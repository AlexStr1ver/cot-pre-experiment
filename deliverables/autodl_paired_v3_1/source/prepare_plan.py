#!/usr/bin/env python3
"""Prepare one answer-blind, shared-cohort plan from saved native20 token IDs.

No model weights are loaded. Existing files are read only; --output must be new.
The historical first paragraph boundary rule is retained, while cohort selection
uses the intersection across targets instead of independent per-target samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

MODEL_REVISION = "2ffdd84494667fbba35525550f9396492b264e23"
DATA_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
PROTOCOL = "shared_cohort_paragraph_signal_plan_v2"
MARKERS = ("<think>", "</think>", "<summary>", "</summary>", "<history>", "</history>")
EXPECTED_MARKER_IDS = [151648, 151649, 151650, 151651, 151652, 151653]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def parse_targets(value):
    try:
        values = [int(x) for x in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Targets must be comma-separated integers") from exc
    if not values or any(x <= 0 for x in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Targets must be distinct positive integers")
    return sorted(values)


def validate_windows(targets, window, limit):
    if not targets or sorted(set(targets)) != list(targets) or min(targets) <= 0:
        raise ValueError("Targets must be sorted distinct positive integers")
    if window < 0 or limit <= 0:
        raise ValueError("boundary-window must be nonnegative and limit positive")
    if any(a + window >= b for a, b in zip(targets, targets[1:])):
        raise ValueError("Target boundary windows must not overlap; repeated positions would not be independent measurements")


def candidate_at(record, tokenizer, target, window):
    """Exact historical boundary rule; never reads reference/predicted answers."""
    first = record["rounds"][0]
    generated = first["generated_token_ids"]
    stop_ids = {tokenizer.convert_tokens_to_ids(x) for x in ("</think>", "<summary>", "</summary>")}
    stop_ids.add(tokenizer.eos_token_id)
    boundary = next((i for i, token in enumerate(generated) if token in stop_ids), len(generated))
    if boundary < target:
        return None, "thinking_ended_before_target"
    cut = next((i for i in range(target, min(boundary, target + window) + 1)
                if tokenizer.decode(generated[:i], skip_special_tokens=False).endswith("\n\n")), None)
    if cut is None:
        return None, "no_paragraph_boundary_in_window"
    ids = list(first["input_token_ids"]) + list(generated[:cut])
    if not ids or ids[0] != tokenizer.bos_token_id or ids.count(tokenizer.bos_token_id) != 1:
        raise ValueError(f"Saved BOS invariant failed for case {record['case']}")
    base = first["prompt"].split("<think>", 1)[0]
    if not tokenizer.decode(ids, skip_special_tokens=False).startswith(base + "<think>\n"):
        raise ValueError(f"Saved prefix/template mismatch for case {record['case']}")
    return {
        "case": record["case"], "unique_id": record["unique_id"],
        "target": target, "cut": cut, "input_tokens": len(ids),
        "prefix_token_ids_sha256": token_hash(ids),
    }, None


def select_common(records, tokenizer, targets, window, limit):
    validate_windows(targets, window, limit)
    records = sorted(records, key=lambda r: r["case"])
    if len({r["case"] for r in records}) != len(records):
        raise ValueError("Duplicate case numbers")
    if len({r["unique_id"] for r in records}) != len(records):
        raise ValueError("Duplicate question IDs")
    by_target, skipped = {}, {}
    for target in targets:
        accepted = {}
        rejected = []
        for record in records:
            item, reason = candidate_at(record, tokenizer, target, window)
            if item is None:
                rejected.append({"case": record["case"], "reason": reason})
            else:
                accepted[item["case"]] = item
        by_target[target], skipped[str(target)] = accepted, rejected
    common = sorted(set.intersection(*(set(x) for x in by_target.values())))
    if len(common) < limit:
        raise ValueError(f"Only {len(common)} common eligible questions, but --limit={limit}; no silent smaller cohort")
    cohort = common[:limit]
    selected = [by_target[target][case] for case in cohort for target in targets]
    return {
        "cohort_cases": cohort,
        "common_eligible_cases": common,
        "eligible_by_target": {str(t): sorted(by_target[t]) for t in targets},
        "skipped_by_target": skipped,
        "selected": selected,
    }


def build_plan(model, baseline, targets, window, limit):
    validate_windows(targets, window, limit)
    cfg_path = baseline / "run_config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("model_revision") != MODEL_REVISION or cfg.get("dataset_revision") != DATA_REVISION:
        raise ValueError("Baseline model/dataset revision differs from audited native20")
    if targets[-1] + window >= cfg["round_tokens"]:
        raise ValueError("Targets/windows must remain below the saved per-round token limit")
    source_hashes = {}
    for name, key in [("native_batch_check.py", "script_sha256"), ("reference_runner.py", "reference_runner_sha256")]:
        source_hashes[name] = sha(baseline / name)
        if source_hashes[name] != cfg[key]:
            raise ValueError(f"Baseline source hash mismatch: {name}")
    paths = sorted((baseline / "cases").glob("*.json"))
    records = [json.loads(path.read_text()) for path in paths]
    if [r["case"] for r in records] != list(range(1, 21)) or not all(r.get("completed") for r in records):
        raise ValueError("Expected the complete saved native20 case set in original order")
    for record in records:
        if not record.get("rounds"):
            raise ValueError("Missing saved round")
        first = record["rounds"][0]
        if not first.get("input_token_ids") or not first.get("generated_token_ids"):
            raise ValueError("Missing saved input/generated token IDs")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    marker_ids = [tokenizer.encode(tag, add_special_tokens=False) for tag in MARKERS]
    if marker_ids != [[x] for x in EXPECTED_MARKER_IDS] or tokenizer.bos_token_id != 151646 or tokenizer.eos_token_id != 151643:
        raise ValueError("Tokenizer control/BOS/EOS IDs differ from audited checkpoint")
    for record in records:
        first = record["rounds"][0]
        if tokenizer.encode(first["prompt"], add_special_tokens=False) != first["input_token_ids"]:
            raise ValueError(f"Saved prompt IDs mismatch for case {record['case']}")
    selection = select_common(records, tokenizer, targets, window, limit)
    hashes_by_case = {r["case"]: sha(p) for p, r in zip(paths, records)}
    for item in selection["selected"]:
        item["case_sha256"] = hashes_by_case[item["case"]]
    tokenizer_files = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]
    return {
        "protocol": PROTOCOL,
        "model_revision": cfg["model_revision"],
        "dataset_revision": cfg["dataset_revision"],
        "baseline_run_config_sha256": sha(cfg_path),
        "baseline_source_sha256": source_hashes,
        "baseline_case_collection_sha256": hashlib.sha256("".join(sha(p) for p in paths).encode()).hexdigest(),
        "planner_sha256": sha(__file__),
        "model_and_tokenizer_file_sha256": {name: sha(model / name) for name in tokenizer_files if (model / name).is_file()},
        "tokenizer_ids": {"bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id,
                          "markers": dict(zip(MARKERS, [x[0] for x in marker_ids]))},
        "targets": list(targets), "boundary_window": window, "limit": limit,
        "source_case_count": len(records),
        "selection_rule": "For every target, first paragraph boundary in [target,target+window] before the first thinking end marker. Intersect eligible cases across all targets, sort case IDs, take first limit. No answer labels used.",
        "population_scope": "Previously explored native20 development trajectories that survive to all requested lengths. Long-reasoning-selected cohort, not an unbiased benchmark or held-out test.",
        "delta_reference_rule": "Earliest selected chronological cut per case, computed after all measurements; not process execution order and not ASAG first ATP.",
        "candidate_semantics": "cut counts saved generated tokens; input_tokens also includes original question/chat prompt. Original IDs are hashed without decode/reencode substitution.",
        **selection,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--targets", type=parse_targets, default=[1024, 2048, 4096])
    parser.add_argument("--boundary-window", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not already exist; use a new plan filename")
    try:
        plan = build_plan(args.model, args.baseline, args.targets, args.boundary_window, args.limit)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"plan": str(args.output), "cohort_cases": plan["cohort_cases"],
                      "common_eligible_cases": plan["common_eligible_cases"],
                      "positions": [(x["case"], x["target"], x["cut"]) for x in plan["selected"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
