#!/usr/bin/env python3
"""Read-only audit of the submitted phase3 smoke run; no model execution."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(root, delivered):
    results = root / "results"
    summary, config, plan = [read(results / name) for name in
                             ("summary.json", "run_config.json", "plan.json")]
    manifest = read(results / "completed_artifacts.json")
    require((root / "exit_status.txt").read_text().strip() == "0", "Nonzero exit")
    require(summary["completed"] and summary["mode"] == "smoke", "Expected completed smoke")
    require(summary["completed_branches"] == summary["planned_branches"] == 2,
            "Expected exactly two smoke branches")
    require(summary["actual_GPU_execution_this_run"] and summary["numerical_audit_passed"],
            "Missing actual GPU/audit success")
    identity = config["experiment_identity"]
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True,
                               separators=(",", ":")).encode()).hexdigest()
    require(fingerprint == config["experiment_fingerprint"] == manifest["experiment_fingerprint"],
            "Experiment fingerprint mismatch")
    package_files = 0
    for line in (root / "code/checksums.sha256").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe manifest path")
        require(sha(root / "code" / name) == expected, "Package hash mismatch: " + name)
        require(sha(delivered / name) == expected, "Delivered code differs: " + name)
        package_files += 1
    for name, expected in identity["code_sha256"].items():
        require(sha(root / "code" / name) == expected, "Execution identity code mismatch")
    rows = list(csv.DictReader((results / "branch_review.csv").open(encoding="utf-8-sig")))
    require(len(rows) == 2, "Unexpected review row count")
    branches = {}
    for path in sorted((results / "branches").glob("*.json")):
        data = read(path)
        require(sha(path) == manifest["branches_sha256"][path.stem], "Branch hash mismatch")
        require(data["record_finished"] and data["answer_completed"], "Incomplete smoke answer")
        require((data["case"], data["cut"], data["replicate_seed"]) == (1, 4103, 2026),
                "Unexpected smoke sample")
        events = data["events"]
        generated = sum(len(e["generated_token_ids"]) for e in events)
        require(generated == data["extra_generated_tokens"], "Generated token mismatch")
        require(data["generation_budget"] == 16384 - data["cut"], "Budget mismatch")
        require(data["total_generated_tokens_with_shared_prefix"] == data["cut"] + generated <= 16384,
                "Budget exceeded")
        require(sum(len(e["input_token_ids"]) for e in events) == data["prefill_tokens_total"],
                "Prefill accounting mismatch")
        require(math.isclose(sum(e["seconds"] for e in events), data["model_call_seconds"], rel_tol=1e-10),
                "Event time mismatch")
        for index, event in enumerate(events):
            require(event["generated_tokens"] == len(event["generated_token_ids"]) ==
                    event["decoder_stats"]["sampled_tokens"], "Event count mismatch")
            require(event["decoder_stats"]["forward_generated_tokens"] == event["generated_tokens"] - 1,
                    "Deferred final forward mismatch")
            if event.get("reset_executed"):
                require(event["reset_verified"] and event["reset_input_token_ids"] ==
                        events[index + 1]["input_token_ids"], "Reset mismatch")
        require(events[-1]["generated_token_ids"][-1] == plan["tokenizer_ids"]["eos"], "Missing EOS")
        require(events[-1]["status"] == "final_eos" and data["pred_boxed"] == "4", "Final outcome changed")
        row = next(r for r in rows if r["branch"] == data["branch"])
        require(row["reference_answer"] == "6" and row["strict_complete_integer_correct"] == "False",
                "Unexpected grade")
        branches[data["branch"]] = data
    A, B = branches["A_continue"], branches["B_summary"]
    require(A["events"][0]["input_token_ids"] + B["events"][0]["inserted_control_token_ids"] ==
            B["events"][0]["input_token_ids"], "A/B initial prefixes differ")
    require(B["forced_summary_succeeded"] and not B["forced_summary_fallback_used"], "B reset failed")
    audits = [read(p) for p in sorted((results / "numerical_audits").glob("*.json"))]
    require({d["case"] for d in audits} == {1, 6, 8}, "Incomplete numerical audit cohort")
    for d in audits:
        require(d["passed"] and d["resource_passed"] and d["chunk_comparison"]["passed"] and
                d["probe_comparison"]["passed"], "Numerical audit failure")
        cross = d["cross_run_phase2_comparison"]
        require(all(v == 0 for v in cross["relative_H_differences"].values()) and
                cross["probe_comparison"]["max_aligned_token_probability_difference"] == 0,
                "Cross-phase signal mismatch")
    numbers = [s for d in audits for s in d["chunk_comparison"]["signals"].values()]
    max_remaining = sum(2 * (16384 - p["cut"]) for p in plan["selected"]
                        if (p["case"], p["cut"]) != (1, 4103))
    return {
        "review_date_beijing": "2026-10-24", "audit_passed": True,
        "execution": "completed smoke; main18 not executed in this upload",
        "checked_package_files": package_files, "fingerprint": fingerprint,
        "completed_action_pairs": 1, "completed_action_branches": 2,
        "numerical_audit_cases": [d["case"] for d in audits],
        "max_chunk_relative_H_difference": max(s["H_relative_difference"] for s in numbers),
        "max_chunk_mean_logit_difference": max(s["logit_mean_absolute_difference"] for s in numbers),
        "max_chunk_probe_token_probability_difference": max(d["probe_comparison"]["max_aligned_token_probability_difference"] for d in audits),
        "cross_phase2_H_and_probe_probability_differences": 0,
        "A": {k: A[k] for k in ["pred_boxed", "answer_completed", "extra_generated_tokens", "branch_wall_seconds", "summary_generated_tokens_total", "prefill_tokens_total", "peak_allocated_gib", "executed_summary_restarts"]},
        "B": {k: B[k] for k in ["pred_boxed", "answer_completed", "extra_generated_tokens", "branch_wall_seconds", "summary_generated_tokens_total", "prefill_tokens_total", "peak_allocated_gib", "executed_summary_restarts"]},
        "reference_answer": "6", "correct_A": False, "correct_B": False,
        "B_minus_A_generated_tokens": B["extra_generated_tokens"] - A["extra_generated_tokens"],
        "relative_generated_token_reduction": 1 - B["extra_generated_tokens"] / A["extra_generated_tokens"],
        "relative_branch_wall_reduction": 1 - B["branch_wall_seconds"] / A["branch_wall_seconds"],
        "next_main_reused_branches": 2, "next_main_new_branches": 16,
        "next_main_max_new_tokens": max_remaining,
        "next_main_max_tokens_at_smoke_throughput_minutes": max_remaining / summary["new_generated_tokens_per_branch_wall_second"] / 60,
        "timing_caveat": "Token-cap extrapolation is not a measured duration or wall-clock upper bound; excludes reload/audits and longer-context throughput changes.",
        "conclusion": "Engineering pass; no demonstrated quality-preserving benefit or H+C action selection advantage. Reuse smoke and complete frozen main18.",
        "manual_validation": "Reference 6 independently checked: omitted quadratics x^2-4 and x^2-1 both satisfy the stated condition. Tokenizer/base/reset fidelity was separately audited against local original native20 and actual tokenizer."
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("delivered_source_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_directory, args.delivered_source_directory), ensure_ascii=False, indent=2))
