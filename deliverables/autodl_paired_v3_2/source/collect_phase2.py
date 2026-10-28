#!/usr/bin/env python3
"""Frozen-plan offline signal measurement; no online controller or ASAG replication."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

from v1_core import (EXPECTED_REVISION, attention_summary, build_prefix, forward_chunk,
                     prefill, replay_attention, sha, synchronize)

ASAG_SUFFIX = "\n\n Final Answer\n\n \\boxed"
H_KEY = "H_sum_over_layers_heads_queries_div_log_keys"
RAW_KEY = "entropy_sum_nats"
SIGNALS = ("raw_tail32", "asag_style_probe_window")


def dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def precision_settings(torch, dtype):
    """Explicit arithmetic mode; never silently fall back or loosen a gate."""
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"requested_model_dtype": dtype,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "interpretation": "FP32 rerun is a precision diagnostic on the same saved prefixes, not a BF16 reproduction or end-to-end speed benchmark."}


def logit_diagnostics(a, b):
    import torch
    a, b = a.float(), b.float()
    if a.shape != b.shape or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError('Logit shapes must agree and values must be finite')
    signed = a - b
    pa, pb = a.softmax(-1), b.softmax(-1)
    token_a, token_b = a.argmax(-1).item(), b.argmax(-1).item()
    return {"logit_max_absolute_difference": signed.abs().max().item(),
            "logit_mean_absolute_difference": signed.abs().mean().item(),
            "logit_mean_signed_difference": signed.mean().item(),
            "logit_centered_mean_absolute_difference": (signed - signed.mean(-1, keepdim=True)).abs().mean().item(),
            "probability_total_variation": (pa - pb).abs().sum(-1).mean().item() / 2,
            "primary_greedy_token_id": token_a, "alternative_greedy_token_id": token_b,
            "greedy_next_token_agrees": token_a == token_b,
            "primary_greedy_token_probability_difference": abs(pa[0, token_a].item() - pb[0, token_a].item())}


def compare_probes(primary, alternate, tolerance):
    """Validate actual C decode paths, whose final prefill sizes differ from H replay."""
    a, b = primary['selected_token_probabilities'], alternate['selected_token_probabilities']
    finite = bool(a) and bool(b) and all(math.isfinite(p) and 0 <= p <= 1 for p in a + b)
    aligned = (primary['generated_token_ids'] == alternate['generated_token_ids']
               and len(a) == len(b) == len(primary['generated_token_ids']))
    difference = max(abs(x - y) for x, y in zip(a, b)) if aligned and finite else None
    same_status = primary['status'] == alternate['status']
    ca, cb = primary['C_mean_selected_token_probability'], alternate['C_mean_selected_token_probability']
    c_difference = abs(ca - cb) if ca is not None and cb is not None else None
    c_valid = (ca is None and cb is None) or (c_difference is not None and math.isfinite(c_difference) and c_difference <= tolerance)
    return {"passed": bool(aligned and finite and same_status and c_valid and difference <= tolerance),
            "confidence_available_in_both_probes": ca is not None and cb is not None,
            "generated_token_ids_agree": aligned, "statuses_agree": same_status,
            "max_aligned_token_probability_difference": difference,
            "C_absolute_difference": c_difference,
            "max_allowed_token_probability_difference": tolerance,
            "interpretation": "Engineering check of full greedy probes across chunk sizes; matching empty/truncated probes may be numerically consistent but still have C=null and are not usable answers."}


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def first_closed_box(text):
    match = re.search(r"\\boxed\s*\{", text)
    if match is None:
        return None
    begin, depth = match.end(), 1
    for index in range(begin, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[begin:index].strip(), index + 1
    return None


def content_probability_diagnostic(tokenizer, generated, probabilities, complete_text):
    """Keep only token spans wholly within outer box content; never split token probabilities."""
    token_pieces = [tokenizer.decode([token], skip_special_tokens=False) for token in generated]
    result = {"token_text_pieces": token_pieces, "strict_content_C": None,
              "strict_content_definition": "Arithmetic mean only over generated token spans fully inside the first outer boxed answer; opening/closing or mixed delimiter/content tokens excluded. Secondary diagnostic, not official ASAG confidence."}
    match = re.search(r"\\boxed\s*\{", complete_text)
    closed = first_closed_box(complete_text)
    if match is None or closed is None or not closed[0]:
        result["strict_content_unavailable_reason"] = "no_complete_nonempty_box"
        return result
    start, end = match.end() - len(ASAG_SUFFIX), closed[1] - 1 - len(ASAG_SUFFIX)
    previous, spans = "", []
    for index in range(len(generated)):
        current = tokenizer.decode(generated[:index + 1], skip_special_tokens=False)
        if not current.startswith(previous):
            result["strict_content_unavailable_reason"] = "nonmonotonic_incremental_decoding"
            return result
        spans.append((len(previous), len(current)))
        previous = current
    included = [i for i, (a, b) in enumerate(spans) if b > a and a >= start and b <= end]
    mixed = [i for i, (a, b) in enumerate(spans) if max(a, start) < min(b, end) and i not in included]
    covered = sum(b - a for i, (a, b) in enumerate(spans) if i in included)
    result.update(strict_content_token_indices=included, excluded_mixed_token_indices=mixed,
                  generated_token_character_spans=spans, answer_content_generated_character_span=[start, end],
                  strict_content_character_coverage=covered / (end - start) if end > start else 0,
                  strict_content_token_count=len(included), generated_token_count=len(generated))
    if included:
        result["strict_content_C"] = sum(probabilities[i] for i in included) / len(included)
    else:
        result["strict_content_unavailable_reason"] = "no_token_wholly_inside_answer_content"
    return result


def verify_plan(plan, baseline, tokenizer):
    """Validate all selected IDs/boundaries before loading weights. Never use gold answers."""
    baseline = Path(baseline)
    cfg_path = baseline / "run_config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("model_revision") != EXPECTED_REVISION or plan.get("model_revision") != EXPECTED_REVISION:
        raise ValueError("Plan/baseline checkpoint revision mismatch")
    if sha(cfg_path) != plan["baseline_run_config_sha256"]:
        raise ValueError("Plan baseline config hash mismatch")
    for filename, key in [("reference_runner.py", "reference_runner_sha256"),
                          ("native_batch_check.py", "script_sha256")]:
        if sha(baseline / filename) != cfg[key]:
            raise ValueError(f"Baseline source hash mismatch: {filename}")
    if plan.get("protocol") != "shared_cohort_paragraph_signal_plan_v2":
        raise ValueError("Unsupported plan protocol")
    cases, targets = plan["cohort_cases"], plan["targets"]
    for name, values in [("cohort_cases", cases), ("targets", targets)]:
        if (not values or any(type(v) is not int or v <= 0 for v in values)
                or values != sorted(set(values))):
            raise ValueError(f"{name} must contain sorted distinct positive integers")
    width = plan.get("boundary_window", 128)
    if type(width) is not int or width < 0:
        raise ValueError("Invalid boundary window")
    if any(left + width >= right for left, right in zip(targets, targets[1:])):
        raise ValueError("Target boundary windows must not overlap")
    expected = {(case, target) for case in cases for target in targets}
    actual = [(x["case"], x["target"]) for x in plan["selected"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Selected positions must be exactly cohort_cases x targets")
    actual_cuts = [(x["case"], x["cut"]) for x in plan["selected"]]
    if len(actual_cuts) != len(set(actual_cuts)):
        raise ValueError("Duplicate actual case/cut pairs would overwrite output filenames")
    prepared = []
    for entry in plan["selected"]:
        case, target, cut = entry["case"], entry["target"], entry["cut"]
        if type(cut) is not int or not target <= cut <= target + width:
            raise ValueError("Cut outside target boundary window")
        path = baseline / "cases" / f"{case:03d}.json"
        if sha(path) != entry["case_sha256"]:
            raise ValueError(f"Case {case} hash mismatch")
        record = json.loads(path.read_text())
        if record["case"] != case or record["unique_id"] != entry["unique_id"]:
            raise ValueError("Case identity mismatch")
        ids, reason = build_prefix(record, cut, tokenizer)
        if reason or ids_sha(ids) != entry["prefix_token_ids_sha256"]:
            raise ValueError(f"Case {case} cut {cut} prefix mismatch or after thinking boundary: {reason}")
        if entry.get("input_tokens") != len(ids):
            raise ValueError("Planned input token count mismatch")
        generated = record["rounds"][0]["generated_token_ids"]
        first_boundary = next((i for i in range(target, cut + 1)
            if tokenizer.decode(generated[:i], skip_special_tokens=False).endswith("\n\n")), None)
        if first_boundary != cut:
            raise ValueError(f"Case {case}: cut is not first paragraph boundary in selected window")
        if len(ids) < 32 or cut < 32:
            raise ValueError("Need at least 32 generated thought tokens for raw-tail window")
        prepared.append({**entry, "input_token_ids": ids})
    return sorted(prepared, key=lambda e: (e["case"], e["cut"]))


def probe_asag_style(model, tokenizer, prefix_ids, max_tokens, chunk, device):
    """Literal ASAG prompt; explicit diagnostic stopping/scope, not exact author implementation."""
    import torch
    if max_tokens <= 0:
        raise ValueError("Probe token budget must be positive")
    suffix_ids = tokenizer.encode(ASAG_SUFFIX, add_special_tokens=False)
    generated, probabilities = [], []
    result, status = None, "token_limit_without_closed_box"
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        cache, logits = prefill(model, prefix_ids + suffix_ids, chunk, device)
        for index in range(max_tokens):
            token = int(logits.argmax(-1).item())
            probability = float(logits.float().softmax(-1)[0, token].item())
            if not math.isfinite(probability):
                raise ValueError("Nonfinite selected-token probability")
            generated.append(token)
            probabilities.append(probability)
            text = tokenizer.decode(generated, skip_special_tokens=False)
            result = first_closed_box(ASAG_SUFFIX + text)
            if result is not None:
                status = "closed_box_review_required" if result[0] else "empty_closed_box"
                break
            if token == tokenizer.eos_token_id:
                status = "eos_without_closed_box"
                break
            if index + 1 < max_tokens:
                out = forward_chunk(model, [token], cache, device)
                cache, logits = out.past_key_values, out.logits[:, -1, :]
    synchronize(device)
    elapsed = time.perf_counter() - started
    usable = result is not None and bool(result[0])
    result_record = {
        "status": status, "answer": result[0] if result is not None else None,
        "C_mean_selected_token_probability": sum(probabilities) / len(probabilities) if usable else None,
        "partial_token_probability_mean_diagnostic": sum(probabilities) / len(probabilities),
        "generated_token_ids": generated, "selected_token_probabilities": probabilities,
        "generated_text": tokenizer.decode(generated, skip_special_tokens=False),
        "generated_tokens": len(generated), "inserted_suffix": ASAG_SUFFIX,
        "inserted_suffix_token_ids": suffix_ids, "prefill_input_tokens": len(prefix_ids) + len(suffix_ids),
        "seconds_including_independent_prefill": elapsed,
        "confidence_scope": "All generated tokens up to and including first balanced box closure, including generated opening/closing format and any characters in the closure token. Empty/truncated boxes have C=null. Not semantic correctness probability.",
    }
    result_record["format_diagnostic"] = content_probability_diagnostic(
        tokenizer, generated, probabilities, ASAG_SUFFIX + result_record["generated_text"])
    result_record["closure_token_probability"] = probabilities[-1] if result is not None else None
    return result_record


def measure_modes(model, tokenizer, ids, chunk, device):
    suffix = tokenizer.encode(ASAG_SUFFIX, add_special_tokens=False)
    if not suffix or len(ids) < len(suffix):
        raise ValueError("Invalid probe instruction window")
    result, logits = {}, {}
    result["raw_tail32"], logits["raw_tail32"] = replay_attention(model, ids, 32, chunk, device)
    result["asag_style_probe_window"], logits["asag_style_probe_window"] = replay_attention(
        model, ids + suffix, 2 * len(suffix), chunk, device)
    return result, logits


def compare_chunks(primary, primary_logits, alternative, alternative_logits,
                   max_relative_h, max_mean_logit):
    comparisons = {}
    for name in SIGNALS:
        h, h2 = primary[name][H_KEY], alternative[name][H_KEY]
        diagnostics = logit_diagnostics(primary_logits[name], alternative_logits[name])
        rel = abs(h - h2) / max(abs(h), 1e-12)
        mean = diagnostics['logit_mean_absolute_difference']
        passed = math.isfinite(rel) and math.isfinite(mean) and rel <= max_relative_h and mean <= max_mean_logit
        comparisons[name] = {
            "H_primary": h, "H_alternative": h2, "H_relative_difference": rel,
            **diagnostics,
            "gate_passed": passed,
        }
    return {"passed": all(v["gate_passed"] for v in comparisons.values()), "signals": comparisons,
            "engineering_gate": {"max_relative_H_difference": max_relative_h,
                                 "max_mean_absolute_logit_difference": max_mean_logit},
            "interpretation": "Numerical sanity comparison of two bounded chunk sizes. These are engineering tolerances, not entropy/confidence policy thresholds and not a full long-sequence attention reference."}


def chronological_deltas(items):
    """Never use canary execution order as the entropy reference."""
    grouped = {}
    for item in items:
        if item.get("status") == "measured_review_required":
            grouped.setdefault(item["case"], []).append(item)
    for group in grouped.values():
        group.sort(key=lambda item: item["cut"])
        reference = group[0]
        for item in group:
            item["delta_reference_cut"] = reference["cut"]
            item["delta_reference_is_first_selected_chronological_point_not_first_ATP"] = True
            item["relative_changes"] = {}
            for name in SIGNALS:
                h0, raw0 = reference["signals"][name][H_KEY], reference["signals"][name][RAW_KEY]
                h, raw = item["signals"][name][H_KEY], item["signals"][name][RAW_KEY]
                item["relative_changes"][name] = {
                    "normalized_H": (h - h0) / h0 if h0 > 0 else None,
                    "raw_entropy_sum": (raw - raw0) / raw0 if raw0 > 0 else None,
                    "normalization_denominator_log_keys": math.log(item["signals"][name]["keys"]),
                }
    return items


def hardware_info(torch, device):
    info = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "device": device}
    if device == "cuda":
        info.update(name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                    total_bytes=torch.cuda.get_device_properties(0).total_memory,
                    bf16_supported_including_emulation=torch.cuda.is_bf16_supported())
        try:
            info["bf16_native_supported"] = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            info["bf16_native_supported"] = None
        info["note"] = "Native BF16 support is diagnostic metadata; FP32 mode does not require BF16. Do not compare wall-clock costs across GPU/dtype changes."
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--probe-tokens", type=int, default=128)
    parser.add_argument("--prefill-chunk", type=int, default=128)
    parser.add_argument("--canary-chunk", type=int, default=64)
    parser.add_argument("--max-canary-relative-h", type=float, default=0.01)
    parser.add_argument("--max-canary-mean-logit", type=float, default=0.05)
    parser.add_argument("--max-canary-probability-difference", type=float, default=0.01)
    parser.add_argument("--max-canary-seconds", type=float, default=120.0)
    parser.add_argument("--max-memory-fraction", type=float, default=0.85)
    args = parser.parse_args()
    if min(args.probe_tokens, args.prefill_chunk, args.canary_chunk) <= 0 or args.prefill_chunk == args.canary_chunk:
        parser.error("Budgets/chunks must be positive and canary chunk must differ")
    if min(args.max_canary_relative_h, args.max_canary_mean_logit, args.max_canary_probability_difference) < 0 or not all(
            math.isfinite(x) for x in [args.max_canary_relative_h, args.max_canary_mean_logit, args.max_canary_probability_difference]):
        parser.error("Engineering tolerances must be finite and nonnegative")
    if not math.isfinite(args.max_canary_seconds) or args.max_canary_seconds <= 0 or not 0 < args.max_memory_fraction < 1:
        parser.error("Resource limits require positive finite seconds and memory fraction in (0,1)")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"completed": False, "planned_positions": 0, "executed_positions": 0,
               "canary_passed": False, "actual_gpu_validation_done_by_this_run": False}
    dump(args.output / 'summary.json', summary)
    items = []
    def checkpoint(stage, item=None, path=None):
        summary['stage'] = stage
        if item is not None:
            item['stage'] = stage
            summary['current_case'] = item['case']
            summary['current_cut'] = item['cut']
            dump(path, item)
        dump(args.output / 'summary.json', summary)
        print(json.dumps({'stage': stage, 'executed_positions': summary['executed_positions'],
                          'case': summary.get('current_case'), 'cut': summary.get('current_cut')}), flush=True)
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if transformers.__version__ != "4.51.3":
            raise ValueError("This runner requires validated Transformers 4.51.3; do not upgrade old environment blindly")
        arithmetic = precision_settings(torch, args.dtype)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is required for this GPU run; use --device cpu only for explicit synthetic checks")
        if args.device == "cuda" and args.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise ValueError("Explicit BF16 mode requires CUDA BF16 support")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        plan = json.loads(args.plan.read_text())
        prepared = verify_plan(plan, args.baseline, tokenizer)
        metadata_hashes = plan.get("model_and_tokenizer_file_sha256", {})
        if not {"config.json", "tokenizer.json", "tokenizer_config.json"}.issubset(metadata_hashes):
            raise ValueError("Plan lacks required model/tokenizer hashes")
        for name, expected_hash in metadata_hashes.items():
            candidate = args.model / name
            if Path(name).name != name or sha(candidate) != expected_hash:
                raise ValueError(f"Plan model/tokenizer metadata hash mismatch: {name}")
        summary["planned_positions"] = len(prepared)
        suffix_ids = tokenizer.encode(ASAG_SUFFIX, add_special_tokens=False)
        cfg = json.loads((args.model / "config.json").read_text())
        if cfg.get("model_type") != "qwen2" or cfg.get("num_hidden_layers", 0) < 4:
            raise ValueError("Validated model architecture is Qwen2 with >=4 layers")
        if any(p["cut"] < len(suffix_ids) for p in prepared):
            raise ValueError("Probe window would include pre-thought prompt tokens")
        canary = max((p for p in prepared if p["case"] == plan["cohort_cases"][0]), key=lambda p: p["cut"])
        order = [canary] + [p for p in prepared if p is not canary]
        config = {
            "protocol": "offline_matched_prefix_dual_H_asag_style_C_v2_1_precision_diagnostic",
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "plan_sha256": sha(args.plan), "plan": plan,
            "baseline_run_config_sha256": sha(args.baseline / "run_config.json"),
            "script_sha256": sha(__file__), "v1_core_sha256": sha(Path(__file__).with_name("v1_core.py")),
            "model_config_sha256": sha(args.model / "config.json"),
            "tokenizer_config_sha256": sha(args.model / "tokenizer_config.json"),
            "tokenizer_json_sha256": sha(args.model / "tokenizer.json"),
            "model_revision_claim": EXPECTED_REVISION, "local_weight_revision_independently_verified": False,
            "weights": [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(args.model.glob("*.safetensors"))],
            "python": sys.version, "platform": platform.platform(), "transformers": transformers.__version__,
            "hardware": hardware_info(torch, args.device), "dtype": args.dtype,
            "arithmetic_settings": arithmetic,
            "ASAG_style_probe_instruction": ASAG_SUFFIX, "probe_instruction_ids": suffix_ids,
            "probe_window_queries": 2 * len(suffix_ids), "raw_tail_queries": 32,
            "attention_definition": "Last four layers/all query heads. Raw and normalized entropy retained. Normalize summed query-row entropy by ln(full key count); causal future entries are zero. ASAG-style window uses last m original thought tokens + m probe instruction tokens. Raw-tail32 excludes probe instruction. HF eager softmax FP32 then explicit model dtype cast; FP32 entropy reduction without row-mass renormalization.",
            "delta_definition": "Relative to minimum selected chronological cut per case, computed after measurements; NOT first Wait ATP. Both raw and normalized changes saved. Scheduled paragraph boundaries and explicit box stopping differ from full ASAG.",
            "probe_definition": "Independent fresh-cache greedy decode with literal ASAG instruction; C averages all generated token probabilities including generated format up to first complete box; empty/truncated C=null. Not equivalent to legacy v1 C prompt/scope.",
            "cost_scope": "Each H replay and C has independent charged prefill. Canary repeated-chunk audit charged separately. No original prefix generation or online/controller latency claim. Model loading measured separately.",
            "not_full_ASAG_reproduction": True, "not_online_controller": True,
        }
        dump(args.output / "run_config.json", config)
        checkpoint('model_loading')
        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            torch_dtype=getattr(torch, args.dtype),
            attn_implementation="eager").to(args.device).eval()
        synchronize(args.device)
        summary["model_load_seconds"] = time.perf_counter() - started
        config['actual_model_dtype'] = str(next(model.parameters()).dtype)
        dump(args.output / 'run_config.json', config)
        for entry in order:
            ids = entry["input_token_ids"]
            item = {k: v for k, v in entry.items() if k != "input_token_ids"}
            item.update(status="measuring", input_tokens=len(ids), is_canary=entry is canary)
            path = args.output / f"case_{entry['case']:03d}_cut_{entry['cut']:05d}.json"
            items.append(item)
            checkpoint('measuring_primary_attention', item, path)
            synchronize(args.device)
            point_started = time.perf_counter()
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            item["signals"], logits = measure_modes(model, tokenizer, ids, args.prefill_chunk, args.device)
            checkpoint('primary_attention_saved', item, path)
            if entry is canary:
                checkpoint('auditing_alternative_attention', item, path)
                alternate, alternate_logits = measure_modes(model, tokenizer, ids, args.canary_chunk, args.device)
                item["canary_chunk_comparison"] = compare_chunks(item["signals"], logits, alternate, alternate_logits,
                    args.max_canary_relative_h, args.max_canary_mean_logit)
                item["canary_audit_seconds"] = sum(v["replay_seconds_including_prefill_and_entropy_reduction"] for v in alternate.values())
                item["canary_alternative_signals"] = alternate
                del alternate_logits
                if not item["canary_chunk_comparison"]["passed"]:
                    item["status"] = "canary_numerical_gate_failed"
                    dump(path, item)
                    raise RuntimeError("Long-prefix chunk consistency gate failed; inspect canary, do not expand")
            del logits
            checkpoint('measuring_primary_confidence', item, path)
            item["probe"] = probe_asag_style(model, tokenizer, ids, args.probe_tokens, args.prefill_chunk, args.device)
            checkpoint('primary_confidence_saved', item, path)
            if entry is canary:
                checkpoint('auditing_alternative_confidence', item, path)
                item['canary_alternative_probe'] = probe_asag_style(
                    model, tokenizer, ids, args.probe_tokens, args.canary_chunk, args.device)
                item['canary_probe_comparison'] = compare_probes(
                    item['probe'], item['canary_alternative_probe'], args.max_canary_probability_difference)
                item['canary_audit_seconds'] += item['canary_alternative_probe']['seconds_including_independent_prefill']
                checkpoint('confidence_audit_saved', item, path)
                if not item['canary_probe_comparison']['passed']:
                    item['status'] = 'canary_confidence_gate_failed'
                    dump(path, item)
                    raise RuntimeError('Long-prefix confidence consistency gate failed; inspect canary, do not expand')
            item["measurement_seconds_excluding_canary_audit"] = sum(
                v["replay_seconds_including_prefill_and_entropy_reduction"] for v in item["signals"].values()) + item["probe"]["seconds_including_independent_prefill"]
            if args.device == "cuda":
                item["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
                item["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
                item["peak_memory_includes_canary_audit"] = entry is canary
            if entry is canary:
                synchronize(args.device)
                elapsed = time.perf_counter() - point_started
                memory_fraction = (torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory) if args.device == "cuda" else None
                resource_pass = elapsed <= args.max_canary_seconds and (memory_fraction is None or memory_fraction <= args.max_memory_fraction)
                item["canary_resource_gate"] = {
                    "passed": resource_pass, "point_seconds_including_audit": elapsed,
                    "peak_allocated_fraction_of_total_memory": memory_fraction,
                    "max_seconds": args.max_canary_seconds, "max_memory_fraction": args.max_memory_fraction,
                    "interpretation": "Resource controls evaluated after the canary finishes, not a hard timeout or scientific policy threshold. Includes extra chunk audits; excludes model loading.",
                }
                if not resource_pass:
                    item["status"] = "canary_resource_gate_failed"
                    summary["canary_passed"] = False
                    dump(path, item)
                    raise RuntimeError("Canary resource gate failed; inspect time/memory before expanding")
                summary['canary_passed'] = True
            item["status"] = "measured_review_required"
            summary["executed_positions"] += 1
            checkpoint('point_completed', item, path)
            print(json.dumps({k: item[k] for k in ["case", "target", "cut", "status", "is_canary"]}), flush=True)
        chronological_deltas(items)
        for item in items:
            dump(args.output / f"case_{item['case']:03d}_cut_{item['cut']:05d}.json", item)
        summary["completed"] = summary["executed_positions"] == summary["planned_positions"] > 0
        summary["actual_gpu_validation_done_by_this_run"] = args.device == "cuda" and summary["completed"]
        summary["measurement_seconds_sum_excluding_canary_audit"] = sum(x["measurement_seconds_excluding_canary_audit"] for x in items)
        summary["canary_audit_seconds_sum"] = sum(x.get("canary_audit_seconds", 0) for x in items)
        summary["closed_nonempty_probes"] = sum(x["probe"]["C_mean_selected_token_probability"] is not None for x in items)
        checkpoint('completed')
    except Exception as exc:
        summary["error"] = repr(exc)
        summary['failed_stage'] = summary.get('stage', 'initialization')
        summary['stage'] = 'failed'
        if items:
            current = items[-1]
            current['error'] = repr(exc)
            if current['status'] == 'measuring':
                current['status'] = 'measurement_failed'
            if args.device == 'cuda' and 'torch' in locals() and torch.cuda.is_initialized():
                current['peak_allocated_gib_at_failure'] = torch.cuda.max_memory_allocated() / 2**30
                current['peak_reserved_gib_at_failure'] = torch.cuda.max_memory_reserved() / 2**30
            dump(args.output / f"case_{current['case']:03d}_cut_{current['cut']:05d}.json", current)
        raise
    finally:
        dump(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
