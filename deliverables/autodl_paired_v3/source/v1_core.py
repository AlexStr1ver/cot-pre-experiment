#!/usr/bin/env python3
"""Offline tail-attention/boxed-confidence diagnostic; NOT an ASAG reproduction.

Reads original native20 token IDs. Writes only a new output directory. No training,
model download, policy decisions, answer grading, or modification of old results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time


PROBE_SUFFIX = "\n</think>\nFinal answer: \\boxed{"
EXPECTED_REVISION = "2ffdd84494667fbba35525550f9396492b264e23"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def integers(value):
    result = [int(v) for v in value.split(",")]
    if not result or any(v <= 0 for v in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Use distinct positive comma-separated integers")
    return sorted(result)


def closed_box(text):
    """Return (answer, end character offset) for first complete nested box."""
    start = text.find("\\boxed{")
    if start < 0:
        return None
    begin = start + len("\\boxed{")
    depth = 1
    for index in range(begin, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[begin:index].strip(), index + 1
    return None


def attention_summary(attentions, selected_layers=4):
    """Sum entropy over layer/head/query, normalized by log full key length.

The returned HF eager probabilities are converted to float32 for reduction.
For BF16 inference these probabilities have already been rounded to BF16 after
the implementation's FP32 softmax. We do not silently renormalize them.
"""
    import torch
    if len(attentions) < selected_layers:
        raise ValueError("Model has fewer layers than requested")
    tail = attentions[-selected_layers:]
    if any(a is None for a in tail):
        raise ValueError("Attention probabilities unavailable; eager backend required")
    shape = tail[0].shape
    if len(shape) != 4 or shape[0] != 1 or shape[-1] <= 1:
        raise ValueError(f"Expected batch=1, heads, queries, keys>1; got {shape}")
    if any(a.shape != shape for a in tail):
        raise ValueError("Selected attention shapes differ")
    heads, queries, keys = shape[1:]
    raw_by_layer = []
    max_row_mass_error = 0.0
    for tensor in tail:
        p = tensor.float()
        if not torch.isfinite(p).all() or (p < 0).any():
            raise ValueError("Nonfinite or negative attention probabilities")
        max_row_mass_error = max(max_row_mass_error, (p.sum(-1) - 1).abs().max().item())
        h = -(p * p.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1)
        raw_by_layer.append(h.sum().item())
    raw_sum = sum(raw_by_layer)
    normalized = raw_sum / math.log(keys)
    return {
        "entropy_sum_nats": raw_sum,
        "H_sum_over_layers_heads_queries_div_log_keys": normalized,
        "H_query_mean_sum_over_layers_heads_div_log_keys": normalized / queries,
        "H_mean_over_layers_heads_queries_div_log_keys": normalized / (queries * heads * selected_layers),
        "raw_entropy_sum_by_layer_nats": raw_by_layer,
        "layer_indices_zero_based": list(range(len(attentions) - selected_layers, len(attentions))),
        "heads_per_layer": heads,
        "queries": queries,
        "keys": keys,
        "max_attention_row_mass_error": max_row_mass_error,
    }


def synchronize(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def forward_chunk(model, ids, cache, device, attentions=False):
    import torch
    x = torch.tensor([ids], device=device, dtype=torch.long)
    return model(input_ids=x, past_key_values=cache, use_cache=True,
                 output_attentions=attentions, return_dict=True, logits_to_keep=1)


def prefill(model, ids, chunk_size, device):
    """Fresh cache each call. Only the last output's logits remain live."""
    if not ids:
        raise ValueError("Empty input")
    cache = None
    for offset in range(0, len(ids), chunk_size):
        out = forward_chunk(model, ids[offset:offset + chunk_size], cache, device)
        cache = out.past_key_values
    return cache, out.logits[:, -1, :]


def replay_attention(model, ids, query_window, prefill_chunk, device):
    import torch
    if len(ids) < query_window:
        raise ValueError("Prefix shorter than query window")
    if query_window <= 0 or prefill_chunk <= 0:
        raise ValueError("Window and chunk sizes must be positive")
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        cache = None
        before_tail = ids[:-query_window]
        if before_tail:
            cache, unused_logits = prefill(model, before_tail, prefill_chunk, device)
            del unused_logits
        out = forward_chunk(model, ids[-query_window:], cache, device, attentions=True)
        summary = attention_summary(out.attentions)
        final_logits = out.logits[:, -1, :].detach().float().cpu()
    synchronize(device)
    summary["replay_seconds_including_prefill_and_entropy_reduction"] = time.perf_counter() - started
    return summary, final_logits


def short_reference_comparison(model, ids, query_window, measured, measured_logits, device):
    """Full eager reference for short sequences only; report differences, not policy validity."""
    import torch
    if len(ids) > 512:
        raise ValueError("Full reference is limited to 512 total input tokens")
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        full = model(input_ids=torch.tensor([ids], device=device), use_cache=False,
                     output_attentions=True, return_dict=True, logits_to_keep=1)
        expected = attention_summary(tuple(a[:, :, -query_window:, :] for a in full.attentions))
        reference_logits = full.logits[:, -1, :].detach().float().cpu()
    h_key = "H_sum_over_layers_heads_queries_div_log_keys"
    difference = (reference_logits - measured_logits).abs()
    synchronize(device)
    return {
        "reference_H": expected[h_key],
        "H_absolute_difference": abs(expected[h_key] - measured[h_key]),
        "H_relative_difference": abs(expected[h_key] - measured[h_key]) / max(abs(expected[h_key]), 1e-12),
        "logit_max_absolute_difference": difference.max().item(),
        "logit_mean_absolute_difference": difference.mean().item(),
        "greedy_next_token_agrees": bool(reference_logits.argmax(-1).item() == measured_logits.argmax(-1).item()),
        "reference_seconds_excluded_from_replay_and_probe": time.perf_counter() - started,
        "interpretation": "Numerical diagnostic in actual dtype/device; inspect differences before longer replay. BF16 need not be bitwise identical. No policy/effectiveness claim.",
    }


def probe_answer(model, tokenizer, prefix_ids, max_tokens, prefill_chunk, device):
    """Independent fresh cache: the signal replay's KV is never mutated/reused."""
    import torch
    suffix_ids = tokenizer.encode(PROBE_SUFFIX, add_special_tokens=False)
    generated, probabilities = [], []
    status, result = "token_limit_without_closed_box", None
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        cache, logits = prefill(model, prefix_ids + suffix_ids, prefill_chunk, device)
        for index in range(max_tokens):
            next_id = int(logits.argmax(-1).item())
            probability = float(logits.float().softmax(-1)[0, next_id].item())
            generated.append(next_id)
            probabilities.append(probability)
            text = tokenizer.decode(generated, skip_special_tokens=False)
            result = closed_box("\\boxed{" + text)
            if result is not None:
                status = "closed_box_review_required"
                break
            if next_id == tokenizer.eos_token_id:
                status = "eos_without_closed_box"
                break
            if index + 1 < max_tokens:
                out = forward_chunk(model, [next_id], cache, device)
                cache, logits = out.past_key_values, out.logits[:, -1, :]
    synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "status": status,
        "answer": result[0] if result is not None else None,
        "confidence_mean_selected_token_probability": sum(probabilities) / len(probabilities) if result is not None else None,
        "partial_token_probability_mean_diagnostic": sum(probabilities) / len(probabilities),
        "generated_token_ids": generated,
        "selected_token_probabilities": probabilities,
        "generated_text": tokenizer.decode(generated, skip_special_tokens=False),
        "generated_tokens": len(generated),
        "inserted_suffix_token_ids": suffix_ids,
        "probe_seconds_including_independent_prefill": elapsed,
        "confidence_scope": "Generated token probabilities through first balanced closing brace, including token containing closure; tokens may also contain trailing characters. No answer-correctness claim.",
    }


def build_prefix(record, cut, tokenizer):
    first = record["rounds"][0]
    generated = first["generated_token_ids"]
    end_ids = {tokenizer.convert_tokens_to_ids(m) for m in ("</think>", "<summary>", "</summary>")}
    end_ids.add(tokenizer.eos_token_id)
    if cut > len(generated):
        return None, "saved_generation_shorter_than_cut"
    if any(token in end_ids for token in generated[:cut]):
        return None, "thinking_already_ended_at_or_before_cut"
    ids = list(first["input_token_ids"]) + list(generated[:cut])
    if ids[0] != tokenizer.bos_token_id or ids.count(tokenizer.bos_token_id) != 1:
        raise ValueError("Saved prefix violates tokenizer BOS invariant")
    return ids, None


def verify_candidate(candidate, record, ids, reason):
    if (reason or ids != candidate["prefix_token_ids"]
            or candidate["unique_id"] != record["unique_id"]
            or candidate["case"] != record["case"]):
        raise ValueError(f"Candidate does not match saved case {record['case']} cut {candidate['cut']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cases", type=integers, help="Default 1,6; mutually exclusive with --candidates")
    parser.add_argument("--cuts", type=integers, help="Default 128,256; mutually exclusive with --candidates")
    parser.add_argument("--candidates", type=Path, nargs="+", help="One or more branch prepare-only candidates.json files; verify exact saved token IDs")
    parser.add_argument("--query-window", type=int, default=32)
    parser.add_argument("--prefill-chunk", type=int, default=128)
    parser.add_argument("--probe-tokens", type=int, default=32)
    parser.add_argument("--check-short-reference", action="store_true", help="Compare each <=512-total-token prefix with full eager attention in the same dtype/device; longer prefixes are explicitly skipped")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.candidates and (args.cases is not None or args.cuts is not None):
        parser.error("--candidates cannot be combined with --cases or --cuts")
    if not args.candidates:
        args.cases = args.cases or [1, 6]
        args.cuts = args.cuts or [128, 256]
    if min(args.query_window, args.prefill_chunk) < 1 or args.probe_tokens < 0:
        parser.error("Windows/chunks must be positive; probe-tokens may be zero to disable")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if transformers.__version__ != "4.51.3":
        raise RuntimeError(f"Validated API is transformers 4.51.3; found {transformers.__version__}; do not upgrade existing GPU environment blindly")
    if args.device == "cuda" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        raise RuntimeError("CUDA with BF16 support required for --device cuda")
    cfg_path = args.baseline / "run_config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("model_revision") != EXPECTED_REVISION:
        raise ValueError("Baseline model revision differs from audited checkpoint")
    for filename, key in [("reference_runner.py", "reference_runner_sha256"), ("native_batch_check.py", "script_sha256")]:
        if sha(args.baseline / filename) != cfg[key]:
            raise ValueError(f"Baseline hash mismatch: {filename}")
    candidate_lookup = {}
    candidate_hashes = {}
    if args.candidates:
        for candidate_path in args.candidates:
            candidate_hashes[str(candidate_path)] = sha(candidate_path)
            for candidate in json.loads(candidate_path.read_text())["selected"]:
                key = (int(candidate["case"]), int(candidate["cut"]))
                if key[0] <= 0 or key[1] <= 0:
                    raise ValueError("Candidate case/cut must be positive")
                if key in candidate_lookup and candidate_lookup[key]["prefix_token_ids"] != candidate["prefix_token_ids"]:
                    raise ValueError("Conflicting candidate token IDs")
                candidate_lookup[key] = candidate
        if not candidate_lookup:
            raise ValueError("No selected candidates")
        selected_cases = sorted({case for case, cut in candidate_lookup})
    else:
        selected_cases = args.cases
    sources = [(case, args.baseline / "cases" / f"{case:03d}.json") for case in selected_cases]
    records = [(case, path, json.loads(path.read_text())) for case, path in sources]
    if any(case != record["case"] for case, _, record in records):
        raise ValueError("Case index mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "protocol": "offline_tail_attention_boxed_confidence_diagnostic_v1",
        "not_asag_reproduction": True,
        "not_online_controller": True,
        "arguments": {k: [str(p) for p in v] if k == "candidates" and v else str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "candidate_file_sha256": candidate_hashes,
        "baseline_run_config_sha256": sha(cfg_path),
        "baseline_declared_model_revision": cfg["model_revision"],
        "local_weight_revision_independently_verified": False,
        "model_config_sha256": sha(args.model / "config.json"),
        "script_sha256": sha(__file__),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dtype": "bfloat16" if args.device == "cuda" else "float32",
        "attention_backend": "eager, bounded prefill chunks; retain all-layer attention only for small final query window; reduce last four layers",
        "attention_definition": "Raw saved reasoning prefix's final query-window tokens; all causally available keys; last four layers and all heads; sum -p ln p over queries/keys/heads/layers, divide by ln(full key count). Also save query mean and layer/head/query mean. HF eager softmax computes FP32 then casts to model dtype; entropy reduction is FP32.",
        "delta_definition": "Diagnostic (H_at_cut - H_first_successful_requested_cut) / H_first_successful_requested_cut, per case. This is not ASAG first-ATP normalization because these are raw tail windows without ASAG probing instruction.",
        "probe_definition": "Independent greedy boxed answer, historical-style suffix. Arithmetic mean probabilities through closure. Unlike historical C, decoding stops at first closed box; no confidence is assigned to truncated/unclosed output.",
        "cost_scope": "Offline replay and independent probe prefills are charged. Timings exclude checkpoint loading and original native prefix generation. No end-to-end speedup claim.",
    }
    dump(args.output / "run_config.json", config)
    status = {"completed": False, "executed_positions": 0, "skipped_positions": 0}
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                    torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
                    attn_implementation="eager").to(args.device).eval()
        if model.config.model_type != "qwen2":
            raise ValueError("This diagnostic is validated only for Qwen2")
        synchronize(args.device)
        status["model_load_seconds"] = time.perf_counter() - started
        if args.device == "cuda":
            status["gpu"] = torch.cuda.get_device_name()
        for case, path, record in records:
            first_h = None
            cuts = sorted(cut for c, cut in candidate_lookup if c == case) if args.candidates else args.cuts
            for cut in cuts:
                item = {"case": case, "unique_id": record["unique_id"], "cut_generated_tokens": cut,
                        "case_sha256": sha(path)}
                ids, reason = build_prefix(record, cut, tokenizer)
                if args.candidates:
                    candidate = candidate_lookup[(case, cut)]
                    verify_candidate(candidate, record, ids, reason)
                    item["candidate_prefix_verified"] = True
                if reason:
                    item.update(status="skipped", reason=reason)
                    status["skipped_positions"] += 1
                else:
                    if args.device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    item["input_tokens"] = len(ids)
                    item["input_token_ids_sha256"] = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
                    item["attention"], logits = replay_attention(model, ids, args.query_window, args.prefill_chunk, args.device)
                    if args.check_short_reference:
                        if len(ids) <= 512:
                            item["short_reference_comparison"] = short_reference_comparison(
                                model, ids, args.query_window, item["attention"], logits, args.device)
                        else:
                            item["short_reference_comparison"] = {"status": "skipped", "reason": "total_input_tokens_exceed_512"}
                    del logits
                    h = item["attention"]["H_sum_over_layers_heads_queries_div_log_keys"]
                    if first_h is None:
                        first_h = h
                    item["relative_change_from_first_diagnostic_cut"] = (h - first_h) / first_h if first_h > 0 else None
                    item["probe"] = probe_answer(model, tokenizer, ids, args.probe_tokens, args.prefill_chunk, args.device) if args.probe_tokens else None
                    if args.device == "cuda":
                        item["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
                        item["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
                    item["status"] = "measured_review_required"
                    status["executed_positions"] += 1
                dump(args.output / f"case_{case:03d}_cut_{cut:05d}.json", item)
                print(json.dumps({k: item[k] for k in ("case", "cut_generated_tokens", "status")}), flush=True)
        if status["executed_positions"] == 0:
            raise RuntimeError("No positions executed; zero-measurement run is not validation")
        status["completed"] = True
    except Exception as exc:
        status["error"] = repr(exc)
        raise
    finally:
        dump(args.output / "summary.json", status)


if __name__ == "__main__":
    main()
