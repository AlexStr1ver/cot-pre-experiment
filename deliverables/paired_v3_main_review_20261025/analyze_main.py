#!/usr/bin/env python3
"""Audit uploaded main18 and join explicit manual grades; no model inference."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit(run, smoke, source, grades):
    root = run / "results"
    config, summary, plan = [read(root / name) for name in
                             ("run_config.json", "summary.json", "plan.json")]
    artifact = read(root / "completed_artifacts.json")
    reuse = read(root / "reuse_manifest.json")
    check((run / "exit_status.txt").read_text().strip() == "0", "Exit status")
    check(summary["mode"] == "main" and summary["completed"] and
          summary["completed_branches"] == summary["planned_branches"] == 18 and
          summary["new_branches"] == 16 and summary["reused_branches"] == 2, "Main completion")
    check(summary["numerical_audit_passed"] and summary["actual_GPU_execution_this_run"], "GPU audit")
    identity = config["experiment_identity"]
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    check(fingerprint == config["experiment_fingerprint"] == artifact["experiment_fingerprint"], "Fingerprint")
    package_count = 0
    for line in (run / "code/checksums.sha256").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        check(not Path(name).is_absolute() and ".." not in Path(name).parts, "Manifest path")
        check(sha(run / "code" / name) == expected == sha(source / name), "Package file: " + name)
        package_count += 1
    for name, expected in identity["code_sha256"].items():
        check(sha(run / "code" / name) == expected, "Executed code: " + name)
    for name in ["run_config.json", "summary.json", "completed_artifacts.json"]:
        check((root / "prior_provenance" / name).read_bytes() == (smoke / "results" / name).read_bytes(),
              "Smoke provenance: " + name)
    check(sha(smoke / "results/run_config.json") == reuse["source_run_config_sha256"], "Reuse config hash")
    for key, item in reuse["branches"].items():
        check(sha(root / "branches" / (key + ".json")) == item["sha256"] ==
              sha(smoke / "results/branches" / (key + ".json")), "Reused record: " + key)
    expected = {f"case_{p['case']:03d}_cut_{p['cut']:05d}_seed_2026_{b}"
                for p in plan["selected"] for b in ["A_continue", "B_summary"]}
    paths = sorted((root / "branches").glob("*.json"))
    check(len(expected) == len(paths) == 18 and expected == {p.stem for p in paths} ==
          {s["key"] for s in config["schedule"]} == set(artifact["branches_sha256"]) == set(grades["branches"]),
          "Schedule, records and manual grades must cover identical 18 branches")
    records = {}
    for path in paths:
        d = read(path)
        grade = grades["branches"][path.stem]
        check(sha(path) == artifact["branches_sha256"][path.stem] == grade["source_branch_sha256"], "Record hash")
        point = next(p for p in plan["selected"] if (p["case"], p["cut"]) == (d["case"], d["cut"]))
        check(d["prefix_token_ids_sha256"] == point["prefix_token_ids_sha256"] and d["replicate_seed"] == 2026,
              "Point identity")
        check(d["record_finished"] and d["answer_completed"] and d["events"][-1]["status"] == "final_eos",
              "Completed EOS answer")
        check(d["events"][-1]["generated_token_ids"][-1] == plan["tokenizer_ids"]["eos"], "EOS token")
        events = d["events"]
        n = sum(len(e["generated_token_ids"]) for e in events)
        check(n == d["extra_generated_tokens"] <= d["generation_budget"] == 16384 - d["cut"], "Token budget")
        check(d["total_generated_tokens_with_shared_prefix"] == n + d["cut"], "Shared cut accounting")
        check(d["prefill_tokens_total"] == sum(len(e["input_token_ids"]) for e in events), "Prefill accounting")
        check(math.isclose(d["model_call_seconds"], sum(e["seconds"] for e in events), rel_tol=1e-10), "Event time")
        for i, e in enumerate(events):
            check(e["generated_tokens"] == len(e["generated_token_ids"]) == e["decoder_stats"]["sampled_tokens"], "Sampled count")
            check(e["decoder_stats"]["forward_generated_tokens"] == e["generated_tokens"] - 1, "Final forward")
            if e.get("reset_executed"):
                check(e["reset_verified"] and e["reset_input_token_ids"] == events[i+1]["input_token_ids"], "Executed reset")
        check(grade["raw_final_box"] == d["pred_boxed"] and isinstance(grade["correct"], bool), "Manual grade target")
        records[path.stem] = d
    check(summary["all_completed_branch_generated_tokens"] == sum(d["extra_generated_tokens"] for d in records.values()), "Summary tokens")
    check(summary["new_generated_tokens"] == sum(d["extra_generated_tokens"] for k,d in records.items() if k not in reuse["branches"]), "New tokens")
    check(math.isclose(summary["newly_executed_branch_wall_seconds"],sum(d["branch_wall_seconds"] for k,d in records.items() if k not in reuse["branches"]),rel_tol=1e-10), "New wall time")
    signals = {(d["case"], d["cut"]): d for d in
               (read(p) for p in sorted((run / "code/evidence_phase2fix/signals").glob("case_*.json")))}
    raw_rows = list(csv.DictReader((root / "branch_review.csv").open(encoding="utf-8-sig")))
    rows, labels = [], collections.Counter()
    for point in sorted(plan["selected"], key=lambda d: (d["case"], d["cut"])):
        prefix = f"case_{point['case']:03d}_cut_{point['cut']:05d}_seed_2026_"
        A, B = [records[prefix + b] for b in ("A_continue", "B_summary")]
        ga, gb = [grades["branches"][prefix + b] for b in ("A_continue", "B_summary")]
        label = {(False,False):"both_wrong",(True,True):"both_correct",(False,True):"B_only_correct",(True,False):"A_only_correct"}[(ga["correct"],gb["correct"])]
        labels[label] += 1
        check(A["events"][0]["input_token_ids"] + B["events"][0]["inserted_control_token_ids"] == B["events"][0]["input_token_ids"], "Shared A/B prefix")
        sig = signals[(point["case"], point["cut"])]
        raw = next(r for r in raw_rows if int(r["case"]) == point["case"] and int(r["cut"]) == point["cut"])
        row = {"case":point["case"],"cut":point["cut"],"target":point["target"],"seed":2026,
               "A_answer":A["pred_boxed"],"B_answer":B["pred_boxed"],"reference":ga["reference"],
               "A_correct_manual":ga["correct"],"B_correct_manual":gb["correct"],"pair_label":label,
               "A_tokens":A["extra_generated_tokens"],"B_tokens":B["extra_generated_tokens"],
               "B_minus_A_tokens":B["extra_generated_tokens"]-A["extra_generated_tokens"],
               "A_wall_seconds":A["branch_wall_seconds"],"B_wall_seconds":B["branch_wall_seconds"],
               "C_answer_reused":sig["probe"]["answer"],"C_correct_reused":raw["C_reused_phase2_integer_correct"] == "True",
               "C_mean_reused":sig["probe"]["C_mean_selected_token_probability"],
               "C_content_diagnostic":sig["probe"]["format_diagnostic"]["strict_content_C"],
               "delta_H_asag_vs_first_selected":float(raw["phase2_delta_H_asag"]),
               "delta_raw_entropy_sum_asag_vs_first_selected":sig["relative_changes"]["asag_style_probe_window"]["raw_entropy_sum"],
               "delta_normalized_raw_tail32_vs_first_selected":sig["relative_changes"]["raw_tail32"]["normalized_H"],
               "delta_raw_entropy_sum_tail32_vs_first_selected":sig["relative_changes"]["raw_tail32"]["raw_entropy_sum"],
               "forced_summary_succeeded":B["forced_summary_succeeded"],"forced_summary_fallback_used":B["forced_summary_fallback_used"]}
        rows.append(row)
    fallback_a = records["case_006_cut_01055_seed_2026_A_continue"]
    fallback_b = records["case_006_cut_01055_seed_2026_B_summary"]
    check(fallback_b["forced_summary_fallback_used"] and fallback_b["events"][0]["generated_tokens"] == 1024,
          "Expected charged fallback")
    check(len(fallback_b["events"][1:]) == len(fallback_a["events"]) and all(
          a[k] == b[k] for a,b in zip(fallback_a["events"],fallback_b["events"][1:])
          for k in ["input_token_ids","generated_token_ids","seed"]), "Fallback continuation must match A")
    audits = [read(p) for p in sorted((root / "numerical_audits").glob("*.json"))]
    check({d["case"] for d in audits} == {1,6,8}, "Numerical audit cohort")
    for d in audits:
        check(d["passed"] and d["resource_passed"] and d["chunk_comparison"]["passed"] and
              d["probe_comparison"]["passed"] and d["cross_run_phase2_comparison"]["probe_comparison"]["passed"], "Numerical audit")
    totals = {}
    for side in ["A","B"]:
        totals[side] = {"correct":sum(r[side+"_correct_manual"] for r in rows),"n_prefixes":len(rows),
                        "tokens":sum(r[side+"_tokens"] for r in rows),"branch_wall_seconds":sum(r[side+"_wall_seconds"] for r in rows)}
    jointly_correct = [r for r in rows if r["pair_label"] == "both_correct"]
    means = {s:{"tokens":sum(r[s+"_tokens"] for r in jointly_correct),
                 "branch_wall_seconds":sum(r[s+"_wall_seconds"] for r in jointly_correct)} for s in ["A","B"]}
    stats = {"date_beijing":"2026-10-25","audit_passed":True,"mode":"main","branches":18,"new_branches":16,"reused_branches":2,
             "package_files_verified":package_count,"events":sum(len(d["events"]) for d in records.values()),
             "executed_resets":sum(d["executed_summary_restarts"] for d in records.values()),
             "manual_unit_equivalence_count":len(grades["manual_unit_equivalence_overrides"]),
             "pair_counts":dict(labels),"totals":totals,
             "relative_token_reduction_descriptive":1-totals["B"]["tokens"]/totals["A"]["tokens"],
             "relative_wall_reduction_descriptive":1-totals["B"]["branch_wall_seconds"]/totals["A"]["branch_wall_seconds"],
             "both_correct_totals":means,
             "forced_summary_successes":sum(r["forced_summary_succeeded"] for r in rows),
             "fallbacks":sum(r["forced_summary_fallback_used"] for r in rows),
             "max_peak_allocated_gib":max(d["peak_allocated_gib"] for d in records.values()),
             "max_chunk_H_relative_difference":max(s["H_relative_difference"] for d in audits for s in d["chunk_comparison"]["signals"].values()),
             "new16_minutes":summary["newly_executed_branch_wall_seconds"]/60,
             "all18_branch_minutes":sum(v["branch_wall_seconds"] for v in totals.values())/60,
             "next_replicate_new_branches":36,"next_replicate_max_tokens":sum(4*(16384-p["cut"]) for p in plan["selected"]),
             "next_similar_lengths_minutes_estimate":2*sum(v["branch_wall_seconds"] for v in totals.values())/60,
             "warning":"Descriptive 9 correlated development prefixes from3 questions/1seed; no trained or online H+C policy, no statistical significance or noninferiority claim. Timings exclude original prefix generation and online monitoring. Seeds add stochastic replicates, not independent questions."}
    return stats, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["run", "smoke", "delivered_source", "manual_grades", "output"]:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    check(not args.output.resolve().is_relative_to(args.run.resolve()), "Never write into raw run")
    stats, rows = audit(args.run,args.smoke,args.delivered_source,read(args.manual_grades))
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/"analysis_summary.json").open("x",encoding="utf-8") as f:
        json.dump(stats,f,ensure_ascii=False,indent=2); f.write("\n")
    with (args.output/"paired_manual_review.csv").open("x",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    print(json.dumps(stats,ensure_ascii=False,indent=2))
