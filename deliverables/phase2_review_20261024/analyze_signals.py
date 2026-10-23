#!/usr/bin/env python3
"""Read-only analysis of the integer-answer phase2fix cohort; no inference/training."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--result', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--previous-bf16', type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted((args.result / 'signals').glob('case_*.json'))
    points, rows = [read(p) for p in paths], []
    run_summary = read(args.result / 'signals/summary.json')
    assert run_summary['completed'] and run_summary['executed_positions'] == len(points)
    for point in points:
        record = read(args.baseline / 'cases' / f"{point['case']:03d}.json")
        probe = point['probe']
        answer, reference = probe['answer'].strip(), str(record['reference_answer']).strip()
        if not all(re.fullmatch(r'[+-]?\d+', text) for text in [answer, reference]):
            raise ValueError('This analysis grades integer answers only; other forms require review')
        assert point['unique_id'] == record['unique_id']
        assert point['status'] == 'measured_review_required'
        probs = probe['selected_token_probabilities']
        assert len(probs) == probe['generated_tokens'] == len(probe['generated_token_ids'])
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probs)
        assert math.isclose(sum(probs) / len(probs), probe['C_mean_selected_token_probability'], abs_tol=1e-12)
        diag = probe['format_diagnostic']
        begin, end = diag['answer_content_generated_character_span']
        spans = diag['generated_token_character_spans']
        assert begin < end and len(spans) == len(probs)
        overlapping = [i for i, (lo, hi) in enumerate(spans) if max(lo, begin) < min(hi, end)]
        covering_mean = sum(probs[i] for i in overlapping) / len(overlapping)
        covered = sum(min(spans[i][1], end) - max(spans[i][0], begin) for i in overlapping)
        assert covered == end - begin
        row = {'case': point['case'], 'target': point['target'], 'cut': point['cut'],
               'input_tokens': point['input_tokens'], 'answer': answer, 'reference': reference,
               'correct_integer_answer': int(answer) == int(reference),
               'C_all_generated_tokens': probe['C_mean_selected_token_probability'],
               'C_strict_content_tokens': diag['strict_content_C'],
               'strict_content_character_coverage': diag['strict_content_character_coverage'],
               'C_content_overlapping_tokens_including_mixed_format': covering_mean,
               'content_overlapping_token_indices': json.dumps(overlapping),
               'closure_token_probability': probe['closure_token_probability'],
               'generated_token_ids': json.dumps(probe['generated_token_ids']),
               'selected_token_probabilities': json.dumps(probs),
               'token_text_pieces': json.dumps(diag['token_text_pieces'], ensure_ascii=False),
               'delta_reference_cut': point['delta_reference_cut']}
        for name, prefix in [('raw_tail32', 'tail32'), ('asag_style_probe_window', 'asag12')]:
            signal = point['signals'][name]
            row.update({prefix + '_raw_entropy_sum_nats': signal['entropy_sum_nats'],
                        prefix + '_normalized_H_mean': signal['H_mean_over_layers_heads_queries_div_log_keys'],
                        prefix + '_keys': signal['keys'],
                        prefix + '_delta_normalized_H': point['relative_changes'][name]['normalized_H'],
                        prefix + '_delta_raw_entropy': point['relative_changes'][name]['raw_entropy_sum']})
        row.update(measurement_seconds=point['measurement_seconds_excluding_canary_audit'],
                   peak_allocated_gib=point['peak_allocated_gib'], peak_reserved_gib=point['peak_reserved_gib'])
        rows.append(row)
    with (args.output / 'signals_review.csv').open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    paired_precision = []
    if args.previous_bf16:
        for path in sorted((args.previous_bf16 / 'signals').glob('case_*.json')):
            old = read(path)
            new = next((p for p in points if (p['case'], p['cut']) == (old['case'], old['cut'])), None)
            if new is None:
                continue
            assert old['prefix_token_ids_sha256'] == new['prefix_token_ids_sha256']
            for name in old['signals']:
                key = 'H_sum_over_layers_heads_queries_div_log_keys'
                a, b = old['signals'][name][key], new['signals'][name][key]
                paired_precision.append({'case': old['case'], 'cut': old['cut'], 'window': name,
                                         'BF16_H': a, 'FP32_H': b, 'relative_difference_FP32_vs_BF16': (b-a)/a,
                                         'interpretation': 'Same saved prefix, separate runs; not a controlled kernel-cause experiment'})
    positive = [r for r in rows if r['correct_integer_answer']]
    negative = [r for r in rows if not r['correct_integer_answer']]
    result = {'grading_method': 'Exact integer match against saved original reference; no learned grader',
              'points': len(rows), 'distinct_questions': len({r['case'] for r in rows}),
              'correct_probes': len(positive), 'closed_nonempty_probes': run_summary['closed_nonempty_probes'],
              'correct_points': [[r['case'],r['cut']] for r in positive],
              'rows_are_within_question_correlated_development_observations': True,
              'minimum_correct_C': min((r['C_all_generated_tokens'] for r in positive), default=None),
              'maximum_wrong_C': max((r['C_all_generated_tokens'] for r in negative), default=None),
              'partial_content_coverage_points': [[r['case'],r['cut']] for r in rows if r['strict_content_character_coverage'] < 1],
              'supplementary_C_definition': 'Mean over tokens with any answer-content overlap. Keeps mixed format/content tokens whole. Not a pure-content or calibrated correctness probability, and does not replace primary C.',
              'precision_comparison': paired_precision,
              'measurement_seconds': sum(r['measurement_seconds'] for r in rows),
              'attention_seconds': sum(s['replay_seconds_including_prefill_and_entropy_reduction'] for p in points for s in p['signals'].values()),
              'probe_seconds': sum(p['probe']['seconds_including_independent_prefill'] for p in points),
              'additional_audit_seconds': run_summary['canary_audit_seconds_sum'],
              'model_load_seconds': run_summary['model_load_seconds'],
              'max_allocated_gib': max(r['peak_allocated_gib'] for r in rows),
              'max_reserved_gib': max(r['peak_reserved_gib'] for r in rows),
              'source_point_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    a = next(r for r in rows if (r['case'],r['cut']) == (8,2055))
    b = next(r for r in rows if (r['case'],r['cut']) == (8,4173))
    assert len(json.loads(a['selected_token_probabilities'])) == len(json.loads(b['selected_token_probabilities'])) == 3
    drop = a['C_all_generated_tokens'] - b['C_all_generated_tokens']
    result['case8_later_C_drop'] = {'drop': drop,
        'closure_contribution_to_drop': (a['closure_token_probability']-b['closure_token_probability'])/3,
        'closure_fraction_of_drop': (a['closure_token_probability']-b['closure_token_probability'])/3/drop}
    (args.output / 'analysis_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
