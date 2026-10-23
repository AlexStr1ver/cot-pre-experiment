#!/usr/bin/env python3
"""Meaningful CPU tests of v2 formulas, windows, probe scopes, pairing, and ordering."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from collect_phase2 import (ASAG_SUFFIX, EXPECTED_REVISION, H_KEY, RAW_KEY, SIGNALS,
    chronological_deltas, compare_chunks, content_probability_diagnostic, first_closed_box,
    ids_sha, measure_modes, probe_asag_style, verify_plan)
from collect_phase2 import compare_probes, logit_diagnostics, precision_settings
from v1_core import attention_summary, sha


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 9
    def encode(self, text, **kwargs):
        assert text == ASAG_SUFFIX
        return [7, 8]
    def decode(self, ids, **kwargs):
        return ''.join({3: '{', 4: '4', 5: '}', 6: '4}', 7: ' ', 8: 'x', 9: '<eos>'}[i] for i in ids)


class ProbeModel:
    def __init__(self, targets=(3, 4, 5)):
        self.targets = targets
    def __call__(self, input_ids, past_key_values, **kwargs):
        length = (past_key_values or 0) + input_ids.shape[1]
        target = self.targets[max(0, min(len(self.targets) - 1, length - 4))]
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, target] = math.log(9)
        return SimpleNamespace(past_key_values=length, logits=logits)


class CollectorTests(unittest.TestCase):
    def test_fp32_configuration_disables_tf32(self):
        settings = precision_settings(torch, 'float32')
        self.assertEqual(settings['float32_matmul_precision'], 'highest')
        self.assertFalse(settings['cuda_matmul_allow_tf32'])
        self.assertFalse(settings['cudnn_allow_tf32'])

    def test_logit_shift_diagnostic_does_not_loosen_original_gate(self):
        a = torch.tensor([[1., 2., 4.]])
        d = logit_diagnostics(a, a + .125)
        self.assertAlmostEqual(d['logit_mean_absolute_difference'], .125)
        self.assertAlmostEqual(d['logit_centered_mean_absolute_difference'], 0.)
        self.assertAlmostEqual(d['probability_total_variation'], 0.)
        signals = {s: {H_KEY: 100.} for s in SIGNALS}
        self.assertFalse(compare_chunks(signals, {s:a for s in SIGNALS},
            signals, {s:a+.125 for s in SIGNALS}, .01, .05)['passed'])
        with self.assertRaises(ValueError):
            logit_diagnostics(a, torch.full_like(a, float('nan')))

    def test_confidence_gate_checks_ids_status_and_probabilities_even_with_null_C(self):
        a = probe_asag_style(ProbeModel(), Tokenizer(), [1, 2], 10, 3, 'cpu')
        b = copy.deepcopy(a)
        self.assertTrue(compare_probes(a, b, .01)['passed'])
        b['selected_token_probabilities'][0] += .02
        self.assertFalse(compare_probes(a, b, .01)['passed'])
        b = copy.deepcopy(a); b['generated_token_ids'][0] = 2
        self.assertFalse(compare_probes(a, b, .01)['passed'])
        a = probe_asag_style(ProbeModel(), Tokenizer(), [1, 2], 2, 3, 'cpu')
        b = copy.deepcopy(a)
        checked = compare_probes(a, b, .01)
        self.assertTrue(checked['passed'])
        self.assertFalse(checked['confidence_available_in_both_probes'])
        b['selected_token_probabilities'][0] += .02
        self.assertFalse(compare_probes(a, b, .01)['passed'])
        b = copy.deepcopy(a); b['status'] = 'empty_closed_box'
        self.assertFalse(compare_probes(a, b, .01)['passed'])

    def test_long_uneven_fp32_cache_and_actual_probe_paths(self):
        torch.manual_seed(20261024)
        torch.set_num_threads(1)
        precision_settings(torch, 'float32')
        cfg = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=48,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=8192, attention_dropout=0, sliding_window=None,
            use_sliding_window=False)
        cfg._attn_implementation = 'eager'
        model = Qwen2ForCausalLM(cfg).eval()
        class LongTok(Tokenizer):
            def encode(self, text, **kwargs):
                return [7, 8, 7, 8, 7, 8]
            def decode(self, ids, **kwargs):
                return ''.join(chr(65 + i % 26) for i in ids)
        tok = LongTok()
        ids = [1] + [10 + i % 70 for i in range(4157)]
        a, la = measure_modes(model, tok, ids, 128, 'cpu')
        b, lb = measure_modes(model, tok, ids, 64, 'cpu')
        self.assertTrue(compare_chunks(a, la, b, lb, 1e-5, 1e-5)['passed'])
        self.assertEqual(a['asag_style_probe_window']['queries'], 12)
        # H replay ends with 12 tokens; normal C prefill ends with 68 or 4.
        with torch.inference_mode():
            _, actual_probe_logits = __import__('v1_core').prefill(model, ids + tok.encode(ASAG_SUFFIX), 128, 'cpu')
        torch.testing.assert_close(actual_probe_logits, la['asag_style_probe_window'], atol=1e-6, rtol=1e-5)
        pa = probe_asag_style(model, tok, ids, 4, 128, 'cpu')
        pb = probe_asag_style(model, tok, ids, 4, 64, 'cpu')
        self.assertTrue(compare_probes(pa, pb, 1e-5)['passed'])

    def test_literal_probe_open_close_probability_and_content_scope(self):
        result = probe_asag_style(ProbeModel(), Tokenizer(), [1, 2], 10, 3, 'cpu')
        self.assertEqual(result['generated_token_ids'], [3, 4, 5])
        self.assertEqual(result['answer'], '4')
        self.assertEqual(result['status'], 'closed_box_review_required')
        self.assertAlmostEqual(result['C_mean_selected_token_probability'], .5, places=6)
        self.assertAlmostEqual(result['format_diagnostic']['strict_content_C'], .5, places=6)
        self.assertEqual(result['format_diagnostic']['strict_content_token_indices'], [1])
        self.assertEqual(result['format_diagnostic']['strict_content_character_coverage'], 1)

    def test_truncated_and_empty_probes_have_null_C(self):
        truncated = probe_asag_style(ProbeModel(), Tokenizer(), [1, 2], 2, 3, 'cpu')
        self.assertEqual(truncated['status'], 'token_limit_without_closed_box')
        self.assertIsNone(truncated['C_mean_selected_token_probability'])
        empty = probe_asag_style(ProbeModel((3, 5)), Tokenizer(), [1, 2], 10, 3, 'cpu')
        self.assertEqual(empty['status'], 'empty_closed_box')
        self.assertIsNone(empty['C_mean_selected_token_probability'])
        self.assertIsNone(empty['format_diagnostic']['strict_content_C'])

    def test_mixed_delimiter_token_excluded_and_uncertain_spans_rejected(self):
        result = content_probability_diagnostic(Tokenizer(), [3, 6], [.9, .6], ASAG_SUFFIX + '{4}')
        self.assertIsNone(result['strict_content_C'])
        self.assertEqual(result['excluded_mixed_token_indices'], [1])
        self.assertEqual(result['strict_content_character_coverage'], 0)
        class Nonmonotonic(Tokenizer):
            def decode(self, ids, **kwargs):
                return {(3,): '\ufffd', (4,): '4', (5,): '}', (3, 4): '{4', (3, 4, 5): '{4}'}[tuple(ids)]
        uncertain = content_probability_diagnostic(Nonmonotonic(), [3, 4, 5], [.8, .2, .9], ASAG_SUFFIX + '{4}')
        self.assertIsNone(uncertain['strict_content_C'])
        self.assertEqual(uncertain['strict_content_unavailable_reason'], 'nonmonotonic_incremental_decoding')

    def test_nested_box_and_non_box_handling(self):
        self.assertEqual(first_closed_box(ASAG_SUFFIX + r'{\frac{1}{2}}')[0], r'\frac{1}{2}')
        self.assertIsNone(first_closed_box(ASAG_SUFFIX + r'{\frac{1}{2}'))
        self.assertEqual(first_closed_box(ASAG_SUFFIX + ' {4} trailing')[0], '4')

    def test_dual_window_chunked_matches_full_tiny_Qwen2(self):
        torch.manual_seed(2026)
        torch.set_num_threads(1)
        cfg = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=48,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, attention_dropout=0, sliding_window=None,
            use_sliding_window=False)
        cfg._attn_implementation = 'eager'
        model = Qwen2ForCausalLM(cfg).eval()
        ids = [1] + list(range(10, 49))
        tok = Tokenizer()
        measured, logits = measure_modes(model, tok, ids, 7, 'cpu')
        alternate, alternate_logits = measure_modes(model, tok, ids, 3, 'cpu')
        comparison = compare_chunks(measured, logits, alternate, alternate_logits, 1e-5, 1e-5)
        self.assertTrue(comparison['passed'])
        for name, seq, q in [('raw_tail32', ids, 32), ('asag_style_probe_window', ids + [7, 8], 4)]:
            with torch.inference_mode():
                full = model(torch.tensor([seq]), use_cache=False, output_attentions=True, logits_to_keep=1)
            expected = attention_summary(tuple(a[:, :, -q:, :] for a in full.attentions))
            self.assertAlmostEqual(measured[name][H_KEY], expected[H_KEY], places=4)
            self.assertEqual(measured[name]['queries'], q)
            self.assertEqual(measured[name]['keys'], len(seq))
            self.assertAlmostEqual(measured[name][RAW_KEY] / math.log(len(seq)), measured[name][H_KEY], places=8)
            torch.testing.assert_close(logits[name], full.logits[:, -1, :], rtol=1e-5, atol=1e-6)
        bad_logits = {name: value + .2 for name, value in alternate_logits.items()}
        self.assertFalse(compare_chunks(measured, logits, alternate, bad_logits, .01, .05)['passed'])

    def test_deltas_chronological_not_canary_order_and_normalization(self):
        def item(cut, raw, keys):
            signal = {RAW_KEY: raw, H_KEY: raw / math.log(keys), 'keys': keys}
            return {'case': 1, 'cut': cut, 'status': 'measured_review_required',
                    'signals': {name: copy.deepcopy(signal) for name in SIGNALS}}
        late, early = item(4096, 110, 4000), item(1024, 100, 1000)
        chronological_deltas([late, early])
        self.assertEqual(late['delta_reference_cut'], 1024)
        self.assertEqual(early['relative_changes']['raw_tail32']['normalized_H'], 0)
        self.assertGreater(late['relative_changes']['raw_tail32']['raw_entropy_sum'], 0)
        self.assertLess(late['relative_changes']['raw_tail32']['normalized_H'], 0)

    def test_plan_hash_identity_and_exact_pairing_rejections(self):
        class BoundaryTok:
            bos_token_id, eos_token_id = 1, 99
            def convert_tokens_to_ids(self, token):
                return {'</think>': 96, '<summary>': 97, '</summary>': 98}[token]
            def decode(self, ids, **kwargs):
                return 'x' * len(ids) + ('\n\n' if len(ids) == 34 else '')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'cases').mkdir()
            for name in ['reference_runner.py', 'native_batch_check.py']:
                (root/name).write_text('# fixture\n')
            cfg = {'model_revision': EXPECTED_REVISION, 'reference_runner_sha256': sha(root/'reference_runner.py'),
                   'script_sha256': sha(root/'native_batch_check.py')}
            (root/'run_config.json').write_text(json.dumps(cfg))
            record = {'case': 1, 'unique_id': 'test/one', 'reference_answer': 'NOT_FOR_PROMPT',
                'rounds': [{'input_token_ids': [1, 2], 'generated_token_ids': [3] * 50}]}
            path = root/'cases/001.json'; path.write_text(json.dumps(record))
            entry = {'case': 1, 'unique_id': 'test/one', 'target': 32, 'cut': 34,
                     'case_sha256': sha(path), 'prefix_token_ids_sha256': ids_sha([1, 2] + [3] * 34), 'input_tokens': 36}
            plan = {'protocol': 'shared_cohort_paragraph_signal_plan_v2', 'model_revision': EXPECTED_REVISION,
                    'baseline_run_config_sha256': sha(root/'run_config.json'),
                    'cohort_cases': [1], 'targets': [32], 'boundary_window': 4, 'selected': [entry]}
            actual = verify_plan(plan, root, BoundaryTok())
            self.assertEqual(actual[0]['input_token_ids'], [1, 2] + [3] * 34)
            self.assertNotIn('reference_answer', actual[0])
            for key, value in [('unique_id', 'other'), ('prefix_token_ids_sha256', '0'*64), ('cut', 33), ('input_tokens', 35)]:
                bad = copy.deepcopy(plan); bad['selected'][0][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    verify_plan(bad, root, BoundaryTok())
            bad = copy.deepcopy(plan); bad['selected'].append(copy.deepcopy(entry))
            with self.assertRaises(ValueError):
                verify_plan(bad, root, BoundaryTok())


if __name__ == '__main__':
    unittest.main(verbosity=2)
