#!/usr/bin/env python3
"""CPU checks for actual tiny-Qwen cache mechanics and sampling semantics."""
import math
from types import SimpleNamespace
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from decoder import CachedDecoder, nucleus_probabilities


class CountingModel:
    training = False
    def __init__(self):
        self.calls = []
    def __call__(self, input_ids, past_key_values, **kwargs):
        previous = 0 if past_key_values is None else past_key_values
        self.calls.append((input_ids.tolist()[0], previous, torch.is_grad_enabled()))
        return SimpleNamespace(past_key_values=previous+input_ids.shape[1],
                               logits=torch.tensor([[[1., 2., 3., 4.]]]))


class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_nucleus_keeps_threshold_crossing_token_and_one_minimum(self):
        logits = torch.tensor([.6, .3, .1]).log()
        actual = nucleus_probabilities(logits, 1, .7)
        torch.testing.assert_close(actual, torch.tensor([2/3, 1/3, 0.]))
        torch.testing.assert_close(nucleus_probabilities(logits, 1, .01), torch.tensor([1., 0., 0.]))
        torch.testing.assert_close(nucleus_probabilities(logits, 1, 1), logits.softmax(-1))
        torch.testing.assert_close(nucleus_probabilities(torch.tensor([1000., 0., -1000.]), .7, .95), torch.tensor([1., 0., 0.]))
        with self.assertRaises(ValueError):
            nucleus_probabilities(torch.tensor([float('nan'), 1.]))

    def test_persistent_cache_and_no_unused_final_forward(self):
        model = CountingModel()
        decoder = CachedDecoder(model, None, 'cpu', prefill_chunk=3)
        decoder.start([1, 2, 3, 1, 2, 3, 1], 2026)
        self.assertEqual([(len(ids), cache) for ids, cache, _ in model.calls], [(3,0),(3,3),(1,6)])
        tokens = [decoder.next_token() for _ in range(8)]
        self.assertEqual(len(tokens), 8)
        self.assertEqual(len(model.calls), 3+7)
        self.assertEqual(model.calls[-1][1], 13)
        self.assertTrue(all(not grad for _, _, grad in model.calls))
        stats = decoder.stats()
        self.assertEqual(stats['sampled_tokens'], 8)
        self.assertEqual(stats['forward_generated_tokens'], 7)
        self.assertEqual(stats['prefill_input_tokens'], 7)
        decoder.close()
        self.assertIsNone(decoder._cache)
        self.assertEqual(decoder.stats(), stats)
        with self.assertRaises(RuntimeError):
            decoder.next_token()

    def test_private_rng_reproducible_and_fresh_start_resets_statistics(self):
        model = CountingModel()
        decoder = CachedDecoder(model, None, 'cpu', prefill_chunk=3, top_p=1)
        torch.manual_seed(42)
        global_state = torch.get_rng_state().clone()
        decoder.start([1,2,3,1], 108)
        first = [decoder.next_token() for _ in range(32)]
        torch.testing.assert_close(torch.get_rng_state(), global_state)
        decoder.start([1,2], 108)
        self.assertEqual(model.calls[-1][1], 0)
        self.assertEqual(decoder.stats()['prefill_input_tokens'], 2)
        self.assertEqual(decoder.stats()['sampled_tokens'], 0)
        second = [decoder.next_token() for _ in range(32)]
        self.assertEqual(first, second)
        decoder.start([1,2], 109)
        third = [decoder.next_token() for _ in range(32)]
        self.assertNotEqual(first, third)

    def test_tiny_Qwen_uneven_prefill_and_cached_decode_match_full(self):
        torch.manual_seed(24)
        config = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=48,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=256, attention_dropout=0, sliding_window=None,
            use_sliding_window=False)
        config._attn_implementation = 'eager'
        model = Qwen2ForCausalLM(config).eval()
        decoder = CachedDecoder(model, None, 'cpu', prefill_chunk=7)
        ids = [1] + list(range(10, 53))
        decoder.start(ids, 2026)
        with torch.inference_mode():
            expected = model(torch.tensor([ids]), use_cache=False, logits_to_keep=1).logits[:, -1, :]
        torch.testing.assert_close(decoder._logits, expected, atol=1e-6, rtol=1e-5)
        generated = []
        for _ in range(6):
            generated.append(decoder.next_token())
            # Last sampled token remains pending; logits condition on earlier samples.
            with torch.inference_mode():
                full = model(torch.tensor([ids+generated[:-1]]), use_cache=False, logits_to_keep=1)
            torch.testing.assert_close(decoder._logits, full.logits[:, -1, :], atol=1e-6, rtol=1e-5)
            self.assertEqual(decoder._cache.get_seq_length(), len(ids)+len(generated)-1)
        decoder.close()

    def test_invalid_inputs_and_precision_rejected(self):
        with self.assertRaises(ValueError):
            CachedDecoder(CountingModel(), None, 'cpu', prefill_chunk=0)
        for temperature, top_p in [(0,.9), (math.nan,.9), (.7,0), (.7,1.1)]:
            with self.assertRaises(ValueError):
                CachedDecoder(CountingModel(), None, 'cpu', temperature=temperature, top_p=top_p)
        decoder = CachedDecoder(CountingModel(), None, 'cpu')
        for ids, seed in [([],1), ([True],1), ([1],-1), ([1],True)]:
            with self.assertRaises(ValueError):
                decoder.start(ids, seed)
        with self.assertRaises(ValueError):
            CachedDecoder(torch.nn.Linear(2,2).eval().to(torch.bfloat16), None, 'cpu')


if __name__ == '__main__':
    unittest.main(verbosity=2)
