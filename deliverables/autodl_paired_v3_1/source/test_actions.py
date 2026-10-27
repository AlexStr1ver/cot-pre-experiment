"""CPU-only state-machine tests with explicit scripted model outputs."""
import copy
import unittest

from actions import (CONTROL_TOKENS, FORCED_SUMMARY_SUFFIX, last_boxed,
                     request_seed, run_action)


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    special = {'<bos>': 1, '<eos>': 2, **{t: i + 3 for i, t in enumerate(CONTROL_TOKENS)}}
    def convert_tokens_to_ids(self, text):
        return self.special.get(text, -1)
    def encode(self, text, **kwargs):
        result = []
        while text:
            found = next((t for t in self.special if text.startswith(t)), None)
            if found:
                result.append(self.special[found]); text = text[len(found):]
            else:
                result.append(ord(text[0]) + 100); text = text[1:]
        return result
    def decode(self, ids, **kwargs):
        special = {v: k for k, v in self.special.items()}
        return ''.join(special[i] if i in special else chr(i - 100) for i in ids)


class ScriptedDecoder:
    def __init__(self, tokenizer, outputs):
        self.outputs = [tokenizer.encode(s) if isinstance(s, str) else list(s) for s in outputs]
        self.starts, self.position, self.session, self.closed = [], 0, -1, True
        self.meta = {}
    def start(self, ids, seed):
        if not self.closed:
            raise AssertionError('Old cache was not closed')
        self.session += 1; self.position = 0; self.closed = False
        if self.session >= len(self.outputs):
            raise AssertionError('Unexpected extra prefill/reset')
        self.starts.append((list(ids), seed))
        self.meta = {'seed': seed, 'prefill_input_tokens': len(ids), 'prefill_forward_calls': 1,
                     'forward_generated_tokens': 0, 'sampled_tokens': 0,
                     'prefill_seconds': 0., 'decode_seconds': 0.}
    def next_token(self):
        token = self.outputs[self.session][self.position]
        if isinstance(token, BaseException):
            raise token
        self.position += 1
        self.meta['sampled_tokens'] = self.position
        self.meta['forward_generated_tokens'] = max(0, self.position - 1)
        return token
    def close(self):
        self.closed = True
    def stats(self):
        return dict(self.meta)


class ActionsTests(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()
        self.base = [self.tok.bos_token_id] + self.tok.encode('question:Q\nassistant:')
        self.prefix = self.base + self.tok.encode('<think>\nold reasoning')
        self.candidate = {'case': 1, 'cut': 7, 'unique_id': 'fixture/one',
                          'input_token_ids': self.prefix, 'base_token_ids': self.base,
                          'prefix_token_ids_sha256': 'fixture-hash'}
    def run_script(self, outputs, branch='A_continue', remaining=100, **kwargs):
        decoder = ScriptedDecoder(self.tok, outputs)
        result = run_action(self.candidate, branch, 2026, decoder, self.tok,
                            total_budget=self.candidate['cut'] + remaining, **kwargs)
        self.assertTrue(decoder.closed)
        self.assertEqual(result['extra_generated_tokens'], sum(len(e['generated_token_ids']) for e in result['events']))
        self.assertEqual(result['total_generated_tokens_with_shared_prefix'], self.candidate['cut'] + result['extra_generated_tokens'])
        self.assertLessEqual(result['extra_generated_tokens'], remaining)
        return result, decoder

    def test_A_crosses_old_first_call_limit_and_heartbeat_does_not_reset(self):
        self.candidate['cut'] = 8190  # Old code would stop after just 2 tokens.
        updates = []
        output = 'abcdef</think>answer \\boxed{6}<eos>'
        result, decoder = self.run_script([output], remaining=50, heartbeat_tokens=2,
                                         on_update=lambda r: updates.append(copy.deepcopy(r)))
        self.assertTrue(result['answer_completed'])
        self.assertEqual(result['pred_boxed'], '6')
        self.assertEqual(len(decoder.starts), 1)
        self.assertGreater(len(updates), 5)
        self.assertEqual(result['extra_generated_tokens'], len(self.tok.encode(output)))
        self.assertEqual(result['events'][0]['decoder_stats']['forward_generated_tokens'], result['extra_generated_tokens'] - 1)
        self.assertTrue(any(u['extra_generated_tokens'] == 4 and u['status'] == 'running' for u in updates))

    def test_B_summary_is_charged_reset_removes_thought_and_continuation_seed_matches_A(self):
        result, decoder = self.run_script(['saved fact</summary>', '</think>\\boxed{6}<eos>'], 'B_summary', remaining=60)
        self.assertTrue(result['forced_summary_succeeded'])
        self.assertEqual(result['executed_summary_restarts'], 1)
        self.assertFalse(result['forced_summary_fallback_used'])
        self.assertEqual(decoder.starts[0][0], self.prefix + self.tok.encode(FORCED_SUMMARY_SUFFIX))
        reset = self.tok.decode(decoder.starts[1][0])
        self.assertEqual(reset, self.tok.decode(self.base) + '<history>saved fact</history><think>\n')
        self.assertNotIn('old reasoning', reset)
        self.assertEqual(decoder.starts[1][0].count(self.tok.bos_token_id), 1)
        self.assertEqual(result['extra_generated_tokens'], len(self.tok.encode('saved fact</summary></think>\\boxed{6}<eos>')))
        a, _ = self.run_script(['</think>\\boxed{6}<eos>'])
        self.assertEqual(result['events'][1]['seed'], a['events'][0]['seed'])
        self.assertNotEqual(result['events'][0]['seed'], result['events'][1]['seed'])

    def test_failed_forced_summary_falls_back_without_free_tokens(self):
        result, decoder = self.run_script(['abc', 'z' * 20], 'B_summary', remaining=11, summary_budget=3)
        self.assertFalse(result['forced_summary_succeeded'])
        self.assertTrue(result['forced_summary_fallback_used'])
        self.assertEqual(result['extra_generated_tokens'], 11)
        self.assertEqual([e['generated_tokens'] for e in result['events']], [3, 8])
        self.assertEqual(result['summary_generated_tokens_total'], 3)
        self.assertEqual(decoder.starts[1][0], self.prefix)
        self.assertEqual(result['executed_summary_restarts'], 0)
        self.assertEqual(result['status'], 'generation_budget_limit')
        self.assertFalse(result['answer_completed'])
        self.assertTrue(result['record_finished'])

    def test_empty_and_malformed_forced_summaries_are_charged_fallbacks(self):
        for text in ['</summary>', 'x<eos>', '<think>x</summary>']:
            with self.subTest(text=text):
                result, decoder = self.run_script([text, '</think>ok<eos>'], 'B_summary', remaining=30)
                self.assertFalse(result['forced_summary_succeeded'])
                self.assertTrue(result['forced_summary_fallback_used'])
                self.assertTrue(result['answer_completed'])
                self.assertEqual(decoder.starts[1][0], self.prefix)
                self.assertEqual(result['extra_generated_tokens'], len(self.tok.encode(text + '</think>ok<eos>')))

    def test_A_and_B_keep_same_remaining_budget_and_no_segment_limit(self):
        outputs = ['</think><summary>s1</summary>', '</think>\n<summary> s2 </summary>',
                   '</think><summary>s3</summary>', '</think>\\boxed{9}<eos>']
        a, _ = self.run_script(outputs, remaining=100)
        b, _ = self.run_script(['force</summary>'] + outputs, 'B_summary', remaining=100)
        self.assertEqual(a['generation_budget'], b['generation_budget'])
        self.assertEqual(a['executed_summary_restarts'], 3)
        self.assertEqual(b['executed_summary_restarts'], 4)
        self.assertEqual(a['pred_boxed'], b['pred_boxed'])
        self.assertEqual(b['extra_generated_tokens'] - a['extra_generated_tokens'], len(self.tok.encode('force</summary>')))
        self.assertEqual(a['summary_generated_tokens_total'], len(self.tok.encode('<summary>s1</summary><summary> s2 </summary><summary>s3</summary>')))
        self.assertTrue(a['record_finished'] and b['record_finished'])

    def test_native_malformed_summary_and_eos_are_explicit_model_outcomes(self):
        cases = {
            '<summary>x</summary>': 'invalid_native_summary_order',
            '</summary>': 'invalid_native_summary_order',
            '</think><summary> </summary>': 'invalid_native_summary',
            '</think><summary>x<eos>': 'eos_with_unclosed_native_summary',
            'thought only<eos>': 'eos_without_final_text',
            '</think><eos>': 'eos_without_final_text',
            '</think><think>': 'invalid_native_control_sequence',
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                result, _ = self.run_script([text])
                self.assertEqual(result['status'], expected)
                self.assertTrue(result['record_finished'])
                self.assertFalse(result['answer_completed'])

    def test_boxed_in_thought_is_not_final_and_budget_box_is_provisional(self):
        result, _ = self.run_script(['\\boxed{1} and more<eos>'])
        self.assertIsNone(result['pred_boxed'])
        self.assertFalse(result['answer_completed'])
        text = '</think>\\boxed{6}'
        result, _ = self.run_script([text], remaining=len(self.tok.encode(text)))
        self.assertEqual(result['status'], 'generation_budget_limit')
        self.assertFalse(result['answer_completed'])
        self.assertEqual(result['candidate_boxed'], '6')
        self.assertEqual(result['final_text'], '\\boxed{6}')

    def test_EOS_on_last_budget_token_and_summary_exhaustion(self):
        text = '</think>done<eos>'
        result, _ = self.run_script([text], remaining=len(self.tok.encode(text)))
        self.assertTrue(result['answer_completed'])
        self.assertIsNone(result['pred_boxed'])
        text = 's</summary>'
        result, decoder = self.run_script([text], 'B_summary', remaining=len(self.tok.encode(text)))
        self.assertTrue(result['forced_summary_succeeded'])
        self.assertEqual(result['executed_summary_restarts'], 0)
        self.assertFalse(result['events'][0]['reset_executed'])
        self.assertEqual(len(decoder.starts), 1)
        self.assertEqual(result['status'], 'generation_budget_limit')

    def test_execution_error_saves_partial_cost_and_reraises(self):
        updates = []
        decoder = ScriptedDecoder(self.tok, [[self.tok.encode('a')[0], RuntimeError('fixture CUDA OOM')]])
        with self.assertRaisesRegex(RuntimeError, 'fixture CUDA OOM'):
            run_action(self.candidate, 'A_continue', 2026, decoder, self.tok,
                       on_update=lambda r: updates.append(copy.deepcopy(r)), heartbeat_tokens=1)
        last = updates[-1]
        self.assertEqual(last['status'], 'execution_error')
        self.assertFalse(last['record_finished'])
        self.assertFalse(last['answer_completed'])
        self.assertEqual(last['extra_generated_tokens'], 1)
        self.assertEqual(last['events'][0]['text'], 'a')
        self.assertTrue(decoder.closed)

    def test_seed_mapping_and_balanced_nested_last_box(self):
        a = request_seed(1, 4103, 2026, 'continue', 0)
        self.assertEqual(a, request_seed(1, 4103, 2026, 'continue', 0))
        for values in [(1,4104,2026,'continue',0),(1,4103,2027,'continue',0),
                       (1,4103,2026,'forced_summary',0),(1,4103,2026,'continue',1)]:
            self.assertNotEqual(a, request_seed(*values))
        self.assertEqual(last_boxed(r'\boxed{1} then \boxed{\frac{1}{2}}'), r'\frac{1}{2}')
        self.assertEqual(last_boxed(r'\boxed{1} then \boxed{incomplete'), '1')

    def test_real_tiny_Qwen_decoder_action_lifecycle(self):
        """Random tiny model is an interface fixture, never task accuracy evidence."""
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from decoder import CachedDecoder
        class CompleteVocabTokenizer(Tokenizer):
            def decode(self, ids, **kwargs):
                special = {v: k for k, v in self.special.items()}
                return ''.join(special[i] if i in special else (chr(i - 100) if i >= 100 else chr(0xe000 + i)) for i in ids)
        torch.manual_seed(20261024)
        torch.set_num_threads(1)
        cfg = Qwen2Config(vocab_size=256, hidden_size=32, intermediate_size=48,
                          num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=256, attention_dropout=0,
                          sliding_window=None, use_sliding_window=False)
        cfg._attn_implementation = 'eager'
        model = Qwen2ForCausalLM(cfg).eval()
        tokenizer = CompleteVocabTokenizer()
        decoder = CachedDecoder(model, tokenizer, 'cpu', prefill_chunk=7)
        result = run_action(self.candidate, 'A_continue', 2026, decoder, tokenizer,
                            total_budget=self.candidate['cut'] + 6, heartbeat_tokens=2)
        self.assertEqual(result['status'], 'generation_budget_limit')
        self.assertEqual(result['extra_generated_tokens'], 6)
        self.assertEqual(len(result['events']), 1)
        stats = result['events'][0]['decoder_stats']
        self.assertEqual(stats['sampled_tokens'], 6)
        self.assertEqual(stats['forward_generated_tokens'], 5)
        self.assertEqual(stats['prefill_input_tokens'], len(self.prefix))
        self.assertEqual(result['prefill_tokens_total'], len(self.prefix))
        self.assertFalse(result['answer_completed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
