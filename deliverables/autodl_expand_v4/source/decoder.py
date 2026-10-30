#!/usr/bin/env python3
"""FP32 HF eager decoding with fresh bounded prefill and persistent decode KV.

This is an explicit sampling implementation, not a promise of token identity with
vLLM or another backend. External orchestration owns budgets and stopping rules.
"""
from __future__ import annotations

import math
import time

import torch


def nucleus_probabilities(logits, temperature=0.7, top_p=0.95):
    """Single sequence: descending nucleus, right-shifted exclusion mask.

Keep the first token taking cumulative mass over top_p. Never impose top_k,
repetition penalties, min_tokens, or a hidden generation_config override.
"""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError('top_p must lie in (0, 1]')
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError('Expected one nonempty vocabulary logits vector')
    scores = logits.float() / temperature
    if not bool(torch.isfinite(scores).all()):
        raise ValueError('Nonfinite logits cannot be sampled')
    if top_p < 1:
        values, indices = torch.sort(scores, descending=True)
        cumulative = values.softmax(-1).cumsum(-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        values = values.masked_fill(remove, -torch.inf)
        scores = torch.empty_like(scores).scatter(0, indices, values)
    return scores.softmax(-1)


class CachedDecoder:
    """One context at a time; start clears KV and resets per-session statistics.

    next_token samples from the current logits. The sampled token is forwarded
    only if another token is requested. Thus stopping does not pay for an unused
    final forward. close releases KV without losing the last session statistics.
    """
    def __init__(self, model, tokenizer, device, prefill_chunk=128,
                 temperature=0.7, top_p=0.95):
        if isinstance(prefill_chunk, bool) or not isinstance(prefill_chunk, int) or prefill_chunk <= 0:
            raise ValueError('prefill_chunk must be a positive integer')
        if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError('Invalid sampling settings')
        self.model, self.tokenizer = model, tokenizer
        self.device = torch.device(device)
        self.prefill_chunk = prefill_chunk
        self.temperature, self.top_p = float(temperature), float(top_p)
        if hasattr(model, 'parameters'):
            parameter = next(model.parameters(), None)
            if parameter is not None and parameter.dtype != torch.float32:
                raise ValueError('Paired diagnostic decoder requires explicit FP32 model weights')
        if getattr(model, 'training', False):
            raise ValueError('Call model.eval() before constructing the decoder')
        self._cache = self._logits = self._pending = self._generator = None
        self._active = False
        self._stats = {}

    def _synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _forward(self, ids):
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        output = self.model(input_ids=inputs, past_key_values=self._cache,
                            use_cache=True, output_attentions=False,
                            return_dict=True, logits_to_keep=1)
        self._cache = output.past_key_values
        self._logits = output.logits[:, -1, :]

    @torch.inference_mode()
    def start(self, input_ids, seed):
        if not isinstance(input_ids, (list, tuple)) or not input_ids:
            raise ValueError('input_ids must be a nonempty list/tuple')
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in input_ids):
            raise ValueError('input_ids must contain nonnegative integer token IDs')
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63)')
        self.close()
        self._stats = {
            'prefill_input_tokens': len(input_ids), 'prefill_forward_calls': 0,
            'forward_generated_tokens': 0, 'sampled_tokens': 0,
            'prefill_seconds': 0.0, 'decode_seconds': 0.0,
            'seed': seed, 'temperature': self.temperature, 'top_p': self.top_p,
            'sampling_algorithm': 'FP32 logits / temperature; descending cumulative probability > top_p; shift mask right to retain first threshold-crossing token; softmax; torch.multinomial with private device Generator; no top_k',
            'timing_scope': 'Prefill is synchronized at its boundaries. Decode sums next_token wall time, including host dispatch and sampling-to-int synchronization, excluding caller processing. Branch wall time is recorded separately by caller.',
        }
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._synchronize()
        started = time.perf_counter()
        try:
            for offset in range(0, len(input_ids), self.prefill_chunk):
                self._forward(input_ids[offset:offset+self.prefill_chunk])
                self._stats['prefill_forward_calls'] += 1
            self._synchronize()
            self._stats['prefill_seconds'] = time.perf_counter() - started
            self._active = True
        except Exception:
            self._stats['prefill_seconds'] = time.perf_counter() - started
            self._cache = self._logits = self._pending = self._generator = None
            self._active = False
            raise

    @torch.inference_mode()
    def next_token(self):
        if not self._active:
            raise RuntimeError('Call start before requesting a token')
        started = time.perf_counter()
        try:
            if self._pending is not None:
                self._forward([self._pending])
                self._stats['forward_generated_tokens'] += 1
                self._pending = None
            probabilities = nucleus_probabilities(self._logits, self.temperature, self.top_p)
            token = int(torch.multinomial(probabilities, 1, generator=self._generator).item())
            self._pending = token
            self._stats['sampled_tokens'] += 1
            return token
        finally:
            self._stats['decode_seconds'] += time.perf_counter() - started

    def close(self):
        self._synchronize()
        self._cache = self._logits = self._pending = self._generator = None
        self._active = False

    def stats(self):
        return dict(self._stats)
