"""Budget-symmetric A/B action state machine. Does not load models or read answer labels."""
from __future__ import annotations

import hashlib
import re
import time

FORCED_SUMMARY_SUFFIX = "\n</think><summary>\n"
CONTROL_TOKENS = ('<think>', '</think>', '<summary>', '</summary>', '<history>', '</history>')


def request_seed(case, cut, replicate_seed, role, index):
    """Branch omitted deliberately: corresponding continuation roles share a seed."""
    text = f'paired_v3|case={case}|cut={cut}|replicate={replicate_seed}|role={role}|index={index}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') % (2**63 - 1)


def last_boxed(text):
    """Last complete balanced boxed expression; only call on final-answer text."""
    found = None
    for match in re.finditer(r'\\boxed\s*\{', text):
        depth = 1
        for i in range(match.end(), len(text)):
            depth += (text[i] == '{') - (text[i] == '}')
            if depth == 0:
                found = text[match.end():i].strip()
                break
    return found


def control_ids(tokenizer):
    markers = {text: tokenizer.convert_tokens_to_ids(text) for text in CONTROL_TOKENS}
    if any(not isinstance(i, int) or i < 0 for i in markers.values()) or len(set(markers.values())) != len(markers):
        raise ValueError('Required single-token control markers missing or aliased')
    for text, value in markers.items():
        if tokenizer.encode(text, add_special_tokens=False) != [value]:
            raise ValueError('Control marker must encode as one token: ' + text)
    return markers


def valid_summary(ids, tokenizer, markers):
    forbidden = set(markers.values()) | {tokenizer.bos_token_id, tokenizer.eos_token_id}
    text = tokenizer.decode(ids, skip_special_tokens=False).strip()
    if not text:
        return None, 'empty_summary'
    if forbidden.intersection(ids) or any(tag in text for tag in CONTROL_TOKENS):
        return None, 'summary_contains_control_marker'
    return text, None


def reset_context(base_ids, summary, tokenizer, markers):
    """Reuse original question/chat IDs; discard all old thought IDs and old summaries."""
    if not base_ids or base_ids[0] != tokenizer.bos_token_id or base_ids.count(tokenizer.bos_token_id) != 1:
        raise ValueError('Reset base must contain exactly one leading BOS')
    if set(base_ids).intersection(markers.values()):
        raise ValueError('Reset base must end before original think and contain no history/control markers')
    text = '<history>' + summary + '</history><think>\n'
    tail = tokenizer.encode(text, add_special_tokens=False)
    ids = list(base_ids) + tail
    decoded = tokenizer.decode(tail, skip_special_tokens=False)
    if decoded != text or ids.count(tokenizer.bos_token_id) != 1:
        raise ValueError('Reset encoding changed exact history or introduced BOS')
    if tail.count(markers['<history>']) != 1 or tail.count(markers['</history>']) != 1 or tail.count(markers['<think>']) != 1:
        raise ValueError('Reset marker counts are invalid')
    return ids, text


def run_action(candidate, branch, replicate_seed, decoder, tokenizer, total_budget=16384,
               summary_budget=1024, heartbeat_tokens=256, on_update=None):
    """Execute one action, preserving KV until a real reset or charged fallback.

    ``on_update`` receives a live JSON-serializable record; callers retaining it in
    memory should copy it. Runtime errors save partial records and re-raise.
    ``record_finished`` means an outcome was obtained, not that its answer is valid.
    """
    if branch not in ('A_continue', 'B_summary'):
        raise ValueError('Unknown branch')
    if min(total_budget, summary_budget, heartbeat_tokens) <= 0 or total_budget <= candidate['cut']:
        raise ValueError('Need positive budgets, heartbeat, and remaining generation budget')
    if candidate['cut'] < 0:
        raise ValueError('Negative saved prefix cut')
    prefix = list(candidate['input_token_ids'])
    base = list(candidate['base_token_ids'])
    if not prefix or prefix[0] != tokenizer.bos_token_id or prefix.count(tokenizer.bos_token_id) != 1:
        raise ValueError('Original prefix must have exactly one leading BOS')
    if prefix[:len(base)] != base:
        raise ValueError('Original prefix does not begin with exact base token IDs')
    markers = control_ids(tokenizer)
    eos = tokenizer.eos_token_id
    if eos is None or eos in markers.values():
        raise ValueError('EOS must be distinct from control markers')
    if tokenizer.bos_token_id == eos:
        raise ValueError('This checkpoint requires distinct tokenizer BOS and EOS')
    if markers['<think>'] not in prefix or any(i in prefix for i in [markers['</think>'], markers['<summary>'], markers['</summary>']]):
        raise ValueError('Action must start inside the first saved thought before any summary')
    record = {
        'case': candidate['case'], 'unique_id': candidate['unique_id'], 'cut': candidate['cut'],
        'prefix_token_ids_sha256': candidate['prefix_token_ids_sha256'],
        'branch': branch, 'replicate_seed': replicate_seed,
        'generation_budget': total_budget - candidate['cut'], 'configured_total_generation_budget': total_budget,
        'summary_generation_cap': summary_budget,
        'status': 'running', 'record_finished': False, 'answer_completed': False,
        'termination_reason': None, 'pred_boxed': None, 'candidate_boxed': None, 'final_text': None,
        'extra_generated_tokens': 0, 'total_generated_tokens_with_shared_prefix': candidate['cut'],
        'events': [], 'forced_summary_succeeded': None if branch == 'A_continue' else False,
        'forced_summary_fallback_planned': False, 'forced_summary_fallback_used': False,
        'executed_summary_restarts': 0, 'inserted_control_tokens': 0,
        'summary_generated_tokens_total': 0,
        'model_call_seconds': 0.0, 'prefill_tokens_total': 0,
        'seed_definition': 'SHA256 of protocol/case/cut/replicate/role/index; corresponding A/B continuation roles share seeds. Not identical random trajectories.',
        'cost_definition': 'All sampled model tokens, including EOS/summary delimiters and failed summaries, are charged. Injected control tokens are input only. Session seconds include host decode/checkpoint overhead; outer branch wall time is recorded separately.',
    }
    current_event, event_started = None, None
    active = False

    def refresh():
        if current_event is not None:
            ids = current_event['generated_token_ids']
            current_event['generated_tokens'] = len(ids)
            current_event['text'] = tokenizer.decode(ids, skip_special_tokens=False)
            if current_event['kind'] == 'forced_summary':
                current_event['summary_generated_tokens'] = len(ids)
            elif markers['<summary>'] in ids:
                current_event['summary_generated_tokens'] = len(ids) - ids.index(markers['<summary>'])
            else:
                current_event['summary_generated_tokens'] = 0
            try:
                stats = decoder.stats() if current_event.get('decoder_start_attempted') else {}
                if stats.get('seed') == current_event['seed']:
                    current_event['decoder_stats'] = stats
            except Exception as exc:
                current_event['stats_error'] = repr(exc)
            if event_started is not None and current_event.get('status') == 'running':
                current_event['seconds'] = time.perf_counter() - event_started
        record['extra_generated_tokens'] = sum(len(e['generated_token_ids']) for e in record['events'])
        record['total_generated_tokens_with_shared_prefix'] = candidate['cut'] + record['extra_generated_tokens']
        record['model_call_seconds'] = sum(e['seconds'] for e in record['events'])
        record['prefill_tokens_total'] = sum(e.get('decoder_stats', {}).get('prefill_input_tokens', 0) for e in record['events'])
        record['summary_generated_tokens_total'] = sum(e.get('summary_generated_tokens', 0) for e in record['events'])
        if record['extra_generated_tokens'] > record['generation_budget']:
            raise AssertionError('Sampled generation budget exceeded')

    def emit():
        refresh()
        if on_update is not None:
            on_update(record)

    def remaining():
        return record['generation_budget'] - record['extra_generated_tokens']

    def begin(ids, kind, role, index):
        nonlocal current_event, event_started, active
        seed = request_seed(candidate['case'], candidate['cut'], replicate_seed, role, index)
        current_event = {
            'kind': kind, 'seed_role': role, 'seed_index': index, 'seed': seed,
            'input_token_ids': list(ids), 'input_tokens': len(ids),
            'generated_token_ids': [], 'generated_tokens': 0, 'summary_generated_tokens': 0, 'text': '',
            'status': 'running', 'seconds': 0.0, 'decoder_stats': {}, 'decoder_start_attempted': False,
        }
        record['events'].append(current_event)
        event_started = time.perf_counter()
        emit()
        active = True  # start may allocate before raising; close it on exceptions.
        current_event['decoder_start_attempted'] = True
        decoder.start(ids, seed)
        emit()
        return current_event

    def sample():
        token = int(decoder.next_token())
        current_event['generated_token_ids'].append(token)
        current_event['generated_tokens'] += 1
        record['extra_generated_tokens'] += 1
        record['total_generated_tokens_with_shared_prefix'] += 1
        if record['extra_generated_tokens'] > record['generation_budget']:
            raise AssertionError('Token sampled beyond remaining budget')
        if current_event['generated_tokens'] % heartbeat_tokens == 0:
            emit()
        return token

    def finish_event(status):
        nonlocal active
        if active:
            decoder.close()
            active = False
        refresh()
        current_event['seconds'] = time.perf_counter() - event_started
        current_event['status'] = status
        emit()

    def outcome(status, reason=None):
        record['status'] = status
        record['termination_reason'] = reason or status
        record['record_finished'] = True
        emit()
        return record

    def prepare_reset(event, summary):
        ids, tail_text = reset_context(base, summary, tokenizer, markers)
        event.update(summary=summary, reset_input_token_ids=ids, reset_prompt=tokenizer.decode(ids, skip_special_tokens=False),
                     reset_tail_text=tail_text, reset_constructed=True, reset_executed=False,
                     reset_input_tokens=len(ids), reset_verified=True)
        return ids

    try:
        emit()
        next_ids = prefix
        pending_reset = None
        if branch == 'B_summary':
            suffix = tokenizer.encode(FORCED_SUMMARY_SUFFIX, add_special_tokens=False)
            record['inserted_control_tokens'] = len(suffix)
            forced = begin(prefix + suffix, 'forced_summary', 'forced_summary', 0)
            forced['inserted_control_token_ids'] = suffix
            forced['sampling_cap'] = min(summary_budget, remaining())
            cap = forced['sampling_cap']
            summary, summary_error, stop = None, None, 'summary_token_limit'
            for _ in range(cap):
                token = sample()
                if token == markers['</summary>']:
                    summary, summary_error = valid_summary(forced['generated_token_ids'][:-1], tokenizer, markers)
                    stop = 'summary_closed' if summary is not None else 'invalid_forced_summary'
                    break
                if token == eos:
                    stop, summary_error = 'eos_without_summary_closure', 'eos_without_summary_closure'
                    break
            if summary is not None:
                record['forced_summary_succeeded'] = True
                next_ids = prepare_reset(forced, summary)
                pending_reset = forced
            else:
                forced['summary_error'] = summary_error or 'summary_token_limit_without_closure'
                record['forced_summary_fallback_planned'] = True
                next_ids = prefix
            finish_event(stop)
        continuation_index = 0
        while remaining() > 0:
            event = begin(next_ids, 'continue_reasoning', 'continue', continuation_index)
            if pending_reset is not None:
                pending_reset['reset_executed'] = True
                record['executed_summary_restarts'] += 1
                pending_reset = None
            if branch == 'B_summary' and record['forced_summary_fallback_planned'] and continuation_index == 0:
                record['forced_summary_fallback_used'] = True
            think_end, summary_start = None, None
            reset_requested = False
            while remaining() > 0:
                token = sample()
                generated = event['generated_token_ids']
                index = len(generated) - 1
                if token == markers['</think>']:
                    if think_end is not None or summary_start is not None:
                        finish_event('invalid_native_control_sequence')
                        return outcome('invalid_native_control_sequence', 'duplicate_or_nested_think_end')
                    think_end = index
                elif token == markers['<summary>']:
                    if think_end is None or summary_start is not None or tokenizer.decode(generated[think_end + 1:index], skip_special_tokens=False).strip():
                        finish_event('invalid_native_summary_order')
                        return outcome('invalid_native_summary_order')
                    summary_start = index
                elif token == markers['</summary>']:
                    if summary_start is None:
                        finish_event('invalid_native_summary_order')
                        return outcome('invalid_native_summary_order', 'summary_close_without_open')
                    summary, error = valid_summary(generated[summary_start + 1:index], tokenizer, markers)
                    if summary is None:
                        event['summary_error'] = error
                        finish_event('invalid_native_summary')
                        return outcome('invalid_native_summary', error)
                    next_ids = prepare_reset(event, summary)
                    pending_reset = event
                    finish_event('native_summary_closed')
                    reset_requested = True
                    break
                elif token in (markers['<think>'], markers['<history>'], markers['</history>'], tokenizer.bos_token_id):
                    finish_event('invalid_native_control_sequence')
                    return outcome('invalid_native_control_sequence', 'unexpected_control_token')
                elif token == eos:
                    if summary_start is not None:
                        finish_event('eos_with_unclosed_native_summary')
                        return outcome('eos_with_unclosed_native_summary')
                    final = tokenizer.decode(generated[think_end + 1:index], skip_special_tokens=False).strip() if think_end is not None else ''
                    if not final:
                        finish_event('eos_without_final_text')
                        return outcome('eos_without_final_text')
                    record['answer_completed'] = True
                    record['final_text'] = final
                    record['pred_boxed'] = last_boxed(final)
                    record['candidate_boxed'] = record['pred_boxed']
                    finish_event('final_eos')
                    return outcome('final_output_review_required')
            if not reset_requested:
                if think_end is not None and summary_start is None:
                    final = tokenizer.decode(event['generated_token_ids'][think_end + 1:], skip_special_tokens=False).strip()
                    record['final_text'] = final or None
                    record['candidate_boxed'] = last_boxed(final)
                    record['pred_boxed'] = record['candidate_boxed']
                finish_event('generation_budget_limit')
                event['budget_ended_with_open_summary'] = summary_start is not None
                return outcome('generation_budget_limit')
            continuation_index += 1
        return outcome('generation_budget_limit')
    except BaseException as exc:
        if active:
            try:
                decoder.close()
            except BaseException as close_exc:
                record['decoder_close_error'] = repr(close_exc)
            active = False
        if current_event is not None:
            try:
                refresh()
            except BaseException as refresh_exc:
                record['partial_refresh_error'] = repr(refresh_exc)
            current_event['status'] = 'execution_error'
            if event_started is not None:
                current_event['seconds'] = time.perf_counter() - event_started
            current_event['error'] = repr(exc)
        record['status'] = 'execution_error'
        record['termination_reason'] = repr(exc)
        record['record_finished'] = False
        record['answer_completed'] = False
        try:
            emit()
        except BaseException as checkpoint_exc:
            record['checkpoint_error'] = repr(checkpoint_exc)
        raise
