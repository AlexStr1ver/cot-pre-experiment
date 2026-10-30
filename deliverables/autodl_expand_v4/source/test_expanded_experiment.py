"""Portable CPU expansion checks using explicitly synthetic model/decoder fixtures.

These tests run the real coordinator, scheduling, budget and resume logic. They
do not load real model weights or claim any GPU or scientific result.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import actions
import expanded_experiment as exp
import paired_experiment as old_exp
from test_actions import ScriptedDecoder, Tokenizer


def frozen_plan():
    return exp.read(Path(__file__).with_name('expansion_plan.json'))


def fixture_candidates():
    tok = Tokenizer()
    base = [tok.bos_token_id] + tok.encode('Question fixture\nAssistant:')
    prefix = base + tok.encode('<think>\nfixture reasoning')
    return [{**entry, 'input_token_ids': list(prefix), 'base_token_ids': list(base),
             'prefix_token_ids_sha256': exp.ids_sha(prefix)} for entry in frozen_plan()['selected']]


class FixtureDecoder(ScriptedDecoder):
    """Deliberately wrong final answers: expansion cannot select only winners."""
    def __init__(self, model, tokenizer, device, **kwargs):
        self.tok = tokenizer
        super().__init__(tokenizer, [])

    def start(self, ids, seed):
        output = ('fixture summary</summary>' if self.tok.decode(ids).endswith('\n</think><summary>\n')
                  else '</think>\\boxed{999}<eos>')
        self.outputs.append(self.tok.encode(output))
        super().start(ids, seed)


class DummyModel:
    def to(self, device): return self
    def eval(self): return self
    def parameters(self): return iter([torch.zeros(1)])


class ExpandedExperimentTests(unittest.TestCase):
    def fixture(self, root):
        phase = root / 'evidence'
        phase.mkdir()
        plan = frozen_plan()
        # Exact bytes are required by the frozen-plan provenance check.
        (phase / 'plan.json').write_bytes(Path(__file__).with_name('expansion_plan.json').read_bytes())
        (phase / 'evidence_manifest.json').write_text('{}\n')
        baseline = root / 'baseline'
        (baseline / 'cases').mkdir(parents=True)
        for case in plan['cohort_cases']:
            exp.dump(baseline / 'cases' / f'{case:03d}.json', {'reference_answer': '0'})
        candidates = fixture_candidates()
        points = {(c['case'], c['cut']): {
            'probe': {'answer': '999', 'C_mean_selected_token_probability': .8},
            'relative_changes': {'asag_style_probe_window': {'normalized_H': 0.}},
        } for c in candidates}
        return phase, baseline, plan, candidates, points

    def run_fixture(self, root, inputs, name, source=None, interrupt_after=None,
                    fail_audit=False, bad_weights=False):
        phase, baseline, plan, candidates, points = inputs
        output = root / name
        argv = ['--model', str(root / 'model'), '--baseline', str(baseline),
                '--phase2-input', str(phase), '--output', str(output), '--device', 'cpu']
        if source is not None:
            argv += ['--resume-from', str(source)]
        actual_action = actions.run_action
        completed = []
        audits = []

        def action(*args, **kwargs):
            if interrupt_after is not None and len(completed) == interrupt_after:
                raise InterruptedError('fixture interruption after committed branches')
            record = actual_action(*args, **kwargs)
            completed.append(record)
            return record

        def audit(model, tokenizer, candidate, saved, device, path):
            audits.append((candidate['case'], candidate['cut']))
            if fail_audit:
                raise RuntimeError('fixture numerical audit failure')
            exp.dump(path, {'passed': True, 'fixture_only': True})
            return {'passed': True}

        weights = copy.deepcopy(exp.EXPECTED_WEIGHTS)
        if bad_weights:
            weights[0]['sha256'] = 'different content'
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()), \
             patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=DummyModel()) as model_load, \
             patch.object(exp, 'load_evidence', return_value=(plan, candidates, points, {})), \
             patch.object(exp, 'weight_fingerprint', return_value=weights), \
             patch.object(exp, 'audit_point', side_effect=audit), \
             patch('decoder.CachedDecoder', FixtureDecoder), \
             patch('actions.run_action', side_effect=action), \
             patch.object(exp.signal, 'signal'), contextlib.redirect_stdout(io.StringIO()):
            try:
                exp.main(argv)
            finally:
                self.last_model_load_count = model_load.call_count
                self.last_actions = len(completed)
                self.last_audits = audits
        return output

    def test_protocol_and_schedule_remain_frozen_without_quality_selection(self):
        self.assertEqual(exp.PROTOCOL, old_exp.PROTOCOL)
        plan = frozen_plan()
        self.assertTrue(exp.validate_frozen_plan(plan))
        jobs = exp.schedule(fixture_candidates())
        self.assertEqual(len(jobs), 60)
        self.assertEqual(len({exp.job_key(*j) for j in jobs}), 60)
        self.assertEqual({c['case'] for c, _, _ in jobs}, {10, 11, 12, 15, 19})
        self.assertEqual({s for _, _, s in jobs}, {2026, 2027})
        self.assertEqual(sum(16384 - c['cut'] for c, _, _ in jobs), 837400)
        for index in range(0, 60, 2):
            self.assertEqual(jobs[index][0], jobs[index + 1][0])
            self.assertEqual(jobs[index][2], jobs[index + 1][2])
            self.assertEqual({jobs[index][1], jobs[index + 1][1]}, {'A_continue', 'B_summary'})
        self.assertEqual(jobs[0][1], 'A_continue')
        self.assertEqual(jobs[30][1], 'B_summary')
        for candidates in [fixture_candidates()[:-1], fixture_candidates() + fixture_candidates()[:1]]:
            with self.assertRaises(ValueError): exp.schedule(candidates)

    def test_frozen_plan_rejects_changed_seed_cohort_or_prefix(self):
        for field in ('replicate_seeds', 'cohort_cases', 'selected'):
            plan = frozen_plan()
            if field == 'selected': plan[field][0]['cut'] += 1
            else: plan[field] = list(reversed(plan[field]))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'frozen'):
                exp.validate_frozen_plan(plan)

    def test_full_coordinator_runs_all_sixty_and_audits_each_longest_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs = self.fixture(root)
            output = self.run_fixture(root, inputs, 'all')
            summary = exp.read(output / 'summary.json')
            self.assertTrue(summary['completed'])
            self.assertEqual((summary['planned_branches'], summary['completed_branches'],
                              summary['new_branches'], summary['reused_branches']), (60, 60, 60, 0))
            self.assertFalse(summary['actual_GPU_execution_this_run'])
            expected = [(case, max(c['cut'] for c in inputs[3] if c['case'] == case))
                        for case in exp.EXPANSION['cases']]
            self.assertEqual(self.last_audits, expected)
            pairs = exp.read(output / 'paired_review.json')['pairs']
            self.assertEqual(len(pairs), 30)
            self.assertTrue(all(p['both_strict_complete_correct'] is False for p in pairs))
            config = exp.read(output / 'run_config.json')
            self.assertEqual(config['experiment_fingerprint'], exp.object_sha(config['experiment_identity']))
            self.assertEqual(set(config['experiment_identity']['code_sha256']), set(exp.CODE_FILES))

    def test_partial_resume_reuses_only_committed_records_without_editing_source(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs = self.fixture(root)
            with self.assertRaisesRegex(InterruptedError, 'fixture interruption'):
                self.run_fixture(root, inputs, 'partial', interrupt_after=2)
            source = root / 'partial'
            self.assertEqual(exp.read(source / 'summary.json')['completed_branches'], 2)
            self.assertTrue(exp.read(source / 'summary.json')['numerical_audit_passed'])
            third = exp.job_key(*exp.schedule(inputs[3])[2])
            exp.dump(source / 'branches' / (third + '.json'), {'record_finished': False, 'status': 'in_progress'})
            snapshots = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
            output = self.run_fixture(root, inputs, 'resumed', source=source)
            summary = exp.read(output / 'summary.json')
            self.assertEqual((summary['completed_branches'], summary['new_branches'], summary['reused_branches']),
                             (60, 58, 2))
            self.assertEqual(self.last_actions, 58)
            self.assertEqual(len(self.last_audits), 5)
            self.assertEqual(snapshots, {str(p.relative_to(source)): p.read_bytes()
                                        for p in source.rglob('*') if p.is_file()})
            reused = exp.read(output / 'reuse_manifest.json')['branches']
            self.assertEqual(len(reused), 2)
            for key in reused:
                self.assertEqual((source / 'branches' / (key + '.json')).read_bytes(),
                                 (output / 'branches' / (key + '.json')).read_bytes())

    def test_failed_current_audit_prevents_every_action(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs = self.fixture(root)
            with self.assertRaisesRegex(RuntimeError, 'numerical audit'):
                self.run_fixture(root, inputs, 'failed', fail_audit=True)
            summary = exp.read(root / 'failed/summary.json')
            self.assertFalse(summary['numerical_audit_passed'])
            self.assertEqual(summary['completed_branches'], 0)
            self.assertEqual(list((root / 'failed/branches').glob('*.json')), [])
            self.assertEqual(self.last_actions, 0)

    def test_changed_weight_content_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'Local weights differ'):
                self.run_fixture(root, inputs, 'bad_weight', bad_weights=True)
            self.assertEqual(self.last_model_load_count, 0)
            self.assertEqual(self.last_actions, 0)

    def make_partial(self, root, inputs):
        with self.assertRaises(InterruptedError):
            self.run_fixture(root, inputs, 'partial', interrupt_after=2)
        return root / 'partial'

    def test_resume_rejects_changed_identity_code_protocol_plan_and_weights(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs = self.fixture(root)
            source = self.make_partial(root, inputs)
            config = exp.read(source / 'run_config.json')
            for field in ('code_sha256', 'protocol', 'plan_sha256', 'weights', 'phase2_manifest_sha256'):
                bad = copy.deepcopy(config)
                identity = bad['experiment_identity']
                if field == 'code_sha256': identity[field]['actions.py'] = 'modified'
                elif field == 'protocol':
                    identity[field]['temperature'] = .8
                    bad['protocol'] = copy.deepcopy(identity[field])
                elif field == 'weights': identity[field][0]['sha256'] = 'modified'
                else: identity[field] = 'modified'
                bad['experiment_fingerprint'] = exp.object_sha(identity)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    exp.validate_prior(source, bad, inputs[3])
            broken = copy.deepcopy(config)
            broken['experiment_fingerprint'] = 'inconsistent digest'
            with self.assertRaises(ValueError): exp.validate_prior(source, broken, inputs[3])
            old = copy.deepcopy(config)
            old['mode'] = 'main'
            exp.dump(source / 'run_config.json', old)
            with self.assertRaisesRegex(ValueError, 'same-v4'):
                exp.validate_prior(source, config, inputs[3])

    def test_resume_rejects_tampered_or_missing_committed_record_and_failed_audit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs = self.fixture(root)
            source = self.make_partial(root, inputs)
            config = exp.read(source / 'run_config.json')
            path = next((source / 'branches').glob('*.json'))
            original = path.read_bytes()
            record = exp.read(path)
            record['pred_boxed'] = 'tampered answer'
            exp.dump(path, record)
            with self.assertRaisesRegex(ValueError, 'checksum'): exp.validate_prior(source, config, inputs[3])
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'missing'): exp.validate_prior(source, config, inputs[3])
            path.write_bytes(original)
            summary = exp.read(source / 'summary.json')
            summary['numerical_audit_passed'] = False
            exp.dump(source / 'summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'numerical audit'): exp.validate_prior(source, config, inputs[3])

    def test_actual_numerical_gate_records_signal_mismatch(self):
        signals = {name: {exp.H_KEY: 100.} for name in ('raw_tail32', 'asag_style_probe_window')}
        logits = {name: torch.zeros(1, 5) for name in signals}
        probe = {'selected_token_probabilities': [.9], 'generated_token_ids': [1],
                 'C_mean_selected_token_probability': .9, 'status': 'closed_box_review_required'}
        saved = {'signals': copy.deepcopy(signals), 'probe': probe}
        saved['signals']['raw_tail32'][exp.H_KEY] = 120.
        with tempfile.TemporaryDirectory() as d, \
             patch.object(exp, 'measure_modes', return_value=(signals, logits)), \
             patch.object(exp, 'probe_asag_style', return_value=probe):
            path = Path(d) / 'audit.json'
            with self.assertRaisesRegex(RuntimeError, 'cross-run'):
                exp.audit_point(None, None, fixture_candidates()[0], saved, 'cpu', path)
            self.assertFalse(exp.read(path)['passed'])

    def test_evidence_loader_rejects_unsealed_or_unvalidated_weight_and_signal_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            phase = Path(d) / 'evidence'
            (phase / 'signals').mkdir(parents=True)
            plan = frozen_plan()
            (phase / 'plan.json').write_bytes(Path(__file__).with_name('expansion_plan.json').read_bytes())
            exp.dump(phase / 'weight_preflight.json', {
                'weights': exp.EXPECTED_WEIGHTS, 'plan_sha256': exp.file_sha(phase / 'plan.json'),
                'tokenizer_and_fifteen_original_prefixes_verified': True,
            })
            exp.dump(phase / 'signals/run_config.json', {
                'script_sha256': exp.file_sha(Path(__file__).with_name('collect_phase2.py')),
                'v1_core_sha256': exp.file_sha(Path(__file__).with_name('v1_core.py')),
                'plan': plan, 'plan_sha256': exp.file_sha(phase / 'plan.json'),
                'dtype': 'float32', 'actual_model_dtype': 'torch.float32',
            })
            exp.dump(phase / 'signals/summary.json', {
                'completed': True, 'planned_positions': 15, 'executed_positions': 15,
                'canary_passed': True, 'actual_gpu_validation_done_by_this_run': True,
            })
            for entry in plan['selected']:
                exp.dump(phase / 'signals' / f"case_{entry['case']:03d}_cut_{entry['cut']:05d}.json", entry)

            def seal():
                return {str(p.relative_to(phase)): exp.file_sha(p)
                        for p in phase.rglob('*.json') if p.name != 'evidence_manifest.json'}

            missing = seal()
            del missing['weight_preflight.json']
            exp.dump(phase / 'evidence_manifest.json', {'files_sha256': missing})
            with self.assertRaisesRegex(ValueError, 'cover every required'):
                exp.load_evidence(phase, Path(d) / 'baseline', Path(d) / 'model', None)

            preflight = exp.read(phase / 'weight_preflight.json')
            preflight['weights'][0]['sha256'] = 'different weight content'
            exp.dump(phase / 'weight_preflight.json', preflight)
            exp.dump(phase / 'evidence_manifest.json', {'files_sha256': seal()})
            with self.assertRaisesRegex(ValueError, 'pre-signal weight/prefix'):
                exp.load_evidence(phase, Path(d) / 'baseline', Path(d) / 'model', None)

            preflight['weights'] = exp.EXPECTED_WEIGHTS
            exp.dump(phase / 'weight_preflight.json', preflight)
            summary = exp.read(phase / 'signals/summary.json')
            summary['actual_gpu_validation_done_by_this_run'] = False
            exp.dump(phase / 'signals/summary.json', summary)
            exp.dump(phase / 'evidence_manifest.json', {'files_sha256': seal()})
            with self.assertRaisesRegex(ValueError, 'completed fifteen-point'):
                exp.load_evidence(phase, Path(d) / 'baseline', Path(d) / 'model', None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
