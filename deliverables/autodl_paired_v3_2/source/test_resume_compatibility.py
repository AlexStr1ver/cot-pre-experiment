"""CPU regressions for honest identity migration and branch reuse; no inference.

legacy_main_identity.json contains the authentic successful main identity.
Branch records constructed below are explicitly synthetic accounting fixtures;
the separate delivery audit validates the real eighteen saved model branches.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import paired_experiment as exp
from actions import run_action
from resume_compatibility import (
    ADDED_CODE_FILES, LEGACY_FINGERPRINT, fingerprint_bridge, identity_sha, local_code_hashes,
)
from test_actions import ScriptedDecoder, Tokenizer
import test_experiment as fixture_experiment
from test_experiment import fixture_candidates


def legacy_config():
    return exp.read(Path(__file__).with_name('legacy_main_identity.json'))


def repaired_config(old):
    current = copy.deepcopy(old)
    current['experiment_identity']['code_sha256'] = local_code_hashes()
    current['experiment_fingerprint'] = identity_sha(current['experiment_identity'])
    current['mode'] = 'replicate'
    current['hardware']['total_bytes'] = 25250627584
    return current


def fixture_prior(root, config, limit=None):
    """Write synthetic completed records with authentic code/science identity metadata."""
    source = root / 'prior'
    (source / 'branches').mkdir(parents=True)
    exp.dump(source / 'run_config.json', config)
    candidates = fixture_candidates()
    jobs = exp.schedule(candidates, config['mode'])
    selected = jobs if limit is None else jobs[:limit]
    hashes = {}
    tok = Tokenizer()
    for candidate, branch, seed in selected:
        outputs = (['</think>\\boxed{999}<eos>'] if branch == 'A_continue' else
                   ['fixture fact</summary>', '</think>\\boxed{999}<eos>'])
        record = run_action(candidate, branch, seed, ScriptedDecoder(tok, outputs), tok)
        key = exp.job_key(candidate, branch, seed)
        path = source / 'branches' / (key + '.json')
        exp.dump(path, record)
        hashes[key] = exp.file_sha(path)
    exp.dump(source / 'completed_artifacts.json', {
        'experiment_fingerprint': config['experiment_fingerprint'], 'branches_sha256': hashes,
    })
    exp.dump(source / 'summary.json', {
        'completed': len(selected) == len(jobs), 'numerical_audit_passed': True,
        'completed_branches': len(selected),
    })
    return source, candidates


class ResumeCompatibilityTests(unittest.TestCase):
    def test_authentic_legacy_identity_migrates_without_mutating_either_config(self):
        old = legacy_config()
        current = repaired_config(old)
        before = copy.deepcopy((old, current))
        self.assertEqual(identity_sha(old['experiment_identity']), LEGACY_FINGERPRINT)
        audit = fingerprint_bridge(old, current)
        self.assertEqual(audit['mode'], 'known_legacy_v3_to_v3_2')
        self.assertNotEqual(audit['prior_experiment_fingerprint'], audit['current_experiment_fingerprint'])
        self.assertEqual(set(audit['added_code_files']), set(ADDED_CODE_FILES))
        self.assertFalse(audit['prior_metadata_modified'])
        self.assertEqual((old, current), before)

    def test_same_repair_identity_uses_exact_resume_path(self):
        current = repaired_config(legacy_config())
        self.assertEqual(fingerprint_bridge(current, copy.deepcopy(current))['mode'], 'identical_v3_2_identity')

    def test_inconsistent_hashes_and_unknown_prior_fingerprints_are_rejected(self):
        old = legacy_config()
        current = repaired_config(old)
        for side in ('prior', 'current'):
            altered = copy.deepcopy(old if side == 'prior' else current)
            altered['experiment_fingerprint'] = 'forged'
            with self.subTest(side=side), self.assertRaisesRegex(ValueError, 'fingerprint'):
                fingerprint_bridge(altered, current) if side == 'prior' else fingerprint_bridge(old, altered)
        unknown = copy.deepcopy(old)
        unknown['experiment_identity']['weights'][0]['sha256'] = 'different'
        unknown['experiment_fingerprint'] = identity_sha(unknown['experiment_identity'])
        with self.assertRaisesRegex(ValueError, 'not an approved legacy'):
            fingerprint_bridge(unknown, current)

    def test_protocol_weights_prefix_evidence_and_metadata_changes_are_rejected(self):
        old = legacy_config()
        for field in ('protocol', 'weights', 'plan_sha256', 'phase2_manifest_sha256',
                      'model_tokenizer_metadata_sha256'):
            current = repaired_config(old)
            identity = current['experiment_identity']
            if field == 'protocol':
                identity[field]['temperature'] = 0.8
                current['protocol'] = copy.deepcopy(identity[field])
            elif field == 'weights':
                identity[field][0]['sha256'] = 'different weights'
            elif field == 'model_tokenizer_metadata_sha256':
                identity[field]['tokenizer.json'] = 'different tokenizer'
            else:
                identity[field] = 'different'
            current['experiment_fingerprint'] = identity_sha(identity)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'scientific identity differs'):
                fingerprint_bridge(old, current)

    def test_code_manifest_and_unfingerprinted_protocol_spoofing_are_rejected(self):
        old = legacy_config()
        for name in ('paired_experiment.py', 'actions.py', 'hardware_compatibility.py', 'resume_compatibility.py'):
            current = repaired_config(old)
            current['experiment_identity']['code_sha256'][name] = 'fake hash'
            current['experiment_fingerprint'] = identity_sha(current['experiment_identity'])
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'executing'):
                fingerprint_bridge(old, current)
        current = repaired_config(old)
        current['protocol']['temperature'] = 0.9
        with self.assertRaisesRegex(ValueError, 'protocol differs'):
            fingerprint_bridge(old, current)

    def test_unreviewed_coordinator_revision_cannot_use_the_legacy_bridge(self):
        old = legacy_config()
        current = repaired_config(old)
        changed_code = local_code_hashes()
        changed_code['paired_experiment.py'] = 'unreviewed coordinator revision'
        current['experiment_identity']['code_sha256'] = changed_code
        current['experiment_fingerprint'] = identity_sha(current['experiment_identity'])
        with patch('resume_compatibility.local_code_hashes', return_value=changed_code):
            with self.assertRaisesRegex(ValueError, 'not the audited v3.2'):
                fingerprint_bridge(old, current)

    def test_legacy_main_all_branches_validate_and_schedule_only_two_new_seeds(self):
        old = legacy_config()
        current = repaired_config(old)
        with tempfile.TemporaryDirectory() as directory:
            source, candidates = fixture_prior(Path(directory), old)
            snapshots = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
            _, _, complete = exp.validate_prior(source, current, candidates, 'replicate')
            self.assertEqual(len(complete), 18)
            jobs = exp.schedule(candidates, 'replicate')
            self.assertEqual(len(jobs), 36)
            self.assertEqual({seed for _, _, seed in jobs}, {2027, 2028})
            self.assertFalse(set(complete) & {exp.job_key(*job) for job in jobs})
            self.assertEqual(snapshots, {str(p.relative_to(source)): p.read_bytes()
                                        for p in source.rglob('*') if p.is_file()})

    def test_legacy_migration_still_rejects_large_capacity_or_model_changes(self):
        old = legacy_config()
        with tempfile.TemporaryDirectory() as directory:
            source, candidates = fixture_prior(Path(directory), old)
            for name, value in (('total_bytes', old['hardware']['total_bytes'] - 64 * 2**20),
                                ('name', 'NVIDIA different GPU'), ('capability', [8, 0]),
                                ('torch', '2.7.0+cu124')):
                current = repaired_config(old)
                current['hardware'][name] = value
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'GPU/runtime is incompatible'):
                    exp.validate_prior(source, current, candidates, 'replicate')

    def test_legacy_bridge_preserves_branch_checksum_and_numerical_audit_gates(self):
        old = legacy_config()
        current = repaired_config(old)
        with tempfile.TemporaryDirectory() as directory:
            source, candidates = fixture_prior(Path(directory), old)
            summary = exp.read(source / 'summary.json')
            summary['numerical_audit_passed'] = False
            exp.dump(source / 'summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'numerical audit'):
                exp.validate_prior(source, current, candidates, 'replicate')
            summary['numerical_audit_passed'] = True
            exp.dump(source / 'summary.json', summary)
            branch = next((source / 'branches').glob('*.json'))
            record = exp.read(branch)
            record['pred_boxed'] = 'tampered'
            exp.dump(branch, record)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                exp.validate_prior(source, current, candidates, 'replicate')

    def test_same_version_partial_replicate_reuses_only_completed_hashed_branches(self):
        current = repaired_config(legacy_config())
        with tempfile.TemporaryDirectory() as directory:
            source, candidates = fixture_prior(Path(directory), current, limit=2)
            _, summary, complete = exp.validate_prior(source, copy.deepcopy(current), candidates, 'replicate')
            self.assertFalse(summary['completed'])
            self.assertEqual(len(complete), 2)
            self.assertTrue(all(record['replicate_seed'] == 2027 for _, record in complete.values()))

    def test_real_coordinator_fixture_saves_both_hardware_and_fingerprint_audits(self):
        fixture = fixture_experiment.ExperimentTests()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            smoke = fixture.run_fixture(root, 'smoke')
            main = fixture.run_fixture(root, 'main', smoke)
            audit = exp.read(main / 'reuse_manifest.json')
            self.assertEqual(audit['source_hardware'], exp.read(smoke / 'run_config.json')['hardware'])
            self.assertEqual(audit['current_hardware'], exp.read(main / 'run_config.json')['hardware'])
            self.assertTrue(audit['hardware_compatibility']['compatible'])
            self.assertEqual(audit['fingerprint_compatibility']['mode'], 'identical_v3_2_identity')


if __name__ == '__main__':
    unittest.main(verbosity=2)
