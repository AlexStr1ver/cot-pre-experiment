"""CPU-only evidence sealing/reuse regressions with explicitly synthetic files.

No checkpoint is downloaded or loaded. Synthetic GPU-success flags below are
input fixtures for validation, never claims about execution by these tests.
"""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import expanded_experiment
import prepare_expansion_evidence as evidence


REAL_PACKAGE = Path(__file__).resolve().parent


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def snapshot(folder):
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob('*') if path.is_file()}


class ExpansionEvidenceTests(unittest.TestCase):
    def fixture(self, root):
        package, model, baseline = (root / name for name in ('package', 'model', 'baseline'))
        for path in (package, model, baseline / 'cases'):
            path.mkdir(parents=True)
        plan = copy.deepcopy(evidence.read(REAL_PACKAGE / 'expansion_plan.json'))
        (model / 'config.json').write_text('{"fixture":true}\n')
        (model / 'model.safetensors').write_bytes(b'fixture weights A')
        weights = [{'name':'model.safetensors', 'bytes':17,
                    'sha256':evidence.sha(model / 'model.safetensors')}]
        self.assertEqual((model / 'model.safetensors').stat().st_size, weights[0]['bytes'])
        plan['model_and_tokenizer_file_sha256'] = {'config.json':evidence.sha(model / 'config.json')}
        for case in range(1, 21):
            dump(baseline / 'cases' / f'{case:03d}.json', {'case':case, 'fixture_only':True})
        plan['baseline_case_collection_sha256'] = hashlib.sha256(''.join(
            evidence.sha(p) for p in sorted((baseline / 'cases').glob('*.json'))).encode()).hexdigest()
        dump(package / 'expansion_plan.json', plan)
        for name in expanded_experiment.CODE_FILES:
            if name != 'expansion_plan.json':
                (package / name).write_text('# synthetic code identity fixture: ' + name + '\n')
        return package, model, baseline, plan, weights

    @contextlib.contextmanager
    def configured(self, directory):
        fixture = self.fixture(Path(directory))
        package, _, _, _, weights = fixture
        with patch.object(evidence, 'PACKAGE', package), patch.object(evidence, 'EXPECTED_WEIGHTS', weights), \
             contextlib.redirect_stdout(io.StringIO()):
            yield fixture

    def complete_evidence(self, root, package, plan, weights, seal=True):
        folder = root / 'evidence'
        (folder / 'signals').mkdir(parents=True)
        (folder / 'plan.json').write_bytes((package / 'expansion_plan.json').read_bytes())
        dump(folder / 'weight_preflight.json', {'weights':weights, 'fixture_only':True, 'checked_unix':123.5,
             'plan_sha256':evidence.sha(folder / 'plan.json'),
             'tokenizer_and_fifteen_original_prefixes_verified':True})
        dump(folder / 'signals/summary.json', {'completed':True, 'planned_positions':15,
             'executed_positions':15, 'canary_passed':True,
             'actual_gpu_validation_done_by_this_run':True, 'fixture_only':True})
        dump(folder / 'signals/run_config.json', {'plan':plan, 'plan_sha256':evidence.sha(folder / 'plan.json'),
             'dtype':'float32', 'actual_model_dtype':'torch.float32',
             'script_sha256':evidence.sha(package / 'collect_phase2.py'),
             'v1_core_sha256':evidence.sha(package / 'v1_core.py'), 'fixture_only':True})
        for entry in plan['selected']:
            dump(folder / 'signals' / f"case_{entry['case']:03d}_cut_{entry['cut']:05d}.json",
                 {**entry, 'status':'measured_review_required', 'fixture_only':True})
        if seal:
            evidence.finalize(folder)
        return folder

    def resumable(self, root, package, plan, weights):
        folder = self.complete_evidence(root, package, plan, weights)
        identity = {'plan_sha256':evidence.sha(folder / 'plan.json'),
                    'phase2_manifest_sha256':evidence.sha(folder / 'evidence_manifest.json'),
                    'weights':copy.deepcopy(weights),
                    'code_sha256':{name:evidence.sha(package / name) for name in expanded_experiment.CODE_FILES}}
        config = {'mode':'expand', 'experiment_identity':identity,
                  'experiment_fingerprint':expanded_experiment.object_sha(identity)}
        dump(root / 'results/run_config.json', config)
        dump(root / 'results/summary.json', {'completed':False, 'numerical_audit_passed':True,
                                           'fixture_only':True})
        return folder

    def test_weight_content_mismatch_stops_before_tokenizer_or_prefix_work(self):
        with tempfile.TemporaryDirectory() as directory, self.configured(directory) as fixture:
            package, model, baseline, _, _ = fixture
            # Same filename and byte count, different contents must not pass.
            (model / 'model.safetensors').write_bytes(b'fixture weights B')
            with patch('transformers.AutoTokenizer.from_pretrained') as tokenizer, \
                 patch('collect_phase2.verify_plan') as verify:
                with self.assertRaisesRegex(ValueError, 'weight bytes differ'):
                    evidence.verify_inputs(model, baseline)
                tokenizer.assert_not_called()
                verify.assert_not_called()

    def test_valid_fresh_prepare_checks_inputs_and_creates_no_signal_directory(self):
        with tempfile.TemporaryDirectory() as directory, self.configured(directory) as fixture:
            package, model, baseline, _, weights = fixture
            output = Path(directory) / 'new run' / 'evidence'
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=object()) as tokenizer, \
                 patch('collect_phase2.verify_plan', return_value=[{}] * 15) as verify:
                evidence.prepare(model, baseline, output)
                tokenizer.assert_called_once_with(model, local_files_only=True)
                verify.assert_called_once()
            self.assertEqual((output / 'plan.json').read_bytes(), (package / 'expansion_plan.json').read_bytes())
            self.assertEqual(evidence.read(output / 'weight_preflight.json')['weights'], weights)
            self.assertFalse((output / 'signals').exists())
            self.assertFalse((output / 'evidence_manifest.json').exists())
            report = evidence.read(output.parent / 'evidence_preparation.json')
            self.assertEqual(report['selection'], 'new_signal_collection_required')
            self.assertFalse(report['current_input_validation']['new_model_inference'])

    def test_complete_fifteen_point_evidence_is_sealed_once_with_exact_hash_coverage(self):
        with tempfile.TemporaryDirectory() as directory, self.configured(directory) as fixture:
            package, _, _, plan, weights = fixture
            folder = self.complete_evidence(Path(directory) / 'source', package, plan, weights, seal=False)
            prior = snapshot(folder)
            evidence.finalize(folder)
            manifest = evidence.read(folder / 'evidence_manifest.json')
            self.assertEqual(set(manifest['files_sha256']), evidence.expected_evidence_names(plan))
            self.assertEqual(len(manifest['files_sha256']), 19)
            self.assertEqual(prior, {key:value for key,value in snapshot(folder).items() if key != 'evidence_manifest.json'})
            for name, digest in manifest['files_sha256'].items():
                self.assertEqual(digest, evidence.sha(folder / name))
            with self.assertRaises(FileExistsError):
                evidence.finalize(folder)

    def test_same_v4_resume_copies_all_evidence_bytes_and_keeps_source_immutable(self):
        with tempfile.TemporaryDirectory() as directory, self.configured(directory) as fixture:
            package, model, baseline, plan, weights = fixture
            source = Path(directory) / 'old run'
            old_evidence = self.resumable(source, package, plan, weights)
            before = snapshot(source)
            output = Path(directory) / 'new run' / 'evidence'
            current = {'weights':weights, 'checked_unix':999.5, 'new_model_inference':False}
            with patch.object(evidence, 'verify_inputs', return_value=current):
                evidence.prepare(model, baseline, output, source)
            self.assertEqual(snapshot(old_evidence), snapshot(output))
            self.assertEqual(snapshot(source), before)
            self.assertEqual(evidence.read(output / 'weight_preflight.json')['checked_unix'], 123.5)
            report = evidence.read(output.parent / 'evidence_preparation.json')
            self.assertEqual(report['selection'], 'exact_complete_evidence_reused')
            self.assertEqual(report['current_input_validation']['checked_unix'], 999.5)

    def test_incomplete_changed_or_missing_evidence_is_rejected(self):
        for issue in ('incomplete', 'canary', 'cpu_claim', 'plan', 'point_hash', 'missing_point', 'manifest_coverage',
                      'preflight_plan_hash', 'preflight_prefix_check'):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as directory, self.configured(directory) as fixture:
                package, _, _, plan, weights = fixture
                folder = self.complete_evidence(Path(directory) / 'old', package, plan, weights)
                if issue in ('incomplete', 'canary', 'cpu_claim'):
                    path = folder / 'signals/summary.json'; data = evidence.read(path)
                    data[{'incomplete':'completed', 'canary':'canary_passed', 'cpu_claim':'actual_gpu_validation_done_by_this_run'}[issue]] = False
                    dump(path, data)
                elif issue == 'plan':
                    path = folder / 'plan.json'; data = evidence.read(path); data['replicate_seeds'] = [9999]; dump(path, data)
                elif issue in ('point_hash', 'missing_point'):
                    path = next((folder / 'signals').glob('case_*.json'))
                    if issue == 'missing_point':path.unlink()
                    else:
                        data=evidence.read(path);data['fixture_tampering']='changed';dump(path,data)
                elif issue.startswith('preflight_'):
                    path = folder / 'weight_preflight.json'; data = evidence.read(path)
                    data['plan_sha256' if issue == 'preflight_plan_hash' else 'tokenizer_and_fifteen_original_prefixes_verified'] = None
                    dump(path, data)
                else:
                    path=folder/'evidence_manifest.json';data=evidence.read(path)
                    del data['files_sha256']['weight_preflight.json'];dump(path,data)
                with self.assertRaises(ValueError):
                    evidence.validate_complete_evidence(folder)

    def test_foreign_resume_identity_or_failed_numerical_audit_is_rejected_before_copy(self):
        for issue in ('mode', 'audit', 'fingerprint', 'weights', 'code', 'plan_hash', 'manifest_hash'):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as directory, self.configured(directory) as fixture:
                package, model, baseline, plan, weights = fixture
                source=Path(directory)/'old';self.resumable(source,package,plan,weights)
                cfg_path=source/'results/run_config.json';config=evidence.read(cfg_path)
                if issue == 'audit':
                    dump(source/'results/summary.json',{'numerical_audit_passed':False})
                elif issue == 'mode':config['mode']='replicate'
                elif issue == 'fingerprint':config['experiment_fingerprint']='forged'
                else:
                    field={'weights':'weights','code':'code_sha256','plan_hash':'plan_sha256','manifest_hash':'phase2_manifest_sha256'}[issue]
                    config['experiment_identity'][field]='changed'
                    config['experiment_fingerprint']=expanded_experiment.object_sha(config['experiment_identity'])
                dump(cfg_path,config)
                output=Path(directory)/'new'/'evidence';before=snapshot(source)
                with patch.object(evidence,'verify_inputs',return_value={'fixture_only':True}):
                    with self.assertRaises(ValueError):evidence.prepare(model,baseline,output,source)
                self.assertFalse(output.exists())
                self.assertEqual(snapshot(source),before)

    def test_existing_output_is_rejected_before_any_inputs_or_files_are_touched(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'evidence';output.mkdir();(output/'sentinel').write_bytes(b'unchanged')
            before=snapshot(output)
            with patch.object(evidence,'verify_inputs') as verify:
                with self.assertRaisesRegex(ValueError,'must be new'):
                    evidence.prepare(Path('model'),Path('baseline'),output)
                verify.assert_not_called()
            self.assertEqual(snapshot(output),before)

    def test_bad_metadata_and_native_collection_fail_before_tokenizer(self):
        for issue in ('metadata','collection'):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as directory, self.configured(directory) as fixture:
                _,model,baseline,_,_=fixture
                path=model/'config.json' if issue=='metadata' else baseline/'cases/001.json'
                path.write_text('{"tampered":true}\n')
                with patch('transformers.AutoTokenizer.from_pretrained') as tokenizer:
                    with self.assertRaises(ValueError):evidence.verify_inputs(model,baseline)
                    tokenizer.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
