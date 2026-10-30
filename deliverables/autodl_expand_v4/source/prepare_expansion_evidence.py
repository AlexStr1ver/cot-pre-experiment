#!/usr/bin/env python3
"""Validate fixed expansion inputs and seal/reuse H/C evidence without inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

PACKAGE = Path(__file__).resolve().parent
EXPECTED_WEIGHTS = [{'name': 'model.safetensors', 'bytes': 3554214720,
                     'sha256': '23ceffc06528eb4d552ea9df4b2ebb98616d6dc8bd1b9d79966e240ca98a42ce'}]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def dump_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def frozen_plan():
    plan = read(PACKAGE / 'expansion_plan.json')
    if (plan['cohort_cases'] != [10, 11, 12, 15, 19] or plan['targets'] != [1024, 2048, 4096]
            or plan['replicate_seeds'] != [2026, 2027] or len(plan['selected']) != 15):
        raise ValueError('Unexpected frozen expansion cohort')
    return plan


def verify_inputs(model, baseline):
    """All disk checks finish before collector is allowed to load model weights."""
    from transformers import AutoTokenizer
    from collect_phase2 import verify_plan
    plan = frozen_plan()
    for name, expected in plan['model_and_tokenizer_file_sha256'].items():
        if Path(name).name != name or sha(model / name) != expected:
            raise ValueError('Model/tokenizer metadata mismatch: ' + name)
    weights = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)}
               for p in sorted(model.glob('*.safetensors'))]
    if weights != EXPECTED_WEIGHTS:
        raise ValueError('Actual model weight bytes differ from frozen successful main')
    records = sorted((baseline / 'cases').glob('*.json'))
    if hashlib.sha256(''.join(sha(p) for p in records).encode()).hexdigest() != plan['baseline_case_collection_sha256']:
        raise ValueError('Original native20 collection differs')
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    prepared = verify_plan(plan, baseline, tokenizer)
    if len(prepared) != 15:
        raise ValueError('Expected exactly fifteen verified prefixes')
    return {'weights': weights, 'plan_sha256': sha(PACKAGE / 'expansion_plan.json'),
            'tokenizer_and_fifteen_original_prefixes_verified': True,
            'new_model_inference': False, 'checked_unix': time.time()}


def expected_evidence_names(plan):
    return {'plan.json', 'weight_preflight.json', 'signals/run_config.json', 'signals/summary.json',
            *(f"signals/case_{p['case']:03d}_cut_{p['cut']:05d}.json" for p in plan['selected'])}


def validate_complete_evidence(folder, require_manifest=True):
    """Reject partial, altered or foreign evidence; never select by answer quality."""
    plan = frozen_plan()
    if read(folder / 'plan.json') != plan or sha(folder / 'plan.json') != sha(PACKAGE / 'expansion_plan.json'):
        raise ValueError('Evidence plan differs from bundled frozen plan')
    preflight = read(folder / 'weight_preflight.json')
    if (preflight.get('weights') != EXPECTED_WEIGHTS or
            preflight.get('plan_sha256') != sha(folder / 'plan.json') or
            preflight.get('tokenizer_and_fifteen_original_prefixes_verified') is not True):
        raise ValueError('Evidence lacks frozen pre-signal weight/prefix identity')
    summary = read(folder / 'signals/summary.json')
    if (summary.get('completed') is not True or summary.get('executed_positions') != 15
            or summary.get('planned_positions') != 15 or summary.get('canary_passed') is not True
            or summary.get('actual_gpu_validation_done_by_this_run') is not True):
        raise ValueError('Source evidence is not a complete audited fifteen-point GPU measurement')
    cfg = read(folder / 'signals/run_config.json')
    if (cfg.get('plan') != plan or cfg.get('plan_sha256') != sha(folder / 'plan.json')
            or cfg.get('dtype') != 'float32' or cfg.get('actual_model_dtype') != 'torch.float32'
            or cfg.get('script_sha256') != sha(PACKAGE / 'collect_phase2.py')
            or cfg.get('v1_core_sha256') != sha(PACKAGE / 'v1_core.py')):
        raise ValueError('Evidence collector code, plan or arithmetic identity differs')
    expected_points = {f"case_{p['case']:03d}_cut_{p['cut']:05d}.json": p for p in plan['selected']}
    if {p.name for p in (folder / 'signals').glob('case_*.json')} != set(expected_points):
        raise ValueError('Evidence point set differs')
    for name, planned in expected_points.items():
        point = read(folder / 'signals' / name)
        if (point.get('status') != 'measured_review_required' or
                any(point.get(k) != planned[k] for k in ('case', 'cut', 'prefix_token_ids_sha256'))):
            raise ValueError('Evidence point identity/status mismatch: ' + name)
    names = expected_evidence_names(plan)
    if require_manifest:
        manifest = read(folder / 'evidence_manifest.json')
        hashes = manifest['files_sha256']
        if set(hashes) != names:
            raise ValueError('Evidence manifest coverage differs')
        for name, expected in hashes.items():
            path = folder / name
            if path.is_symlink() or sha(path) != expected:
                raise ValueError('Evidence checksum mismatch: ' + name)
    return names


def validate_resume_source(source_run):
    evidence = source_run / 'evidence'
    names = validate_complete_evidence(evidence)
    config = read(source_run / 'results/run_config.json')
    summary = read(source_run / 'results/summary.json')
    identity = config['experiment_identity']
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if config.get('mode') != 'expand' or summary.get('numerical_audit_passed') is not True:
        raise ValueError('Resume source must be a v4 expansion with passed numerical audit')
    if config.get('experiment_fingerprint') != identity_sha:
        raise ValueError('Resume source identity hash differs')
    if (identity.get('plan_sha256') != sha(evidence / 'plan.json') or
            identity.get('phase2_manifest_sha256') != sha(evidence / 'evidence_manifest.json') or
            identity.get('weights') != EXPECTED_WEIGHTS):
        raise ValueError('Resume source evidence/weights identity differs')
    # Coordinator subsequently validates every completed branch. This earlier
    # check prevents copying measurements from another code revision.
    from expanded_experiment import CODE_FILES
    if identity.get('code_sha256') != {name: sha(PACKAGE / name) for name in CODE_FILES}:
        raise ValueError('Resume source code differs from this expansion package')
    return names


def prepare(model, baseline, output, source_run=None):
    if output.exists():
        raise ValueError('Evidence output must be new; existing results are immutable')
    current = verify_inputs(model, baseline)
    if source_run is None:
        output.mkdir(parents=True)
        shutil.copyfile(PACKAGE / 'expansion_plan.json', output / 'plan.json')
        dump_new(output / 'weight_preflight.json', current)
        mode = 'new_signal_collection_required'
    else:
        names = validate_resume_source(source_run)
        output.mkdir(parents=True)
        for name in sorted(names | {'evidence_manifest.json'}):
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_run / 'evidence' / name, target)
        validate_complete_evidence(output)
        mode = 'exact_complete_evidence_reused'
    report = {'completed': True, 'selection': mode, 'source_run': str(source_run) if source_run else None,
              'current_input_validation': current, 'previous_files_modified': False,
              'note': 'Signal reuse preserves manifest bytes; fresh action numerical audits remain mandatory.'}
    dump_new(output.parent / 'evidence_preparation.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def finalize(evidence):
    names = validate_complete_evidence(evidence, require_manifest=False)
    manifest = {'protocol': 'expanded_frozen_prefix_signal_evidence_v4',
                'files_sha256': {name: sha(evidence / name) for name in sorted(names)},
                'note': 'Fresh fixed-cohort H/C measurements; immutable and copied byte-for-byte on same-v4 resume.'}
    dump_new(evidence / 'evidence_manifest.json', manifest)
    validate_complete_evidence(evidence)
    print('EVIDENCE_SEALED=' + str(evidence), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    first = commands.add_parser('prepare')
    for name in ('model', 'baseline', 'output'):
        first.add_argument('--' + name, type=Path, required=True)
    first.add_argument('--source-run', type=Path)
    last = commands.add_parser('finalize')
    last.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.model, args.baseline, args.output, args.source_run)
    else:
        finalize(args.evidence)


if __name__ == '__main__':
    main()
