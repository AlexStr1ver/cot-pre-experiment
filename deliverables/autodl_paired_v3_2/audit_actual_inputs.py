"""Offline read-only audit of actual returned artifacts; no model inference."""
import argparse
import copy
import difflib
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshots(folder):
    return {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser()
    for name in ('package', 'failed-run', 'failed-archive', 'prior-run', 'previous-package', 'baseline', 'model'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package))
    import paired_experiment as exp
    import resume_compatibility as lineage
    from environment_probe import compare_compatibility
    from hardware_compatibility import compare_hardware
    from transformers import AutoTokenizer

    source = args.prior_run / 'results'
    prior_snapshot = snapshots(args.prior_run)
    failed_snapshot = snapshots(args.failed_run)
    previous_snapshot = snapshots(args.previous_package)
    old = exp.read(source / 'run_config.json')
    failed = exp.read(args.failed_run / 'environment/existing_0_isolated.json')
    assert set(failed['mismatches'][0]) == {'kind', 'name', 'expected', 'actual'}
    assert len(failed['mismatches']) == 1 and failed['mismatches'][0]['name'] == 'total_bytes'
    assert not (args.failed_run / 'results').exists()
    assert not (args.failed_run / 'tests.log').exists()
    assert not (args.failed_run / 'experiment.log').exists()
    assert (args.failed_run / 'exit_status.txt').read_text().strip() == '1'
    returned_files = snapshots(args.failed_run / 'code')
    assert all(previous_snapshot[name] == sha for name, sha in returned_files.items())
    with zipfile.ZipFile(args.failed_archive) as z:
        assert z.testzip() is None

    rechecked = compare_compatibility(failed, old, True)
    assert rechecked['hardware_compatible'] is True and not rechecked['mismatches']
    assert all(rechecked[k] is True for k in ('package_versions_match', 'python_version_compatible', 'cuda_ok'))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    plan, candidates, _, _ = exp.load_evidence(args.package / 'evidence_phase2fix', args.baseline, args.model, tokenizer)
    current = copy.deepcopy(old)
    current['experiment_identity']['code_sha256'] = lineage.local_code_hashes()
    current['experiment_fingerprint'] = exp.object_sha(current['experiment_identity'])
    current['hardware'] = copy.deepcopy(failed['hardware'])
    current['mode'] = 'replicate'
    bridge = lineage.fingerprint_bridge(old, current)
    _, summary, complete = exp.validate_prior(source, current, candidates, 'replicate')
    assert summary['completed'] and len(complete) == 18
    jobs = exp.schedule(candidates, 'replicate')
    assert len(jobs) == 36 and {seed for _, _, seed in jobs} == {2027, 2028}
    assert not set(complete) & {exp.job_key(*job) for job in jobs}
    for name in lineage.UNCHANGED_CODE_FILES:
        assert (args.package / name).read_bytes() == (args.previous_package / name).read_bytes()
    assert (args.package / 'PROTOCOL.json').read_bytes() == (args.previous_package / 'PROTOCOL.json').read_bytes()
    assert old['protocol'] == exp.PROTOCOL
    assert prior_snapshot == snapshots(args.prior_run)
    assert failed_snapshot == snapshots(args.failed_run)
    assert previous_snapshot == snapshots(args.previous_package)

    diagnosis = {
        'date_beijing': '2026-10-28', 'source_archive': args.failed_archive.name,
        'source_archive_bytes': args.failed_archive.stat().st_size,
        'source_archive_sha256': digest(args.failed_archive), 'zip_crc_valid': True,
        'returned_code_files_identical_to_v3_1': len(returned_files),
        'exit_status': 1, 'failed_stage': 'environment_preparation',
        'dependency_and_cuda_checks_passed': True, 'sole_observed_mismatch': failed['mismatches'][0],
        'capacity_difference_bytes': failed['hardware']['total_bytes'] - old['hardware']['total_bytes'],
        'capacity_difference_percent': 100 * (failed['hardware']['total_bytes'] - old['hardware']['total_bytes']) / old['hardware']['total_bytes'],
        'root_cause': 'Exact total memory byte equality incorrectly blocked otherwise compatible runtime/GPU model metadata',
        'same_physical_gpu_proven': False, 'cause_of_capacity_report_change_known': False,
        'cpu_tests_started': False, 'new_environment_install_attempted': False, 'new_model_branches': 0,
        'old_results_and_failed_evidence_unchanged': True,
    }
    preflight = {
        'date_beijing': '2026-10-28', 'actual_local_tokenizer_and_original_prefixes_verified': True,
        'main_records_validated': len(complete), 'new_schedule_count': len(jobs),
        'seeds': sorted({seed for _, _, seed in jobs}),
        'new_schedule_overlaps_main': False, 'max_additional_generated_tokens': sum(16384 - c['cut'] for c, _, _ in jobs),
        'experiment_fingerprint_changed_honestly': old['experiment_fingerprint'] != current['experiment_fingerprint'],
        'fingerprint_compatibility': bridge,
        'hardware_compatibility': compare_hardware(current['hardware'], old['hardware']),
        'failed_probe_replay': rechecked,
        'real_model_inference': False, 'actual_GPU_execution': False,
        'note': 'Read-only replay uses uploaded GPU metadata; actual AutoDL runtime and numerical audits still required.',
    }
    core = {
        'date_beijing': '2026-10-28', 'scientific_protocol_unchanged': True,
        'unchanged_scientific_files': list(lineage.UNCHANGED_CODE_FILES),
        'original_six_files_all_byte_identical': False,
        'coordinator_change_scope': 'hardware compatibility, exact approved fingerprint migration, provenance logging only',
        'code_sha256': lineage.local_code_hashes(),
        'original_main_experiment_fingerprint': old['experiment_fingerprint'],
        'new_experiment_fingerprint_for_frozen_main_identity': current['experiment_fingerprint'],
        'successful_main_source': '/root/autodl-tmp/infty_paired_v3_kt2eZTc2',
        'failed_runs_not_resume_sources': ['infty_paired_v3_gddcYOhk', 'infty_paired_v3_FCuKe8gI'],
    }
    for name, data in [('FAILURE_DIAGNOSIS.json', diagnosis), ('RESUME_PREFLIGHT.json', preflight), ('CORE_IDENTITY.json', core)]:
        exp.dump(args.package / name, data)
    diff = ''.join(difflib.unified_diff((args.previous_package / 'paired_experiment.py').read_text().splitlines(True),
                                      (args.package / 'paired_experiment.py').read_text().splitlines(True),
                                      fromfile='v3.1/paired_experiment.py', tofile='v3.2/paired_experiment.py', n=0))
    (args.package / 'COORDINATOR_DIFF.txt').write_text(diff)
    print(json.dumps({'diagnosis': diagnosis, 'preflight': preflight, 'core': core}, indent=2))


if __name__ == '__main__':
    main()
