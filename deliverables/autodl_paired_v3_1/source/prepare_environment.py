#!/usr/bin/env python3
"""Diagnose imports, select a healthy interpreter or build a NEW pinned venv.

All commands inherit the runner worker process group, so TERM also stops pip.
Never installs into the original environment or changes model/results files.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def dump(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run_probe(python, package, folder, previous, name, isolated):
    report = folder / (name + '.json')
    argv = [str(python)]
    if isolated:
        argv += ['-I', '-u', str(package / 'environment_probe.py')]
    else:
        # Preserve the previous code cwd as a possible shadowing source. -c
        # leaves that cwd on sys.path, unlike executing a probe from the new ZIP.
        argv += ['-u', '-c', 'import runpy,sys;sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")',
                 str(package / 'environment_probe.py')]
    argv += ['--output', str(report), '--require-cuda', '--expected-main', str(previous)]
    cwd = previous / 'code'
    if not cwd.is_dir():
        cwd = package
    try:
        with (folder / (name + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(argv, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, timeout=180)
        if report.exists():
            data = read(report)
        else:
            data = {'dependencies_ok': False, 'probe_process_error': 'No JSON report; inspect log'}
        data['probe_exit_status'] = result.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        data = {'dependencies_ok': False, 'probe_process_error': str(exc), 'probe_exit_status': 1}
    dump(report, data)
    return data


def usable(data):
    return data.get('probe_exit_status') == 0 and data.get('passed') is True


def cannot_fix_with_dependencies(data):
    # Healthy libraries with wrong device/driver/physical GPU should stop, not
    # spend gigabytes reinstalling identical wheels. Python mismatch also stops.
    if data.get('python_version_compatible') is False:
        return True
    if any(item.get('kind') == 'hardware' and item.get('name') in
           ('device', 'name', 'capability', 'total_bytes') for item in data.get('mismatches', [])):
        return True
    if data.get('classification') == 'probe_or_expected_input_error':
        return True
    return bool(data.get('dependencies_ok') and data.get('package_versions_match') and
                (data.get('cuda_ok') is not True or data.get('hardware_compatible') is not True))


def install_new_environment(root, package, folder, previous, state):
    cfg = read(previous / 'results/run_config.json')
    expected_python = tuple(int(v) for v in cfg['python'].split()[0].split('.')[:2])
    if sys.version_info[:2] != expected_python:
        raise RuntimeError('Need the original Python major.minor to create a compatible venv: ' + str(expected_python))
    # Everything large lives outside the result directory (and its return ZIP).
    free = shutil.disk_usage(root).free
    if free < 12 * 2**30:
        raise RuntimeError('Fresh CUDA environment needs at least 12 GiB free on ROOT; no old data was deleted')
    venvs = root / 'venvs'
    venvs.mkdir(parents=True, exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix='infty-paired-v3_1_', dir=venvs))
    state.update(stage='creating_new_venv', new_venv=str(target))
    dump(folder / 'preparation.json', state)
    subprocess.run([sys.executable, '-I', '-m', 'venv', str(target)], check=True)
    python = target / 'bin/python'
    cache = root / '.cache/infty-paired-v3_1'
    (cache / 'tmp').mkdir(parents=True, exist_ok=True)
    (cache / 'pip').mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env['TMPDIR'] = str(cache / 'tmp')
    base = [str(python), '-I', '-m', 'pip', '--isolated', '--disable-pip-version-check',
            '--cache-dir', str(cache / 'pip'), 'install', '--retries', '3', '--timeout', '60',
            '--constraint', str(package / 'constraints-recovery.txt')]
    commands = [base + ['--index-url', 'https://download.pytorch.org/whl/cu124', 'torch==2.6.0+cu124'],
                base + ['--index-url', 'https://pypi.org/simple', '--requirement', str(package / 'requirements-recovery.txt')],
                [str(python), '-I', '-m', 'pip', '--isolated', 'check']]
    dump(folder / 'install_plan.json', {'new_environment': str(target), 'commands': commands,
                                       'original_environment_modified': False})
    for index, command in enumerate(commands):
        state['stage'] = 'installing_torch' if index == 0 else 'installing_transformers' if index == 1 else 'pip_check'
        dump(folder / 'preparation.json', state)
        print('ENVIRONMENT_STAGE=' + state['stage'], flush=True)
        subprocess.run(command, env=env, check=True)
    return python


def prepare(root, package, run_dir, previous, candidate_python=None):
    folder = run_dir / 'environment'
    folder.mkdir()
    state = {'completed': False, 'stage': 'starting', 'started_unix': time.time(),
             'original_environment_modified': False, 'bootstrap_python': sys.executable,
             'previous_run': str(previous), 'actual_new_model_inference': False}
    dump(folder / 'preparation.json', state)
    try:
        config_path = previous / 'results/run_config.json'
        summary_path = previous / 'results/summary.json'
        if not config_path.is_file() or not summary_path.is_file():
            raise RuntimeError('Source must be completed main kt2eZTc2 or resumable replicate; failed preflight gddcYOhk has no model results')
        cfg, old_summary = read(config_path), read(summary_path)
        if cfg.get('mode') not in ('smoke','main','replicate') or not old_summary.get('numerical_audit_passed'):
            raise RuntimeError('Source lacks a passed numerical audit')
        if cfg['mode'] == 'main' and not old_summary.get('completed'):
            raise RuntimeError('Seed replication requires a completed main')
        # Do not adopt arbitrary expected versions from a changed prior file.
        if cfg['hardware']['torch'] != '2.6.0+cu124' or cfg['transformers'] != '4.51.3':
            raise RuntimeError('Recovery is pinned to main torch2.6.0+cu124 / Transformers4.51.3')
        default = root / 'venvs/infty-official/bin/python'
        candidates = [Path(candidate_python)] if candidate_python else [default]
        marker = root / 'venvs/infty-paired-v3_1-ready.json'
        if not candidate_python and marker.is_file():
            try:
                cached = Path(read(marker)['python'])
                if cached.is_absolute() and cached.parent.parent.resolve().is_relative_to((root / 'venvs').resolve()):
                    if cached not in candidates:
                        candidates.append(cached)
            except (KeyError, ValueError, OSError):
                pass
        selected = None
        for index, candidate in enumerate(candidates):
            state.update(stage='probing_existing', candidate=str(candidate))
            dump(folder / 'preparation.json', state)
            normal = run_probe(candidate, package, folder, previous, f'existing_{index}_normal', False)
            isolated = run_probe(candidate, package, folder, previous, f'existing_{index}_isolated', True)
            if usable(isolated):
                selected = candidate
                state['selection'] = 'existing_healthy' if usable(normal) else 'isolated_import_recovery'
                break
            if cannot_fix_with_dependencies(isolated):
                raise RuntimeError('Python or CUDA/GPU mismatch is not repaired by reinstalling dependencies; inspect environment probe JSON')
        if selected is None:
            selected = install_new_environment(root, package, folder, previous, state)
            fixed = run_probe(selected, package, folder, previous, 'new_environment_isolated', True)
            if not usable(fixed):
                raise RuntimeError('New environment still fails API/version/GPU check; no model run started')
            state['selection'] = 'new_isolated_venv'
            dump(marker, {'python': str(selected), 'created_by_run': str(run_dir),
                          'note': 'Re-probed each use; no original environment replaced'})
        (folder / 'selected_python.txt').write_text(str(selected) + '\n', encoding='utf-8')
        # Inventory is diagnostic only. The real import/API/version/GPU checks
        # above are mandatory; missing pip in an otherwise healthy env is not.
        try:
            inventory = subprocess.run([str(selected), '-I', '-m', 'pip', '--isolated', 'list', '--format=json'],
                                       capture_output=True, text=True, timeout=60)
            if inventory.returncode != 0:
                raise RuntimeError('pip inventory failed: ' + str(inventory.stderr))
            dump(folder / 'selected_packages.json', json.loads(inventory.stdout))
            dump(folder / 'inventory_status.json', {'completed': True})
        except Exception as exc:
            dump(folder / 'inventory_status.json', {'completed': False, 'error': repr(exc),
                                                   'note': 'Import/API/version/GPU checks passed separately'})
        state.update(completed=True, stage='ready_for_original_CPU_tests', selected_python=str(selected))
        print('SELECTED_PYTHON=' + str(selected), flush=True)
        return selected
    except BaseException as exc:
        state.update(error=repr(exc), failed_stage=state.get('stage'), stage='failed')
        raise
    finally:
        state['finished_unix'] = time.time()
        dump(folder / 'preparation.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['root','package','run-dir','previous']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--candidate-python')
    args = parser.parse_args()
    prepare(args.root.resolve(), args.package.resolve(), args.run_dir.resolve(), args.previous.resolve(), args.candidate_python)
