#!/usr/bin/env python3
"""Stdlib-only lifecycle fixtures for recovery wrappers; never real GPU evidence."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile

from status_paired import process_alive

PACKAGE = Path(__file__).resolve().parent


def pid_running(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z'
    except FileNotFoundError:
        return False


class RecoveryRunnerTests(unittest.TestCase):
    def fixture(self, directory, preparation='healthy', experiment_delay=0):
        root = Path(directory) / 'autodl root with spaces'; root.mkdir()
        package = Path(directory) / 'recovery package with spaces'; package.mkdir()
        candidate = root / 'selected env with spaces/bin/python'
        candidate.parent.mkdir(parents=True)
        # Preserve the active venv under -I while testing an override path with spaces.
        candidate.write_text('#!/usr/bin/env bash\nexport INFTY_FIXTURE_PYTHON_BRIDGE=1\nexec ' +
                             shlex.quote(sys.executable) + ' "$@"\n')
        candidate.chmod(0o755)
        for name in ['run_paired.sh', 'launch_paired.sh', 'status_paired.py', 'isolated_entry.py']:
            shutil.copyfile(PACKAGE / name, package / name)
        for name in ['test_decoder.py', 'test_actions.py', 'test_experiment.py']:
            (package / name).write_text('# Deliberately empty stdlib lifecycle fixture.\n')
        (package / 'test_experiment.py').write_text(
            'import sys,unittest\n'
            'class IsolatedFixtureTests(unittest.TestCase):\n'
            ' def test_isolated_interpreter_is_used(self):\n'
            '  self.assertEqual(sys.flags.isolated,1)\n')
        evidence = package / 'evidence_phase2fix'; evidence.mkdir()
        (evidence / 'plan.json').write_text('{"fixture": true}\n')
        preparer = (
            'import argparse,json,os,pathlib,subprocess,sys,time\n'
            'p=argparse.ArgumentParser()\n'
            "for n in ['root','package','run-dir','previous']:p.add_argument('--'+n)\n"
            "p.add_argument('--candidate-python');a=p.parse_args()\n"
            "out=pathlib.Path(a.run_dir)/'environment';out.mkdir()\n"
            "(out/'preparer_argv.json').write_text(json.dumps(vars(a)))\n"
            "(out/'normal_probe.json').write_text(json.dumps({'fixture_only':True,'dependencies_ok':False}))\n"
        )
        if preparation == 'failed':
            preparer += (
                "(out/'isolated_probe.json').write_text(json.dumps({'fixture_only':True,'dependencies_ok':True,'cuda_ok':False}))\n"
                "(out/'preparation.json').write_text(json.dumps({'status':'blocked_hardware','fixture_only':True}))\n"
                "print('FIXTURE_PREFLIGHT_FAILED_NO_INSTALL',flush=True)\n"
                'sys.exit(32)\n'
            )
        elif preparation == 'installing':
            preparer += (
                "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
                "(out/'installer_child.pid').write_text(str(child.pid))\n"
                "(out/'preparation.json').write_text(json.dumps({'status':'installing','fixture_only':True}))\n"
                "print('FIXTURE_INSTALLER_CHILD_READY',flush=True)\n"
                'child.wait()\n'
                "print('FIXTURE_INSTALLER_SHOULD_NOT_FINISH',flush=True)\n"
            )
        else:
            preparer += (
                "selected=a.candidate_python or sys.executable\n"
                "(out/'selected_python.txt').write_text(selected+'\\n')\n"
                "(out/'preparation.json').write_text(json.dumps({'status':'ready','selected_python':selected,'installer_started':False,'fixture_only':True}))\n"
                "print('FIXTURE_HEALTHY_ENV_SELECTED_NO_INSTALL',flush=True)\n"
            )
        (package / 'prepare_environment.py').write_text(preparer)
        (package / 'paired_experiment.py').write_text(
            'import json,os,pathlib,sys,time\n'
            "out=pathlib.Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir()\n"
            "(out/'fixture_argv.json').write_text(json.dumps(sys.argv[1:]))\n"
            "(out/'interpreter.json').write_text(json.dumps({'executable':sys.executable,'isolated':sys.flags.isolated,'bridge':os.environ.get('INFTY_FIXTURE_PYTHON_BRIDGE')}))\n"
            "print('FIXTURE_EXPERIMENT_STARTED',flush=True)\n"
            f'time.sleep({experiment_delay})\n'
            "(out/'summary.json').write_text(json.dumps({'completed':True,'fixture_only':True}))\n"
            "print('FIXTURE_EXPERIMENT_ENDED',flush=True)\n"
        )
        (package / 'checksums.sha256').write_text(''.join(
            hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + str(p.relative_to(package)) + '\n'
            for p in sorted(package.rglob('*')) if p.is_file()))
        previous = Path(directory) / 'completed main with spaces'; (previous / 'results').mkdir(parents=True)
        (previous / 'results/run_config.json').write_text('{"mode":"main","fixture_only":true}\n')
        (previous / 'results/summary.json').write_text('{"completed":true,"fixture_only":true}\n')
        env = dict(os.environ)
        env['INFTY_PAIRED_PYTHON'] = str(candidate)
        return root, package, previous, candidate, env

    def output_directory(self, text):
        return Path(next(line.split('=', 1)[1] for line in text.splitlines()
                         if line.startswith('OUTPUT_DIRECTORY=')))

    def wait_for(self, path, seconds=12):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not path.exists():
            time.sleep(.05)
        self.assertTrue(path.exists(), str(path))

    def archive(self, result, expected_exit):
        self.assertEqual((result / 'exit_status.txt').read_text().strip(), str(expected_exit))
        archive = result.with_name(result.name + '_return.zip')
        with zipfile.ZipFile(archive) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(z.read('exit_status.txt').decode().strip(), str(expected_exit))
            self.assertIn('environment/preparation.json', z.namelist())
            self.assertIn('environment.log', z.namelist())
            self.assertIn('code/prepare_environment.py', z.namelist())
            return set(z.namelist())

    def test_preflight_failure_preserves_diagnostics_and_never_starts_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            root, package, previous, candidate, env = self.fixture(directory, 'failed')
            run = subprocess.run(['bash', str(package / 'run_paired.sh'), str(root), 'replicate', str(previous)],
                                 env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 32, run.stdout + run.stderr)
            result = self.output_directory(run.stdout)
            names = self.archive(result, 32)
            self.assertIn('environment/normal_probe.json', names)
            self.assertIn('environment/isolated_probe.json', names)
            self.assertNotIn('results/fixture_argv.json', names)
            self.assertFalse((result / 'experiment.log').exists())
            self.assertIn('FIXTURE_PREFLIGHT_FAILED_NO_INSTALL', (result / 'environment.log').read_text())

    def test_override_interpreter_and_previous_paths_with_spaces_are_forwarded(self):
        for mode, prior_flag in [('main', '--resume-from'), ('replicate', '--prior-run')]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root, package, previous, candidate, env = self.fixture(directory)
                run = subprocess.run(['bash', str(package / 'run_paired.sh'), str(root), mode, str(previous)],
                                     env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                result = self.output_directory(run.stdout)
                self.archive(result, 0)
                prep_args = json.loads((result / 'environment/preparer_argv.json').read_text())
                self.assertEqual(prep_args['candidate_python'], str(candidate))
                self.assertEqual(prep_args['previous'], str(previous))
                args = json.loads((result / 'results/fixture_argv.json').read_text())
                self.assertEqual(args[args.index(prior_flag)+1], str(previous / 'results'))
                self.assertEqual(args[args.index('--mode')+1], mode)
                selected = json.loads((result / 'results/interpreter.json').read_text())
                self.assertEqual(selected['isolated'], 1)
                self.assertEqual(Path(selected['executable']).resolve(), Path(sys.executable).resolve())
                self.assertEqual(selected['bridge'], '1')

    def test_healthy_preparation_starts_tests_and_experiment_without_installer_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root, package, previous, candidate, env = self.fixture(directory)
            run = subprocess.run(['bash', str(package / 'run_paired.sh'), str(root), 'replicate', str(previous)],
                                 env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = self.output_directory(run.stdout)
            self.archive(result, 0)
            prep = json.loads((result / 'environment/preparation.json').read_text())
            self.assertFalse(prep['installer_started'])
            self.assertIn('OK', (result / 'tests.log').read_text())
            self.assertIn('FIXTURE_EXPERIMENT_ENDED', (result / 'experiment.log').read_text())
            self.assertFalse((result / 'environment/installer_child.pid').exists())

    def test_term_stops_installation_process_group_and_archives_flushed_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root, package, previous, candidate, env = self.fixture(directory, 'installing')
            run = subprocess.run(['bash', str(package / 'launch_paired.sh'), str(root), 'replicate', str(previous)],
                                 env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = self.output_directory(run.stdout)
            child_file = result / 'environment/installer_child.pid'
            self.wait_for(child_file)
            child = int(child_file.read_text())
            runner = int((result / 'runner.pid').read_text())
            try:
                self.assertTrue(pid_running(child))
                self.assertTrue(process_alive(result, 'worker'))
                os.kill(runner, signal.SIGTERM)
                self.wait_for(result.with_name(result.name + '_return.zip'))
                names = self.archive(result, 143)
                self.assertFalse(pid_running(child))
                self.assertFalse(process_alive(result, 'worker'))
                self.assertNotIn('results/fixture_argv.json', names)
                text = (result / 'environment.log').read_text()
                self.assertIn('FIXTURE_INSTALLER_CHILD_READY', text)
                self.assertNotIn('FIXTURE_INSTALLER_SHOULD_NOT_FINISH', text)
            finally:
                for pid in [runner, child]:
                    if pid_running(pid):
                        try: os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError: pass

    def test_detached_launcher_and_status_work_with_python_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root, package, previous, candidate, env = self.fixture(directory, experiment_delay=2)
            started = time.monotonic()
            run = subprocess.run(['bash', str(package / 'launch_paired.sh'), str(root), 'replicate', str(previous)],
                                 env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertLess(time.monotonic() - started, 1.5)
            result = self.output_directory(run.stdout)
            self.wait_for(result.with_name(result.name + '_return.zip'))
            self.archive(result, 0)
            status = subprocess.run([str(candidate), '-I', str(package / 'isolated_entry.py'),
                                     'script', 'status_paired.py', str(result)],
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertEqual(json.loads(status.stdout)['state'], 'completed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
