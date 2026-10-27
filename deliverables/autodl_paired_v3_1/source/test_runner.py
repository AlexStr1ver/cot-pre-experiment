#!/usr/bin/env python3
"""Runner lifecycle tests with explicit stdlib-only experiment fixtures, no GPU."""
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

from status_paired import pack_result, process_alive, runner_alive

PACKAGE = Path(__file__).resolve().parent


class RunnerTests(unittest.TestCase):
    def fixture(self, directory, exit_code=0, delay=0):
        root = Path(directory) / 'autodl with spaces'; root.mkdir()
        (root / 'infty_paired_v3_kt2eZTc2/results').mkdir(parents=True)
        venv = root / 'venvs/infty-official/bin'; venv.mkdir(parents=True)
        # A symlink outside the active venv loses its site-packages under -I.
        (venv / 'python').write_text('#!/usr/bin/env bash\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
        (venv / 'python').chmod(0o755)
        package = Path(directory) / 'package with spaces'; package.mkdir()
        for name in ('run_paired.sh', 'launch_paired.sh', 'status_paired.py', 'isolated_entry.py'):
            shutil.copyfile(PACKAGE / name, package / name)
        (package / 'prepare_environment.py').write_text(
            'import argparse,json,pathlib,sys\n'
            'p=argparse.ArgumentParser()\n'
            "for n in ['root','package','run-dir','previous']:p.add_argument('--'+n)\n"
            "p.add_argument('--candidate-python');a=p.parse_args()\n"
            "out=pathlib.Path(a.run_dir)/'environment';out.mkdir()\n"
            "(out/'selected_python.txt').write_text(sys.executable+'\\n')\n"
            "(out/'preparation.json').write_text(json.dumps({'status':'ready','fixture_only':True}))\n")
        for name in ('test_decoder.py', 'test_actions.py', 'test_experiment.py'):
            (package / name).write_text('# Explicit lifecycle fixture module; no inference tests.\n')
        (package / 'test_experiment.py').write_text(
            'import json,pathlib,unittest\n'
            'class FixtureInputTests(unittest.TestCase):\n'
            ' def test_wrapper_selected_fixture_working_directory(self):\n'
            "  self.assertEqual(json.loads(pathlib.Path('evidence_phase2fix/plan.json').read_text()),{'fixture':True})\n")
        evidence = package / 'evidence_phase2fix'; evidence.mkdir()
        (evidence / 'plan.json').write_text('{"fixture": true}')
        (package / 'paired_experiment.py').write_text(
            'import json,os,pathlib,sys,time\n'
            "out=pathlib.Path(sys.argv[sys.argv.index('--output')+1]); out.mkdir()\n"
            "(out/'fixture_argv.json').write_text(json.dumps(sys.argv[1:]))\n"
            "(out/'worker_ready.txt').write_text(str(os.getpid()))\n"
            "print('FIXTURE_EXPERIMENT_STARTED',flush=True)\n"
            f'time.sleep({delay})\n'
            f"(out/'summary.json').write_text(json.dumps({{'completed': {exit_code==0}, 'fixture_only': True}}))\n"
            "print('FIXTURE_EXPERIMENT_ENDED',flush=True)\n"
            f'sys.exit({exit_code})\n')
        (package / 'checksums.sha256').write_text(''.join(
            hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + str(p.relative_to(package)) + '\n'
            for p in sorted(package.rglob('*')) if p.is_file()))
        return root, package

    def result_path(self, text):
        return Path(next(line.split('=', 1)[1] for line in text.splitlines()
                         if line.startswith('OUTPUT_DIRECTORY=')))

    def wait_archive(self, result, seconds=12):
        archive = result.with_name(result.name + '_return.zip')
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not archive.exists():
            time.sleep(.05)
        self.assertTrue(archive.exists(), (result / 'launcher.log').read_text())
        return archive

    def verify_archive(self, result, code):
        self.assertEqual((result / 'exit_status.txt').read_text().strip(), str(code))
        with zipfile.ZipFile(result.with_name(result.name + '_return.zip')) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(z.read('exit_status.txt').decode().strip(), str(code))
            self.assertIn('code/paired_experiment.py', z.namelist())
            self.assertIn('code/evidence_phase2fix/plan.json', z.namelist())
            self.assertIn('FIXTURE_EXPERIMENT_STARTED', z.read('experiment.log').decode())

    def test_success_and_failure_exit_preserved_and_logs_archived(self):
        for code in (0, 7):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as d:
                root, package = self.fixture(d, code)
                run = subprocess.run(['bash',str(package/'run_paired.sh'),str(root)],capture_output=True,text=True,timeout=15)
                self.assertEqual(run.returncode, code, run.stdout + run.stderr)
                result = self.result_path(run.stdout)
                self.verify_archive(result, code)
                self.assertIn('FIXTURE_EXPERIMENT_ENDED', (result/'experiment.log').read_text())

    def test_main_and_replicate_forward_previous_paths_with_spaces(self):
        for mode, flag in [('main','--resume-from'),('replicate','--prior-run')]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                root, package = self.fixture(d)
                previous = Path(d)/'old experiment'; (previous/'results').mkdir(parents=True)
                run = subprocess.run(['bash',str(package/'run_paired.sh'),str(root),mode,str(previous)],capture_output=True,text=True,timeout=15)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                args=json.loads((self.result_path(run.stdout)/'results/fixture_argv.json').read_text())
                self.assertEqual(args[args.index('--mode')+1],mode)
                self.assertEqual(args[args.index(flag)+1],str(previous/'results'))
                self.assertEqual(args[args.index('--phase2-input')+1],str(package/'evidence_phase2fix'))

    def test_detached_launcher_returns_before_work_finishes(self):
        with tempfile.TemporaryDirectory() as d:
            root,package=self.fixture(d,delay=2)
            started=time.monotonic()
            run=subprocess.run(['bash',str(package/'launch_paired.sh'),str(root)],capture_output=True,text=True,timeout=5)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertLess(time.monotonic()-started,1.5)
            result=self.result_path(run.stdout)
            self.wait_archive(result)
            self.verify_archive(result,0)
            status=subprocess.run([sys.executable,str(package/'status_paired.py'),str(result)],capture_output=True,text=True,timeout=5)
            self.assertEqual(json.loads(status.stdout)['state'],'completed')

    def test_term_stops_active_worker_before_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root,package=self.fixture(d,delay=30)
            run=subprocess.run(['bash',str(package/'launch_paired.sh'),str(root)],capture_output=True,text=True,timeout=5)
            result=self.result_path(run.stdout)
            deadline=time.monotonic()+15
            while time.monotonic()<deadline and not (result/'results/worker_ready.txt').exists():
                time.sleep(.05)
            self.assertTrue((result/'results/worker_ready.txt').exists())
            self.assertTrue(process_alive(result,'worker'))
            os.kill(int((result/'runner.pid').read_text()),signal.SIGTERM)
            self.wait_archive(result)
            self.verify_archive(result,143)
            self.assertFalse(process_alive(result,'worker'))
            self.assertNotIn('FIXTURE_EXPERIMENT_ENDED',(result/'experiment.log').read_text())

    def test_repack_keeps_existing_archive_and_reuse_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root,package=self.fixture(d)
            previous=root/'prior result';previous.mkdir();(previous/'partial.json').write_text('{}')
            first=pack_result(previous);old=first.read_bytes()
            (previous/'later.log').write_text('later\n')
            second=pack_result(previous)
            self.assertNotEqual(first,second);self.assertEqual(first.read_bytes(),old)
            with zipfile.ZipFile(second) as z:self.assertIn('later.log',z.namelist())
            run=subprocess.run(['bash',str(package/'run_paired.sh'),str(root),'smoke','',str(previous)],capture_output=True,text=True,timeout=5)
            self.assertEqual(run.returncode,2);self.assertEqual((previous/'partial.json').read_text(),'{}')
            sourced=subprocess.run(['bash','-c','source "$1"; printf "PARENT_ALIVE\\n"','test',str(package/'run_paired.sh')],capture_output=True,text=True,timeout=5)
            self.assertIn('PARENT_ALIVE',sourced.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
