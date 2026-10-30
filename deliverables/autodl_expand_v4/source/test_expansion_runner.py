"""CPU lifecycle fixtures for v4 launch/collect/resume/archive, never GPU evidence."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile

from status_expanded import pack_result, process_alive

PACKAGE = Path(__file__).resolve().parent


class ExpansionRunnerTests(unittest.TestCase):
    def fixture(self, directory, failing_stage=None, delay=0):
        root = Path(directory) / 'autodl root with spaces'; root.mkdir()
        package = Path(directory) / 'v4 package with spaces'; package.mkdir()
        anchor = root / 'infty_paired_v3_rsiFEcc2'; anchor.mkdir()
        candidate = root / 'healthy env with spaces/bin/python'
        candidate.parent.mkdir(parents=True)
        candidate.write_text('#!/usr/bin/env bash\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
        candidate.chmod(0o755)
        env = dict(os.environ, INFTY_PAIRED_PYTHON=str(candidate))
        for name in ('run_expanded.sh', 'launch_expanded.sh', 'isolated_expand_entry.py', 'status_expanded.py'):
            shutil.copyfile(PACKAGE / name, package / name)
        for name in ('test_decoder', 'test_actions', 'test_experiment', 'test_hardware_compatibility', 'test_expanded_experiment'):
            (package / (name + '.py')).write_text('# Explicit empty lifecycle fixture.\n')
        (package / 'test_expanded_experiment.py').write_text(
            'import sys,unittest\n'
            'class FixtureTests(unittest.TestCase):\n'
            ' def test_isolated_mode(self):self.assertEqual(sys.flags.isolated,1)\n')
        historical = package / 'evidence_phase2fix'; historical.mkdir()
        (historical / 'plan.json').write_text('{"fixture":true}\n')
        (package / 'expansion_plan.json').write_text('{"fixture":true}\n')
        (package / 'prepare_environment.py').write_text(
            'import argparse,json,pathlib,sys\n'
            'p=argparse.ArgumentParser()\n'
            "for n in ['root','package','run-dir','previous']:p.add_argument('--'+n)\n"
            "p.add_argument('--candidate-python');a=p.parse_args()\n"
            "out=pathlib.Path(a.run_dir)/'environment';out.mkdir()\n"
            "(out/'fixture_args.json').write_text(json.dumps(vars(a)))\n"
            "(out/'preparation.json').write_text(json.dumps({'completed':True,'fixture_only':True}))\n"
            "(out/'selected_python.txt').write_text(a.candidate_python+'\\n')\n")
        (package / 'prepare_expansion_evidence.py').write_text(
            'import argparse,json,pathlib,shutil,sys\n'
            'p=argparse.ArgumentParser();s=p.add_subparsers(dest="command",required=True)\n'
            'a=s.add_parser("prepare")\n'
            "for n in ['model','baseline','output']:a.add_argument('--'+n,required=True)\n"
            "a.add_argument('--source-run');s.add_parser('finalize').add_argument('--evidence',required=True)\n"
            'a=p.parse_args()\n'
            "if a.command=='prepare':\n"
            ' out=pathlib.Path(a.output);out.mkdir()\n'
            " (out.parent/'fixture_prepare_args.json').write_text(json.dumps(vars(a)))\n"
            " (out.parent/'evidence_preparation.json').write_text(json.dumps({'completed':True,'fixture_only':True}))\n"
            " print('FIXTURE_EVIDENCE_PREPARED',flush=True)\n"
            + (" sys.exit(37)\n" if failing_stage == 'prepare' else '') +
            " (out/'plan.json').write_text('{\"fixture\":true}\\n')\n"
            " if a.source_run:\n"
            "  shutil.copytree(pathlib.Path(a.source_run)/'evidence',out,dirs_exist_ok=True)\n"
            'else:\n'
            ' out=pathlib.Path(a.evidence)\n'
            " (out/'evidence_manifest.json').write_text('{\"fixture\":true}\\n')\n"
            " print('FIXTURE_EVIDENCE_FINALIZED',flush=True)\n")
        (package / 'collect_phase2.py').write_text(
            'import json,pathlib,subprocess,sys,time\n'
            "out=pathlib.Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir()\n"
            "(out/'fixture_args.json').write_text(json.dumps(sys.argv[1:]))\n"
            "print('FIXTURE_SIGNALS_STARTED',flush=True)\n"
            + ("child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
               "(out/'child.pid').write_text(str(child.pid))\nchild.wait()\n" if failing_stage == 'long_signal' else '') +
            ("sys.exit(31)\n" if failing_stage == 'signals' else '') +
            "(out/'summary.json').write_text(json.dumps({'completed':True,'executed_positions':15,'fixture_only':True}))\n"
            "print('FIXTURE_SIGNALS_COMPLETED',flush=True)\n")
        (package / 'expanded_experiment.py').write_text(
            'import json,pathlib,sys,time\n'
            "out=pathlib.Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir()\n"
            "(out/'fixture_args.json').write_text(json.dumps(sys.argv[1:]))\n"
            "print('FIXTURE_ACTIONS_STARTED',flush=True)\n"
            f'time.sleep({delay})\n' +
            ("sys.exit(7)\n" if failing_stage == 'actions' else '') +
            "(out/'summary.json').write_text(json.dumps({'completed':True,'completed_branches':60,'fixture_only':True}))\n"
            "print('FIXTURE_ACTIONS_COMPLETED',flush=True)\n")
        (package / 'checksums.sha256').write_text(''.join(
            hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + str(p.relative_to(package)) + '\n'
            for p in sorted(package.rglob('*')) if p.is_file()))
        return root, package, anchor, candidate, env

    def result_path(self, stdout):
        return Path(next(line.split('=', 1)[1] for line in stdout.splitlines()
                         if line.startswith('OUTPUT_DIRECTORY=')))

    def wait_for(self, path, seconds=15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not path.exists():
            time.sleep(.05)
        self.assertTrue(path.exists(), str(path))

    def archive(self, result, expected_exit):
        self.assertEqual((result / 'exit_status.txt').read_text().strip(), str(expected_exit))
        with zipfile.ZipFile(result.with_name(result.name + '_return.zip')) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(archive.read('exit_status.txt').decode().strip(), str(expected_exit))
            self.assertIn('code/expanded_experiment.py', archive.namelist())
            return set(archive.namelist())

    def foreground(self, root, package, anchor, env, resume=None):
        args = ['bash', str(package / 'run_expanded.sh'), str(root), str(anchor)]
        if resume is not None:
            args.append(str(resume))
        return subprocess.run(args, env=env, capture_output=True, text=True, timeout=25)

    def test_fresh_flow_collects_then_seals_before_actions_and_forwards_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root, package, anchor, candidate, env = self.fixture(directory)
            run = self.foreground(root, package, anchor, env)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = self.result_path(run.stdout); names = self.archive(result, 0)
            self.assertIn('evidence/signals/summary.json', names)
            self.assertIn('evidence/evidence_manifest.json', names)
            self.assertLess(run.stdout.index('FIXTURE_SIGNALS_COMPLETED'), run.stdout.index('FIXTURE_EVIDENCE_FINALIZED'))
            self.assertLess(run.stdout.index('FIXTURE_EVIDENCE_FINALIZED'), run.stdout.index('FIXTURE_ACTIONS_STARTED'))
            preparation = json.loads((result / 'environment/fixture_args.json').read_text())
            self.assertEqual(preparation['previous'], str(anchor))
            self.assertEqual(preparation['candidate_python'], str(candidate))
            signal_args = json.loads((result / 'evidence/signals/fixture_args.json').read_text())
            self.assertEqual(signal_args[signal_args.index('--dtype') + 1], 'float32')
            self.assertEqual(signal_args[signal_args.index('--prefill-chunk') + 1], '128')
            action_args = json.loads((result / 'results/fixture_args.json').read_text())
            self.assertEqual(action_args[action_args.index('--phase2-input') + 1], str(result / 'evidence'))
            self.assertNotIn('--resume-from', action_args)

    def test_resume_keeps_source_bytes_and_skips_signal_collection_and_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root, package, anchor, _, env = self.fixture(directory)
            previous = root / 'prior expansion with spaces'
            (previous / 'evidence/signals').mkdir(parents=True)
            (previous / 'results').mkdir()
            (previous / 'evidence/evidence_manifest.json').write_text('{"old_manifest":"exact bytes"}\n')
            (previous / 'evidence/signals/summary.json').write_text('{"completed":true,"fixture_only":true}\n')
            before = {str(p.relative_to(previous)): p.read_bytes() for p in previous.rglob('*') if p.is_file()}
            run = self.foreground(root, package, anchor, env, previous)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = self.result_path(run.stdout); names = self.archive(result, 0)
            self.assertNotIn('signals.log', names)
            self.assertNotIn('evidence_finalization.log', names)
            self.assertNotIn('FIXTURE_SIGNALS_STARTED', run.stdout)
            action_args = json.loads((result / 'results/fixture_args.json').read_text())
            self.assertEqual(action_args[action_args.index('--resume-from') + 1], str(previous / 'results'))
            prep_args = json.loads((result / 'fixture_prepare_args.json').read_text())
            self.assertEqual(prep_args['source_run'], str(previous))
            self.assertEqual(before, {str(p.relative_to(previous)): p.read_bytes() for p in previous.rglob('*') if p.is_file()})
            self.assertEqual((result / 'evidence/evidence_manifest.json').read_bytes(),
                             (previous / 'evidence/evidence_manifest.json').read_bytes())

    def test_failures_preserve_exit_and_partial_diagnostics_without_later_stages(self):
        for stage, code in [('prepare', 37), ('signals', 31), ('actions', 7)]:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                root, package, anchor, _, env = self.fixture(directory, failing_stage=stage)
                run = self.foreground(root, package, anchor, env)
                self.assertEqual(run.returncode, code, run.stdout + run.stderr)
                result = self.result_path(run.stdout); names = self.archive(result, code)
                self.assertIn('evidence_preparation.log', names)
                if stage != 'actions':
                    self.assertFalse((result / 'results').exists())
                    self.assertNotIn('experiment.log', names)
                    self.assertNotIn('evidence/evidence_manifest.json', names)
                if stage == 'prepare':
                    self.assertNotIn('signals.log', names)
                if stage == 'signals':
                    self.assertIn('evidence/signals/fixture_args.json', names)

    def test_term_stops_collector_process_group_and_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root, package, anchor, _, env = self.fixture(directory, failing_stage='long_signal')
            run = subprocess.run(['bash', str(package / 'launch_expanded.sh'), str(root), str(anchor)],
                                 env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = self.result_path(run.stdout)
            self.wait_for(result / 'evidence/signals/child.pid')
            runner = int((result / 'runner.pid').read_text())
            child = int((result / 'evidence/signals/child.pid').read_text())
            try:
                self.assertTrue(process_alive(result, 'worker'))
                os.kill(runner, signal.SIGTERM)
                self.wait_for(result.with_name(result.name + '_return.zip'))
                names = self.archive(result, 143)
                self.assertFalse(process_alive(result, 'worker'))
                stat = Path(f'/proc/{child}/stat')
                if stat.exists():
                    self.assertEqual(stat.read_text().rsplit(')', 1)[1].split()[0], 'Z')
                self.assertNotIn('experiment.log', names)
                self.assertNotIn('FIXTURE_SIGNALS_COMPLETED', (result / 'signals.log').read_text())
            finally:
                for pid in (runner, child):
                    try: os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError: pass

    def test_detached_launch_and_stdlib_status_report_full_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            root, package, anchor, candidate, env = self.fixture(directory, delay=1)
            started = time.monotonic()
            run = subprocess.run(['bash', str(package / 'launch_expanded.sh'), str(root), str(anchor)],
                                 env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertLess(time.monotonic() - started, 1.5)
            result = self.result_path(run.stdout)
            self.wait_for(result.with_name(result.name + '_return.zip'))
            self.archive(result, 0)
            status = subprocess.run([str(candidate), '-I', str(package / 'isolated_expand_entry.py'),
                                     'script', 'status_expanded.py', str(result)],
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(status.returncode, 0, status.stderr)
            result_status = json.loads(status.stdout)
            self.assertEqual(result_status['state'], 'completed')
            self.assertEqual(result_status['summary']['completed_branches'], 60)
            self.assertEqual(result_status['signals']['executed_positions'], 15)

    def test_existing_run_refused_and_repack_preserves_prior_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root, package, anchor, _, env = self.fixture(directory)
            prior = root / 'previous run'; prior.mkdir()
            (prior / 'partial.json').write_text('{"untouched":true}\n')
            archive = pack_result(prior); original = archive.read_bytes()
            (prior / 'later.log').write_text('later\n')
            repack = pack_result(prior)
            self.assertNotEqual(archive, repack)
            self.assertEqual(archive.read_bytes(), original)
            run = subprocess.run(['bash', str(package / 'run_expanded.sh'), str(root), str(anchor), '', str(prior)],
                                 env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(run.returncode, 2)
            self.assertEqual((prior / 'partial.json').read_text(), '{"untouched":true}\n')
            self.assertFalse((prior / 'exit_status.txt').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
