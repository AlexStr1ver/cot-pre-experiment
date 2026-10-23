#!/usr/bin/env python3
"""CPU-only runner lifecycle tests, with explicit fake model/planner fixtures."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile

from status_phase2 import pack_result, runner_alive

PACKAGE = Path(__file__).resolve().parent


class RunnerTests(unittest.TestCase):
    def make_fixture(self, directory, exit_code, delay=0):
        root = Path(directory) / "autodl"
        root.mkdir()
        venv = root / "venvs/infty-official/bin"
        venv.mkdir(parents=True)
        (venv / "python").symlink_to(sys.executable)
        package = Path(directory) / "package"
        package.mkdir()
        for name in ("run_phase2.sh", "launch_phase2.sh", "status_phase2.py"):
            shutil.copyfile(PACKAGE / name, package / name)
        for name in ("test_prepare_plan.py", "test_collect_phase2.py", "v1_core.py"):
            (package / name).write_text("# Explicit test fixture, no model inference\n")
        (package / "prepare_plan.py").write_text(
            "import pathlib, sys\npathlib.Path(sys.argv[sys.argv.index('--output')+1]).write_text('{}')\n")
        (package / "collect_phase2.py").write_text(
            "import json, pathlib, sys, time\n"
            "assert sys.argv[sys.argv.index('--dtype')+1] == 'float32'\n"
            "out=pathlib.Path(sys.argv[sys.argv.index('--output')+1]); out.mkdir()\n"
            f"time.sleep({delay})\n"
            f"(out/'summary.json').write_text(json.dumps({{'completed': {exit_code == 0}, 'fixture_only': True}}))\n"
            "print('FIXTURE_COLLECTOR_EXECUTED', flush=True)\n"
            f"sys.exit({exit_code})\n")
        (package / "checksums.sha256").write_text("".join(
            hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n"
            for p in sorted(package.iterdir()) if p.is_file()))
        return root, package

    def result_path(self, output):
        return Path(next(line.split("=", 1)[1] for line in output.splitlines()
                         if line.startswith("OUTPUT_DIRECTORY=")))

    def verify_archive(self, root, expected_code):
        self.assertEqual((root / "exit_status.txt").read_text().strip(), str(expected_code))
        archive = root.with_name(root.name + "_return.zip")
        with zipfile.ZipFile(archive) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(z.read("exit_status.txt").decode().strip(), str(expected_code))
            self.assertIn("code/collect_phase2.py", z.namelist())
            self.assertIn("FIXTURE_COLLECTOR_EXECUTED", z.read("signals.log").decode())

    def test_success_and_failure_preserve_exit_and_archive(self):
        for code in (0, 7):
            with self.subTest(exit_code=code), tempfile.TemporaryDirectory() as directory:
                root, package = self.make_fixture(directory, code)
                run = subprocess.run(["bash", str(package / "run_phase2.sh"), str(root)],
                                     capture_output=True, text=True, timeout=15)
                self.assertEqual(run.returncode, code, run.stdout + run.stderr)
                self.verify_archive(self.result_path(run.stdout), code)

    def test_detached_launcher_returns_then_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root, package = self.make_fixture(directory, 0, delay=2)
            started = time.monotonic()
            run = subprocess.run(["bash", str(package / "launch_phase2.sh"), str(root)],
                                 capture_output=True, text=True, timeout=5)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertLess(time.monotonic() - started, 1.5)
            result = self.result_path(run.stdout)
            deadline = time.monotonic() + 12
            observed_live = False
            while time.monotonic() < deadline:
                observed_live |= runner_alive(result)
                if result.with_name(result.name + "_return.zip").exists():
                    break
                time.sleep(.05)
            self.assertTrue(observed_live)
            self.verify_archive(result, 0)

    def test_repack_never_overwrites_and_manual_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            (root / "partial.json").write_text('{"status":"measuring"}')
            first = pack_result(root)
            first_bytes = first.read_bytes()
            (root / "later.log").write_text("interrupted process retained logs\n")
            second = pack_result(root)
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), first_bytes)
            with zipfile.ZipFile(second) as z:
                self.assertIn("later.log", z.namelist())
            self.assertFalse(runner_alive(root))

    def test_reuse_refused_and_sourcing_does_not_exit_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root, package = self.make_fixture(directory, 0)
            previous = root / "previous"
            previous.mkdir()
            (previous / "exit_status.txt").write_text("old\n")
            run = subprocess.run(["bash", str(package / "run_phase2.sh"), str(root), "3", str(previous)],
                                 capture_output=True, text=True, timeout=5)
            self.assertEqual(run.returncode, 2)
            self.assertEqual((previous / "exit_status.txt").read_text(), "old\n")
            sourced = subprocess.run(["bash", "-c", 'source "$1"; printf "PARENT_ALIVE\\n"',
                                      "test", str(package / "run_phase2.sh")],
                                     capture_output=True, text=True, timeout=5)
            self.assertEqual(sourced.returncode, 0)
            self.assertIn("PARENT_ALIVE", sourced.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
