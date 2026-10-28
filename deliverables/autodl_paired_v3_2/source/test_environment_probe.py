"""Focused checks for environment diagnostics; no package installation or model load."""
import contextlib
import copy
import importlib.machinery
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
import subprocess
import sys

from environment_probe import collect_environment, compare_compatibility, main, module_description


class EnvironmentProbeTests(unittest.TestCase):
    def test_actual_failed_capacity_report_passes_without_changing_measurements(self):
        fixture = json.loads(Path(__file__).with_name('capacity_failure_fixture.json').read_text())
        report, expected = fixture['report'], fixture['expected']
        before = copy.deepcopy(fixture)
        self.assertFalse(report['hardware_compatible'])  # v3.1's observed failure
        result = compare_compatibility(report, expected, True)
        self.assertTrue(result['hardware_compatible'], result)
        self.assertEqual(result['mismatches'], [])
        self.assertEqual(fixture, before)
        self.assertEqual(report['hardware']['total_bytes'], 25250627584)
        self.assertEqual(expected['hardware']['total_bytes'], 25252724736)
        for field in ['package_versions_match', 'python_version_compatible', 'cuda_ok']:
            self.assertTrue(result[field])

    def test_direct_isolated_probe_loads_shared_policy_without_cwd_imports(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'probe.json'
            process = subprocess.run([sys.executable, '-I', str(Path(__file__).with_name('environment_probe.py')),
                                      '--output', str(output)], cwd=d, capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            result = json.loads(output.read_text())
            self.assertTrue(result['passed'])
            self.assertTrue(result['isolated_mode'])
            self.assertNotIn(d, result['sys_path'])

    def test_actual_healthy_cpu_imports_and_operations(self):
        report = collect_environment()
        self.assertTrue(report["dependencies_ok"], report["errors"])
        self.assertTrue(report["basic_torch_ops_ok"])
        self.assertTrue(report["transformers_apis_ok"])
        self.assertTrue(report["passed"], report)
        self.assertFalse(report["new_model_inference"])
        self.assertEqual(set(report["selected_environment"]), {"PYTHONPATH", "PYTHONHOME"})

    def test_namespace_modules_report_missing_apis_without_crash(self):
        def importer(name):
            module = types.ModuleType(name)
            module.__path__ = ["/fixture/incomplete/" + name]
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            return module
        report = collect_environment(importer=importer, distribution_version=lambda name: "fixture-metadata")
        self.assertFalse(report["passed"])
        self.assertFalse(report["dependencies_ok"])
        self.assertEqual(report["classification"], "dependency_import_or_api_failure")
        self.assertTrue(report["modules"]["torch"]["is_namespace_package"])
        self.assertIsNone(report["modules"]["transformers"]["file"])
        self.assertEqual({e["module"] for e in report["errors"]}, {"torch", "transformers"})

    def test_import_errors_are_captured_as_dependency_failures(self):
        def importer(name):
            raise ImportError("fixture missing dependency: " + name)
        report = collect_environment(importer=importer)
        self.assertFalse(report["dependencies_ok"])
        self.assertFalse(report["passed"])
        self.assertTrue(all(e["kind"] == "dependency_import" for e in report["errors"]))

    def test_expected_gpu_versions_hardware_and_python_are_separate(self):
        hardware = {"torch":"2.6.0+cu124", "cuda_build":"12.4", "device":"cuda",
                    "name":"NVIDIA GeForce RTX 4090 D", "capability":[8,9], "total_bytes":25252724736}
        expected = {"hardware":hardware, "python":"3.10.8 fixture", "transformers":"4.51.3"}
        report = {"module_versions":{"torch":"2.6.0+cu124", "transformers":"4.51.3", "cuda_build":"12.4"},
                  "python_major_minor":[3,10], "cuda":{"available":True}, "hardware":hardware}
        result = compare_compatibility(report, expected, True)
        self.assertTrue(all(result[k] for k in ["package_versions_match","python_version_compatible","cuda_ok","hardware_compatible"]))
        changed = copy.deepcopy(report); changed["hardware"]["name"] = "different GPU"
        result = compare_compatibility(changed, expected, True)
        self.assertFalse(result["hardware_compatible"])
        self.assertTrue(result["package_versions_match"])
        changed = copy.deepcopy(report); changed["module_versions"]["torch"] = "2.6.0+cpu"
        self.assertFalse(compare_compatibility(changed, expected)["package_versions_match"])
        changed = copy.deepcopy(report); changed["python_major_minor"] = [3,12]
        self.assertFalse(compare_compatibility(changed, expected)["python_version_compatible"])
        changed = copy.deepcopy(report); changed["cuda"]["available"] = None; changed["hardware"] = None
        result = compare_compatibility(changed, expected, True)
        self.assertIsNone(result["cuda_ok"])
        self.assertIsNone(result["hardware_compatible"])

    def test_cli_always_records_bad_expected_input_as_json(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(["--output", str(output), "--expected-main", str(Path(d)/"missing")])
            self.assertEqual(status, 1)
            report = json.loads(output.read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["classification"], "probe_or_expected_input_error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
