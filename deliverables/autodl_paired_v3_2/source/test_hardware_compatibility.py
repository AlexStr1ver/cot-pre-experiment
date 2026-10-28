"""Regression tests for the returned 2 MiB discrepancy and rejection boundaries."""
import copy
import unittest

from hardware_compatibility import compare_hardware, MAX_CAPACITY_DELTA_BYTES


class HardwareCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.main = {"device": "cuda", "name": "NVIDIA GeForce RTX 4090 D",
                     "capability": [8, 9], "torch": "2.6.0+cu124",
                     "cuda_build": "12.4", "total_bytes": 25252724736}

    def test_returned_two_MiB_difference_passes_and_preserves_observations(self):
        current = dict(self.main, total_bytes=25250627584)
        before_current, before_main = copy.deepcopy(current), copy.deepcopy(self.main)
        report = compare_hardware(current, self.main)
        self.assertTrue(report["compatible"], report)
        self.assertEqual(report["mismatches"], [])
        memory = report["memory_capacity_check"]
        self.assertEqual(memory["expected"], 25252724736)
        self.assertEqual(memory["actual"], 25250627584)
        self.assertEqual(memory["signed_delta_bytes"], -2097152)
        self.assertEqual(memory["max_allowed_delta_bytes"], 16 * 2**20)
        self.assertEqual((current, self.main), (before_current, before_main))

    def test_absolute_and_relative_bounds_are_inclusive_and_symmetric(self):
        for total in (25252724736, 2 * 2**30):
            prior = dict(self.main, total_bytes=total)
            allowed = min(MAX_CAPACITY_DELTA_BYTES, total // 1000)
            for sign in (-1, 1):
                with self.subTest(total=total, sign=sign):
                    self.assertTrue(compare_hardware(dict(prior, total_bytes=total + sign * allowed), prior)["compatible"])
                    report = compare_hardware(dict(prior, total_bytes=total + sign * (allowed + 1)), prior)
                    self.assertFalse(report["compatible"])
                    self.assertEqual(report["mismatches"][0]["name"], "total_bytes")

    def test_different_models_capability_device_or_runtime_still_fail(self):
        changed = {"name": "NVIDIA GeForce RTX 4090", "capability": [8, 6],
                   "torch": "2.6.0+cpu", "cuda_build": "12.6", "device": "cpu"}
        for key, value in changed.items():
            with self.subTest(key=key):
                report = compare_hardware(dict(self.main, **{key: value}), self.main)
                self.assertFalse(report["compatible"])
                self.assertIn(key, [item["name"] for item in report["mismatches"]])

    def test_missing_cuda_fields_are_not_accepted_even_when_missing_in_both(self):
        for key in self.main:
            changed = self.main.copy()
            del changed[key]
            for actual, expected in ((changed, self.main), (self.main, changed), (changed, changed)):
                with self.subTest(key=key, actual=actual, expected=expected):
                    self.assertFalse(compare_hardware(actual, expected)["compatible"])

    def test_invalid_capacity_types_and_values_are_rejected(self):
        for value in (None, True, False, 0, -1, 25252724736.0, "25252724736"):
            for changed_side in ("actual", "expected", "both"):
                actual, expected = self.main.copy(), self.main.copy()
                if changed_side in ("actual", "both"):
                    actual["total_bytes"] = value
                if changed_side in ("expected", "both"):
                    expected["total_bytes"] = value
                with self.subTest(value=value, side=changed_side):
                    report = compare_hardware(actual, expected)
                    self.assertFalse(report["compatible"])
                    self.assertFalse(report["memory_capacity_check"]["passed"])

    def test_invalid_cuda_metadata_rejected_even_when_identical(self):
        bad = {"name": [None, "", "  ", 4090], "torch": [None, "", 2.6],
               "cuda_build": [None, "", 12.4], "capability": [None, [], [8], [True, 9], [8, 9.0], [0, 0]]}
        for key, values in bad.items():
            for value in values:
                changed = dict(self.main, **{key: value})
                with self.subTest(key=key, value=value):
                    self.assertFalse(compare_hardware(changed, changed)["compatible"])

    def test_uninspectable_hardware_remains_unknown(self):
        report = compare_hardware(None, self.main)
        self.assertIsNone(report["compatible"])
        self.assertIsNone(report["memory_capacity_check"]["passed"])
        self.assertFalse(compare_hardware([], self.main)["compatible"])
        self.assertFalse(compare_hardware(self.main, None)["compatible"])
        self.assertFalse(compare_hardware({}, {})["compatible"])

    def test_cpu_fixture_compatibility_does_not_require_cuda_metadata(self):
        cpu = {"device": "cpu", "torch": "2.6.0+cpu", "cuda_build": None}
        report = compare_hardware(cpu, cpu)
        self.assertTrue(report["compatible"], report)
        self.assertFalse(report["memory_capacity_check"]["enabled"])
        self.assertFalse(compare_hardware(dict(cpu, torch="2.5.0+cpu"), cpu)["compatible"])
        self.assertFalse(compare_hardware(dict(cpu, total_bytes=1024), cpu)["compatible"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
