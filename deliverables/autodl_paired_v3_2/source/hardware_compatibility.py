"""Bounded GPU-capacity compatibility for environment and immutable-result checks.

Capacity is the driver's reported *total* memory, not current free memory.  A
small change in that report does not identify a different GPU model.  This
policy does not claim that two runs used the same physical GPU or guarantee
numerical equivalence; the independent numerical audits remain mandatory.
"""
from __future__ import annotations

POLICY = "same_runtime_and_gpu_model_capacity_16MiB_and_0.1percent_v1"
MAX_CAPACITY_DELTA_BYTES = 16 * 2**20
CAPACITY_RELATIVE_DENOMINATOR = 1000
STRICT_KEYS = ("device", "name", "capability", "torch", "cuda_build")


def _positive_integer(value):
    return type(value) is int and value > 0


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _valid_field(key, value, device):
    if key == "device":
        return value in ("cuda", "cpu")
    if key == "torch":
        return _nonempty_string(value)
    if device == "cpu":
        # hardware_info omits GPU fields for CPU-only test runs.  A CUDA-built
        # torch can still run CPU tests, so either build value is permitted.
        if key == "cuda_build":
            return value is None or _nonempty_string(value)
        return value is None
    if key in ("name", "cuda_build"):
        return _nonempty_string(value)
    if key == "capability":
        return (isinstance(value, list) and len(value) == 2 and
                type(value[0]) is int and value[0] > 0 and
                type(value[1]) is int and value[1] >= 0)
    return False


def compare_hardware(actual: dict | None, expected: dict) -> dict:
    """Compare immutable metadata without modifying or normalizing either input.

    ``compatible=None`` means actual hardware could not be inspected.  Invalid
    or mismatching observed metadata gives ``False``.  CUDA capacity tolerance
    is inclusive and bounded by BOTH 16 MiB and 0.1% of the prior total.
    """
    result = {
        "policy": POLICY, "compatible": False, "mismatches": [],
        "memory_capacity_check": {
            "enabled": False, "expected": None, "actual": None,
            "signed_delta_bytes": None, "absolute_delta_bytes": None,
            "max_allowed_delta_bytes": None,
            "absolute_limit_bytes": MAX_CAPACITY_DELTA_BYTES,
            "relative_limit_denominator": CAPACITY_RELATIVE_DENOMINATOR,
            "passed": None, "basis": "driver_reported_total_memory_not_free_memory",
        },
    }
    if not isinstance(expected, dict):
        result["mismatches"].append({"kind": "hardware", "name": "metadata",
                                     "expected": expected, "actual": actual,
                                     "reason": "expected_hardware_is_not_a_dictionary"})
        return result
    if actual is None:
        result["compatible"] = None
        result["memory_capacity_check"]["reason"] = "actual_hardware_uninspectable"
        return result
    if not isinstance(actual, dict):
        result["mismatches"].append({"kind": "hardware", "name": "metadata",
                                     "expected": expected, "actual": actual,
                                     "reason": "actual_hardware_is_not_a_dictionary"})
        return result
    for key in STRICT_KEYS:
        wanted, observed = expected.get(key), actual.get(key)
        valid = (_valid_field(key, wanted, expected.get("device")) and
                 _valid_field(key, observed, actual.get("device")))
        if not valid or observed != wanted:
            result["mismatches"].append({"kind": "hardware", "name": key,
                                         "expected": wanted, "actual": observed,
                                         "reason": "invalid_or_missing_metadata" if not valid else "exact_match_required"})
    memory = result["memory_capacity_check"]
    memory.update(expected=expected.get("total_bytes"), actual=actual.get("total_bytes"))
    if expected.get("device") == "cuda" or actual.get("device") == "cuda":
        memory["enabled"] = True
        wanted, observed = memory["expected"], memory["actual"]
        if _positive_integer(wanted) and _positive_integer(observed):
            limit = min(MAX_CAPACITY_DELTA_BYTES, wanted // CAPACITY_RELATIVE_DENOMINATOR)
            delta = observed - wanted
            memory.update(signed_delta_bytes=delta, absolute_delta_bytes=abs(delta),
                          max_allowed_delta_bytes=limit, passed=abs(delta) <= limit,
                          reason="within_bounded_tolerance" if abs(delta) <= limit else "capacity_difference_exceeds_tolerance")
        else:
            memory.update(passed=False, reason="capacity_must_be_a_positive_integer")
        if not memory["passed"]:
            result["mismatches"].append({"kind": "hardware", "name": "total_bytes",
                                         "expected": wanted, "actual": observed,
                                         "reason": memory["reason"],
                                         "max_allowed_delta_bytes": memory["max_allowed_delta_bytes"]})
    else:
        # No GPU capacity field is expected from hardware_info(device='cpu').
        memory.update(passed=memory["expected"] is None and memory["actual"] is None,
                      reason="not_applicable_to_cpu")
        if not memory["passed"]:
            result["mismatches"].append({"kind": "hardware", "name": "total_bytes",
                                         "expected": memory["expected"], "actual": memory["actual"],
                                         "reason": "unexpected_gpu_capacity_for_cpu"})
    result["compatible"] = not result["mismatches"]
    return result
