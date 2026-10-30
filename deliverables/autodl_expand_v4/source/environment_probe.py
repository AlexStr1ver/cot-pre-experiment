#!/usr/bin/env python3
"""Inspect package resolution, API health and prior-run compatibility without inference."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import traceback

# This diagnostic is also launched directly with Python -I, which excludes the
# script directory. Load only our named stdlib helper by its explicit path;
# never add cwd/PYTHONPATH before probing the installed torch/transformers.
_hardware_spec = importlib.util.spec_from_file_location(
    '_infty_hardware_compatibility', Path(__file__).with_name('hardware_compatibility.py'))
_hardware_module = importlib.util.module_from_spec(_hardware_spec)
_hardware_spec.loader.exec_module(_hardware_module)
compare_hardware = _hardware_module.compare_hardware


def version_pair(value):
    match = re.match(r"(\d+)\.(\d+)", str(value))
    return [int(match[1]), int(match[2])] if match else None


def module_description(module):
    spec = getattr(module, "__spec__", None)
    loader = getattr(spec, "loader", None)
    paths = getattr(module, "__path__", None)
    return {
        "file": getattr(module, "__file__", None),
        "origin": getattr(spec, "origin", None),
        "path": list(paths) if paths is not None else None,
        "loader_type": type(loader).__name__ if loader is not None else None,
        "version": getattr(module, "__version__", None),
        "is_namespace_package": bool(spec is not None and getattr(spec, "origin", None) is None and paths is not None),
    }


def compare_compatibility(report, expected=None, require_cuda=False):
    """False means an observed mismatch; None means hardware could not be inspected."""
    versions = report["module_versions"]
    result = {"package_versions_match": True, "python_version_compatible": True,
              "cuda_ok": True, "hardware_compatible": True, "mismatches": []}
    targets = {"transformers": "4.51.3"}
    if expected is not None:
        hardware = expected["hardware"]
        targets.update(torch=hardware.get("torch"), cuda_build=hardware.get("cuda_build"))
        expected_pair = version_pair(expected.get("python", ""))
        result["python_version_compatible"] = expected_pair is not None and report["python_major_minor"] == expected_pair
        if not result["python_version_compatible"]:
            result["mismatches"].append({"kind": "python_version", "expected": expected_pair,
                                         "actual": report["python_major_minor"]})
    else:
        # The cloud's CPU-only 2.6.0 build is useful for offline probe validation.
        targets["torch_base"] = "2.6.0"
    actual_versions = dict(versions)
    actual_versions["torch_base"] = str(versions.get("torch", "")).split("+", 1)[0]
    for name, wanted in targets.items():
        actual = actual_versions.get(name)
        if actual != wanted:
            result["package_versions_match"] = False
            result["mismatches"].append({"kind": "package_version", "name": name,
                                         "expected": wanted, "actual": actual})
    if require_cuda or expected is not None:
        available = report["cuda"]["available"]
        result["cuda_ok"] = available
        if available is not True:
            result["mismatches"].append({"kind": "cuda_unavailable_or_unknown", "actual": available})
    if expected is not None:
        compatibility = compare_hardware(report.get("hardware"), expected["hardware"])
        result["hardware_compatibility"] = compatibility
        result["hardware_compatible"] = compatibility["compatible"]
        result["mismatches"].extend(compatibility["mismatches"])
    result["expected_versions"] = targets
    return result


def collect_environment(expected=None, require_cuda=False, importer=None, distribution_version=None):
    importer = importer or importlib.import_module
    distribution_version = distribution_version or importlib.metadata.version
    report = {
        "probe_schema": 2,
        "sys_executable": sys.executable, "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix, "python_version": sys.version,
        "python_major_minor": [sys.version_info.major, sys.version_info.minor],
        "sys_path": list(sys.path), "cwd": os.getcwd(),
        "isolated_mode": bool(sys.flags.isolated), "no_user_site": bool(sys.flags.no_user_site),
        "selected_environment": {name: os.environ.get(name) for name in ("PYTHONPATH", "PYTHONHOME")},
        "modules": {}, "distribution_versions": {}, "module_versions": {},
        "dependencies_ok": False, "basic_torch_ops_ok": False, "transformers_apis_ok": False,
        "cuda": {"available": None, "error": None}, "hardware": None,
        "errors": [], "new_model_inference": False,
    }
    modules = {}
    for name in ["torch", "transformers"]:
        try:
            module = importer(name)
            modules[name] = module
            report["modules"][name] = module_description(module)
            report["module_versions"][name] = getattr(module, "__version__", None)
        except Exception as exc:
            report["modules"][name] = {"import_error": repr(exc)}
            report["module_versions"][name] = None
            report["errors"].append({"kind": "dependency_import", "module": name,
                                      "error": repr(exc), "traceback": traceback.format_exc()})
    for name in ["torch", "transformers", "tokenizers", "safetensors", "huggingface-hub", "numpy"]:
        try:
            report["distribution_versions"][name] = distribution_version(name)
        except Exception as exc:
            report["distribution_versions"][name] = {"error": repr(exc)}
    torch = modules.get("torch")
    if torch is not None:
        report["module_versions"]["cuda_build"] = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            z = torch.zeros(2, 2)
            a = torch.tensor([[1., 2.], [3., 4.]])
            if z.tolist() != [[0., 0.], [0., 0.]] or (a @ a).tolist() != [[7., 10.], [15., 22.]]:
                raise RuntimeError("Basic CPU tensor/matmul returned unexpected values")
            report["basic_torch_ops_ok"] = True
        except Exception as exc:
            report["errors"].append({"kind": "dependency_api", "module": "torch",
                                      "error": repr(exc), "traceback": traceback.format_exc()})
        try:
            available = bool(torch.cuda.is_available())
            report["cuda"]["available"] = available
            if available:
                report["hardware"] = {
                    "torch": torch.__version__, "cuda_build": torch.version.cuda,
                    "device": "cuda", "name": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0)),
                    "total_bytes": torch.cuda.get_device_properties(0).total_memory,
                }
        except Exception as exc:
            report["cuda"]["error"] = repr(exc)
    transformers = modules.get("transformers")
    if transformers is not None:
        try:
            names = ("Qwen2Config", "Qwen2ForCausalLM", "AutoTokenizer", "AutoModelForCausalLM")
            apis = {name: getattr(transformers, name) for name in names}
            if not all(callable(value) for value in apis.values()):
                raise TypeError("A required Transformers API is not callable")
            report["transformers_apis_ok"] = True
            report["transformers_api_modules"] = {name: getattr(value, "__module__", None) for name, value in apis.items()}
        except Exception as exc:
            report["errors"].append({"kind": "dependency_api", "module": "transformers",
                                      "error": repr(exc), "traceback": traceback.format_exc()})
    report["dependencies_ok"] = report["basic_torch_ops_ok"] and report["transformers_apis_ok"]
    report.update(compare_compatibility(report, expected=expected, require_cuda=require_cuda))
    report["passed"] = bool(report["dependencies_ok"] and report["package_versions_match"] and
                            report["python_version_compatible"] and report["cuda_ok"] is True and
                            report["hardware_compatible"] is True)
    if not report["dependencies_ok"]:
        report["classification"] = "dependency_import_or_api_failure"
    elif not report["package_versions_match"]:
        report["classification"] = "package_version_mismatch"
    elif not report["python_version_compatible"]:
        report["classification"] = "python_version_mismatch"
    elif report["cuda_ok"] is not True:
        report["classification"] = "cuda_unavailable_or_uninspectable"
    elif report["hardware_compatible"] is not True:
        report["classification"] = "hardware_mismatch_or_uninspectable"
    else:
        report["classification"] = "passed"
    return report


def load_expected(run_directory):
    path = Path(run_directory).resolve() / "results/run_config.json"
    expected = json.loads(path.read_text(encoding="utf-8"))
    hardware = expected["hardware"]
    if (hardware.get("torch") != "2.6.0+cu124" or hardware.get("cuda_build") != "12.4" or
            expected.get("transformers") != "4.51.3" or hardware.get("device") != "cuda" or
            version_pair(expected.get("python")) is None):
        raise ValueError("Expected main does not identify the frozen CUDA12.4/PyTorch2.6.0/Transformers4.51.3 experiment")
    return expected, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-main", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)
    try:
        expected, expected_path = load_expected(args.expected_main) if args.expected_main else (None, None)
        report = collect_environment(expected=expected, require_cuda=args.require_cuda)
        report["expected_main_config"] = str(expected_path) if expected_path else None
    except Exception as exc:
        report = {"probe_schema": 1, "passed": False, "dependencies_ok": False,
                  "classification": "probe_or_expected_input_error", "error": repr(exc),
                  "traceback": traceback.format_exc(), "sys_executable": sys.executable,
                  "new_model_inference": False}
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, args.output)
    except Exception as exc:
        report.update(passed=False, output_error=repr(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
