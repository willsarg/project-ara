# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""No-model verification of ARA's installed isolated engine environments.

The core never imports an ML runtime.  Each probe executes inside the selected
engine environment and returns JSON.  A probe may initialize a device and run one
scalar operation, so it is an explicit Doctor/install action rather than recon.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from ara import engine_env, engines


_LLAMA_PROBE = r"""
import importlib.metadata as metadata
import json
import platform
from llama_cpp import llama_cpp

supported = bool(llama_cpp.llama_supports_gpu_offload())
raw = llama_cpp.llama_print_system_info()
info = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
print(json.dumps({
    "kind": "llama_cpp",
    "package_version": metadata.version("llama-cpp-python"),
    "python_arch": platform.machine(),
    "system_info": info,
    "gpu_offload": supported,
}))
"""


_CUDA_PROBE = r"""
import json
import torch

available = bool(torch.cuda.is_available())
operation_ok = False
error = None
device = None
capability = None
if available:
    try:
        device = torch.cuda.get_device_name(0)
        capability = ".".join(str(n) for n in torch.cuda.get_device_capability(0))
        value = torch.ones(1, device="cuda")
        torch.cuda.synchronize()
        operation_ok = bool(value.item() == 1)
    except Exception as exc:
        error = str(exc)
print(json.dumps({
    "kind": "torch_cuda",
    "package_version": str(torch.__version__),
    "cuda_build": torch.version.cuda,
    "arch_list": list(torch.cuda.get_arch_list()),
    "available": available,
    "operation_ok": operation_ok,
    "device": device,
    "capability": capability,
    "error": error,
}))
"""


_MLX_PROBE = r"""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import ara_engine_mlx
import mlx.core as mx

engine_root = Path(ara_engine_mlx.__file__).resolve().parent
source_hash = hashlib.sha256()
for source in sorted(
        path for path in engine_root.rglob("*.py")
        if "__pycache__" not in path.parts):
    source_hash.update(source.relative_to(engine_root).as_posix().encode())
    source_hash.update(b"\0")
    source_hash.update(source.read_bytes())

metal_available = bool(mx.metal.is_available())
gpu_count = int(mx.device_count(mx.gpu))
operation_ok = False
error = None
info = {}
try:
    if metal_available and gpu_count:
        mx.set_default_device(mx.gpu)
        value = mx.ones((1,))
        mx.eval(value)
        operation_ok = bool(value.item() == 1)
    info = dict(mx.device_info())
except Exception as exc:
    error = str(exc)
print(json.dumps({
    "kind": "mlx",
    "package_version": metadata.version("mlx"),
    "mlx_lm_version": metadata.version("mlx-lm"),
    "engine_package_version": metadata.version("ara-engine-mlx"),
    "engine_source_digest": "sha256:" + source_hash.hexdigest(),
    "metal_available": metal_available,
    "gpu_count": gpu_count,
    "operation_ok": operation_ok,
    "device": info.get("device_name"),
    "architecture": info.get("architecture"),
    "error": error,
}))
"""


def _probe(key: str, backend: str) -> dict[str, Any]:
    code = _MLX_PROBE if key == "mlx" else _CUDA_PROBE if key == "cuda" else _LLAMA_PROBE
    return engine_env.run_python_json(backend, code, timeout=30.0)


def _finding(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "severity": "warning", "detail": detail}


def _status(status: str, detail: str) -> dict[str, str]:
    return {"status": status, "detail": detail}


def _stable_probe(probe: dict[str, Any]) -> dict[str, Any]:
    """Facts that identify a build/device pairing without transient health fields."""
    kind = probe.get("kind")
    keys = {
        "llama_cpp": ("kind", "package_version", "python_arch", "system_info"),
        "torch_cuda": ("kind", "package_version", "cuda_build", "arch_list",
                       "device", "capability"),
        "mlx": ("kind", "package_version", "mlx_lm_version",
                "engine_package_version", "engine_source_digest", "device", "architecture"),
    }.get(kind, tuple(sorted(probe)))
    return {key: probe.get(key) for key in keys}


def _fingerprint(key: str, version: str | None, schema: str | None,
                 probe: dict[str, Any]) -> str:
    payload = {
        "engine": key,
        "ara_version": version,
        "schema": schema,
        "probe": _stable_probe(probe),
    }
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"engine:v2:sha256:{digest}"


def _strongest_simd(host_features: list[str]) -> str | None:
    for feature in ("AVX-512", "AVX2", "AVX", "SSE4.2", "NEON"):
        if feature in host_features:
            return feature
    return None


def _has_feature(system_info: str, feature: str) -> bool:
    normalized = re.sub(r"[^A-Z0-9]", "", system_info.upper())
    return re.sub(r"[^A-Z0-9]", "", feature.upper()) + "1" in normalized


def _llama_device(system_info: str, expected: str) -> str | None:
    match = re.search(
        rf"(?:^|\|)\s*{re.escape(expected)}\s*:\s*(.*?)(?=\|\s*[A-Za-z0-9_-]+\s*:|$)",
        system_info, re.IGNORECASE)
    if not match:
        return None
    name = re.search(r"(?:NAME|DEVICE)\s*=\s*([^|]+)", match.group(1), re.IGNORECASE)
    return name.group(1).strip() if name else None


def _cuda_build_status(probe: dict[str, Any]) -> tuple[dict[str, str], dict[str, str] | None]:
    if not probe.get("cuda_build"):
        return (_status("mismatch", "PyTorch is not a CUDA build"),
                _finding("backend_missing", "installed PyTorch has no CUDA runtime"))
    capability = re.fullmatch(r"(\d+)\.(\d+)", str(probe.get("capability") or ""))
    if capability is None:
        return _status("unknown", "host CUDA capability is unavailable"), None
    capability_code = int(capability.group(1)) * 10 + int(capability.group(2))
    capability_label = f"{capability.group(1)}.{capability.group(2)}"
    architectures = {str(value).lower() for value in probe.get("arch_list") or []}
    if f"sm_{capability_code}" in architectures:
        return _status(
            "matched", f"CUDA build includes host SM {capability_label}"), None
    ptx_codes = [
        int(match.group(1))
        for architecture in architectures
        if (match := re.fullmatch(r"compute_(\d+)", architecture)) is not None
    ]
    if any(code <= capability_code for code in ptx_codes):
        return _status(
            "matched", f"CUDA build includes PTX compatible with host SM {capability_label}"), None
    if probe.get("operation_ok") is True:
        return _status(
            "matched", f"CUDA build executed successfully on host SM {capability_label}"), None
    detail = f"CUDA build does not include host SM {capability_label} or compatible PTX"
    return _status("mismatch", detail), _finding("cuda_arch_unsupported", detail)


def _workload_status(fingerprint: str | None, rows: list[dict]) -> dict[str, str]:
    measured = [row for row in rows if row.get("safe_context") is not None]
    if not measured:
        return _status("not_verified", "run ara characterize to verify a model workload")
    fingerprints = []
    for row in measured:
        evidence = row.get("evidence")
        observed = (evidence.get("engine", {}).get("fingerprint")
                    if isinstance(evidence, dict) else None)
        if observed:
            fingerprints.append(observed)
    if fingerprint is not None and fingerprint in fingerprints:
        return _status("verified", "characterize verified this installed engine build")
    if fingerprints:
        return _status("stale", "characterization belongs to a different engine build")
    return _status("unknown", "stored characterization predates engine fingerprinting")


def audit_engine(key: str, *, host_features: list[str] | None = None,
                 characterization_rows: list[dict] | None = None) -> dict[str, Any]:
    """Audit one catalog engine without loading a model."""
    if key not in engines.ENGINES:
        raise ValueError(f"unknown engine {key!r}")
    entry = engines.ENGINES[key]
    backend = entry["backend"]
    findings: list[dict[str, str]] = []
    report: dict[str, Any] = {
        "key": key,
        "backend": backend,
        "package": entry["package"],
        "package_version": None,
        "device": None,
        "installation": _status("absent", "engine environment is not installed"),
        "build": _status("not_checked", "engine environment is not installed"),
        "runtime": _status("not_checked", "engine environment is not installed"),
        "workload": _status("not_verified", "run ara characterize to verify a model workload"),
        "capabilities": {},
        "fingerprint": None,
        "findings": findings,
    }
    if not engine_env.exists(backend):
        return report

    stamped = engine_env.stamped_version(backend)
    schema = engine_env.stamped_schema(backend)
    current = engines._ara_version()
    expected_schema = entry.get("env_schema")
    if stamped != current or (expected_schema is not None and schema != expected_schema):
        report["installation"] = _status(
            "stale", "engine stamps do not match the current ARA release")
        findings.append(_finding(
            "installation_stale", f"reinstall with ara install --engine {key} --refresh"))
    else:
        report["installation"] = _status("matched", "engine stamps match this ARA release")

    try:
        probe = _probe(key, backend)
    except (engine_env.EngineEnvError, OSError, ValueError, json.JSONDecodeError) as exc:
        report["build"] = _status("unknown", "engine probe failed")
        report["runtime"] = _status("unknown", "engine probe failed")
        findings.append(_finding("probe_failed", str(exc)))
        report["workload"] = _workload_status(
            None, list(characterization_rows or []))
        return report

    report["package_version"] = probe.get("package_version")
    report["capabilities"] = _stable_probe(probe)
    report["fingerprint"] = _fingerprint(key, stamped, schema, probe)
    kind = probe.get("kind")
    if kind == "llama_cpp":
        info = str(probe.get("system_info") or "")
        if key == "cpu":
            feature = _strongest_simd(list(host_features or []))
            if feature is not None and not _has_feature(info, feature):
                report["build"] = _status(
                    "mismatch", f"installed build does not report host {feature}")
                findings.append(_finding(
                    "cpu_simd_missing",
                    f"host reports {feature} but the installed llama.cpp build does not"))
            elif feature is None:
                report["build"] = _status(
                    "unknown", "host SIMD capability is unknown")
            else:
                report["build"] = _status("matched", f"build reports host {feature}")
            report["runtime"] = _status("matched", "llama.cpp runtime loaded")
        else:
            expected = "Vulkan" if key == "vulkan" else "CUDA"
            registered = re.search(
                rf"(?:^|\|)\s*{re.escape(expected)}\s*:", info, re.IGNORECASE) is not None
            if registered:
                report["build"] = _status("matched", f"{expected} backend is registered")
            else:
                report["build"] = _status("mismatch", f"{expected} backend is not registered")
                findings.append(_finding(
                    "backend_missing", f"installed llama.cpp build does not report {expected}"))
            if registered and probe.get("gpu_offload") is True:
                report["runtime"] = _status("matched", "GPU offload device is available")
            else:
                report["runtime"] = _status("mismatch", "GPU offload device is unavailable")
                findings.append(_finding(
                    "accelerator_unavailable", f"{expected} GPU offload is not available"))
            report["device"] = _llama_device(info, expected)
    elif kind == "torch_cuda":
        report["build"], build_finding = _cuda_build_status(probe)
        if build_finding is not None:
            findings.append(build_finding)
        report["device"] = probe.get("device")
        if probe.get("available") is True and probe.get("operation_ok") is True:
            report["runtime"] = _status("matched", "a CUDA tensor operation completed")
        elif probe.get("available") is not True:
            report["runtime"] = _status("mismatch", "CUDA device is unavailable")
            findings.append(_finding("accelerator_unavailable", "PyTorch cannot see a CUDA device"))
        else:
            report["runtime"] = _status("mismatch", "CUDA tensor operation failed")
            findings.append(_finding(
                "runtime_operation_failed", str(probe.get("error") or "CUDA operation failed")))
    elif kind == "mlx":
        report["build"] = _status("matched", "MLX runtime imported")
        report["device"] = probe.get("device")
        if (probe.get("metal_available") is True and int(probe.get("gpu_count") or 0) > 0
                and probe.get("operation_ok") is True):
            report["runtime"] = _status("matched", "an MLX GPU operation completed")
        else:
            report["runtime"] = _status("mismatch", "MLX Metal GPU operation failed")
            findings.append(_finding(
                "runtime_operation_failed", str(probe.get("error") or "MLX GPU unavailable")))
    else:
        report["build"] = _status("unknown", "engine returned an unknown probe payload")
        report["runtime"] = _status("unknown", "engine returned an unknown probe payload")
        findings.append(_finding("probe_invalid", f"unknown probe kind {kind!r}"))

    report["workload"] = _workload_status(
        report["fingerprint"], list(characterization_rows or []))
    return report


def audit_installed(*, host_features: list[str] | None = None,
                    characterization_rows: list[dict] | None = None) -> list[dict[str, Any]]:
    """Audit every environment that physically exists, in catalog order."""
    rows = list(characterization_rows or [])
    reports = []
    for key, entry in engines.ENGINES.items():
        if not engine_env.exists(entry["backend"]):
            continue
        relevant = [row for row in rows if row.get("engine") == key]
        reports.append(audit_engine(
            key, host_features=host_features, characterization_rows=relevant))
    return reports


def characterization_evidence(report: dict[str, Any]) -> dict[str, Any]:
    """Durable proof tying a successful characterization to this engine build."""
    return {
        "engine": {
            "fingerprint": report.get("fingerprint"),
            "package_version": report.get("package_version"),
            "build_status": report["build"]["status"],
            "runtime_status": report["runtime"]["status"],
        },
        "workload": {"status": "verified", "method": "characterize"},
    }
