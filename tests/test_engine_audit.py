# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Truthful installed-engine verification for ``ara doctor --engines``."""
from __future__ import annotations

from ara import engine_audit

import pytest


def _installed(monkeypatch, *, version="0.1.3", schema=None):
    monkeypatch.setattr(engine_audit.engine_env, "exists", lambda _backend: True)
    monkeypatch.setattr(engine_audit.engine_env, "stamped_version", lambda _backend: version)
    monkeypatch.setattr(engine_audit.engine_env, "stamped_schema", lambda _backend: schema)
    monkeypatch.setattr(engine_audit.engines, "_ara_version", lambda: "0.1.3")


def test_cpu_audit_matches_reported_host_simd(monkeypatch):
    _installed(monkeypatch)
    monkeypatch.setattr(engine_audit, "_probe", lambda _key, _backend: {
        "kind": "llama_cpp",
        "package_version": "0.3.34",
        "python_arch": "arm64",
        "system_info": "CPU : NEON = 1 | ARM_FMA = 1 |",
        "gpu_offload": True,
    })

    report = engine_audit.audit_engine("cpu", host_features=["NEON", "BF16"])

    assert report["installation"]["status"] == "matched"
    assert report["build"]["status"] == "matched"
    assert report["runtime"]["status"] == "matched"
    assert report["workload"]["status"] == "not_verified"
    assert report["package_version"] == "0.3.34"
    assert report["fingerprint"].startswith("engine:v1:sha256:")
    assert report["findings"] == []


def test_cpu_audit_reports_missing_strongest_host_simd(monkeypatch):
    _installed(monkeypatch)
    monkeypatch.setattr(engine_audit, "_probe", lambda _key, _backend: {
        "kind": "llama_cpp",
        "package_version": "0.3.19",
        "python_arch": "x86_64",
        "system_info": "CPU : AVX = 1 | SSE4_2 = 1 |",
        "gpu_offload": False,
    })

    report = engine_audit.audit_engine("cpu", host_features=["AVX-512", "AVX2", "AVX"])

    assert report["build"]["status"] == "mismatch"
    assert report["runtime"]["status"] == "matched"
    assert report["findings"] == [{
        "code": "cpu_simd_missing",
        "severity": "warning",
        "detail": "host reports AVX-512 but the installed llama.cpp build does not",
    }]


def test_vulkan_audit_rejects_cpu_only_llama_build(monkeypatch):
    _installed(monkeypatch)
    monkeypatch.setattr(engine_audit, "_probe", lambda _key, _backend: {
        "kind": "llama_cpp",
        "package_version": "0.3.31",
        "python_arch": "x86_64",
        "system_info": "CPU : AVX2 = 1 |",
        "gpu_offload": False,
    })

    report = engine_audit.audit_engine("vulkan", host_features=["AVX2"])

    assert report["build"]["status"] == "mismatch"
    assert report["runtime"]["status"] == "mismatch"
    assert {finding["code"] for finding in report["findings"]} == {
        "backend_missing", "accelerator_unavailable",
    }


def test_vulkan_audit_accepts_registered_backend_and_device(monkeypatch):
    _installed(monkeypatch)
    monkeypatch.setattr(engine_audit, "_probe", lambda _key, _backend: {
        "kind": "llama_cpp",
        "package_version": "0.3.31",
        "python_arch": "x86_64",
        "system_info": "Vulkan : NAME = AMD Radeon | CPU : AVX2 = 1 |",
        "gpu_offload": True,
    })

    report = engine_audit.audit_engine("vulkan", host_features=["AVX2"])

    assert report["build"]["status"] == "matched"
    assert report["runtime"]["status"] == "matched"
    assert report["device"] == "AMD Radeon"


def test_cuda_audit_requires_a_live_cuda_operation(monkeypatch):
    _installed(monkeypatch, schema="ara-engine-cuda:ara_engine_cuda:v1")
    monkeypatch.setattr(engine_audit, "_probe", lambda _key, _backend: {
        "kind": "torch_cuda",
        "package_version": "2.9.0+cu128",
        "cuda_build": "12.8",
        "arch_list": ["sm_75", "sm_80", "sm_89"],
        "available": True,
        "operation_ok": False,
        "device": "NVIDIA RTX 2080",
        "capability": "7.5",
        "error": "CUDA kernel launch failed",
    })

    report = engine_audit.audit_engine("cuda")

    assert report["build"]["status"] == "matched"
    assert report["runtime"]["status"] == "mismatch"
    assert report["findings"][-1]["code"] == "runtime_operation_failed"


def test_cuda_audit_rejects_build_without_host_architecture(monkeypatch):
    _installed(monkeypatch, schema="ara-engine-cuda:ara_engine_cuda:v1")
    monkeypatch.setattr(engine_audit, "_probe", lambda _key, _backend: {
        "kind": "torch_cuda",
        "package_version": "2.9.0+cu128",
        "cuda_build": "12.8",
        "arch_list": ["sm_80", "sm_86"],
        "available": True,
        "operation_ok": False,
        "device": "NVIDIA RTX 2080",
        "capability": "7.5",
        "error": "no kernel image is available",
    })

    report = engine_audit.audit_engine("cuda")

    assert report["build"]["status"] == "mismatch"
    assert "SM 7.5" in report["build"]["detail"]
    assert {finding["code"] for finding in report["findings"]} == {
        "cuda_arch_unsupported", "runtime_operation_failed",
    }


def test_cuda_audit_accepts_ptx_forward_compatibility(monkeypatch):
    _installed(monkeypatch, schema="ara-engine-cuda:ara_engine_cuda:v1")
    monkeypatch.setattr(engine_audit, "_probe", lambda _key, _backend: {
        "kind": "torch_cuda",
        "package_version": "2.9.0+cu128",
        "cuda_build": "12.8",
        "arch_list": ["sm_80", "compute_80"],
        "available": True,
        "operation_ok": False,
        "device": "NVIDIA RTX 4090",
        "capability": "8.9",
        "error": "driver initialization failed",
    })

    report = engine_audit.audit_engine("cuda")

    assert report["build"] == {
        "status": "matched",
        "detail": "CUDA build includes PTX compatible with host SM 8.9",
    }


def test_cuda_audit_is_unknown_without_host_capability(monkeypatch):
    _installed(monkeypatch, schema="ara-engine-cuda:ara_engine_cuda:v1")
    monkeypatch.setattr(engine_audit, "_probe", lambda _key, _backend: {
        "kind": "torch_cuda",
        "package_version": "2.9.0+cu128",
        "cuda_build": "12.8",
        "arch_list": ["sm_80"],
        "available": False,
        "operation_ok": False,
        "device": None,
        "capability": None,
        "error": None,
    })

    report = engine_audit.audit_engine("cuda")

    assert report["build"]["status"] == "unknown"
    assert report["build"]["detail"] == "host CUDA capability is unavailable"


def test_mlx_audit_reports_visible_metal_device(monkeypatch):
    _installed(monkeypatch, schema="ara-engine-mlx:ara_engine_mlx:v1")
    monkeypatch.setattr(engine_audit, "_probe", lambda _key, _backend: {
        "kind": "mlx",
        "package_version": "0.32.0",
        "metal_available": True,
        "gpu_count": 1,
        "operation_ok": True,
        "device": "Apple M4 Pro",
        "architecture": "applegpu_g16s",
    })

    report = engine_audit.audit_engine("mlx")

    assert report["build"]["status"] == "matched"
    assert report["runtime"]["status"] == "matched"
    assert report["device"] == "Apple M4 Pro"


def test_audit_absent_engine_does_not_probe(monkeypatch):
    monkeypatch.setattr(engine_audit.engine_env, "exists", lambda _backend: False)
    monkeypatch.setattr(engine_audit, "_probe", lambda *_args: (_ for _ in ()).throw(
        AssertionError("absent engine was probed")))

    report = engine_audit.audit_engine("cpu")

    assert report["installation"]["status"] == "absent"
    assert report["build"]["status"] == "not_checked"
    assert report["runtime"]["status"] == "not_checked"
    assert report["fingerprint"] is None


def test_matching_characterization_fingerprint_verifies_workload(monkeypatch):
    _installed(monkeypatch)
    monkeypatch.setattr(engine_audit, "_probe", lambda _key, _backend: {
        "kind": "llama_cpp", "package_version": "0.3.34", "python_arch": "arm64",
        "system_info": "CPU : NEON = 1 |", "gpu_offload": False,
    })
    first = engine_audit.audit_engine("cpu", host_features=["NEON"])
    rows = [{
        "safe_context": 8192,
        "evidence": {"engine": {"fingerprint": first["fingerprint"]}},
    }]

    report = engine_audit.audit_engine("cpu", host_features=["NEON"],
                                       characterization_rows=rows)

    assert report["workload"]["status"] == "verified"


def test_legacy_characterization_is_unknown_for_current_build(monkeypatch):
    _installed(monkeypatch)
    monkeypatch.setattr(engine_audit, "_probe", lambda _key, _backend: {
        "kind": "llama_cpp", "package_version": "0.3.34", "python_arch": "arm64",
        "system_info": "CPU : NEON = 1 |", "gpu_offload": False,
    })

    report = engine_audit.audit_engine(
        "cpu", host_features=["NEON"],
        characterization_rows=[{"safe_context": 8192, "evidence": None}],
    )

    assert report["workload"]["status"] == "unknown"
    assert "predates engine fingerprinting" in report["workload"]["detail"]


def test_changed_engine_fingerprint_marks_workload_evidence_stale(monkeypatch):
    _installed(monkeypatch)
    monkeypatch.setattr(engine_audit, "_probe", lambda _key, _backend: {
        "kind": "llama_cpp", "package_version": "0.3.34", "python_arch": "arm64",
        "system_info": "CPU : NEON = 1 |", "gpu_offload": False,
    })

    report = engine_audit.audit_engine(
        "cpu", host_features=["NEON"],
        characterization_rows=[{
            "safe_context": 8192,
            "evidence": {"engine": {"fingerprint": "engine:v1:sha256:old"}},
        }],
    )

    assert report["workload"]["status"] == "stale"


def test_characterization_evidence_keeps_only_stable_audit_facts(monkeypatch):
    _installed(monkeypatch)
    monkeypatch.setattr(engine_audit, "_probe", lambda _key, _backend: {
        "kind": "llama_cpp", "package_version": "0.3.34", "python_arch": "arm64",
        "system_info": "CPU : NEON = 1 |", "gpu_offload": False,
    })
    report = engine_audit.audit_engine("cpu", host_features=["NEON"])

    evidence = engine_audit.characterization_evidence(report)

    assert evidence == {
        "engine": {
            "fingerprint": report["fingerprint"],
            "package_version": "0.3.34",
            "build_status": "matched",
            "runtime_status": "matched",
        },
        "workload": {
            "status": "verified",
            "method": "characterize",
        },
    }


@pytest.mark.parametrize("key,marker", [
    ("mlx", "import mlx.core"),
    ("cuda", "import torch"),
    ("cpu", "from llama_cpp import llama_cpp"),
])
def test_probe_selects_engine_specific_isolated_script(monkeypatch, key, marker):
    seen = {}
    monkeypatch.setattr(
        engine_audit.engine_env, "run_python_json",
        lambda backend, code, *, timeout: seen.update(
            backend=backend, code=code, timeout=timeout) or {"ok": True})

    backend = engine_audit.engines.ENGINES[key]["backend"]
    assert engine_audit._probe(key, backend) == {"ok": True}
    assert seen["backend"] == backend and marker in seen["code"]
    assert seen["timeout"] == 30.0


def test_unknown_engine_is_rejected():
    with pytest.raises(ValueError, match="unknown engine"):
        engine_audit.audit_engine("bogus")


def test_stale_engine_stamp_is_reported_while_probe_still_runs(monkeypatch):
    _installed(monkeypatch, version="0.0.1")
    monkeypatch.setattr(engine_audit, "_probe", lambda *_args: {
        "kind": "llama_cpp", "package_version": "0.3.34", "python_arch": "arm64",
        "system_info": "CPU : NEON = 1 |", "gpu_offload": False,
    })

    report = engine_audit.audit_engine("cpu", host_features=["NEON"])

    assert report["installation"]["status"] == "stale"
    assert report["build"]["status"] == "matched"
    assert report["findings"][0]["code"] == "installation_stale"


def test_probe_failure_is_unknown_and_preserves_stale_workload_signal(monkeypatch):
    _installed(monkeypatch)
    monkeypatch.setattr(
        engine_audit, "_probe",
        lambda *_args: (_ for _ in ()).throw(engine_audit.engine_env.EngineEnvError("boom")))

    report = engine_audit.audit_engine(
        "cpu", characterization_rows=[{
            "safe_context": 8192,
            "evidence": {"engine": {"fingerprint": "engine:v1:sha256:old"}},
        }])

    assert report["build"]["status"] == "unknown"
    assert report["runtime"]["status"] == "unknown"
    assert report["workload"]["status"] == "stale"
    assert report["findings"][-1] == {
        "code": "probe_failed", "severity": "warning", "detail": "boom"}


def test_cpu_audit_is_unknown_when_host_simd_is_unknown(monkeypatch):
    _installed(monkeypatch)
    monkeypatch.setattr(engine_audit, "_probe", lambda *_args: {
        "kind": "llama_cpp", "package_version": "0.3.34", "python_arch": "mystery",
        "system_info": "CPU :", "gpu_offload": False,
    })

    assert engine_audit.audit_engine("cpu")["build"]["status"] == "unknown"


def test_cuda_audit_accepts_live_operation(monkeypatch):
    _installed(monkeypatch, schema="ara-engine-cuda:ara_engine_cuda:v1")
    monkeypatch.setattr(engine_audit, "_probe", lambda *_args: {
        "kind": "torch_cuda", "package_version": "2.9.0+cu128", "cuda_build": "12.8",
        "arch_list": ["sm_80"], "available": True, "operation_ok": True,
        "device": "RTX 2080", "capability": "7.5", "error": None,
    })

    report = engine_audit.audit_engine("cuda")

    assert report["build"]["status"] == "matched"
    assert report["runtime"]["status"] == "matched"


def test_cuda_audit_rejects_cpu_torch_and_no_visible_device(monkeypatch):
    _installed(monkeypatch, schema="ara-engine-cuda:ara_engine_cuda:v1")
    monkeypatch.setattr(engine_audit, "_probe", lambda *_args: {
        "kind": "torch_cuda", "package_version": "2.9.0", "cuda_build": None,
        "arch_list": [], "available": False, "operation_ok": False,
        "device": None, "capability": None, "error": None,
    })

    report = engine_audit.audit_engine("cuda")

    assert report["build"]["status"] == "mismatch"
    assert report["runtime"]["status"] == "mismatch"
    assert {finding["code"] for finding in report["findings"]} == {
        "backend_missing", "accelerator_unavailable"}


def test_mlx_audit_reports_failed_gpu_operation(monkeypatch):
    _installed(monkeypatch, schema="ara-engine-mlx:ara_engine_mlx:v1")
    monkeypatch.setattr(engine_audit, "_probe", lambda *_args: {
        "kind": "mlx", "package_version": "0.32.0", "metal_available": False,
        "gpu_count": 0, "operation_ok": False, "device": None,
        "architecture": None, "error": "Metal unavailable",
    })

    report = engine_audit.audit_engine("mlx")

    assert report["runtime"]["status"] == "mismatch"
    assert report["findings"][-1]["detail"] == "Metal unavailable"


def test_unknown_probe_payload_is_reported_without_guessing(monkeypatch):
    _installed(monkeypatch)
    monkeypatch.setattr(engine_audit, "_probe", lambda *_args: {
        "kind": "future_runtime", "package_version": "1", "volatile": "ignored"})

    report = engine_audit.audit_engine("cpu")

    assert report["build"]["status"] == "unknown"
    assert report["runtime"]["status"] == "unknown"
    assert report["capabilities"]["kind"] == "future_runtime"
    assert report["findings"][-1]["code"] == "probe_invalid"


def test_llama_device_parser_is_conservative():
    assert engine_audit._llama_device("CPU : AVX2 = 1 |", "Vulkan") is None
    assert engine_audit._llama_device("Vulkan : COOPMAT = 1 | CPU :", "Vulkan") is None


def test_audit_installed_filters_absent_envs_and_rows_by_engine(monkeypatch):
    catalog = {
        "cpu": {"backend": "cpu"},
        "mlx": {"backend": "apple"},
    }
    seen = []
    monkeypatch.setattr(engine_audit.engines, "ENGINES", catalog)
    monkeypatch.setattr(
        engine_audit.engine_env, "exists", lambda backend: backend == "cpu")

    def audit_engine(key, *, host_features, characterization_rows):
        seen.append((key, host_features, characterization_rows))
        return {"key": key}

    monkeypatch.setattr(engine_audit, "audit_engine", audit_engine)
    rows = [{"engine": "cpu", "safe_context": 1},
            {"engine": "mlx", "safe_context": 2}]

    assert engine_audit.audit_installed(
        host_features=["AVX2"], characterization_rows=rows) == [{"key": "cpu"}]
    assert seen == [("cpu", ["AVX2"], [{"engine": "cpu", "safe_context": 1}])]
