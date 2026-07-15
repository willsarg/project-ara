# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Engine contract conformance — every backend adapter speaks the same interface and shapes.

ARA's whole design is a swappable backend behind a fixed contract. These tests assert that
*every* adapter (apple/cpu/cuda) exposes the same interface and returns the same result shapes,
each driven through its own (faked) engine seam. When a new backend lands, adding it to
``BACKENDS`` forces it to conform — or the suite fails.
"""
from __future__ import annotations

import pytest

from ara import acquire, catalog
from ara.backends import apple, cpu, cuda, cuda_gguf, vulkan


@pytest.fixture(autouse=True)
def _no_catalog_network(monkeypatch):
    """Keep the suite offline: characterize now calls catalog.describe; stub it out."""
    monkeypatch.setattr(catalog, "describe", lambda m: None)

# The interface every ramp-class backend must expose.
REQUIRED = ["characterize", "safe_limits", "calibrate", "calibration_model_cached",
            "download_calibration_model", "prepare_download", "download_prepared_model",
            "CALIBRATION_MODEL"]

# The canonical key set safe_limits() must return (engine facts + ARA's calibration overlay).
SAFE_LIMITS_KEYS = {"device", "total_gb", "wall_gb", "safe_budget_gb", "margin_gb",
                    "headroom_gb", "swap_free_gb", "overhead_gb", "calibrated", "calibrated_at"}

_FACTS = {"device": "Test Device", "total_gb": 48.0, "wall_gb": 40.0, "safe_budget_gb": 36.0,
          "margin_gb": 4.0, "headroom_gb": 28.0, "swap_free_gb": 2.0}


def _seam_apple(monkeypatch):
    def worker(name, argv):
        if "limits" in argv:
            return dict(_FACTS)
        if "calibrate" in argv:
            return {"measured_overhead_gb": 5.0, "default_overhead_gb": 6.0, "n_points": 4}
        if "--preflight" in argv:
            return {"base_gb": 5.0, "ref_baseline_gb": 0.0, "slope_gb_per_k": 1.0,
                    "budget_gb": 36.0, "max_context": 16000}
        return {"context": int(argv[3]), "mem_gb": 5.0 + int(argv[3]) / 1000}
    monkeypatch.setattr(apple, "engine_env", type("E", (), {"run_worker": staticmethod(worker)}))
    monkeypatch.setattr(apple, "_budget_params", lambda: (2.0, 1.0))


def _seam_cpu(monkeypatch):
    def worker(name, argv, *, stream=False):
        if "--limits" in argv:
            return dict(_FACTS)
        if "--preflight" in argv:
            return {"base_gb": 5.0, "ref_baseline_gb": 0.0, "slope_gb_per_k": 1.0,
                    "budget_gb": 36.0, "max_context": 16000}
        return {"context": int(argv[2]), "mem_gb": 5.0 + int(argv[2]) / 1000}
    monkeypatch.setattr(cpu, "engine_env", type("E", (), {"run_worker": staticmethod(worker)}))
    monkeypatch.setattr(cpu, "_budget_params", lambda: (2.0, 1.0))


def _seam_cuda(monkeypatch):
    def worker(name, argv):
        if "limits" in argv:
            return dict(_FACTS)
        if "calibrate" in argv:
            return {"device": "Test Device", "measured_overhead_gb": 0.9,
                    "default_overhead_gb": 0.6, "n_points": 1}
        if "--preflight" in argv:
            return {"base_gb": 5.0, "ref_baseline_gb": 0.0, "slope_gb_per_k": 1.0,
                    "budget_gb": 36.0, "max_context": 16000}
        return {"context": int(argv[3]), "mem_gb": 5.0 + int(argv[3]) / 1000}
    monkeypatch.setattr(cuda, "engine_env", type("E", (), {"run_worker": staticmethod(worker)}))
    monkeypatch.setattr(cuda, "_budget_params", lambda: (1.0, 0.6))


# (label, backend module, seam installer)
BACKENDS = [("apple", apple, _seam_apple), ("cpu", cpu, _seam_cpu), ("cuda", cuda, _seam_cuda)]
_IDS = [b[0] for b in BACKENDS]

# Every backend is now a worker-model adapter held to the STRICT result-shape contract.
SHIPPING = BACKENDS
_SHIP_IDS = _IDS


@pytest.mark.parametrize("mod", [apple, cpu, cuda, cuda_gguf, vulkan])
def test_backend_prepared_download_delegates_to_immutable_acquisition(mod, monkeypatch):
    plan = acquire.AcquisitionPlan("org/model", "org/model", "a" * 40, None, 1.0)
    seen = []
    monkeypatch.setattr(acquire, "prepare_download",
                        lambda model, *, gguf: seen.append(("prepare", model, gguf)) or plan)
    monkeypatch.setattr(acquire, "download_prepared",
                        lambda received, *, progress=False: seen.append(
                            ("download", received, progress)))

    assert mod.prepare_download("org/model") is plan
    assert mod.download_prepared_model(plan, progress=True) is None
    assert seen[0][0:2] == ("prepare", "org/model")
    assert seen[1] == ("download", plan, True)


@pytest.mark.parametrize("label, mod, _seam", BACKENDS, ids=_IDS)
def test_backend_exposes_the_interface(label, mod, _seam):
    for attr in REQUIRED:
        assert hasattr(mod, attr), f"backend {label} is missing {attr!r}"
    assert isinstance(mod.CALIBRATION_MODEL, str) and mod.CALIBRATION_MODEL


@pytest.mark.parametrize("label, mod, seam", BACKENDS, ids=_IDS)
def test_safe_limits_returns_the_canonical_shape(label, mod, seam, monkeypatch):
    seam(monkeypatch)
    m = mod.safe_limits()
    assert SAFE_LIMITS_KEYS <= set(m), f"{label} safe_limits missing {SAFE_LIMITS_KEYS - set(m)}"
    assert isinstance(m["calibrated"], bool)
    assert isinstance(m["safe_budget_gb"], (int, float))


@pytest.mark.parametrize("label, mod, seam", SHIPPING, ids=_SHIP_IDS)
def test_characterize_returns_the_canonical_shape(label, mod, seam, monkeypatch):
    seam(monkeypatch)
    r = mod.characterize("org/model")
    assert {"model", "safe_context", "binding", "points"} <= set(r), f"{label} characterize shape"
    assert r["model"] == "org/model"
    assert r["safe_context"] is None or isinstance(r["safe_context"], int)
    assert r["binding"] in ("memory", "context_window")
    # points must be the canonical dict shape — NOT just "a list" (the loose check let cuda's
    # raw (ctx, mem) tuples pass and diverge silently; this is what the conformance test exists for)
    assert isinstance(r["points"], list)
    for p in r["points"]:
        assert isinstance(p, dict) and {"context", "mem_gb"} <= set(p), f"{label} point shape {p}"


@pytest.mark.parametrize("label, mod, seam", BACKENDS, ids=_IDS)
def test_calibrate_carries_limits_and_a_measurement(label, mod, seam, monkeypatch):
    seam(monkeypatch)
    m = mod.calibrate(mod.CALIBRATION_MODEL)
    assert m["calibrated"] is True
    # each backend attaches what it measured (apple: 'calibration'; cpu/cuda: 'characterization')
    assert ("calibration" in m) or ("characterization" in m)
