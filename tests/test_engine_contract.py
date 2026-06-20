"""Engine contract conformance — every backend adapter speaks the same interface and shapes.

ARA's whole design is a swappable backend behind a fixed contract. These tests assert that
*every* adapter (apple/cpu/cuda) exposes the same interface and returns the same result shapes,
each driven through its own (faked) engine seam. When a new backend lands, adding it to
``BACKENDS`` forces it to conform — or the suite fails.
"""
from __future__ import annotations

import sys
import types

import pytest

from ara.backends import apple, cpu, cuda

# The interface every ramp-class backend must expose.
REQUIRED = ["characterize", "safe_limits", "calibrate", "calibration_model_cached",
            "download_calibration_model", "CALIBRATION_MODEL"]

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
    def worker(name, argv):
        if "--limits" in argv:
            return dict(_FACTS)
        if "--preflight" in argv:
            return {"base_gb": 5.0, "ref_baseline_gb": 0.0, "slope_gb_per_k": 1.0,
                    "budget_gb": 36.0, "max_context": 16000}
        return {"context": int(argv[2]), "mem_gb": 5.0 + int(argv[2]) / 1000}
    monkeypatch.setattr(cpu, "engine_env", type("E", (), {"run_worker": staticmethod(worker)}))
    monkeypatch.setattr(cpu, "_budget_params", lambda: (2.0, 1.0))


def _seam_cuda(monkeypatch):
    limits = types.SimpleNamespace(device="Test Device", total_gb=48.0, wall_gb=40.0,
                                   used_gb=12.0, safe_threshold_gb=lambda margin: 40.0 - margin)
    system = types.ModuleType("wcx_suite.system"); system.read_limits = lambda: limits
    config = types.ModuleType("wcx_suite.config"); config.margin_gb = lambda v=None: 4.0
    probe = types.ModuleType("wcx_suite.probe")
    probe.characterize = lambda model, budget_gb: types.SimpleNamespace(
        safe_context=16000, points=[(512, 1.4)])
    pkg = types.ModuleType("wcx_suite"); pkg.system, pkg.config, pkg.probe = system, config, probe
    pkg.__path__ = []
    for name, mod in {"wcx_suite": pkg, "wcx_suite.system": system,
                      "wcx_suite.config": config, "wcx_suite.probe": probe}.items():
        monkeypatch.setitem(sys.modules, name, mod)


# (label, backend module, seam installer)
BACKENDS = [("apple", apple, _seam_apple), ("cpu", cpu, _seam_cpu), ("cuda", cuda, _seam_cuda)]
_IDS = [b[0] for b in BACKENDS]


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


@pytest.mark.parametrize("label, mod, seam", BACKENDS, ids=_IDS)
def test_characterize_returns_the_canonical_shape(label, mod, seam, monkeypatch):
    seam(monkeypatch)
    r = mod.characterize("org/model")
    assert {"model", "safe_context", "points"} <= set(r), f"{label} characterize shape"
    assert r["model"] == "org/model"
    assert r["safe_context"] is None or isinstance(r["safe_context"], int)
    assert isinstance(r["points"], list)


@pytest.mark.parametrize("label, mod, seam", BACKENDS, ids=_IDS)
def test_calibrate_carries_limits_and_a_measurement(label, mod, seam, monkeypatch):
    seam(monkeypatch)
    m = mod.calibrate(mod.CALIBRATION_MODEL)
    assert m["calibrated"] is True
    # each backend attaches what it measured (apple: 'calibration'; cpu/cuda: 'characterization')
    assert ("calibration" in m) or ("characterization" in m)
