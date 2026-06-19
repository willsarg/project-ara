"""backends/cuda.py — the wcx-suite seam, exercised against a fake engine."""
from __future__ import annotations

import sys
import types

import pytest

from ara.backends import cuda


class _FakeGPULimits:
    device = "NVIDIA GeForce RTX 2070"
    total_gb = 8.0
    wall_gb = 8.0
    free_gb = 6.9
    used_gb = 0.9

    def safe_threshold_gb(self, margin):
        return self.wall_gb - margin


@pytest.fixture
def fake_wcx(monkeypatch):
    """Inject a fake ``wcx_suite`` package; return knobs (.limits / .characterize_result)."""
    state = types.SimpleNamespace(
        limits=_FakeGPULimits(),
        margin=1.0,
        characterize_result=types.SimpleNamespace(
            safe_context=16000, points=[(512, 1.4), (2048, 2.0)]),
    )
    system = types.ModuleType("wcx_suite.system")
    system.read_limits = lambda: state.limits
    config = types.ModuleType("wcx_suite.config")
    config.margin_gb = lambda v=None: state.margin
    probe = types.ModuleType("wcx_suite.probe")
    probe.characterize = lambda model, budget_gb: state.characterize_result
    pkg = types.ModuleType("wcx_suite")
    pkg.system, pkg.config, pkg.probe = system, config, probe
    pkg.__path__ = []
    for name, mod in {"wcx_suite": pkg, "wcx_suite.system": system,
                      "wcx_suite.config": config, "wcx_suite.probe": probe}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return state


def test_safe_limits(fake_wcx):
    m = cuda.safe_limits()
    assert m["device"] == "NVIDIA GeForce RTX 2070"
    assert m["total_gb"] == 8.0 and m["wall_gb"] == 8.0
    assert m["safe_budget_gb"] == 7.0     # wall − 1 GB margin
    assert m["margin_gb"] == 1.0
    assert m["headroom_gb"] == 6.1        # safe 7 − used 0.9
    assert m["swap_free_gb"] is None
    assert m["overhead_gb"] is None
    assert m["calibrated"] is True and m["calibrated_at"] is None


def test_safe_limits_raises_without_gpu(fake_wcx):
    fake_wcx.limits = None
    with pytest.raises(RuntimeError):
        cuda.safe_limits()


def test_calibrate_attaches_characterization(fake_wcx):
    m = cuda.calibrate("smol")
    assert m["calibrated"] is True
    assert m["characterization"]["model"] == "smol"
    assert m["characterization"]["safe_context"] == 16000
    assert m["characterization"]["points"] == [(512, 1.4), (2048, 2.0)]


def test_calibrate_handles_failed_characterize(fake_wcx):
    fake_wcx.characterize_result = None
    m = cuda.calibrate("smol")
    assert m["characterization"]["safe_context"] is None
    assert m["characterization"]["points"] == []


def test_characterize_returns_ceiling(fake_wcx):
    r = cuda.characterize("smol")
    assert r["model"] == "smol"
    assert r["safe_context"] == 16000
    assert r["points"] == [(512, 1.4), (2048, 2.0)]


def test_characterize_none_when_failed(fake_wcx):
    fake_wcx.characterize_result = None
    r = cuda.characterize("smol")
    assert r["safe_context"] is None and r["points"] == []


def test_calibration_model_cached_true(monkeypatch):
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache",
                        lambda model, fn: "/path/to/config.json")
    assert cuda.calibration_model_cached("smol") is True


def test_calibration_model_cached_false(monkeypatch):
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda model, fn: None)
    assert cuda.calibration_model_cached("smol") is False


def test_calibration_model_cached_handles_error(monkeypatch):
    def boom(model, fn):
        raise RuntimeError("hf down")
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", boom)
    assert cuda.calibration_model_cached("smol") is False


def test_download_delegates_to_acquire(monkeypatch):
    called = {}
    monkeypatch.setattr("ara.acquire.download", lambda m: called.setdefault("model", m))
    cuda.download_calibration_model("smol")
    assert called["model"] == "smol"
