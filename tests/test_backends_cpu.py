# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""backends/cpu.py — the second real engine, proving the abstraction isn't Apple-shaped.

The CPU/llama.cpp adapter is intentionally a near-twin of backends/apple.py: it supplies only
its own specifics (the isolated ``cpu`` env, the built-in ``cpu_llama`` worker script, budget
params, schedule) into the SAME ``contracts.driver.characterize``. These tests drive it with a
mocked engine env — no llama.cpp, no model download — exactly as the apple tests do.
"""
from __future__ import annotations

import pytest

from ara import catalog
from ara.backends import cpu


@pytest.fixture(autouse=True)
def _no_catalog_network(monkeypatch):
    """Keep the suite offline: characterize now calls catalog.describe; stub it out."""
    monkeypatch.setattr(catalog, "describe", lambda m: None)


def test_worker_is_a_builtin_script_under_ara(tmp_path=None):
    # built into ARA (no separate repo), run by path — not an installed ``-m`` module
    assert cpu.WORKER.name == "cpu_llama.py"
    assert cpu.WORKER.parent.name == "workers"


class _FakeEngine:
    """Stand-in for engine_env.run_worker over the cpu env: preflight + linear measurements."""
    def __init__(self, est, intercept=5.0, slope_per_k=1.0, refuse_at=None):
        self.est, self.intercept, self.slope = est, intercept, slope_per_k
        self.refuse_at = refuse_at
        self.measured: list[int] = []

    def __call__(self, name, argv):
        assert name == "cpu"
        assert argv[0].endswith("cpu_llama.py")     # script by path, no ``-m``
        model = argv[1]
        ctx = int(argv[2])
        assert model == "org/model"
        if "--preflight" in argv:
            return dict(self.est)
        self.measured.append(ctx)
        if self.refuse_at is not None and ctx >= self.refuse_at:
            return {"context": ctx, "refused": True, "reason": "engine veto"}
        return {"context": ctx, "mem_gb": self.intercept + self.slope * (ctx / 1000)}


def _patch(monkeypatch, fake, margin=2.0, overhead=1.0):
    monkeypatch.setattr(cpu, "_budget_params", lambda: (margin, overhead))
    monkeypatch.setattr(cpu, "engine_env",
                        type("E", (), {"run_worker": staticmethod(fake)}))


def test_characterize_drives_shared_driver_over_cpu_env(monkeypatch):
    est = {"base_gb": 5.0, "slope_gb_per_k": 1.0, "budget_gb": 36.0,
           "max_context": 16000, "ref_baseline_gb": 0.0}
    fake = _FakeEngine(est)
    _patch(monkeypatch, fake)
    r = cpu.characterize("org/model")
    assert r["model"] == "org/model"
    # fitted memory ceiling ~31k exceeds the model's 16k window → capped, window-bound
    assert r["safe_context"] == 16_000
    assert r["binding"] == "context_window"
    assert all(c <= 16000 for c in fake.measured)
    assert r["points"][0] == {"context": 2000, "mem_gb": 7.0}


def test_characterize_none_when_preflight_errors(monkeypatch):
    fake = _FakeEngine({"error": "no GGUF for model"})
    _patch(monkeypatch, fake)
    assert cpu.characterize("org/model") == {
        "model": "org/model", "safe_context": None, "points": [],
        "error": "no GGUF for model"}


def test_budget_params_uses_stored_calibration(monkeypatch):
    monkeypatch.setattr(cpu, "db", type("D", (), {"connect": staticmethod(lambda: None)}))
    monkeypatch.setattr(cpu, "calibration",
                        type("P", (), {"get_calibration": staticmethod(
                            lambda con, eng: {"fixed_overhead_gb": 5.5})}), raising=False)
    assert cpu._budget_params() == (cpu.DEFAULT_MARGIN_GB, 5.5)


def test_budget_params_falls_back_to_default_overhead(monkeypatch):
    monkeypatch.setattr(cpu, "db", type("D", (), {"connect": staticmethod(lambda: None)}))
    monkeypatch.setattr(cpu, "calibration",
                        type("P", (), {"get_calibration": staticmethod(lambda con, eng: None)}),
                        raising=False)
    assert cpu._budget_params() == (cpu.DEFAULT_MARGIN_GB, cpu.DEFAULT_OVERHEAD_GB)


# --------------------------------------------------------------------------- #
# safe_limits / calibrate — the profile flow (exact RAM wall, like CUDA's VRAM)
# --------------------------------------------------------------------------- #
def test_safe_limits_exact_wall_needs_no_calibration(monkeypatch):
    facts = {"device": "x86_64", "total_gb": 24.0, "wall_gb": 24.0, "safe_budget_gb": 22.0,
             "margin_gb": 2.0, "headroom_gb": 12.0, "swap_free_gb": 1.0}
    seen = {}

    def worker(name, argv):
        seen["name"], seen["argv"] = name, argv
        return dict(facts)

    monkeypatch.setattr(cpu, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    m = cpu.safe_limits()
    assert seen["name"] == "cpu" and "--limits" in seen["argv"]
    assert m["wall_gb"] == 24.0 and m["safe_budget_gb"] == 22.0
    assert m["calibrated"] is True          # the wall is read exactly
    assert m["overhead_gb"] is None and m["calibrated_at"] is None


def test_calibration_model_cached_is_always_true():
    assert cpu.calibration_model_cached() is True   # worker downloads lazily


def test_download_calibration_model_is_noop():
    assert cpu.download_calibration_model() is None


def test_calibrate_attaches_characterization(monkeypatch):
    monkeypatch.setattr(cpu, "safe_limits", lambda: {"wall_gb": 24.0, "calibrated": True})
    monkeypatch.setattr(cpu, "characterize",
                        lambda model: {"model": model, "safe_context": 8192, "points": []})
    out = cpu.calibrate("org/m")
    assert out["calibrated"] is True
    assert out["characterization"]["safe_context"] == 8192


def test_calibrate_returns_uncalibrated_when_characterize_errors(monkeypatch):
    """characterize() returns an error → calibrate() must NOT claim calibrated=True (Rule #3)."""
    monkeypatch.setattr(cpu, "safe_limits",
                        lambda: {"wall_gb": 24.0, "calibrated": True, "overhead_gb": None})
    monkeypatch.setattr(cpu, "characterize",
                        lambda model: {"model": model, "safe_context": None,
                                       "points": [], "error": "no GGUF found for org/m"})
    out = cpu.calibrate("org/m")
    assert out["calibrated"] is False
    assert "calibration_error" in out
    assert "org/m" in out["calibration_error"]
    assert "no GGUF found for org/m" in out["calibration_error"]


# --------------------------------------------------------------------------- #
# generate — governed one-shot inference (Spec 2026-06-23-capability-pipeline, Slice 4)
# --------------------------------------------------------------------------- #
def test_generate_drives_worker_capped_at_context(monkeypatch):
    seen = {}

    def worker(name, argv, *, input=None):
        seen["name"], seen["argv"], seen["input"] = name, argv, input
        return {"context": 8192, "completion": "hello there"}

    monkeypatch.setattr(cpu, "_budget_params", lambda: (2.0, 1.0))
    monkeypatch.setattr(cpu, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    out = cpu.generate("org/model", "say hi", max_context=8192, max_tokens=64)
    assert out["completion"] == "hello there"
    assert seen["name"] == "cpu"
    assert seen["argv"][0].endswith("cpu_llama.py")
    assert seen["argv"][1] == "org/model" and seen["argv"][2] == "8192"   # governed ceiling
    assert "--generate" in seen["argv"]
    assert "--max-tokens" in seen["argv"] and "64" in seen["argv"]
    assert seen["input"] == "say hi"          # prompt over stdin, not argv
