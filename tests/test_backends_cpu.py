"""backends/cpu.py — the second real engine, proving the abstraction isn't Apple-shaped.

The CPU/llama.cpp adapter is intentionally a near-twin of backends/apple.py: it supplies only
its own specifics (the isolated ``cpu`` env, the built-in ``cpu_llama`` worker script, budget
params, schedule) into the SAME ``contracts.driver.characterize``. These tests drive it with a
mocked engine env — no llama.cpp, no model download — exactly as the apple tests do.
"""
from __future__ import annotations

from ara.backends import cpu


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
        "model": "org/model", "safe_context": None, "points": []}


def test_budget_params_uses_stored_calibration(monkeypatch):
    monkeypatch.setattr(cpu, "db", type("D", (), {"connect": staticmethod(lambda: None)}))
    monkeypatch.setattr(cpu, "profiles",
                        type("P", (), {"get_calibration": staticmethod(
                            lambda con, eng: {"fixed_overhead_gb": 5.5})}), raising=False)
    assert cpu._budget_params() == (cpu.DEFAULT_MARGIN_GB, 5.5)


def test_budget_params_falls_back_to_default_overhead(monkeypatch):
    monkeypatch.setattr(cpu, "db", type("D", (), {"connect": staticmethod(lambda: None)}))
    monkeypatch.setattr(cpu, "profiles",
                        type("P", (), {"get_calibration": staticmethod(lambda con, eng: None)}),
                        raising=False)
    assert cpu._budget_params() == (cpu.DEFAULT_MARGIN_GB, cpu.DEFAULT_OVERHEAD_GB)
