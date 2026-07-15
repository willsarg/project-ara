# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""backends/vulkan.py — GPU-offload GGUF inference via llama.cpp's Vulkan backend.

The Vulkan adapter is intentionally a near-twin of backends/cpu.py: it drives the SAME
``contracts.driver.characterize`` and supplies only its own specifics (the isolated ``vulkan``
env, the built-in ``vulkan_llama`` worker, budget params, schedule). The wall is exact system
RAM (the GPU's GTT pool is carved from it), so — like CPU, unlike Apple — there's nothing to
calibrate for the budget itself. These tests drive it with a mocked engine env — no llama.cpp,
no GPU, no model download.

Slug: 2026-06-25-vulkan-amd-engine-lane
"""
from __future__ import annotations

import contextlib

import pytest

from ara import catalog
from ara.backends import vulkan


@pytest.fixture(autouse=True)
def _no_catalog_network(monkeypatch):
    """Keep the suite offline: characterize calls catalog.describe; stub it out."""
    monkeypatch.setattr(catalog, "describe", lambda m: None)


def test_worker_is_a_builtin_script_under_ara():
    # built into ARA (no separate repo), run by path — not an installed ``-m`` module
    assert vulkan.WORKER.name == "vulkan_llama.py"
    assert vulkan.WORKER.parent.name == "workers"


class _FakeEngine:
    """Stand-in for engine_env.run_worker over the vulkan env: preflight + linear measurements."""
    def __init__(self, est, intercept=5.0, slope_per_k=1.0, refuse_at=None):
        self.est, self.intercept, self.slope = est, intercept, slope_per_k
        self.refuse_at = refuse_at
        self.measured: list[int] = []

    def __call__(self, name, argv, *, stream=False):
        assert name == "vulkan"
        assert argv[0].endswith("vulkan_llama.py")     # script by path, no ``-m``
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
    monkeypatch.setattr(vulkan, "_budget_params", lambda: (margin, overhead))
    monkeypatch.setattr(vulkan, "engine_env",
                        type("E", (), {"run_worker": staticmethod(fake)}))


# --------------------------------------------------------------------------- #
# characterize(progress=...) — stream kwarg threading (download-progress parity)
# --------------------------------------------------------------------------- #
class _StreamCaptureFakeEngine:
    """Like _FakeEngine but records the stream= kwarg on every call."""
    def __init__(self, est, intercept=5.0, slope_per_k=1.0):
        self.est, self.intercept, self.slope = est, intercept, slope_per_k
        self.stream_kwargs: list[bool] = []

    def __call__(self, name, argv, *, stream=False):
        self.stream_kwargs.append(stream)
        if "--preflight" in argv:
            return dict(self.est)
        ctx = int(argv[2])
        return {"context": ctx, "mem_gb": self.intercept + self.slope * (ctx / 1000)}


_EST = {"base_gb": 5.0, "slope_gb_per_k": 1.0, "budget_gb": 36.0,
        "max_context": 4000, "ref_baseline_gb": 0.0}


def test_characterize_progress_true_passes_stream_true_to_run_worker(monkeypatch):
    monkeypatch.setattr(catalog, "describe", lambda m: None)
    fake = _StreamCaptureFakeEngine(_EST)
    monkeypatch.setattr(vulkan, "_budget_params", lambda: (2.0, 1.0))
    monkeypatch.setattr(vulkan, "engine_env",
                        type("E", (), {"run_worker": staticmethod(fake)}))
    vulkan.characterize("org/model", progress=True)
    assert len(fake.stream_kwargs) >= 2
    assert all(s is True for s in fake.stream_kwargs)


def test_characterize_progress_false_passes_stream_false_to_run_worker(monkeypatch):
    monkeypatch.setattr(catalog, "describe", lambda m: None)
    fake = _StreamCaptureFakeEngine(_EST)
    monkeypatch.setattr(vulkan, "_budget_params", lambda: (2.0, 1.0))
    monkeypatch.setattr(vulkan, "engine_env",
                        type("E", (), {"run_worker": staticmethod(fake)}))
    vulkan.characterize("org/model", progress=False)
    assert len(fake.stream_kwargs) >= 2
    assert all(s is False for s in fake.stream_kwargs)


# --------------------------------------------------------------------------- #
# flash-attention plumbing (on by default; --no-flash-attn disables it)
# Slug: 2026-06-25-vulkan-flash-attention
# --------------------------------------------------------------------------- #
def test_worker_argv_omits_flag_by_default_and_adds_it_when_disabled():
    assert "--no-flash-attn" not in vulkan._worker_argv("m", 100, 2.0, 1.0)
    assert "--no-flash-attn" in vulkan._worker_argv("m", 100, 2.0, 1.0, flash_attn=False)


def test_characterize_threads_no_flash_attn_to_every_worker_call(monkeypatch):
    seen = []

    def fake(name, argv, *, stream=False):
        seen.append(argv)
        if "--preflight" in argv:
            return dict(_EST)
        return {"context": int(argv[2]), "mem_gb": 5.0 + int(argv[2]) / 1000}

    monkeypatch.setattr(catalog, "describe", lambda m: None)
    monkeypatch.setattr(vulkan, "_budget_params", lambda: (2.0, 1.0))
    monkeypatch.setattr(vulkan, "engine_env",
                        type("E", (), {"run_worker": staticmethod(fake)}))
    vulkan.characterize("org/model", flash_attn=False)
    assert seen and all("--no-flash-attn" in a for a in seen)


def test_characterize_default_has_flash_attn_on(monkeypatch):
    seen = []

    def fake(name, argv, *, stream=False):
        seen.append(argv)
        if "--preflight" in argv:
            return dict(_EST)
        return {"context": int(argv[2]), "mem_gb": 5.0 + int(argv[2]) / 1000}

    monkeypatch.setattr(catalog, "describe", lambda m: None)
    monkeypatch.setattr(vulkan, "_budget_params", lambda: (2.0, 1.0))
    monkeypatch.setattr(vulkan, "engine_env",
                        type("E", (), {"run_worker": staticmethod(fake)}))
    vulkan.characterize("org/model")
    assert seen and not any("--no-flash-attn" in a for a in seen)


def test_generate_adds_no_flash_attn_when_disabled(monkeypatch):
    seen = {}

    def worker(name, argv, *, input=None):
        seen["argv"] = argv
        return {"context": 8192, "completion": "x"}

    monkeypatch.setattr(vulkan, "_budget_params", lambda: (2.0, 1.0))
    monkeypatch.setattr(vulkan, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    vulkan.generate("org/model", "hi", max_context=8192, flash_attn=False)
    assert "--no-flash-attn" in seen["argv"]
    # default keeps FA on (no flag)
    vulkan.generate("org/model", "hi", max_context=8192)
    assert "--no-flash-attn" not in seen["argv"]


# --------------------------------------------------------------------------- #
# KV-cache quantization plumbing (default f16 = no flag; symmetric K=V)
# Slug: 2026-06-25-vulkan-kv-cache-quant
# --------------------------------------------------------------------------- #
def test_worker_argv_carries_kv_quant_only_when_not_f16():
    assert "--kv-quant" not in vulkan._worker_argv("m", 100, 2.0, 1.0)            # f16 default
    argv = vulkan._worker_argv("m", 100, 2.0, 1.0, kv_quant="q8_0")
    assert argv[-2:] == ["--kv-quant", "q8_0"]


def test_characterize_threads_kv_quant_to_every_worker_call(monkeypatch):
    seen = []

    def fake(name, argv, *, stream=False):
        seen.append(argv)
        if "--preflight" in argv:
            return dict(_EST)
        return {"context": int(argv[2]), "mem_gb": 5.0 + int(argv[2]) / 1000}

    monkeypatch.setattr(catalog, "describe", lambda m: None)
    monkeypatch.setattr(vulkan, "_budget_params", lambda: (2.0, 1.0))
    monkeypatch.setattr(vulkan, "engine_env",
                        type("E", (), {"run_worker": staticmethod(fake)}))
    vulkan.characterize("org/model", kv_quant="q8_0")
    assert seen and all(a[-2:] == ["--kv-quant", "q8_0"] for a in seen)


def test_generate_adds_kv_quant_when_not_f16(monkeypatch):
    seen = {}

    def worker(name, argv, *, input=None):
        seen["argv"] = argv
        return {"context": 8192, "completion": "x"}

    monkeypatch.setattr(vulkan, "_budget_params", lambda: (2.0, 1.0))
    monkeypatch.setattr(vulkan, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    vulkan.generate("org/model", "hi", max_context=8192, kv_quant="q4_0")
    assert seen["argv"][-2:] == ["--kv-quant", "q4_0"]
    vulkan.generate("org/model", "hi", max_context=8192)   # f16 default → no flag
    assert "--kv-quant" not in seen["argv"]


def test_characterize_passes_kv_dtype_bytes_from_quant_to_driver(monkeypatch):
    # The decode-ceiling estimate is engine-agnostic in the driver; vulkan maps kv_quant → the
    # per-element byte count so the estimate reflects the KV cache type. Slug: 2026-06-25-vulkan-kv-cache-quant
    seen = {}

    def fake_driver(model, *, preflight, measure, schedule, kv_dtype_bytes=2.0):
        seen["kv_dtype_bytes"] = kv_dtype_bytes
        return {"model": model, "safe_context": 1, "points": []}

    monkeypatch.setattr(vulkan.driver, "characterize", fake_driver)
    monkeypatch.setattr(vulkan, "_budget_params", lambda: (2.0, 1.0))
    vulkan.characterize("org/model", kv_quant="q8_0")
    assert seen["kv_dtype_bytes"] == vulkan._KV_BYTES["q8_0"]
    vulkan.characterize("org/model")                       # f16 default
    assert seen["kv_dtype_bytes"] == vulkan._KV_BYTES["f16"]


def test_characterize_drives_shared_driver_over_vulkan_env(monkeypatch):
    est = {"base_gb": 5.0, "slope_gb_per_k": 1.0, "budget_gb": 36.0,
           "max_context": 16000, "ref_baseline_gb": 0.0}
    fake = _FakeEngine(est)
    _patch(monkeypatch, fake)
    r = vulkan.characterize("org/model")
    assert r["model"] == "org/model"
    assert r["safe_context"] == 16_000
    assert r["binding"] == "context_window"
    assert all(c <= 16000 for c in fake.measured)
    assert r["points"][0] == {"context": 2000, "mem_gb": 7.0}


def test_characterize_none_when_preflight_errors(monkeypatch):
    fake = _FakeEngine({"error": "Vulkan offload not active"})
    _patch(monkeypatch, fake)
    assert vulkan.characterize("org/model") == {
        "model": "org/model", "safe_context": None, "points": [],
        "error": "Vulkan offload not active"}


def test_budget_params_uses_stored_calibration(monkeypatch):
    monkeypatch.setattr(vulkan, "db", type("D", (), {"connected": staticmethod(lambda: contextlib.nullcontext(None))}))
    monkeypatch.setattr(vulkan, "calibration",
                        type("P", (), {"get_calibration": staticmethod(
                            lambda con, eng: {"fixed_overhead_gb": 5.5})}), raising=False)
    assert vulkan._budget_params() == (vulkan.DEFAULT_MARGIN_GB, 5.5)


def test_budget_params_falls_back_to_default_overhead(monkeypatch):
    monkeypatch.setattr(vulkan, "db", type("D", (), {"connected": staticmethod(lambda: contextlib.nullcontext(None))}))
    monkeypatch.setattr(vulkan, "calibration",
                        type("P", (), {"get_calibration": staticmethod(lambda con, eng: None)}),
                        raising=False)
    assert vulkan._budget_params() == (vulkan.DEFAULT_MARGIN_GB, vulkan.DEFAULT_OVERHEAD_GB)


# --------------------------------------------------------------------------- #
# safe_limits / calibrate — the profile flow (exact RAM wall, like CPU's)
# --------------------------------------------------------------------------- #
def test_safe_limits_exact_wall_needs_no_calibration(monkeypatch):
    facts = {"device": "x86_64", "total_gb": 11.0, "wall_gb": 11.0, "safe_budget_gb": 9.9,
             "margin_gb": 1.1, "headroom_gb": 8.0, "swap_free_gb": 1.0}
    seen = {}

    def worker(name, argv):
        seen["name"], seen["argv"] = name, argv
        return dict(facts)

    monkeypatch.setattr(vulkan, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    m = vulkan.safe_limits()
    assert seen["name"] == "vulkan" and "--limits" in seen["argv"]
    assert m["wall_gb"] == 11.0 and m["safe_budget_gb"] == 9.9
    assert m["calibrated"] is True          # the wall is read exactly
    assert m["overhead_gb"] is None and m["calibrated_at"] is None


def test_calibration_model_cached_uses_artifact_authority(monkeypatch):
    monkeypatch.setattr(vulkan.staleness, "artifact_identity", lambda model: None)
    assert vulkan.calibration_model_cached("org/model") is False
    monkeypatch.setattr(vulkan.staleness, "artifact_identity", lambda model: "artifact")
    assert vulkan.calibration_model_cached("org/model") is True


def test_download_calibration_model_acquires_selected_gguf(monkeypatch):
    calls = []
    monkeypatch.setattr(vulkan.acquire, "download_gguf",
                        lambda model, *, progress=False: calls.append((model, progress)))
    assert vulkan.download_calibration_model("org/model", progress=True) is None
    assert calls == [("org/model", True)]


def test_download_calibration_model_passes_progress(monkeypatch):
    calls = []
    monkeypatch.setattr(vulkan.acquire, "download_gguf",
                        lambda model, *, progress=False: calls.append(progress))
    assert vulkan.download_calibration_model(progress=True) is None
    assert vulkan.download_calibration_model(progress=False) is None
    assert calls == [True, False]


def test_calibrate_attaches_characterization(monkeypatch):
    monkeypatch.setattr(vulkan, "safe_limits", lambda: {"wall_gb": 11.0, "calibrated": True})
    monkeypatch.setattr(vulkan, "characterize",
                        lambda model: {"model": model, "safe_context": 8192, "points": []})
    out = vulkan.calibrate("org/m")
    assert out["calibrated"] is True
    assert out["characterization"]["safe_context"] == 8192


def test_calibrate_returns_uncalibrated_when_characterize_errors(monkeypatch):
    """characterize() returns an error → calibrate() must NOT claim calibrated=True (Rule #3)."""
    monkeypatch.setattr(vulkan, "safe_limits",
                        lambda: {"wall_gb": 11.0, "calibrated": True, "overhead_gb": None})
    monkeypatch.setattr(vulkan, "characterize",
                        lambda model: {"model": model, "safe_context": None,
                                       "points": [], "error": "Vulkan offload not active"})
    out = vulkan.calibrate("org/m")
    assert out["calibrated"] is False
    assert "calibration_error" in out
    assert "org/m" in out["calibration_error"]
    assert "Vulkan offload not active" in out["calibration_error"]


# --------------------------------------------------------------------------- #
# generate — governed one-shot inference
# --------------------------------------------------------------------------- #
def test_generate_drives_worker_capped_at_context(monkeypatch):
    seen = {}

    def worker(name, argv, *, input=None):
        seen["name"], seen["argv"], seen["input"] = name, argv, input
        return {"context": 8192, "completion": "hello there"}

    monkeypatch.setattr(vulkan, "_budget_params", lambda: (2.0, 1.0))
    monkeypatch.setattr(vulkan, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    out = vulkan.generate("org/model", "say hi", max_context=8192, max_tokens=64)
    assert out["completion"] == "hello there"
    assert seen["name"] == "vulkan"
    assert seen["argv"][0].endswith("vulkan_llama.py")
    assert seen["argv"][1] == "org/model" and seen["argv"][2] == "8192"   # governed ceiling
    assert "--generate" in seen["argv"]
    assert "--max-tokens" in seen["argv"] and "64" in seen["argv"]
    assert seen["input"] == "say hi"          # prompt over stdin, not argv


# ── benchmark (multi-prompt, load-once) ───────────────────────────────────────
import json as _json  # noqa: E402


def test_benchmark_builds_argv_and_passes_prompts_as_json_stdin(monkeypatch):
    seen = {}

    def worker(name, argv, *, input=None):
        seen.update(name=name, argv=argv, input=input)
        return {"context": 4096, "results": []}

    monkeypatch.setattr(vulkan, "_budget_params", lambda: (2.0, 1.0))
    monkeypatch.setattr(vulkan, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    out = vulkan.benchmark("org/m", ["p1", "p2"], max_context=4096)
    assert seen["name"] == "vulkan"
    assert seen["argv"][:4] == [str(vulkan.WORKER), "org/m", "4096", "--benchmark"]
    assert _json.loads(seen["input"]) == ["p1", "p2"]
    assert out == {"context": 4096, "results": []}


def test_benchmark_threads_no_flash_attn_and_kv_quant(monkeypatch):
    seen = {}

    def worker(name, argv, *, input=None):
        seen["argv"] = argv
        return {"context": 4096, "results": []}

    monkeypatch.setattr(vulkan, "_budget_params", lambda: (2.0, 1.0))
    monkeypatch.setattr(vulkan, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    vulkan.benchmark("m", [], max_context=4096, flash_attn=False, kv_quant="q8_0")
    assert "--no-flash-attn" in seen["argv"]
    assert "--kv-quant" in seen["argv"] and "q8_0" in seen["argv"]


def test_benchmark_omits_kv_quant_for_f16(monkeypatch):
    seen = {}

    def worker(name, argv, *, input=None):
        seen["argv"] = argv
        return {"context": 4096, "results": []}

    monkeypatch.setattr(vulkan, "_budget_params", lambda: (2.0, 1.0))
    monkeypatch.setattr(vulkan, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    vulkan.benchmark("m", [], max_context=4096)  # f16 default
    assert "--kv-quant" not in seen["argv"]
    assert "--no-flash-attn" not in seen["argv"]


def test_benchmark_threads_max_tokens(monkeypatch):
    seen = {}

    def worker(name, argv, *, input=None):
        seen["argv"] = argv
        return {"context": 4096, "results": []}

    monkeypatch.setattr(vulkan, "_budget_params", lambda: (2.0, 1.0))
    monkeypatch.setattr(vulkan, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    vulkan.benchmark("m", [], max_context=4096, max_tokens=128)
    assert seen["argv"][seen["argv"].index("--max-tokens") + 1] == "128"


def test_benchmark_returns_worker_dict_verbatim(monkeypatch):
    result = {"context": 4096, "results": [{"prompt_index": 0, "completion": "ok"}]}

    monkeypatch.setattr(vulkan, "_budget_params", lambda: (2.0, 1.0))
    monkeypatch.setattr(vulkan, "engine_env",
                        type("E", (), {"run_worker": staticmethod(lambda *a, **k: dict(result))}))
    assert vulkan.benchmark("m", ["hi"], max_context=4096) == result
