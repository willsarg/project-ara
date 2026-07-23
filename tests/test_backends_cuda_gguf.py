# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""backends/cuda_gguf.py — CUDA-GGUF hybrid engine (partial GPU offload, two walls).

The cuda_gguf adapter is a near-twin of backends/vulkan.py: it drives the SAME
``contracts.driver.characterize`` and supplies only its own specifics (the isolated
``cuda_gguf`` env, the built-in ``cuda_gguf_llama`` worker, budget params, schedule). The two
walls (VRAM + RAM) are read exactly, so — like CPU/Vulkan, unlike Apple — there's nothing to
calibrate for the budget itself. These tests drive it with a mocked engine env — no llama.cpp,
no GPU, no model download.

Key differences from vulkan that the tests verify:
  * TWO margins: ``--vram-margin`` and ``--ram-margin`` (not ``--margin`` / ``--overhead``)
  * No ``flash_attn`` or ``kv_quant`` levers (not exposed by the cuda_gguf worker)
  * ``kv_dtype_bytes=2.0`` is always passed to the driver (fp16 fixed)
  * ``ENV_NAME`` = ``"cuda_gguf"``

Slug: 2026-06-29-cuda-gguf-hybrid-two-wall-engine
"""
from __future__ import annotations

import json as _json

import contextlib

import pytest

from ara import catalog
from ara.backends import cuda_gguf


@pytest.fixture(autouse=True)
def _no_catalog_network(monkeypatch):
    """Keep the suite offline: characterize calls catalog.describe; stub it out."""
    monkeypatch.setattr(catalog, "describe", lambda m: None)


# --------------------------------------------------------------------------- #
# Worker identity
# --------------------------------------------------------------------------- #
def test_worker_is_a_builtin_script_under_ara():
    # built into ARA (no separate repo), run by path — not an installed ``-m`` module
    assert cuda_gguf.WORKER.name == "cuda_gguf_llama.py"
    assert cuda_gguf.WORKER.parent.name == "workers"


def test_env_name_is_cuda_gguf():
    assert cuda_gguf.ENV_NAME == "cuda_gguf"


# --------------------------------------------------------------------------- #
# _worker_argv — both margins always present, no --margin / --overhead
# --------------------------------------------------------------------------- #
def test_worker_argv_contains_both_margins():
    argv = cuda_gguf._worker_argv("m", 100, 1.0, 2.0)
    assert "--vram-margin" in argv
    idx = argv.index("--vram-margin")
    assert argv[idx + 1] == "1.0"
    assert "--ram-margin" in argv
    idx2 = argv.index("--ram-margin")
    assert argv[idx2 + 1] == "2.0"


def test_worker_argv_has_no_old_margin_or_overhead_flags():
    argv = cuda_gguf._worker_argv("m", 100, 1.0, 2.0)
    assert "--margin" not in argv
    assert "--overhead" not in argv


def test_worker_argv_no_preflight_by_default():
    argv = cuda_gguf._worker_argv("m", 100, 1.0, 2.0)
    assert "--preflight" not in argv


def test_worker_argv_adds_preflight_when_requested():
    argv = cuda_gguf._worker_argv("m", 100, 1.0, 2.0, preflight=True)
    assert "--preflight" in argv


def test_worker_argv_starts_with_worker_model_ctx():
    argv = cuda_gguf._worker_argv("org/m", 4096, 1.0, 2.0)
    assert argv[0] == str(cuda_gguf.WORKER)
    assert argv[1] == "org/m"
    assert argv[2] == "4096"


# --------------------------------------------------------------------------- #
# Helpers for faking the engine env
# --------------------------------------------------------------------------- #
class _FakeEngine:
    """Stand-in for engine_env.run_worker over the cuda_gguf env: preflight + measurements."""
    def __init__(self, est, intercept=5.0, slope_per_k=1.0, refuse_at=None):
        self.est = est
        self.intercept = intercept
        self.slope = slope_per_k
        self.refuse_at = refuse_at
        self.measured: list[int] = []

    def __call__(self, name, argv, *, stream=False):
        assert name == "cuda_gguf"
        assert argv[0].endswith("cuda_gguf_llama.py")   # script by path, no ``-m``
        model = argv[1]
        ctx = int(argv[2])
        assert model == "org/model"
        # Both margin flags must be present in every call
        assert "--vram-margin" in argv
        assert "--ram-margin" in argv
        if "--preflight" in argv:
            return dict(self.est)
        self.measured.append(ctx)
        if self.refuse_at is not None and ctx >= self.refuse_at:
            return {"context": ctx, "refused": True, "reason": "engine veto"}
        return _measurement(
            ctx,
            self.intercept + self.slope * (ctx / 1000),
            ram_budget=self.est["ram_budget_gb"],
        )


def _patch(monkeypatch, fake, vram_margin=1.0, ram_margin=2.0):
    monkeypatch.setattr(cuda_gguf, "_budget_params", lambda: (vram_margin, ram_margin))
    monkeypatch.setattr(cuda_gguf, "engine_env",
                        type("E", (), {"run_worker": staticmethod(fake)}))


# --------------------------------------------------------------------------- #
# characterize(progress=...) — stream kwarg threading
# --------------------------------------------------------------------------- #
class _StreamCaptureFakeEngine:
    """Like _FakeEngine but records the stream= kwarg on every call."""
    def __init__(self, est, intercept=5.0, slope_per_k=1.0):
        self.est = est
        self.intercept = intercept
        self.slope = slope_per_k
        self.stream_kwargs: list[bool] = []

    def __call__(self, name, argv, *, stream=False):
        self.stream_kwargs.append(stream)
        if "--preflight" in argv:
            return dict(self.est)
        ctx = int(argv[2])
        return _measurement(
            ctx,
            self.intercept + self.slope * (ctx / 1000),
            ram_budget=self.est["ram_budget_gb"],
        )


_EST = {"base_gb": 5.0, "slope_gb_per_k": 1.0, "budget_gb": 20.0,
        "max_context": 4000, "ref_baseline_gb": 0.0,
        "n_layers": 32, "fit_layers": 16,
        "vram_budget_gb": 8.0, "ram_budget_gb": 20.0,
        "fit_dimension": "ram_absolute", "memory_unit": "GiB"}


def _measurement(context, absolute_ram, *, ram_budget=20.0):
    return {
        "context": context,
        "mem_gb": absolute_ram,
        "telemetry": {
            "schema": "cuda-gguf-two-wall-telemetry:v1",
            "fit_dimension": "ram_absolute",
            "unit": "GiB",
            "gpu_layers": 16,
            "vram": {"observed_gb": 4.0, "budget_gb": 8.0},
            "ram": {
                "observed_buffers_gb": absolute_ram - 1.0,
                "baseline_gb": 1.0,
                "observed_absolute_gb": absolute_ram,
                "budget_gb": ram_budget,
            },
            "provenance": {
                "source": "llama.cpp-load-log",
                "aggregation": "median",
                "repeat_count": 3,
                "vram_buffer_lines": 3,
                "ram_buffer_lines": 2,
            },
        },
    }


def test_characterize_progress_true_passes_stream_true_to_run_worker(monkeypatch):
    monkeypatch.setattr(catalog, "describe", lambda m: None)
    fake = _StreamCaptureFakeEngine(_EST)
    monkeypatch.setattr(cuda_gguf, "_budget_params", lambda: (1.0, 2.0))
    monkeypatch.setattr(cuda_gguf, "engine_env",
                        type("E", (), {"run_worker": staticmethod(fake)}))
    cuda_gguf.characterize("org/model", progress=True)
    assert len(fake.stream_kwargs) >= 2
    assert all(s is True for s in fake.stream_kwargs)


def test_characterize_progress_false_passes_stream_false_to_run_worker(monkeypatch):
    monkeypatch.setattr(catalog, "describe", lambda m: None)
    fake = _StreamCaptureFakeEngine(_EST)
    monkeypatch.setattr(cuda_gguf, "_budget_params", lambda: (1.0, 2.0))
    monkeypatch.setattr(cuda_gguf, "engine_env",
                        type("E", (), {"run_worker": staticmethod(fake)}))
    cuda_gguf.characterize("org/model", progress=False)
    assert len(fake.stream_kwargs) >= 2
    assert all(s is False for s in fake.stream_kwargs)


# --------------------------------------------------------------------------- #
# kv_dtype_bytes — always 2.0 (fp16 fixed; no kv_quant lever)
# --------------------------------------------------------------------------- #
def test_characterize_passes_kv_dtype_bytes_2_0_to_driver(monkeypatch):
    """cuda_gguf always passes kv_dtype_bytes=2.0 (fp16 fixed, no KV quant lever)."""
    seen = {}

    def fake_driver(model, *, preflight, measure, schedule, kv_dtype_bytes=2.0,
                    methodology_descriptor=None):
        seen["kv_dtype_bytes"] = kv_dtype_bytes
        return {"model": model, "safe_context": 1, "points": []}

    monkeypatch.setattr(cuda_gguf.driver, "characterize", fake_driver)
    monkeypatch.setattr(cuda_gguf, "_budget_params", lambda: (1.0, 2.0))
    cuda_gguf.characterize("org/model")
    assert seen["kv_dtype_bytes"] == 2.0


def test_characterization_methodology_names_dimension_bound_two_wall_protocol():
    descriptor = cuda_gguf.characterization_methodology()

    assert descriptor["worker_protocol"] == "ara-cuda-gguf-llama-measurement:v2"
    assert descriptor["telemetry_failure_policy"] == (
        "dimension-bound-two-wall-fail-closed:v2")


# --------------------------------------------------------------------------- #
# characterize — drives the shared driver over the cuda_gguf env
# --------------------------------------------------------------------------- #
def test_characterize_drives_shared_driver_over_cuda_gguf_env(monkeypatch):
    est = {"base_gb": 5.0, "slope_gb_per_k": 1.0, "budget_gb": 36.0,
           "max_context": 16000, "ref_baseline_gb": 0.0,
           "n_layers": 32, "fit_layers": 16,
           "vram_budget_gb": 8.0, "ram_budget_gb": 36.0,
           "fit_dimension": "ram_absolute", "memory_unit": "GiB"}
    fake = _FakeEngine(est)
    _patch(monkeypatch, fake)
    r = cuda_gguf.characterize("org/model")
    assert r["model"] == "org/model"
    assert r["safe_context"] == 16_000
    assert r["binding"] == "context_window"
    assert all(c <= 16000 for c in fake.measured)
    assert r["points"][0]["mem_gb"] == 7.0
    assert r["points"][0]["measurement_dimension"] == "ram_absolute"
    assert r["points"][0]["memory_unit"] == "GiB"
    assert r["points"][0]["telemetry"]["vram"]["observed_gb"] == 4.0


def test_characterize_none_when_preflight_errors(monkeypatch):
    fake = _FakeEngine({"error": "CUDA offload not active"})
    _patch(monkeypatch, fake)
    assert cuda_gguf.characterize("org/model") == {
        "model": "org/model", "safe_context": None,
        "direct_context": None, "fitted_context": None, "points": [],
        "error": "CUDA offload not active"}


# --------------------------------------------------------------------------- #
# _budget_params — reads stored calibration overrides, else defaults
# --------------------------------------------------------------------------- #
def test_budget_params_uses_stored_calibration_for_both_margins(monkeypatch):
    seen = []
    monkeypatch.setattr(cuda_gguf, "db",
                        type("D", (), {"connected": staticmethod(lambda: contextlib.nullcontext(None))}))
    monkeypatch.setattr(cuda_gguf, "calibration",
                        type("P", (), {"get_calibration": staticmethod(
                            lambda con, eng, **kwargs: seen.append((eng, kwargs))
                            or {"vram_margin_gb": 0.5, "ram_margin_gb": 3.0})}),
                        raising=False)
    assert cuda_gguf._budget_params(
        engine_fingerprint="engine:v1:sha256:cuda-gguf") == (0.5, 3.0)
    assert seen == [(cuda_gguf.CALIBRATION_ENGINE, {
        "engine_fingerprint": "engine:v1:sha256:cuda-gguf",
    })]


def test_budget_params_does_not_reuse_an_unscoped_engine_build(monkeypatch):
    monkeypatch.setattr(
        cuda_gguf.db, "connected",
        lambda: pytest.fail("unscoped lookup must not read calibration"))

    assert cuda_gguf._budget_params() == (
        cuda_gguf.DEFAULT_VRAM_MARGIN_GB, cuda_gguf.DEFAULT_RAM_MARGIN_GB)


def test_budget_params_falls_back_to_defaults_when_no_calibration(monkeypatch):
    monkeypatch.setattr(cuda_gguf, "db",
                        type("D", (), {"connected": staticmethod(lambda: contextlib.nullcontext(None))}))
    monkeypatch.setattr(cuda_gguf, "calibration",
                        type("P", (), {"get_calibration": staticmethod(
                            lambda con, eng, **kwargs: None)}),
                        raising=False)
    assert cuda_gguf._budget_params(
        engine_fingerprint="engine:v1:sha256:cuda-gguf") == (
        cuda_gguf.DEFAULT_VRAM_MARGIN_GB, cuda_gguf.DEFAULT_RAM_MARGIN_GB)


# --------------------------------------------------------------------------- #
# safe_limits / calibrate — the profile flow (exact two-wall reads, like CPU)
# --------------------------------------------------------------------------- #
def test_safe_limits_exact_walls_need_no_calibration(monkeypatch):
    facts = {"device": "GPU+CPU (CUDA hybrid)", "vram_total_gb": 8.0, "vram_used_gb": 1.0,
             "vram_budget_gb": 6.5, "ram_total_gb": 32.0, "ram_used_gb": 12.0,
             "ram_budget_gb": 28.0}
    seen = {}

    def worker(name, argv):
        seen["name"], seen["argv"] = name, argv
        return dict(facts)

    monkeypatch.setattr(cuda_gguf, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    m = cuda_gguf.safe_limits()
    assert seen["name"] == "cuda_gguf" and "--limits" in seen["argv"]
    assert "--vram-margin" in seen["argv"] and "--ram-margin" in seen["argv"]
    assert m["vram_budget_gb"] == 6.5 and m["ram_budget_gb"] == 28.0
    assert m["calibrated"] is True          # the walls are read exactly
    assert m["overhead_gb"] is None and m["calibrated_at"] is None


def test_calibration_model_cached_uses_artifact_authority(monkeypatch):
    monkeypatch.setattr(cuda_gguf.staleness, "artifact_identity", lambda model: None)
    assert cuda_gguf.calibration_model_cached("org/model") is False
    monkeypatch.setattr(cuda_gguf.staleness, "artifact_identity", lambda model: "artifact")
    assert cuda_gguf.calibration_model_cached("org/model") is True


def test_download_calibration_model_acquires_selected_gguf(monkeypatch):
    calls = []
    monkeypatch.setattr(cuda_gguf.acquire, "download_gguf",
                        lambda model, *, progress=False: calls.append((model, progress)))
    assert cuda_gguf.download_calibration_model("org/model", progress=True) is None
    assert calls == [("org/model", True)]


def test_download_calibration_model_passes_progress(monkeypatch):
    calls = []
    monkeypatch.setattr(cuda_gguf.acquire, "download_gguf",
                        lambda model, *, progress=False: calls.append(progress))
    assert cuda_gguf.download_calibration_model(progress=True) is None
    assert cuda_gguf.download_calibration_model(progress=False) is None
    assert calls == [True, False]


def test_calibrate_attaches_characterization(monkeypatch):
    monkeypatch.setattr(cuda_gguf, "safe_limits",
                        lambda: {"vram_budget_gb": 6.5, "calibrated": True})
    monkeypatch.setattr(cuda_gguf, "characterize",
                        lambda model: {"model": model, "safe_context": 8192, "points": []})
    out = cuda_gguf.calibrate("org/m")
    assert out["calibrated"] is True
    assert out["characterization"]["safe_context"] == 8192


def test_calibrate_returns_uncalibrated_when_characterize_errors(monkeypatch):
    """characterize() returns an error → calibrate() must NOT claim calibrated=True (Rule #3)."""
    monkeypatch.setattr(cuda_gguf, "safe_limits",
                        lambda: {"vram_budget_gb": 6.5, "calibrated": True, "overhead_gb": None})
    monkeypatch.setattr(cuda_gguf, "characterize",
                        lambda model: {"model": model, "safe_context": None,
                                       "points": [], "error": "CUDA offload not active"})
    out = cuda_gguf.calibrate("org/m")
    assert out["calibrated"] is False
    assert "calibration_error" in out
    assert "org/m" in out["calibration_error"]
    assert "CUDA offload not active" in out["calibration_error"]


# --------------------------------------------------------------------------- #
# generate — governed one-shot inference
# --------------------------------------------------------------------------- #
def test_generate_drives_worker_capped_at_context(monkeypatch):
    seen = {}

    def worker(name, argv, *, input=None):
        seen["name"], seen["argv"], seen["input"] = name, argv, input
        return {"context": 8192, "gpu_layers": 28, "completion": "hello there"}

    monkeypatch.setattr(cuda_gguf, "_budget_params", lambda: (1.0, 2.0))
    monkeypatch.setattr(cuda_gguf, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    out = cuda_gguf.generate("org/model", "say hi", max_context=8192, max_tokens=64)
    assert out["completion"] == "hello there"
    assert seen["name"] == "cuda_gguf"
    assert seen["argv"][0].endswith("cuda_gguf_llama.py")
    assert seen["argv"][1] == "org/model" and seen["argv"][2] == "8192"  # governed ceiling
    assert "--generate" in seen["argv"]
    assert "--max-tokens" in seen["argv"] and "64" in seen["argv"]
    assert "--vram-margin" in seen["argv"] and "--ram-margin" in seen["argv"]
    assert seen["input"] == "say hi"          # prompt over stdin, not argv


def test_generate_returns_worker_dict_verbatim(monkeypatch):
    result = {"context": 4096, "gpu_layers": 14, "completion": "done"}
    monkeypatch.setattr(cuda_gguf, "_budget_params", lambda: (1.0, 2.0))
    monkeypatch.setattr(cuda_gguf, "engine_env",
                        type("E", (), {"run_worker": staticmethod(lambda *a, **k: dict(result))}))
    assert cuda_gguf.generate("m", "hi", max_context=4096) == result


def test_generate_no_flash_attn_or_kv_quant_flags(monkeypatch):
    """cuda_gguf never passes --no-flash-attn or --kv-quant — the worker has no such flags."""
    seen = {}

    def worker(name, argv, *, input=None):
        seen["argv"] = argv
        return {"context": 4096, "gpu_layers": 14, "completion": "x"}

    monkeypatch.setattr(cuda_gguf, "_budget_params", lambda: (1.0, 2.0))
    monkeypatch.setattr(cuda_gguf, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    cuda_gguf.generate("m", "hi", max_context=4096)
    assert "--no-flash-attn" not in seen["argv"]
    assert "--kv-quant" not in seen["argv"]


# --------------------------------------------------------------------------- #
# benchmark — multi-prompt, load-once
# --------------------------------------------------------------------------- #
def test_benchmark_builds_argv_and_passes_prompts_as_json_stdin(monkeypatch):
    seen = {}

    def worker(name, argv, *, input=None):
        seen.update(name=name, argv=argv, input=input)
        return {"context": 4096, "gpu_layers": 14, "results": []}

    monkeypatch.setattr(cuda_gguf, "_budget_params", lambda: (1.0, 2.0))
    monkeypatch.setattr(cuda_gguf, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    out = cuda_gguf.benchmark("org/m", ["p1", "p2"], max_context=4096)
    assert seen["name"] == "cuda_gguf"
    assert seen["argv"][:4] == [str(cuda_gguf.WORKER), "org/m", "4096", "--benchmark"]
    assert "--vram-margin" in seen["argv"] and "--ram-margin" in seen["argv"]
    assert _json.loads(seen["input"]) == ["p1", "p2"]
    assert out == {"context": 4096, "gpu_layers": 14, "results": []}


def test_benchmark_threads_max_tokens(monkeypatch):
    seen = {}

    def worker(name, argv, *, input=None):
        seen["argv"] = argv
        return {"context": 4096, "gpu_layers": 14, "results": []}

    monkeypatch.setattr(cuda_gguf, "_budget_params", lambda: (1.0, 2.0))
    monkeypatch.setattr(cuda_gguf, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    cuda_gguf.benchmark("m", [], max_context=4096, max_tokens=128)
    assert seen["argv"][seen["argv"].index("--max-tokens") + 1] == "128"


def test_benchmark_returns_worker_dict_verbatim(monkeypatch):
    result = {"context": 4096, "gpu_layers": 14,
              "results": [{"prompt_index": 0, "completion": "ok"}]}

    monkeypatch.setattr(cuda_gguf, "_budget_params", lambda: (1.0, 2.0))
    monkeypatch.setattr(cuda_gguf, "engine_env",
                        type("E", (), {"run_worker": staticmethod(lambda *a, **k: dict(result))}))
    assert cuda_gguf.benchmark("m", ["hi"], max_context=4096) == result


def test_benchmark_no_flash_attn_or_kv_quant_flags(monkeypatch):
    """cuda_gguf benchmark never passes --no-flash-attn or --kv-quant."""
    seen = {}

    def worker(name, argv, *, input=None):
        seen["argv"] = argv
        return {"context": 4096, "gpu_layers": 14, "results": []}

    monkeypatch.setattr(cuda_gguf, "_budget_params", lambda: (1.0, 2.0))
    monkeypatch.setattr(cuda_gguf, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    cuda_gguf.benchmark("m", [], max_context=4096)
    assert "--no-flash-attn" not in seen["argv"]
    assert "--kv-quant" not in seen["argv"]
