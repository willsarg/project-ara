# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""backends/cuda.py — the native CUDA engine seam (stateless; ARA owns persistence).

The CUDA twin of test_backends_apple.py: cuda drives device + measure_one workers
out-of-process through engine_env, never importing the engine in ARA's interpreter.
"""
from __future__ import annotations

import contextlib

import pytest

from ara import acquire, catalog
from ara.backends import cuda


@pytest.fixture(autouse=True)
def _no_catalog_network(monkeypatch):
    """Keep the suite offline: characterize now calls catalog.describe; stub it out."""
    monkeypatch.setattr(catalog, "describe", lambda m: None)


def _fake_worker(monkeypatch, fn):
    monkeypatch.setattr(cuda, "engine_env",
                        type("E", (), {"run_worker": staticmethod(fn)}))


# Engine facts the CUDA `device limits` worker returns (ARA overlays its own calibration fields).
_LIMITS_FACTS = {
    "device": "NVIDIA GeForce RTX 2070", "total_gb": 8.0, "wall_gb": 8.0,
    "safe_budget_gb": 7.0, "margin_gb": 1.0, "headroom_gb": 5.0, "swap_free_gb": None,
}


def test_safe_limits_drives_device_worker_and_overlays(monkeypatch):
    calls = []

    def worker(name, argv):
        calls.append((name, argv))
        return dict(_LIMITS_FACTS)

    _fake_worker(monkeypatch, worker)
    m = cuda.safe_limits()
    assert calls == [("cuda", ["-m", "ara_engine_cuda.device", "limits"])]
    assert m["device"] == "NVIDIA GeForce RTX 2070"
    assert m["total_gb"] == 8.0 and m["wall_gb"] == 8.0
    assert m["safe_budget_gb"] == 7.0 and m["margin_gb"] == 1.0
    assert m["headroom_gb"] == 5.0 and m["swap_free_gb"] is None
    # no stored calibration in the engine — ARA overlays it from its own store
    assert m["calibrated"] is False
    assert m["overhead_gb"] is None
    assert m["calibrated_at"] is None


def test_safe_limits_raises_when_no_gpu(monkeypatch):
    _fake_worker(monkeypatch, lambda name, argv: {"error": "no NVIDIA GPU visible to nvidia-smi"})
    try:
        cuda.safe_limits()
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "no NVIDIA GPU" in str(e)


def test_calibration_model_cached_true(monkeypatch):
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache",
                        lambda m, fn: "/path/to/config.json")
    assert cuda.calibration_model_cached("any/model") is True


def test_calibration_model_cached_false_when_absent(monkeypatch):
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda m, fn: None)
    assert cuda.calibration_model_cached("any/model") is False


def test_calibration_model_cached_false_on_error(monkeypatch):
    def boom(m, fn):
        raise RuntimeError("hf down")
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", boom)
    assert cuda.calibration_model_cached("any/model") is False


def test_download_calibration_model_delegates_to_acquire(monkeypatch):
    calls = []
    monkeypatch.setattr(acquire, "download", lambda repo_id, *, progress=False: calls.append(repo_id))
    cuda.download_calibration_model("org/calib-model")
    assert calls == ["org/calib-model"]


def test_download_calibration_model_passes_progress_to_acquire(monkeypatch):
    """download_calibration_model(progress=True) passes progress=True to acquire.download.

    Slug: 2026-06-24-download-progress
    """
    captured = {}
    monkeypatch.setattr(acquire, "download",
                        lambda repo_id, *, progress=False: captured.update(progress=progress))
    cuda.download_calibration_model("org/m", progress=True)
    assert captured["progress"] is True


def test_download_calibration_model_default_progress_false(monkeypatch):
    """download_calibration_model() default passes progress=False to acquire.download.

    Slug: 2026-06-24-download-progress
    """
    captured = {}
    monkeypatch.setattr(acquire, "download",
                        lambda repo_id, *, progress=False: captured.update(progress=progress))
    cuda.download_calibration_model("org/m")
    assert captured["progress"] is False


def test_characterize_accepts_progress_and_does_not_stream(monkeypatch):
    """cuda.characterize(progress=True) accepts progress for symmetry but does NOT pass stream
    to run_worker — bars already ran in-process during the pre-fetch step.

    Slug: 2026-06-24-download-progress
    """
    stream_kwargs = []

    def worker(name, argv, **kwargs):
        stream_kwargs.append(kwargs.get("stream", False))
        ctx = int(argv[3])
        if "--preflight" in argv:
            return {"base_gb": 1.0, "slope_gb_per_k": 0.2, "budget_gb": 7.0,
                    "max_context": 4000, "ref_baseline_gb": 0.0}
        return {"context": ctx, "mem_gb": 1.0 + 0.2 * (ctx / 1000)}

    monkeypatch.setattr(cuda, "_budget_params", lambda: (1.0, 0.6))
    monkeypatch.setattr(cuda, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    cuda.characterize("org/m", progress=True)
    # stream must never be True for cuda — bars ran in-process during pre-fetch
    assert all(s is False for s in stream_kwargs)


def _calibrate_worker(monkeypatch, calibration, calls=None):
    def worker(name, argv):
        if calls is not None:
            calls.append(argv)
        return dict(calibration) if argv[2] == "calibrate" else dict(_LIMITS_FACTS)
    _fake_worker(monkeypatch, worker)


def test_calibrate_surfaces_effective_overhead(monkeypatch):
    calls = []
    _calibrate_worker(monkeypatch, {
        "device": "RTX 2070", "measured_overhead_gb": 0.9,
        "default_overhead_gb": 0.6, "n_points": 1,
    }, calls)
    m = cuda.calibrate("org/calib-model")
    assert m["device"] == "NVIDIA GeForce RTX 2070"     # carries fresh limits …
    assert m["overhead_gb"] == 0.9                       # effective = max(default 0.6, measured 0.9)
    assert m["calibrated"] is True
    assert m["calibration"]["n_points"] == 1             # … plus what it measured
    assert ["-m", "ara_engine_cuda.device", "calibrate", "org/calib-model"] in calls


def test_calibrate_overhead_falls_back_to_default(monkeypatch):
    _calibrate_worker(monkeypatch, {"default_overhead_gb": 0.6})   # no measurement key
    assert cuda.calibrate("org/calib-model")["overhead_gb"] == 0.6


def test_calibrate_overhead_none_when_nothing_measured(monkeypatch):
    _calibrate_worker(monkeypatch, {"n_points": 0})               # no overhead keys at all
    assert cuda.calibrate("org/calib-model")["overhead_gb"] is None


def test_calibrate_returns_uncalibrated_on_worker_error(monkeypatch):
    """Worker returns an error dict → calibrate() must NOT claim calibrated=True (Rule #3)."""
    def worker(name, argv):
        if argv[2] == "calibrate":
            return {"error": "CUDA context initialisation failed"}
        return dict(_LIMITS_FACTS)
    _fake_worker(monkeypatch, worker)
    m = cuda.calibrate("org/calib-model")
    assert m["calibrated"] is False
    assert m["overhead_gb"] is None
    assert "calibration_error" in m
    assert "org/calib-model" in m["calibration_error"]
    assert "CUDA context initialisation failed" in m["calibration_error"]


def test_calibrate_returns_uncalibrated_on_worker_exception(monkeypatch):
    """Worker raises → calibrate() must NOT crash; must return uncalibrated + error (Rule #3)."""
    def worker(name, argv):
        if argv[2] == "calibrate":
            raise RuntimeError("CUDA engine env not installed")
        return dict(_LIMITS_FACTS)
    _fake_worker(monkeypatch, worker)
    m = cuda.calibrate("org/calib-model")
    assert m["calibrated"] is False
    assert m["overhead_gb"] is None
    assert "calibration_error" in m
    assert "org/calib-model" in m["calibration_error"]
    assert "CUDA engine env not installed" in m["calibration_error"]


class _FakeEngine:
    """Stand-in for engine_env.run_worker: answers preflight + per-ctx measurements, driven by a
    canned estimate and a linear memory model. Records every spawn."""
    def __init__(self, est, intercept=1.0, slope_per_k=0.2, refuse_at=None):
        self.est, self.intercept, self.slope = est, intercept, slope_per_k
        self.refuse_at = refuse_at
        self.measured: list[int] = []

    def __call__(self, name, argv):
        assert name == "cuda"
        ctx = int(argv[3])
        if "--preflight" in argv:
            return dict(self.est)
        self.measured.append(ctx)
        if self.refuse_at is not None and ctx >= self.refuse_at:
            return {"context": ctx, "refused": True, "reason": "engine veto"}
        return {"context": ctx, "mem_gb": self.intercept + self.slope * (ctx / 1000)}


def _patch_budget(monkeypatch, margin=1.0, overhead=0.6):
    monkeypatch.setattr(cuda, "_budget_params", lambda: (margin, overhead))


def test_characterize_drives_ramp_over_engine_env(monkeypatch):
    _patch_budget(monkeypatch)
    est = {"base_gb": 1.6, "slope_gb_per_k": 0.2, "budget_gb": 7.0,
           "max_context": 16000, "ref_baseline_gb": 0.0}
    fake = _FakeEngine(est, intercept=1.0, slope_per_k=0.2)
    _fake_worker(monkeypatch, fake)
    r = cuda.characterize("org/model")
    assert r["model"] == "org/model"
    # fitted ceiling ~30k exceeds the 16k window → capped, window-bound
    assert r["safe_context"] == 16_000
    assert r["binding"] == "context_window"
    assert all(c <= 16000 for c in fake.measured)
    assert r["points"][0] == {"context": 2000, "mem_gb": 1.4}


def test_characterize_subtracts_live_ref_baseline_from_ceiling(monkeypatch):
    _patch_budget(monkeypatch)
    # delta fit: model base 1, slope 0.2; live VRAM baseline 2 GB → ceiling (7-2-1)/0.2 = 20k
    est = {"base_gb": 3.0, "slope_gb_per_k": 0.2, "budget_gb": 7.0,
           "max_context": None, "ref_baseline_gb": 2.0}
    fake = _FakeEngine(est, intercept=1.0, slope_per_k=0.2)
    _fake_worker(monkeypatch, fake)
    r = cuda.characterize("org/model")
    assert r["safe_context"] == 19_999    # (7-2-1)/0.2 = 20k, −1 to stay strictly under budget


def test_characterize_none_when_preflight_errors(monkeypatch):
    _patch_budget(monkeypatch)
    fake = _FakeEngine({"error": "model not found in HF cache"})
    _fake_worker(monkeypatch, fake)
    out = cuda.characterize("missing/model")
    assert out == {"model": "missing/model", "safe_context": None, "points": [],
                   "error": "model not found in HF cache"}


def test_characterize_stops_on_engine_refusal(monkeypatch):
    _patch_budget(monkeypatch)
    est = {"base_gb": 1.0, "slope_gb_per_k": 0.2, "budget_gb": 7.0,
           "max_context": None, "ref_baseline_gb": 0.0}
    fake = _FakeEngine(est, refuse_at=8000)
    _fake_worker(monkeypatch, fake)
    r = cuda.characterize("org/model")
    # 8000 refused → hard wall: bisect [4000, 8000), report a confirmed-safe context under it
    assert 4000 <= r["safe_context"] < 8000
    assert r["binding"] == "memory"
    assert r["safe_context"] in {p["context"] for p in r["points"]}


def test_characterize_l1_scheduler_skips_dispatch_when_predicted_breach(monkeypatch):
    _patch_budget(monkeypatch)
    # base already at budget → L1 refuses the first rung; nothing is dispatched
    est = {"base_gb": 6.95, "slope_gb_per_k": 0.2, "budget_gb": 7.0,
           "max_context": None, "ref_baseline_gb": 0.0}
    fake = _FakeEngine(est)
    _fake_worker(monkeypatch, fake)
    r = cuda.characterize("org/model")
    assert fake.measured == []
    assert r["safe_context"] is None


def test_characterize_l2_stops_when_actual_measurement_reaches_budget(monkeypatch):
    _patch_budget(monkeypatch)
    # L1 predicts safe (tiny slope), but the ACTUAL measured VRAM is high → L2 catches it
    est = {"base_gb": 1.0, "slope_gb_per_k": 0.0001, "budget_gb": 7.0,
           "max_context": None, "ref_baseline_gb": 0.0}
    fake = _FakeEngine(est, intercept=8.0, slope_per_k=0.0)   # every measurement reports 8 GB
    _fake_worker(monkeypatch, fake)
    r = cuda.characterize("org/model")
    assert fake.measured == [2000] and r["safe_context"] is None


def test_budget_params_uses_stored_calibration(monkeypatch):
    monkeypatch.setattr(cuda, "db", type("D", (), {"connected": staticmethod(lambda: contextlib.nullcontext(None))}))
    monkeypatch.setattr(cuda, "calibration",
                        type("P", (), {"get_calibration": staticmethod(
                            lambda con, eng: {"fixed_overhead_gb": 1.2})}), raising=False)
    margin, overhead = cuda._budget_params()
    assert (margin, overhead) == (cuda.DEFAULT_MARGIN_GB, 1.2)


def test_budget_params_falls_back_to_default_overhead(monkeypatch):
    monkeypatch.setattr(cuda, "db", type("D", (), {"connected": staticmethod(lambda: contextlib.nullcontext(None))}))
    monkeypatch.setattr(cuda, "calibration",
                        type("P", (), {"get_calibration": staticmethod(lambda con, eng: None)}),
                        raising=False)
    margin, overhead = cuda._budget_params()
    assert (margin, overhead) == (cuda.DEFAULT_MARGIN_GB, cuda.DEFAULT_OVERHEAD_GB)


def test_calibration_model_constant_is_transformers_format():
    # transformers-format (torch loads this), NOT the mlx-community 4-bit build apple uses
    assert cuda.CALIBRATION_MODEL == "HuggingFaceTB/SmolLM-135M-Instruct"


# --------------------------------------------------------------------------- #
# generate — governed one-shot CUDA inference (Spec 2026-06-23-capability-pipeline)
# --------------------------------------------------------------------------- #
def test_generate_drives_worker_capped_at_context(monkeypatch):
    seen = {}

    def worker(name, argv, *, input=None):
        seen["name"], seen["argv"], seen["input"] = name, argv, input
        return {"context": 8192, "completion": "hello there"}

    _patch_budget(monkeypatch, margin=1.0, overhead=0.6)
    monkeypatch.setattr(cuda, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    out = cuda.generate("org/m", "hi", max_context=8192, max_tokens=64)
    assert out == {"context": 8192, "completion": "hello there"}   # worker dict verbatim
    assert seen["name"] == "cuda"
    assert seen["argv"] == ["-m", "ara_engine_cuda.generate", "org/m", "8192",
                            "--margin", "1.0", "--overhead", "0.6", "--max-tokens", "64"]
    assert seen["input"] == "hi"               # prompt over stdin, not argv


# --------------------------------------------------------------------------- #
# KV-quant lever: --kv-quant {f16,q8_0,q4_0} → CUDA --kv-bits {None,8,4}
# --------------------------------------------------------------------------- #
def test_worker_argv_appends_kv_bits_for_quant():
    q4 = cuda._worker_argv("org/m", 2000, 1.0, 0.6, kv_quant="q4_0")
    assert q4[q4.index("--kv-bits") + 1] == "4"
    q8 = cuda._worker_argv("org/m", 2000, 1.0, 0.6, kv_quant="q8_0")
    assert q8[q8.index("--kv-bits") + 1] == "8"
    assert "--kv-bits" not in cuda._worker_argv("org/m", 2000, 1.0, 0.6)   # f16 default omits it


def test_characterize_passes_kv_dtype_bytes_to_driver(monkeypatch):
    # The decode-ceiling estimate must reflect the chosen cache size (the slope-bug class).
    _patch_budget(monkeypatch)
    seen = {}
    monkeypatch.setattr(cuda.driver, "characterize",
                        lambda model, **kw: seen.update(kw) or {"model": model})
    cuda.characterize("org/m", kv_quant="q4_0")
    assert seen["kv_dtype_bytes"] == cuda._CUDA_KV_BYTES["q4_0"]


def test_generate_appends_kv_bits_for_quant(monkeypatch):
    seen = {}
    _patch_budget(monkeypatch, margin=1.0, overhead=0.6)
    monkeypatch.setattr(cuda, "engine_env", type("E", (), {"run_worker": staticmethod(
        lambda name, argv, *, input=None: seen.update(argv=argv) or {"context": 8192, "completion": "x"})}))
    cuda.generate("org/m", "hi", max_context=8192, max_tokens=64, kv_quant="q8_0")
    assert seen["argv"][seen["argv"].index("--kv-bits") + 1] == "8"


# --------------------------------------------------------------------------- #
# Flash-attention: SDPA default, FA2 opt-in (--flash-attn), availability-gated
# --------------------------------------------------------------------------- #
def test_worker_argv_appends_flash_attn_when_opted_in():
    assert "--flash-attn" in cuda._worker_argv("org/m", 2000, 1.0, 0.6, flash_attn=True)
    assert "--flash-attn" not in cuda._worker_argv("org/m", 2000, 1.0, 0.6)   # SDPA default


def test_generate_appends_flash_attn_when_opted_in(monkeypatch):
    seen = {}
    _patch_budget(monkeypatch, margin=1.0, overhead=0.6)
    monkeypatch.setattr(cuda, "engine_env", type("E", (), {"run_worker": staticmethod(
        lambda name, argv, *, input=None: seen.update(argv=argv) or {"context": 8192, "completion": "x"})}))
    cuda.generate("org/m", "hi", max_context=8192, max_tokens=64, flash_attn=True)
    assert "--flash-attn" in seen["argv"]


def test_flash_attn_capable_reads_device_worker(monkeypatch):
    monkeypatch.setattr(cuda, "safe_limits", lambda: {"flash_attn_capable": True})
    assert cuda.flash_attn_capable() is True
    monkeypatch.setattr(cuda, "safe_limits", lambda: {"flash_attn_capable": False})
    assert cuda.flash_attn_capable() is False


def test_flash_attn_capable_false_when_worker_errors(monkeypatch):
    def boom():
        raise RuntimeError("no GPU")
    monkeypatch.setattr(cuda, "safe_limits", boom)
    assert cuda.flash_attn_capable() is False


# --------------------------------------------------------------------------- #
# Weight-quant lever (--weight-quant {none,int8,int4,fp8}) + FP8 capability
# --------------------------------------------------------------------------- #
def test_worker_argv_appends_weight_quant():
    assert cuda._worker_argv("org/m", 2000, 1.0, 0.6, weight_quant="int4")[-2:] == \
        ["--weight-quant", "int4"]
    assert "--weight-quant" not in cuda._worker_argv("org/m", 2000, 1.0, 0.6)   # none → omitted


def test_generate_appends_weight_quant(monkeypatch):
    seen = {}
    _patch_budget(monkeypatch, margin=1.0, overhead=0.6)
    monkeypatch.setattr(cuda, "engine_env", type("E", (), {"run_worker": staticmethod(
        lambda name, argv, *, input=None: seen.update(argv=argv) or {"context": 8192, "completion": "x"})}))
    cuda.generate("org/m", "hi", max_context=8192, max_tokens=64, weight_quant="int8")
    assert seen["argv"][-2:] == ["--weight-quant", "int8"]


# --------------------------------------------------------------------------- #
# Chunked prefill lever (--prefill-chunk N) → CUDA --prefill-chunk N
# --------------------------------------------------------------------------- #
def test_worker_argv_appends_prefill_chunk():
    argv = cuda._worker_argv("org/m", 2000, 1.0, 0.6, prefill_chunk=256)
    assert argv[-2:] == ["--prefill-chunk", "256"]
    assert "--prefill-chunk" not in cuda._worker_argv("org/m", 2000, 1.0, 0.6)   # off by default


def test_characterize_threads_prefill_chunk_to_measure(monkeypatch):
    # The certified ceiling must be measured with the same chunking run will use.
    _patch_budget(monkeypatch)
    seen = {}
    monkeypatch.setattr(cuda.driver, "characterize",
                        lambda model, measure=None, **kw: seen.update(
                            argv=measure(model, 2000)) or {"model": model})
    monkeypatch.setattr(cuda, "engine_env", type("E", (), {"run_worker": staticmethod(
        lambda name, argv, *, input=None: argv)}))
    cuda.characterize("org/m", prefill_chunk=256)
    assert seen["argv"][-2:] == ["--prefill-chunk", "256"]


def test_generate_appends_prefill_chunk(monkeypatch):
    seen = {}
    _patch_budget(monkeypatch, margin=1.0, overhead=0.6)
    monkeypatch.setattr(cuda, "engine_env", type("E", (), {"run_worker": staticmethod(
        lambda name, argv, *, input=None: seen.update(argv=argv) or {"context": 8192, "completion": "x"})}))
    cuda.generate("org/m", "hi", max_context=8192, max_tokens=64, prefill_chunk=256)
    assert seen["argv"][-2:] == ["--prefill-chunk", "256"]


# --------------------------------------------------------------------------- #
# benchmark — load-once multi-prompt completion via ara_engine_cuda.benchmark
# --------------------------------------------------------------------------- #
def test_benchmark_drives_worker_with_json_prompts(monkeypatch):
    import json

    seen = {}

    def worker(name, argv, *, input=None):
        seen["name"], seen["argv"], seen["input"] = name, argv, input
        return {"context": 4096, "results": [{"prompt_index": 0, "completion": "x"}]}

    _patch_budget(monkeypatch, margin=1.0, overhead=0.6)
    monkeypatch.setattr(cuda, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    out = cuda.benchmark("org/m", ["p0", "p1"], max_context=4096, max_tokens=128)
    assert out["results"][0]["completion"] == "x"          # worker dict verbatim
    assert seen["name"] == "cuda"
    assert seen["argv"] == ["-m", "ara_engine_cuda.benchmark", "org/m", "4096",
                            "--margin", "1.0", "--overhead", "0.6", "--max-tokens", "128"]
    assert json.loads(seen["input"]) == ["p0", "p1"]       # prompts as JSON array over stdin


def test_benchmark_appends_levers(monkeypatch):
    seen = {}
    _patch_budget(monkeypatch, margin=1.0, overhead=0.6)
    monkeypatch.setattr(cuda, "engine_env", type("E", (), {"run_worker": staticmethod(
        lambda name, argv, *, input=None: seen.update(argv=argv) or {"context": 4096, "results": []})}))
    cuda.benchmark("org/m", ["p"], max_context=4096, max_tokens=64,
                   kv_quant="q4_0", flash_attn=True, weight_quant="int8", prefill_chunk=256)
    a = seen["argv"]
    assert a[a.index("--kv-bits") + 1] == "4"
    assert "--flash-attn" in a
    assert a[a.index("--weight-quant") + 1] == "int8"
    assert a[a.index("--prefill-chunk") + 1] == "256"


def test_fp8_capable_reads_device_worker(monkeypatch):
    monkeypatch.setattr(cuda, "safe_limits", lambda: {"fp8_capable": True})
    assert cuda.fp8_capable() is True
    monkeypatch.setattr(cuda, "safe_limits", lambda: {"fp8_capable": False})
    assert cuda.fp8_capable() is False


def test_fp8_capable_false_when_worker_errors(monkeypatch):
    def boom():
        raise RuntimeError("no GPU")
    monkeypatch.setattr(cuda, "safe_limits", boom)
    assert cuda.fp8_capable() is False
