# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""backends/apple.py — a lean wmx-suite seam (stateless; ARA owns persistence)."""
from __future__ import annotations

import pytest

from ara import acquire, catalog, engine_env
from ara.backends import apple


@pytest.fixture(autouse=True)
def _no_catalog_network(monkeypatch):
    """Keep the suite offline: characterize now calls catalog.describe; stub it out."""
    monkeypatch.setattr(catalog, "describe", lambda m: None)


def _fake_worker(monkeypatch, fn):
    """Patch engine_env.run_worker on the apple module (the only engine seam)."""
    monkeypatch.setattr(apple, "engine_env",
                        type("E", (), {"run_worker": staticmethod(fn)}))


# The engine facts the wmx `device limits` worker returns (ARA overlays its own fields).
_LIMITS_FACTS = {
    "device": "Apple M4 Pro", "total_gb": 48.0, "wall_gb": 40.0,
    "safe_budget_gb": 36.0, "margin_gb": 4.0, "headroom_gb": 28.0, "swap_free_gb": 2.0,
}


def test_safe_limits_drives_device_worker_and_overlays(monkeypatch):
    calls = []

    def worker(name, argv):
        calls.append((name, argv))
        return dict(_LIMITS_FACTS)

    _fake_worker(monkeypatch, worker)
    m = apple.safe_limits()
    assert calls == [("apple", ["-m", "wmx_suite.device", "limits"])]
    assert m["device"] == "Apple M4 Pro"
    assert m["total_gb"] == 48.0 and m["wall_gb"] == 40.0
    assert m["safe_budget_gb"] == 36.0 and m["margin_gb"] == 4.0
    assert m["headroom_gb"] == 28.0 and m["swap_free_gb"] == 2.0
    # no stored calibration in the engine — ARA overlays it from its own store
    assert m["calibrated"] is False
    assert m["overhead_gb"] is None
    assert m["calibrated_at"] is None


def test_calibration_model_cached_true(monkeypatch):
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache",
                        lambda m, fn: "/path/to/config.json")
    assert apple.calibration_model_cached("any/model") is True


def test_calibration_model_cached_false_when_absent(monkeypatch):
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda m, fn: None)
    assert apple.calibration_model_cached("any/model") is False


def test_calibration_model_cached_false_on_error(monkeypatch):
    def boom(m, fn):
        raise RuntimeError("hf down")
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", boom)
    assert apple.calibration_model_cached("any/model") is False


def test_download_calibration_model_delegates_to_acquire(monkeypatch):
    calls = []
    monkeypatch.setattr(acquire, "download", lambda repo_id, *, progress=False: calls.append(repo_id))
    apple.download_calibration_model("org/calib-model")
    assert calls == ["org/calib-model"]


def test_download_calibration_model_passes_progress_to_acquire(monkeypatch):
    """download_calibration_model(progress=True) passes progress=True to acquire.download.

    Slug: 2026-06-24-download-progress
    """
    captured = {}
    monkeypatch.setattr(acquire, "download",
                        lambda repo_id, *, progress=False: captured.update(progress=progress))
    apple.download_calibration_model("org/m", progress=True)
    assert captured["progress"] is True


def test_download_calibration_model_default_progress_false(monkeypatch):
    """download_calibration_model() default passes progress=False to acquire.download.

    Slug: 2026-06-24-download-progress
    """
    captured = {}
    monkeypatch.setattr(acquire, "download",
                        lambda repo_id, *, progress=False: captured.update(progress=progress))
    apple.download_calibration_model("org/m")
    assert captured["progress"] is False


def test_characterize_accepts_progress_and_does_not_stream(monkeypatch):
    """apple.characterize(progress=True) accepts progress for symmetry but does NOT pass stream
    to run_worker — bars already ran in-process during the pre-fetch step.

    Slug: 2026-06-24-download-progress
    """
    stream_kwargs = []

    def worker(name, argv, **kwargs):
        stream_kwargs.append(kwargs.get("stream", False))
        model = argv[2]
        ctx = int(argv[3])
        if "--preflight" in argv:
            return {"base_gb": 5.0, "slope_gb_per_k": 1.0, "budget_gb": 36.0,
                    "max_context": 4000, "ref_baseline_gb": 0.0}
        return {"context": ctx, "mem_gb": 5.0 + (ctx / 1000)}

    monkeypatch.setattr(apple, "_budget_params", lambda: (2.0, 1.0))
    monkeypatch.setattr(apple, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    apple.characterize("org/m", progress=True)
    # stream must never be True for apple — bars ran in-process during pre-fetch
    assert all(s is False for s in stream_kwargs)


def _calibrate_worker(monkeypatch, calibration, calls=None):
    """Worker that answers the device 'calibrate' call with *calibration*, 'limits' with facts."""
    def worker(name, argv):
        if calls is not None:
            calls.append(argv)
        return dict(calibration) if argv[2] == "calibrate" else dict(_LIMITS_FACTS)
    _fake_worker(monkeypatch, worker)


def test_calibrate_surfaces_effective_overhead(monkeypatch):
    calls = []
    _calibrate_worker(monkeypatch, {
        "measured_overhead_gb": 5.0, "default_overhead_gb": 6.0, "n_points": 4,
    }, calls)
    m = apple.calibrate("org/calib-model")
    assert m["device"] == "Apple M4 Pro"               # carries fresh limits …
    assert m["overhead_gb"] == 6.0                      # effective = max(default 6, measured 5)
    assert m["calibrated"] is True
    assert m["calibration"]["n_points"] == 4           # … plus what it measured
    assert ["-m", "wmx_suite.device", "calibrate", "org/calib-model"] in calls


def test_calibrate_overhead_none_when_no_measurement(monkeypatch):
    _calibrate_worker(monkeypatch, {"n_points": 0})    # no overhead keys at all
    m = apple.calibrate("org/calib-model")
    assert m["overhead_gb"] is None


def test_calibrate_returns_uncalibrated_on_worker_error(monkeypatch):
    """Worker returns an error dict → calibrate() must NOT claim calibrated=True (Rule #3)."""
    def worker(name, argv):
        if argv[2] == "calibrate":
            return {"error": "model not found in HF cache"}
        return dict(_LIMITS_FACTS)
    _fake_worker(monkeypatch, worker)
    m = apple.calibrate("org/calib-model")
    assert m["calibrated"] is False
    assert m["overhead_gb"] is None
    assert "calibration_error" in m
    assert "org/calib-model" in m["calibration_error"]
    assert "model not found in HF cache" in m["calibration_error"]


def test_calibrate_returns_uncalibrated_on_worker_exception(monkeypatch):
    """Worker raises → calibrate() must NOT crash; must return uncalibrated + error (Rule #3)."""
    def worker(name, argv):
        if argv[2] == "calibrate":
            raise RuntimeError("engine env failed to start")
        return dict(_LIMITS_FACTS)
    _fake_worker(monkeypatch, worker)
    m = apple.calibrate("org/calib-model")
    assert m["calibrated"] is False
    assert m["overhead_gb"] is None
    assert "calibration_error" in m
    assert "org/calib-model" in m["calibration_error"]
    assert "engine env failed to start" in m["calibration_error"]


class _FakeEngine:
    """Stand-in for engine_env.run_worker: answers preflight + per-ctx measurements,
    driven by a canned estimate and a linear memory model. Records every spawn."""
    def __init__(self, est, intercept=5.0, slope_per_k=1.0, refuse_at=None):
        self.est, self.intercept, self.slope = est, intercept, slope_per_k
        self.refuse_at = refuse_at
        self.measured: list[int] = []

    def __call__(self, name, argv):
        assert name == "apple"
        model = argv[2]
        ctx = int(argv[3])
        if "--preflight" in argv:
            return dict(self.est)
        self.measured.append(ctx)
        if self.refuse_at is not None and ctx >= self.refuse_at:
            return {"context": ctx, "refused": True, "reason": "engine veto"}
        return {"context": ctx, "mem_gb": self.intercept + self.slope * (ctx / 1000)}


def _patch_budget(monkeypatch, margin=2.0, overhead=1.0):
    monkeypatch.setattr(apple, "_budget_params", lambda: (margin, overhead))


def test_characterize_drives_ramp_over_engine_env(monkeypatch):
    _patch_budget(monkeypatch)
    est = {"base_gb": 5.0, "slope_gb_per_k": 1.0, "budget_gb": 36.0, "max_context": 16000, "ref_baseline_gb": 0.0}
    fake = _FakeEngine(est, intercept=5.0, slope_per_k=1.0)
    monkeypatch.setattr(apple, "engine_env",
                        type("E", (), {"run_worker": staticmethod(fake)}))
    r = apple.characterize("org/model")
    assert r["model"] == "org/model"
    # fitted memory ceiling ~31k exceeds the model's 16k window → capped, window-bound
    assert r["safe_context"] == 16_000
    assert r["binding"] == "context_window"
    assert all(c <= 16000 for c in fake.measured)
    assert r["points"][0] == {"context": 2000, "mem_gb": 7.0}


def test_characterize_subtracts_live_ref_baseline_from_ceiling(monkeypatch):
    _patch_budget(monkeypatch)
    # delta fit: model base 5, slope 1; live OS baseline 8 GB → ceiling (36-8-5)/1 = 23k
    est = {"base_gb": 13.0, "slope_gb_per_k": 1.0, "budget_gb": 36.0,
           "max_context": None, "ref_baseline_gb": 8.0}
    fake = _FakeEngine(est, intercept=5.0, slope_per_k=1.0)
    monkeypatch.setattr(apple, "engine_env",
                        type("E", (), {"run_worker": staticmethod(fake)}))
    r = apple.characterize("org/model")
    assert r["safe_context"] == 22_999    # (36-8-5)/1 = 23k, −1 to stay strictly under budget


def test_characterize_none_when_preflight_errors(monkeypatch):
    _patch_budget(monkeypatch)
    fake = _FakeEngine({"error": "model not found in HF cache"})
    monkeypatch.setattr(apple, "engine_env",
                        type("E", (), {"run_worker": staticmethod(fake)}))
    out = apple.characterize("missing/model")
    assert out == {"model": "missing/model", "safe_context": None, "points": [],
                   "error": "model not found in HF cache"}


def test_characterize_stops_on_engine_refusal(monkeypatch):
    _patch_budget(monkeypatch)
    est = {"base_gb": 5.0, "slope_gb_per_k": 1.0, "budget_gb": 36.0, "max_context": None, "ref_baseline_gb": 0.0}
    fake = _FakeEngine(est, refuse_at=8000)
    monkeypatch.setattr(apple, "engine_env",
                        type("E", (), {"run_worker": staticmethod(fake)}))
    r = apple.characterize("org/model")
    # 2000 + 4000 measured; 8000 refused → the abort is a hard wall, so ARA bisects [4000, 8000)
    # and reports a confirmed-safe context strictly under it (never extrapolating past the abort).
    assert 4000 <= r["safe_context"] < 8000
    assert r["binding"] == "memory"
    assert r["safe_context"] in {p["context"] for p in r["points"]}


def test_characterize_l1_scheduler_skips_dispatch_when_predicted_breach(monkeypatch):
    _patch_budget(monkeypatch)
    # base already at budget → L1 plan_next refuses the first rung; nothing is dispatched
    est = {"base_gb": 35.9, "slope_gb_per_k": 1.0, "budget_gb": 36.0, "max_context": None, "ref_baseline_gb": 0.0}
    fake = _FakeEngine(est)
    monkeypatch.setattr(apple, "engine_env",
                        type("E", (), {"run_worker": staticmethod(fake)}))
    r = apple.characterize("org/model")
    assert fake.measured == []          # L1 prevented any measurement dispatch
    assert r["safe_context"] is None


def test_characterize_l2_stops_when_actual_measurement_reaches_budget(monkeypatch):
    _patch_budget(monkeypatch)
    # L1 predicts safe (low slope), but the ACTUAL measured memory is high → L2 catches it
    est = {"base_gb": 5.0, "slope_gb_per_k": 0.001, "budget_gb": 36.0, "max_context": None, "ref_baseline_gb": 0.0}
    fake = _FakeEngine(est, intercept=40.0, slope_per_k=0.0)  # every measurement reports 40 GB
    monkeypatch.setattr(apple, "engine_env",
                        type("E", (), {"run_worker": staticmethod(fake)}))
    r = apple.characterize("org/model")
    # first rung dispatched, measured 40 >= 36 → L2 refuses → stop with no usable points
    assert fake.measured == [2000] and r["safe_context"] is None


# --- KV-quant lever (parity with the Vulkan lane) ----------------------------------- #

def test_worker_argv_omits_kv_bits_for_fp16():
    argv = apple._worker_argv("m", 4000, 2.0, 1.0, kv_quant="f16")
    assert "--kv-bits" not in argv


@pytest.mark.parametrize("kv_quant,bits", [("q8_0", "8"), ("q4_0", "4")])
def test_worker_argv_maps_kv_quant_to_kv_bits(kv_quant, bits):
    argv = apple._worker_argv("m", 4000, 2.0, 1.0, kv_quant=kv_quant)
    assert argv[argv.index("--kv-bits") + 1] == bits


def _capture_driver_kv_dtype_bytes(monkeypatch):
    captured = {}

    def fake_characterize(model, *, preflight, measure, schedule, kv_dtype_bytes=2.0):
        captured["kv_dtype_bytes"] = kv_dtype_bytes
        return {"model": model, "safe_context": 1, "points": []}

    monkeypatch.setattr(apple.driver, "characterize", fake_characterize)
    _patch_budget(monkeypatch)
    return captured


def test_characterize_decode_estimate_defaults_to_fp16(monkeypatch):
    captured = _capture_driver_kv_dtype_bytes(monkeypatch)
    apple.characterize("org/m")
    assert captured["kv_dtype_bytes"] == 2.0


def test_characterize_decode_estimate_reflects_kv_quant(monkeypatch):
    captured = _capture_driver_kv_dtype_bytes(monkeypatch)
    apple.characterize("org/m", kv_quant="q8_0")
    assert captured["kv_dtype_bytes"] == pytest.approx(apple._MLX_KV_BYTES["q8_0"])


def test_characterize_threads_kv_bits_to_worker(monkeypatch):
    _patch_budget(monkeypatch)
    seen = []
    est = {"base_gb": 5.0, "slope_gb_per_k": 1.0, "budget_gb": 36.0,
           "max_context": 8000, "ref_baseline_gb": 0.0}

    def worker(name, argv):
        seen.append(argv)
        if "--preflight" in argv:
            return dict(est)
        return {"context": int(argv[3]), "mem_gb": 5.0 + int(argv[3]) / 1000}

    monkeypatch.setattr(apple, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    apple.characterize("org/m", kv_quant="q8_0")
    assert seen and all(a[a.index("--kv-bits") + 1] == "8" for a in seen)


def test_generate_threads_kv_bits_to_worker(monkeypatch):
    _patch_budget(monkeypatch)
    seen = {}

    def worker(name, argv, *, input=None):
        seen["argv"] = argv
        return {"context": 4000, "completion": "ok"}

    monkeypatch.setattr(apple, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    apple.generate("org/m", "hi", max_context=4000, max_tokens=16, kv_quant="q4_0")
    assert seen["argv"][seen["argv"].index("--kv-bits") + 1] == "4"


def test_generate_omits_kv_bits_for_fp16(monkeypatch):
    _patch_budget(monkeypatch)
    seen = {}

    def worker(name, argv, *, input=None):
        seen["argv"] = argv
        return {"context": 4000, "completion": "ok"}

    monkeypatch.setattr(apple, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    apple.generate("org/m", "hi", max_context=4000, max_tokens=16)
    assert "--kv-bits" not in seen["argv"]


def test_budget_params_uses_stored_calibration(monkeypatch):
    monkeypatch.setattr(apple, "db", type("D", (), {"connect": staticmethod(lambda: None)}))
    monkeypatch.setattr(apple, "calibration",
                        type("P", (), {"get_calibration": staticmethod(
                            lambda con, eng: {"fixed_overhead_gb": 5.5})}), raising=False)
    margin, overhead = apple._budget_params()
    assert (margin, overhead) == (apple.DEFAULT_MARGIN_GB, 5.5)


def test_budget_params_falls_back_to_default_overhead(monkeypatch):
    monkeypatch.setattr(apple, "db", type("D", (), {"connect": staticmethod(lambda: None)}))
    monkeypatch.setattr(apple, "calibration",
                        type("P", (), {"get_calibration": staticmethod(lambda con, eng: None)}),
                        raising=False)
    margin, overhead = apple._budget_params()
    assert (margin, overhead) == (apple.DEFAULT_MARGIN_GB, apple.DEFAULT_OVERHEAD_GB)


def test_calibration_model_constant_is_small_instruct():
    assert apple.CALIBRATION_MODEL == "mlx-community/SmolLM-135M-Instruct-4bit"


# --------------------------------------------------------------------------- #
# generate — governed one-shot MLX inference (Spec 2026-06-23-capability-pipeline)
# --------------------------------------------------------------------------- #
def test_generate_drives_worker_capped_at_context(monkeypatch):
    seen = {}

    def worker(name, argv, *, input=None):
        seen["name"], seen["argv"], seen["input"] = name, argv, input
        return {"context": 8192, "completion": "hello there"}

    _patch_budget(monkeypatch, margin=2.0, overhead=1.0)
    monkeypatch.setattr(apple, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    out = apple.generate("org/m", "hi", max_context=8192, max_tokens=64)
    assert out == {"context": 8192, "completion": "hello there"}   # worker dict verbatim
    assert seen["name"] == "apple"
    assert seen["argv"] == ["-m", "wmx_suite.generate", "org/m", "8192",
                            "--margin", "2.0", "--overhead", "1.0", "--max-tokens", "64"]
    assert seen["input"] == "hi"               # prompt over stdin, not argv


# --------------------------------------------------------------------------- #
# serve — governed long-lived MLX server
# --------------------------------------------------------------------------- #

_SENTINEL_PROC = object()   # a distinct sentinel for "the Popen object"


def _fake_start_server(monkeypatch, url="http://127.0.0.1:8080", ctx=4096):
    """Monkeypatch engine_env.start_worker_server to capture argv and return a canned result."""
    captured = {}

    def fake_start(name, argv):
        captured["name"] = name
        captured["argv"] = argv
        return _SENTINEL_PROC, {"url": url, "context": ctx}

    monkeypatch.setattr(engine_env, "start_worker_server", fake_start)
    return captured


def test_serve_builds_exact_argv(monkeypatch):
    """serve() builds the exact wmx_suite.serve argv and passes it to start_worker_server."""
    _patch_budget(monkeypatch)
    captured = _fake_start_server(monkeypatch)
    apple.serve("org/m", port=8080, max_context=4096)
    assert captured["name"] == "apple"
    assert captured["argv"] == [
        "-m", "wmx_suite.serve", "org/m", "4096",
        "--margin", "2.0", "--overhead", "1.0",
        "--port", "8080",
    ]


def test_serve_appends_kv_bits_for_q4_0(monkeypatch):
    """kv_quant='q4_0' → --kv-bits 4 appended after the core argv."""
    _patch_budget(monkeypatch)
    captured = _fake_start_server(monkeypatch)
    apple.serve("org/m", port=8080, max_context=4096, kv_quant="q4_0")
    argv = captured["argv"]
    assert "--kv-bits" in argv
    assert argv[argv.index("--kv-bits") + 1] == "4"


def test_serve_omits_kv_bits_for_f16(monkeypatch):
    """kv_quant='f16' (default) → --kv-bits must NOT appear in argv."""
    _patch_budget(monkeypatch)
    captured = _fake_start_server(monkeypatch)
    apple.serve("org/m", port=8080, max_context=4096)
    assert "--kv-bits" not in captured["argv"]


def test_serve_returns_proc_url_context(monkeypatch):
    """serve() unpacks the ready signal into (proc, url, context)."""
    _patch_budget(monkeypatch)
    _fake_start_server(monkeypatch, url="http://127.0.0.1:9999", ctx=8192)
    proc, url, ctx = apple.serve("org/m", port=9999, max_context=8192)
    assert proc is _SENTINEL_PROC
    assert url == "http://127.0.0.1:9999"
    assert ctx == 8192
