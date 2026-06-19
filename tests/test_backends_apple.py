"""backends/apple.py — a lean wmx-suite seam (stateless; ARA owns persistence)."""
from __future__ import annotations

from ara import acquire
from ara.backends import apple


def test_safe_limits_is_stateless(fake_wmx):
    m = apple.safe_limits()
    assert m["device"] == "Apple M4 Pro"
    assert m["total_gb"] == 48.0 and m["wall_gb"] == 40.0
    assert m["safe_budget_gb"] == 36.0 and m["margin_gb"] == 4.0
    assert m["headroom_gb"] == 28.0       # safe 36 − wired 8
    assert m["swap_free_gb"] == 2.0
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
    monkeypatch.setattr(acquire, "download", lambda repo_id: calls.append(repo_id))
    apple.download_calibration_model("org/calib-model")
    assert calls == ["org/calib-model"]


def test_calibrate_surfaces_effective_overhead(fake_wmx):
    fake_wmx.calibrate_return = {
        "measured_overhead_gb": 5.0, "default_overhead_gb": 6.0, "n_points": 4,
    }
    m = apple.calibrate("org/calib-model")
    assert m["device"] == "Apple M4 Pro"               # carries fresh limits …
    assert m["overhead_gb"] == 6.0                      # effective = max(default 6, measured 5)
    assert m["calibrated"] is True
    assert m["calibration"] == fake_wmx.calibrate_return  # … plus what it measured
    assert fake_wmx.calibrate_calls == ["org/calib-model"]


def test_calibrate_overhead_none_when_no_measurement(fake_wmx):
    fake_wmx.calibrate_return = {"n_points": 0}        # no overhead keys at all
    m = apple.calibrate("org/calib-model")
    assert m["overhead_gb"] is None


def test_calibration_model_constant_is_small_instruct():
    assert apple.CALIBRATION_MODEL == "mlx-community/SmolLM-135M-Instruct-4bit"
