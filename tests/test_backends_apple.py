"""backends/apple.py — the wmx-suite seam, exercised against a fake engine."""
from __future__ import annotations

from ara import acquire
from ara.backends import apple


def test_safe_limits_uncalibrated(fake_wmx):
    fake_wmx.profile = None
    m = apple.safe_limits()
    assert m["device"] == "Apple M4 Pro"
    assert m["total_gb"] == 48.0
    assert m["wall_gb"] == 40.0
    assert m["safe_budget_gb"] == 36.0
    assert m["margin_gb"] == 4.0
    assert m["headroom_gb"] == 28.0      # safe 36 − wired 8
    assert m["swap_free_gb"] == 2.0
    assert m["calibrated"] is False
    assert m["overhead_gb"] is None
    assert m["calibrated_at"] is None


def test_safe_limits_calibrated(fake_wmx):
    fake_wmx.profile = {"fixed_overhead_gb": 5.5, "calibrated_at": "2026-06-18T09:30:00Z"}
    m = apple.safe_limits()
    assert m["calibrated"] is True
    assert m["overhead_gb"] == 5.5
    assert m["calibrated_at"] == "2026-06-18"  # trimmed to the date


def test_calibration_model_cached_true(fake_wmx):
    fake_wmx.describe_return = {"id": "model"}
    assert apple.calibration_model_cached("any/model") is True


def test_calibration_model_cached_false_when_absent(fake_wmx):
    fake_wmx.describe_return = None
    assert apple.calibration_model_cached("any/model") is False


def test_calibration_model_cached_false_on_error(fake_wmx):
    fake_wmx.describe_raises = True
    assert apple.calibration_model_cached("any/model") is False


def test_download_calibration_model_delegates_to_acquire(fake_wmx, monkeypatch):
    calls = []
    monkeypatch.setattr(acquire, "download", lambda repo_id: calls.append(repo_id))
    apple.download_calibration_model("org/calib-model")
    assert calls == ["org/calib-model"]


def test_calibrate_returns_limits_with_calibration_subdict(fake_wmx):
    fake_wmx.calibrate_return = {
        "measured_overhead_gb": 5.0, "default_overhead_gb": 6.0, "n_points": 4,
    }
    m = apple.calibrate("org/calib-model")
    # carries fresh limits …
    assert m["device"] == "Apple M4 Pro"
    # … plus exactly what calibration measured
    assert m["calibration"] == fake_wmx.calibrate_return
    assert fake_wmx.calibrate_calls == ["org/calib-model"]


def test_calibration_model_constant_is_small_instruct():
    assert apple.CALIBRATION_MODEL == "mlx-community/SmolLM-135M-Instruct-4bit"
