# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""calibration.py — this machine's stored per-engine baseline overhead."""
from __future__ import annotations

from ara import calibration


def test_save_and_get_calibration(store, monkeypatch):
    monkeypatch.setattr(calibration, "machine_key", lambda: "mkey")
    calibration.save_calibration(store, "wcx", fixed_overhead_gb=0.14, calibrated_at="2026-06-19")
    row = calibration.get_calibration(store, "wcx")
    assert row["fixed_overhead_gb"] == 0.14 and row["calibrated_at"] == "2026-06-19"


def test_get_calibration_missing_is_none(store, monkeypatch):
    monkeypatch.setattr(calibration, "machine_key", lambda: "mkey")
    assert calibration.get_calibration(store, "wmx") is None


def test_save_calibration_defaults_timestamp(store, monkeypatch):
    monkeypatch.setattr(calibration, "machine_key", lambda: "mkey")
    calibration.save_calibration(store, "wcx", fixed_overhead_gb=0.2)   # no calibrated_at
    assert calibration.get_calibration(store, "wcx")["calibrated_at"]   # something was stamped
