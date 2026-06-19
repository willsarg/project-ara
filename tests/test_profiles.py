"""profiles.py — this machine's identity + its stored per-engine calibration."""
from __future__ import annotations

import types

from ara import profiles


def test_machine_key_is_composite_and_stable(monkeypatch):
    monkeypatch.setattr(profiles.detect, "chip_name", lambda: "TestCPU")
    monkeypatch.setattr(profiles.detect, "accelerator",
                        lambda chip: types.SimpleNamespace(name="TestGPU"))
    monkeypatch.setattr(profiles.psutil, "virtual_memory",
                        lambda: types.SimpleNamespace(total=34359738368))
    monkeypatch.setattr(profiles.platform, "system", lambda: "Linux")
    k = profiles.machine_key()
    assert k == "TestCPU|TestGPU|34359738368|Linux"
    assert profiles.machine_key() == k          # stable across calls


def test_save_and_get_calibration(store, monkeypatch):
    monkeypatch.setattr(profiles, "machine_key", lambda: "mkey")
    profiles.save_calibration(store, "wcx", fixed_overhead_gb=0.14, calibrated_at="2026-06-19")
    row = profiles.get_calibration(store, "wcx")
    assert row["fixed_overhead_gb"] == 0.14 and row["calibrated_at"] == "2026-06-19"


def test_get_calibration_missing_is_none(store, monkeypatch):
    monkeypatch.setattr(profiles, "machine_key", lambda: "mkey")
    assert profiles.get_calibration(store, "wmx") is None


def test_save_calibration_defaults_timestamp(store, monkeypatch):
    monkeypatch.setattr(profiles, "machine_key", lambda: "mkey")
    profiles.save_calibration(store, "wcx", fixed_overhead_gb=0.2)   # no calibrated_at
    assert profiles.get_calibration(store, "wcx")["calibrated_at"]   # something was stamped
