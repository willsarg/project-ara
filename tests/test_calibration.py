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


def test_save_calibration_persists_measured_wall(store, monkeypatch):
    # Spec 2026-06-23-capability-pipeline: the measured wall rides with the overhead so profile
    # can report the engine's real numbers, not the heuristic.
    monkeypatch.setattr(calibration, "machine_key", lambda: "mkey")
    calibration.save_calibration(store, "wcx", fixed_overhead_gb=0.6,
                                 wall_gb=24.0, safe_budget_gb=23.0)
    row = calibration.get_calibration(store, "wcx")
    assert row["wall_gb"] == 24.0 and row["safe_budget_gb"] == 23.0


def test_calibration_round_trips_exact_measurement_authority(store, monkeypatch):
    monkeypatch.setattr(calibration, "machine_key", lambda: "mkey")
    calibration.save_calibration(
        store,
        "mlx",
        fixed_overhead_gb=0.6,
        wall_gb=17.760009765625,
        safe_budget_gb=15.760009765625,
        wall_bytes=19_069_665_280,
        safe_budget_bytes=16_922_181_632,
        memory_unit="GiB",
        environment_key="mlx-env",
        authority_key="mlx-authority",
        authority_evidence={"schema": "mlx-memory-authority:v1"},
    )

    row = calibration.get_calibration(
        store, "mlx", authority_key="mlx-authority")
    assert row["wall_bytes"] == 19_069_665_280
    assert row["safe_budget_bytes"] == 16_922_181_632
    assert row["authority_evidence"]["schema"] == "mlx-memory-authority:v1"


def test_calibration_round_trips_exact_engine_fingerprint(store, monkeypatch):
    monkeypatch.setattr(calibration, "machine_key", lambda: "mkey")
    calibration.save_calibration(
        store,
        "cuda",
        fixed_overhead_gb=0.9,
        authority_key="authority:cuda",
        engine_fingerprint="engine:v2:sha256:cuda-a",
    )

    assert calibration.get_calibration(
        store,
        "cuda",
        authority_key="authority:cuda",
        engine_fingerprint="engine:v2:sha256:cuda-a",
    )["fixed_overhead_gb"] == 0.9
    assert calibration.get_calibration(
        store,
        "cuda",
        authority_key="authority:cuda",
        engine_fingerprint="engine:v2:sha256:cuda-b",
    ) is None


def test_save_calibration_wall_defaults_none(store, monkeypatch):
    # Existing callers (overhead only) still work — wall columns stay NULL.
    monkeypatch.setattr(calibration, "machine_key", lambda: "mkey")
    calibration.save_calibration(store, "wcx", fixed_overhead_gb=0.6)
    row = calibration.get_calibration(store, "wcx")
    assert row["wall_gb"] is None and row["safe_budget_gb"] is None
