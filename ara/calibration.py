# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Per-engine baseline calibration — this machine's measured fixed memory overhead for an engine.

Keyed by ``profile.machine_key()`` so a calibration is reused only on the same machine. ARA owns
this (the engine measures; ARA remembers), stored in the ``calibrations`` table via ``ara.db``.
"""
from __future__ import annotations

from ara import db
from ara.profile import machine_key


def save_calibration(con, engine: str, *, fixed_overhead_gb: float,
                     calibrated_at: str | None = None,
                     wall_gb: float | None = None,
                     safe_budget_gb: float | None = None) -> None:
    """Persist this machine's measured overhead for *engine* (and, when known, the measured
    memory wall + safe budget the engine read, so profile/recommend can report reality)."""
    db.upsert_calibration(con, machine_key(), engine,
                          fixed_overhead_gb=fixed_overhead_gb,
                          calibrated_at=calibrated_at or db._now(),
                          wall_gb=wall_gb, safe_budget_gb=safe_budget_gb)


def get_calibration(con, engine: str) -> dict | None:
    """This machine's stored calibration for *engine*, or None if never calibrated."""
    return db.get_calibration(con, machine_key(), engine)
