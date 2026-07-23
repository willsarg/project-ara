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
                     safe_budget_gb: float | None = None,
                     environment_key: str = db.UNSCOPED_ENVIRONMENT_KEY,
                     authority_key: str = db.UNSCOPED_AUTHORITY_KEY,
                     memory_unit: str = "GiB",
                     wall_bytes: int | None = None,
                     safe_budget_bytes: int | None = None,
                     authority_evidence: dict | None = None,
                     engine_fingerprint: str = db.LEGACY_ENGINE_FINGERPRINT) -> None:
    """Persist this machine's measured overhead for *engine* (and, when known, the measured
    memory wall + safe budget the engine read, so profile/recommend can report reality)."""
    db.upsert_calibration(con, machine_key(), engine,
                          fixed_overhead_gb=fixed_overhead_gb,
                          calibrated_at=calibrated_at or db._now(),
                          wall_gb=wall_gb, safe_budget_gb=safe_budget_gb,
                          environment_key=environment_key,
                          authority_key=authority_key,
                          memory_unit=memory_unit,
                          wall_bytes=wall_bytes,
                          safe_budget_bytes=safe_budget_bytes,
                          authority_evidence=authority_evidence,
                          engine_fingerprint=engine_fingerprint)


def get_calibration(
    con,
    engine: str,
    *,
    authority_key: str | None = None,
    engine_fingerprint: str | None = None,
) -> dict | None:
    """This machine's stored calibration for *engine*, or None if never calibrated."""
    return db.get_calibration(
        con, machine_key(), engine, authority_key=authority_key,
        engine_fingerprint=engine_fingerprint)
