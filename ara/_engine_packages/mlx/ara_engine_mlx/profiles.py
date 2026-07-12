# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Per-machine cold-start constants for the pre-flight base estimate.

The crash wall, the ambient baseline, and per-model fits already adapt to the host
at runtime. The only part still tied to the testbed is the cold-start base estimate for
models that haven't been characterized yet: until a machine is calibrated it falls back to
priors measured on the M4 Pro testbed. The native MLX calibration worker measures this machine's real
FIXED_OVERHEAD_GB once and stores it (keyed by chip + RAM + macOS major) so the estimate
is trustworthy on any Apple Silicon SKU. `characterize` remains the per-model mechanism.
"""
from __future__ import annotations

import sqlite3

from . import db, system

# Loose priors, measured loosely on the M4 Pro testbed (see probe.py); the per-machine
# fallback until `calibrate` runs. The resident factor is
# HELD FIXED (effective factor measured 0.88-1.10 across models); only the overhead is
# calibrated per machine, and never stored below this default (calibration only tightens).
DEFAULT_RESIDENT_FACTOR = 1.05
DEFAULT_FIXED_OVERHEAD_GB = 1.0


def machine_key() -> tuple[str, int, int]:
    """(device_name, total_ram_bytes, macos_major) identifying the current machine.

    total RAM is read in BYTES from mx.device_info()['memory_size'] (SystemLimits.total_gb
    is already divided by 1e9, so it can't supply a stable integer key).
    """
    import mlx.core as mx
    d = mx.device_info()
    return (str(d.get("device_name", "")), int(d.get("memory_size", 0)), system.macos_major())


def cold_start_constants(con: sqlite3.Connection) -> tuple[float, float, str]:
    """Return (resident_factor, fixed_overhead_gb, source).

    source == "profile" when a stored profile matches this machine, else "default".
    The factor is always the default (held fixed); only the overhead is profile-specific.
    """
    profile = db.get_profile(con, machine_key())
    if profile is not None:
        return DEFAULT_RESIDENT_FACTOR, float(profile["fixed_overhead_gb"]), "profile"
    return DEFAULT_RESIDENT_FACTOR, DEFAULT_FIXED_OVERHEAD_GB, "default"


def embedding_machine_key(model_id: str, mlx_version: str) -> tuple[str, int, int, str, str]:
    """5-part key for an embedding profile: machine identity + MLX version + model."""
    dev, ram, osv = machine_key()
    return (dev, ram, osv, mlx_version, model_id)


def embedding_coeffs(con: sqlite3.Connection, model_id: str,
                     mlx_version: str) -> tuple[float, float, float] | None:
    """Stored (intercept_gb, linear, quad) gate coefficients for this machine+mlx+model,
    or None when there is no matching calibration profile (→ cold-start fallback)."""
    row = db.get_embedding_profile(con, embedding_machine_key(model_id, mlx_version))
    if row is None:
        return None
    return (float(row["coef_intercept_gb"]), float(row["coef_linear"]),
            float(row["coef_quad"]))


def upsert_embedding_coeffs(con: sqlite3.Connection, model_id: str, mlx_version: str, *,
                            coef_intercept_gb: float, coef_linear: float,
                            coef_quad: float, n_points: int) -> None:
    db.upsert_embedding_profile(
        con, embedding_machine_key(model_id, mlx_version),
        coef_intercept_gb=coef_intercept_gb, coef_linear=coef_linear,
        coef_quad=coef_quad, n_points=n_points,
    )
