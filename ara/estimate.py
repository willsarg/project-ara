# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Engine-free analytic capability estimate — the reasoning behind ``ara profile``.

ARA's *analytic* layer reasons over ``detect`` facts (no engine, no model load) and checks whether
a model's weights + context window fit an available budget. Apple/MLX has no current budget at
this seam because Metal authority is read only inside the isolated engine. ``characterize`` later
measures the real ceiling. See Spec 2026-06-23-capability-pipeline.
"""
from __future__ import annotations

import math

from ara.contracts import ramp

# ARA policy (not engine-measured): the analytic safety margin below a known wall.
MARGIN_GB = 2.0

# The analytic layer's unit contract is binary GiB (matching detect's RAM/VRAM and the KV slope).
# Weight sizes arrive as DECIMAL GB (on-disk bytes / 1e9, the disk-space denomination) and are
# converted at the model_fit boundary. Slug 2026-07-02-analytic-units-gib.
GIB = 1024 ** 3
MEASUREMENT_BYTE_TOLERANCE = 1


def _nonnegative_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def measurement_validation_error(measured: dict) -> str | None:
    """Return the first invariant violated by stored memory evidence, or ``None`` when valid."""
    wall = measured.get("wall_gb")
    safe = measured.get("safe_budget_gb")
    if wall is None:
        return "missing_wall_gb"
    if not _nonnegative_number(wall):
        return "invalid_wall_gb"
    if safe is None:
        return "missing_safe_budget_gb"
    if not _nonnegative_number(safe):
        return "invalid_safe_budget_gb"
    if safe > wall:
        return "safe_budget_exceeds_wall"

    wall_bytes = measured.get("wall_bytes")
    safe_bytes = measured.get("safe_budget_bytes")
    if wall_bytes is not None and (
            not isinstance(wall_bytes, int) or isinstance(wall_bytes, bool) or wall_bytes < 0):
        return "invalid_wall_bytes"
    if safe_bytes is not None and (
            not isinstance(safe_bytes, int) or isinstance(safe_bytes, bool) or safe_bytes < 0):
        return "invalid_safe_budget_bytes"
    if wall_bytes is not None and safe_bytes is not None and safe_bytes > wall_bytes:
        return "safe_budget_bytes_exceed_wall_bytes"
    if (wall_bytes is not None
            and abs(wall * GIB - wall_bytes) > MEASUREMENT_BYTE_TOLERANCE):
        return "wall_bytes_mismatch"
    if (safe_bytes is not None
            and abs(safe * GIB - safe_bytes) > MEASUREMENT_BYTE_TOLERANCE):
        return "safe_budget_bytes_mismatch"
    return None


def _budget_status(safe_budget, basis: str) -> tuple[str, str | None]:
    if safe_budget is None:
        return "unknown", "no_current_budget"
    if safe_budget <= 0:
        return "insufficient", "insufficient_memory"
    return basis, None


def limits(machine, measured: dict | None = None, *, sharded: bool = False,
           backend: str | None = None) -> dict:
    """Analytic memory limits from detect facts — no engine, no model load.

    Mirrors the wall available from engine-free facts: CUDA → a single device's VRAM by default (the
    shipped engine reads device 0 only and does not shard a model across GPUs; pass
    ``sharded=True`` for a future engine that does, which sums VRAM across all cards), Apple → a
    current wall is unknown until a live measurement is supplied, CPU → physical RAM. ``total_gb``
    always reports the
    true physical total across all cards; ``wall_gb`` is the governable budget. The safe budget
    is the wall minus ARA's margin. Shape is compatible with the limits dict the engines return.

    *backend* overrides the machine's detected backend for an explicit analytic engine choice
    (for example, the CPU fallback on a CUDA host). Pure: the heuristic never touches a database.
    The CALLER may pass *measured* — a stored
    calibration dict carrying ``wall_gb``/``safe_budget_gb`` from the engine's own ``safe_limits``.
    When it holds a usable wall, those measured numbers replace the analytic value and the result
    is labelled ``basis="measured"`` / ``calibrated=True``. Otherwise Apple is ``unknown`` and
    CPU/CUDA remain ``estimated``. The label always matches the data source.
    """
    selected_backend = backend or machine.backend
    if selected_backend == "cuda":
        per_device = machine.accel.vram_gb
        count = machine.accel.count or 1
        total = per_device * count if per_device is not None else None
        wall = total if sharded else per_device
        device = machine.accel.name
    else:
        total = machine.ram_total_gb
        device = machine.chip
        wall = None if selected_backend == "apple" else total
    if not _nonnegative_number(total):
        total = None
    if not _nonnegative_number(wall):
        wall = None
    safe_budget = max(wall - MARGIN_GB, 0.0) if wall is not None else None
    basis = "unknown" if selected_backend == "apple" else "estimated"
    budget_status, budget_reason = _budget_status(safe_budget, basis)
    measurement_reason = (
        measurement_validation_error(measured) if measured is not None else None)
    out = {
        "device": device,
        "physical_memory_bytes": getattr(machine, "physical_memory_bytes", None),
        "total_gb": total,
        "wall_gb": wall,
        "safe_budget_gb": safe_budget,
        "margin_gb": MARGIN_GB,
        "headroom_gb": None,          # a live quantity — belongs to detect/status, not the estimate
        "overhead_gb": None,          # measured cold-start overhead is characterize's job
        "swap_free_gb": machine.swap_gb,
        "calibrated": False,
        "calibrated_at": None,
        "basis": basis,
        "budget_status": budget_status,
        "budget_reason": budget_reason,
        "measurement_status": (
            "absent" if measured is None else "invalid" if measurement_reason else "valid"),
        "measurement_reason": measurement_reason,
    }
    # A real measurement for this machine + engine wins over the heuristic — but only when it
    # actually carries a wall (older/partial calibration rows fall back to the estimate honestly).
    measured_wall = (measured or {}).get("wall_gb")
    if measured is not None and measurement_reason is None:
        if wall is not None:
            out["estimated_wall_gb"] = wall
            out["estimated_safe_budget_gb"] = safe_budget
        out["wall_gb"] = measured_wall
        out["safe_budget_gb"] = (measured or {}).get("safe_budget_gb")
        out["calibrated"] = True
        out["calibrated_at"] = (measured or {}).get("calibrated_at")
        out["basis"] = "measured"
        out["budget_status"], out["budget_reason"] = _budget_status(
            out["safe_budget_gb"], out["basis"])
    return out


def model_fit(limits_dict: dict, meta: dict, weights_gb: float | None) -> dict:
    """Does *meta*'s model fit the estimated budget, and what context does it support?

    Engine-free: the weights footprint is estimated by the caller (≈ on-disk size); the KV growth
    is the analytic fp16 slope from the model's architecture. ``binding`` reports what limits the
    context — ``"context_window"`` (the budget covers the model's whole window) or ``"memory"``
    (the budget binds first) — or None when the slope can't be estimated. ``fits`` is False only
    when known weights equal or exceed the budget. It is None when the fit is unknowable:
    ``reason="no_current_budget"`` when the read-only seam has no admissible current budget, or
    ``reason="size_unknown"`` when the model footprint is unavailable. A model whose known
    weights fit but whose KV slope cannot be derived remains ``fits=True`` with
    ``reason="architecture_unknown"``.

    Units: *weights_gb* arrives in DECIMAL GB (on-disk bytes / 1e9 — the callers pass catalog /
    Hub sizes) while the budget and KV slope are binary GiB, so the weights are converted here
    before any comparison; the returned ``weights_gb`` is the converted GiB in-memory footprint.
    Comparing the raw decimal figure against a GiB budget overstated the weights term ~7.4% and
    skewed the decode ceiling. Slug 2026-07-02-analytic-units-gib.
    """
    budget = limits_dict["safe_budget_gb"]
    weights_gib = weights_gb * 1e9 / GIB if weights_gb is not None else None
    if budget is None:
        fits = None
        reason = "no_current_budget"
    elif not _nonnegative_number(budget):
        fits = None
        reason = "invalid_budget"
    elif budget == 0:
        fits = None
        reason = "insufficient_memory"
    elif weights_gib is None:
        fits = None
        reason = "size_unknown"
    elif weights_gib < budget:
        fits = True
        reason = None
    else:
        fits = False
        reason = "exceeds_safe_budget"
    slope = ramp.analytic_kv_slope_gb_per_k(meta.get("n_layers"), meta.get("kv_heads"),
                                            meta.get("head_dim"))
    est_context, binding = None, None
    if fits and slope:
        est_context, binding = ramp.decode_ceiling(
            weights_gib, slope, budget, max_context=meta.get("max_context"))
    elif fits:
        reason = "architecture_unknown"
    return {
        "weights_gb": weights_gib,
        "fits": fits,
        "est_context": est_context,
        "max_context": meta.get("max_context"),
        "binding": binding,
        "reason": reason,
    }
