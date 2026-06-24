# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Engine-free analytic capability estimate — the reasoning behind ``ara profile``.

ARA's *analytic* layer: it mirrors the engine memory wall from ``detect`` facts (no engine, no
model load) and checks whether a model's weights + context window fit the estimated budget.
``characterize`` later *measures* the real ceiling; this only predicts it. Everything here is an
estimate and is labelled as such. See Spec 2026-06-23-capability-pipeline.
"""
from __future__ import annotations

from ara.contracts import ramp

# ARA policy (not engine-measured): the analytic safety margin below the wall, and Apple's
# unified-memory working-set fraction (Metal's recommendedMaxWorkingSetSize ≈ 75% of RAM).
MARGIN_GB = 2.0
APPLE_WORKING_SET = 0.75


def limits(machine, measured: dict | None = None) -> dict:
    """Analytic memory limits from detect facts — no engine, no model load.

    Mirrors the wall each backend would read: CUDA → total VRAM, Apple → a working-set fraction
    of unified RAM, CPU → physical RAM. The safe budget is the wall minus ARA's margin. Shape is
    compatible with the limits dict the engines return.

    Pure: the heuristic never touches a database. The CALLER may pass *measured* — a stored
    calibration dict carrying ``wall_gb``/``safe_budget_gb`` from the engine's own ``safe_limits``.
    When it holds a usable wall, those measured numbers replace the heuristic and the result is
    labelled ``basis="measured"`` / ``calibrated=True`` (the heuristic value is kept alongside as
    ``estimated_wall_gb``/``estimated_safe_budget_gb`` so the correction is visible). Otherwise the
    result is the honest heuristic: ``basis="estimated"`` / ``calibrated=False``. The label always
    matches the data source — a heuristic is never reported as measured.
    """
    if machine.backend == "cuda":
        vram = machine.accel.vram_gb
        total = vram * (machine.accel.count or 1) if vram is not None else None
        wall = total
        device = machine.accel.name
    else:
        total = machine.ram_total_gb
        device = machine.chip
        wall = total * APPLE_WORKING_SET if (machine.backend == "apple" and total is not None) \
            else total
    safe_budget = wall - MARGIN_GB if wall is not None else None
    out = {
        "device": device,
        "total_gb": total,
        "wall_gb": wall,
        "safe_budget_gb": safe_budget,
        "margin_gb": MARGIN_GB,
        "headroom_gb": None,          # a live quantity — belongs to detect/status, not the estimate
        "overhead_gb": None,          # measured cold-start overhead is characterize's job
        "swap_free_gb": machine.swap_gb,
        "calibrated": False,
        "calibrated_at": None,
        "basis": "estimated",
    }
    # A real measurement for this machine + engine wins over the heuristic — but only when it
    # actually carries a wall (older/partial calibration rows fall back to the estimate honestly).
    measured_wall = (measured or {}).get("wall_gb")
    if measured_wall is not None:
        out["estimated_wall_gb"] = wall              # keep the heuristic visible for comparison
        out["estimated_safe_budget_gb"] = safe_budget
        out["wall_gb"] = measured_wall
        out["safe_budget_gb"] = (measured or {}).get("safe_budget_gb")
        out["calibrated"] = True
        out["calibrated_at"] = (measured or {}).get("calibrated_at")
        out["basis"] = "measured"
    return out


def model_fit(limits_dict: dict, meta: dict, weights_gb: float | None) -> dict:
    """Does *meta*'s model fit the estimated budget, and what context does it support?

    Engine-free: the weights footprint is estimated by the caller (≈ on-disk size); the KV growth
    is the analytic fp16 slope from the model's architecture. ``binding`` reports what limits the
    context — ``"context_window"`` (the budget covers the model's whole window) or ``"memory"``
    (the budget binds first) — or None when the slope can't be estimated. ``fits`` is False when
    the weights alone exceed the budget.
    """
    budget = limits_dict["safe_budget_gb"]
    fits = weights_gb is not None and budget is not None and weights_gb < budget
    slope = ramp.analytic_kv_slope_gb_per_k(meta.get("n_layers"), meta.get("kv_heads"),
                                            meta.get("head_dim"))
    est_context, binding = None, None
    if fits and slope:
        est_context, binding = ramp.decode_ceiling(
            weights_gb, slope, budget, max_context=meta.get("max_context"))
    return {
        "weights_gb": weights_gb,
        "fits": fits,
        "est_context": est_context,
        "max_context": meta.get("max_context"),
        "binding": binding,
    }
