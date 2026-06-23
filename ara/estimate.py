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


def limits(machine) -> dict:
    """Analytic memory limits from detect facts — no engine, no model load.

    Mirrors the wall each backend would read: CUDA → total VRAM, Apple → a working-set fraction
    of unified RAM, CPU → physical RAM. The safe budget is the wall minus ARA's margin. Shape is
    compatible with the limits dict the engines return, but ``calibrated`` is always False and
    ``basis`` is ``"estimated"`` — this is predicted, not measured.
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
    return {
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
