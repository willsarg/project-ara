"""Methodology invariant matrix — the safe-context math across realistic model shapes.

100% line+branch coverage proved every line *runs*; it did not prove the methodology is
*correct* across the range of real models. A small-window model (SmolLM-135M, window 2048) once
returned "couldn't fit" despite trivially fitting — invisible to coverage because the synthetic
unit data never included a window below the 2nd schedule rung.

These tests assert the *invariants* that must hold for ANY (window, slope, budget), parametrized
over the real spread of context windows. They drive the engine-agnostic ``driver`` with a linear
synthetic engine, so a regression in the scheduling/fit/cap math is caught regardless of which
lines happen to execute.
"""
from __future__ import annotations

import pytest

from ara.contracts import driver

BUDGET = 15.0          # safe budget (GB)
REF = 2.0              # live OS baseline (GB), added back at solve time
SCHEDULE = [2000, 4000, 8000, 16000, 32000, 65536, 131072]

# The real spread: tiny windows (below the 2nd rung), mid, large, and unbounded.
WINDOWS = [512, 1024, 2048, 4096, 8192, 40960, 131072, None]


def _drive(*, max_context, intercept, slope_per_k):
    """Run the driver with a linear engine; return (result, contexts_probed).

    ``intercept``/``slope_per_k`` describe the model's DELTA footprint over the OS baseline;
    the absolute footprint at a context is ``REF + intercept + slope·(ctx/1000)``.
    """
    seen: list[int] = []

    def measure(model, ctx):
        seen.append(ctx)
        return {"context": ctx, "mem_gb": intercept + slope_per_k * (ctx / 1000)}

    est = {"base_gb": REF + intercept, "ref_baseline_gb": REF,
           "slope_gb_per_k": slope_per_k, "budget_gb": BUDGET, "max_context": max_context}
    result = driver.characterize("m", preflight=lambda m: est, measure=measure,
                                 schedule=list(SCHEDULE))
    return result, seen


def _absolute(intercept, slope_per_k, ctx):
    return REF + intercept + slope_per_k * (ctx / 1000)


# --------------------------------------------------------------------------- #
# A fitting model ALWAYS yields a usable ceiling (the invariant the bug broke).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("max_context", WINDOWS)
def test_fitting_model_reports_a_usable_ceiling(max_context):
    # Featherweight model (0.5 GB base, 0.02 GB/1k): fits comfortably at any window.
    r, _ = _drive(max_context=max_context, intercept=0.5, slope_per_k=0.02)
    assert r["safe_context"] is not None and r["safe_context"] > 0
    if max_context is not None:
        assert r["safe_context"] <= max_context          # never above the window
        # tiny slope → memory allows far past the window → window-bound
        assert r["binding"] == "context_window"
        assert r["safe_context"] == max_context


@pytest.mark.parametrize("max_context", WINDOWS)
def test_ceiling_never_exceeds_window(max_context):
    # Heavier model: whatever the math says, the ceiling can never exceed the trained window.
    r, _ = _drive(max_context=max_context, intercept=1.0, slope_per_k=0.5)
    if r["safe_context"] is not None and max_context is not None:
        assert r["safe_context"] <= max_context


@pytest.mark.parametrize("max_context", WINDOWS)
def test_never_probes_a_context_that_breaches_budget(max_context):
    # L1 must never dispatch a probe whose predicted footprint reaches the budget.
    _, seen = _drive(max_context=max_context, intercept=1.0, slope_per_k=1.0)
    assert all(_absolute(1.0, 1.0, c) < BUDGET for c in seen)


# --------------------------------------------------------------------------- #
# Binding is honest: memory when memory binds first, window when the window does.
# --------------------------------------------------------------------------- #
def test_memory_bound_when_window_is_huge():
    # Steep slope + huge window → memory is the limit, well below the window.
    r, _ = _drive(max_context=131072, intercept=1.0, slope_per_k=1.0)
    assert r["binding"] == "memory"
    assert r["safe_context"] is not None and r["safe_context"] < 131072


def test_window_bound_when_window_is_small():
    # Gentle slope + small window → the window is the limit, memory has headroom to spare.
    r, _ = _drive(max_context=2048, intercept=1.0, slope_per_k=0.05)
    assert r["binding"] == "context_window" and r["safe_context"] == 2048


# --------------------------------------------------------------------------- #
# An oversized model is honestly refused — no ceiling, and never probed (L1).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("max_context", [2048, 8192, None])
def test_oversized_model_reports_no_ceiling_and_never_probes(max_context):
    # Base alone (REF + 20) already over the 15 GB budget → can't load → refuse before probing.
    r, seen = _drive(max_context=max_context, intercept=20.0, slope_per_k=0.1)
    assert r["safe_context"] is None
    assert seen == []          # L1 prevented every dispatch — nothing was loaded
