# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Shared RULE #1 safety gating for the Kokoro benchmark workers.

Each Kokoro worker runs its whole sweep in a single process, so it needs:
  * a settled-baseline pre-flight that reserves headroom for the model load and is run
    BEFORE the heavy MLX/Metal import (so the import itself can't cross the wall first), and
  * a fresh per-step check that treats a read failure as UNSAFE (never trusting a stale
    startup snapshot).

This is the reactive Kokoro analogue of the predictive gate the LLM/embeddings paths use.
It is appropriate because Kokoro's footprint is small and nearly static, but unlike the
old per-worker checks it (a) reserves model-load headroom, (b) settles the baseline, and
(c) never falls back to a stale value on a failed read. For the concurrent batch worker it
also predicts the per-call transient × concurrency before running a rung.
"""
from __future__ import annotations

from . import system

# Headroom (GiB) reserved for loading Kokoro-82M (bf16 weights + Metal init) before a run is
# cleared to start. Conservative for an ~82M-param model.
MODEL_WEIGHT_EST_GB = 0.5


def preflight(margin_gb: float) -> tuple[float, float, bool]:
    """Settled-baseline pre-flight with model-load headroom.

    Returns (threshold, baseline, safe). ``safe`` is False when the settled baseline plus
    the model-load headroom would reach the safe threshold — the caller MUST abort without
    loading the model. Call this BEFORE importing mlx/kokoro so the Metal init never runs
    on an already-pressured machine.
    """
    limits = system.read_limits()
    threshold = limits.safe_threshold_gb(margin_gb)
    baseline = system.sample_settled_baseline()
    safe = baseline + MODEL_WEIGHT_EST_GB < threshold
    return threshold, baseline, safe


def over_threshold(threshold: float) -> bool:
    """Fresh OS-wired read for a per-step safeguard. A read failure is treated as UNSAFE
    (returns True) rather than trusting a stale snapshot — the safe direction."""
    try:
        return system.wired_gb() >= threshold
    except Exception:
        return True


def predicted_concurrent_peak(current_wired_gb: float, per_call_gb: float,
                              batch_size: int) -> float:
    """Predicted OS-wired peak for a concurrent batch rung: current residency plus the
    measured worst-case per-call transient times the concurrency. Used by the batch worker
    to refuse a rung whose transient spike would breach the wall, since the between-rung
    check alone cannot catch an in-rung concurrent spike."""
    return current_wired_gb + per_call_gb * max(0, batch_size)
