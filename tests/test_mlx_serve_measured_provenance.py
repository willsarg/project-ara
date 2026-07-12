# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""wmx safety gate — measured-provenance slope (Rule #1).

When ARA serves a model it has CHARACTERIZED, the measured ramp already fit the real (smaller)
growth slope and certified the ceiling safe on this machine. Re-predicting that ceiling with the
conservative *a-priori* slope over-predicts and refuses a serve that characterize already proved
safe (Qwen3-0.6B at its measured 40960). ``safety_gate`` gains ``measured_slope_gb_per_k``: when
present it predicts with the MEASURED slope instead of the a-priori one — still adding the current
live baseline, still running the base-load check. Absent → behaviour is unchanged (a-priori).

Vendored wmx code is outside the coverage gate (omit list) but tested here directly — this is
Rule #1 logic. Slug: 2026-07-02-wmx-serve-measured-provenance-gate.
"""
from __future__ import annotations

from ara._engine_packages.mlx.ara_engine_mlx import measure_one


class _Info:
    """Minimal ModelInfo double: a fixed a-priori slope + weights, quantizable KV."""

    is_causal = True
    can_quantize_kv = True

    def __init__(self, weights_gb: float, apriori_slope: float):
        self.weights_gb = weights_gb
        self._apriori = apriori_slope

    def estimated_slope_gb_per_k(self, kv_bits=None) -> float:
        return self._apriori


class _Limits:
    def __init__(self, threshold_gb: float, wired_now_gb: float):
        self._threshold = threshold_gb
        self.wired_now_gb = wired_now_gb

    def safe_threshold_gb(self, margin_gb: float) -> float:
        return self._threshold


# a-priori 0.5 GB/1k refuses at 40k; the measured 0.21 GB/1k (Qwen3-0.6B, real fit) fits.
_INFO = _Info(weights_gb=1.0, apriori_slope=0.5)
_LIMITS = _Limits(threshold_gb=16.0, wired_now_gb=3.0)
_KW = dict(margin_gb=2.0, overhead_gb=1.0, live_base=3.0)


def test_apriori_slope_refuses_the_measured_ceiling():
    # Baseline: without provenance, the a-priori slope over-predicts and refuses at 40k.
    reason = measure_one.safety_gate(_INFO, _LIMITS, 40000, **_KW)
    assert reason is not None and "predicted" in reason


def test_measured_slope_admits_the_ceiling_the_apriori_refused():
    # Same ceiling, measured provenance: the real slope fits under budget → no refusal.
    reason = measure_one.safety_gate(_INFO, _LIMITS, 40000,
                                     measured_slope_gb_per_k=0.21, **_KW)
    assert reason is None


def test_measured_provenance_still_honours_the_live_baseline():
    # A busy machine (high live_base) at serve time still refuses even under measured provenance —
    # the measured slope replaces the a-priori slope, it does NOT bypass current conditions.
    busy = dict(_KW, live_base=10.0)
    reason = measure_one.safety_gate(_INFO, _LIMITS, 40000,
                                     measured_slope_gb_per_k=0.21, **busy)
    assert reason is not None and "predicted" in reason


def test_measured_provenance_still_runs_the_base_load_check():
    # The model must still fit to load; a measured slope never bypasses the base-estimate gate.
    heavy = _Info(weights_gb=20.0, apriori_slope=0.21)
    reason = measure_one.safety_gate(heavy, _LIMITS, 1000,
                                     measured_slope_gb_per_k=0.21, **_KW)
    assert reason is not None and "won't load" in reason


def test_none_measured_slope_is_byte_identical_to_apriori():
    # Explicit None (uncharacterized / --ctx path) → unchanged a-priori behaviour.
    a = measure_one.safety_gate(_INFO, _LIMITS, 40000, measured_slope_gb_per_k=None, **_KW)
    b = measure_one.safety_gate(_INFO, _LIMITS, 40000, **_KW)
    assert a == b and a is not None
