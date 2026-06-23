# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The ramp contract — linear fit of memory vs context + safe-ceiling solve.

Pure math, no engine. Mirrors wmx-suite's proven methodology (least-squares
y = a + b·x with x in thousands of tokens, then solve for the context where
memory meets the safe budget), moved up into ARA so every ramp-class backend
shares one comparable methodology.
"""
from __future__ import annotations

import pytest

from ara.contracts import ramp


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
def test_fit_recovers_a_clean_line():
    # memory = 5 + 1.0 * (ctx/1000): intercept 5 GB, slope 1 GB per 1k tokens
    f = ramp.fit([(1000, 6.0), (2000, 7.0), (3000, 8.0)])
    assert f.intercept_gb == pytest.approx(5.0)
    assert f.slope_gb_per_k == pytest.approx(1.0)
    assert f.r2 == pytest.approx(1.0)
    assert f.n_points == 3


def test_fit_reports_imperfect_r2_for_noisy_points():
    f = ramp.fit([(1000, 6.0), (2000, 7.2), (3000, 7.9)])
    assert f.slope_gb_per_k > 0
    assert 0.0 <= f.r2 < 1.0


def test_fit_needs_at_least_two_points():
    with pytest.raises(ramp.RampError, match="at least two"):
        ramp.fit([(1000, 6.0)])


def test_fit_needs_distinct_contexts():
    # all measurements at one context → no slope is determinable
    with pytest.raises(ramp.RampError, match="distinct"):
        ramp.fit([(1000, 6.0), (1000, 7.0)])


# --------------------------------------------------------------------------- #
# safe_ceiling
# --------------------------------------------------------------------------- #
def test_safe_ceiling_solves_for_budget():
    f = ramp.Fit(intercept_gb=5.0, slope_gb_per_k=1.0, r2=1.0, n_points=3)
    # headroom 15-5 = 10 GB; 10/(1 GB/1k) = 10k, minus 1 token to stay STRICTLY under budget
    # (predicted memory at 10000 would equal 15.0 == the budget, which every gate treats as unsafe)
    assert ramp.safe_ceiling(f, budget_gb=15.0) == 9_999


def test_safe_ceiling_truncates_to_int_tokens():
    f = ramp.Fit(intercept_gb=5.0, slope_gb_per_k=3.0, r2=1.0, n_points=3)
    # headroom 10 / 3 = 3.333k → 3333 tokens (floor)
    assert ramp.safe_ceiling(f, budget_gb=15.0) == 3333


def test_safe_ceiling_zero_when_model_base_exceeds_budget():
    f = ramp.Fit(intercept_gb=20.0, slope_gb_per_k=1.0, r2=1.0, n_points=3)
    # base already over budget → won't fit even at minimal context
    assert ramp.safe_ceiling(f, budget_gb=15.0) == 0


def test_safe_ceiling_subtracts_live_ref_baseline():
    # fit is the model DELTA (intercept 5 = model base); live OS baseline 8 GB eats headroom
    f = ramp.Fit(intercept_gb=5.0, slope_gb_per_k=1.0, r2=1.0, n_points=3)
    # headroom = 36 - 8 (ref) - 5 (model) = 23 → 23k, minus 1 token to stay strictly under budget
    assert ramp.safe_ceiling(f, budget_gb=36.0, ref_baseline_gb=8.0) == 22_999


def test_run_threads_ref_baseline_into_ceiling():
    # delta points: model base 5, slope 1; ref baseline 8 → (36-8-5)/1 = 23k, −1 strictly under
    res = ramp.run(_linear_measure(5.0, 1.0), schedule=[2000, 4000, 8000],
                   base_gb=13.0, slope_gb_per_k=1.0, budget_gb=36.0, ref_baseline_gb=8.0)
    assert res.safe_context == 22_999


def test_safe_ceiling_none_when_no_measurable_growth():
    flat = ramp.Fit(intercept_gb=5.0, slope_gb_per_k=0.0, r2=1.0, n_points=3)
    assert ramp.safe_ceiling(flat, budget_gb=15.0) is None
    falling = ramp.Fit(intercept_gb=5.0, slope_gb_per_k=-0.2, r2=1.0, n_points=3)
    assert ramp.safe_ceiling(falling, budget_gb=15.0) is None


# --------------------------------------------------------------------------- #
# predict-before-probe — the crash-prevention gate (mirrors wmx probe.py)
# --------------------------------------------------------------------------- #
def test_predict_gb_is_base_plus_slope_times_kilotokens():
    # base 6 GB + 0.5 GB/1k * 8k tokens = 10 GB
    assert ramp.predict_gb(6.0, 0.5, 8000) == pytest.approx(10.0)


def test_would_breach_true_at_or_over_budget():
    # predicted 6 + 0.5*8 = 10 >= budget 10 → unsafe (never probe at/over the wall)
    assert ramp.would_breach(6.0, 0.5, 8000, budget_gb=10.0) is True


def test_would_breach_false_with_headroom():
    assert ramp.would_breach(6.0, 0.5, 8000, budget_gb=12.0) is False


# --------------------------------------------------------------------------- #
# plan_next — safe inductive escalation (never returns an unsafe context)
# --------------------------------------------------------------------------- #
SCHED = [2000, 4000, 8000, 16000, 32000]


def test_plan_next_starts_at_lowest_unmeasured_when_safe():
    nxt = ramp.plan_next(SCHED, measured=set(), base_gb=6.0, slope_gb_per_k=0.1, budget_gb=36.0)
    assert nxt == 2000


def test_plan_next_skips_already_measured():
    nxt = ramp.plan_next(SCHED, measured={2000, 4000}, base_gb=6.0,
                         slope_gb_per_k=0.1, budget_gb=36.0)
    assert nxt == 8000


def test_plan_next_stops_when_next_rung_would_breach():
    # base 30 + 0.5/1k: even 16k → 30+8=38 >= 36 budget → stop rather than probe it
    nxt = ramp.plan_next(SCHED, measured={2000, 4000, 8000}, base_gb=30.0,
                         slope_gb_per_k=0.5, budget_gb=36.0)
    assert nxt is None


def test_plan_next_none_when_all_measured():
    nxt = ramp.plan_next(SCHED, measured=set(SCHED), base_gb=6.0,
                         slope_gb_per_k=0.1, budget_gb=36.0)
    assert nxt is None


# --------------------------------------------------------------------------- #
# run — the safe ramp driver (injectable measure_fn, duck-typed Measurement)
# --------------------------------------------------------------------------- #
class FakeM:
    """Stand-in for a worker Measurement (duck-typed: .refused, .mem_gb)."""
    def __init__(self, mem_gb=None, refused=False):
        self.mem_gb, self.refused, self.reason = mem_gb, refused, None


def _linear_measure(intercept, slope_per_k):
    # mem = intercept + slope_per_k * (ctx/1000)
    return lambda ctx: FakeM(mem_gb=intercept + slope_per_k * (ctx / 1000))


def test_run_collects_points_fits_and_solves():
    res = ramp.run(_linear_measure(5.0, 1.0), schedule=[2000, 4000, 8000],
                   base_gb=5.0, slope_gb_per_k=1.0, budget_gb=36.0)
    assert len(res.points) == 3
    # fitted intercept 5, slope 1 → ceiling (36-5)/1 = 31k
    assert res.safe_context == 30_999
    assert res.stopped_reason == "ok"


def test_run_returns_none_when_scheduler_refuses_immediately():
    # base already over budget → plan_next gives nothing, no measurement attempted
    calls = []
    res = ramp.run(lambda c: calls.append(c) or FakeM(mem_gb=1.0),
                   schedule=[2000, 4000], base_gb=40.0, slope_gb_per_k=1.0, budget_gb=36.0)
    assert calls == [] and res.points == [] and res.safe_context is None
    assert res.stopped_reason == "insufficient points"


def test_run_bisects_below_abort_instead_of_extrapolating_past_it():
    # A refusal at 8000 is HARD evidence the ceiling is below it — extrapolating the 2000/4000
    # fit to 31k would claim unsafe contexts are safe (the bug this guards against). The ramp
    # must bracket [4000, 8000) and report a confirmed-safe context strictly under the abort.
    def measure(ctx):
        return FakeM(refused=True) if ctx >= 8000 else FakeM(mem_gb=5.0 + ctx / 1000)
    res = ramp.run(measure, schedule=[2000, 4000, 8000, 16000],
                   base_gb=5.0, slope_gb_per_k=1.0, budget_gb=36.0)
    assert res.binding == "memory"
    assert 4000 <= res.safe_context < 8000      # bisected into the bracket, never past it
    assert res.safe_context != 30_999           # the old extrapolate-past-abort answer is gone
    assert res.safe_context in {c for c, _ in res.points}   # the ceiling is a measured context


def test_run_finds_the_wall_when_growth_is_super_linear():
    # KV-style linear up to 4000, then a prefill-style wall: anything >= 5000 aborts. Bisection
    # locates the wall (~5000) far better than the coarse schedule (which jumps 4000 → 8000).
    def measure(ctx):
        return FakeM(refused=True) if ctx >= 5000 else FakeM(mem_gb=5.0 + ctx / 1000)
    res = ramp.run(measure, schedule=[2000, 4000, 8000, 16000],
                   base_gb=5.0, slope_gb_per_k=1.0, budget_gb=36.0)
    assert 4000 <= res.safe_context < 5000      # pinned just below the real wall
    assert res.binding == "memory"


def test_run_reports_single_safe_point_when_gate_stops_without_abort():
    # base 33 + slope 1: 2000 → 35 < 36 (probe it), 4000 → 37 ≥ 36 (a-priori gate stops, no abort).
    # One measured-safe point is a real lower bound — report it, don't discard it as None (#45).
    res = ramp.run(_linear_measure(33.0, 1.0), schedule=[2000, 4000],
                   base_gb=33.0, slope_gb_per_k=1.0, budget_gb=36.0)
    assert res.safe_context == 2000 and res.binding == "memory"
    assert res.stopped_reason == "single safe point"


def test_run_none_when_smallest_context_aborts_with_no_safe_point():
    # first rung refused and nothing was ever safe → None, and we do NOT keep reloading the model
    # at smaller contexts (a model that can't do the smallest rung is reported as not fitting).
    calls = []

    def measure(ctx):
        calls.append(ctx)
        return FakeM(refused=True)

    res = ramp.run(measure, schedule=[2000, 4000], base_gb=5.0,
                   slope_gb_per_k=1.0, budget_gb=36.0)
    assert res.safe_context is None and res.points == []
    assert calls == [2000]               # bisection is NOT attempted without a safe lower bound


def test_run_bisection_is_bounded():
    # every midpoint below 8000 is safe → bisection keeps climbing, but the probe count is capped
    # (each probe is a full model load); assert it doesn't run away.
    calls = []

    def measure(ctx):
        calls.append(ctx)
        return FakeM(refused=True) if ctx >= 8000 else FakeM(mem_gb=5.0 + ctx / 1000)

    ramp.run(measure, schedule=[2000, 4000, 8000], base_gb=5.0,
             slope_gb_per_k=1.0, budget_gb=36.0)
    # 3 schedule probes (2000/4000/8000) + at most BISECT_MAX_STEPS bisection probes
    assert len(calls) <= 3 + ramp.BISECT_MAX_STEPS


def test_run_refines_gate_from_measurements_to_escalate_safely():
    # a-priori slope is STEEP (2.0) → a static gate would stop escalating after ~8000.
    # The real measured slope is shallow, so once ≥2 points exist the gate uses the refined
    # fit and safely climbs higher — mirroring wmx (more data → more accurate gate).
    seen = []

    def measure(ctx):
        seen.append(ctx)
        return FakeM(mem_gb=1.0 + 0.1 * (ctx / 1000))   # shallow real growth

    ramp.run(measure, schedule=[2000, 4000, 8000, 16000, 32000],
             base_gb=10.0, slope_gb_per_k=2.0, budget_gb=36.0)
    assert 32000 in seen          # refined fit let escalation pass the steep a-priori gate


def test_run_caps_ceiling_at_model_context_window():
    # memory would allow ~31k, but the model's window is 20k → report 20k, window-bound
    res = ramp.run(_linear_measure(5.0, 1.0), schedule=[2000, 4000, 8000],
                   base_gb=5.0, slope_gb_per_k=1.0, budget_gb=36.0, max_context=20000)
    assert res.safe_context == 20000
    assert res.binding == "context_window"


def test_run_memory_bound_when_ceiling_below_window():
    res = ramp.run(_linear_measure(5.0, 1.0), schedule=[2000, 4000, 8000],
                   base_gb=5.0, slope_gb_per_k=1.0, budget_gb=36.0, max_context=100000)
    assert res.safe_context == 30999 and res.binding == "memory"


def test_run_none_when_fewer_than_two_points():
    res = ramp.run(lambda c: FakeM(refused=True), schedule=[2000, 4000],
                   base_gb=5.0, slope_gb_per_k=1.0, budget_gb=36.0)
    assert res.points == [] and res.safe_context is None and res.fit is None


# --------------------------------------------------------------------------- #
# analytic_kv_slope_gb_per_k — decode slope from model metadata
# --------------------------------------------------------------------------- #
def test_analytic_kv_slope_known_numbers():
    # 2 * 32 layers * 8 kv_heads * 128 head_dim * 2 bytes / 1e9 * 1000 = ~0.13107 GB/1k
    slope = ramp.analytic_kv_slope_gb_per_k(32, 8, 128)
    expected = 2 * 32 * 8 * 128 * 2 / 1e9 * 1000
    assert slope == pytest.approx(expected)


def test_analytic_kv_slope_none_when_n_layers_is_none():
    assert ramp.analytic_kv_slope_gb_per_k(None, 8, 128) is None


def test_analytic_kv_slope_none_when_kv_heads_is_none():
    assert ramp.analytic_kv_slope_gb_per_k(32, None, 128) is None


def test_analytic_kv_slope_none_when_head_dim_is_none():
    assert ramp.analytic_kv_slope_gb_per_k(32, 8, None) is None


def test_analytic_kv_slope_none_when_n_layers_is_zero():
    assert ramp.analytic_kv_slope_gb_per_k(0, 8, 128) is None


def test_analytic_kv_slope_none_when_kv_heads_is_zero():
    assert ramp.analytic_kv_slope_gb_per_k(32, 0, 128) is None


def test_analytic_kv_slope_none_when_head_dim_is_zero():
    assert ramp.analytic_kv_slope_gb_per_k(32, 8, 0) is None


# --------------------------------------------------------------------------- #
# decode_ceiling — analytic decode-safe context ceiling
# --------------------------------------------------------------------------- #
def test_decode_ceiling_larger_than_prefill_for_shallower_kv_slope():
    # Prefill: intercept=5, slope=2.0 → safe_ceiling budget=36 → (36-5)/2 = 15.5k
    # Decode: intercept=5, kv_slope=0.5 → (36-5)/0.5 = 62k, much bigger
    prefill_f = ramp.Fit(intercept_gb=5.0, slope_gb_per_k=2.0, r2=1.0, n_points=3)
    prefill_c = ramp.safe_ceiling(prefill_f, budget_gb=36.0)
    kv_slope = 0.5
    decode_c, binding = ramp.decode_ceiling(5.0, kv_slope, 36.0)
    assert decode_c is not None and prefill_c is not None
    assert decode_c > prefill_c
    assert binding == "memory"


def test_decode_ceiling_none_when_kv_slope_zero():
    c, binding = ramp.decode_ceiling(5.0, 0.0, 36.0)
    assert c is None and binding == "memory"


def test_decode_ceiling_none_when_kv_slope_negative():
    c, binding = ramp.decode_ceiling(5.0, -0.5, 36.0)
    assert c is None and binding == "memory"


def test_decode_ceiling_capped_at_max_context():
    # kv_slope so small the analytic ceiling would far exceed 20k
    c, binding = ramp.decode_ceiling(5.0, 0.01, 36.0, max_context=20000)
    assert c == 20000 and binding == "context_window"


def test_decode_ceiling_memory_bound_when_below_max_context():
    # kv_slope steep enough that decode ceiling is below max_context
    c, binding = ramp.decode_ceiling(5.0, 5.0, 36.0, max_context=100000)
    assert c is not None and c < 100000
    assert binding == "memory"
