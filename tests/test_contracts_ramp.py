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
    # headroom = 15 - 5 = 10 GB; 10 GB / (1 GB/1k) = 10k tokens
    assert ramp.safe_ceiling(f, budget_gb=15.0) == 10_000


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
    # headroom = 36 - 8 (ref) - 5 (model) = 23 → 23k tokens
    assert ramp.safe_ceiling(f, budget_gb=36.0, ref_baseline_gb=8.0) == 23_000


def test_run_threads_ref_baseline_into_ceiling():
    # delta points: model base 5, slope 1; ref baseline 8 → ceiling (36-8-5)/1 = 23k
    res = ramp.run(_linear_measure(5.0, 1.0), schedule=[2000, 4000, 8000],
                   base_gb=13.0, slope_gb_per_k=1.0, budget_gb=36.0, ref_baseline_gb=8.0)
    assert res.safe_context == 23_000


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
    assert res.safe_context == 31_000
    assert res.stopped_reason == "ok"


def test_run_returns_none_when_scheduler_refuses_immediately():
    # base already over budget → plan_next gives nothing, no measurement attempted
    calls = []
    res = ramp.run(lambda c: calls.append(c) or FakeM(mem_gb=1.0),
                   schedule=[2000, 4000], base_gb=40.0, slope_gb_per_k=1.0, budget_gb=36.0)
    assert calls == [] and res.points == [] and res.safe_context is None
    assert res.stopped_reason == "insufficient points"


def test_run_stops_on_engine_refusal_but_uses_points_so_far():
    def measure(ctx):
        return FakeM(refused=True) if ctx >= 8000 else FakeM(mem_gb=5.0 + ctx / 1000)
    res = ramp.run(measure, schedule=[2000, 4000, 8000, 16000],
                   base_gb=5.0, slope_gb_per_k=1.0, budget_gb=36.0)
    assert len(res.points) == 2          # 2000 and 4000 measured; 8000 refused → stop
    assert res.safe_context == 31_000    # still fits from the two safe points


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


def test_run_none_when_fewer_than_two_points():
    res = ramp.run(lambda c: FakeM(refused=True), schedule=[2000, 4000],
                   base_gb=5.0, slope_gb_per_k=1.0, budget_gb=36.0)
    assert res.points == [] and res.safe_context is None and res.fit is None
