"""The ramp contract: fit memory-vs-context, solve for the safe context ceiling.

For hardware with a hard memory wall (Apple, CUDA, ROCm, …), a model's footprint grows
~linearly with context as the KV cache fills. ARA owns this methodology so the number means
the same thing on every ramp-class backend; engines only supply safe ``(context, memory)``
measurements. Mirrors wmx-suite's proven fit (least-squares ``y = a + b·x``, x in thousands
of tokens) and ceiling solve.
"""
from __future__ import annotations

from dataclasses import dataclass


class RampError(ValueError):
    """The measured points can't support a fit (too few, or no distinct contexts)."""


@dataclass(frozen=True)
class Fit:
    """A fitted memory curve: ``mem_gb = intercept_gb + slope_gb_per_k · (ctx/1000)``."""
    intercept_gb: float    # memory extrapolated to zero context (model + OS base)
    slope_gb_per_k: float  # GB added per 1000 tokens of context
    r2: float              # goodness of fit, 0..1
    n_points: int


def fit(points: list[tuple[int, float]]) -> Fit:
    """Least-squares fit of memory (GB) against context (tokens) from safe measurements.

    *points* are ``(context_tokens, mem_gb)``. Needs at least two points at two distinct
    contexts, else a line is undetermined — raises :class:`RampError`.
    """
    if len(points) < 2:
        raise RampError("need at least two measurements to fit a ramp")
    xs_k = [ctx / 1000 for ctx, _ in points]   # thousands of tokens
    ys = [mem for _, mem in points]
    n = len(points)
    mx = sum(xs_k) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs_k)
    if sxx == 0:
        raise RampError("need measurements at distinct contexts to fit a slope")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs_k, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs_k, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    return Fit(intercept_gb=intercept, slope_gb_per_k=slope, r2=r2, n_points=n)


def predict_gb(base_gb: float, slope_gb_per_k: float, ctx_tokens: int) -> float:
    """Conservative a-priori memory prediction at *ctx_tokens*: ``base + slope·(ctx/1000)``.

    Used *before* probing, from a model's pre-estimated base + slope (not a fitted line),
    so a rung is never attempted unless it's predicted to stay under budget.
    """
    return base_gb + slope_gb_per_k * (ctx_tokens / 1000)


def would_breach(base_gb: float, slope_gb_per_k: float, ctx_tokens: int,
                 budget_gb: float) -> bool:
    """Would probing at *ctx_tokens* reach/exceed the safe budget? (RULE #1 gate.)

    Mirrors wmx-suite's predict-before-probe abort: ``>=`` is unsafe — rounding *toward*
    the wall is how you crash, so the budget is a hard ceiling, never a target.
    """
    return predict_gb(base_gb, slope_gb_per_k, ctx_tokens) >= budget_gb


@dataclass(frozen=True)
class RampResult:
    """Outcome of a safe ramp: the ceiling (or None), the fit, and the points gathered.

    ``binding`` says what limits the ceiling — ``"memory"`` (the safe budget) or
    ``"context_window"`` (the model's own ``max_context``, which memory would otherwise exceed).
    """
    safe_context: int | None
    fit: Fit | None
    points: list[tuple[int, float]]
    stopped_reason: str
    binding: str = "memory"


# Bisection bounds for refining the ceiling between the highest safe context and a measured
# abort. Each probe is a full model load, so the step count is capped; the gap tolerance stops
# once the bracket is tight enough that more precision isn't worth another load.
BISECT_MIN_GAP = 256       # tokens — stop when the safe/abort bracket is this tight
BISECT_MAX_STEPS = 6       # hard cap on bisection probes


def _bisect_ceiling(measure_fn, lo: int, hi: int, points: list[tuple[int, float]]) -> int:
    """Refine the highest safe context in ``[lo, hi)``: *lo* is known-safe, *hi* a measured abort.

    Probes midpoints — each guarded by the engine's own L4/L5, since the abort proved the
    a-priori fit can't be trusted here. A safe midpoint raises the floor (and is recorded);
    an aborted one lowers the ceiling. Returns the highest confirmed-safe context. Bounded by
    :data:`BISECT_MIN_GAP` / :data:`BISECT_MAX_STEPS` (each probe is a full model load).
    """
    steps = 0
    while hi - lo > BISECT_MIN_GAP and steps < BISECT_MAX_STEPS:
        mid = (lo + hi) // 2
        steps += 1
        m = measure_fn(mid)
        if m.refused:
            hi = mid
        else:
            lo = mid
            points.append((mid, m.mem_gb))
    return lo


def run(measure_fn, schedule: list[int], base_gb: float, slope_gb_per_k: float,
        budget_gb: float, ref_baseline_gb: float = 0.0,
        max_context: int | None = None) -> RampResult:
    """Drive the safe ramp: schedule a rung (L1 gate), measure it, repeat, then fit + solve.

    *measure_fn(ctx)* returns a Measurement (duck-typed ``.refused`` / ``.mem_gb``) whose
    ``mem_gb`` is the model's DELTA at that context — the adapter wires it to the engine
    worker. The gate predicts the ABSOLUTE footprint from *base_gb* + *slope_gb_per_k*; the
    ceiling solve adds *ref_baseline_gb* to the fitted delta. Escalation only visits contexts
    :func:`plan_next` deems safe.

    Two regimes produce the ceiling. If no rung ever aborts, the model grows gently and the
    fitted line extrapolates the ceiling (capped at the model's window). If a rung **aborts**
    (engine L4/L5 veto, or ARA's L2), that abort is a hard wall — extrapolating past it would
    claim unsafe contexts are safe — so the ramp bisects ``[highest safe, abort)`` and reports
    the highest *confirmed-safe* context (always memory-bound). A single safe measurement with
    no abort is still a real lower bound and is reported as the ceiling, not discarded.
    """
    points: list[tuple[int, float]] = []
    measured: set[int] = set()
    aborted_at: int | None = None
    while True:
        # Gate inputs: the conservative a-priori estimate until we have ≥2 points, then the
        # REFINED fit (delta intercept + live ref_baseline → absolute). Refining as data
        # arrives lets escalation safely climb higher when growth is gentler than estimated —
        # and stop earlier when it's steeper. (The engine's L4 veto backstops either way.)
        if len(points) >= 2:
            f = fit(points)
            gate_base, gate_slope = ref_baseline_gb + f.intercept_gb, f.slope_gb_per_k
        else:
            gate_base, gate_slope = base_gb, slope_gb_per_k
        ctx = plan_next(schedule, measured, gate_base, gate_slope, budget_gb)
        if ctx is None:
            break
        m = measure_fn(ctx)
        measured.add(ctx)
        if m.refused:
            aborted_at = ctx
            break
        points.append((ctx, m.mem_gb))

    safe_points = [c for c, _ in points]

    # A rung aborted: the ceiling is below it. Bisect to pin the wall — but only with a safe
    # lower bound to anchor on. If even the smallest rung aborts, the model doesn't fit here;
    # report None rather than reloading it at ever-smaller contexts (RULE #1 + cost).
    if aborted_at is not None:
        if not safe_points:
            return RampResult(None, None, points, "aborted at smallest context")
        ceiling = _bisect_ceiling(measure_fn, max(safe_points), aborted_at, points)
        points.sort()
        f = fit(points) if len(points) >= 2 else None
        return RampResult(ceiling, f, points, "bracketed below abort", "memory")

    # No abort. With <2 points a line is undetermined, but a single safe measurement is still a
    # real lower bound — report it (None only if nothing was ever measured safely).
    if len(points) < 2:
        ceiling = max(safe_points) if safe_points else None
        reason = "single safe point" if safe_points else "insufficient points"
        return RampResult(ceiling, None, points, reason)

    # Gentle linear regime: fit + extrapolate to the budget, capped at the model's window.
    f = fit(points)
    ceiling = safe_ceiling(f, budget_gb, ref_baseline_gb)
    binding = "memory"
    # Honesty: a memory ceiling past the model's own context window is unusable — cap it,
    # and say the limit is the window, not memory.
    if ceiling is not None and max_context is not None and ceiling > max_context:
        ceiling, binding = max_context, "context_window"
    return RampResult(ceiling, f, points, "ok", binding)


def plan_next(schedule: list[int], measured, base_gb: float,
              slope_gb_per_k: float, budget_gb: float) -> int | None:
    """The next safe context to probe from *schedule* (ascending), or None to stop.

    The L1 crash-safety gate: returns the lowest not-yet-*measured* rung whose
    conservative prediction stays under budget. If that rung would breach, returns None
    (stop escalating) — so this never hands back a context that's unsafe to probe.
    """
    for ctx in schedule:
        if ctx in measured:
            continue
        if would_breach(base_gb, slope_gb_per_k, ctx, budget_gb):
            return None
        return ctx
    return None


def safe_ceiling(f: Fit, budget_gb: float, ref_baseline_gb: float = 0.0) -> int | None:
    """Largest context (tokens) whose predicted memory stays within *budget_gb*.

    The fit is on the model's DELTA over its own launch baseline (so ``intercept_gb`` is the
    model's footprint at context→0, free of ambient noise); *ref_baseline_gb* is the live OS
    baseline added back at solve time — mirroring wmx's ``ref_baseline + model_base + slope·c``.
    Returns ``None`` when there's no measurable growth (slope ≤ 0), ``0`` when the base already
    exceeds budget, else the floored token count.
    """
    if f.slope_gb_per_k <= 0:
        return None
    headroom = budget_gb - ref_baseline_gb - f.intercept_gb
    tokens = int(max(0.0, headroom / f.slope_gb_per_k) * 1000)
    # The budget is the line predicted peak must never reach: ``>=`` is unsafe everywhere else
    # (would_breach, plan_next, the L2/L4 gates). When headroom divides evenly, the floored token
    # count predicts *exactly* the budget — so step one token below to stay strictly under it.
    if tokens > 0 and predict_gb(ref_baseline_gb + f.intercept_gb, f.slope_gb_per_k, tokens) >= budget_gb:
        tokens -= 1
    return tokens
