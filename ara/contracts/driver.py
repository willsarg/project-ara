"""The engine-agnostic thin-path driver: methodology in ARA, facts from the engine.

Every ramp-class backend (Apple, CPU/llama.cpp, CUDA, …) measures the same way — a no-load
preflight estimate, then a safe escalation up a context schedule, then a fit + ceiling solve.
That methodology lives here, once. A backend adapter supplies only what's engine-specific:

  * ``preflight(model) -> est``  — the no-load estimate ``{base_gb, slope_gb_per_k, budget_gb,
    max_context, ref_baseline_gb}`` (or ``{"error": ...}`` if the model can't be measured).
  * ``measure(model, ctx) -> dict`` — one raw worker reading (the canonical worker contract).
  * ``schedule`` — the context rungs to try (ascending).

The driver knows nothing about MLX, llama.cpp, or which env it's talking to — that's the point.
Crash-safety is layered: ``ramp.run`` gates each rung a-priori (L1), the driver re-checks the
*actual* measured footprint against budget (L2), and the engine's own worker refuses-before-load
(L4) and aborts mid-probe (L5). Returns ``{model, safe_context, binding, points}`` (or, when
preflight can't measure the model, ``{model, safe_context: None, points: []}``).
"""
from __future__ import annotations

from collections.abc import Callable

from ara.contracts import ramp, worker


def characterize(model: str, *, preflight: Callable[[str], dict],
                 measure: Callable[[str, int], dict], schedule: list[int]) -> dict:
    """Drive the safe ramp for *model* using engine-supplied *preflight*/*measure* callables."""
    est = preflight(model)
    if "error" in est:
        return {"model": model, "safe_context": None, "points": []}

    def measure_fn(ctx: int):
        m = worker.parse(measure(model, ctx))
        # L2 (independent of L1's prediction): mem_gb is the model DELTA, so the ACTUAL
        # absolute footprint is ref_baseline + delta. If that reached the budget, stop
        # escalating and don't trust higher contexts — even though L1 predicted it safe.
        if not m.refused and m.mem_gb is not None \
                and est["ref_baseline_gb"] + m.mem_gb >= est["budget_gb"]:
            return worker.Measurement(context=ctx, mem_gb=None, refused=True,
                                      reason="ARA L2: measured at/over safe budget")
        return m

    rungs = [c for c in schedule
             if est["max_context"] is None or c <= est["max_context"]]
    res = ramp.run(measure_fn, rungs, est["base_gb"], est["slope_gb_per_k"],
                   est["budget_gb"], ref_baseline_gb=est["ref_baseline_gb"],
                   max_context=est["max_context"])
    return {"model": model, "safe_context": res.safe_context, "binding": res.binding,
            "points": [{"context": c, "mem_gb": m} for c, m in res.points]}
