# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
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

from ara import acquire, catalog, methodology
from ara.contracts import ramp, worker


def _describe_ref(model: str) -> str:
    """The reference to look up in the catalog. For a ``repo:filename.gguf`` quant selector,
    that's the repo half — the catalog can't resolve the ``:filename``, which otherwise left
    ``decode_context`` null for pinned-quant GGUFs (#101). Other refs pass through unchanged."""
    return model.split(":", 1)[0] if acquire.valid_repo_gguf_ref(model) else model


def _rungs(schedule: list[int], max_context: int | None) -> list[int]:
    """The context rungs to probe, ascending, never above the model's window.

    Beyond the standard schedule clamped to *max_context*, this always probes ``max_context``
    itself — so a model whose window is small (e.g. 2048, below the 2nd schedule rung) still
    gets the ≥2 distinct points a fit needs, and is measured right at its window. For a window
    below every standard rung, a lower anchor (half the window) is added so two points exist.
    """
    if max_context is None:
        return list(schedule)
    if max_context <= 0:
        return []
    rungs = {c for c in schedule if c <= max_context} | {max_context}
    if len(rungs) < 2:
        anchor = max(1, max_context // 2)
        rungs.add(anchor)                  # max_context > 0, so this never exceeds the window
    return sorted(rungs)


def characterize(model: str, *, preflight: Callable[[str], dict],
                 measure: Callable[[str, int], dict], schedule: list[int],
                 kv_dtype_bytes: float = 2.0,
                 methodology_descriptor: dict | None = None) -> dict:
    """Drive the safe ramp for *model* using engine-supplied *preflight*/*measure* callables.

    *kv_dtype_bytes* is the engine's KV-cache element size for the analytic **decode** ceiling —
    default 2 (fp16). An engine that quantizes its KV cache passes the smaller per-element size
    (e.g. ~1.06 for q8_0, ~0.56 for q4_0) so the decode estimate reflects the cache actually in
    use; the driver stays engine-agnostic — it just takes a byte count, not a quant name.
    """
    est = preflight(model)
    if "error" in est:
        return {"model": model, "safe_context": None, "direct_context": None,
                "fitted_context": None, "points": [], "error": est["error"]}

    def measure_fn(ctx: int):
        m = worker.parse(measure(model, ctx))
        if m.context != ctx:
            raise worker.WorkerProtocolError(
                f"worker context mismatch: requested {ctx}, returned {m.context}")
        # L2 (independent of L1's prediction): mem_gb is the model DELTA, so the ACTUAL
        # absolute footprint is ref_baseline + delta. If that reached the budget, stop
        # escalating and don't trust higher contexts — even though L1 predicted it safe.
        if not m.refused and m.mem_gb is not None \
                and est["ref_baseline_gb"] + m.mem_gb >= est["budget_gb"]:
            return worker.Measurement(context=ctx, mem_gb=None, refused=True,
                                      reason="ARA L2: measured at/over safe budget",
                                      telemetry=m.telemetry)
        return m

    rungs = _rungs(schedule, est["max_context"])
    res = ramp.run(measure_fn, rungs, est["base_gb"], est["slope_gb_per_k"],
                   est["budget_gb"], ref_baseline_gb=est["ref_baseline_gb"],
                   max_context=est["max_context"])
    decode_context = None
    if res.fit is not None:
        meta = catalog.describe(_describe_ref(model)) or {}
        kv_slope = ramp.analytic_kv_slope_gb_per_k(
            meta.get("n_layers"), meta.get("kv_heads"), meta.get("head_dim"),
            kv_dtype_bytes=kv_dtype_bytes)
        if kv_slope:
            decode_context, _ = ramp.decode_ceiling(
                res.fit.intercept_gb, kv_slope, est["budget_gb"],
                est["ref_baseline_gb"], est["max_context"])
    points = []
    for context, mem_gb in res.points:
        point = {"context": context, "mem_gb": mem_gb}
        if context in res.telemetry:
            point["telemetry"] = res.telemetry[context]
        points.append(point)
    successful_contexts = {context for context, _mem_gb in res.points}
    refusal_telemetry = [
        {"context": context, "telemetry": telemetry}
        for context, telemetry in sorted(res.telemetry.items())
        if context not in successful_contexts
    ]
    out = {"model": model, "safe_context": res.safe_context,
           "direct_context": res.direct_context, "fitted_context": res.fitted_context,
           "binding": res.binding, "stopped_reason": res.stopped_reason,
           "aborted_at": res.aborted_at,
           "decode_context": decode_context,
           "points": points}
    if refusal_telemetry:
        out["refusal_telemetry"] = refusal_telemetry
    if methodology_descriptor is not None:
        out["methodology"] = methodology_descriptor
        out["methodology_key"] = methodology.key(methodology_descriptor)
    if res.safe_context is None:
        # Surface the stop reason and budget numbers so callers can explain the null — e.g.
        # "all contexts predicted over safe budget" — rather than silently emitting bare null.
        out["base_gb"] = est.get("base_gb")
        out["budget_gb"] = est.get("budget_gb")
    return out
