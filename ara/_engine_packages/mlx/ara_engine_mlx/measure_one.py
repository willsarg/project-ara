# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Self-vetoing single-measurement primitive — the engine-side hard veto (L4).

ARA's thin path drives the ramp, but crash-prevention is checked at every layer. This is
the engine's: given a model and one context, it reads the *live* wall, estimates the
footprint conservatively, and **refuses before loading anything** if the base alone — or
the prediction at that context — would reach the safe budget. Only when safe does it spawn
the isolated probe worker (one fresh process, so wired residue never contaminates the
reading). It emits a single canonical JSON object:

    safe:    {"context": <int>, "mem_gb": <os-wired high-water>}
    refused: {"context": <int>, "refused": true, "reason": "<why>"}

Usage: ``python -m ara_engine_mlx.measure_one <hf_id> <ctx> --margin G --overhead G``
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

from . import models, probe, profiles, system

# Repeat each rung in fresh processes and take the median: a single prefill peak lands
# between sampler windows (±~1GiB jitter), so one shot makes ceilings erratic.
DEFAULT_REPEATS = 3


def _effective_kv_bits(info, kv_bits: int | None) -> int | None:
    """fp16 (None) for non-quantizable cache types — quantizing a RotatingKVCache model
    raises NotImplementedError past the quant threshold and crashes (Rule #1). The lever is
    honoured only where it is safe."""
    return kv_bits if info.can_quantize_kv else None


def safety_gate(info, limits, ctx: int, *, margin_gb: float, overhead_gb: float,
                live_base: float, kv_bits: int | None = None,
                measured_slope_gb_per_k: float | None = None) -> str | None:
    """Return a refusal reason if probing (model, ctx) is unsafe, else None.

    Two independent refusals, both conservative (``>=`` — never round toward the wall):
    the base estimate must fit on its own (the model can load), and the predicted
    footprint at *ctx* (live baseline + model base + slope) must stay under budget.

    The growth slope is the *a-priori* estimate from architecture (reflecting ``kv_bits``,
    fp16 by default) UNLESS ``measured_slope_gb_per_k`` is supplied — the ramp slope ARA fit
    from a real characterization of this model on this machine. Measured provenance is used
    only when ARA serves the model's own measured ceiling: the ramp already survived the real
    prefill spike and certified that ceiling, so the a-priori conservatism (which exists for
    UNcharacterized models) over-predicts and would refuse a safe serve. Crucially this only
    swaps the slope — the base-load check and the *current* live baseline still apply, so a
    busier machine at serve time still refuses honestly. Rule #1 is preserved: the ceiling was
    empirically measured safe, and the L5 runtime watchdog remains the hard backstop.
    Slug 2026-07-02-wmx-serve-measured-provenance-gate.
    """
    threshold = limits.safe_threshold_gb(margin_gb)
    est_base = probe.estimate_base_gb(info, limits, overhead_gb)
    if est_base >= threshold:
        return (f"base estimate {est_base:.2f}GiB >= safe budget {threshold:.2f}GiB — "
                f"won't load")
    model_base = info.weights_gb * profiles.DEFAULT_RESIDENT_FACTOR + overhead_gb
    slope = (measured_slope_gb_per_k if measured_slope_gb_per_k is not None
             else info.estimated_slope_gb_per_k(_effective_kv_bits(info, kv_bits)))
    predicted = live_base + model_base + slope * (ctx / 1000)
    if predicted >= threshold:
        return (f"predicted {predicted:.2f}GiB at {ctx} tok >= safe budget "
                f"{threshold:.2f}GiB")
    return None


def preflight(hf_id: str, *, margin_gb: float, overhead_gb: float,
              kv_bits: int | None = None) -> dict:
    """No-load estimate for ARA's scheduler: the context→0 base, a-priori slope, budget.

    ARA owns the ramp methodology but not model knowledge, so the engine supplies the
    facts (from the model config + the live wall); ARA fits and solves on top.
    """
    info = models.describe(hf_id)
    if info is None:
        return {"error": f"model not found in HF cache: {hf_id}"}
    if not info.is_causal:
        return {"error": f"{hf_id} is not a supported causal language model"}
    limits = system.read_limits()
    live_base = system.sample_settled_baseline()
    model_base = info.weights_gb * profiles.DEFAULT_RESIDENT_FACTOR + overhead_gb
    return {
        "base_gb": round(live_base + model_base, 4),   # absolute, for ARA's a-priori gate
        "ref_baseline_gb": round(live_base, 4),        # live OS baseline, added at solve time
        "slope_gb_per_k": info.estimated_slope_gb_per_k(_effective_kv_bits(info, kv_bits)),
        "budget_gb": limits.safe_threshold_gb(margin_gb),
        "max_context": info.max_context,
    }


def _spawn_worker(hf_id: str, ctx: int, kv_bits, abort_wired_gb) -> dict:
    """Run the isolated probe worker in this interpreter; return its raw result dict.

    The worker is handed the hard wired limit (the safe budget) so its own watchdog (L5)
    can abort if the live footprint reaches it despite the pre-flight gate.
    """
    return probe._run_worker(sys.executable, hf_id, ctx, kv_bits,
                             abort_wired_gb=abort_wired_gb)


def _refused(ctx: int, reason: str) -> dict:
    return {"context": ctx, "refused": True, "reason": reason}


def run(hf_id: str, ctx: int, *, margin_gb: float, overhead_gb: float,
        repeats: int = DEFAULT_REPEATS, kv_bits: int | None = None) -> dict:
    """Gate then (if safe) measure; return the canonical result dict.

    ``mem_gb`` is the model's DELTA over its own launch baseline (``os_wired - baseline``),
    median over *repeats* fresh runs — this removes ambient cross-process drift and per-run
    jitter, so the fit is stable. ARA adds the live ref_baseline back at solve time.
    """
    info = models.describe(hf_id)
    if info is None:
        return _refused(ctx, f"model not found in HF cache: {hf_id}")
    if not info.is_causal:
        return _refused(ctx, f"{hf_id} is not a supported causal language model")

    limits = system.read_limits()
    live_base = system.sample_settled_baseline()
    kv_bits = _effective_kv_bits(info, kv_bits)
    reason = safety_gate(info, limits, ctx, margin_gb=margin_gb,
                         overhead_gb=overhead_gb, live_base=live_base, kv_bits=kv_bits)
    if reason is not None:
        return _refused(ctx, reason)

    threshold = limits.safe_threshold_gb(margin_gb)
    deltas = []
    for _ in range(max(1, repeats)):
        raw = _spawn_worker(hf_id, ctx, kv_bits, abort_wired_gb=threshold)
        if raw.get("status") != "ok":
            return _refused(ctx, f"probe failed: {raw.get('note', 'no output')}")
        deltas.append(raw["os_wired_gb"] - raw["baseline_wired_gb"])
    return {"context": ctx, "mem_gb": round(statistics.median(deltas), 3)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Safe single-context memory measurement.")
    ap.add_argument("hf_id")
    ap.add_argument("ctx", type=int)
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--overhead", type=float, required=True)
    ap.add_argument("--preflight", action="store_true",
                    help="print the no-load estimate (base/slope/budget) and exit")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--kv-bits", type=int, default=None,
                    help="KV-cache quantization bits (8 or 4); omit for fp16. Ignored for "
                         "non-quantizable (sliding-window) models.")
    args = ap.parse_args(argv)
    if args.preflight:
        result = preflight(args.hf_id, margin_gb=args.margin, overhead_gb=args.overhead,
                           kv_bits=args.kv_bits)
    else:
        result = run(args.hf_id, args.ctx, margin_gb=args.margin,
                     overhead_gb=args.overhead, repeats=args.repeats, kv_bits=args.kv_bits)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
