# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Self-vetoing single-measurement primitive — the engine-side hard veto (L4).

ARA's thin path drives the ramp, but crash-prevention is checked at every layer. This is the
engine's: given a model and one context, it reads the *live* VRAM wall, estimates the footprint
conservatively, and **refuses before loading anything** if the model's base alone — or the
prediction at that context — would reach the safe budget. Only when safe does it spawn the
isolated GPU probe worker (a fresh process, so an OOM/TDR kills the worker, not the suite), which
carries its own watchdog (L5). It emits one canonical JSON object:

    safe:    {"context": <int>, "mem_gb": <model VRAM delta over its launch baseline>}
    refused: {"context": <int>, "refused": true, "reason": "<why>"}

Usage: ``python -m wcx_suite.measure_one <hf_id> <ctx> --margin G --overhead G [--preflight]``
"""
from __future__ import annotations

import argparse
import json
import statistics

from . import models, probe, system

# fp16 weights load into VRAM at ~their on-disk size (already fp16/bf16 checkpoints); a checkpoint
# stored at higher precision only over-estimates here, which is safe. No MLX-style wired inflation.
RESIDENT_FACTOR = 1.0

# nvidia-smi `used` after a synchronized forward is deterministic (it captures torch's retained
# reserved pool), unlike Apple's jittery wired sampler — so a single measurement is stable.
DEFAULT_REPEATS = 1


def _effective_weight_factor(info, weight_quant: str) -> float:
    """The weight-size factor to estimate with: 1.0 for a pre-quantized repo (its on-disk weights
    are already small and load as-is), else the requested weight-quant factor."""
    if models.is_prequantized(info.hf_id):
        return 1.0
    return models.weight_bytes_factor(weight_quant)


def _model_base_gb(info, overhead_gb: float, weight_quant: str = "none") -> float:
    """The model's own resident VRAM at context 0: (quantized) weights + the CUDA-context overhead."""
    return (info.weights_gb * RESIDENT_FACTOR * _effective_weight_factor(info, weight_quant)
            + overhead_gb)


def _effective_kv_bits(info, kv_bits: int | None) -> int | None:
    """The KV precision we'll actually use: the request, forced to fp16 (None) when the model's
    cache can't be quantized (sliding-window). Mirrors the apple/wmx rule — never crash a model
    that can't quantize; silently fall back to the safe fp16 path."""
    return kv_bits if info.can_quantize_kv else None


def safety_gate(info, limits, ctx: int, *, margin_gb: float, overhead_gb: float,
                live_base: float, kv_bits: int | None = None,
                weight_quant: str = "none") -> str | None:
    """Return a refusal reason if probing (model, ctx) is unsafe, else None.

    Two independent, conservative (``>=``) refusals: the absolute base footprint must fit on its
    own (the model can load alongside whatever already uses the card), and the predicted footprint
    at *ctx* (live baseline + model base + a-priori KV slope) must stay strictly under budget. The
    slope is KV-aware and the base is weight-quant-aware, so opting into quant raises the gated
    ceiling instead of leaving it at fp16.
    """
    threshold = limits.safe_threshold_gb(margin_gb)
    base = live_base + _model_base_gb(info, overhead_gb, weight_quant)
    if base >= threshold:
        return f"base estimate {base:.2f}GB >= safe budget {threshold:.2f}GB — won't load"
    slope = info.estimated_slope_gb_per_k(_effective_kv_bits(info, kv_bits))
    predicted = base + slope * (ctx / 1000)
    if predicted >= threshold:
        return f"predicted {predicted:.2f}GB at {ctx} tok >= safe budget {threshold:.2f}GB"
    return None


def preflight(hf_id: str, *, margin_gb: float, overhead_gb: float,
              kv_bits: int | None = None, weight_quant: str = "none") -> dict:
    """No-load estimate for ARA's scheduler: the context→0 base, a-priori slope, budget, window.

    ARA owns the ramp methodology but not CUDA model knowledge, so the engine supplies the facts
    (model config + the live VRAM wall); ARA fits and solves on top. The slope reflects the
    effective KV precision (fp16 unless quant was requested and the cache supports it).
    """
    info = models.describe(hf_id)
    if info is None:
        return {"error": f"model not found in HF cache: {hf_id}"}
    if not info.is_causal:
        return {"error": f"{hf_id} is not a supported causal language model"}
    limits = system.read_limits()
    if limits is None:
        return {"error": "no NVIDIA GPU visible to nvidia-smi"}
    live_base = limits.used_gb              # live VRAM in use (display + other procs) right now
    return {
        "base_gb": round(live_base + _model_base_gb(info, overhead_gb, weight_quant), 4),
        "ref_baseline_gb": round(live_base, 4),
        "slope_gb_per_k": info.estimated_slope_gb_per_k(_effective_kv_bits(info, kv_bits)),
        "budget_gb": limits.safe_threshold_gb(margin_gb),
        "max_context": info.max_context,
    }


def _spawn_worker(hf_id: str, ctx: int, abort_gb: float, kv_bits: int | None = None,
                  prefer_flash: bool = False, weight_quant: str = "none",
                  chunk: int | None = None) -> dict:
    """Run the isolated GPU probe worker for one context; return its raw result dict.

    The worker gets the absolute safe budget as its hard abort limit, so its watchdog (L5) can
    kill the process if live VRAM reaches it despite the pre-flight gate. *kv_bits* (already
    resolved to the effective precision) tells the worker to measure with a quantized KV cache;
    *prefer_flash* measures with FlashAttention-2 (else SDPA).
    """
    return probe.measure_once(hf_id, ctx, abort_gb=abort_gb, kv_bits=kv_bits,
                              prefer_flash=prefer_flash, weight_quant=weight_quant, chunk=chunk)


def _refused(ctx: int, reason: str) -> dict:
    return {"context": ctx, "refused": True, "reason": reason}


def run(hf_id: str, ctx: int, *, margin_gb: float, overhead_gb: float,
        repeats: int = DEFAULT_REPEATS, kv_bits: int | None = None,
        prefer_flash: bool = False, weight_quant: str = "none",
        chunk: int | None = None) -> dict:
    """Gate then (if safe) measure; return the canonical result dict.

    ``mem_gb`` is the model's VRAM DELTA over its own launch baseline (``used − baseline``),
    median over *repeats* fresh runs. ARA adds the live ref_baseline back at solve time. *kv_bits*
    (the requested KV precision) is measured AT that precision so the certified ceiling matches how
    ``run`` will execute — never certify q4 but measure fp16 (the inconsistency we killed on MLX).
    """
    info = models.describe(hf_id)
    if info is None:
        return _refused(ctx, f"model not found in HF cache: {hf_id}")
    if not info.is_causal:
        return _refused(ctx, f"{hf_id} is not a supported causal language model")
    limits = system.read_limits()
    if limits is None:
        return _refused(ctx, "no NVIDIA GPU visible to nvidia-smi")

    eff_kv_bits = _effective_kv_bits(info, kv_bits)
    live_base = limits.used_gb
    reason = safety_gate(info, limits, ctx, margin_gb=margin_gb, overhead_gb=overhead_gb,
                         live_base=live_base, kv_bits=eff_kv_bits, weight_quant=weight_quant)
    if reason is not None:
        return _refused(ctx, reason)

    threshold = limits.safe_threshold_gb(margin_gb)
    deltas = []
    for _ in range(max(1, repeats)):
        raw = _spawn_worker(hf_id, ctx, abort_gb=threshold, kv_bits=eff_kv_bits,
                            prefer_flash=prefer_flash, weight_quant=weight_quant, chunk=chunk)
        if raw.get("status") != "ok":
            return _refused(ctx, f"probe failed: {raw.get('note', 'no output')}")
        deltas.append(raw["used_gb"] - raw["baseline_gb"])
    return {"context": ctx, "mem_gb": round(statistics.median(deltas), 3)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Safe single-context VRAM measurement.")
    ap.add_argument("hf_id")
    ap.add_argument("ctx", type=int)
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--overhead", type=float, required=True)
    ap.add_argument("--preflight", action="store_true",
                    help="print the no-load estimate (base/slope/budget) and exit")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--kv-bits", type=int, default=None,
                    help="quantize the KV cache to N bits (8 or 4); default fp16")
    ap.add_argument("--flash-attn", action="store_true",
                    help="opt into FlashAttention-2 (Ampere+ only; else falls back to SDPA)")
    ap.add_argument("--weight-quant", default="none",
                    help="load weights quantized: int8 / int4 / fp8 (default none)")
    ap.add_argument("--prefill-chunk", type=int, default=None,
                    help="prefill in segments of N tokens (chunked prefill); default single-shot")
    args = ap.parse_args(argv)
    if args.preflight:
        result = preflight(args.hf_id, margin_gb=args.margin, overhead_gb=args.overhead,
                           kv_bits=args.kv_bits, weight_quant=args.weight_quant)
    else:
        result = run(args.hf_id, args.ctx, margin_gb=args.margin,
                     overhead_gb=args.overhead, repeats=args.repeats, kv_bits=args.kv_bits,
                     prefer_flash=args.flash_attn, weight_quant=args.weight_quant,
                     chunk=args.prefill_chunk)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
