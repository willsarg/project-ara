# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Load-once governed multi-prompt inference — the engine-side `benchmark` verb.

ARA drives this out-of-process to complete MANY prompts with the model loaded a SINGLE time:
given a model, a context ceiling, and a JSON array of prompts on stdin, it gates each prompt
under the same hard L4 veto as ``generate``, then (load-once) completes the ones that fit. The
CUDA sibling of the Vulkan/CPU benchmark workers — identical contract and safety discipline, but
torch + transformers (imported LAZILY) instead of llama.cpp.

    success:  {"context": <int>, "results": [{"prompt_index": i, "completion": "<text>"} | ...]}
    per-prompt refusal:  {"prompt_index": i, "refused": true, "reason": "<why>"}
    whole-run refusal:   {"context": <int>, "refused": true, "reason": "<why>"}

Like ``generate``, each prompt is gated on its EFFECTIVE context (``min(ctx, prompt_tokens +
max_tokens)``), not the raw ceiling — transformers grows the KV cache dynamically, so a one-shot
from a short prompt only reaches that much context. The gate uses the tokenizer ONLY (no weights),
and the weights are loaded LAZILY on the first prompt whose gate passes — so a run that can't fit
even its first prompt never loads a byte of weights (RULE #1: refuse before load).

Usage: ``python -m wcx_suite.benchmark <hf_id> <ctx> --margin G --overhead G --max-tokens N``
       (the PROMPTS are a JSON array read from stdin, never argv.)
"""
from __future__ import annotations

import argparse
import json
import sys

from . import measure_one, models, probe_worker, system

# Reuse the canonical refusal shape so every engine verb speaks the same dialect.
_refused = measure_one._refused


def _generate_one(model, tokenizer, prompt: str, max_tokens: int, kv=None, ch=None) -> str:
    """Generate up to *max_tokens* new tokens for one *prompt* on an ALREADY-LOADED model+tokenizer.

    The per-prompt twin of ``generate._generate`` minus the load — the model is loaded once by the
    caller and reused here. *kv* is the effective KV-cache precision (None = fp16), *ch* opts into
    chunked prefill (segment the prompt, then greedy-decode) so the run reproduces the chunked
    footprint the ceiling was certified at. Returns the NEW text only."""
    import torch                                            # lazy (NVIDIA-only [cuda] extra)

    from .generate import _apply_prompt, _greedy_decode
    from .probe_worker import _chunked_prefill, _new_cache

    input_ids = _apply_prompt(tokenizer, prompt)
    prompt_len = input_ids.shape[1]
    if ch is not None:
        with torch.no_grad():
            cache, last_logits = _chunked_prefill(
                model, input_ids, torch, chunk=ch, cache=_new_cache(kv, torch, model))
            new_tokens = _greedy_decode(model, cache, last_logits, torch, max_tokens=max_tokens,
                                        start_pos=prompt_len, eos_id=tokenizer.eos_token_id)
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, max_new_tokens=max_tokens,
                             **models.kv_cache_kwargs(kv))
    new_tokens = out[0][prompt_len:]                         # decode only the newly generated span
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run(hf_id: str, ctx: int, *, margin_gb: float, overhead_gb: float, max_tokens: int,
        prompts: list, kv_bits: int | None = None, prefer_flash: bool = False,
        weight_quant: str = "none", chunk: int | None = None) -> dict:
    """Gate each prompt on its effective context, then (load-once) complete the ones that fit.

    The three whole-run guards (model in cache, causal LM, GPU visible) mirror ``generate.run`` and
    return before any torch/transformers import. Then, for each prompt, the L4 ``safety_gate`` is
    evaluated against ``min(ctx, prompt_tokens + max_tokens)`` — a per-prompt refusal records that
    index and continues (it does NOT abort the run). The model is loaded LAZILY on the first prompt
    whose gate passes (its gate already proved the weights-base fits), then reused for every later
    prompt — so the load happens at most once, and never before a gate has cleared it. Reports the
    ceiling *ctx* (consistent with the wmx/cpu/vulkan verbs), not the smaller effective context."""
    info = models.describe(hf_id)
    if info is None:
        return _refused(ctx, f"model not found in HF cache: {hf_id}")
    if not info.is_causal:
        return _refused(ctx, f"{hf_id} is not a supported causal language model")
    limits = system.read_limits()
    if limits is None:
        return _refused(ctx, "no NVIDIA GPU visible to nvidia-smi")

    eff_kv_bits = measure_one._effective_kv_bits(info, kv_bits)
    import torch                                            # lazy (NVIDIA-only [cuda] extra)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_id)        # tokenizer only — no weights, no GPU
    model = None                                            # loaded lazily on the first passing gate
    results = []
    for i, prompt in enumerate(prompts):
        prompt_tokens = len(tokenizer.encode(prompt))
        effective_ctx = min(ctx, prompt_tokens + max_tokens)
        reason = measure_one.safety_gate(info, limits, effective_ctx, margin_gb=margin_gb,
                                         overhead_gb=overhead_gb, live_base=limits.used_gb,
                                         kv_bits=eff_kv_bits, weight_quant=weight_quant)
        if reason is not None:
            results.append({"prompt_index": i, "refused": True, "reason": reason})
            continue
        if model is None:                                   # refuse-before-load: only after a pass
            model = probe_worker._load_model(hf_id, torch, prefer_flash=prefer_flash,
                                             weight_quant=weight_quant)
        # A per-prompt exception (decode failure, OOM) must NOT crash the whole run and lose the
        # completed work — capture it as an error result and press on (Rule #3, mirrors wmx).
        try:
            completion = _generate_one(model, tokenizer, prompt, max_tokens, eff_kv_bits, chunk)
            results.append({"prompt_index": i, "completion": completion})
        except Exception as exc:
            results.append({"prompt_index": i, "error": str(exc)})
    return {"context": ctx, "results": results}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Safe load-once multi-prompt inference under the L4 veto.")
    ap.add_argument("hf_id")
    ap.add_argument("ctx", type=int)
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--overhead", type=float, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--kv-bits", type=int, default=None,
                    help="quantize the KV cache to N bits (8 or 4); default fp16")
    ap.add_argument("--flash-attn", action="store_true",
                    help="opt into FlashAttention-2 (Ampere+ only; else falls back to SDPA)")
    ap.add_argument("--weight-quant", default="none",
                    help="load weights quantized: int8 / int4 / fp8 (default none)")
    ap.add_argument("--prefill-chunk", type=int, default=None,
                    help="prefill in segments of N tokens (chunked prefill); default single-shot")
    args = ap.parse_args(argv)
    prompts = json.loads(sys.stdin.read())  # PROMPTS are a JSON array on stdin, never argv
    result = run(args.hf_id, args.ctx, margin_gb=args.margin, overhead_gb=args.overhead,
                 max_tokens=args.max_tokens, prompts=prompts, kv_bits=args.kv_bits,
                 prefer_flash=args.flash_attn, weight_quant=args.weight_quant,
                 chunk=args.prefill_chunk)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
