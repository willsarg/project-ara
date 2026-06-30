# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""One-shot governed inference — the engine-side `run` verb, under the same hard veto as L4.

ARA drives this out-of-process for a single completion: given a model, a context ceiling, and a
prompt on stdin, refuse-before-loading if the run wouldn't be safe, else load on CUDA, generate,
and emit one JSON line. The CUDA sibling of ``wmx_suite.generate`` — identical contract and safety
discipline, but torch + transformers (imported LAZILY) instead of mlx_lm.

    success:  {"context": <int>, "completion": "<text>"}
    refusal:  {"context": <int>, "refused": true, "reason": "<why>"}

Crucially we gate the EFFECTIVE context, not the raw ceiling. transformers/torch grow the KV cache
dynamically during generation, so a one-shot from a short prompt only reaches
``prompt_tokens + max_tokens`` of context — not the full ceiling. Gating the ceiling over-predicts
memory and would refuse runs that ``characterize`` already certified. The token count uses the
tokenizer ONLY (no model weights), so the gate runs before a single byte of weights loads (RULE #1).

Usage: ``python -m wcx_suite.generate <hf_id> <ctx> --margin G --overhead G --max-tokens N``
       (PROMPT is read from stdin, never argv.)
"""
from __future__ import annotations

import argparse
import json
import sys

from . import measure_one, models, system

# Reuse the canonical refusal shape so every engine verb speaks the same dialect.
_refused = measure_one._refused


def _count_prompt_tokens(hf_id: str, prompt: str) -> int:
    """Prompt length in tokens via the tokenizer ONLY — no model weights touched.

    Empty prompt short-circuits to 0 before any import, so the no-prompt case never pays the
    transformers import cost (and never needs the cache warm)."""
    if not prompt:
        return 0
    from transformers import AutoTokenizer  # lazy: heavy, CUDA-box-only extra

    return len(AutoTokenizer.from_pretrained(hf_id).encode(prompt))


def _greedy_decode(model, cache, last_logits, torch, *, max_tokens, start_pos, eos_id) -> list:
    """Greedy decode after a chunked prefill: argmax the running logits, feed one token at a time
    back into the accumulated *cache* (with the correct ``cache_position``), stop at *max_tokens* or
    EOS. Greedy matches the non-chunked default (``do_sample=False``). Returns the new token ids."""
    ids, logits, pos = [], last_logits, start_pos
    for _ in range(max_tokens):
        nxt = int(torch.argmax(logits[:, -1, :], dim=-1)[0])
        ids.append(nxt)
        if eos_id is not None and nxt == eos_id:
            break
        out = model(torch.tensor([[nxt]], device="cuda"), past_key_values=cache,
                    cache_position=torch.arange(pos, pos + 1, device="cuda"),
                    use_cache=True, logits_to_keep=1)
        cache, logits, pos = out.past_key_values, out.logits, pos + 1
    return ids


def _apply_prompt(tokenizer, prompt):
    """Tokenize *prompt* for generation, applying the chat template for instruct models.

    Instruct models (``chat_template`` is set) require the prompt wrapped in their template;
    a raw prompt yields empty or garbage output (confirmed live with gemma). Base models
    (no ``chat_template``) fall back to raw tokenisation. Returns the input-ids tensor already
    on CUDA, ready to pass directly to ``model.generate`` or ``_chunked_prefill``.

    The template is rendered with ``tokenize=False`` (a string) and then tokenised, NOT via
    ``apply_chat_template(return_tensors="pt")`` — under transformers 5.x the latter returns a
    ``BatchEncoding`` (dict-like), not a bare tensor, so ``.shape`` would crash (caught live on
    transformers 5.12.1). Rendering to a string keeps both branches identical and version-stable.
    """
    if getattr(tokenizer, "chat_template", None) is not None:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False)
        return tokenizer(text, return_tensors="pt").to("cuda").input_ids
    return tokenizer(prompt, return_tensors="pt").to("cuda").input_ids


def _generate(hf_id: str, prompt: str, max_tokens: int, kv_bits: int | None = None,
              prefer_flash: bool = False, weight_quant: str = "none",
              chunk: int | None = None) -> str:
    """Load *hf_id* on CUDA, generate up to *max_tokens* new tokens, return the NEW text only.

    *kv_bits* (when set) quantizes the KV cache during generation — the same precision the ceiling
    was certified at, so the run executes exactly as characterized; *prefer_flash* opts into
    FlashAttention-2 (else SDPA); *weight_quant* loads weights quantized — all matching characterize.
    With *chunk* set, the prompt is prefilled in segments (``_chunked_prefill``) then greedily
    decoded, so ``run`` reproduces the chunked footprint the ceiling was certified at.
    Reuses probe_worker's load (quantized weights need device_map, not .to())."""
    import torch                                            # lazy (NVIDIA-only [cuda] extra)
    from transformers import AutoTokenizer

    from .probe_worker import _chunked_prefill, _load_model, _new_cache

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = _load_model(hf_id, torch, prefer_flash=prefer_flash, weight_quant=weight_quant)

    input_ids = _apply_prompt(tokenizer, prompt)
    prompt_len = input_ids.shape[1]
    if chunk is not None:
        with torch.no_grad():
            cache, last_logits = _chunked_prefill(
                model, input_ids, torch, chunk=chunk, cache=_new_cache(kv_bits, torch, model))
            new_tokens = _greedy_decode(model, cache, last_logits, torch, max_tokens=max_tokens,
                                        start_pos=prompt_len, eos_id=tokenizer.eos_token_id)
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, max_new_tokens=max_tokens,
                             **models.kv_cache_kwargs(kv_bits))
    new_tokens = out[0][prompt_len:]                         # decode only the newly generated span
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run(hf_id: str, ctx: int, *, margin_gb: float, overhead_gb: float, max_tokens: int,
        prompt: str = "", kv_bits: int | None = None, prefer_flash: bool = False,
        weight_quant: str = "none", chunk: int | None = None) -> dict:
    """Gate on the effective context, then (if safe) load + generate. Returns the result dict.

    The refusal/success both report the ceiling *ctx* (consistent with the wmx verb and the cpu
    worker), even though the gate is evaluated against the smaller effective context. *kv_bits* is
    resolved to the effective precision (fp16 unless quant was requested and the cache supports it)
    and used for BOTH the gate's a-priori slope and the actual generation, so they never disagree.
    """
    info = models.describe(hf_id)
    if info is None:
        return _refused(ctx, f"model not found in HF cache: {hf_id}")
    if not info.is_causal:
        return _refused(ctx, f"{hf_id} is not a supported causal language model")
    limits = system.read_limits()
    if limits is None:
        return _refused(ctx, "no NVIDIA GPU visible to nvidia-smi")

    eff_kv_bits = measure_one._effective_kv_bits(info, kv_bits)
    # A one-shot only reaches prompt + new tokens of context; never gate the raw ceiling.
    prompt_tokens = _count_prompt_tokens(hf_id, prompt)
    effective_ctx = min(ctx, prompt_tokens + max_tokens)

    live_base = limits.used_gb              # mirrors measure_one.run's exact call site
    reason = measure_one.safety_gate(info, limits, effective_ctx, margin_gb=margin_gb,
                                     overhead_gb=overhead_gb, live_base=live_base,
                                     kv_bits=eff_kv_bits, weight_quant=weight_quant)
    if reason is not None:
        return _refused(ctx, reason)        # report the ceiling, not the effective ctx

    completion = _generate(hf_id, prompt, max_tokens, eff_kv_bits, prefer_flash, weight_quant, chunk)
    return {"context": ctx, "completion": completion}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Safe one-shot inference under the L4 veto.")
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
    prompt = sys.stdin.read()               # PROMPT comes from stdin, never argv
    result = run(args.hf_id, args.ctx, margin_gb=args.margin, overhead_gb=args.overhead,
                 max_tokens=args.max_tokens, prompt=prompt, kv_bits=args.kv_bits,
                 prefer_flash=args.flash_attn, weight_quant=args.weight_quant,
                 chunk=args.prefill_chunk)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
