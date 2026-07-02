# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Governed one-shot text generation — the engine-side generate verb for ARA's `run`.

Same Safety-first (Rule #1) discipline as ``measure_one``: describe the model and run the
existing L4 safety gate *before* loading any weights. An unknown/non-causal model, or a
gate that vetoes, refuses with no load. The gate runs at the *effective* context the
one-shot will actually reach — ``min(ctx, prompt_tokens + max_tokens)`` — because MLX
grows its KV cache dynamically; the ceiling ``ctx`` stays the hard cap. Prompt tokens are
counted with the tokenizer only (no weights). Only a safe (model, ctx) reaches mlx_lm. The
prompt is read from **stdin**, never argv. Emits one JSON line:

    success: {"context": <int>, "completion": "<generated text>"}
    refused: {"context": <int>, "refused": true, "reason": "<why>"}

Usage:
    python -m wmx_suite.generate <hf_id> <ctx> --margin G --overhead G --max-tokens N
    (prompt on stdin)
"""
from __future__ import annotations

import argparse
import json
import sys

from . import measure_one, models, system
from .serve import register_turn_end_tokens

# reuse measure_one's canonical refusal shape for consistency across workers
_refused = measure_one._refused


def _prepare_prompt(hf_id: str, prompt: str):
    """Render the model's chat template to TOKEN IDS (single ``<bos>``) and count, no weights.

    Wraps *prompt* as a single ``{"role": "user", "content": prompt}`` turn. For instruct
    models (tokenizer has a chat template) it returns the ``apply_chat_template`` **token ids**,
    not the rendered string — so the later ``mlx_generate`` does NOT prepend a SECOND ``<bos>``
    by re-encoding an already-templated string (gemma-3: ``[2,2,...]`` vs ``[2,...]``), which
    degrades output (#107). The transformers tokenizer shares the model's vocab, so its ids are
    exactly what mlx_lm would produce. Base/completion models (no template) get the raw string,
    which ``mlx_generate`` tokenizes normally. Returns ``(prompt_input, token_count)``.

    ``transformers.AutoTokenizer.from_pretrained`` reads only tokenizer artefacts from the
    HF cache — it never touches weight tensors — so the refuse-before-load property
    (Rule #1) is preserved.  Lazy-imported so the module loads without transformers
    installed and tests can monkeypatch ``sys.modules["transformers"]``.
    """
    if not prompt:
        return prompt, 0
    from transformers import AutoTokenizer

    return _render_and_count(AutoTokenizer.from_pretrained(hf_id), prompt)


def _render_and_count(tok, prompt: str):
    """The render-to-ids + count core of :func:`_prepare_prompt`, on an already-loaded
    tokenizer — shared with ``benchmark._max_effective_ctx`` so a batch counts every prompt
    with ONE tokenizer load. Returns ``(prompt_input, token_count)``; empty prompt → 0."""
    if not prompt:
        return prompt, 0
    if getattr(tok, "chat_template", None):
        # Render to STRING, then encode with add_special_tokens=False -> a flat list[int] with a
        # SINGLE <bos> (the template's own), no double-<bos> (#107). Going via tokenize=True here
        # is unsafe: transformers 5.x returns a BatchEncoding that mlx_lm's mx.array() rejects.
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        )
        ids = tok.encode(text, add_special_tokens=False)
        return ids, len(ids)
    return prompt, len(tok.encode(prompt))


def generate(hf_id: str, ctx: int, *, prompt: str, margin_gb: float,
             overhead_gb: float, max_tokens: int, kv_bits: int | None = None) -> dict:
    """Gate then (if safe) load + generate; return the canonical result dict.

    Refuses before loading if the model is unknown/non-causal or if the shared safety
    gate predicts the footprint at the *effective* context would reach the safe budget.

    ``ctx`` is the characterized ceiling — the hard cap we never gate or generate beyond.
    But MLX grows its KV cache dynamically, so a one-shot from a short prompt only reaches
    ``prompt_tokens + max_tokens`` of context, not the full ceiling. We therefore gate on
    the *effective* context the run will actually reach, capped at the ceiling::

        effective_ctx = min(ctx, prompt_tokens + max_tokens)

    Gating the raw ceiling here would over-predict memory and refuse runs that
    ``characterize`` already certified safe. The reported context stays the ceiling.
    """
    info = models.describe(hf_id)
    if info is None:
        return _refused(ctx, f"model not found in HF cache: {hf_id}")
    if not info.is_causal:
        return _refused(ctx, f"{hf_id} is not a supported causal language model")

    # Apply the chat template (tokenizer only, no weights) then count the rendered tokens.
    # Instruct models need the template; base/completion models fall back to raw.
    # Gate on the *effective* context the one-shot will actually reach — capped at ceiling.
    prompt_input, prompt_tokens = _prepare_prompt(hf_id, prompt)
    effective_ctx = min(ctx, prompt_tokens + max_tokens)

    # fp16 (None) unless the cache type can quantize — keeps run consistent with characterize
    # and never quantizes a RotatingKVCache model (which would crash past the quant threshold).
    kv_bits = measure_one._effective_kv_bits(info, kv_bits)

    limits = system.read_limits()
    live_base = system.sample_settled_baseline()
    reason = measure_one.safety_gate(info, limits, effective_ctx, margin_gb=margin_gb,
                                     overhead_gb=overhead_gb, live_base=live_base,
                                     kv_bits=kv_bits)
    if reason is not None:
        return _refused(ctx, reason)

    # Lazy import (like probe_worker) so the module imports without mlx installed and
    # tests can monkeypatch sys.modules["mlx_lm"].
    from mlx_lm import generate as mlx_generate, load

    # Match production quant knobs when quantizing; pass nothing for fp16 (mlx_lm default).
    kv_kwargs = ({} if kv_bits is None
                 else {"kv_bits": kv_bits, "kv_group_size": 64, "quantized_kv_start": 5000})
    try:
        model, tok = load(hf_id)
    except Exception as exc:
        first = str(exc).splitlines()[0][:200] if str(exc) else ""
        return _refused(ctx, f"failed to load {hf_id}: {type(exc).__name__}: {first}")
    # Register instruct turn-end tokens so generation self-stops (mlx_lm registers only the
    # scalar <eos>, not <end_of_turn> etc.) — else it rambles to max_tokens (#107).
    register_turn_end_tokens(tok)
    text = mlx_generate(model, tok, prompt=prompt_input, max_tokens=max_tokens, **kv_kwargs)
    return {"context": ctx, "completion": text}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Governed one-shot text generation.")
    ap.add_argument("hf_id")
    ap.add_argument("ctx", type=int)
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--overhead", type=float, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--kv-bits", type=int, default=None,
                    help="KV-cache quantization bits (8 or 4); omit for fp16. Ignored for "
                         "non-quantizable (sliding-window) models.")
    args = ap.parse_args(argv)
    prompt = sys.stdin.read()
    result = generate(args.hf_id, args.ctx, prompt=prompt, margin_gb=args.margin,
                      overhead_gb=args.overhead, max_tokens=args.max_tokens,
                      kv_bits=args.kv_bits)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
