# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Governed multi-prompt MLX benchmark — the engine-side benchmark verb for ARA.

Same Safety-first (Rule #1) discipline as ``generate``: the safety gate runs *before*
any weight load. A vetoed gate prints one JSON refusal to stdout and exits 1. Only a safe
(model, context) pair reaches mlx_lm. The model and tokenizer load ONCE; all prompts run
in that single session. Per-prompt governance (``governed_max_tokens``) enforces the
ceiling for each prompt individually — a prompt that fills the ceiling receives a per-item
refusal dict, not a crash.

stdin = JSON array of raw prompt strings.
stdout = one JSON line (flushed):

    gate refused:  {"context": <int>, "refused": true, "reason": "<why>"}
    success:       {"context": <int>, "results": [<per-prompt>, ...]}

Per-prompt items:
    completion: {"prompt_index": <int>, "completion": "<text>"}
    per-item veto: {"prompt_index": <int>, "refused": true, "reason": "<why>"}

Usage:
    python -m ara_engine_mlx.benchmark <hf_id> <ctx_ceiling> --margin G --overhead G
        [--kv-bits N] [--max-tokens N]
    (JSON array of prompt strings on stdin)
"""
from __future__ import annotations

import argparse
import json
import sys

from .generate import _render_and_count
from .serve import _pre_load_gate, governed_max_tokens, register_turn_end_tokens

DEFAULT_MAX_TOKENS = 256


def _max_effective_ctx(hf_id: str, prompts: list[str], *, max_tokens: int,
                       ceiling: int) -> int:
    """Worst-case context this batch can actually reach, capped at *ceiling* — no weights.

    MLX grows its KV cache dynamically, so a batch of prompts only ever reaches
    ``longest_rendered_prompt + max_tokens`` of context — the honest target for the pre-load
    memory gate. Gating the raw ceiling instead over-predicts and refuses runs that
    ``characterize`` already certified safe (the same rationale as ``generate.run``'s
    ``effective_ctx``; the llama.cpp-family workers rightly gate the raw ceiling because
    llama.cpp allocates the full KV at ``n_ctx`` up front — MLX does not).

    Counts the RENDERED prompt (chat template applied) with one tokenizer-only load —
    ``AutoTokenizer.from_pretrained`` never touches weight tensors, preserving
    refuse-before-load (Rule #1).
    """
    longest = 0
    if prompts:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(hf_id)
        longest = max(_render_and_count(tok, p)[1] for p in prompts)
    return min(ceiling, longest + max_tokens)


def _run_prompts(prompts: list[str], tokenizer, *, max_tokens: int, ceiling: int,
                 effective_kv: int | None, mlx_generate, model) -> list[dict]:
    """Pure inner loop — process *prompts* with a pre-loaded *model* / *tokenizer*.

    Unit-testable without weights: pass a fake tokenizer and fake mlx_generate.
    kv_kwargs mirror the production quant knobs from generate.py and serve.py:
    fp16 gets no extra kwargs; quantized caches get kv_bits + kv_group_size +
    quantized_kv_start (consistent with measure_one).
    """
    kv_kwargs = ({} if effective_kv is None
                 else {"kv_bits": effective_kv, "kv_group_size": 64,
                       "quantized_kv_start": 5000})
    results: list[dict] = []
    for i, p in enumerate(prompts):
        # Apply the chat template when the tokenizer supports it so instruct models
        # receive the formatted input they need.  Pass the rendered TOKEN IDS (not the
        # re-encoded string) to mlx_generate: a templated string already contains <bos>,
        # and mlx_generate re-encoding it with add_special_tokens=True prepends a SECOND
        # <bos> (gemma-3: [2,2,105,...] vs [2,105,...]), degrading output (#107). Falls
        # back to the raw prompt for base/completion models or when templating fails.
        try:
            prompt_input = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,   # tokenize=True (default) -> token ids, single bos
            )
            prompt_tokens = len(prompt_input)
        except Exception:
            prompt_input = p
            prompt_tokens = len(tokenizer.encode(p))
        allowed = governed_max_tokens(prompt_tokens, max_tokens, ceiling)
        if allowed is None:
            results.append({"prompt_index": i, "refused": True,
                            "reason": f"prompt fills context ceiling {ceiling}"})
        else:
            try:
                text = mlx_generate(model, tokenizer, prompt=prompt_input,
                                    max_tokens=allowed, **kv_kwargs)
                results.append({"prompt_index": i, "completion": text})
            except Exception as exc:
                results.append({"prompt_index": i, "error": str(exc)})
    return results


def benchmark(hf_id: str, ceiling: int, *, prompts: list[str], margin_gb: float,
              overhead_gb: float, max_tokens: int = DEFAULT_MAX_TOKENS,
              kv_bits: int | None = None) -> dict:
    """Gate then (if safe) load once + generate for each prompt; return the result dict.

    Refuses before loading if the model is unknown/non-causal or the memory gate vetoes.
    The gate runs at the batch's EFFECTIVE context (see :func:`_max_effective_ctx`), not the
    raw ceiling — a short-prompt batch under a large measured window-bound ceiling must not be
    refused by the uncharacterized-model spike prior for memory it will never touch. If the
    tokenizer-only count fails (model not cached, template error), it degrades CONSERVATIVELY
    to gating the raw ceiling, so ``_pre_load_gate`` still reports those cases cleanly.
    The model and tokenizer load once; all prompts are generated in that session.
    Per-prompt governance enforces the ceiling for each prompt individually.
    """
    try:
        gate_ctx = _max_effective_ctx(hf_id, prompts, max_tokens=max_tokens, ceiling=ceiling)
    except Exception:
        gate_ctx = ceiling
    refusal, effective_kv = _pre_load_gate(hf_id, gate_ctx, margin_gb=margin_gb,
                                           overhead_gb=overhead_gb, kv_bits=kv_bits)
    if refusal is not None:
        return {"context": ceiling, "refused": True, "reason": refusal["reason"]}

    # Lazy import — preserves refuse-before-load (Rule #1) and allows test monkeypatching.
    from mlx_lm import generate as mlx_generate, load  # type: ignore[import]

    try:
        model, tokenizer = load(hf_id)
    except Exception as exc:
        first = str(exc).splitlines()[0][:200] if str(exc) else ""
        return {"context": ceiling, "refused": True,
                "reason": f"failed to load {hf_id}: {type(exc).__name__}: {first}"}
    # Register instruct turn-end tokens so generation self-stops (mlx_lm only knows the
    # scalar <eos>, not <end_of_turn> etc.) — else models ramble to max_tokens (#107).
    register_turn_end_tokens(tokenizer)
    results = _run_prompts(prompts, tokenizer, max_tokens=max_tokens, ceiling=ceiling,
                           effective_kv=effective_kv, mlx_generate=mlx_generate, model=model)
    return {"context": ceiling, "results": results}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Governed multi-prompt MLX benchmark.")
    ap.add_argument("hf_id")
    ap.add_argument("ctx_ceiling", type=int)
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--overhead", type=float, required=True)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help=f"Max new tokens per prompt (default {DEFAULT_MAX_TOKENS}).")
    ap.add_argument("--kv-bits", type=int, default=None,
                    help="KV-cache quantization bits (8 or 4); omit for fp16. "
                         "Ignored for non-quantizable (RotatingKVCache) models.")
    args = ap.parse_args(argv)
    prompts = json.loads(sys.stdin.read())
    result = benchmark(args.hf_id, args.ctx_ceiling, prompts=prompts,
                       margin_gb=args.margin, overhead_gb=args.overhead,
                       max_tokens=args.max_tokens, kv_bits=args.kv_bits)
    # A structured refusal (gate veto or load failure) is a valid result, NOT a crash — exit 0 so
    # run_worker returns the dict and ARA renders the reason cleanly (a non-zero exit would make
    # run_worker raise with stderr instead, losing the reason).
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
