# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The process that actually touches the GPU — run in isolation so a crash/OOM kills it,
not the suite. Emits exactly one JSON line on stdout.

    python -m ara_engine_cuda.probe_worker calibrate
    python -m ara_engine_cuda.probe_worker measure <hf_id> <ctx> <abort_gb>

torch is imported lazily inside each mode, so the module loads without the (heavy,
NVIDIA-only) ``[cuda]`` extras installed — only running a GPU mode needs them.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

from . import models, system

WATCHDOG_INTERVAL_S = 0.25   # how often the L5 watchdog re-reads live VRAM


def _used_mb() -> float | None:
    """VRAM in use right now, in MiB (via nvidia-smi), or None."""
    lim = system.read_limits()
    return None if lim is None else lim.used_gb * system.MIB_PER_GB


def _used_gb() -> float | None:
    """VRAM in use right now, in GB (via nvidia-smi), or None."""
    mb = _used_mb()
    return None if mb is None else mb / system.MIB_PER_GB


def _poll_abort(abort_gb: float, *, sampler=_used_gb, on_breach=lambda: None) -> bool:
    """One watchdog tick: if live VRAM has reached *abort_gb*, fire *on_breach* and report True.

    ``>=`` is intentional — the budget is the line live VRAM must never reach (RULE #1). An
    unreadable sampler (None) is treated as 'can't confirm a breach', so it never false-fires.
    """
    used = sampler()
    if used is not None and used >= abort_gb:
        on_breach()
        return True
    return False


def _start_watchdog(abort_gb: float, *, interval: float = WATCHDOG_INTERVAL_S) -> None:
    """Spawn a daemon thread that hard-kills this process (``os._exit(3)``) the instant live VRAM
    reaches *abort_gb* — the last line of defence (L5) if the pre-flight gate mis-predicted."""
    def loop() -> None:
        while not _poll_abort(abort_gb, on_breach=lambda: os._exit(3)):
            time.sleep(interval)
    threading.Thread(target=loop, daemon=True).start()


def calibrate() -> dict:
    """Fixed VRAM the CUDA context itself costs — the nvidia-smi delta across torch CUDA
    init. Every model pays this before its own weights. Tiny allocation; safe."""
    import torch

    if not torch.cuda.is_available():
        return {"ok": False, "error": "CUDA not available"}
    before = _used_mb()
    torch.cuda.init()
    torch.zeros(1, device="cuda")      # force the context to materialize
    # cuBLAS/cuDNN load lazily on first use — but every real model run pays for them,
    # so a tiny matmul forces them in now. Otherwise we'd undercount overhead (unsafe).
    _m = torch.randn(64, 64, device="cuda")
    (_m @ _m).sum().item()
    torch.cuda.synchronize()
    after = _used_mb()
    if before is None or after is None:
        return {"ok": False, "error": "nvidia-smi unavailable"}
    return {
        "ok": True,
        "device": torch.cuda.get_device_name(0),
        "overhead_gb": max(0.0, after - before) / system.MIB_PER_GB,
    }


def _chunked_prefill(model, ids, torch, *, chunk, cache):
    """Prefill in fixed-size segments, accumulating the KV cache, so per-forward attention is
    ``chunk × N`` instead of ``N × N`` — the long-context unlock on cards without an efficient
    attention kernel (Turing). Feeds ``chunk`` tokens at a time with an explicit ``cache_position``
    (keeps RoPE + the causal mask correct across the growing cache) and ``logits_to_keep=1`` (the
    per-position logits tensor is never needed during prefill). Returns ``(cache, last_logits)``."""
    n = ids.shape[1]
    last_logits = None
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out = model(ids[:, s:e], past_key_values=cache,
                    cache_position=torch.arange(s, e, device=ids.device),
                    use_cache=True, logits_to_keep=1)
        cache, last_logits = out.past_key_values, out.logits
        # Return the freed transient attention scratch to the driver after each chunk. The live KV
        # cache is still referenced, so this only releases the per-chunk scratch — keeping torch's
        # reserved pool ~= the working set instead of bloating past dedicated VRAM and spilling to
        # shared RAM (the ~4x slowdown). Negligible overhead; keeps long-context prefill in-VRAM.
        torch.cuda.empty_cache()
    return cache, last_logits


def _new_cache(kv_bits, torch, model):
    """The KV cache object chunked prefill fills incrementally: a plain ``DynamicCache`` (fp16) or an
    HQQ ``QuantizedCache`` (kv_bits), matching the precision the ceiling is certified at. transformers
    5.x builds the quantized cache from ``(backend, model.config, nbits, q_group_size)`` — both live
    in ``cache_utils``. GPU/transformers-only and live-validated; the chunking *logic* around it is
    unit-tested separately."""
    from transformers.cache_utils import DynamicCache
    if kv_bits is None:
        return DynamicCache()
    from transformers.cache_utils import QuantizedCache
    return QuantizedCache(backend="hqq", config=model.config, nbits=kv_bits,
                          q_group_size=models.KV_GROUP_SIZE)


# Chunked prefill into a QUANTIZED cache is novel (HQQ is built for single-token decode updates); if
# the cache can't be built or rejects a multi-token chunk fill, we fall back to single-shot for that
# path — and measurement + generation both fall back identically (shared code), so the gate stays
# honest. ImportError/AttributeError cover a transformers cache-API shift; NotImplementedError/Type/
# Value cover a cache that won't take the update. Narrow on purpose: never swallow an
# OutOfMemoryError (a real wall hit must surface, not silently retry a bigger single-shot).
_CHUNK_FALLBACK_ERRORS = (NotImplementedError, TypeError, ValueError, ImportError, AttributeError)


def _fill_kv(model, ids, kv_bits, torch, *, chunk=None) -> None:
    """Populate the KV cache to *ids* length at the requested precision, so the VRAM read reflects
    how it'll actually run. With *chunk* set, prefill is segmented (``_chunked_prefill``) so the
    measured footprint matches a chunked ``run``; else fp16 is a plain forward and quantized goes
    through ``generate`` (one step) so transformers builds the quantized cache via its public API."""
    with torch.no_grad():
        if chunk is not None:
            try:
                _chunked_prefill(model, ids, torch, chunk=chunk, cache=_new_cache(kv_bits, torch, model))
                torch.cuda.synchronize()
                return
            except _CHUNK_FALLBACK_ERRORS:
                pass        # quantized cache won't take chunk fills → single-shot below (symmetric)
        if kv_bits is None:
            model(ids)
        else:
            model.generate(ids, max_new_tokens=1, do_sample=False,
                           **models.kv_cache_kwargs(kv_bits))
    torch.cuda.synchronize()


def _load_model(model_id, torch, *, prefer_flash=False, weight_quant="none"):
    """Load the causal LM with the requested attention + weight quantization. Quantized loads
    (bitsandbytes int8/int4, FP8, or a pre-quantized GPTQ/AWQ repo) place weights via ``device_map``
    — you can't ``.to()`` a quantized model after loading; an fp16 load goes straight to CUDA."""
    from transformers import AutoModelForCausalLM

    prequant = models.is_prequantized(model_id)
    kw = {"torch_dtype": torch.float16,
          "attn_implementation": models.attn_implementation(prefer_flash)}
    kw.update(models.weight_quant_kwargs(weight_quant, prequantized=prequant))
    if "quantization_config" in kw or prequant:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", **kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw).to("cuda")
    model.eval()
    return model


def characterize(model_id: str, contexts_csv: str, kv_bits_str: str | None = None, *,
                 prefer_flash: bool = False, weight_quant: str = "none",
                 chunk: int | None = None) -> dict:
    """Load *model_id* on the GPU and measure total VRAM at each context in *contexts_csv*.

    Probes small→large; stops at the first context that errors/OOMs, keeping the safe points
    below it. Total VRAM is nvidia-smi ``used`` after a synchronized forward — it captures the
    display baseline, the CUDA context, and torch's (retained) reserved pool, so it's the
    conservative number to compare against the wall. *kv_bits_str* (when given) measures with a
    quantized KV cache; *prefer_flash* opts into FlashAttention-2 (else SDPA); *weight_quant*
    loads the weights quantized (int8/int4/fp8).
    """
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        return {"ok": False, "error": "CUDA not available"}
    device = torch.cuda.get_device_name(0)
    contexts = [int(c) for c in contexts_csv.split(",") if c]
    kv_bits = int(kv_bits_str) if kv_bits_str is not None else None
    try:
        AutoTokenizer.from_pretrained(model_id)   # validate + warm the cache
        model = _load_model(model_id, torch, prefer_flash=prefer_flash, weight_quant=weight_quant)
    except Exception as exc:
        return {"ok": False, "error": f"load failed: {type(exc).__name__}: {exc}"}

    points: list[dict] = []
    for ctx in contexts:
        try:
            ids = torch.randint(0, int(model.config.vocab_size), (1, ctx), device="cuda")
            _fill_kv(model, ids, kv_bits, torch, chunk=chunk)
        except Exception:
            break   # OOM/error at this context — stop, keep the safe points below it
        used = _used_mb()
        if used is None:
            return {"ok": False, "error": "nvidia-smi unavailable"}
        points.append({"context": ctx, "used_gb": used / system.MIB_PER_GB})

    if not points:
        return {"ok": False, "error": "no contexts measured"}
    return {"ok": True, "device": device, "model": model_id, "points": points}


def measure(model_id: str, ctx_str: str, abort_gb_str: str | None = None,
            kv_bits_str: str | None = None, *, prefer_flash: bool = False,
            weight_quant: str = "none", chunk: int | None = None) -> dict:
    """Load *model_id* on the GPU at one context and return its VRAM delta over the launch baseline.

    Single-context primitive ARA's driver drives (via :func:`probe.measure_once`). Refuses before
    importing torch if no abort limit is given — running without the L5 watchdog would mean
    probing toward the wall with no backstop (RULE #1, fail-closed). Reports ``baseline_gb`` (live
    VRAM before the model loads, in this fresh process) and ``used_gb`` (after a synchronized
    forward); their difference is the model's own footprint, which ARA re-bases on the live wall.
    *kv_bits_str* (when given) measures with a quantized KV cache, so the certified ceiling matches
    how ``run`` will execute; *prefer_flash* opts into FlashAttention-2 (else SDPA).
    """
    if abort_gb_str is None:
        return {"ok": False, "error": "refusing to probe without an abort limit (L5)"}
    ctx, abort_gb = int(ctx_str), float(abort_gb_str)
    kv_bits = int(kv_bits_str) if kv_bits_str is not None else None

    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        return {"ok": False, "error": "CUDA not available"}
    baseline = _used_gb()
    if baseline is None:
        return {"ok": False, "error": "nvidia-smi unavailable"}

    _start_watchdog(abort_gb)        # L5 armed before a single byte of weights loads
    device = torch.cuda.get_device_name(0)
    try:
        AutoTokenizer.from_pretrained(model_id)
        model = _load_model(model_id, torch, prefer_flash=prefer_flash, weight_quant=weight_quant)
        ids = torch.randint(0, int(model.config.vocab_size), (1, ctx), device="cuda")
        _fill_kv(model, ids, kv_bits, torch, chunk=chunk)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    used = _used_gb()
    if used is None:
        return {"ok": False, "error": "nvidia-smi unavailable"}
    return {"ok": True, "device": device, "used_gb": used, "baseline_gb": baseline}


_MODES = {"calibrate": calibrate, "characterize": characterize, "measure": measure}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    mode = argv[0] if argv else ""
    fn = _MODES.get(mode)
    if fn is None:
        result: dict = {"ok": False, "error": f"unknown mode: {mode!r}"}
    else:
        # --flash-attn (bare) and --weight-quant <val> are named flags stripped to kwargs, so they
        # never collide with the conditional positional kv_bits; everything else stays positional.
        rest = [a for a in argv[1:] if a != "--flash-attn"]
        kwargs = {"prefer_flash": True} if "--flash-attn" in argv[1:] else {}
        if "--weight-quant" in rest:
            i = rest.index("--weight-quant")
            kwargs["weight_quant"] = rest[i + 1]
            del rest[i:i + 2]
        if "--prefill-chunk" in rest:
            i = rest.index("--prefill-chunk")
            kwargs["chunk"] = int(rest[i + 1])
            del rest[i:i + 2]
        try:
            result = fn(*rest, **kwargs)
        except Exception as exc:   # never let a GPU fault escape as a traceback
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
