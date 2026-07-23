# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Single isolated memory measurement for one (model, context) pair.

Run as a subprocess — one fresh process per context — so wired-memory residue from a
previous context never contaminates the high-water reading. Prints one JSON line.

Usage:
    python -m ara_engine_mlx.probe_worker <hf_id> <context> [--kv-bits N]
"""
from __future__ import annotations

import argparse
import json


def _allocator_counters(mx) -> dict[str, int]:
    """Exact MLX allocator observations; host VM telemetry belongs to the parent."""
    return {
        "mlx_peak_bytes": int(mx.get_peak_memory()),
        "mlx_active_plus_cache_bytes": int(
            mx.get_active_memory() + mx.get_cache_memory()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_id")
    ap.add_argument("context", type=int)
    ap.add_argument("--kv-bits", type=int, default=None)
    ap.add_argument("--kv-group-size", type=int, default=64)
    ap.add_argument("--quantized-kv-start", type=int, default=5000)
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import generate, load

    model, tok = load(args.hf_id)

    # build a prompt of exactly `context` tokens from repeated filler
    filler = "The quick brown fox jumps over the lazy dog. " * 20000
    ids = tok.encode(filler)
    result = {"hf_id": args.hf_id, "context": args.context}
    if args.context > len(ids):
        result.update(status="error", note="not enough filler tokens")
        print(json.dumps(result), flush=True)
        return

    prompt = tok.decode(ids[: args.context])
    gen_kwargs = dict(max_tokens=args.max_tokens, verbose=False)
    if args.kv_bits is not None:
        gen_kwargs.update(kv_bits=args.kv_bits, kv_group_size=args.kv_group_size,
                          quantized_kv_start=args.quantized_kv_start)

    mx.clear_cache()
    mx.reset_peak_memory()
    try:
        generate(model, tok, prompt=prompt, **gen_kwargs)
    except Exception as e:  # e.g. RotatingKVCache Quantization NYI
        result.update(status="error", note=f"{type(e).__name__}: {e}")
        print(json.dumps(result), flush=True)
        return

    result.update(status="ok", **_allocator_counters(mx))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
