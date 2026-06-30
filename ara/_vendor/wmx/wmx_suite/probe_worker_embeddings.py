# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Single isolated memory measurement for one ModernBERT embeddings (batch, seq) cell.

Run as a subprocess — one fresh process per cell — so wired-memory residue from a previous
cell never contaminates the high-water reading. Prints one JSON line.

Usage:
    python -m wmx_suite.probe_worker_embeddings --model <id> --batch B --seq S \
        [--repeats N] [--margin GB]

Import convention: modules are imported (not their members) so tests can patch
`mlx_embeddings.load`, `mx.*`, and `system.*` on these shared module objects.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time

import mlx.core as mx

from . import system

# Headroom (GB) the worker reserves for loading the model weights before allocating
# activations. Conservative for a ModernBERT-base bf16 (~0.3 GB weights + overhead).
MODEL_WEIGHT_EST_GB = 0.6

try:
    import mlx_embeddings
except ImportError:  # pragma: no cover - exercised via patched import in tests
    mlx_embeddings = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--margin", type=float, default=2.0)
    args = ap.parse_args()

    limits = system.read_limits()
    threshold = limits.safe_threshold_gb(args.margin)
    if limits.wired_now_gb + MODEL_WEIGHT_EST_GB >= threshold:
        print(json.dumps({
            "status": "error",
            "note": (f"Pre-flight aborted: wired {limits.wired_now_gb:.2f} GB + "
                     f"weight headroom {MODEL_WEIGHT_EST_GB} GB >= threshold "
                     f"{threshold:.2f} GB. Model not loaded."),
        }), flush=True)
        sys.exit(0)

    if mlx_embeddings is None:
        print(json.dumps({
            "status": "error",
            "note": "mlx-embeddings not installed; add it to run the embeddings benchmark.",
        }), flush=True)
        sys.exit(1)

    # Background OS-wired sampler: MLX may free per-layer buffers mid-forward, so a single
    # post-eval read can miss the true high-water that gates LARGER cells. Start it BEFORE
    # loading the model so the weight-load transient is captured too; seed with the current
    # wired reading so an unscheduled sampler thread never reports a pathological 0.0.
    hi = [system.wired_gb()]
    stop = [False]

    def sampler():
        while not stop[0]:
            hi[0] = max(hi[0], system.wired_gb())
            time.sleep(0.05)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()

    # Always stop the sampler (and join it before reading hi[0]), even if load or the
    # forward pass raises (e.g. OOM mid-run), so the thread can't outlive the work and the
    # final high-water read isn't racing an in-flight sample.
    try:
        model, _tokenizer = mlx_embeddings.load(args.model)
        embed_dtype = model.model.embeddings.tok_embeddings.weight.dtype
        input_ids = mx.zeros((args.batch, args.seq), dtype=mx.int32)
        attention_mask = mx.ones((args.batch, args.seq), dtype=embed_dtype)

        # Warmup (compile Metal graphs); not measured.
        out = model(input_ids, attention_mask=attention_mask)
        mx.eval(out.last_hidden_state)
        mx.clear_cache()

        compute_times = []
        peaks = []
        for _ in range(max(1, args.repeats)):
            mx.clear_cache()
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            out = model(input_ids, attention_mask=attention_mask)
            mx.eval(out.last_hidden_state)
            compute_times.append(time.perf_counter() - t0)
            peaks.append(mx.get_peak_memory() / 1e9)
    finally:
        stop[0] = True
        t.join(timeout=0.2)

    compute_time = statistics.median(compute_times)
    throughput_tps = (args.batch * args.seq) / compute_time if compute_time > 0 else 0.0
    print(json.dumps({
        "status": "rung_done",
        "batch": args.batch,
        "seq": args.seq,
        "os_wired_gb": round(hi[0], 3),
        "peak_gb": round(max(peaks), 3),
        "compute_time": round(compute_time, 4),
        "throughput_tps": round(throughput_tps, 2),
        "latency_ms": round(compute_time * 1000, 3),
    }), flush=True)


if __name__ == "__main__":
    main()
