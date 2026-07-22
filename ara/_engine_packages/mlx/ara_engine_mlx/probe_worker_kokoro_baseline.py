# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Kokoro TTS active synthesis memory baseline benchmark worker.

Measures the OS-wired memory overhead (GiB) of a short warm synthesis over the idle
baseline. NOTE: this is only the floor — Kokoro's footprint is NOT static; it grows with
utterance length (longer output audio allocates more) and with concurrency (~GiB per
concurrent call). See the synthesis and batch benchmarks for the scaling curves.
Prints JSON lines to stdout.
"""
from __future__ import annotations

import argparse
import json
import time
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Kokoro-82M-bf16")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--margin", type=float, default=2.0)
    args = ap.parse_args()

    # Pre-flight BEFORE importing mlx/kokoro, so the Metal init can't cross the wall first.
    # Reserves model-load headroom on top of the settled baseline.
    from ara_engine_mlx import kokoro_safety
    threshold, baseline_gb, safe = kokoro_safety.preflight(args.margin)
    if not safe:
        print(json.dumps({
            "status": "error",
            "note": (f"Pre-flight aborted: settled baseline ({baseline_gb:.2f} GiB) + "
                     f"{kokoro_safety.MODEL_WEIGHT_EST_GB} GiB model-load headroom would reach "
                     f"the safe threshold ({threshold:.2f} GiB); model not loaded."),
        }), flush=True)
        sys.exit(0)

    try:
        import mlx.core as mx
        from kokoro_mlx import KokoroTTS
        from kokoro_mlx.generate import generate
        from ara_engine_mlx.system import sample_settled_baseline
    except ImportError as e:
        print(json.dumps({"status": "error", "note": f"Import failed: {e}"}), flush=True)
        sys.exit(1)

    # 2. Load model and run warm synthesis
    try:
        tts = KokoroTTS.from_pretrained(args.model)
        phonemizer = tts._get_phonemizer(None, args.voice)
        
        # Warmup active generation to initialize Metal graphs and buffers
        generate(
            text="This is a warm baseline memory measurement.",
            model=tts._model,
            config=tts._config,
            voice_manager=tts._voices,
            voice=args.voice,
            phonemizer=phonemizer
        )
        mx.clear_cache()
    except Exception as e:
        print(json.dumps({"status": "error", "note": f"Model initialization or synthesis failed: {e}"}), flush=True)
        sys.exit(1)

    # 3. Measure active system memory (settled)
    active_gb = sample_settled_baseline(settle=0.5, n=3, interval=0.2)
    
    # Calculate overhead
    overhead_gb = active_gb - baseline_gb
    if overhead_gb < 0:
        # Prevent negative values due to OS caching flux
        overhead_gb = 0.0

    print(json.dumps({
        "status": "rung_done",
        "baseline_gb": round(baseline_gb, 3),
        "active_gb": round(active_gb, 3),
        "overhead_gb": round(overhead_gb, 3)
    }), flush=True)

    tts.close()


if __name__ == "__main__":
    main()
