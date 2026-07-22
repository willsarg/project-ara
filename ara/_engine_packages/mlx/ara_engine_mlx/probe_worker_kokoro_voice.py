# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Kokoro TTS voice switching latency benchmark worker.

Runs sequential sweeps within a single process to measure latency differences between:
1. Static Same-Voice Baseline (no voice changes)
2. Warm Switching (alternating between cached style vectors)
3. Cold Loading (clearing cache and reading a voice vector from disk)

Prints JSON lines to stdout.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Kokoro-82M-bf16")
    ap.add_argument("--voice-a", default="af_heart")
    ap.add_argument("--voice-b", default="am_adam")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--margin", type=float, default=2.0)
    args = ap.parse_args()

    # Pre-flight BEFORE importing mlx/kokoro, so the Metal init can't cross the wall first.
    from ara_engine_mlx import kokoro_safety
    threshold, _baseline, safe = kokoro_safety.preflight(args.margin)
    if not safe:
        print(json.dumps({
            "status": "error",
            "note": (f"Pre-flight aborted: settled baseline + "
                     f"{kokoro_safety.MODEL_WEIGHT_EST_GB} GiB model-load headroom would reach "
                     f"the safe threshold ({threshold:.2f} GiB); model not loaded."),
        }), flush=True)
        sys.exit(0)

    try:
        import mlx.core as mx
        from kokoro_mlx import KokoroTTS
        from kokoro_mlx.generate import generate
    except ImportError as e:
        print(json.dumps({"status": "error", "note": f"Import failed: {e}"}), flush=True)
        sys.exit(1)

    # 1. Load model
    try:
        tts = KokoroTTS.from_pretrained(args.model)
    except Exception as e:
        print(json.dumps({"status": "error", "note": f"Failed to load model: {e}"}), flush=True)
        sys.exit(1)

    phonemizer = tts._get_phonemizer(None, args.voice_a)
    test_text = "This is a voice switching test."

    # 2. Warmup both voices
    try:
        tts._voices.load_voice(args.voice_a)
        tts._voices.load_voice(args.voice_b)
        # compile JIT graphs
        generate(
            text=test_text,
            model=tts._model,
            config=tts._config,
            voice_manager=tts._voices,
            voice=args.voice_a,
            phonemizer=phonemizer
        )
        generate(
            text=test_text,
            model=tts._model,
            config=tts._config,
            voice_manager=tts._voices,
            voice=args.voice_b,
            phonemizer=phonemizer
        )
        mx.clear_cache()
        print(json.dumps({"status": "warmup_done"}), flush=True)
    except Exception as e:
        print(json.dumps({"status": "error", "note": f"Warmup run failed: {e}"}), flush=True)
        sys.exit(1)

    # 3. Runs for each condition
    def _safeguard(stage: str) -> bool:
        """Fresh per-step check (failed read = unsafe). Emits + signals stop if at/over."""
        if kokoro_safety.over_threshold(threshold):
            print(json.dumps({
                "status": "safeguard_triggered",
                "note": f"Active memory safeguard {stage}: OS-wired memory reached the safe threshold ({threshold:.2f} GiB)."
            }), flush=True)
            return True
        return False

    for r in range(max(1, args.repeats)):
        if _safeguard("before voice repeat"):
            break

        # A. Static same-voice baseline
        # Warmup/populate voice_a cache
        _ = generate(
            text=test_text,
            model=tts._model,
            config=tts._config,
            voice_manager=tts._voices,
            voice=args.voice_a,
            phonemizer=phonemizer
        )
        # Measure duration of second same-voice run
        t0 = time.perf_counter()
        _ = generate(
            text=test_text,
            model=tts._model,
            config=tts._config,
            voice_manager=tts._voices,
            voice=args.voice_a,
            phonemizer=phonemizer
        )
        t1 = time.perf_counter()
        static_duration = (t1 - t0) * 1000.0  # ms
        
        print(json.dumps({
            "status": "rung_done",
            "cond_type": "static_baseline",
            "voice_from": args.voice_a,
            "voice_to": args.voice_a,
            "duration_ms": round(static_duration, 2)
        }), flush=True)

        if _safeguard("before warm-switch step"):
            break

        # B. Warm Switch (cached)
        # Warmup/populate voice_a cache
        _ = generate(
            text=test_text,
            model=tts._model,
            config=tts._config,
            voice_manager=tts._voices,
            voice=args.voice_a,
            phonemizer=phonemizer
        )
        # Measure switching to voice_b (already cached)
        t0 = time.perf_counter()
        _ = generate(
            text=test_text,
            model=tts._model,
            config=tts._config,
            voice_manager=tts._voices,
            voice=args.voice_b,
            phonemizer=phonemizer
        )
        t1 = time.perf_counter()
        warm_duration = (t1 - t0) * 1000.0  # ms

        print(json.dumps({
            "status": "rung_done",
            "cond_type": "warm_switch",
            "voice_from": args.voice_a,
            "voice_to": args.voice_b,
            "duration_ms": round(warm_duration, 2)
        }), flush=True)

        if _safeguard("before cold-load step"):
            break

        # C. Cold Load (clear cache, load from disk)
        tts._voices._cache.clear()
        mx.clear_cache()
        # Measure loading voice_b from disk and synthesizing
        t0 = time.perf_counter()
        _ = generate(
            text=test_text,
            model=tts._model,
            config=tts._config,
            voice_manager=tts._voices,
            voice=args.voice_b,
            phonemizer=phonemizer
        )
        t1 = time.perf_counter()
        cold_duration = (t1 - t0) * 1000.0  # ms

        print(json.dumps({
            "status": "rung_done",
            "cond_type": "cold_load",
            "voice_from": "none",
            "voice_to": args.voice_b,
            "duration_ms": round(cold_duration, 2)
        }), flush=True)

    tts.close()


if __name__ == "__main__":
    main()
