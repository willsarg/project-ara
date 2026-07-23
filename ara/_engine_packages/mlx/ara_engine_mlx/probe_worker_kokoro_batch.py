# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Kokoro TTS batch performance benchmark worker.

Runs concurrent synthesis sweeps within a single process to measure the impact of 
batch size (concurrency) on aggregate throughput (CPS) and peak GPU memory footprint,
enforcing active RAM safeguards. Prints JSON lines to stdout.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import threading
import time
import sys

from . import units

# 16 sentences of approximately 100 characters each to benchmark concurrency levels 1 to 16
BENCHMARK_SENTENCES = [
    "The sun rises gracefully over the sleeping mountains, casting a golden glow on the quiet valley below.",
    "Technology continues to evolve at an incredible speed, transforming how we work and connect every day.",
    "Reading books allows us to travel to distant places and experience life through many different eyes.",
    "A balanced diet and regular physical exercise are essential components of a healthy and happy lifestyle.",
    "The soft sound of rain falling on the leaves creates a peaceful and calming atmosphere in the forest.",
    "Learning a new language opens up exciting opportunities to explore diverse cultures around the world.",
    "Deep learning models are achieving remarkable performance in speech synthesis and computer vision.",
    "The historic city features narrow cobblestone streets, ancient stone buildings, and beautiful plazas.",
    "Developing software requires patience, logical thinking, and a commitment to continuous learning.",
    "Exploring the deep ocean reveals mysterious creatures that thrive in complete darkness and high pressure.",
    "Kindness is a simple yet powerful gift that can brighten someone's day and build stronger communities.",
    "The international space station orbits the Earth every ninety minutes, hosting research from many nations.",
    "Fresh coffee in the morning provides a wonderful aroma and a gentle energy boost to start the day.",
    "Protecting natural habitats is crucial for preserving the diverse plants and animals on our planet.",
    "Artistic expression takes many forms, including painting, music, poetry, and theatrical performance.",
    "Understanding historical events helps us make better decisions and navigate the challenges of the future."
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Kokoro-82M-bf16")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--margin", type=float, default=2.0)
    args = ap.parse_args()

    try:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    except ValueError:
        print(json.dumps({"status": "error", "note": "Invalid batch-sizes argument"}), flush=True)
        sys.exit(1)

    # Pre-flight BEFORE importing mlx/kokoro, so the Metal init can't cross the wall first.
    from ara_engine_mlx import kokoro_safety
    threshold, baseline, safe = kokoro_safety.preflight(args.margin)
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
        from ara_engine_mlx.system import wired_gb
    except ImportError as e:
        print(json.dumps({"status": "error", "note": f"Import failed: {e}"}), flush=True)
        sys.exit(1)

    # 1. Load model
    try:
        tts = KokoroTTS.from_pretrained(args.model)
    except Exception as e:
        print(json.dumps({"status": "error", "note": f"Failed to load model: {e}"}), flush=True)
        sys.exit(1)

    phonemizer = tts._get_phonemizer(None, args.voice)

    # 2. Warmup run to compile Metal GPU kernels
    try:
        # Pre-load voice
        tts._voices.load_voice(args.voice)
        generate(
            text="warmup",
            model=tts._model,
            config=tts._config,
            voice_manager=tts._voices,
            voice=args.voice,
            phonemizer=phonemizer
        )
        mx.clear_cache()
        print(json.dumps({"status": "warmup_done"}), flush=True)
    except Exception as e:
        print(json.dumps({"status": "error", "note": f"Warmup run failed: {e}"}), flush=True)
        sys.exit(1)

    def run_one_generation(text: str):
        # Bypasses the KokoroTTS instance lock and calls the underlying generate() directly
        # from multiple threads. Thread-safety of concurrent generate() is an ASSUMPTION;
        # no race has been observed. Plain Python strings keep MLX allocations in-thread.
        return generate(
            text=text,
            model=tts._model,
            config=tts._config,
            voice_manager=tts._voices,
            voice=args.voice,
            phonemizer=phonemizer
        )

    # 3. Sweep over batch sizes.
    # The between-rung check alone cannot catch the transient spike DURING concurrent
    # synthesis, so we PREDICT each rung's peak = current wired + (measured worst-case
    # per-call transient) x concurrency, and refuse the rung if that would breach the wall.
    # per_call_gb is learned from each rung's measured high-water (a background sampler).
    per_call_gb = 0.0
    for batch_size in batch_sizes:
        if batch_size <= 0:
            continue
        if batch_size > len(BENCHMARK_SENTENCES):
            # Limit to the max available sentences
            batch_size = len(BENCHMARK_SENTENCES)

        # Fresh residency (a failed read counts as unsafe), then PREDICT the concurrent peak.
        try:
            current_wired = wired_gb()
        except Exception:
            current_wired = float("inf")
        predicted_peak = kokoro_safety.predicted_concurrent_peak(
            current_wired, per_call_gb, batch_size)
        if current_wired >= threshold or predicted_peak >= threshold:
            print(json.dumps({
                "status": "safeguard_triggered",
                "note": (f"Active memory safeguard: predicted concurrent peak "
                         f"{predicted_peak:.2f} GiB (current {current_wired:.2f} + "
                         f"{batch_size}x{per_call_gb:.3f} GiB/call) would reach the safe "
                         f"threshold ({threshold:.2f} GiB) at batch {batch_size}.")
            }), flush=True)
            break

        sentences_to_use = BENCHMARK_SENTENCES[:batch_size]
        total_chars = sum(len(txt) for txt in sentences_to_use)

        run_times = []
        run_cpss = []
        run_peaks = []
        success = True

        # Background sampler captures the in-rung concurrent transient high-water, which
        # feeds per_call_gb for gating the NEXT (larger) rung.
        hi = [current_wired]
        stop = [False]

        def _sampler():
            while not stop[0]:
                try:
                    hi[0] = max(hi[0], wired_gb())
                except Exception:
                    pass
                time.sleep(0.02)

        sampler_thread = threading.Thread(target=_sampler, daemon=True)
        sampler_thread.start()
        try:
            for _ in range(max(1, args.repeats)):
                mx.clear_cache()
                mx.reset_peak_memory()
                time.sleep(0.05)  # small settle

                t0 = time.perf_counter()
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                        futures = [executor.submit(run_one_generation, text) for text in sentences_to_use]
                        _ = [fut.result() for fut in futures]
                    t1 = time.perf_counter()
                except Exception as e:
                    print(json.dumps({"status": "error", "note": f"Generation failed at batch size {batch_size}: {e}"}), flush=True)
                    success = False
                    break

                elapsed = t1 - t0
                if elapsed <= 0:
                    elapsed = 0.001

                run_times.append(elapsed)
                run_cpss.append(total_chars / elapsed)
                run_peaks.append(units.bytes_to_gib(mx.get_peak_memory()))
        finally:
            stop[0] = True
            sampler_thread.join(timeout=0.2)

        if not success:
            continue

        # Learn the worst-case per-call transient from this rung's measured high-water so
        # the gate for the next (larger) batch reflects real concurrent cost.
        rung_delta = max(0.0, hi[0] - baseline)
        per_call_gb = max(per_call_gb, rung_delta / batch_size)

        # Emit median values for the rung
        print(json.dumps({
            "status": "rung_done",
            "batch_size": batch_size,
            "total_time": round(statistics.median(run_times), 3),
            "cps": round(statistics.median(run_cpss), 2),
            "peak_gb": round(statistics.median(run_peaks), 3)
        }), flush=True)

    tts.close()


if __name__ == "__main__":
    main()
