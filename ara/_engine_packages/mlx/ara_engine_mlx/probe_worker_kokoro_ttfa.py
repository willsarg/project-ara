# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Kokoro TTS TTFA latency benchmark worker.

Benchmarks streaming Time-to-First-Audio (TTFA) latency against non-streaming 
total generation time across varying utterance lengths, enforcing active RAM safeguards.
Prints JSON lines to stdout.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import sys

BASE_TEXT = (
    "The MLX framework is an array framework for machine learning research on Apple Silicon, "
    "brought to you by Apple's machine learning research team. MLX is designed by machine learning "
    "researchers for machine learning researchers. The framework is intended to be user-friendly, "
    "yet still efficient to train and deploy models. The design of the framework itself is also "
    "conceptual-simple, and we want to make it easy for researchers to extend and explore. "
    "Kokoro is an open-weight TTS model with 82 million parameters. Despite its small size, it delivers "
    "exceptional quality, competing with much larger models. It is designed to run efficiently on local "
    "hardware. By running Kokoro on Apple Silicon via the MLX framework, we can achieve high-throughput "
    "audio synthesis with very low latency. This makes it ideal for real-time applications such as screen "
    "readers, interactive assistants, and voiceovers. Let us test the generation speed and efficiency. "
    "In order to perform a thorough benchmark, we must generate utterances of varying lengths. "
    "Utterance length is directly related to the computational complexity of the synthesis process. "
    "While causal language models have KV cache memory scaling issues, Kokoro uses a non-causal feedforward "
    "architecture; even so, its OS-wired memory footprint grows with utterance length, since longer output "
    "audio allocates larger buffers, and grows further under concurrent synthesis. "
    "Instead, the primary performance metric for text-to-speech is the Real-Time Factor, which measures "
    "how many seconds of audio can be generated per second of computation. A Real-Time Factor of less than "
    "one indicates that generation is faster than real-time. The smaller the Real-Time Factor, the faster "
    "the synthesis. Characters-per-second throughput is another key metric, indicating the raw processing speed. "
    "By measuring these factors across a range of text lengths, we can characterize the performance profile of "
    "the system and identify any non-linearities or overheads in the pipeline, such as tokenization, phonemization, "
    "and model dispatch latency. This benchmark provides crucial data for developers looking to integrate local "
    "TTS engines into their applications on macOS. We will evaluate performance under various conditions."
)


def get_text_of_length(length: int) -> str:
    """Slice and repeat BASE_TEXT to construct a natural-looking string of roughly L characters."""
    if length <= 0:
        return ""
    text = BASE_TEXT
    while len(text) < length:
        text += " " + BASE_TEXT

    # Find the nearest space close to target length to avoid cutting words in half
    space_idx = text.find(" ", length)
    if space_idx != -1 and space_idx - length < 20:
        return text[:space_idx].strip()
    return text[:length].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Kokoro-82M-bf16")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--lengths", default="10,50,100,200,500,1000,2000,3000")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--margin", type=float, default=2.0)
    args = ap.parse_args()

    try:
        lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    except ValueError:
        print(json.dumps({"status": "error", "note": "Invalid lengths argument"}), flush=True)
        sys.exit(1)

    # Pre-flight BEFORE importing mlx/kokoro, so the Metal init can't cross the wall first.
    from ara_engine_mlx import kokoro_safety
    threshold, _baseline, safe = kokoro_safety.preflight(args.margin)
    if not safe:
        print(json.dumps({
            "status": "error",
            "note": (f"Pre-flight aborted: settled baseline + "
                     f"{kokoro_safety.MODEL_WEIGHT_EST_GB} GB model-load headroom would reach "
                     f"the safe threshold ({threshold:.2f} GB); model not loaded."),
        }), flush=True)
        sys.exit(0)

    try:
        import mlx.core as mx
        from kokoro_mlx import KokoroTTS
    except ImportError as e:
        print(json.dumps({"status": "error", "note": f"Import failed: {e}"}), flush=True)
        sys.exit(1)

    # 1. Load model
    try:
        tts = KokoroTTS.from_pretrained(args.model)
    except Exception as e:
        print(json.dumps({"status": "error", "note": f"Failed to load model: {e}"}), flush=True)
        sys.exit(1)

    # 2. Warmup both streaming and non-streaming pipelines
    try:
        # Warmup streaming
        stream = tts.generate_stream("warmup", voice=args.voice)
        list(stream)
        # Warmup non-streaming
        tts.generate("warmup", voice=args.voice)
        mx.clear_cache()
        print(json.dumps({"status": "warmup_done"}), flush=True)
    except Exception as e:
        print(json.dumps({"status": "error", "note": f"Warmup run failed: {e}"}), flush=True)
        sys.exit(1)

    # 3. Sweep over lengths
    for length in lengths:
        # Fresh per-rung safeguard (a failed read counts as unsafe).
        if kokoro_safety.over_threshold(threshold):
            print(json.dumps({
                "status": "safeguard_triggered",
                "note": f"Active memory safeguard: OS-wired memory reached the safe threshold ({threshold:.2f} GB)."
            }), flush=True)
            break

        text = get_text_of_length(length)
        if not text:
            continue

        run_ttfas = []
        run_totals = []
        run_ratios = []
        run_chunk_durs = []
        run_peaks = []

        success = True
        for _ in range(max(1, args.repeats)):
            mx.clear_cache()
            mx.reset_peak_memory()
            time.sleep(0.05)

            # Measure Streaming TTFA
            t0 = time.perf_counter()
            try:
                stream = tts.generate_stream(text, voice=args.voice)
                first_chunk = next(stream)
                ttfa = time.perf_counter() - t0
                list(stream)  # exhaust stream
            except Exception as e:
                print(json.dumps({"status": "error", "note": f"Streaming failed at length {length}: {e}"}), flush=True)
                success = False
                break

            chunk_dur = len(first_chunk) / tts.SAMPLE_RATE

            # Per-sub-step safeguard: don't start the non-streaming pass if the streaming
            # pass already pushed wired memory to the threshold.
            mx.clear_cache()
            if kokoro_safety.over_threshold(threshold):
                print(json.dumps({
                    "status": "safeguard_triggered",
                    "note": f"Active memory safeguard before non-streaming pass: OS-wired memory reached the safe threshold ({threshold:.2f} GB)."
                }), flush=True)
                success = False
                break

            # Measure Non-Streaming Total Time
            time.sleep(0.05)
            t_start = time.perf_counter()
            try:
                tts.generate(text, voice=args.voice)
                total = time.perf_counter() - t_start
            except Exception as e:
                print(json.dumps({"status": "error", "note": f"Non-streaming failed at length {length}: {e}"}), flush=True)
                success = False
                break

            if ttfa <= 0:
                ttfa = 0.001

            run_ttfas.append(ttfa)
            run_totals.append(total)
            run_ratios.append(total / ttfa)
            run_chunk_durs.append(chunk_dur)
            run_peaks.append(mx.get_peak_memory() / 1e9)  # GB

        if not success:
            continue

        # Emit medians
        print(json.dumps({
            "status": "rung_done",
            "length": len(text),
            "ttfa_sec": round(statistics.median(run_ttfas), 3),
            "total_sec": round(statistics.median(run_totals), 3),
            "speedup_ratio": round(statistics.median(run_ratios), 2),
            "first_chunk_duration": round(statistics.median(run_chunk_durs), 3),
            "peak_gb": round(statistics.median(run_peaks), 3)
        }), flush=True)

    tts.close()


if __name__ == "__main__":
    main()
