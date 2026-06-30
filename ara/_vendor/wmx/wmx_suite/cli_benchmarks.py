# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Benchmark subcommands (embeddings + Kokoro TTS), split out of cli.py.

Thin orchestration handlers: parse args, stream a worker subprocess, log to the
DB, and render through the views. Shared CLI helpers (`_stream_worker`,
`_configured_margin`, `CONSOLE`) stay in cli.py and are reached via the `cli`
module object so existing tests' monkeypatches keep intercepting.
"""
from __future__ import annotations

import json
import signal
import sys

from . import cli as _cli
from . import config, db, profiles
from .views import benchmark_embeddings as view_bench_embed
from .views import benchmark_kokoro as view_bench_kokoro
from .views import benchmark_kokoro_concurrency as view_bench_kokoro_conc


def cmd_benchmark_embeddings(args):
    """Run the ModernBERT embeddings 2D (batch x seq) memory-surface sweep and log it."""
    from . import embeddings_probe

    margin_val = _cli._configured_margin(args.margin)
    batches = [int(x) for x in str(args.batches).split(",") if x.strip()]
    seqs = [int(x) for x in str(args.seqs).split(",") if x.strip()]

    import mlx.core as mx
    mlx_version = mx.__version__

    con = db.connect()
    run_id = db.start_embeddings_run(con, args.model, mlx_version)

    ignore_profile = getattr(args, "ignore_profile", False)
    if not ignore_profile and profiles.embedding_coeffs(con, args.model, mlx_version):
        profile_source = "loaded"
    elif ignore_profile:
        profile_source = "ignored"
    else:
        profile_source = "cold"

    console = getattr(args, "console", None) or _cli.CONSOLE

    # Accumulate cells for the final render.  Aborted flags come from events.
    cells: dict = {}
    aborted = False
    safeguard_data: dict | None = None

    def on_event(event):
        nonlocal aborted, safeguard_data
        ev = event.get("event")
        if ev == "preflight_abort":
            aborted = True
            # Render the safeguard guidance immediately so the user sees it.
            view_bench_embed.render_safeguard(console, {
                "model": args.model,
                "reason": event.get("note", "pre-flight check refused this run"),
                "predicted_gb": event.get("predicted_gb") or 0.0,
                "safe_budget_gb": event.get("safe_budget_gb") or 0.0,
                "margin_gb": margin_val,
            })
        elif ev == "error":
            aborted = True
            console.emit(console.style(
                "warn",
                f"ERROR at batch {event.get('batch')} seq {event.get('seq')}: "
                f"{event.get('note')}"
            ))
        elif ev == "row_skipped":
            b, s = event["batch"], event["seq"]
            cells[(b, s)] = {
                "status": "skipped",
                "throughput_tps": None,
                "latency_ms": None,
                "peak_gb": None,
                "os_wired_gb": None,
                "predicted_gb": event.get("predicted_gb"),
            }
        elif ev == "cell_done":
            b, s = event["batch"], event["seq"]
            cells[(b, s)] = {
                "status": "measured",
                "throughput_tps": event.get("throughput_tps"),
                "latency_ms": event.get("latency_ms"),
                "peak_gb": event.get("peak_gb"),
                "os_wired_gb": event.get("os_wired_gb"),
                "predicted_gb": event.get("predicted_gb"),
            }

    summary = embeddings_probe.sweep(
        con, run_id, args.model, batches=batches, seqs=seqs,
        repeats=args.repeats, margin_gb=margin_val, mlx_version=mlx_version,
        ignore_profile=ignore_profile, on_event=on_event,
    )

    if aborted:
        sys.exit(1)

    view_bench_embed.render_surface(console, {
        "model": args.model,
        "margin_gb": margin_val,
        "profile_source": profile_source,
        "mlx_version": mlx_version,
        "batches": batches,
        "seqs": seqs,
        "cells": cells,
        "n_cells_measured": summary.get("n_cells_measured", 0),
        "n_cells_skipped": summary.get("n_cells_skipped", 0),
    })


def cmd_benchmark_kokoro(args):
    """Run Kokoro TTS performance sweep and log to database."""
    import json

    margin_val = _cli._configured_margin(args.margin)

    py = sys.executable
    cmd = [
        py, "-m", "wmx_suite.probe_worker_kokoro",
        "--model", args.model,
        "--voice", args.voice,
        "--lengths", args.lengths,
        "--repeats", str(args.repeats),
        "--margin", str(margin_val)
    ]

    import mlx.core as mx
    mlx_version = mx.__version__

    con = db.connect()
    run_id = db.start_kokoro_run(con, args.model, args.voice, mlx_version)

    console = getattr(args, "console", None) or _cli.CONSOLE

    rows: list[dict] = []
    safeguard_triggered = False
    safeguard_note: str = ""
    error_triggered = False

    def on_line(raw_line):
        nonlocal safeguard_triggered, safeguard_note, error_triggered
        line = raw_line.strip()
        if not line.startswith("{"):
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        status = data.get("status")
        if status == "warmup_done":
            pass  # no output — the view renders the full table at the end
        elif status == "safeguard_triggered":
            safeguard_triggered = True
            safeguard_note = data.get("note", "")
        elif status == "rung_done":
            length = data["length"]
            audio_dur = data["audio_duration"]
            comp_time = data["compute_time"]
            rtf = data["rtf"]
            cps = data["cps"]
            peak_gb = data["peak_gb"]
            os_wired = data.get("os_wired_gb")
            rows.append({
                "length": length,
                "audio_dur": audio_dur,
                "compute_time": comp_time,
                "rtf": rtf,
                "cps": cps,
                "peak_gb": peak_gb,
                "os_wired_gb": os_wired,
            })
            db.add_kokoro_measurement(
                con, run_id,
                text_length=length,
                audio_duration=audio_dur,
                compute_time=comp_time,
                rtf=rtf,
                cps=cps,
                peak_gb=peak_gb,
                os_wired_gb=os_wired
            )
        elif status == "error":
            console.emit(console.style("warn", f"ERROR: {data.get('note')}"))
            error_triggered = True
            return True  # signal _stream_worker to terminate the child immediately
        return None

    try:
        returncode, stderr_text = _cli._stream_worker(cmd, on_line)
    except KeyboardInterrupt:
        console.emit(console.style("warn", "Benchmark interrupted by user."))
        sys.exit(1)

    if error_triggered:
        sys.exit(1)

    if returncode != 0:
        console.emit(console.style("warn", f"Worker exited with code {returncode}."))
        if stderr_text:
            console.emit(console.style("gloss", f"Stderr: {stderr_text}"))
        sys.exit(returncode)

    if safeguard_triggered:
        view_bench_kokoro.render_safeguard(console, {
            "note": safeguard_note,
            "next_cmd": "wmx-suite benchmark-kokoro --lengths <shorter>",
        })

    view_bench_kokoro.render_perf(console, {
        "model": args.model,
        "voice": args.voice,
        "lengths": args.lengths,
        "repeats": args.repeats,
        "margin": margin_val,
        "rows": rows,
        "safeguard_triggered": safeguard_triggered,
        "run_id": run_id,
    })


def cmd_benchmark_kokoro_ttfa(args):
    """Run Kokoro TTS TTFA latency benchmark sweep and log to database."""
    import json

    margin_val = _cli._configured_margin(args.margin)

    py = sys.executable
    cmd = [
        py, "-m", "wmx_suite.probe_worker_kokoro_ttfa",
        "--model", args.model,
        "--voice", args.voice,
        "--lengths", args.lengths,
        "--repeats", str(args.repeats),
        "--margin", str(margin_val)
    ]

    import mlx.core as mx
    mlx_version = mx.__version__

    con = db.connect()
    run_id = db.start_kokoro_ttfa_run(con, args.model, args.voice, mlx_version)

    console = getattr(args, "console", None) or _cli.CONSOLE

    rows: list[dict] = []
    safeguard_triggered = False
    safeguard_note: str = ""
    error_triggered = False

    def on_line(raw_line):
        nonlocal safeguard_triggered, safeguard_note, error_triggered
        line = raw_line.strip()
        if not line.startswith("{"):
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        status = data.get("status")
        if status == "warmup_done":
            pass
        elif status == "safeguard_triggered":
            safeguard_triggered = True
            safeguard_note = data.get("note", "")
        elif status == "rung_done":
            length = data["length"]
            ttfa_sec = data["ttfa_sec"]
            total_sec = data["total_sec"]
            speedup = data["speedup_ratio"]
            chunk_dur = data["first_chunk_duration"]
            peak_gb = data["peak_gb"]
            rows.append({
                "length": length,
                "ttfa_sec": ttfa_sec,
                "total_sec": total_sec,
                "speedup_ratio": speedup,
                "first_chunk_dur": chunk_dur,
                "peak_gb": peak_gb,
            })
            db.add_kokoro_ttfa_measurement(
                con, run_id,
                text_length=length,
                ttfa_sec=ttfa_sec,
                total_sec=total_sec,
                speedup_ratio=speedup,
                first_chunk_duration=chunk_dur,
                peak_gb=peak_gb
            )
        elif status == "error":
            console.emit(console.style("warn", f"ERROR: {data.get('note')}"))
            error_triggered = True
            return True  # signal _stream_worker to terminate the child immediately
        return None

    try:
        returncode, stderr_text = _cli._stream_worker(cmd, on_line)
    except KeyboardInterrupt:
        console.emit(console.style("warn", "Benchmark interrupted by user."))
        sys.exit(1)

    if error_triggered:
        sys.exit(1)

    if returncode != 0:
        console.emit(console.style("warn", f"Worker exited with code {returncode}."))
        if stderr_text:
            console.emit(console.style("gloss", f"Stderr: {stderr_text}"))
        sys.exit(returncode)

    if safeguard_triggered:
        view_bench_kokoro.render_safeguard(console, {
            "note": safeguard_note,
            "next_cmd": "wmx-suite benchmark-kokoro-ttfa --lengths <shorter>",
        })

    view_bench_kokoro.render_ttfa(console, {
        "model": args.model,
        "voice": args.voice,
        "lengths": args.lengths,
        "repeats": args.repeats,
        "margin": margin_val,
        "rows": rows,
        "safeguard_triggered": safeguard_triggered,
        "run_id": run_id,
    })


def cmd_benchmark_kokoro_batch(args):
    """Run Kokoro TTS batch performance sweep and log to database."""
    import json

    margin_val = _cli._configured_margin(args.margin)

    py = sys.executable
    cmd = [
        py, "-m", "wmx_suite.probe_worker_kokoro_batch",
        "--model", args.model,
        "--voice", args.voice,
        "--batch-sizes", args.batch_sizes,
        "--repeats", str(args.repeats),
        "--margin", str(margin_val)
    ]

    import mlx.core as mx
    mlx_version = mx.__version__

    con = db.connect()
    run_id = db.start_kokoro_batch_run(con, args.model, args.voice, mlx_version)

    console = getattr(args, "console", None) or _cli.CONSOLE

    rows: list[dict] = []
    safeguard_triggered = False
    safeguard_note: str = ""
    error_triggered = False

    def on_line(raw_line):
        nonlocal safeguard_triggered, safeguard_note, error_triggered
        line = raw_line.strip()
        if not line.startswith("{"):
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        status = data.get("status")
        if status == "warmup_done":
            pass
        elif status == "safeguard_triggered":
            safeguard_triggered = True
            safeguard_note = data.get("note", "")
        elif status == "rung_done":
            batch_size = data["batch_size"]
            total_time = data["total_time"]
            cps = data["cps"]
            peak_gb = data["peak_gb"]
            rows.append({
                "batch_size": batch_size,
                "total_time": total_time,
                "cps": cps,
                "peak_gb": peak_gb,
            })
            db.add_kokoro_batch_measurement(
                con, run_id,
                batch_size=batch_size,
                total_time=total_time,
                cps=cps,
                peak_gb=peak_gb
            )
        elif status == "error":
            console.emit(console.style("warn", f"ERROR: {data.get('note')}"))
            error_triggered = True
            return True  # signal _stream_worker to terminate the child immediately
        return None

    try:
        returncode, stderr_text = _cli._stream_worker(cmd, on_line)
    except KeyboardInterrupt:
        console.emit(console.style("warn", "Benchmark interrupted by user."))
        sys.exit(1)

    if error_triggered:
        sys.exit(1)

    if returncode != 0:
        console.emit(console.style("warn", f"Worker exited with code {returncode}."))
        if stderr_text:
            console.emit(console.style("gloss", f"Stderr: {stderr_text}"))
        sys.exit(returncode)

    if safeguard_triggered:
        view_bench_kokoro_conc.render_safeguard(console, {
            "note": safeguard_note,
            "next_cmd": "wmx-suite benchmark-kokoro-batch --batch-sizes <smaller>",
        })

    view_bench_kokoro_conc.render_batch(console, {
        "model": args.model,
        "voice": args.voice,
        "batch_sizes": args.batch_sizes,
        "repeats": args.repeats,
        "margin": margin_val,
        "rows": rows,
        "safeguard_triggered": safeguard_triggered,
        "run_id": run_id,
    })


def cmd_benchmark_kokoro_voice(args):
    """Run Kokoro TTS voice switching performance sweep and log to database."""
    import json

    margin_val = _cli._configured_margin(args.margin)

    py = sys.executable
    cmd = [
        py, "-m", "wmx_suite.probe_worker_kokoro_voice",
        "--model", args.model,
        "--voice-a", args.voice_a,
        "--voice-b", args.voice_b,
        "--repeats", str(args.repeats),
        "--margin", str(margin_val)
    ]

    import mlx.core as mx
    mlx_version = mx.__version__

    con = db.connect()
    run_id = db.start_kokoro_voice_run(con, args.model, mlx_version)

    console = getattr(args, "console", None) or _cli.CONSOLE

    rows: list[dict] = []
    safeguard_triggered = False
    safeguard_note: str = ""
    error_triggered = False

    def on_line(raw_line):
        nonlocal safeguard_triggered, safeguard_note, error_triggered
        line = raw_line.strip()
        if not line.startswith("{"):
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        status = data.get("status")
        if status == "warmup_done":
            pass
        elif status == "safeguard_triggered":
            safeguard_triggered = True
            safeguard_note = data.get("note", "")
        elif status == "rung_done":
            cond_type = data["cond_type"]
            voice_from = data["voice_from"]
            voice_to = data["voice_to"]
            duration_ms = data["duration_ms"]
            rows.append({
                "cond_type": cond_type,
                "voice_from": voice_from,
                "voice_to": voice_to,
                "duration_ms": duration_ms,
            })
            db.add_kokoro_voice_measurement(
                con, run_id,
                cond_type=cond_type,
                voice_from=voice_from,
                voice_to=voice_to,
                duration_ms=duration_ms
            )
        elif status == "error":
            console.emit(console.style("warn", f"ERROR: {data.get('note')}"))
            error_triggered = True
            return True  # signal _stream_worker to terminate the child immediately
        return None

    try:
        returncode, stderr_text = _cli._stream_worker(cmd, on_line)
    except KeyboardInterrupt:
        console.emit(console.style("warn", "Benchmark interrupted by user."))
        sys.exit(1)

    if error_triggered:
        sys.exit(1)

    if returncode != 0:
        console.emit(console.style("warn", f"Worker exited with code {returncode}."))
        if stderr_text:
            console.emit(console.style("gloss", f"Stderr: {stderr_text}"))
        sys.exit(returncode)

    if safeguard_triggered:
        view_bench_kokoro_conc.render_safeguard(console, {
            "note": safeguard_note,
            "next_cmd": "wmx-suite benchmark-kokoro-voice --repeats <fewer>",
        })

    view_bench_kokoro_conc.render_voice(console, {
        "model": args.model,
        "voice_a": args.voice_a,
        "voice_b": args.voice_b,
        "repeats": args.repeats,
        "margin": margin_val,
        "rows": rows,
        "safeguard_triggered": safeguard_triggered,
        "run_id": run_id,
    })


def cmd_benchmark_kokoro_cache(args):
    """Run Kokoro TTS cache scaling performance sweep and log to database."""
    import json

    margin_val = _cli._configured_margin(args.margin)

    py = sys.executable
    cmd = [
        py, "-m", "wmx_suite.probe_worker_kokoro_cache",
        "--model", args.model,
        "--cache-sizes", args.cache_sizes,
        "--margin", str(margin_val)
    ]

    import mlx.core as mx
    mlx_version = mx.__version__

    con = db.connect()
    run_id = db.start_kokoro_cache_run(con, args.model, mlx_version)

    console = getattr(args, "console", None) or _cli.CONSOLE

    rows: list[dict] = []
    safeguard_triggered = False
    safeguard_note: str = ""
    error_triggered = False

    def on_line(raw_line):
        nonlocal safeguard_triggered, safeguard_note, error_triggered
        line = raw_line.strip()
        if not line.startswith("{"):
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        status = data.get("status")
        if status == "warmup_done":
            pass
        elif status == "safeguard_triggered":
            safeguard_triggered = True
            safeguard_note = data.get("note", "")
        elif status == "rung_done":
            cache_size = data["cache_size"]
            os_wired_gb = data["os_wired_gb"]
            peak_gb = data["peak_gb"]
            rows.append({
                "cache_size": cache_size,
                "os_wired_gb": os_wired_gb,
                "peak_gb": peak_gb,
            })
            db.add_kokoro_cache_measurement(
                con, run_id,
                cache_size=cache_size,
                os_wired_gb=os_wired_gb,
                peak_gb=peak_gb
            )
        elif status == "error":
            console.emit(console.style("warn", f"ERROR: {data.get('note')}"))
            error_triggered = True
            return True  # signal _stream_worker to terminate the child immediately
        return None

    try:
        returncode, stderr_text = _cli._stream_worker(cmd, on_line)
    except KeyboardInterrupt:
        console.emit(console.style("warn", "Benchmark interrupted by user."))
        sys.exit(1)

    if error_triggered:
        sys.exit(1)

    if returncode != 0:
        console.emit(console.style("warn", f"Worker exited with code {returncode}."))
        if stderr_text:
            console.emit(console.style("gloss", f"Stderr: {stderr_text}"))
        sys.exit(returncode)

    if safeguard_triggered:
        view_bench_kokoro_conc.render_safeguard(console, {
            "note": safeguard_note,
            "next_cmd": "wmx-suite benchmark-kokoro-cache --cache-sizes <smaller>",
        })

    view_bench_kokoro_conc.render_cache(console, {
        "model": args.model,
        "cache_sizes": args.cache_sizes,
        "margin": margin_val,
        "rows": rows,
        "safeguard_triggered": safeguard_triggered,
        "run_id": run_id,
    })


def cmd_benchmark_kokoro_baseline(args):
    """Run Kokoro TTS active baseline memory benchmark and log to database."""
    import json

    margin_val = _cli._configured_margin(args.margin)

    py = sys.executable
    cmd = [
        py, "-m", "wmx_suite.probe_worker_kokoro_baseline",
        "--model", args.model,
        "--voice", args.voice,
        "--margin", str(margin_val)
    ]

    import mlx.core as mx
    mlx_version = mx.__version__

    con = db.connect()
    run_id = db.start_kokoro_baseline_run(con, args.model, mlx_version)

    console = getattr(args, "console", None) or _cli.CONSOLE

    baseline_result: dict = {}
    error_triggered = False

    def on_line(raw_line):
        nonlocal error_triggered
        line = raw_line.strip()
        if not line.startswith("{"):
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        status = data.get("status")
        if status == "rung_done":
            baseline_gb = data["baseline_gb"]
            active_gb = data["active_gb"]
            overhead_gb = data["overhead_gb"]
            baseline_result["baseline_gb"] = baseline_gb
            baseline_result["active_gb"] = active_gb
            baseline_result["overhead_gb"] = overhead_gb
            db.add_kokoro_baseline_measurement(
                con, run_id,
                baseline_gb=baseline_gb,
                active_gb=active_gb,
                overhead_gb=overhead_gb
            )
        elif status == "error":
            console.emit(console.style("warn", f"ERROR: {data.get('note')}"))
            error_triggered = True
            return True  # signal _stream_worker to terminate the child immediately
        return None

    try:
        # capture_stderr=False: inherit terminal stderr, matching original behaviour.
        returncode, stderr_text = _cli._stream_worker(cmd, on_line, capture_stderr=False)
    except KeyboardInterrupt:
        console.emit(console.style("warn", "Benchmark interrupted by user."))
        sys.exit(1)

    if error_triggered:
        sys.exit(1)

    if returncode != 0:
        console.emit(console.style("warn", f"Worker exited with code {returncode}."))
        sys.exit(returncode)

    view_bench_kokoro.render_baseline(console, {
        "model": args.model,
        "voice": args.voice,
        "margin": margin_val,
        "baseline_gb": baseline_result.get("baseline_gb", 0.0),
        "active_gb": baseline_result.get("active_gb", 0.0),
        "overhead_gb": baseline_result.get("overhead_gb", 0.0),
        "run_id": run_id,
    })
