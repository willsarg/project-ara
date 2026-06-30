# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Pure render functions for Kokoro TTS concurrency benchmark commands.

Each function is PURE: only ``console`` + ``data`` → ``console.emit`` calls.
No DB, no system, no MLX, no I/O beyond ``console.emit``.

Data schemas
------------

### ``render_batch(console, data)``

``data`` is a dict with:

    model               : str   — HuggingFace model ID (e.g. "hexgrad/Kokoro-82M")
    voice               : str   — voice label used for the sweep (e.g. "af_heart")
    batch_sizes         : str   — comma-separated batch sizes spec (e.g. "1,2,4,8")
    repeats             : int   — repetitions per batch size rung
    margin              : float — safety margin in GB (e.g. 2.0)

    rows                : list[dict]
        Each dict has:
            batch_size  : int   — number of utterances synthesised in parallel
            total_time  : float — wall-clock time to process the whole batch (seconds)
            cps         : float — characters per second (throughput metric)
            peak_gb     : float — MLX-reported peak wired memory in GB

    safeguard_triggered : bool  — True if sweep stopped early by RULE #1
    run_id              : int | None — DB run ID (display only)

### ``render_voice(console, data)``

``data`` is a dict with:

    model               : str   — HuggingFace model ID
    voice_a             : str   — first voice label (e.g. "af_heart")
    voice_b             : str   — second voice label (e.g. "am_adam")
    repeats             : int   — repetitions per condition
    margin              : float — safety margin in GB

    rows                : list[dict]
        Each dict has:
            cond_type   : str   — condition label: "warm_switch" or "cold_load"
            voice_from  : str   — voice before the switch
            voice_to    : str   — voice after the switch
            duration_ms : float — measured latency in milliseconds

    safeguard_triggered : bool  — True if sweep stopped early by RULE #1
    run_id              : int | None — DB run ID (display only)

### ``render_cache(console, data)``

``data`` is a dict with:

    model               : str   — HuggingFace model ID
    cache_sizes         : str   — comma-separated cache size spec (e.g. "1,2,4,8")
    margin              : float — safety margin in GB

    rows                : list[dict]
        Each dict has:
            cache_size  : int   — number of voices held in the cache
            os_wired_gb : float — OS vm_stat wired pages in GB (measured overhead)
            peak_gb     : float — MLX-reported peak wired memory in GB

    safeguard_triggered : bool  — True if sweep stopped early by RULE #1
    run_id              : int | None — DB run ID (display only)

### ``render_safeguard(console, data)``

``data`` is a dict with:

    note        : str         — human-readable reason from the worker
                                (e.g. "predicted peak 16.4 GB exceeds safe ceiling 15.18 GB")
    batch_size  : int | None  — the batch-size / cache-size rung that triggered the stop
    peak_gb     : float | None — the predicted or measured peak at trigger point
    wall_gb     : float | None — the crash-wall limit
    safe_gb     : float | None — the safe ceiling (wall minus margin)
    next_cmd    : str | None   — the command to adjust and retry

### ``render_preflight_abort(console, data)``

``data`` is a dict with:

    reason       : str          — why the run was refused
    predicted_gb : float | None — predicted peak memory demand
    available_gb : float | None — estimated available safe headroom
    wall_gb      : float | None — crash-wall limit
    margin_gb    : float | None — configured safety margin
    next_cmd     : str | None   — suggested follow-up command
"""
from __future__ import annotations


# --------------------------------------------------------------------------- #
# render_batch — concurrency sweep (batch size vs throughput)
# --------------------------------------------------------------------------- #

def render_batch(console, data: dict) -> None:
    """Render the Kokoro TTS batch-size concurrency sweep results.

    Normal output:
        Section header + config status line, results table (batch size, total
        time, chars/sec, peak GB).  Safeguard early-stop shown via ``guidance``.
        Ends with ``next_block``.

    Verbose appends a ``raw`` appendix with per-rung full-precision detail.
    """
    model = data["model"]
    voice = data["voice"]
    batch_sizes = data.get("batch_sizes", "")
    repeats = data.get("repeats", 1)
    margin = data.get("margin", 2.0)
    rows = data.get("rows", [])
    safeguard_triggered = data.get("safeguard_triggered", False)
    run_id = data.get("run_id")

    # ── section header + config ──────────────────────────────────────────────
    console.emit(console.section("Kokoro TTS Batch Concurrency Benchmark"))
    console.emit(console.status_line([
        (model, "accent"),
        (f"voice {voice}", "label"),
        (f"batches {batch_sizes}", "gloss"),
        (f"{repeats} rep/batch", "gloss"),
        (f"margin {margin:.1f} GB", "gloss"),
    ]))
    console.emit()

    # ── safeguard early-stop ─────────────────────────────────────────────────
    if safeguard_triggered:
        console.emit(console.style("warn",
            "⚠  Sweep stopped early — RULE #1 memory safeguard triggered."
        ))
        console.emit()

    # ── results table ────────────────────────────────────────────────────────
    if rows:
        columns = [
            ("Batch size",   "r", "metric"),
            ("Total time (s)", "r", "value"),
            ("chars/sec",    "r", "value"),
            ("Peak GB",      "r", "metric"),
        ]
        table_rows = []
        for r in rows:
            table_rows.append((
                str(r["batch_size"]),
                f"{r['total_time']:.2f}",
                f"{r['cps']:.1f}",
                f"{r['peak_gb']:.2f}",
            ))
        console.emit(console.table(columns, table_rows))
        console.emit()

    # ── verbose appendix ─────────────────────────────────────────────────────
    if console.verbose and rows:
        raw_lines = []
        for r in rows:
            raw_lines.append(
                f"  batch={r['batch_size']:>3}"
                f"  total={r['total_time']:.3f}s"
                f"  cps={r['cps']:.2f}"
                f"  peak={r['peak_gb']:.3f}GB"
            )
        raw_out = console.raw("per-rung detail (--verbose)", raw_lines)
        if raw_out:
            console.emit(raw_out)
            console.emit()

    # ── next block ───────────────────────────────────────────────────────────
    console.emit(console.next_block([
        ("wmx-suite benchmark kokoro-voice", "measure voice-switch latency (warm vs cold)"),
        ("wmx-suite benchmark kokoro-cache", "measure voice-cache memory overhead"),
    ]))


# --------------------------------------------------------------------------- #
# render_voice — voice-switch latency (warm switch vs cold load)
# --------------------------------------------------------------------------- #

def render_voice(console, data: dict) -> None:
    """Render the Kokoro TTS voice-switching latency benchmark results.

    Normal output:
        Section header + config status line, table of condition/from/to/latency
        rows with warm-switch vs cold-load rows glossed in the header.
        Ends with ``next_block``.

    Verbose appends a ``raw`` appendix with full-precision timings.
    """
    model = data["model"]
    voice_a = data.get("voice_a", "")
    voice_b = data.get("voice_b", "")
    repeats = data.get("repeats", 1)
    margin = data.get("margin", 2.0)
    rows = data.get("rows", [])
    safeguard_triggered = data.get("safeguard_triggered", False)
    run_id = data.get("run_id")

    # ── section header + config ──────────────────────────────────────────────
    console.emit(console.section("Kokoro TTS Voice-Switch Latency Benchmark"))
    console.emit(console.status_line([
        (model, "accent"),
        (f"{voice_a} → {voice_b}", "label"),
        (f"{repeats} reps", "gloss"),
        (f"margin {margin:.1f} GB", "gloss"),
    ]))
    console.emit()

    # ── safeguard early-stop ─────────────────────────────────────────────────
    if safeguard_triggered:
        console.emit(console.style("warn",
            "⚠  Sweep stopped early — RULE #1 memory safeguard triggered."
        ))
        console.emit()

    # ── results table ────────────────────────────────────────────────────────
    if rows:
        columns = [
            ("Condition",       "l", "value"),
            ("Voice from",      "l", "value"),
            ("Voice to",        "l", "value"),
            ("Latency (ms)",    "r", "metric"),
        ]
        table_rows = []
        for r in rows:
            # highlight warm vs cold in the condition cell
            cond = r["cond_type"]
            if cond == "warm_switch":
                cond_cell = (cond, "good")
            elif cond == "cold_load":
                cond_cell = (cond, "warn")
            else:
                cond_cell = (cond, "value")
            table_rows.append((
                cond_cell,
                r["voice_from"],
                r["voice_to"],
                f"{r['duration_ms']:.1f}",
            ))
        console.emit(console.table(columns, table_rows))
        console.emit()

        # inline legend below the table
        console.emit(console.style("gloss",
            "  warm_switch   voice already resident in cache — only embeddings swap\n"
            "  cold_load     voice not in cache — full embedding load from disk"
        ))
        console.emit()

    # ── verbose appendix ─────────────────────────────────────────────────────
    if console.verbose and rows:
        raw_lines = []
        for r in rows:
            raw_lines.append(
                f"  cond={r['cond_type']:<12}"
                f"  from={r['voice_from']:<12}"
                f"  to={r['voice_to']:<12}"
                f"  duration={r['duration_ms']:.3f}ms"
            )
        raw_out = console.raw("per-rung detail (--verbose)", raw_lines)
        if raw_out:
            console.emit(raw_out)
            console.emit()

    # ── next block ───────────────────────────────────────────────────────────
    console.emit(console.next_block([
        ("wmx-suite benchmark kokoro-batch", "throughput sweep across batch sizes"),
        ("wmx-suite benchmark kokoro-cache", "measure voice-cache memory overhead"),
    ]))


# --------------------------------------------------------------------------- #
# render_cache — voice-cache memory overhead sweep
# --------------------------------------------------------------------------- #

def render_cache(console, data: dict) -> None:
    """Render the Kokoro TTS voice-cache memory overhead sweep results.

    Normal output:
        Section header + config status line, table of cache-size vs overhead
        (OS wired GB and peak GB).  Ends with ``next_block``.

    Verbose appends a ``raw`` appendix with full-precision measurements.
    """
    model = data["model"]
    cache_sizes = data.get("cache_sizes", "")
    margin = data.get("margin", 2.0)
    rows = data.get("rows", [])
    safeguard_triggered = data.get("safeguard_triggered", False)
    run_id = data.get("run_id")

    # ── section header + config ──────────────────────────────────────────────
    console.emit(console.section("Kokoro TTS Voice-Cache Memory Benchmark"))
    console.emit(console.status_line([
        (model, "accent"),
        (f"cache sizes {cache_sizes}", "gloss"),
        (f"margin {margin:.1f} GB", "gloss"),
    ]))
    console.emit()

    # ── safeguard early-stop ─────────────────────────────────────────────────
    if safeguard_triggered:
        console.emit(console.style("warn",
            "⚠  Sweep stopped early — RULE #1 memory safeguard triggered."
        ))
        console.emit()

    # ── results table ────────────────────────────────────────────────────────
    if rows:
        columns = [
            ("Cache size (voices)", "r", "metric"),
            ("Overhead GB",         "r", "value"),
            ("Peak GB",             "r", "metric"),
        ]
        table_rows = []
        for r in rows:
            table_rows.append((
                str(r["cache_size"]),
                f"{r['os_wired_gb']:.3f}",
                f"{r['peak_gb']:.3f}",
            ))
        console.emit(console.table(columns, table_rows))
        console.emit()

        # inline legend
        console.emit(console.style("gloss",
            "  overhead GB   OS vm_stat wired pages — true memory cost of the cache\n"
            "  peak GB       MLX-reported peak allocation for the rung"
        ))
        console.emit()

    # ── verbose appendix ─────────────────────────────────────────────────────
    if console.verbose and rows:
        raw_lines = []
        for r in rows:
            raw_lines.append(
                f"  cache={r['cache_size']:>3}"
                f"  os_wired={r['os_wired_gb']:.6f}GB"
                f"  peak={r['peak_gb']:.6f}GB"
            )
        raw_out = console.raw("per-rung detail (--verbose)", raw_lines)
        if raw_out:
            console.emit(raw_out)
            console.emit()

    # ── next block ───────────────────────────────────────────────────────────
    console.emit(console.next_block([
        ("wmx-suite benchmark kokoro-batch", "throughput sweep across batch sizes"),
        ("wmx-suite benchmark kokoro-voice", "measure voice-switch latency"),
    ]))


# --------------------------------------------------------------------------- #
# render_safeguard — RULE #1 safeguard-triggered stop
# --------------------------------------------------------------------------- #

def render_safeguard(console, data: dict) -> None:
    """Render a RULE #1 safeguard-triggered early-stop message.

    RENDER-ONLY — never re-implements gating logic.

    Data schema: see module docstring (``render_safeguard``).
    """
    note = data.get("note", "predicted peak would exceed the crash wall")
    batch_size = data.get("batch_size")
    peak_gb = data.get("peak_gb")
    wall_gb = data.get("wall_gb")
    safe_gb = data.get("safe_gb")
    next_cmd = data.get("next_cmd",
                        "wmx-suite benchmark kokoro-batch --batch-sizes <smaller>")

    why_lines = [note]
    if batch_size is not None:
        why_lines.append(f"triggered at batch size {batch_size}")
    if peak_gb is not None and safe_gb is not None:
        why_lines.append(
            f"predicted {peak_gb:.2f} GB > safe ceiling {safe_gb:.2f} GB"
        )
    if wall_gb is not None:
        why_lines.append(f"crash wall is {wall_gb:.2f} GB")

    console.emit(console.guidance(
        "Sweep stopped — memory safeguard triggered",
        why_lines,
        [
            (next_cmd, "use a smaller batch to stay under the safe ceiling"),
            ("wmx-suite benchmark kokoro-baseline", "confirm the active memory floor"),
        ],
    ))


# --------------------------------------------------------------------------- #
# render_preflight_abort — pre-flight refusal
# --------------------------------------------------------------------------- #

def render_preflight_abort(console, data: dict) -> None:
    """Render a pre-flight refusal before any synthesis starts.

    RENDER-ONLY — never re-implements gating logic.

    Data schema: see module docstring (``render_preflight_abort``).
    """
    reason = data.get("reason", "pre-flight check refused this run")
    predicted_gb = data.get("predicted_gb")
    available_gb = data.get("available_gb")
    wall_gb = data.get("wall_gb")
    margin_gb = data.get("margin_gb")
    next_cmd = data.get("next_cmd",
                        "wmx-suite benchmark kokoro-batch --batch-sizes <smaller>")

    why_lines = [reason]
    if predicted_gb is not None:
        why_lines.append(f"predicted load: {predicted_gb:.2f} GB")
    if available_gb is not None:
        why_lines.append(f"safe headroom:  {available_gb:.2f} GB")
    if wall_gb is not None and margin_gb is not None:
        why_lines.append(
            f"crash wall {wall_gb:.2f} GB minus margin {margin_gb:.2f} GB"
            f" = ceiling {wall_gb - margin_gb:.2f} GB"
        )

    console.emit(console.guidance(
        "Won't start — pre-flight check failed",
        why_lines,
        [
            (next_cmd, "adjust the run to fit within the safe ceiling"),
            ("wmx-suite characterize <model>", "re-measure this model's memory profile"),
        ],
    ))
