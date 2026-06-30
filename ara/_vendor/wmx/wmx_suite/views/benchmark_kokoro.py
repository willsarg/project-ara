# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Pure render functions for Kokoro TTS benchmark commands.

Each function is PURE: only ``console`` + ``data`` → ``console.emit`` calls.
No DB, no system, no MLX, no I/O beyond ``console.emit``.

Visual source of truth: ``docs/mockups/cli-output-mockup.html``.
"""
from __future__ import annotations


# --------------------------------------------------------------------------- #
# render_perf — Kokoro TTS throughput sweep
# --------------------------------------------------------------------------- #

def render_perf(console, data: dict) -> None:
    """Render the Kokoro TTS performance (throughput) sweep results.

    Data schema
    -----------
    ``data`` is a dict with:

        model   : str   — HuggingFace model ID (e.g. "hexgrad/Kokoro-82M")
        voice   : str   — voice label (e.g. "af_heart")
        lengths : str   — sweep spec as passed to the CLI (e.g. "50,100,200")
        repeats : int   — repetitions per sweep length
        margin  : float — safety margin in GB (e.g. 2.0)

        rows    : list[dict]
            Each dict has:
                length       : int   — text length in chars
                audio_dur    : float — synthesised audio duration in seconds
                compute_time : float — wall-clock synthesis time in seconds
                rtf          : float — real-time factor (compute / audio)
                cps          : float — characters per second
                peak_gb      : float — MLX-reported peak wired memory in GB
                os_wired_gb  : float | None — OS vm_stat wired pages in GB

        safeguard_triggered : bool — True if sweep stopped early by RULE #1
        run_id  : int | None — DB run ID (for reference only; may be omitted)
    """
    model = data["model"]
    voice = data["voice"]
    lengths = data.get("lengths", "")
    repeats = data.get("repeats", 1)
    margin = data.get("margin", 2.0)
    rows = data.get("rows", [])
    safeguard_triggered = data.get("safeguard_triggered", False)
    run_id = data.get("run_id")

    # Section header + config status line
    console.emit(console.section("Kokoro TTS Performance Benchmark"))
    console.emit(console.status_line([
        (model, "accent"),
        (f"voice {voice}", "label"),
        (f"sweeps {lengths}", "gloss"),
        (f"{repeats} rep/len", "gloss"),
        (f"margin {margin:.1f} GB", "gloss"),
    ]))
    console.emit()

    if safeguard_triggered:
        console.emit(console.style("warn",
            "⚠  Sweep stopped early — RULE #1 memory safeguard triggered."
        ))
        console.emit()

    if rows:
        columns = [
            ("Length (char)", "r", "metric"),
            ("Audio (s)",     "r", "value"),
            ("Compute (s)",   "r", "value"),
            ("RTF",           "r", "value"),
            ("chars/sec",     "r", "value"),
            ("Peak GB",       "r", "metric"),
            ("OS Wired GB",   "r", "metric"),
        ]
        table_rows = []
        for r in rows:
            os_wired = r.get("os_wired_gb")
            os_wired_str = f"{os_wired:.2f}" if os_wired is not None else "—"
            table_rows.append((
                str(r["length"]),
                f"{r['audio_dur']:.2f}",
                f"{r['compute_time']:.2f}",
                f"{r['rtf']:.4f}",
                f"{r['cps']:.1f}",
                f"{r['peak_gb']:.2f}",
                os_wired_str,
            ))
        console.emit(console.table(columns, table_rows))
        console.emit()

    # Verbose appendix: per-rung raw detail
    if console.verbose and rows:
        raw_lines = []
        for r in rows:
            os_wired = r.get("os_wired_gb")
            os_wired_str = f"{os_wired:.3f}" if os_wired is not None else "—"
            raw_lines.append(
                f"  length={r['length']:>5}  audio={r['audio_dur']:.3f}s"
                f"  compute={r['compute_time']:.3f}s  rtf={r['rtf']:.5f}"
                f"  cps={r['cps']:.2f}  peak={r['peak_gb']:.3f}GB"
                f"  os_wired={os_wired_str}GB"
            )
        raw_out = console.raw("per-rung detail (--verbose)", raw_lines)
        if raw_out:
            console.emit(raw_out)
            console.emit()

    run_ref = f"  run id {run_id}" if run_id is not None else ""
    next_items = [
        ("wmx-suite benchmark kokoro-ttfa", "measure streaming vs non-streaming latency"),
        ("wmx-suite benchmark kokoro-baseline", "measure the active synthesis memory floor"),
    ]
    console.emit(console.next_block(next_items))


# --------------------------------------------------------------------------- #
# render_ttfa — Kokoro TTS streaming latency (TTFA) sweep
# --------------------------------------------------------------------------- #

def render_ttfa(console, data: dict) -> None:
    """Render the Kokoro TTS Time-to-First-Audio (TTFA) latency sweep.

    Data schema
    -----------
    ``data`` is a dict with:

        model   : str   — HuggingFace model ID
        voice   : str   — voice label
        lengths : str   — sweep spec (e.g. "50,100,200")
        repeats : int   — repetitions per length
        margin  : float — safety margin in GB

        rows    : list[dict]
            Each dict has:
                length             : int   — text length in chars
                ttfa_sec           : float — time to first audio chunk in seconds
                total_sec          : float — total synthesis wall time in seconds
                speedup_ratio      : float — streaming speedup vs non-streaming
                first_chunk_dur    : float — duration of the first audio chunk in seconds
                peak_gb            : float — MLX-reported peak wired memory in GB

        safeguard_triggered : bool
        run_id  : int | None
    """
    model = data["model"]
    voice = data["voice"]
    lengths = data.get("lengths", "")
    repeats = data.get("repeats", 1)
    margin = data.get("margin", 2.0)
    rows = data.get("rows", [])
    safeguard_triggered = data.get("safeguard_triggered", False)
    run_id = data.get("run_id")

    console.emit(console.section("Kokoro TTS Time-to-First-Audio Benchmark"))
    console.emit(console.status_line([
        (model, "accent"),
        (f"voice {voice}", "label"),
        (f"sweeps {lengths}", "gloss"),
        (f"{repeats} rep/len", "gloss"),
        (f"margin {margin:.1f} GB", "gloss"),
    ]))
    console.emit()

    if safeguard_triggered:
        console.emit(console.style("warn",
            "⚠  Sweep stopped early — RULE #1 memory safeguard triggered."
        ))
        console.emit()

    if rows:
        columns = [
            ("Length (char)",    "r", "metric"),
            ("TTFA (s)",         "r", "value"),
            ("Total (s)",        "r", "value"),
            ("Speedup",          "r", "good"),
            ("First Chunk (s)",  "r", "value"),
            ("Peak GB",          "r", "metric"),
        ]
        table_rows = []
        for r in rows:
            table_rows.append((
                str(r["length"]),
                f"{r['ttfa_sec']:.3f}",
                f"{r['total_sec']:.3f}",
                f"{r['speedup_ratio']:.1f}x",
                f"{r['first_chunk_dur']:.2f}",
                f"{r['peak_gb']:.2f}",
            ))
        console.emit(console.table(columns, table_rows))
        console.emit()

    # Verbose appendix
    if console.verbose and rows:
        raw_lines = []
        for r in rows:
            raw_lines.append(
                f"  length={r['length']:>5}  ttfa={r['ttfa_sec']:.4f}s"
                f"  total={r['total_sec']:.4f}s  speedup={r['speedup_ratio']:.2f}x"
                f"  first_chunk={r['first_chunk_dur']:.3f}s"
                f"  peak={r['peak_gb']:.3f}GB"
            )
        raw_out = console.raw("per-rung detail (--verbose)", raw_lines)
        if raw_out:
            console.emit(raw_out)
            console.emit()

    console.emit(console.next_block([
        ("wmx-suite benchmark kokoro",          "throughput sweep across text lengths"),
        ("wmx-suite benchmark kokoro-baseline", "measure the active synthesis memory floor"),
    ]))


# --------------------------------------------------------------------------- #
# render_baseline — Kokoro TTS active memory floor
# --------------------------------------------------------------------------- #

def render_baseline(console, data: dict) -> None:
    """Render the Kokoro TTS active-synthesis memory baseline result.

    Data schema
    -----------
    ``data`` is a dict with:

        model       : str   — HuggingFace model ID
        voice       : str   — voice label
        margin      : float — safety margin in GB

        baseline_gb : float — settled OS-wired baseline before model load (GB)
        active_gb   : float — OS-wired with model loaded and synthesis active (GB)
        overhead_gb : float — active_gb minus baseline_gb (GB)

        run_id      : int | None
    """
    model = data["model"]
    voice = data["voice"]
    margin = data.get("margin", 2.0)
    baseline_gb = data["baseline_gb"]
    active_gb = data["active_gb"]
    overhead_gb = data["overhead_gb"]

    console.emit(console.section("Kokoro TTS Active Memory Baseline"))
    console.emit(console.status_line([
        (model, "accent"),
        (f"voice {voice}", "label"),
        (f"margin {margin:.1f} GB", "gloss"),
    ]))
    console.emit()

    console.emit(console.field(
        "System baseline", f"{baseline_gb:.3f} GB",
        "settled OS-wired pages before Kokoro loads",
    ))
    console.emit(console.field(
        "Active synthesis", f"{active_gb:.3f} GB",
        "OS-wired pages with model loaded and synthesis running",
        value_role="metric",
    ))
    console.emit(console.field(
        "Static overhead", f"{overhead_gb:.3f} GB",
        "active minus baseline — cost of keeping Kokoro resident",
        value_role="warn" if overhead_gb > 1.5 else "good",
    ))
    console.emit()

    # Verbose appendix: repeat the raw scalars for scripting
    if console.verbose:
        raw_lines = [
            f"  baseline_gb   {baseline_gb:.6f}",
            f"  active_gb     {active_gb:.6f}",
            f"  overhead_gb   {overhead_gb:.6f}",
        ]
        raw_out = console.raw("raw measurement (--verbose)", raw_lines)
        if raw_out:
            console.emit(raw_out)
            console.emit()

    console.emit(console.next_block([
        ("wmx-suite benchmark kokoro",      "throughput sweep across text lengths"),
        ("wmx-suite benchmark kokoro-ttfa", "streaming latency sweep"),
    ]))


# --------------------------------------------------------------------------- #
# render_safeguard — RULE #1 safeguard-triggered stop (shared)
# --------------------------------------------------------------------------- #

def render_safeguard(console, data: dict) -> None:
    """Render a RULE #1 safeguard-triggered early-stop message.

    This is a RENDER-ONLY function — it never re-implements gating logic.

    Data schema
    -----------
    ``data`` is a dict with:

        note        : str   — human-readable reason from the worker
                              (e.g. "predicted peak 16.4 GB exceeds wall 15.18 GB")
        length      : int | None — the text length rung that triggered the stop
        peak_gb     : float | None — the predicted or measured peak at trigger point
        wall_gb     : float | None — the crash-wall limit
        safe_gb     : float | None — the safe ceiling (wall minus margin)
        next_cmd    : str | None   — the command to adjust and retry
    """
    note = data.get("note", "predicted peak would exceed the crash wall")
    length = data.get("length")
    peak_gb = data.get("peak_gb")
    wall_gb = data.get("wall_gb")
    safe_gb = data.get("safe_gb")
    next_cmd = data.get("next_cmd", "wmx-suite benchmark kokoro --lengths <shorter>")

    why_lines = [note]
    if length is not None:
        why_lines.append(f"triggered at length {length} chars")
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
            (next_cmd, "use shorter lengths to stay under the safe ceiling"),
            ("wmx-suite benchmark kokoro-baseline", "confirm the active memory floor"),
        ],
    ))


# --------------------------------------------------------------------------- #
# render_preflight_abort — pre-flight refusal (shared)
# --------------------------------------------------------------------------- #

def render_preflight_abort(console, data: dict) -> None:
    """Render a pre-flight refusal before any synthesis starts.

    This is a RENDER-ONLY function — it never re-implements gating logic.

    Data schema
    -----------
    ``data`` is a dict with:

        reason      : str         — why the run was refused
                                    (e.g. "not enough headroom to load model safely")
        predicted_gb : float | None — predicted peak memory demand
        available_gb : float | None — estimated available safe headroom
        wall_gb      : float | None — crash-wall limit
        margin_gb    : float | None — configured safety margin
        next_cmd     : str | None   — suggested follow-up command
    """
    reason = data.get("reason", "pre-flight check refused this run")
    predicted_gb = data.get("predicted_gb")
    available_gb = data.get("available_gb")
    wall_gb = data.get("wall_gb")
    margin_gb = data.get("margin_gb")
    next_cmd = data.get("next_cmd", "wmx-suite benchmark kokoro --margin <smaller>")

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
