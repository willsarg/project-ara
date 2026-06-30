# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Render function for ``wmx-suite list``.

Data schema
-----------
``data`` is a dict with the following keys:

    models : list[dict]
        Each dict has:
            hf_id           : str   — full HuggingFace model ID
            loads_gb        : float — human-rounded base load size in GB (e.g. 5.0)
            safe_ctx        : int   — safe context ceiling in tokens (e.g. 28533)
            speed_tps       : float | None  — measured generation speed tok/s, or None
            fit             : str   — confidence label: "good", "ok", or "poor"
            tight           : bool  — True if safe_ctx is dangerously small (show ⚠ tight)

            # verbose-only raw fields
            slope_gb_per_k  : float — GB per 1k tokens slope
            r2              : float — R² of the fit
            hard_wall_ctx   : int   — context where memory hits the crash wall
            n_runs          : int   — number of generation runs logged
"""
from __future__ import annotations

_PREFIX = "mlx-community/"


def _strip(hf_id: str) -> str:
    return hf_id[len(_PREFIX):] if hf_id.startswith(_PREFIX) else hf_id


def _fmt_ctx(tok: int) -> str:
    """Human-round a token count: ~28,500 tok."""
    # Round to nearest 100 for display
    rounded = round(tok / 100) * 100
    return f"~{rounded:,} tok"


def render(console, data: dict) -> None:
    """Render the list of measured models to *console*.

    Normal output:
        Section header, aligned table (model, loads at, safe context, speed, fit)
        with ⚠ tight flag where applicable, inline legend gloss, next_block.
        Crash context is HIDDEN in normal mode.

    Verbose appends a raw-fit table (GB/1k slope, R², crash ctx, runs).
    """
    models = data["models"]

    # Compute column widths
    names = [_strip(m["hf_id"]) for m in models]
    name_w = max(max(len(n) for n in names), len("model"))

    # Build table rows
    columns = [
        ("model",        "l", "value"),
        ("loads at",     "r", "value"),
        ("safe context", "r", "good"),
        ("speed",        "r", "value"),
        ("fit",          "l", "value"),
    ]

    rows = []
    for m, short in zip(models, names):
        loads_str = f"{m['loads_gb']:.1f} GB"

        safe_str = _fmt_ctx(m["safe_ctx"])

        if m["speed_tps"] is not None:
            speed_str = f"~{m['speed_tps']:.0f} tok/s"
        else:
            speed_str = "—"

        fit_str = m["fit"]
        if m.get("tight"):
            fit_cell = (fit_str, "value")
        else:
            fit_cell = (fit_str, "value")

        # Build the row; the tight flag is appended as an extra inline piece
        # but the table primitive has fixed columns so we embed the flag in fit
        if m.get("tight"):
            fit_with_flag = fit_str + "   " + console.style("warn", "⚠ tight")
        else:
            fit_with_flag = fit_str

        rows.append((
            short,
            (loads_str, "value"),
            (safe_str, "good"),
            (speed_str, "value"),
            (fit_with_flag, "value"),
        ))

    console.emit(console.section("Models measured on this machine"))
    console.emit()
    console.emit(console.table(columns, rows))

    # Verbose appendix: raw fit table
    if console.verbose:
        verbose_columns = [
            ("model",     "l", "dim"),
            ("GB/1k",     "r", "dim"),
            ("R²",        "r", "dim"),
            ("crash ctx", "r", "dim"),
            ("runs",      "r", "dim"),
        ]
        verbose_rows = []
        for m, short in zip(models, names):
            verbose_rows.append((
                (short,                        "dim"),
                (f"{m['slope_gb_per_k']:.4f}", "dim"),
                (str(m["r2"]),                 "dim"),
                (f"{m['hard_wall_ctx']:,}",    "dim"),
                (str(m["n_runs"]),             "dim"),
            ))
        raw_out = console.raw(
            "raw fit (--verbose)",
            [],
        )
        # raw() returns "" when not verbose; but we ARE verbose here.
        # We build the table separately and emit it under the raw section header.
        console.emit()
        console.emit(
            console.section("raw fit (--verbose)")
            + console.style("dim",
                "   safe = ceiling kept below the wall; crash = where the fit hits it")
        )
        console.emit(console.table(verbose_columns, verbose_rows))

    # Legend gloss (always shown)
    console.emit()
    console.emit(console.style("gloss",
        "  loads at      memory just to load the model\n"
        "  safe context  largest context we guarantee won't cross the crash wall\n"
        "  speed         measured tokens/sec from real runs (— = not run yet)\n"
        "  fit           confidence of the measurement (good / ok / poor)"
    ))

    console.emit(console.next_block([
        ("wmx-suite run --model <model>", "launch one safely"),
    ]))
