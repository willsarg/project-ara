# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Render function for ``wmx-suite system``.

Data schema
-----------
``data`` is a dict with the following keys:

    device          : str   — chip name, e.g. "Apple M4 Pro"
    total_gb        : float — total physical RAM in GB (e.g. 25.77)
    wall_gb         : float — crash wall in GB (max_recommended_working_set_size, e.g. 17.18)
    safe_budget_gb  : float — wall_gb minus margin (e.g. 15.18)
    wired_gb        : float — memory wired right now / baseline (e.g. 3.14)
    free_headroom_gb: float — safe_budget_gb minus wired_gb (e.g. 12.04)
    max_buffer_gb   : float — largest single Metal allocation allowed (e.g. 12.88)
    swap_free_gb    : float | None — swap free in GB, or None if unknown
    swap_warn       : bool  — True when swap_free_gb is not None and < 2.0
    margin_gb       : float — configured margin in GB (e.g. 2.0)
    margin_source   : str   — where the margin came from, e.g. "default (set WMX_SUITE_MARGIN_GB or --margin)"
    wall_bytes      : int   — raw wall bytes from the kernel (e.g. 18446744073)
    calibrated      : bool  — True if a calibration profile exists for this machine
    cal_model_short : str   — short model name used for calibration, e.g. "Qwen2.5-VL-7B-Instruct-4bit"
    cal_overhead_gb : float — fixed overhead from the calibration profile (e.g. 1.0)
    cal_date        : str   — ISO date portion of calibrated_at (e.g. "2026-06-14")
    wired_sample    : str   — description of the sampling approach, e.g. "median of 3 @ 0.2s"
"""
from __future__ import annotations

_PREFIX = "mlx-community/"


def _strip(hf_id: str) -> str:
    return hf_id[len(_PREFIX):] if hf_id.startswith(_PREFIX) else hf_id


def render(console, data: dict) -> None:
    """Render the system memory budget to *console*.

    Normal output:
        device field, blank line, "memory budget" section (total RAM → crash wall
        → safe budget → wired now → free headroom), blank line, "details" section
        (max GPU buffer, swap, calibration), next_block.

    Verbose appends a raw appendix: wall source/bytes, margin source, cold-start
    calibration detail, wired sampling note.
    """
    c = console

    # ── device ──────────────────────────────────────────────────────────────
    c.emit(c.field("device", data["device"]))
    c.emit()

    # ── memory budget ───────────────────────────────────────────────────────
    c.emit(c.section("memory budget"))
    c.emit(c.field(
        "total RAM",
        f"{data['total_gb']:.2f} GB",
        "physical memory installed",
    ))
    c.emit(c.field(
        "crash wall",
        f"{data['wall_gb']:.2f} GB",
        "most the GPU can safely lock — past this the Mac can hard-lock",
        value_role="warn",
    ))
    c.emit(c.field(
        "  safe budget",
        f"{data['safe_budget_gb']:.2f} GB",
        f"crash wall − {data['margin_gb']:g} GB margin — the suite never plans above this",
        value_role="good",
    ))
    c.emit(c.field(
        "wired now",
        f"{data['wired_gb']:.2f} GB",
        "memory locked right now (your starting point)",
    ))
    c.emit(c.field(
        "free headroom",
        f"{data['free_headroom_gb']:.2f} GB",
        "safe budget − wired now → what a model may use",
        value_role="metric",
    ))
    c.emit()

    # ── details ─────────────────────────────────────────────────────────────
    c.emit(c.section("details"))
    c.emit(c.field(
        "max GPU buffer",
        f"{data['max_buffer_gb']:.2f} GB",
        "largest single allocation Metal allows",
    ))

    swap = data["swap_free_gb"]
    if swap is None:
        c.emit(c.field("swap free", "unknown"))
    elif data["swap_warn"]:
        swap_val = f"{swap:.2f} GB  " + c.glyph("warn") + " tiny"
        c.emit(c.field(
            "swap free",
            swap_val,
            "almost no fallback if you cross the wall",
            value_role="warn",
        ))
    else:
        c.emit(c.field("swap free", f"{swap:.2f} GB"))

    if data["calibrated"]:
        cal_val = c.style("good", "tuned")
        cal_gloss = (
            f"this machine's cold-start overhead is measured"
            f" ({data['cal_model_short']}, {data['cal_date']})"
        )
        c.emit(c.field("calibration", cal_val, cal_gloss))
    else:
        c.emit(c.field(
            "calibration",
            c.style("warn", "uncalibrated"),
            "run 'wmx-suite calibrate' to tune for this machine",
        ))

    # ── verbose raw appendix ────────────────────────────────────────────────
    raw_lines = [
        c.field("  wall source", "max_recommended_working_set_size", value_role="dim"),
        c.field("  wall bytes", f"{data['wall_bytes']:,} B", value_role="dim"),
        c.field(
            "  margin",
            f"{data['margin_gb']:.2f} GB",
            data["margin_source"],
            value_role="dim",
        ),
    ]
    if data["calibrated"]:
        raw_lines.append(c.field(
            "  cold-start",
            f"overhead {data['cal_overhead_gb']:.2f} GB",
            f"{data['cal_model_short']} @ {data['cal_date']}T00:00:00Z",
            value_role="dim",
        ))
    raw_lines.append(c.field(
        "  wired sample",
        data["wired_sample"],
        "vm_stat 'Pages wired down'",
        value_role="dim",
    ))
    raw_out = c.raw("raw (--verbose)", raw_lines)
    if raw_out:
        c.emit()
        c.emit(raw_out)

    # ── next block ──────────────────────────────────────────────────────────
    c.emit(c.next_block([
        ("wmx-suite health",    "which of your models can run right now, and how big a context"),
        ("wmx-suite list",      "safe context ceilings already measured on this machine"),
        ("wmx-suite calibrate", "re-tune this machine's cold-start estimate"),
    ]))
