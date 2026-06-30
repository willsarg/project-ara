# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Render functions for ``wmx-suite calibrate`` — progress + outcome.

PURE: ``(console, data)`` in, styled output via console primitives. Data schemas
documented per function.
"""
from __future__ import annotations


def render_header(console, data: dict) -> None:
    """data: model, weights_gb, threshold_gb, kv_mode"""
    c = console
    c.emit(c.section("Calibrating this machine's cold-start estimate")
           + c.style("dim", f"   (using {data['model']})"))
    c.emit(c.style("dim", "  measuring the fixed memory overhead so future "
                          "predictions start accurate."))
    c.emit()
    c.emit(c.field("safe budget", f"{data['threshold_gb']:.2f} GB",
                   "the ceiling these probes stay under"))
    c.emit(c.field("model", data["model"],
                   f"{data['weights_gb']:.1f} GB weights · {data['kv_mode']} KV"))
    c.emit()


def render_rung(console, data: dict) -> None:
    """data: ctx, delta_gb, repeats"""
    c = console
    c.emit(
        "  " + c.glyph("ok") + " "
        + c.style("value", f"{data['ctx']:>6,} tok")
        + c.style("dim", "   ")
        + c.style("metric", f"+{data['delta_gb']:.2f} GB over base")
        + c.style("dim", f"   (median of {data['repeats']})")
    )


def render_abort(console, data: dict) -> None:
    """data: reason (str), kind ("memory"|"load"|"fit")"""
    kind = data.get("kind", "fit")
    if kind == "load":
        # The chosen model failed to load (e.g. an incompatible checkpoint) —
        # the fix is a different model, not more memory.
        tries = [
            ("wmx-suite calibrate --model <other>", "calibrate with a model that loads"),
            ("huggingface.co/mlx-community", "find a working build to download"),
        ]
    elif kind == "memory":
        tries = [
            ("free up memory", "close other apps, then retry"),
            ("wmx-suite calibrate --model <smaller>", "or calibrate with a smaller model"),
        ]
    else:  # fit / unknown
        tries = [
            ("wmx-suite calibrate", "run it again (transient measurement noise)"),
            ("wmx-suite calibrate --model <other>", "or try a different model"),
        ]
    console.emit(console.guidance(
        "Couldn't calibrate this machine.", [data["reason"]], tries))


def render_summary(console, data: dict) -> None:
    """data: machine, model, n_points, measured_overhead_gb, default_overhead_gb,
    fixed_overhead_gb, changed (bool)
    """
    c = console
    c.emit()
    c.emit(c.style("good", "✓ Calibrated this machine's cold-start estimate."))
    c.emit(c.field("machine", data["machine"], "where the profile is stored"))
    c.emit(c.field("model used", data["model"], f"{data['n_points']} probe rungs"))
    c.emit(c.field("measured", f"{data['measured_overhead_gb']:.2f} GB overhead",
                   f"default is {data['default_overhead_gb']:.2f} GB"))
    if data["changed"]:
        note = "tightened the estimate for this machine"
    else:
        note = "kept the safe default (measurement didn't beat the floor)"
    c.emit(c.field("stored", f"{data['fixed_overhead_gb']:.2f} GB", note))
    c.emit(c.next_block([
        ("wmx-suite system", "confirm the updated calibration"),
        ("wmx-suite health", "re-check per-model go/no-go with the tighter estimate"),
    ]))
