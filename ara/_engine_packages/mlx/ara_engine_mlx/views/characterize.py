# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Render functions for native MLX characterization progress + outcome.

``characterize`` is a streaming command (it probes context rungs one at a time),
so it has several small renderers instead of one. Each is PURE: ``(console,
data)`` in, styled output emitted via the console primitives.

Data schemas are documented per function below.
"""
from __future__ import annotations


def _short(model_id: str) -> str:
    return model_id.split("/", 1)[-1] if "/" in model_id else model_id


def render_header(console, data: dict) -> None:
    """Opening block.

    data: model, cache_type, kv_mode, wall_gb, safe_budget_gb, baseline_gb,
          est_gb, weights_gb
    """
    c = console
    c.emit(c.section(f"Characterizing {_short(data['model'])}")
           + c.style("dim", f"   ({data['cache_type']} · {data['kv_mode']} KV)"))
    c.emit(c.style("dim", "  measuring how much context fits — probing upward from "
                          "small, never into the danger zone."))
    c.emit()
    c.emit(c.field("safe budget", f"{data['safe_budget_gb']:.2f} GB",
                   f"crash wall {data['wall_gb']:.2f} GB − margin"))
    c.emit(c.field("memory now", f"{data['baseline_gb']:.2f} GB",
                   "wired before the model loads (your starting point)"))
    c.emit(c.field("pre-flight est", f"{data['est_gb']:.2f} GB",
                   f"rough load footprint from {data['weights_gb']:.1f} GB of weights"))
    c.emit()


def render_note(console, text: str, *, role: str = "dim") -> None:
    """A one-off progress line (e.g. the supervised min-probe steps)."""
    console.emit(console.style(role, "  " + text))


def render_rung(console, data: dict) -> None:
    """A measured context rung (progress).

    data: ctx, os_wired_gb, delta_gb, peak_gb, repeats, spread_gb
    """
    c = console
    c.emit(
        "  " + c.glyph("ok") + " "
        + c.style("value", f"{data['ctx']:>6,} tok")
        + c.style("dim", "   ")
        + c.style("metric", f"{data['os_wired_gb']:.2f} GB wired")
        + c.style("dim", f"   (+{data['delta_gb']:.2f} GB vs base, "
                         f"median of {data['repeats']})")
    )


def render_stop(console, data: dict) -> None:
    """Safe stop — the next rung would breach the budget.

    data: ctx, predicted_gb, safe_budget_gb
    """
    c = console
    c.emit(
        "  " + c.style("warn", "■")
        + c.style("value", f" ceiling reached")
        + c.style("dim", f" — {data['ctx']:,} tok would need ~{data['predicted_gb']:.2f} GB, "
                         f"over the {data['safe_budget_gb']:.2f} GB budget. Stopping safely.")
    )


def render_refusal(console, data: dict) -> None:
    """Pre-flight refusal — won't even probe.

    data: model, kind ("hopeless"|"borderline"), est_gb, threshold_gb, wall_gb
    """
    model = _short(data["model"])
    if data["kind"] == "hopeless":
        why = [f"Its estimated load footprint ({data['est_gb']:.2f} GB) meets or "
               f"exceeds the crash wall ({data['wall_gb']:.2f} GB) — it can't load "
               f"without risking a hard lock, so we never probe it."]
        tries = [
            ("huggingface.co/mlx-community", "browse for a smaller / more-quantized build"),
            ("ara models recommend", "see what does fit right now"),
        ]
    else:  # borderline
        why = [f"Its estimated load footprint ({data['est_gb']:.2f} GB) is above the "
               f"safe budget ({data['threshold_gb']:.2f} GB) but below the wall — the "
               f"estimate may just be pessimistic."]
        tries = [
            (f"ara characterize {model} --engine mlx --min-probe",
             "measure the true base with one supervised safe probe"),
            ("huggingface.co/mlx-community", "or pick a smaller build"),
        ]
    console.emit(console.guidance(
        f"Can't characterize {model} — it won't fit safely on this Mac.", why, tries))


def render_failure(console, data: dict) -> None:
    """The probe couldn't measure the model (e.g. a load error).

    data: model, note (already a one-line summary)
    """
    model = _short(data["model"])
    console.emit(console.guidance(
        f"Couldn't measure {model}.",
        [data.get("note") or "The model failed to load during probing."],
        [("huggingface.co/mlx-community", "find a different build that loads"),
         ("ara models recommend", "see what already fits this Mac")],
    ))


def render_summary(console, data: dict) -> None:
    """Final outcome on success.

    data: model, safe_ctx, hard_wall_ctx, r2, n_points
    """
    c = console
    model = _short(data["model"])
    c.emit()
    c.emit(c.style("good", f"✓ {model} is safe to run up to "
                           f"~{data['safe_ctx']:,} tokens of context."))
    c.emit(c.field("safe context", f"~{data['safe_ctx']:,} tok",
                   "guaranteed to stay under the crash wall"))
    c.emit(c.field("hard wall", f"~{data['hard_wall_ctx']:,} tok",
                   "where memory would actually hit the wall"))
    c.emit(c.field("fit quality", f"R²={data['r2']}",
                   f"from {data['n_points']} measured points"))
    c.emit(c.next_block([
        (f"ara run {model} '<prompt>' --engine mlx", "launch it safely now"),
        (f"ara models show {model}", "inspect its stored model details"),
    ]))
