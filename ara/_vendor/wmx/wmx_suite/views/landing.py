# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Render function for ``wmx-suite`` (no args / --help landing screen).

Data schema
-----------
``data`` is a dict with the following keys:

    device          : str   — chip name, e.g. "Apple M4 Pro"
    free_gb         : float — free headroom in GB (safe_budget − wired now), e.g. 12.04
    safe_budget_gb  : float — safe budget in GB (wall − margin), e.g. 15.18
    models_ready    : int   — count of characterized models ready to run (e.g. 3)
    calibrated      : bool  — True if a calibration profile exists for this machine
"""
from __future__ import annotations

# Width for command column in group listings (must align with widest entry in
# the standard groups; benchmarks use their own auto-width).
_CMD_W = 22


def _cmd(console, name: str, why: str) -> str:
    """One command row: accent-colored name padded to _CMD_W, dim gloss."""
    return (
        "  "
        + console.style("accent", name.ljust(_CMD_W))
        + console.style("gloss", why)
    )


def render(console, data: dict) -> None:
    """Render the landing screen to *console*.

    Normal output:
        Tagline, blank, status line (device · free · models ready · calibrated),
        blank, NEW HERE numbered onboarding path, blank, three command groups
        (YOUR MACHINE / MODELS / RUN), benchmarks one-liner, blank, help footer.

    Verbose appends the full 7-command BENCHMARKS group and removes the
    one-liner in favour of the expanded list; also removes the --verbose hint
    from the footer.
    """
    c = console
    d = data

    cal_str = "calibrated" if d["calibrated"] else "uncalibrated"
    cal_role = "good" if d["calibrated"] else "warn"

    # ── tagline ─────────────────────────────────────────────────────────────
    c.emit(
        "  " + c.style("accent", "wmx-suite")
        + c.style("dim", "  —  run local MLX models on Apple Silicon without crashing your Mac")
    )
    c.emit()

    # ── status line ─────────────────────────────────────────────────────────
    c.emit(
        c.style("dim", "  this machine: ")
        + c.style("metric", d["device"])
        + c.style("dim", " · ")
        + c.style("good", f"{d['free_gb']:.2f} GB free")
        + c.style("dim", " of a ")
        + c.style("good", f"{d['safe_budget_gb']:.2f} GB")
        + c.style("dim", " safe budget · ")
        + c.style("good", f"{d['models_ready']} models ready")
        + c.style("dim", " · ")
        + c.style(cal_role, cal_str)
    )
    c.emit()

    # ── NEW HERE onboarding ──────────────────────────────────────────────────
    c.emit(
        c.section("  NEW HERE?  ")
        + c.style("dim", "first time on this machine, run these in order:")
    )
    c.emit(
        "     "
        + c.style("accent", "1) wmx-suite characterize <m>")
        + c.style("gloss", "  measure a model's safe context ceiling")
    )
    c.emit(
        "     "
        + c.style("accent", "2) wmx-suite run --model <m>")
        + c.style("gloss", "   launch it safely (auto KV-bits + context cap)")
    )
    c.emit()

    # ── YOUR MACHINE ────────────────────────────────────────────────────────
    c.emit(c.section("  YOUR MACHINE"))
    c.emit(_cmd(c, "system", "specs + memory budget (what your Mac can safely give a model)"))
    c.emit(_cmd(c, "health", "live go/no-go: which models can run right now, and how big"))
    c.emit()

    # ── MODELS ──────────────────────────────────────────────────────────────
    c.emit(c.section("  MODELS"))
    c.emit(_cmd(c, "characterize <model>", "safely measure a model's max safe context"))
    c.emit(_cmd(c, "list",                 "show measured ceilings stored for this machine"))
    c.emit(_cmd(c, "calibrate",            "tune this machine's cold-start memory estimate"))
    c.emit()

    # ── RUN ─────────────────────────────────────────────────────────────────
    c.emit(c.section("  RUN"))
    c.emit(_cmd(c, "run --model <model>", "safe launch — refuses anything that would crash the Mac"))
    c.emit()

    # ── BENCHMARKS ──────────────────────────────────────────────────────────
    if c.verbose:
        c.emit(
            c.section("  BENCHMARKS")
            + c.style("dim", "  (power users)")
        )
        c.emit(_cmd(c, "benchmark-kokoro",          "Kokoro TTS throughput (RTF / chars-per-sec) vs length"))
        c.emit(_cmd(c, "benchmark-kokoro-ttfa",     "streaming time-to-first-audio latency"))
        c.emit(_cmd(c, "benchmark-kokoro-batch",    "concurrency vs throughput & peak memory"))
        c.emit(_cmd(c, "benchmark-kokoro-voice",    "voice-switch cost (warm vs cold load)"))
        c.emit(_cmd(c, "benchmark-kokoro-cache",    "voice-cache memory overhead"))
        c.emit(_cmd(c, "benchmark-kokoro-baseline", "idle synthesis memory floor"))
        c.emit(_cmd(c, "benchmark-embeddings",      "ModernBERT memory surface (batch x sequence)"))
    else:
        c.emit(
            c.style("dim", "  benchmarks (TTS / embeddings, for power users): ")
            + c.style("accent", "wmx-suite benchmark-* --help")
        )
    c.emit()

    # ── footer ──────────────────────────────────────────────────────────────
    footer = (
        c.style("dim", "  details on any command:  ")
        + c.style("accent", "wmx-suite <command> --help")
    )
    if not c.verbose:
        footer += (
            c.style("dim", "   ·   everything at once:  ")
            + c.style("accent", "wmx-suite --verbose")
        )
    c.emit(footer)
