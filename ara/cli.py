"""ARA command-line front door.

``ara`` with no arguments renders the landing screen (mirrors wmx-suite's feel):
a tagline, a live 'this machine' status line, and the planned command path.
Subcommands aren't built yet — this is an early scaffold.
"""
from __future__ import annotations

import sys

from ara import detect
from ara.registry import engine_status
from ara.ui import Console

_CMD_W = 16


def _cmd(c: Console, name: str, why: str) -> str:
    """One command row: accent name padded, dim gloss."""
    return "  " + c.style("accent", name.ljust(_CMD_W)) + c.style("gloss", why)


def render_landing(c: Console) -> None:
    chip = detect.chip_name()
    backend = detect.backend_name()
    engine_ok, engine = engine_status()

    supported = backend != "unsupported"
    backend_role = "good" if supported else "warn"
    engine_role = "good" if engine_ok else "warn"
    engine_str = f"{engine} ready" if engine_ok else f"{engine} not installed"

    # ── tagline ──────────────────────────────────────────────────────────────
    c.emit(
        "  " + c.style("accent", "ara")
        + c.style("dim", "  —  AI Runs Anywhere: run local models on whatever hardware you've got")
    )
    c.emit()

    # ── status line ──────────────────────────────────────────────────────────
    c.emit(
        c.style("dim", "  this machine: ")
        + c.style("metric", chip)
        + c.style("dim", " · backend ")
        + c.style(backend_role, backend)
        + c.style("dim", " · engine ")
        + c.style(engine_role, engine_str)
    )
    c.emit()

    # ── planned command path ─────────────────────────────────────────────────
    c.emit(c.section("  GETTING STARTED") + c.style("dim", "  (the planned v1 path)"))
    c.emit(_cmd(c, "detect", "inspect this machine and choose a safe backend"))
    c.emit(_cmd(c, "recommend", "best model per modality that fits this machine"))
    c.emit(_cmd(c, "run <model>", "launch it safely — right up to the edge, never over"))
    c.emit()

    # ── footer ───────────────────────────────────────────────────────────────
    if not supported:
        c.emit(c.style("warn", "  no supported backend for this machine yet — Apple Silicon only for now"))
    c.emit(c.style("dim", "  early scaffold — only this screen is wired up so far"))


def main() -> int:
    argv = sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv
    rest = [a for a in argv if a not in ("--verbose", "-v")]
    c = Console.from_env(verbose=verbose)

    if not rest or rest[0] in ("-h", "--help"):
        render_landing(c)
        return 0

    c.emit(c.style("warn", f"  '{rest[0]}' isn't built yet — ARA is an early scaffold."))
    c.emit(
        c.style("dim", "  run ") + c.style("accent", "ara")
        + c.style("dim", " with no arguments to see the planned commands.")
    )
    return 1
