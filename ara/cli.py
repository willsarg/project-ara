"""ARA command-line front door.

``ara`` with no arguments renders the landing screen (mirrors wmx-suite's feel).
``ara detect`` renders read-only machine recon. Subcommands beyond detect aren't
built yet — this is an early scaffold.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict

from ara import detect
from ara.registry import engine_status
from ara.ui import Console

_CMD_W = 16


def _cmd(c: Console, name: str, why: str) -> str:
    """One command row: accent name padded, dim gloss."""
    return "  " + c.style("accent", name.ljust(_CMD_W)) + c.style("gloss", why)


def _fmt_gb(v: float | None, decimals: int = 0) -> str:
    return f"{v:.{decimals}f} GB" if v is not None else "unknown"


# --------------------------------------------------------------------------- #
# landing
# --------------------------------------------------------------------------- #
def render_landing(c: Console) -> None:
    chip = detect.chip_name()
    backend = detect.backend_name()
    engine_ok, engine = engine_status()

    supported = backend != "unsupported"
    backend_role = "good" if supported else "warn"
    engine_role = "good" if engine_ok else "warn"
    engine_str = f"{engine} ready" if engine_ok else f"{engine} not installed"

    c.emit(
        "  " + c.style("accent", "ara")
        + c.style("dim", "  —  AI Runs Anywhere: run local models on whatever hardware you've got")
    )
    c.emit()
    c.emit(
        c.style("dim", "  this machine: ")
        + c.style("metric", chip)
        + c.style("dim", " · backend ")
        + c.style(backend_role, backend)
        + c.style("dim", " · engine ")
        + c.style(engine_role, engine_str)
    )
    c.emit()
    c.emit(c.section("  GETTING STARTED") + c.style("dim", "  (the planned v1 path)"))
    c.emit(_cmd(c, "detect", "inspect this machine and choose a safe backend"))
    c.emit(_cmd(c, "recommend", "best model per modality that fits this machine"))
    c.emit(_cmd(c, "run <model>", "launch it safely — right up to the edge, never over"))
    c.emit()
    if not supported:
        c.emit(c.style("warn", "  no supported backend for this machine yet — Apple Silicon only for now"))
    c.emit(
        c.style("dim", "  try ") + c.style("accent", "ara detect")
        + c.style("dim", "  ·  recommend / run are next")
    )


# --------------------------------------------------------------------------- #
# detect (recon only — never profiles or loads an engine)
# --------------------------------------------------------------------------- #
def render_detect(c: Console, *, as_json: bool = False) -> None:
    m = detect.profile()

    if as_json:
        print(json.dumps(asdict(m), indent=2))
        return

    c.emit()
    c.emit(c.section("  SYSTEM"))
    c.emit(c.field("chip", m.chip))
    c.emit(c.field("os", m.os_version))
    c.emit(c.field("arch", m.arch))
    if m.cpu_physical:
        cores = f"{m.cpu_physical} cores"
        if c.verbose and m.cpu_logical:
            cores = f"{m.cpu_physical} physical · {m.cpu_logical} logical"
        c.emit(c.field("cpu", cores))
    c.emit()

    c.emit(c.section("  MEMORY"))
    c.emit(c.field("total", _fmt_gb(m.ram_total_gb)))
    if m.ram_available_gb is not None:
        c.emit(c.field("available", _fmt_gb(m.ram_available_gb, 1), "free right now"))
    c.emit()

    c.emit(c.section("  ACCELERATOR"))
    a = m.accel
    if a.kind == "nvidia":
        gloss = f"{a.vram_gb:.0f} GB VRAM · {a.api}" if a.vram_gb else (a.api or "")
    elif a.kind == "apple":
        gloss = "Metal · unified memory (shared with system)"
    else:
        gloss = "no GPU detected"
    c.emit(c.field("gpu", a.name, gloss, value_role="metric" if a.kind != "none" else "warn"))
    c.emit()

    c.emit(c.section("  STORAGE"))
    c.emit(c.field("disk free", _fmt_gb(m.disk_free_gb), "on the home volume"))
    c.emit()

    c.emit(c.section("  ARA"))
    c.emit(c.field(
        "backend", m.backend,
        "auto-picked for this hardware" if m.supported else "no adapter for this hardware yet",
        value_role="good" if m.supported else "warn",
    ))
    c.emit(c.field(
        "engine", f"{m.engine} {'ready' if m.engine_ready else 'not installed'}",
        None if m.engine_ready else ("install: uv sync" if m.supported else None),
        value_role="good" if m.engine_ready else "warn",
    ))
    c.emit()

    c.emit(c.section("  ALREADY HERE"))
    c.emit(c.field("hf cache", "found" if m.hf_cache else "not found",
                   value_role="good" if m.hf_cache else "dim"))
    c.emit(c.field("ollama", "found" if m.ollama else "not found",
                   value_role="good" if m.ollama else "dim"))
    c.emit()

    if not m.supported:
        c.emit(c.style("warn", "  no ARA backend for this hardware yet — recon works, running comes later"))
        c.emit()


# --------------------------------------------------------------------------- #
# entry
# --------------------------------------------------------------------------- #
def main() -> int:
    argv = sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv
    as_json = "--json" in argv
    rest = [a for a in argv if a not in ("--verbose", "-v", "--json")]
    c = Console.from_env(verbose=verbose)

    if not rest or rest[0] in ("-h", "--help"):
        render_landing(c)
        return 0

    if rest[0] == "detect":
        render_detect(c, as_json=as_json)
        return 0

    c.emit(c.style("warn", f"  '{rest[0]}' isn't built yet — ARA is an early scaffold."))
    c.emit(
        c.style("dim", "  run ") + c.style("accent", "ara")
        + c.style("dim", " with no arguments to see the planned commands.")
    )
    return 1
