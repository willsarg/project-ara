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
from ara.registry import engine_status, get_backend
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
    c.emit(_cmd(c, "detect", "inspect this machine — read-only recon"))
    c.emit(_cmd(c, "profile", "measure this machine's safe memory limits"))
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

    a = m.accel

    c.emit()
    c.emit(c.section("  SYSTEM"))
    c.emit(c.field("chip", m.chip))
    c.emit(c.field("os", m.os_version))
    c.emit(c.field("arch", m.arch))
    if m.cpu_physical:
        cores = f"{m.cpu_physical} cores"
        if c.verbose and m.cpu_logical:
            cores = f"{m.cpu_physical} physical · {m.cpu_logical} logical"
        if m.cpu_features:
            cores += "   " + " · ".join(m.cpu_features)
        c.emit(c.field("cpu", cores))
    if m.python_version:
        c.emit(c.field("python", m.python_version, "ambient python3"))
    c.emit()

    c.emit(c.section("  MEMORY"))
    c.emit(c.field("total", _fmt_gb(m.ram_total_gb)))
    if m.ram_available_gb is not None:
        c.emit(c.field("available", _fmt_gb(m.ram_available_gb, 1), "free right now"))
    if m.swap_gb:
        c.emit(c.field("swap", _fmt_gb(m.swap_gb, 1)))
    c.emit()

    c.emit(c.section("  ACCELERATOR"))
    if a.kind == "nvidia":
        bits = []
        if a.vram_gb:
            bits.append(f"{a.vram_gb:.0f} GB VRAM")
        if a.compute:
            bits.append(f"SM {a.compute}")
        if a.cuda_version:
            bits.append(f"CUDA {a.cuda_version}")
        gloss = " · ".join(bits)
        name = f"{a.name}  (x{a.count})" if a.count > 1 else a.name
        c.emit(c.field("gpu", name, gloss))
    elif a.kind == "apple":
        cores = f"{a.cores}-core " if a.cores else ""
        c.emit(c.field("gpu", a.name, f"{cores}Metal · unified memory (shared with system)"))
    else:
        c.emit(c.field("gpu", a.name, "no GPU detected", value_role="warn"))
    c.emit()

    c.emit(c.section("  STORAGE"))
    c.emit(c.field("disk free", _fmt_gb(m.disk_free_gb), "on the home volume"))
    c.emit()

    c.emit(c.section("  RUNTIMES"))
    for rt in m.runtimes:
        if rt.present:
            val = f"{rt.name} {rt.version}" if rt.version else rt.name
            c.emit(c.field("·", val, "found", value_role="good"))
        elif c.verbose:
            c.emit(c.field("·", rt.name, "not found", value_role="dim"))
    if not any(rt.present for rt in m.runtimes):
        c.emit(c.style("dim", "  none detected"))
    c.emit()

    c.emit(c.section("  MODELS"))
    for store in m.model_stores:
        if store.present and store.count:
            c.emit(c.field(store.name, f"{store.count} models",
                           f"{store.size_gb:.0f} GB", value_role="good"))
        elif store.present:
            c.emit(c.field(store.name, "empty", value_role="dim"))
        elif c.verbose:
            c.emit(c.field(store.name, "not found", value_role="dim"))
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
    c.emit(c.field("hf token", "present" if m.hf_token else "none",
                   None if m.hf_token else "needed for gated models",
                   value_role="good" if m.hf_token else "dim"))
    c.emit(c.field("power", m.power))
    c.emit()

    if not m.supported:
        c.emit(c.style("warn", "  no ARA backend for this hardware yet — recon works, running comes later"))
        c.emit()


# --------------------------------------------------------------------------- #
# profile (measures — crosses the seam into the engine)
# --------------------------------------------------------------------------- #
def render_profile(c: Console, *, recalibrate: bool = False, as_json: bool = False) -> int:
    backend = detect.backend_name()
    if backend == "unsupported":
        c.emit(c.style("warn", "  profiling needs an ARA backend — none for this hardware yet."))
        return 1
    engine_ok, engine = engine_status()
    if not engine_ok:
        c.emit(c.style("warn", f"  the {engine} engine isn't installed here — run: ")
               + c.style("accent", "uv sync"))
        return 1

    try:
        m = get_backend().machine_profile(recalibrate=recalibrate)
    except SystemExit:
        return 1  # engine already printed a clean reason (e.g. no cached model)
    except Exception as exc:
        c.emit(c.style("bad", f"  profiling failed: {exc}"))
        return 1

    if as_json:
        print(json.dumps(m, indent=2))
        return 0

    c.emit()
    c.emit(c.section("  SAFE LIMITS"))
    c.emit(c.field("device", f"{m['device']} · {m['total_gb']:.0f} GB"))
    c.emit(c.field("crash wall", _fmt_gb(m["wall_gb"], 1),
                   "the hard ceiling — never cross", value_role="bad"))
    c.emit(c.field("safe budget", _fmt_gb(m["safe_budget_gb"], 1),
                   f"wall − {m['margin_gb']:.0f} GB margin", value_role="good"))
    c.emit(c.field("headroom", _fmt_gb(m["headroom_gb"], 1), "free under budget right now"))
    if m["overhead_gb"] is not None:
        c.emit(c.field("overhead", _fmt_gb(m["overhead_gb"], 1),
                       f"measured cold-start · calibrated {m['calibrated_at']}"))
    if m["swap_free_gb"] is not None:
        c.emit(c.field("swap", f"{m['swap_free_gb']:.1f} GB free"))
    c.emit()

    if not m["calibrated"]:
        c.emit(c.style("warn", "  estimated only — not calibrated. run ")
               + c.style("accent", "ara profile --recalibrate"))
    elif not m["just_measured"]:
        c.emit(c.style("dim", "  cached from a prior run — ")
               + c.style("accent", "ara profile --recalibrate")
               + c.style("dim", " to re-measure"))
    c.emit()
    return 0


# --------------------------------------------------------------------------- #
# entry
# --------------------------------------------------------------------------- #
def main() -> int:
    argv = sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv
    as_json = "--json" in argv
    recalibrate = "--recalibrate" in argv
    rest = [a for a in argv if a not in ("--verbose", "-v", "--json", "--recalibrate")]
    c = Console.from_env(verbose=verbose)

    if not rest or rest[0] in ("-h", "--help"):
        render_landing(c)
        return 0

    if rest[0] == "detect":
        render_detect(c, as_json=as_json)
        return 0

    if rest[0] == "profile":
        return render_profile(c, recalibrate=recalibrate, as_json=as_json)

    c.emit(c.style("warn", f"  '{rest[0]}' isn't built yet — ARA is an early scaffold."))
    c.emit(
        c.style("dim", "  run ") + c.style("accent", "ara")
        + c.style("dim", " with no arguments to see the planned commands.")
    )
    return 1
