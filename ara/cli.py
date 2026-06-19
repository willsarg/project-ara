"""ARA command-line front door.

``ara`` with no arguments renders the landing screen (mirrors wmx-suite's feel).
``ara detect`` renders read-only machine recon. Subcommands beyond detect aren't
built yet — this is an early scaffold.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from ara import acquire, detect, pythons, status
from ara.registry import engine_status, get_backend
from ara.ui import Console

_CMD_W = 16


def _cmd(c: Console, name: str, why: str) -> str:
    """One command row: accent name padded, dim gloss."""
    return "  " + c.style("accent", name.ljust(_CMD_W)) + c.style("gloss", why)


def _fmt_gb(v: float | None, decimals: int = 0) -> str:
    return f"{v:.{decimals}f} GB" if v is not None else "unknown"


def _fmt_size(gb: float | None) -> str:
    """Human download size: MB under a gigabyte, GB above. 'size unknown' if None."""
    if gb is None:
        return "size unknown"
    return f"~{gb * 1000:.0f} MB" if gb < 1 else f"~{gb:.1f} GB"


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
    c.emit(_cmd(c, "status", "show AI/ML processes running right now"))
    c.emit(_cmd(c, "python", "list every Python interpreter + its AI libraries"))
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
        gloss = "your default python3" if m.framework_python else "ARA's python (no user env found)"
        c.emit(c.field("python", m.python_version, gloss))
    n_py = pythons.count()
    if n_py > 1:
        c.emit(c.field("pythons", str(n_py), "interpreters on this machine — run: ara python"))
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

    engines = [rt for rt in m.runtimes if rt.kind == "engine"]
    frameworks = [rt for rt in m.runtimes if rt.kind == "framework"]

    c.emit(c.section("  ENGINES") + c.style("dim", "  (what ARA can launch models through)"))
    for rt in engines:
        if rt.present:
            val = f"{rt.name} {rt.version}" if rt.version else rt.name
            if rt.requires:  # installed, but can't accelerate on this hardware
                c.emit(c.field("·", val, f"installed · {rt.requires}", value_role="warn"))
            else:
                c.emit(c.field("·", val, "found", value_role="good"))
        elif c.verbose:
            c.emit(c.field("·", rt.name, "not found", value_role="dim"))
    if not any(rt.present for rt in engines) and not c.verbose:
        c.emit(c.style("dim", "  none detected"))
    c.emit()

    # Frameworks reflect the USER's own python, not ARA's bundled deps.
    fw_gloss = f"  ({m.framework_python})" if m.framework_python \
        else "  (no separate user python — ARA's env only)"
    c.emit(c.section("  FRAMEWORKS") + c.style("dim", fw_gloss))
    for rt in frameworks:
        if rt.present:
            val = f"{rt.name} {rt.version}" if rt.version else rt.name
            c.emit(c.field("·", val, "found", value_role="good"))
        elif c.verbose:
            c.emit(c.field("·", rt.name, "not found", value_role="dim"))
    if not any(rt.present for rt in frameworks):
        # The default python is bare — an empty section is misleading if the AI stack
        # actually lives in another interpreter, so surface the richest one (only pay the
        # interpreter probe in this already-empty case; full list via `ara python`).
        others = sorted((i for i in pythons.discover() if i.ai_present and not i.is_default),
                        key=lambda i: len(i.ai_present), reverse=True)
        if others:
            top = others[0]
            libs = " · ".join(f"{k} {v}" for k, v in top.ai_present.items())
            c.emit(c.style("dim", "  none in your default — found in another interpreter:"))
            c.emit("  " + c.style("good", f"{top.origin} {top.version or ''}".strip())
                   + "   " + c.style("accent", _tilde(top.path)))
            c.emit("       " + c.style("good", libs))
            more = len(others) - 1
            c.emit(c.style("dim", "  run ") + c.style("accent", "ara python")
                   + c.style("dim", " for the full list" + (f" (+{more} more)" if more else "")))
        elif not c.verbose:
            c.emit(c.style("dim", "  none in this env"))
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
# status (live recon — running AI/ML processes; never crosses the engine seam)
# --------------------------------------------------------------------------- #
def _fmt_uptime(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h"
    return f"{s // 86400}d"


def _fmt_mem(gb: float) -> str:
    """Process RSS, MB under a gigabyte and GB above (binary, matching detect)."""
    return f"{gb * 1024:.0f} MB" if gb < 1 else f"{gb:.1f} GB"


def render_status(c: Console, *, as_json: bool = False) -> None:
    procs = status.scan()

    if as_json:
        print(json.dumps([asdict(p) for p in procs], indent=2))
        return

    c.emit()
    c.emit(c.section("  RUNNING AI/ML"))
    if not procs:
        c.emit(c.style("dim", "  nothing running right now"))
        c.emit()
        return

    for p in procs:
        bits = [f"pid {p.pid}", f"up {_fmt_uptime(p.uptime_s)}"]
        if p.port:
            bits.append(f":{p.port}")
        if p.gpu_mb:
            bits.append(f"{p.gpu_mb:.0f} MB GPU")
        if p.detail:
            bits.append(p.detail)
        c.emit(c.field(p.label, _fmt_mem(p.rss_gb), " · ".join(bits), value_role="metric"))

    total = sum(p.rss_gb for p in procs)
    plural = "process" if len(procs) == 1 else "processes"
    c.emit()
    c.emit(c.field("total", _fmt_mem(total), f"RSS across {len(procs)} {plural}",
                   value_role="good"))
    c.emit()


# --------------------------------------------------------------------------- #
# python (interpreter discovery — read-only; which pythons, which have AI libs)
# --------------------------------------------------------------------------- #
def _tilde(p: str) -> str:
    home = str(Path.home())
    return "~" + p[len(home):] if p.startswith(home) else p


def render_python(c: Console, *, as_json: bool = False) -> None:
    ints = pythons.discover()

    if as_json:
        print(json.dumps([asdict(i) for i in ints], indent=2))
        return

    c.emit()
    c.emit(c.section("  PYTHON INTERPRETERS"))
    sub = " " * 13  # aligns continuation lines under the path column
    with_ai = 0
    last_origin = None
    for i in ints:
        if i.origin != last_origin:          # group header per origin
            c.emit()
            c.emit(c.style("accent", f"  {i.origin}"))
            last_origin = i.origin
        mark = c.style("good", "●") if i.is_default else " "
        c.emit(f"  {mark} " + c.style("metric", f"{i.version or '?':8} ") + _tilde(i.path))
        # When the path you'd type is a symlink, show where it really lives — this is
        # what explains the origin label and untangles symlink chains.
        if _tilde(i.real) != _tilde(i.path):
            c.emit(sub + c.style("dim", f"→ {_tilde(i.real)}"))
        present = i.ai_present
        if present:
            with_ai += 1
            c.emit(sub + c.style("good", " · ".join(f"{k} {v}" for k, v in present.items())))
        else:
            c.emit(sub + c.style("dim", "no AI libraries"))
        if i.caution:
            c.emit(sub + c.style("warn", f"⚠ {i.caution}"))

    c.emit()
    managed = sum(1 for i in ints if i.caution)
    summary = f"  {len(ints)} interpreters · {with_ai} with AI libraries"
    if managed:
        summary += f" · {managed} managed (install into a venv, not the interpreter)"
    c.emit(c.style("dim", summary))
    c.emit(c.style("dim", "  ") + c.style("good", "●") + c.style("dim", " = your default python3"))
    c.emit()
    c.emit(c.style("gloss", "  how this was found: your PATH + standard install homes "
                            "(Homebrew, python.org, pyenv, conda, uv, asdf, macOS)."))
    c.emit(c.style("gloss", "  missing one? it's likely a virtualenv or a custom folder "
                            "not on PATH — add its directory to PATH and re-run."))
    c.emit()


# --------------------------------------------------------------------------- #
# profile (measures — crosses the seam into the engine)
# --------------------------------------------------------------------------- #
def _emit_limits(c: Console, m: dict) -> None:
    c.emit()
    c.emit(c.section("  SAFE LIMITS")
           + c.style("dim", "" if m["calibrated"] else "  (estimated — not calibrated)"))
    c.emit(c.field("device", f"{m['device']} · {m['total_gb']:.0f} GB"))
    c.emit(c.field("crash wall", _fmt_gb(m["wall_gb"], 1),
                   "the hard ceiling — never cross", value_role="bad"))
    c.emit(c.field("safe budget", _fmt_gb(m["safe_budget_gb"], 1),
                   f"wall − {m['margin_gb']:.0f} GB margin", value_role="good"))
    c.emit(c.field("headroom", _fmt_gb(m["headroom_gb"], 1), "free under budget right now"))
    if m["overhead_gb"] is not None:
        gloss = "default estimate" if not m["calibrated"] else \
            f"measured cold-start · calibrated {m['calibrated_at']}"
        c.emit(c.field("overhead", _fmt_gb(m["overhead_gb"], 1), gloss))
    if m["swap_free_gb"] is not None:
        c.emit(c.field("swap", f"{m['swap_free_gb']:.1f} GB free"))
    c.emit()


def _confirm(question: str) -> bool:
    try:
        return input(f"  {question} [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _emit_calibration(c: Console, m: dict, fallback_model: str) -> None:
    """One honest line on what calibration measured vs the built-in default."""
    cal = m.get("calibration") or {}
    measured = cal.get("measured_overhead_gb")
    default = cal.get("default_overhead_gb")
    n = cal.get("n_points")
    short = cal.get("hf_id", fallback_model).split("/")[-1]
    if measured is None or default is None:
        return
    if measured < default:
        verdict = (f"hardware is lean — measured overhead under the {default:.0f} GB "
                   f"default, keeping the default")
    elif measured > default:
        verdict = (f"measured {measured:.1f} GB → {measured - default:.1f} GB more "
                   f"conservative than the {default:.0f} GB default")
    else:
        verdict = f"measured {measured:.1f} GB, matching the default"
    rungs = f" · {n} rungs" if n else ""
    c.emit(c.style("dim", f"  overhead: {verdict}{rungs} · {short}"))


def render_profile(c: Console, *, recalibrate: bool = False, as_json: bool = False,
                   assume_yes: bool = False, model: str | None = None) -> int:
    backend = detect.backend_name()
    if backend == "unsupported":
        c.emit(c.style("warn", "  profiling needs an ARA backend — none for this hardware yet."))
        return 1
    engine_ok, engine = engine_status()
    if not engine_ok:
        c.emit(c.style("warn", f"  the {engine} engine isn't installed here — run: ")
               + c.style("accent", "uv sync"))
        return 1

    bk = get_backend()
    try:
        m = bk.safe_limits()
    except Exception as exc:
        c.emit(c.style("bad", f"  couldn't read limits: {exc}"))
        return 1

    if as_json:
        print(json.dumps(m, indent=2))
        return 0

    _emit_limits(c, m)

    # Naming an explicit --model means "calibrate against this", so it bypasses the
    # cached early-return the way --recalibrate does.
    explicit_model = model is not None
    model = model or bk.CALIBRATION_MODEL

    # Calibration is opt-in: offered only when it'd help, and only interactively.
    if m["calibrated"] and not recalibrate and not explicit_model:
        c.emit(c.style("dim", "  cached — ") + c.style("accent", "ara profile --recalibrate")
               + c.style("dim", " to re-measure"))
        c.emit()
        return 0

    interactive = assume_yes or sys.stdin.isatty()
    if not interactive:
        c.emit(c.style("dim", "  estimated — run ") + c.style("accent", "ara profile")
               + c.style("dim", " in a terminal (or pass ") + c.style("accent", "--yes")
               + c.style("dim", ") to calibrate against a real model"))
        c.emit()
        return 0

    cached = bk.calibration_model_cached(model)
    if cached:
        q = f"Calibrate now against {model}?  (loads it, stays under the safe budget)"
    else:
        size_gb = acquire.repo_size_gb(model)
        free_gb = acquire.free_disk_gb()
        if size_gb and free_gb is not None and free_gb < size_gb + acquire.DISK_BUFFER_GB:
            c.emit(c.style("bad",
                           f"  not enough disk for {model}: needs ~{size_gb:.1f} GB + "
                           f"{acquire.DISK_BUFFER_GB:.0f} GB headroom, only {free_gb:.1f} GB free."))
            c.emit()
            return 1
        q = f"Download {model} from Hugging Face and calibrate?  ({_fmt_size(size_gb)})"

    if not (assume_yes or _confirm(q)):
        c.emit(c.style("dim", "  skipped — showing estimated limits."))
        c.emit()
        return 0

    try:
        if not cached:
            c.emit(c.style("dim", f"  downloading {model} …"))
            bk.download_calibration_model(model)
        m = bk.calibrate(model)
    except SystemExit:
        return 1  # engine printed a clean reason
    except Exception as exc:
        c.emit(c.style("bad", f"  calibration failed: {exc}"))
        return 1

    c.emit(c.style("good", "  calibrated."))
    _emit_calibration(c, m, model)
    _emit_limits(c, m)
    return 0


# --------------------------------------------------------------------------- #
# entry
# --------------------------------------------------------------------------- #
def main() -> int:
    argv = sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv
    as_json = "--json" in argv
    recalibrate = "--recalibrate" in argv
    assume_yes = "--yes" in argv or "-y" in argv

    # --model <id> takes a value; pull it (and its value) out before the rest.
    model: str | None = None
    rest: list[str] = []
    skip = False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a == "--model":
            model = argv[i + 1] if i + 1 < len(argv) else None
            skip = True
            continue
        if a.startswith("--model="):
            model = a.split("=", 1)[1] or None
            continue
        if a in ("--verbose", "-v", "--json", "--recalibrate", "--yes", "-y"):
            continue
        rest.append(a)
    c = Console.from_env(verbose=verbose)

    if not rest or rest[0] in ("-h", "--help"):
        render_landing(c)
        return 0

    if rest[0] == "detect":
        render_detect(c, as_json=as_json)
        return 0

    if rest[0] == "status":
        render_status(c, as_json=as_json)
        return 0

    if rest[0] == "python":
        render_python(c, as_json=as_json)
        return 0

    if rest[0] == "profile":
        return render_profile(c, recalibrate=recalibrate, as_json=as_json,
                              assume_yes=assume_yes, model=model)

    c.emit(c.style("warn", f"  '{rest[0]}' isn't built yet — ARA is an early scaffold."))
    c.emit(
        c.style("dim", "  run ") + c.style("accent", "ara")
        + c.style("dim", " with no arguments to see the planned commands.")
    )
    return 1
