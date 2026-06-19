"""ARA command-line front door.

``ara`` with no arguments renders the landing screen (mirrors wmx-suite's feel).
``ara detect`` renders read-only machine recon. Subcommands beyond detect aren't
built yet — this is an early scaffold.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from ara import (acquire, apps, catalog, db, detect, engines, mlx, profiles,
                 pythons, status, versions)
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
# section filtering — shared across the recon commands (--include / --exclude)
# --------------------------------------------------------------------------- #
# The sections each recon command can show, in display order. Single-section
# commands list one key so the flags behave consistently everywhere.
_RECON_SECTIONS: dict[str, tuple[str, ...]] = {
    "detect": ("system", "memory", "accelerator", "storage",
               "engines", "frameworks", "models", "apps", "ara"),
    "apps": ("runner", "image", "speech", "toolkit", "assistant", "coding"),
    "mlx": ("readiness", "libraries"),
    "status": ("processes",),
    "python": ("interpreters",),
}
_SECTION_ALIASES = {
    "gpu": "accelerator", "app": "apps", "framework": "frameworks", "engine": "engines",
    "model": "models", "lib": "libraries", "libs": "libraries", "library": "libraries",
    "ready": "readiness", "proc": "processes", "procs": "processes", "process": "processes",
    "interpreter": "interpreters", "interps": "interpreters", "interp": "interpreters",
    "runners": "runner", "toolkits": "toolkit", "assistants": "assistant",
    "models-runner": "runner",
}


def _csv(value: str) -> list[str]:
    """Split a comma-separated flag value into trimmed, non-empty parts."""
    return [s.strip() for s in value.split(",") if s.strip()]


def _section_filter(include, exclude):
    """A predicate over section keys: a whitelist if *include* is given, else a blacklist."""
    inc, exc = set(include or []), set(exclude or [])
    return (lambda k: k in inc) if inc else (lambda k: k not in exc)


def _resolve_want(cmd: str, include: list[str], exclude: list[str], c: Console):
    """Build the section predicate for *cmd*, normalizing aliases and warning on unknowns.
    Returns None when the command has no sections to filter."""
    valid = _RECON_SECTIONS.get(cmd)
    if valid is None:
        if include or exclude:
            c.emit(c.style("warn", f"  --include/--exclude don't apply to '{cmd}'"))
            c.emit()
        return None

    def norm(xs):
        return [_SECTION_ALIASES.get(x.lower().strip(), x.lower().strip()) for x in xs]

    inc, exc = norm(include), norm(exclude)
    unknown = [s for s in (*inc, *exc) if s not in valid]
    if unknown:
        c.emit(c.style("warn", f"  unknown section(s) for {cmd}: {', '.join(dict.fromkeys(unknown))}"))
        c.emit(c.style("dim", f"  valid: {', '.join(valid)}"))
        c.emit()
    return _section_filter([s for s in inc if s in valid], [s for s in exc if s in valid])


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
    c.emit(_cmd(c, "apps", "list installed AI/ML apps + versions"))
    if supported:  # MLX ecosystem view is Apple-Silicon only
        c.emit(_cmd(c, "mlx", "inspect the MLX ecosystem — libraries + readiness"))
    c.emit(_cmd(c, "install", "install the engine matched to this machine"))
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
def _det_system(c: Console, m) -> None:
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


def _det_memory(c: Console, m) -> None:
    c.emit(c.section("  MEMORY"))
    c.emit(c.field("total", _fmt_gb(m.ram_total_gb)))
    if m.ram_available_gb is not None:
        c.emit(c.field("available", _fmt_gb(m.ram_available_gb, 1), "free right now"))
    if m.swap_gb:
        c.emit(c.field("swap", _fmt_gb(m.swap_gb, 1)))
    c.emit()


def _det_accelerator(c: Console, m) -> None:
    a = m.accel
    c.emit(c.section("  ACCELERATOR"))
    if a.kind == "nvidia":
        bits = []
        if a.vram_gb:
            bits.append(f"{a.vram_gb:.0f} GB VRAM")
        if a.compute:
            bits.append(f"SM {a.compute}")
        if a.cuda_version:
            bits.append(f"CUDA {a.cuda_version}")
        if a.driver_version:
            bits.append(f"driver {a.driver_version}")
        gloss = " · ".join(bits)
        name = f"{a.name}  (x{a.count})" if a.count > 1 else a.name
        c.emit(c.field("gpu", name, gloss))
    elif a.kind == "apple":
        cores = f"{a.cores}-core " if a.cores else ""
        c.emit(c.field("gpu", a.name, f"{cores}Metal · unified memory (shared with system)"))
    else:
        c.emit(c.field("gpu", a.name, "no GPU detected", value_role="warn"))
    c.emit()


def _det_storage(c: Console, m) -> None:
    c.emit(c.section("  STORAGE"))
    c.emit(c.field("disk free", _fmt_gb(m.disk_free_gb), "on the home volume"))
    c.emit()


def _det_engines(c: Console, m) -> None:
    engines = [rt for rt in m.runtimes if rt.kind == "engine"]
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


def _det_frameworks(c: Console, m) -> None:
    # Frameworks reflect the USER's own python, not ARA's bundled deps.
    frameworks = [rt for rt in m.runtimes if rt.kind == "framework"]
    c.emit(c.section("  FRAMEWORKS"))
    present_fw = [rt for rt in frameworks if rt.present]
    default_py = m.framework_python or "ARA's env (no separate user python)"

    if present_fw:
        libs = " · ".join(f"{rt.name} {rt.version}".strip() for rt in present_fw)
        c.emit(c.style("dim", "  Your default python has AI frameworks:"))
        c.emit("      " + c.style("accent", default_py))
        c.emit("      " + c.style("good", libs))
    else:
        c.emit(c.style("dim", "  Your default python has no AI frameworks:"))
        c.emit("      " + c.style("accent", default_py))
        # An empty section is misleading when the stack actually lives in another
        # interpreter — surface the richest one (probe paid only in this empty case).
        others = sorted((i for i in pythons.discover() if i.ai_present and not i.is_default),
                        key=lambda i: len(i.ai_present), reverse=True)
        if others:
            top = others[0]
            libs = " · ".join(f"{k} {v}" for k, v in top.ai_present.items())
            c.emit()
            c.emit(c.style("dim", "  But you've got them in ")
                   + c.style("good", f"{top.origin} {top.version or ''}".strip())
                   + c.style("dim", ":"))
            c.emit("      " + c.style("accent", _tilde(top.path)))
            c.emit("      " + c.style("good", libs))
            c.emit()
            more = len(others) - 1
            tail = f" ({more} more with AI libraries)" if more else ""
            c.emit(c.style("dim", "  Run ") + c.style("accent", "ara python")
                   + c.style("dim", f" to see every interpreter{tail}."))
        else:
            c.emit(c.style("dim", "  None found in any interpreter on this machine."))
    c.emit()


def _det_models(c: Console, m) -> None:
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


def _det_apps(c: Console, m) -> None:
    # Detect shows a per-category summary; `ara apps` has the full list with versions.
    c.emit(c.section("  AI/ML APPS"))
    if not m.apps:
        c.emit(c.style("dim", "  none detected"))
        c.emit()
        return
    by_cat: dict[str, list] = {}
    for app in m.apps:
        by_cat.setdefault(app.category, []).append(app)
    for cat in apps._ORDER:
        items = by_cat.get(cat)
        if not items:
            continue
        recent = sorted(items, key=lambda a: a.installed_at or 0.0, reverse=True)
        names = ", ".join(a.label for a in recent[:3])
        extra = len(items) - len(recent[:3])
        if extra:
            names += f"  (+{extra} more)"
        c.emit("  " + c.style("metric", f"{apps.CATEGORY_LABEL[cat]:17}")
               + c.style("good", f"{len(items):>2}") + c.style("dim", f"   {names}"))
    c.emit(c.style("dim", "  newest first per category — run ")
           + c.style("accent", "ara apps") + c.style("dim", " for the full list with versions"))
    c.emit()


def _det_ara(c: Console, m) -> None:
    c.emit(c.section("  ARA"))
    c.emit(c.field(
        "backend", m.backend,
        "auto-picked for this hardware" if m.supported else "no adapter for this hardware yet",
        value_role="good" if m.supported else "warn",
    ))
    c.emit(c.field(
        "engine", f"{m.engine} {'ready' if m.engine_ready else 'not installed'}",
        None if m.engine_ready else ("install: ara install" if m.supported else None),
        value_role="good" if m.engine_ready else "warn",
    ))
    c.emit(c.field("hf cli",
                   ("not found" if not m.hf_cli
                    else f"present {m.hf_cli_version}" if m.hf_cli_version else "present"),
                   "the hf command" if m.hf_cli else "pip install huggingface_hub",
                   value_role="good" if m.hf_cli else "dim"))
    c.emit(c.field("hf token", "present" if m.hf_token else "none",
                   None if m.hf_token else "needed for gated models",
                   value_role="good" if m.hf_token else "dim"))
    c.emit(c.field("power", m.power))
    c.emit()
    if not m.supported:
        c.emit(c.style("warn", "  no ARA backend for this hardware yet — recon works, running comes later"))
        c.emit()


_DETECT_RENDERERS: tuple[tuple[str, object], ...] = (
    ("system", _det_system),
    ("memory", _det_memory),
    ("accelerator", _det_accelerator),
    ("storage", _det_storage),
    ("engines", _det_engines),
    ("frameworks", _det_frameworks),
    ("models", _det_models),
    ("apps", _det_apps),
    ("ara", _det_ara),
)


def render_detect(c: Console, *, as_json: bool = False, want=None) -> None:
    m = detect.profile()
    if as_json:
        print(json.dumps(asdict(m), indent=2))
        return
    want = want or (lambda _key: True)
    c.emit()
    for key, fn in _DETECT_RENDERERS:
        if want(key):
            fn(c, m)


# --------------------------------------------------------------------------- #
# apps (full AI/ML software inventory — the detailed list detect summarizes)
# --------------------------------------------------------------------------- #
def render_apps(c: Console, *, as_json: bool = False, want=None) -> None:
    inventory = apps.scan()
    # auto_updates lookup (one batched brew call) lives here, in the dedicated command —
    # never in the detect summary. True = brew defers, so drift is expected, not a conflict.
    defers = versions.cask_auto_updates()
    if as_json:
        print(json.dumps([{
            "label": a.label, "category": a.category, "version": a.version,
            "source": a.source, "duplicate": a.duplicate, "drift": a.drift,
            "brew_recorded": a.brew_recorded, "cask_token": a.cask_token,
            "auto_updates": defers.get(a.cask_token),
            "in_app": a.in_app, "cask": a.cask, "formula": a.formula,
            "installed_at": a.installed_at,
        } for a in inventory], indent=2))
        return
    want = want or (lambda _key: True)
    c.emit()
    c.emit(c.section("  AI/ML APPS"))
    shown = [a for a in inventory if want(a.category)]
    if not shown:
        c.emit(c.style("dim", "  none detected"))
        c.emit()
        return
    last_cat = None
    for app in shown:
        if app.category != last_cat:
            c.emit()
            c.emit(c.style("accent", f"  {apps.CATEGORY_LABEL[app.category]}"))
            last_cat = app.category
        name = f"{app.label} {app.version}".strip() if app.version else app.label
        auto = defers.get(app.cask_token)            # True / False / None(unknown)
        # The real problem = the app self-updated outside brew (drift) AND brew has no
        # auto_updates to account for it. Omitting auto_updates alone is fine — an app
        # that updates THROUGH brew (e.g. Claude Code) correctly omits it and won't drift.
        clueless = bool(app.drift and not auto)
        problem = app.duplicate or clueless
        gloss = app.source
        if clueless:
            gloss += (f"  ⚠ self-updated past brew (records {app.brew_recorded}); "
                      f"no auto_updates, so brew upgrade will clobber it")
        elif app.drift:  # auto_updates declared → expected, brew won't fight it
            gloss += f"  · self-updates; brew defers (records {app.brew_recorded})"
        if app.duplicate:
            gloss += "  ⚠ likely duplicate"
        c.emit(c.field("·", name, gloss, value_role="warn" if problem else "good"))
    c.emit()
    c.emit(c.style("gloss", "  curated catalog; a Homebrew cask installs the .app, "
                            "so cask + app is one install, not a duplicate."))
    c.emit(c.style("gloss", "  ⚠ = the app self-updated past Homebrew's record and the cask "
                            "has no auto_updates, so brew upgrade can clobber it."))
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


def render_status(c: Console, *, as_json: bool = False, want=None) -> None:
    procs = status.scan()

    if as_json:
        print(json.dumps([asdict(p) for p in procs], indent=2))
        return

    if not (want or (lambda _key: True))("processes"):
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


def render_python(c: Console, *, as_json: bool = False, want=None) -> None:
    ints = pythons.discover()

    if as_json:
        print(json.dumps([asdict(i) for i in ints], indent=2))
        return

    if not (want or (lambda _key: True))("interpreters"):
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
    homes = ("python.org (Programs\\Python), conda, uv, pyenv-win, the Store"
             if os.name == "nt" else "Homebrew, python.org, pyenv, conda, uv, asdf, macOS")
    c.emit(c.style("gloss", "  how this was found: your PATH + standard install homes "
                            f"({homes})."))
    c.emit(c.style("gloss", "  missing one? it's likely a virtualenv or a custom folder "
                            "not on PATH — add its directory to PATH and re-run."))
    c.emit()


# --------------------------------------------------------------------------- #
# mlx (the MLX ecosystem — libraries by modality + Apple readiness; Apple-only)
# --------------------------------------------------------------------------- #
def render_mlx(c: Console, *, as_json: bool = False, want=None) -> None:
    is_apple = detect.backend_name() == "apple"
    interps = mlx.scan() if is_apple else []
    runtimes = mlx.lmstudio_mlx_runtimes() if is_apple else []
    n_models = mlx.mlx_community_model_count() if is_apple else 0
    accel = detect.accelerator(detect.chip_name()) if is_apple else None

    if as_json:
        print(json.dumps({
            "apple_silicon": is_apple,
            "gpu": {"name": accel.name, "cores": accel.cores} if accel else None,
            "mlx_community_models": n_models,
            "lmstudio_mlx_runtimes": runtimes,
            "interpreters": [
                {"path": m.path, "origin": m.origin, "version": m.version, "packages": m.packages}
                for m in interps
            ],
        }, indent=2))
        return

    c.emit()
    c.emit(c.section("  MLX"))
    if not is_apple:
        c.emit(c.style("warn", "  MLX is Apple-Silicon only — not applicable on this machine."))
        c.emit()
        return
    want = want or (lambda _key: True)

    if want("readiness"):
        c.emit()
        c.emit(c.style("dim", "  READINESS"))
        cores = f"{accel.cores}-core " if accel and accel.cores else ""
        c.emit(c.field("GPU", accel.name, f"{cores}Metal · unified memory"))
        c.emit(c.field("models", f"{n_models} cached", "mlx-community models in your HF cache"))
        if runtimes:
            extra = f"  (+{len(runtimes) - 1} older)" if len(runtimes) > 1 else ""
            c.emit(c.field("LM Studio", f"MLX runtime {runtimes[0]}{extra}", "Apple MLX engine"))
        else:
            c.emit(c.field("LM Studio", "no MLX runtime", value_role="dim"))

    if want("libraries"):
        c.emit()
        c.emit(c.style("dim", "  LIBRARIES"))
        if not interps:
            c.emit("  " + c.style("dim", "No MLX packages installed in any interpreter."))
            c.emit("  " + c.style("dim", "Install into a venv, e.g. ")
                   + c.style("accent", "pip install mlx-lm"))
        else:
            present: set[str] = set()
            for mi in interps:
                c.emit()
                c.emit("  " + c.style("good", f"{mi.origin} {mi.version or ''}".strip())
                       + c.style("dim", "  ·  ") + c.style("accent", _tilde(mi.path)))
                mgr = pythons.manager_of(mi.origin, mi.externally_managed)
                if mgr:  # MLX was pip-installed into an interpreter managed by someone else
                    c.emit("      " + c.style("warn", f"⚠ this interpreter is managed by {mgr} — "
                                                      f"MLX shouldn't be installed into it; use a venv"))
                for label, pkgs in mlx.GROUPS:
                    got = [(p, mi.packages[p]) for p in pkgs if p in mi.packages]
                    if got:
                        present.update(p for p, _ in got)
                        c.emit("      " + c.style("metric", f"{label:14}")
                               + c.style("good", " · ".join(f"{p} {v}" for p, v in got)))
            missing = [(label, pkgs) for label, pkgs in mlx.GROUPS
                       if not any(p in present for p in pkgs)]
            if missing:
                c.emit()
                items = " · ".join(f"{label} ({'/'.join(pkgs)})" for label, pkgs in missing)
                c.emit("  " + c.style("dim", "not installed: " + items))
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


def render_install(c: Console, *, engine: str = "auto", as_json: bool = False) -> int:
    """Install the matched engine. ``--engine`` is the consent; exit 0 once the
    engine is present (installed or already), nonzero otherwise."""
    key = engines.resolve(engine)
    if key is None:
        if as_json:
            print(json.dumps({"status": "no_match", "engine": engine}))
        else:
            c.emit(c.style("warn", f"  no engine matches '{engine}' on this hardware"))
        return 1

    result = engines.install(key)
    if as_json:
        print(json.dumps({"key": result.key, "status": result.status,
                          "detail": result.detail}))
        return 0 if result.status in ("installed", "already") else 1

    pkg = engines.ENGINES[key]["package"]
    if result.status == "installed":
        c.emit(c.style("good", f"  installed {pkg}"))
    elif result.status == "already":
        c.emit(c.style("dim", f"  {pkg} already installed"))
    elif result.status == "coming_soon":
        c.emit(c.style("warn", f"  {pkg} — coming soon (not installable yet)"))
    else:  # failed
        c.emit(c.style("bad", f"  installing {pkg} failed:"))
        c.emit(c.style("dim", f"  {result.detail}"))
    return 0 if result.status in ("installed", "already") else 1


def render_uninstall(c: Console, *, engine: str = "auto", as_json: bool = False) -> int:
    """Remove the matched engine. Exit 0 once it's gone (removed or already absent)."""
    key = engines.resolve(engine)
    if key is None:
        if as_json:
            print(json.dumps({"status": "no_match", "engine": engine}))
        else:
            c.emit(c.style("warn", f"  no engine matches '{engine}' on this hardware"))
        return 1

    result = engines.uninstall(key)
    if as_json:
        print(json.dumps({"key": result.key, "status": result.status,
                          "detail": result.detail}))
        return 0 if result.status in ("removed", "absent") else 1

    pkg = engines.ENGINES[key]["package"]
    if result.status == "removed":
        c.emit(c.style("good", f"  removed {pkg}"))
    elif result.status == "absent":
        c.emit(c.style("dim", f"  {pkg} not installed"))
    else:  # failed
        c.emit(c.style("bad", f"  removing {pkg} failed:"))
        c.emit(c.style("dim", f"  {result.detail}"))
    return 0 if result.status in ("removed", "absent") else 1


def _overlay_stored_calibration(m: dict, engine_key: str | None) -> None:
    """Reuse ARA's stored overhead for this machine+engine, if any. ARA owns this record
    now — so a machine calibrated once shows as calibrated without re-measuring."""
    if engine_key is None:
        return
    stored = profiles.get_calibration(db.connect(), engine_key)
    if stored and stored.get("fixed_overhead_gb") is not None:
        m["overhead_gb"] = stored["fixed_overhead_gb"]
        m["calibrated"] = True
        m["calibrated_at"] = (stored.get("calibrated_at") or "")[:10] or None


def _persist_calibration(m: dict, engine_key: str | None) -> None:
    """Remember what the engine just measured: an overhead and/or a characterization."""
    if engine_key is None:
        return
    con = db.connect()
    if m.get("overhead_gb") is not None:
        profiles.save_calibration(con, engine_key, fixed_overhead_gb=m["overhead_gb"])
    ch = m.get("characterization")
    if ch and ch.get("safe_context") is not None:
        db.save_characterization(con, profiles.machine_key(), engine_key, ch["model"],
                                 safe_context=ch["safe_context"], points=ch["points"])
        catalog.remember(con, ch["model"])


def render_profile(c: Console, *, recalibrate: bool = False, as_json: bool = False,
                   assume_yes: bool = False, model: str | None = None,
                   engine: str | None = None) -> int:
    backend = detect.backend_name()
    if backend == "unsupported":
        c.emit(c.style("warn", "  profiling needs an ARA backend — none for this hardware yet."))
        return 1
    engine_ok, engine_pkg = engine_status()
    if not engine_ok:
        # --engine is consent to install. We can't import a just-installed package
        # in this process, so install then ask for a re-run rather than fake it.
        if engine is not None:
            if render_install(c, engine=engine) == 0:
                c.emit(c.style("accent", "  re-run ara profile") + c.style("dim", " to measure"))
            return 1
        c.emit(c.style("warn", f"  the {engine_pkg} engine isn't installed here — run: ")
               + c.style("accent", "ara install"))
        return 1

    bk = get_backend()
    try:
        m = bk.safe_limits()
    except Exception as exc:
        c.emit(c.style("bad", f"  couldn't read limits: {exc}"))
        return 1

    engine_key = engines.for_backend(backend)
    _overlay_stored_calibration(m, engine_key)   # reuse a stored measurement if we have one

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
        # Offer a re-measure only when there's a measured overhead to redo. An exactly-read
        # wall (e.g. CUDA VRAM) is calibrated with nothing to recalibrate.
        if m["overhead_gb"] is not None:
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

    _persist_calibration(m, engine_key)   # ARA remembers it, so next run is cached
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

    # --model / --engine / --include / --exclude take values; pull them out first.
    model: str | None = None
    engine: str | None = None
    include: list[str] = []
    exclude: list[str] = []
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
        if a == "--engine":
            engine = argv[i + 1] if i + 1 < len(argv) else None
            skip = True
            continue
        if a.startswith("--engine="):
            engine = a.split("=", 1)[1] or None
            continue
        if a in ("--include", "--exclude"):
            (include if a == "--include" else exclude).extend(
                _csv(argv[i + 1] if i + 1 < len(argv) else ""))
            skip = True
            continue
        if a.startswith("--include="):
            include.extend(_csv(a.split("=", 1)[1]))
            continue
        if a.startswith("--exclude="):
            exclude.extend(_csv(a.split("=", 1)[1]))
            continue
        if a in ("--verbose", "-v", "--json", "--recalibrate", "--yes", "-y"):
            continue
        rest.append(a)
    c = Console.from_env(verbose=verbose)

    if not rest or rest[0] in ("-h", "--help"):
        render_landing(c)
        return 0

    cmd = rest[0]
    # Section filtering is shared across the recon commands; build the predicate once.
    want = _resolve_want(cmd, include, exclude, c) if (include or exclude) else None

    if cmd == "detect":
        render_detect(c, as_json=as_json, want=want)
        return 0

    if cmd == "status":
        render_status(c, as_json=as_json, want=want)
        return 0

    if cmd == "python":
        render_python(c, as_json=as_json, want=want)
        return 0

    if cmd == "apps":
        render_apps(c, as_json=as_json, want=want)
        return 0

    if cmd == "mlx":
        render_mlx(c, as_json=as_json, want=want)
        return 0

    if cmd == "profile":
        return render_profile(c, recalibrate=recalibrate, as_json=as_json,
                              assume_yes=assume_yes, model=model, engine=engine)

    if cmd == "install":
        return render_install(c, engine=engine or "auto", as_json=as_json)

    if cmd == "uninstall":
        return render_uninstall(c, engine=engine or "auto", as_json=as_json)

    c.emit(c.style("warn", f"  '{rest[0]}' isn't built yet — ARA is an early scaffold."))
    c.emit(
        c.style("dim", "  run ") + c.style("accent", "ara")
        + c.style("dim", " with no arguments to see the planned commands.")
    )
    return 1
