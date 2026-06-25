# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
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

from ara import (acquire, apps, catalog, db, detect, engines, estimate, hub, hf_auth, mlx,
                 profile, calibration, pythons, serialize, status, versions)
from ara.registry import UnknownEngine, engine_status, get_backend, resolve_engine
from ara.ui import Console

_CMD_W = 16


def _cmd(c: Console, name: str, why: str) -> str:
    """One command row: accent name padded, dim gloss. Names wider than the column
    (e.g. ``characterize <model>``) still keep a gap so they don't collide with the gloss."""
    width = _CMD_W if len(name) < _CMD_W else len(name) + 2
    return "  " + c.style("accent", name.ljust(width)) + c.style("gloss", why)


def _fmt_gb(v: float | None, decimals: int = 0) -> str:
    return f"{v:.{decimals}f} GB" if v is not None else "unknown"


def _fmt_size(gb: float | None) -> str:
    """Human download size: MB under a gigabyte, GB above. 'size unknown' if None."""
    if gb is None:
        return "size unknown"
    return f"~{gb * 1000:.0f} MB" if gb < 1 else f"~{gb:.1f} GB"


def _fetch_error_msg(model: str, reason: str) -> str:
    """Turn an acquire reason code into a one-line, actionable user message."""
    if reason == "gated":
        return (f"{model} is gated — accept its terms on huggingface.co/{model} "
                f"then set HF_TOKEN")
    if reason == "not_found":
        return f"{model} not found or you don't have access"
    if reason == "offline":
        return (f"can't reach hugging face (and {model} isn't cached) "
                f"— check your connection")
    if reason == "auth":
        return f"{model}: authentication failed — check your HF_TOKEN"
    return f"couldn't fetch {model}: unknown error"


# --------------------------------------------------------------------------- #
# section filtering — shared across the recon commands (--include / --exclude)
# --------------------------------------------------------------------------- #
# The sections each recon command can show, in display order. Single-section
# commands list one key so the flags behave consistently everywhere.
_RECON_SECTIONS: dict[str, tuple[str, ...]] = {
    "detect": ("system", "memory", "accelerator", "storage", "board",
               "engines", "frameworks", "models", "apps", "ara"),
    "apps": ("runner", "image", "speech", "toolkit", "assistant", "coding"),
    "mlx": ("readiness", "libraries"),
    "status": ("processes", "apps"),
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

    accelerated = backend in ("apple", "cuda")
    backend_role = "good" if accelerated else "metric"   # cpu is a real backend, just not GPU
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
    if backend == "apple":  # MLX ecosystem view is Apple-Silicon only
        c.emit(_cmd(c, "mlx", "inspect the MLX ecosystem — libraries + readiness"))
    c.emit(_cmd(c, "search <query>", "find models on the Hugging Face Hub"))
    c.emit(_cmd(c, "models", "catalog the models on this machine + their safe ceilings"))
    c.emit(_cmd(c, "characterize <model>", "measure a model's safe context ceiling here"))
    c.emit(_cmd(c, "install", "install the engine matched to this machine"))
    c.emit(_cmd(c, "profile", "estimate this machine's capability (analytic — no engine)"))
    c.emit(_cmd(c, "recommend", "catalog models that fit, ranked by usable context"))
    c.emit(_cmd(c, "run <model>", "launch it safely — right up to the edge, never over"))
    c.emit()
    if not accelerated:
        c.emit(c.style("dim", "  no GPU backend detected — using the CPU fallback (llama.cpp); "
                              "install with ") + c.style("accent", "ara install --engine cpu"))
    c.emit(
        c.style("dim", "  try ") + c.style("accent", "ara detect")
        + c.style("dim", "  ·  run is next")
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
    if c.verbose:
        _det_cpu_detail(c, m)
    if m.python_version:
        gloss = "your default python3" if m.framework_python else "ARA's python (no user env found)"
        c.emit(c.field("python", m.python_version, gloss))
    n_py = pythons.count()
    if n_py > 1:
        c.emit(c.field("pythons", str(n_py), "interpreters on this machine — run: ara python"))
    c.emit()


def _det_cpu_detail(c: Console, m) -> None:
    """Verbose-only CPU detail block: vendor, clocks, caches, features."""
    cpu = m.cpu
    if cpu.vendor is not None:
        c.emit(c.field("  vendor", cpu.vendor))
    if cpu.logical is not None:
        c.emit(c.field("  threads", str(cpu.logical)))
    clocks = []
    if cpu.base_mhz is not None:
        clocks.append(f"base {cpu.base_mhz} MHz")
    if cpu.max_mhz is not None:
        clocks.append(f"max {cpu.max_mhz} MHz")
    if clocks:
        c.emit(c.field("  clocks", " · ".join(clocks)))
    caches = []
    if cpu.l1_kb is not None:
        caches.append(f"L1 {cpu.l1_kb} KB")
    if cpu.l2_kb is not None:
        caches.append(f"L2 {cpu.l2_kb} KB")
    if cpu.l3_kb is not None:
        caches.append(f"L3 {cpu.l3_kb} KB")
    if caches:
        c.emit(c.field("  cache", " · ".join(caches)))
    if cpu.features:
        c.emit(c.field("  features", " · ".join(cpu.features)))


def _det_memory(c: Console, m) -> None:
    c.emit(c.section("  MEMORY"))
    c.emit(c.field("total", _fmt_gb(m.ram_total_gb)))
    if m.ram_available_gb is not None:
        c.emit(c.field("available", _fmt_gb(m.ram_available_gb, 1), "free right now"))
    if m.swap_gb:
        c.emit(c.field("swap", _fmt_gb(m.swap_gb, 1)))
    if c.verbose:
        _det_memory_detail(c, m)
    c.emit()


def _det_memory_detail(c: Console, m) -> None:
    """Verbose-only memory detail: kind, speed, slot summary, per-module list."""
    mem = m.memory
    if mem.kind is not None:
        c.emit(c.field("  kind", mem.kind))
    if mem.speed_mts is not None:
        c.emit(c.field("  speed", f"{mem.speed_mts} MT/s"))
    # Slot summary
    if mem.slots_used is not None or mem.slots_total is not None:
        used = str(mem.slots_used) if mem.slots_used is not None else "?"
        total = str(mem.slots_total) if mem.slots_total is not None else "?"
        c.emit(c.field("  slots", f"{used} / {total} used"))
    # Per-module list
    if mem.modules:
        for mod in mem.modules:
            parts = []
            if mod.slot is not None:
                parts.append(mod.slot)
            if mod.capacity_gb is not None:
                parts.append(f"{mod.capacity_gb:.0f} GB")
            if mod.speed_mts is not None:
                parts.append(f"{mod.speed_mts} MT/s")
            if mod.manufacturer is not None:
                parts.append(mod.manufacturer)
            if mod.part_number is not None:
                parts.append(mod.part_number)
            if parts:
                c.emit(c.field("  module", " · ".join(parts)))
    elif mem.slots_used is None and mem.slots_total is None:
        # No slot info at all and no modules → platform doesn't expose them
        c.emit(c.field("  modules", "(not reported on this system)"))


_ARA_ENGINE_BACKENDS = {"cuda", "mlx"}   # backends ARA can actually run today
_RUNTIME_LABEL = {"vulkan": "Vulkan", "cuda": "CUDA", "mlx": "MLX", "rocm": "ROCm"}


def _gpu_line(c: Console, g) -> None:
    """Render one GpuInfo entry: a name·VRAM line, then a hint sub-line."""
    parts = [g.name or g.vendor.upper()]
    if g.vram_gb is not None:
        parts.append(f"{g.vram_gb:.0f} GB" + (" (shared)" if g.integrated else ""))
    if g.integrated:
        parts.append("integrated")
    c.emit(c.field("gpu", parts[0], " · ".join(parts[1:]) or None))
    # hint line
    if g.usable_backend:
        label = _RUNTIME_LABEL.get(g.usable_backend, g.usable_backend)
        rt = g.compute_runtime or label
        if g.usable_backend in _ARA_ENGINE_BACKENDS:
            hint = f"{rt} — usable"
        else:
            hint = f"{rt} — usable via {label}, ARA engine coming (not yet runnable)"
    elif g.compute_runtime:
        hint = f"{g.compute_runtime} present — not ARA's path"
    else:
        hint = "no usable GPU runtime detected"
    c.emit(c.field("", "", hint))


def _is_usable_accel(a, g) -> bool:
    """True when g is the same GPU already shown in the rich accelerator block."""
    return (a.kind == "nvidia" and g.vendor == "nvidia") or \
           (a.kind == "apple" and g.vendor == "apple")


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
        for g in getattr(m, "gpus", []):
            if not _is_usable_accel(a, g):
                _gpu_line(c, g)
    elif a.kind == "apple":
        cores = f"{a.cores}-core " if a.cores else ""
        c.emit(c.field("gpu", a.name, f"{cores}Metal · unified memory (shared with system)"))
        for g in getattr(m, "gpus", []):
            if not _is_usable_accel(a, g):
                _gpu_line(c, g)
    else:
        if getattr(m, "gpus", None):
            for g in m.gpus:
                _gpu_line(c, g)
        else:
            c.emit(c.field("gpu", a.name, "no GPU detected", value_role="warn"))
    c.emit()


def _det_storage(c: Console, m) -> None:
    c.emit(c.section("  STORAGE"))
    c.emit(c.field("disk free", _fmt_gb(m.disk_free_gb), "on the home volume"))
    if c.verbose:
        _det_storage_detail(c, m)
    c.emit()


def _det_storage_detail(c: Console, m) -> None:
    """Verbose-only storage detail: per-drive list."""
    drives = m.storage.drives
    if drives:
        for drive in drives:
            parts = []
            if drive.model is not None:
                parts.append(drive.model)
            if drive.media is not None:
                parts.append(drive.media)
            if drive.size_gb is not None:
                parts.append(f"{drive.size_gb:.0f} GB")
            if parts:
                c.emit(c.field("  drive", " · ".join(parts)))


def _det_board(c: Console, m) -> None:
    """Verbose-only BOARD section: board vendor/model, BIOS version/date, system vendor/model."""
    if not c.verbose:
        return
    board = m.board
    # Check if there's anything to show at all
    any_board = any(v is not None for v in (
        board.board_vendor, board.board_model, board.bios_version,
        board.bios_date, board.system_vendor, board.system_model,
    ))
    if not any_board:
        return
    c.emit(c.section("  BOARD"))
    # Labels here are up to 13 chars ("system vendor") — widen the pad past the 12 default so the
    # value never butts against the label.
    w = 14
    if board.board_vendor is not None:
        c.emit(c.field("board vendor", board.board_vendor, label_width=w))
    if board.board_model is not None:
        c.emit(c.field("board model", board.board_model, label_width=w))
    if board.bios_version is not None:
        c.emit(c.field("bios", board.bios_version, label_width=w))
    if board.bios_date is not None:
        c.emit(c.field("bios date", board.bios_date, label_width=w))
    if board.system_vendor is not None:
        c.emit(c.field("system vendor", board.system_vendor, label_width=w))
    if board.system_model is not None:
        c.emit(c.field("system model", board.system_model, label_width=w))
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
        "auto-picked for this hardware" if m.accelerated
        else "CPU fallback — no GPU backend detected",
        value_role="good",
    ))
    c.emit(c.field(
        "engine", f"{m.engine} {'ready' if m.engine_ready else 'not installed'}",
        None if m.engine_ready else "install: ara install",
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


_DETECT_RENDERERS: tuple[tuple[str, object], ...] = (
    ("system", _det_system),
    ("memory", _det_memory),
    ("accelerator", _det_accelerator),
    ("storage", _det_storage),
    ("board", _det_board),
    ("engines", _det_engines),
    ("frameworks", _det_frameworks),
    ("models", _det_models),
    ("apps", _det_apps),
    ("ara", _det_ara),
)


def render_detect(c: Console, *, as_json: bool = False, want=None) -> None:
    m = detect.machine()
    if as_json:
        # serialize.machine(m) is the single source of truth for the detect --json shape:
        # asdict(m) (nested cpu/memory/storage/board already included as dicts) PLUS the
        # `accelerated` @property asdict drops — otherwise the CPU-fallback distinction the
        # design introduced would be invisible to machine consumers.
        print(json.dumps(serialize.machine(m), indent=2))
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


def _emit_workloads(c: Console, procs) -> None:
    """RUNNING AI/ML — local inference workloads consuming memory/GPU right now."""
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


def _emit_apps(c: Console, apps_) -> None:
    """AI APPS — running client apps (remote-API GUIs/CLIs). RSS is ordinary RAM, not ML."""
    c.emit(c.section("  AI APPS"))
    if not apps_:
        c.emit(c.style("dim", "  no AI apps running right now"))
        c.emit()
        return
    for a in apps_:
        plural = "proc" if a.n_procs == 1 else "procs"
        bits = [f"{a.n_procs} {plural}", f"up {_fmt_uptime(a.uptime_s)}"]
        c.emit(c.field(a.label, _fmt_mem(a.rss_gb), " · ".join(bits), value_role="metric"))
    c.emit()


def render_status(c: Console, *, as_json: bool = False, want=None) -> None:
    procs = status.scan()
    apps_ = status.scan_apps()

    if as_json:
        print(json.dumps({"workloads": [asdict(p) for p in procs],
                          "apps": [asdict(a) for a in apps_]}, indent=2))
        return

    show = want or (lambda _key: True)
    if show("processes"):
        _emit_workloads(c, procs)
    if show("apps"):
        _emit_apps(c, apps_)


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
# characterize (measures — crosses the seam into the engine)
# --------------------------------------------------------------------------- #
def _emit_limits(c: Console, m: dict) -> None:
    # The tag must match the data source: a measured wall reads as measured; without one it's
    # honestly flagged as an uncalibrated estimate. Spec 2026-06-23-capability-pipeline.
    measured = m.get("basis") == "measured"
    tag = "  (measured)" if measured else "  (estimated — not calibrated)"
    c.emit()
    c.emit(c.section("  SAFE LIMITS") + c.style("dim", tag))
    c.emit(c.field("device", f"{m['device']} · {m['total_gb']:.0f} GB"))
    c.emit(c.field("crash wall", _fmt_gb(m["wall_gb"], 1),
                   "the hard ceiling — never cross", value_role="bad"))
    c.emit(c.field("safe budget", _fmt_gb(m["safe_budget_gb"], 1),
                   f"wall − {m['margin_gb']:.0f} GB margin", value_role="good"))
    if m.get("headroom_gb") is not None:
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


def render_model_detail(c: Console, model_id: str, *, as_json: bool = False) -> int:
    """Detail for one model: architecture (from its HF config) + its safe ceiling here."""
    meta = catalog.describe(model_id)
    if meta is None:
        if as_json:
            print(json.dumps({"error": f"couldn't describe {model_id}"}))
        else:
            c.emit(c.style("warn", f"  couldn't describe {model_id} — is it downloaded / a valid repo?"))
        return 1
    con = db.connect()
    mk = profile.machine_key()
    # Per-engine: a model can be characterized under several engines on one machine (GPU + CPU).
    per_engine = {}                       # engine_key -> (safe_context, decode_context)
    for key in engines.ENGINES:
        row = db.get_characterization(con, mk, key, model_id)
        if row is not None:
            per_engine[key] = (row["safe_context"], row.get("decode_context"))
    best = max((sc for (sc, _) in per_engine.values() if sc is not None), default=None)
    # Pair decode_context with the engine that owns the best safe_context, so the two
    # top-level JSON scalars describe the same engine — not independent max() picks.
    best_engine_pair = max(
        ((sc, dc) for (sc, dc) in per_engine.values() if sc is not None),
        key=lambda t: t[0], default=None,
    )
    best_decode = best_engine_pair[1] if best_engine_pair is not None else None
    if as_json:
        print(json.dumps({"model_id": model_id, **meta, "safe_context": best,
                          "decode_context": best_decode,
                          "engines": {k: sc for k, (sc, _) in per_engine.items()},
                          "characterized": bool(per_engine)}, indent=2))
        return 0
    kvh, hd = meta["kv_heads"], meta["head_dim"]
    c.emit()
    c.emit(c.section(f"  {model_id}"))
    c.emit(c.field("modality", meta["modality"] or "?"))
    c.emit(c.field("layers", str(meta["n_layers"]) if meta["n_layers"] else "?"))
    c.emit(c.field("kv cache", f"{kvh} heads × {hd} dim" if (kvh and hd) else "?"))
    c.emit(c.field("max context", str(meta["max_context"]) if meta["max_context"] else "?"))
    c.emit(c.field("quant", meta["quant"] or "none"))
    if per_engine:                        # one ceiling line per engine that measured it
        for key, (sc, dc) in per_engine.items():
            ceiling_str = f"~{sc} tokens" if sc else "no safe ceiling"
            if sc and dc and dc > sc:
                ceiling_str += f"  · ~{dc} stream-only (est.)"
            c.emit(c.field(f"{key} ceiling", ceiling_str))
    else:
        c.emit(c.field("ceiling", "not characterized"))
    c.emit()
    return 0


def render_characterize(c: Console, model: str, *, engine: str | None = None,
                        as_json: bool = False) -> int:
    """Measure a model's safe context ceiling on an engine, and store it.

    Defaults to the detected engine; ``--engine`` overrides it so you can target a non-detected
    backend (e.g. the CPU fallback on a GPU box). ARA owns the result, so it shows up in
    `ara models` regardless of which engine measured it."""
    try:
        sel = resolve_engine(engine)
    except UnknownEngine:
        msg = f"unknown engine {engine!r} — try one of: {', '.join(engines.ENGINES)}"
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1
    if not acquire.valid_model_id(model):
        msg = f"invalid model id {model!r} — expected a Hugging Face repo id (org/name)"
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1
    engine_ok, engine_pkg = engine_status(sel.backend)
    if not engine_ok:
        if as_json:
            print(json.dumps({"error": f"{engine_pkg} engine not installed"}))
        else:
            c.emit(c.style("warn", f"  the {engine_pkg} engine isn't installed — run: ")
                   + c.style("accent", f"ara install --engine {sel.engine_key}"))
        return 1
    bk = get_backend(sel.backend)
    progress = (not as_json) and sys.stderr.isatty()
    # Pre-fetch: ensure weights are in the HF cache before the engine's preflight runs.
    # Without this, the worker's blobs/ scan yields weights_gb≈0 for uncached transformers
    # models, so the a-priori safety gates (L1/L4) under-predict memory on the first rung.
    # cpu.calibration_model_cached() always returns True, so this only fires for apple/cuda.
    incompatible = engines.engine_for_model(model) not in (None, sel.engine_key)
    if not incompatible and not bk.calibration_model_cached(model):
        size_gb = acquire.repo_size_gb(model)
        free_gb = acquire.free_disk_gb()
        if size_gb and free_gb is not None and free_gb < size_gb + acquire.DISK_BUFFER_GB:
            msg = (f"not enough disk for {model}: needs ~{size_gb:.1f} GB + "
                   f"{acquire.DISK_BUFFER_GB:.0f} GB headroom, only {free_gb:.1f} GB free.")
            if as_json:
                print(json.dumps({"error": msg}))
            else:
                c.emit(c.style("bad", f"  {msg}"))
            return 1
        c.emit(c.style("dim", f"  downloading {model} … ({_fmt_size(size_gb)})"))
        try:
            bk.download_calibration_model(model, progress=progress)
        except Exception as exc:
            reason = acquire.classify_repo_error(exc)
            msg = _fetch_error_msg(model, reason)
            print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
            return 1
    # characterize owns calibration: measure + persist the engine baseline once (when none is
    # stored) so the ramp uses the real overhead, not the default. Spec 2026-06-23-capability-pipeline.
    cal_con = db.connect()
    if hasattr(bk, "calibrate") and calibration.get_calibration(cal_con, sel.engine_key) is None:
        c.emit(c.style("dim", f"  calibrating {sel.engine_key} … (first run on this machine)"))
        cal = bk.calibrate()
        overhead = (cal or {}).get("overhead_gb")
        wall = (cal or {}).get("wall_gb")
        # Honesty (Rule #3): if calibration couldn't run (model missing, worker error), say so —
        # never let the conservative default masquerade as a measurement. The ramp still proceeds
        # safely on the default overhead; we just don't hide that it's a fallback.
        cal_err = (cal or {}).get("calibration_error")
        if cal_err:
            c.emit(c.style("warn", f"  calibration skipped: {cal_err}"
                                   " — using conservative default overhead"))
        # Persist whatever the engine measured: the cold-start overhead (Apple) and/or the exact
        # wall + safe budget (CPU/CUDA read an exact wall, so overhead is None there). Storing the
        # wall regardless of overhead is what lets profile/recommend report reality on every engine,
        # not just the ones with a measured overhead. Spec 2026-06-23-capability-pipeline.
        if overhead is not None or wall is not None:
            calibration.save_calibration(
                cal_con, sel.engine_key, fixed_overhead_gb=overhead,
                wall_gb=wall, safe_budget_gb=(cal or {}).get("safe_budget_gb"))
        # Surface the measured wall right where it's measured — otherwise the user sees only the
        # ceiling and the calibrated reality stays invisible. Guard on a real wall so engines that
        # measure only cold-start overhead don't print an empty line. Spec 2026-06-23-capability-pipeline.
        if wall is not None:
            budget = (cal or {}).get("safe_budget_gb")
            line = c.field("measured wall", _fmt_gb(wall, 1), label_width=15)
            if budget is not None:
                line += "  · " + c.style("dim", f"safe budget {_fmt_gb(budget, 1)}")
            c.emit(line)
    c.emit(c.style("dim", f"  characterizing {model} … (loads the model on the device)"))
    try:
        result = bk.characterize(model, progress=progress)
    except (SystemExit, Exception) as exc:   # engine may refuse/abort/OOM-guard
        c.emit(c.style("bad", f"  characterization failed: {exc}"))
        return 1

    # An engine that couldn't even load the model returns an `error` (not a measurement) — don't
    # persist a misleading null row. Suggest a compatible engine when we can tell cheaply (e.g. a
    # GGUF handed to the torch-based wcx → suggest the CPU/llama.cpp engine).
    if result.get("error"):
        suggest = engines.engine_for_model(model)
        hint = ("  — try " + c.style("accent", f"ara characterize {model} --engine {suggest}")
                if suggest and suggest != sel.engine_key else "")
        if as_json:
            print(json.dumps({"error": result["error"]}))
        else:
            c.emit(c.style("warn", f"  {engine_pkg} couldn't load {model}: {result['error']}") + hint)
        return 1

    ceiling = result["safe_context"]
    con = db.connect()
    db.save_characterization(con, profile.machine_key(), sel.engine_key,
                             model, safe_context=ceiling, points=result["points"],
                             decode_context=result.get("decode_context"))
    catalog.remember(con, model)

    if as_json:
        print(json.dumps({"model": model, "safe_context": ceiling,
                          "decode_context": result.get("decode_context")}, indent=2))
        return 0
    if ceiling:
        c.emit(c.style("good", f"  safe context ceiling  ~{ceiling} tokens")
               + c.style("dim", "  · stored (see ara models)"))
        dc = result.get("decode_context")
        if dc and dc > ceiling:
            c.emit(c.style("good", f"  decode ceiling (est.)  ~{dc} tokens")
                   + c.style("dim", "  · grow-by-streaming, not a prompt size"))
    else:
        c.emit(c.style("warn", "  couldn't fit a ceiling — the model may be too big or borderline"))
    c.emit()
    return 0


def render_search(c: Console, query: str, *, as_json: bool = False) -> int:
    """Search the Hugging Face Hub for models (engine-agnostic)."""
    results = hub.search(query)
    if results is None:
        c.emit(c.style("warn", "  couldn't search — is the hf CLI installed? ")
               + c.style("accent", "pip install huggingface_hub"))
        return 1
    if as_json:
        print(json.dumps(results, indent=2))
        return 0
    c.emit()
    c.emit(c.section(f"  HUB SEARCH: {query}"))
    for r in results:
        c.emit("  " + c.style("metric", r["id"])
               + c.style("dim", f"  ↓{r['downloads']} · ♥{r['likes']}"))
    if not results:
        c.emit(c.style("dim", "  no models found"))
    c.emit()
    return 0


def _best_ceilings(con) -> dict[str, tuple[int | None, str, int | None]]:
    """Best safe-context per model across engines: ``{model_id: (safe_context, engine_key, decode_context)}``.

    A model can be characterized under several engines on one machine (GPU + CPU); ``ara models``
    shows the largest ceiling and which engine reached it. A real ceiling beats a null
    (measured-but-unfit) one; ties favour the detected default engine (considered first)."""
    mk = profile.machine_key()
    default = engines.for_backend(detect.backend_name())
    best: dict[str, tuple[int | None, str, int | None]] = {}
    for key in dict.fromkeys([default, *engines.ENGINES]):
        if key is None:
            continue
        for r in db.list_characterizations(con, mk, key):
            mid, sc = r["model_id"], r["safe_context"]
            cur = best.get(mid)
            if cur is None or (sc is not None and (cur[0] is None or sc > cur[0])):
                best[mid] = (sc, key, r.get("decode_context"))
    return best


def render_models(c: Console, *, as_json: bool = False, want=None) -> None:
    """The model catalog: scan the HF cache, then list each model + its best safe ceiling here."""
    con = db.connect()
    catalog.scan(con)
    models = catalog.all_models(con)
    best = _best_ceilings(con)

    if as_json:
        print(json.dumps(
            [{**m,
              "safe_context": best[m["model_id"]][0] if m["model_id"] in best else None,
              "engine": best[m["model_id"]][1] if m["model_id"] in best else None,
              "decode_context": best[m["model_id"]][2] if m["model_id"] in best else None,
              "characterized": m["model_id"] in best} for m in models], indent=2))
        return

    c.emit()
    c.emit(c.section("  MODEL CATALOG"))
    for m in models:
        mid = m["model_id"]
        if mid in best:                           # measured under at least one engine
            ceiling, ekey, decode = best[mid]
            tail = f"~{ceiling} tokens ({ekey})" if ceiling else "no safe ceiling"
            if ceiling and decode and decode > ceiling:
                tail = f"~{ceiling} tokens ({ekey}) · ~{decode} stream-only (est.)"
            role = "good" if ceiling else "dim"   # measured-but-unfit mirrors profile's '—'
        else:
            tail, role = "not characterized", "dim"
        c.emit("  " + c.style("metric", mid)
               + c.style("dim", f"  {m['modality'] or '?'}  →  ")
               + c.style(role, tail))
    if not models:
        c.emit(c.style("dim", "  empty — download a model and it'll be cataloged here"))
    c.emit()
    n_char = sum(1 for m in models if m["model_id"] in best)
    c.emit(c.style("dim", f"  {len(models)} cataloged · {n_char} characterized on this machine"))
    c.emit()


def render_recommend(c: Console, *, as_json: bool = False) -> int:
    """Analytic recommendations — which cataloged models fit this machine, ranked by the context
    the estimated budget supports (most first), marking those already characterized here.

    Engine-free: reuses ``estimate.limits``/``model_fit`` (profile's math, anti-silo) over the
    catalog (which records each model's on-disk weight). No engine, no model load. Only models
    with a rankable context estimate are listed. Spec 2026-06-23-capability-pipeline."""
    con = db.connect()
    catalog.scan(con)                 # refresh the catalog from the local cache first
    # Prefer the measured wall for the detected engine (anti-silo: same grounding as profile).
    default_engine = engines.for_backend(detect.backend_name())
    measured = (calibration.get_calibration(con, default_engine)
                if default_engine is not None else None)
    lim = estimate.limits(detect.machine(), measured=measured)
    best = _best_ceilings(con)        # model_id -> (safe_context, engine_key, decode_context)

    recs = []
    unrankable = 0                    # weights fit, but we can't read the arch to estimate context
    for row in catalog.all_models(con):
        fit = estimate.model_fit(lim, row, row.get("weights_gb"))
        if not fit["fits"]:
            continue
        if fit["est_context"] is None:
            unrankable += 1           # honest: count it rather than drop it silently
            continue
        recs.append({"model_id": row["model_id"], "modality": row.get("modality"),
                     "est_context": fit["est_context"], "max_context": fit["max_context"],
                     "binding": fit["binding"], "fits": True,
                     "characterized": row["model_id"] in best})
    recs.sort(key=lambda r: r["est_context"], reverse=True)

    if as_json:
        print(json.dumps(recs, indent=2))
        return 0

    def _unrankable_note() -> None:
        if unrankable:
            c.emit(c.style("dim", f"  {unrankable} more fit but can't be ranked "
                                   "(architecture unknown) — try ara profile --model <model>"))

    c.emit()
    c.emit(c.section("  RECOMMENDED MODELS")
           + c.style("dim", "  (estimated — fits this machine, most context first)"))
    if not recs:
        c.emit(c.style("dim", "  nothing in the catalog fits the estimated budget — "
                              "try a smaller / more-quantized model"))
        _unrankable_note()
        c.emit()
        return 0
    for r in recs:
        tail = f"~{r['est_context']} tok est."
        if r["binding"] == "context_window":
            tail += " (full window)"
        mark = c.style("good", "  · characterized here") if r["characterized"] else ""
        c.emit("  " + c.style("metric", r["model_id"])
               + c.style("dim", f"  {r['modality'] or '?'}  →  ")
               + c.style("accent", tail) + mark)
    n_char = sum(1 for r in recs if r["characterized"])
    c.emit()
    c.emit(c.style("dim", f"  {len(recs)} fit · {n_char} characterized here"))
    _unrankable_note()
    c.emit()
    return 0


RUN_MAX_TOKENS = 256


def render_run(c: Console, model: str, *, prompt: str | None = None, engine: str | None = None,
               assume_yes: bool = False, as_json: bool = False,
               max_tokens: int = RUN_MAX_TOKENS) -> int:
    """Governed one-shot inference: generate a completion for *model*, capped at its characterized
    safe context ceiling (launch under the wall, never over). Requires a *measured* ceiling — if
    the model isn't characterized here, it refuses and points at ``ara characterize``. Loads the
    engine + model out-of-process. Spec 2026-06-23-capability-pipeline."""
    def err(msg: str) -> int:
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1

    try:
        sel = resolve_engine(engine)
    except UnknownEngine:
        return err(f"unknown engine {engine!r} — try one of: {', '.join(engines.ENGINES)}")
    if not prompt:
        return err("usage: ara run <model> <prompt>")
    if not acquire.valid_model_id(model):
        return err(f"invalid model id {model!r} — expected a Hugging Face repo id (org/name)")

    con = db.connect()
    mk = profile.machine_key()
    suffix = "" if engine is None else f" --engine {sel.engine_key}"

    if engine is not None:
        # Pinned: use exactly the named engine — honour the explicit choice, don't second-guess it.
        row = db.get_characterization(con, mk, sel.engine_key, model)
        if row is None:
            return err(f"{model} isn't characterized on {sel.engine_key} yet — run: "
                       f"ara characterize {model}{suffix}")
        if row.get("safe_context") is None:
            return err(f"{model} was characterized but didn't fit on {sel.engine_key} — "
                       f"too big for this machine")
        engine_key, backend, safe = sel.engine_key, sel.backend, row["safe_context"]
    else:
        # No --engine: scan every engine this model is characterized under on this machine and pick
        # the largest measured ceiling whose backend can actually run (has `generate`). A model
        # characterized on the CPU fallback runs there even when the detected backend differs.
        # Mirror _best_ceilings' iteration: detected default first so ties favour it. The default
        # is never None here — resolve_engine(None) above would have raised if the detected backend
        # had no engine — so [default, *ENGINES] holds only real keys.
        default = engines.for_backend(detect.backend_name())
        per_engine = {}                  # engine_key -> (safe_context, backend, can_run)
        for key in dict.fromkeys([default, *engines.ENGINES]):
            row = db.get_characterization(con, mk, key, model)
            if row is None:
                continue
            backend = engines.ENGINES[key]["backend"]
            per_engine[key] = (row.get("safe_context"), backend,
                               hasattr(get_backend(backend), "generate"))
        if not per_engine:
            return err(f"{model} isn't characterized on {sel.engine_key} yet — run: "
                       f"ara characterize {model}")
        fitted = {k: v for k, v in per_engine.items() if v[0] is not None}
        if not fitted:
            return err(f"{model} was characterized but didn't fit on {sel.engine_key} — "
                       f"too big for this machine")
        runnable = {k: v for k, v in fitted.items() if v[2]}
        if not runnable:
            # Characterized + fits, but only on engine(s) ARA can't run through yet (apple/cuda).
            # Be honest about that — don't masquerade as uncharacterized.
            where = ", ".join(fitted)
            return err(f"{model} is characterized on {where}, but run isn't supported on "
                       f"that engine yet")
        # Largest ceiling wins; the dict is detected-first, so a strict `>` lets ties favour it.
        engine_key = max(runnable, key=lambda k: runnable[k][0])
        safe, backend, _ = runnable[engine_key]

    engine_ok, engine_pkg = engine_status(backend)
    if not engine_ok:
        return err(f"the {engine_pkg} engine isn't installed — run: ara install{suffix}")
    bk = get_backend(backend)
    if not hasattr(bk, "generate"):
        return err(f"run isn't supported on the {engine_pkg} engine yet")

    # Consent before load (a courtesy — the ceiling already makes it wall-safe). Interactive only;
    # --yes or a non-tty (scripts/--json) proceed straight to the governed run.
    if not as_json and not assume_yes and sys.stdin.isatty():
        if not _confirm(f"Load {model} on {engine_pkg} and generate (≤ ~{safe} tokens)?"):
            c.emit(c.style("dim", "  skipped."))
            return 0

    if not as_json:
        c.emit(c.style("dim", f"  running {model} on {engine_pkg} … (≤ ~{safe} tokens)"))
    try:
        result = bk.generate(model, prompt, max_context=safe, max_tokens=max_tokens)
    except (SystemExit, Exception) as exc:        # engine may refuse/abort/OOM-guard
        return err(f"run failed: {exc}")
    if result.get("refused"):
        return err(f"the {engine_pkg} engine refused: {result.get('reason', 'no reason given')}")

    completion = result.get("completion", "")
    if as_json:
        print(json.dumps({"model": model, "engine": engine_key,
                          "safe_context": safe, "completion": completion}, indent=2))
        return 0
    c.emit()
    c.emit(completion)
    c.emit()
    return 0


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


def _emit_characterized(c: Console, engine_key: str | None) -> None:
    """Show models ARA has characterized on this machine + engine (from the store)."""
    if engine_key is None:
        return
    rows = db.list_characterizations(db.connect(), profile.machine_key(), engine_key)
    if not rows:
        return
    c.emit(c.section("  CHARACTERIZED MODELS"))
    for r in rows:
        name = r["model_id"].split("/")[-1]
        if r["safe_context"]:
            dc = r.get("decode_context")
            ceiling = f"~{r['safe_context']} tokens"
            if dc and dc > r["safe_context"]:
                ceiling += f"  · ~{dc} stream-only (est.)"
        else:
            ceiling = "—"
        c.emit("  " + c.style("metric", name) + c.style("dim", "  →  ")
               + c.style("good", ceiling) + c.style("dim", "  safe context ceiling"))
    c.emit()


def _model_fit(lim: dict, model: str) -> dict | None:
    """Analytic fit for *model* against the estimated limits — no engine, no model load.

    Combines the model's architecture (for the KV slope) with its weight footprint. The weight
    comes from the catalog's stored ``weights_gb`` first (local, from the HF cache — exactly what
    ``recommend`` uses, so the two compute identically with no network call); only when the
    catalog has no weight for it do we fall back to the HF API (``acquire.repo_size_gb``). Returns
    None when the model can't be described (bad repo / not cached)."""
    meta = catalog.describe(model)
    if meta is None:
        return None
    con = db.connect()
    row = catalog.get(con, model) or catalog.remember(con, model)
    weights_gb = row.get("weights_gb") if row else None
    if weights_gb is None:
        weights_gb = acquire.repo_size_gb(model)
    return estimate.model_fit(lim, meta, weights_gb)


def _emit_model_fit(c: Console, lim: dict, model: str) -> None:
    """Render the per-model analytic verdict: does it fit, and what context does the budget hold?"""
    fit = _model_fit(lim, model)
    c.emit()
    c.emit(c.section(f"  MODEL FIT: {model}") + c.style("dim", "  (estimated)"))
    if fit is None:
        c.emit(c.style("warn", f"  couldn't describe {model} — is it a valid repo / downloaded?"))
        c.emit()
        return
    c.emit(c.field("weights", _fmt_gb(fit["weights_gb"], 1), "estimated in-memory footprint"))
    if not fit["fits"]:
        c.emit(c.field("verdict", "won't fit", "weights alone exceed the estimated budget",
                       value_role="bad"))
    elif fit["binding"] == "context_window":
        c.emit(c.field("verdict", f"fits — full {fit['max_context']} ctx",
                       "budget covers the model's whole window", value_role="good"))
    elif fit["est_context"]:
        c.emit(c.field("verdict", f"context-limited ~{fit['est_context']} tok",
                       f"the budget binds before the model's {fit['max_context']} window",
                       value_role="warn"))
    else:
        c.emit(c.field("verdict", "fits", "context estimate unavailable (unknown architecture)",
                       value_role="good"))
    c.emit(c.style("dim", "  estimated — run ") + c.style("accent", f"ara characterize {model}")
           + c.style("dim", " to measure the real ceiling"))
    c.emit()


def render_profile(c: Console, *, as_json: bool = False, model: str | None = None,
                   engine: str | None = None) -> int:
    """Analytic capability assessment — engine-free. Reasons over ``detect.machine()`` facts +
    ARA's heuristics to estimate the memory budget, persists the profile, and (with ``--model``)
    checks whether a model's weights + context window fit the estimate. It never loads an engine
    or a model; ``characterize`` does that to measure the real ceiling.
    Spec 2026-06-23-capability-pipeline."""
    try:
        sel = resolve_engine(engine)
    except UnknownEngine:
        msg = f"unknown engine {engine!r} — try one of: {', '.join(engines.ENGINES)}"
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1

    m = detect.machine()
    # Ground the estimate in reality: if this engine has a measured wall stored from a prior
    # characterize, report the MEASURED budget (labelled), not the heuristic. Read-only — still
    # engine-free (no engine import/load). Spec 2026-06-23-capability-pipeline.
    measured = calibration.get_calibration(db.connect(), sel.engine_key)
    lim = estimate.limits(m, measured=measured)

    # profile is the persister (detect stays ephemeral): record this analytic snapshot, history kept.
    profile.capture(db.connect())

    if as_json:
        payload = {**lim}
        if model is not None:
            payload["model_fit"] = _model_fit(lim, model)
        print(json.dumps(payload, indent=2))
        return 0

    _emit_limits(c, lim)
    _emit_characterized(c, sel.engine_key)   # models ARA has already measured here

    if model is not None:
        _emit_model_fit(c, lim, model)
    else:
        # The footer must agree with the SAFE LIMITS tag: once the wall is measured, dropping the
        # "estimated —" framing keeps it honest. Spec 2026-06-23-capability-pipeline.
        if lim["basis"] == "measured":
            c.emit(c.style("dim", "  run ")
                   + c.style("accent", "ara characterize <model>")
                   + c.style("dim", " to measure a model's real ceiling"))
        else:
            c.emit(c.style("dim", "  estimated — run ")
                   + c.style("accent", "ara characterize <model>")
                   + c.style("dim", " to measure a real ceiling"))
        c.emit()
    return 0


# --------------------------------------------------------------------------- #
# hf login / logout / status
# --------------------------------------------------------------------------- #

def _read_token(c) -> str:
    """Read a HF token interactively (TTY) or from piped stdin. Testable seam."""
    import getpass
    if sys.stdin.isatty():
        return getpass.getpass("  paste your hugging face token (hidden): ")
    return sys.stdin.readline()


def render_hf(c: Console, sub: str | None, *, token: str | None = None,
              as_json: bool = False) -> int:
    """Dispatch hf subcommands: login / logout / status."""

    def _err(msg: str) -> int:
        if as_json:
            print(json.dumps({"error": msg}))
        else:
            c.emit(c.style("bad", f"  {msg}"))
        return 1

    if sub == "login":
        # Warn when token comes from --token flag (may persist in shell history).
        if token is not None and not as_json:
            c.emit(c.style("warn",
                           "  note: --token may be saved in your shell history; "
                           "prefer the interactive prompt"))
        if token is None:
            token = _read_token(c)
        if not token or not token.strip():
            return _err("no token provided")
        res = hf_auth.set_token(token)
        if not res["saved"]:
            msg = ("that token was rejected by the hub"
                   if res["error"] == "invalid" else "no token provided")
            return _err(msg)
        if as_json:
            print(json.dumps({"saved": True, "user": res["user"], "verified": res["verified"]}))
            return 0
        if res["verified"]:
            c.emit(c.style("good", f"  logged in as {res['user']}"))
        else:
            c.emit(c.style("warn",
                           f"  token saved — couldn't verify ({res['error']})"))
        return 0

    if sub == "logout":
        res = hf_auth.clear_token()
        if as_json:
            print(json.dumps(res))
            return 0
        if res["removed"]:
            c.emit(c.style("good", "  removed the stored hugging face token"))
        else:
            c.emit(c.style("dim", "  no stored hugging face token to remove"))
        if res["shadowed_by_env"]:
            c.emit(c.style("warn",
                           "  an HF_TOKEN env var is still set — "
                           "ARA can't unset your environment"))
        return 0

    if sub == "status":
        st = hf_auth.status()
        if as_json:
            print(json.dumps(st))
            return 0
        if not st["present"]:
            c.emit(c.style("dim", "  not logged in — run ")
                   + c.style("accent", "ara hf login")
                   + c.style("dim", " (needed for gated models)"))
            return 0
        if st["verified"]:
            c.emit(c.style("good", f"  logged in as {st['user']}"))
            c.emit(c.style("dim", f"  · token from {st['source']}"))
        else:
            c.emit(c.style("warn",
                           f"  token present ({st['source']}) but couldn't verify ({st['error']})"))
        return 0

    if sub is None:
        return _err("specify an hf subcommand — try one of: login, logout, status")
    return _err(f"unknown hf subcommand {sub!r} — try one of: login, logout, status")


# --------------------------------------------------------------------------- #
# entry
# --------------------------------------------------------------------------- #
def main() -> int:
    argv = sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv
    as_json = "--json" in argv
    assume_yes = "--yes" in argv or "-y" in argv

    # --model / --engine / --token / --include / --exclude take values; pull them out first.
    model: str | None = None
    engine: str | None = None
    token: str | None = None
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
        if a == "--token":
            token = argv[i + 1] if i + 1 < len(argv) else None
            skip = True
            continue
        if a.startswith("--token="):
            token = a.split("=", 1)[1]
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
        if a in ("--verbose", "-v", "--json", "--yes", "-y"):
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

    if cmd == "models":
        if len(rest) > 1:                       # `ara models <id>` → one model's detail
            return render_model_detail(c, rest[1], as_json=as_json)
        render_models(c, as_json=as_json, want=want)
        return 0

    if cmd == "search":
        if len(rest) < 2:
            c.emit(c.style("warn", "  usage: ara search <query>"))
            return 1
        return render_search(c, " ".join(rest[1:]), as_json=as_json)

    if cmd == "characterize":
        if len(rest) < 2:
            c.emit(c.style("warn", "  usage: ara characterize <model>"))
            return 1
        return render_characterize(c, rest[1], engine=engine, as_json=as_json)

    if cmd == "profile":
        return render_profile(c, as_json=as_json, model=model, engine=engine)

    if cmd == "recommend":
        return render_recommend(c, as_json=as_json)

    if cmd == "run":
        if len(rest) < 2:
            c.emit(c.style("warn", "  usage: ara run <model> <prompt>"))
            return 1
        return render_run(c, rest[1], prompt=" ".join(rest[2:]) or None,
                          engine=engine, assume_yes=assume_yes, as_json=as_json)

    if cmd == "install":
        return render_install(c, engine=engine or "auto", as_json=as_json)

    if cmd == "uninstall":
        return render_uninstall(c, engine=engine or "auto", as_json=as_json)

    if cmd == "hf":
        return render_hf(c, rest[1] if len(rest) > 1 else None, token=token, as_json=as_json)

    c.emit(c.style("warn", f"  '{rest[0]}' isn't built yet — ARA is an early scaffold."))
    c.emit(
        c.style("dim", "  run ") + c.style("accent", "ara")
        + c.style("dim", " with no arguments to see the planned commands.")
    )
    return 1
