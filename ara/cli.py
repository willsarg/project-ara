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

from ara import (acquire, apps, benchmark, catalog, db, detect, engines, estimate, hub,
                 hf_auth, locking, mlx, ollama, profile, calibration, pythons, scoring, serialize,
                 staleness, status, versions)
from ara.contracts import ramp
from ara.engines import _ara_version    # single source of truth (also stamps engine envs)
from ara.engine_env import EngineEnvError
from ara.locking import MeasurementBusy
from ara.registry import UnknownEngine, engine_status, get_backend, resolve_engine
from ara.ui import Console


def _hf_hint(c: Console, as_json: bool) -> None:
    """A one-line nudge toward `ara hf login` when a Hub op runs unauthenticated (higher rate limits
    + faster downloads). Skipped under --json (would corrupt the parse) and when already
    authenticated. Pairs with hf_auth.quiet_hub_warnings(), which mutes HF's own generic warning."""
    if not as_json and not hf_auth.has_token():
        c.emit(c.style("dim", "  tip: run ") + c.style("accent", "ara hf login")
               + c.style("dim", " for higher rate limits + faster downloads"))


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


def _resolve_want(cmd: str, include: list[str], exclude: list[str], c: Console, *,
                  as_json: bool = False):
    """Build the section predicate for *cmd*, normalizing aliases and warning on unknowns.
    Returns None when the command has no sections to filter. The advisory warnings are styled
    text, so they're suppressed under --json (Rule #3) — they'd corrupt the JSON parse; the
    filter still applies."""
    valid = _RECON_SECTIONS.get(cmd)
    if valid is None:
        if (include or exclude) and not as_json:
            c.emit(c.style("warn", f"  --include/--exclude don't apply to '{cmd}'"))
            c.emit()
        return None

    def norm(xs):
        return [_SECTION_ALIASES.get(x.lower().strip(), x.lower().strip()) for x in xs]

    inc, exc = norm(include), norm(exclude)
    unknown = [s for s in (*inc, *exc) if s not in valid]
    if unknown and not as_json:
        c.emit(c.style("warn", f"  unknown section(s) for {cmd}: {', '.join(dict.fromkeys(unknown))}"))
        c.emit(c.style("dim", f"  valid: {', '.join(valid)}"))
        c.emit()
    return _section_filter([s for s in inc if s in valid], [s for s in exc if s in valid])


# --------------------------------------------------------------------------- #
# help (per-subcommand usage for `ara <cmd> --help` / `-h`)
# --------------------------------------------------------------------------- #
_COMMAND_HELP = {
    "detect": "ara detect [--json] [--include S] [--exclude S] — read-only machine recon",
    "status": "ara status [--json] — AI/ML processes running right now",
    "python": "ara python [--json] — interpreters + their AI libraries",
    "apps": "ara apps [--json] — installed AI/ML apps",
    "mlx": "ara mlx [--json] — MLX ecosystem readiness",
    "search": "ara search <query> [--json] — find models on the Hugging Face Hub",
    "models": "ara models [<model-id>] [--json] — model catalog + safe ceilings",
    "characterize": ("ara characterize <model> [--engine E] [--kv-quant f16|q8_0|q4_0] "
                     "[--weight-quant none|int8|int4|fp8] [--chunked-prefill] [--json]"),
    "install": "ara install [<engine>] [--engine E] — install an engine (default: matched to this machine)",
    "uninstall": "ara uninstall [<engine>] [--engine E] — remove an engine",
    "profile": "ara profile [--model M] [--engine E] [--json] — analytic capability estimate",
    "recommend": "ara recommend [--json] — models that fit here, ranked by usable context",
    "run": ("ara run <model> <prompt> [--engine E] [--kv-quant ...] [--weight-quant ...] "
            "[--chunked-prefill] [--json]"),
    "serve": ("ara serve <model> [--ctx N] [--name X] [--yes] [--json] — stand a model up on "
              "Ollama, governed at a safe context ceiling, and return the OpenAI-compatible endpoint"),
    "benchmark": ("ara benchmark <model> --use-case <coding|reasoning|agentic|extraction|rag> "
                  "[--exec-consent] [--engine E] [--ctx N] [--max-tokens N] [--repeat N] [--yes] "
                  "[--json] — run a capability probe set and store the measured score "
                  "(--exec-consent is REQUIRED for the coding probe, which runs model-written code)"),
    "hf": "ara hf <login|logout|status> [--token T] [--json] — Hugging Face auth",
}


def render_help(c: Console, cmd: str | None, *, as_json: bool = False) -> int:
    """`ara <cmd> --help` — print *cmd*'s usage. No/unknown command → the full landing catalog."""
    usage = _COMMAND_HELP.get(cmd) if cmd else None
    if usage is None:
        render_landing(c)
        return 0
    if as_json:
        print(json.dumps({"command": cmd, "usage": usage}))
    else:
        c.emit(c.style("dim", f"  usage: {usage}"))
    return 0


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
    c.emit(_cmd(c, "serve <model>", "stand it up safely on Ollama + hand back the endpoint"))
    c.emit(_cmd(c, "benchmark <model>", "run a capability probe and store the measured score"))
    c.emit(_cmd(c, "node <sub>", "run ARA as a push-only daemon that phones home (enroll/run/install/…)"))
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


_ARA_ENGINE_BACKENDS = {"cuda", "mlx", "vulkan"}   # backends ARA can actually run today
_KV_QUANT_CHOICES = ("f16", "q8_0", "q4_0")        # vulkan KV-cache quant (symmetric K=V)
_WEIGHT_QUANT_CHOICES = ("none", "int8", "int4", "fp8")   # CUDA runtime weight quant (bitsandbytes/FP8)
_DEFAULT_PREFILL_CHUNK = 512   # chunk size a bare --chunked-prefill uses (cuda); tunable via --prefill-chunk
_RUNTIME_LABEL = {"vulkan": "Vulkan", "cuda": "CUDA", "mlx": "MLX", "rocm": "ROCm"}


def _int_or_none(s: str) -> int | None:
    """Parse a CLI integer value, or None if it's empty/not an int (so a bad value disables the
    lever rather than crashing — the engine-level reject/validation still applies downstream)."""
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def _gpu_line(c: Console, g) -> None:
    """Render one GpuInfo entry: a name·VRAM line, then a hint sub-line."""
    parts = [g.name or g.vendor.upper()]
    apu_gtt = g.gtt_gb is not None and g.integrated
    if g.vram_gb is not None:
        # On an APU the vram figure is a small carveout, not the usable pool — say so and show GTT.
        if apu_gtt:
            parts.append(f"{g.vram_gb:.0f} GB VRAM carveout")
        else:
            parts.append(f"{g.vram_gb:.0f} GB" + (" (shared)" if g.integrated else ""))
    if apu_gtt:
        parts.append(f"{g.gtt_gb:.0f} GB shared (GTT)")
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
    c.emit(c.section("  ENGINES") + c.style("dim", "  (third-party launchers found on this system)"))
    for rt in engines:
        if rt.present:
            val = f"{rt.name} {rt.version}" if rt.version else rt.name
            if rt.requires:  # installed, but can't accelerate on this hardware
                c.emit(c.field("·", val, f"installed · {rt.requires}", value_role="warn"))
            elif rt.serving is False:  # a server runtime that's installed but not up
                c.emit(c.field("·", val, "installed · not serving — run `ollama serve`",
                               value_role="warn"))
            elif rt.serving is True:
                c.emit(c.field("·", val, "serving", value_role="good"))
            else:
                c.emit(c.field("·", val, "found", value_role="good"))
        elif c.verbose:
            c.emit(c.field("·", rt.name, "not found", value_role="dim"))
    if not any(rt.present for rt in engines) and not c.verbose:
        # NOT a bare "none" — that read as contradicting the ARA section's own-engine readiness.
        c.emit(c.style("dim", "  none found — ARA runs models through its own engine (see ARA below)"))
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
    # auto_updates lookup (one batched brew call, scoped to the casks actually in the
    # scanned inventory — not every installed cask) lives here, in the dedicated command —
    # never in the detect summary. True = brew defers, so drift is expected, not a conflict.
    tokens = sorted({a.cask_token for a in inventory if a.cask_token})
    defers = versions.cask_auto_updates(tokens)
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
    procs, apps_ = status._scan_all()  # one process_iter pass feeds both sections

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


def _ctx_gate_msg(ctx: int, measured: int | None, model: str) -> str | None:
    """Rule #1 gate for an explicit ``--ctx``: never proceed past this machine's MEASURED safe
    ceiling. Returns the refusal message, or None when the request is admissible.

    A measured ceiling exists → ``--ctx`` must stay ≤ it; raising the ceiling goes through
    ``ara characterize`` (which re-measures safely) — not past the wall on a hunch. No measured
    ceiling → an explicit ``--ctx`` is allowed as before (explicit beats silent guess; there is
    no measurement to violate). Replaces the old proceed-with-a-note advisory (2026-06-28 audit
    follow-up). Slug 2026-07-02-rule1-ctx-gate."""
    if measured is not None and ctx > measured:
        return (f"--ctx {ctx} exceeds the measured safe ceiling {measured} for {model} on this "
                f"machine — refusing (Rule #1: never exceed the measured memory wall). Use "
                f"--ctx ≤ {measured}, or re-run `ara characterize {model}` if the hardware or "
                f"engine setup changed.")
    return None


def _stale_ceiling_note(c: Console, model: str, measured_at: str | None, *,
                        as_json: bool) -> bool:
    """Warn (Rule #3) when a stored ceiling predates the model's current cache files — the number
    was measured against a since-changed model. Advisory, never a block: the measured ceiling is
    still the best on record until ``ara characterize`` re-measures. Returns True when stale, so a
    ``--json`` caller can carry a ``stale_ceiling`` flag rather than print. Slug
    2026-07-02-ara-ceiling-staleness."""
    if not staleness.ceiling_is_stale(model, measured_at):
        return False
    if not as_json:
        c.emit(c.style("warn", f"  ⚠ measured ceiling may be stale — {model}'s cache files changed "
                               f"since it was characterized ({measured_at}); re-run: "
                               f"ara characterize {model}"))
    return True


def _measured_ramp_slope(row: dict | None) -> float | None:
    """Fit the measured growth slope (GB per 1k tok) from a characterization row's stored ramp
    points, or None when it can't (missing/too-few points, degenerate fit). The points are in the
    engine's native units (wmx: decimal GB) — the SAME units the wmx serve gate predicts in — so
    the slope passes straight through with no conversion. Lets ``serve`` gate a measured ceiling
    with the real slope instead of the conservative a-priori one; None falls back to a-priori.
    Slug 2026-07-02-wmx-serve-measured-provenance-gate."""
    pts = [(p["context"], p["mem_gb"]) for p in (row or {}).get("points", [])
           if p.get("context") is not None and p.get("mem_gb") is not None]
    if len(pts) < 2:
        return None
    try:
        return ramp.fit(pts).slope_gb_per_k
    except ramp.RampError:
        return None


def render_model_detail(c: Console, model_id: str, *, as_json: bool = False) -> int:
    """Detail for one model: architecture (from its HF config) + its safe ceiling here."""
    meta = catalog.describe(model_id)
    if meta is None:
        if as_json:
            print(json.dumps({"error": f"couldn't describe {model_id}"}))
        else:
            c.emit(c.style("warn", f"  couldn't describe {model_id} — is it downloaded / a valid repo?"))
        return 1
    mk = profile.machine_key()
    # Per-engine: a model can be characterized under several engines on one machine (GPU + CPU).
    per_engine = {}                       # engine_key -> (safe_context, decode_context, measured_at)
    with db.connected() as con:
        for key in engines.ENGINES:
            row = db.get_characterization(con, mk, key, model_id)
            if row is not None:
                per_engine[key] = (row["safe_context"], row.get("decode_context"),
                                   row.get("measured_at"))
    # Best (largest) ceiling, carrying its decode_context AND measured_at so the top-level scalars
    # and the staleness flag all describe the SAME engine — not independent max() picks.
    best_triple = max(((sc, dc, at) for (sc, dc, at) in per_engine.values() if sc is not None),
                      key=lambda t: t[0], default=None)
    best = best_triple[0] if best_triple else None
    best_decode = best_triple[1] if best_triple else None
    # Rule #3: a stored ceiling whose cache changed since it was measured isn't authoritative —
    # flag it here just as serve/run do, so no command shows a stale number unqualified.
    best_stale = best_triple is not None and staleness.ceiling_is_stale(model_id, best_triple[2])
    if as_json:
        print(json.dumps({"model_id": model_id, **meta, "safe_context": best,
                          "decode_context": best_decode, "stale_ceiling": best_stale,
                          "engines": {k: sc for k, (sc, _, _) in per_engine.items()},
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
        for key, (sc, dc, at) in per_engine.items():
            ceiling_str = f"~{sc} tokens" if sc else "no safe ceiling"
            if sc and dc and dc > sc:
                ceiling_str += f"  · ~{dc} stream-only (est.)"
            if sc and staleness.ceiling_is_stale(model_id, at):
                ceiling_str += "  · ⚠ stale — re-characterize"
            c.emit(c.field(f"{key} ceiling", ceiling_str))
    else:
        c.emit(c.field("ceiling", "not characterized"))
    c.emit()
    return 0


def _kv_quant_error(kv_quant: str) -> str:
    return (f"invalid --kv-quant {kv_quant!r} — choose one of: {', '.join(_KV_QUANT_CHOICES)}")


# Which context levers each engine actually honors at the ARA level. Rule #3: a lever the engine
# can't honor is REJECTED with a clear message, never silently dropped. Mirrors _kv_fa_kwargs.
_ENGINE_LEVERS = {
    "vulkan": {"kv_quant", "flash_attn"},
    # NVIDIA-native runtime weight quant + chunked prefill (long-context unlock on Turing/no-FA cards)
    "cuda": {"kv_quant", "flash_attn", "weight_quant", "prefill_chunk"},
    "apple": {"kv_quant"},        # MLX's SDPA is always fused — no flash-attn knob
    "cpu": set(),                  # llama.cpp cache-type / flash aren't wired at the ARA level yet
}


def _unsupported_lever_error(backend: str, *, kv_quant: str, flash_attn: bool,
                             flash_attn_optin: bool, weight_quant: str = "none",
                             prefill_chunk: int | None = None) -> str | None:
    """A clear message when the user EXPLICITLY set a context lever the selected engine can't honor
    — else None. Honesty (Rule #3): reject rather than silently ignore the flag. A flash flag is
    'explicit' when --no-flash-attn turned the default off, or --flash-attn opted in."""
    levers = _ENGINE_LEVERS.get(backend, set())
    if kv_quant != "f16" and "kv_quant" not in levers:
        return f"--kv-quant isn't supported on the {backend} engine (it runs an fp16 KV cache)"
    if ((not flash_attn) or flash_attn_optin) and "flash_attn" not in levers:
        return f"flash-attention isn't a tunable setting on the {backend} engine"
    if weight_quant != "none" and "weight_quant" not in levers:
        return f"--weight-quant is only supported on the cuda engine (not {backend})"
    if prefill_chunk is not None and "prefill_chunk" not in levers:
        return f"chunked prefill is only supported on the cuda engine (not {backend})"
    return None


def _kv_fa_kwargs(backend: str, *, flash_attn: bool, flash_attn_optin: bool,
                 kv_quant: str, weight_quant: str = "none",
                 prefill_chunk: int | None = None) -> dict:
    """The context-lever kwargs each backend's characterize/generate accepts. KV-quant is a lever
    on the AMD iGPU (vulkan), Apple (wmx), and NVIDIA (wcx/cuda) lanes. Flash-attention has
    OPPOSITE defaults per engine: vulkan's llama.cpp FA is on-by-default (``flash_attn``, the
    ``--no-flash-attn`` opt-out), while CUDA defaults to SDPA with FA2 an availability-gated opt-in
    (``flash_attn_optin``, the ``--flash-attn`` flag). MLX's SDPA is always fused (no knob). Runtime
    weight-quant is CUDA-only (the others ship pre-quantized files)."""
    if backend == "vulkan":
        return {"flash_attn": flash_attn, "kv_quant": kv_quant}
    if backend == "cuda":
        return {"kv_quant": kv_quant, "flash_attn": flash_attn_optin, "weight_quant": weight_quant,
                "prefill_chunk": prefill_chunk}
    if backend == "apple":
        return {"kv_quant": kv_quant}
    return {}


def _flash_sdpa_note(c: Console, bk, backend: str, flash_attn_optin: bool,
                     as_json: bool) -> None:
    """Honesty (Rule #3): when the user opts into --flash-attn but this GPU can't run FA2, say so
    — the run silently uses SDPA otherwise. Only the CUDA backend exposes the capability check.
    Skipped under --json (a styled line would corrupt the parse)."""
    if (not as_json and flash_attn_optin and backend == "cuda"
            and hasattr(bk, "flash_attn_capable") and not bk.flash_attn_capable()):
        c.emit(c.style("dim", "  flash-attn (FA2) needs an Ampere+ GPU — using SDPA"))


def _weight_quant_hw_error(bk, backend: str, weight_quant: str) -> str | None:
    """FP8 weights need Ada/Hopper (sm_89+); reject upfront on older CUDA GPUs (Rule #3) rather
    than failing deep in the model load. Only CUDA reaches here (weight-quant is CUDA-only)."""
    if (weight_quant == "fp8" and backend == "cuda"
            and hasattr(bk, "fp8_capable") and not bk.fp8_capable()):
        return "--weight-quant fp8 needs an Ada/Hopper GPU (sm_89+) — this GPU can't run FP8"
    return None


def _prefetch_weights(c: Console, model: str, bk, engine_key: str | None,
                      *, as_json: bool, progress: bool) -> int | None:
    """Ensure a transformers/MLX model's weights are in the HF cache before the engine runs.

    So wcx/wmx fetch on demand like the GGUF engines (which download in-worker), instead of the
    worker refusing an uncached model (#109). Without it the worker's ``blobs/`` scan also yields
    ``weights_gb≈0`` for uncached transformers models, under-predicting the a-priori memory gate.
    No-op when the model's engine doesn't match *engine_key* or it's already cached — cpu/vulkan/
    cuda-gguf report cached (they acquire the GGUF in-worker), so this only fetches for apple/cuda.
    Returns 1 (after printing) on a disk-space or fetch error, else None.
    """
    incompatible = engines.engine_for_model(model) not in (None, engine_key)
    cached = getattr(bk, "calibration_model_cached", None)
    if incompatible or cached is None or cached(model):
        return None
    size_gb = acquire.repo_size_gb(model)
    free_gb = acquire.free_disk_gb()
    if size_gb and free_gb is not None and free_gb < size_gb + acquire.DISK_BUFFER_GB:
        msg = (f"not enough disk for {model}: needs ~{size_gb:.1f} GB + "
               f"{acquire.DISK_BUFFER_GB:.0f} GB headroom, only {free_gb:.1f} GB free.")
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1
    _hf_hint(c, as_json)        # nudge to `ara hf login` before the (visible) HF rate-limit warning
    c.emit(c.style("dim", f"  downloading {model} … ({_fmt_size(size_gb)})"))
    try:
        bk.download_calibration_model(model, progress=progress)
    except Exception as exc:
        msg = _fetch_error_msg(model, acquire.classify_repo_error(exc))
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1
    return None


def render_characterize(c: Console, model: str, *, engine: str | None = None,
                        as_json: bool = False, flash_attn: bool = True,
                        flash_attn_optin: bool = False, kv_quant: str = "f16",
                        weight_quant: str = "none", prefill_chunk: int | None = None) -> int:
    """Measure a model's safe context ceiling on an engine, and store it.

    Defaults to the detected engine; ``--engine`` overrides it so you can target a non-detected
    backend (e.g. the CPU fallback on a GPU box). ARA owns the result, so it shows up in
    `ara models` regardless of which engine measured it.

    ``--engine ollama`` routes to a dedicated residency-ramp path (Slice 2): Ollama isn't a registry
    engine, and its model names (``qwen3:0.6b``) aren't HF refs — so it branches before both
    ``resolve_engine`` and ``valid_model_ref``. Spec 2026-07-04-characterize-through-ollama-ramp."""
    if engine == "ollama":
        return _render_characterize_ollama(c, model, as_json=as_json)
    try:
        sel = resolve_engine(engine)
    except UnknownEngine:
        msg = f"unknown engine {engine!r} — try one of: {', '.join(engines.ENGINES)}"
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1
    if not acquire.valid_model_ref(model):
        msg = (f"invalid model {model!r} — expected a Hugging Face repo id (org/name) "
               f"or a local .gguf file path")
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1
    if kv_quant not in _KV_QUANT_CHOICES:
        msg = _kv_quant_error(kv_quant)
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1
    if weight_quant not in _WEIGHT_QUANT_CHOICES:
        msg = f"invalid --weight-quant {weight_quant!r} — choose one of: {', '.join(_WEIGHT_QUANT_CHOICES)}"
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1
    lever_err = _unsupported_lever_error(sel.backend, kv_quant=kv_quant, flash_attn=flash_attn,
                                         flash_attn_optin=flash_attn_optin, weight_quant=weight_quant,
                                         prefill_chunk=prefill_chunk)
    if lever_err is not None:
        print(json.dumps({"error": lever_err})) if as_json else c.emit(c.style("bad", f"  {lever_err}"))
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
    hw_err = _weight_quant_hw_error(bk, sel.backend, weight_quant)
    if hw_err is not None:
        print(json.dumps({"error": hw_err})) if as_json else c.emit(c.style("bad", f"  {hw_err}"))
        return 1
    progress = (not as_json) and sys.stderr.isatty()
    # Pre-fetch weights into the HF cache before the engine's preflight runs (#109).
    if (rc := _prefetch_weights(c, model, bk, sel.engine_key,
                                as_json=as_json, progress=progress)) is not None:
        return rc
    # characterize owns calibration: measure + persist the engine baseline once (when none is
    # stored) so the ramp uses the real overhead, not the default. Spec 2026-06-23-capability-pipeline.
    with db.connected() as cal_con:
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
    fa_kw = _kv_fa_kwargs(sel.backend, flash_attn=flash_attn, flash_attn_optin=flash_attn_optin,
                          kv_quant=kv_quant, weight_quant=weight_quant, prefill_chunk=prefill_chunk)
    _flash_sdpa_note(c, bk, sel.backend, flash_attn_optin, as_json)
    try:
        result = bk.characterize(model, progress=progress, **fa_kw)
    except (SystemExit, Exception) as exc:   # engine may refuse/abort/OOM-guard
        msg = f"characterization failed: {exc}"
        # Rule #3 (Honesty): under --json a consumer parses stdout — emit a structured error, never
        # styled text or a traceback that would break the parse.
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
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
    with db.connected() as con:
        db.save_characterization(con, profile.machine_key(), sel.engine_key,
                                 model, safe_context=ceiling, points=result["points"],
                                 decode_context=result.get("decode_context"))
        catalog.remember(con, model)

    if as_json:
        out: dict = {"model": model, "safe_context": ceiling,
                     "decode_context": result.get("decode_context")}
        if ceiling is None:
            # Carry through the diagnostic fields the driver surfaced so automated callers
            # can explain why — not just a bare null.
            for k in ("stopped_reason", "base_gb", "budget_gb"):
                if result.get(k) is not None:
                    out[k] = result[k]
        print(json.dumps(out, indent=2))
        return 0
    if ceiling:
        c.emit(c.style("good", f"  safe context ceiling  ~{ceiling} tokens")
               + c.style("dim", "  · stored (see ara models)"))
        dc = result.get("decode_context")
        if dc and dc > ceiling:
            c.emit(c.style("good", f"  decode ceiling (est.)  ~{dc} tokens")
                   + c.style("dim", "  · grow-by-streaming, not a prompt size"))
    else:
        base = result.get("base_gb")
        budget = result.get("budget_gb")
        if base is not None and budget is not None:
            c.emit(c.style("warn",
                           f"  couldn't fit a ceiling — estimated base {base:.2f} GiB"
                           f" already near {budget:.1f} GiB safe budget"))
        else:
            c.emit(c.style("warn", "  couldn't fit a ceiling — the model may be too big or borderline"))
        c.emit(c.style("dim", "  try: --weight-quant int4 or int8, or a smaller/quantized model"))
    c.emit()
    return 0


def render_search(c: Console, query: str, *, as_json: bool = False) -> int:
    """Search the Hugging Face Hub for models (engine-agnostic)."""
    results = hub.search(query)
    if results is None:
        if as_json:
            print(json.dumps({"error": "couldn't search — is the hf CLI installed? "
                                       "(pip install huggingface_hub)"}))
        else:
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
    _hf_hint(c, as_json)
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
    with db.connected() as con:
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


def render_recommend(c: Console, *, as_json: bool = False, use_case: str | None = None) -> int:
    """Analytic recommendations — which cataloged models fit this machine, ranked by the context
    the estimated budget supports (most first), marking those already characterized here.

    Engine-free: reuses ``estimate.limits``/``model_fit`` (profile's math, anti-silo) over the
    catalog (which records each model's on-disk weight). No engine, no model load. Only models
    with a rankable context estimate are listed. With ``use_case``, ranks by a capability score —
    *measured here* (a local benchmark on the actual quant) or *imported* (a published number,
    labelled), never a guess (Rule #3). Stays engine-free either way. Spec
    2026-06-23-capability-pipeline + 2026-06-28-recommend-use-case-and-serve-selection."""
    if use_case is not None and use_case not in scoring.USE_CASES:
        msg = (f"unknown use-case {use_case!r} — choose one of: "
               f"{', '.join(scoring.USE_CASES)}")
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1
    with db.connected() as con:
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
            quant = row.get("quant") or scoring.quant_key(row["model_id"])
            recs.append({"model_id": row["model_id"], "modality": row.get("modality"),
                         "est_context": fit["est_context"], "max_context": fit["max_context"],
                         "binding": fit["binding"], "fits": True,
                         "characterized": row["model_id"] in best,
                         "quant": quant, "quant_bits": scoring.quant_bits(quant),
                         "base": scoring.base_key(row["model_id"])})
        if use_case is not None:
            rows = db.list_benchmark_results(con, profile.machine_key())
            bench_measured = ({(r["model_id"], r["use_case"]):
                               {"score": r["score"], "source": r["source"],
                                "sample_size": r.get("sample_size"),
                                "refused_n": r.get("refused_n"), "errored_n": r.get("errored_n")}
                               for r in rows} or None)
            recs = scoring.rank(recs, use_case, measured=bench_measured,
                                imported=scoring.load_imported())
        else:
            recs.sort(key=lambda r: r["est_context"], reverse=True)

    if as_json:
        if use_case is not None:
            recs = [{**r, "score": (None if r["score"] is None else
                                    {"tier": r["score"].tier, "value": r["score"].value,
                                     "source": r["score"].source,
                                     "sample_size": r["score"].sample_size,
                                     "refused_n": r["score"].refused_n,
                                     "errored_n": r["score"].errored_n,
                                     "inversion": r["score"].inversion})} for r in recs]
        print(json.dumps(recs, indent=2))
        return 0

    def _unrankable_note() -> None:
        if unrankable:
            c.emit(c.style("dim", f"  {unrankable} more fit but can't be ranked "
                                   "(architecture unknown) — try ara profile --model <model>"))

    c.emit()
    sub = ("  (estimated — fits this machine, most context first)" if use_case is None
           else f"  (for {use_case} — capability-ranked; measured-here or imported)")
    c.emit(c.section("  RECOMMENDED MODELS") + c.style("dim", sub))
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
        if use_case is not None:
            s = r.get("score")
            if s is None:
                head = "unknown (not measured or imported)"
            else:
                head = f"{use_case} {s.value * 100:.0f}% ({s.tier})"
                if s.tier == "measured":
                    # Disclose a depressed / shaky measurement rather than ranking on it silently.
                    if s.refused_n or s.errored_n:
                        partial = []
                        if s.refused_n:
                            partial.append(f"{s.refused_n} refused")
                        if s.errored_n:
                            partial.append(f"{s.errored_n} errored")
                        head += f" [partial: {', '.join(partial)}]"
                    if s.sample_size is not None and s.sample_size < 100:
                        head += f" [low-confidence n={s.sample_size}]"
                    if s.inversion:
                        head += f" [quant-inversion: {s.inversion}]"
            tail = f"{head} · {tail}"
        mark = c.style("good", "  · characterized here") if r["characterized"] else ""
        quant_tag = c.style("dim", f" [{r['quant']}]") if r["quant"] else ""
        c.emit("  " + c.style("metric", r["model_id"]) + quant_tag
               + c.style("dim", f"  {r['modality'] or '?'}  →  ")
               + c.style("accent", tail) + mark)
    n_char = sum(1 for r in recs if r["characterized"])
    c.emit()
    c.emit(c.style("dim", f"  {len(recs)} fit · {n_char} characterized here"))
    by_base: dict[str, dict[str, dict]] = {}
    for r in recs:
        if r["quant"] is None:
            continue
        by_base.setdefault(r["base"], {})[r["quant"]] = r
    tradeoffs = {base: variants for base, variants in by_base.items() if len(variants) > 1}
    if tradeoffs:
        c.emit(c.style("dim", "  tradeoff — same base at multiple quants (fewer bits → more "
                              "context; more bits → closer to the original weights):"))
        for base, variants in tradeoffs.items():
            # bits desc, but a quant whose bit-width we can't map (quant_bits None) sorts last
            # rather than crashing the comparison (Rule #2 — recommend never blows up).
            ordered = sorted(variants.values(),
                             key=lambda r: (r["quant_bits"] is None, -(r["quant_bits"] or 0.0)))
            parts = ", ".join(f"{r['quant']}(~{r['est_context']} tok)" for r in ordered)
            c.emit(c.style("dim", f"    {base}: {parts}"))
    _unrankable_note()
    c.emit()
    return 0


def render_benchmark(c: Console, model: str, *, use_case: str, engine: str | None = None,
                     ctx: int | None = None, max_tokens: int | None = None,
                     repeat: int = 1, assume_yes: bool = False,
                     exec_consent: bool = False, as_json: bool = False) -> int:
    """Run a capability probe set against *model* and store the score as a measured tier result.

    Requires a characterization ceiling (or explicit ``--ctx``) and an engine backend that supports
    ``benchmark`` (Apple/MLX, CPU, Vulkan, and CUDA). Spec 2026-06-28-recommend-use-case-and-serve-selection."""
    def err(msg: str) -> int:
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1

    if use_case not in benchmark.USE_CASES:
        return err(f"unknown use-case {use_case!r} — choose one of: "
                   f"{', '.join(benchmark.USE_CASES)}")
    if max_tokens is not None and max_tokens <= 0:
        return err("--max-tokens must be a positive integer")
    if repeat < 1:
        return err("--repeat must be a positive integer")

    # Hard gate on code execution — un-bypassable by --json/--yes/non-tty (those only suppress the
    # interactive prompt). The coding benchmark runs model-generated Python with full user
    # privileges (NOT a security sandbox); require deliberate, explicit consent in every mode.
    if use_case == "coding" and not exec_consent:
        return err("the coding benchmark executes model-generated Python on this machine "
                   "(NOT a security sandbox) — re-run with --exec-consent to allow it")

    key = engines.resolve(engine) if engine else engines.for_hardware()
    backend = engines.ENGINES.get(key, {}).get("backend") if key else None
    bk = get_backend(backend) if backend else None
    if bk is None or not hasattr(bk, "benchmark"):
        be_name = backend or "none"
        return err(f"benchmark isn't supported on the {be_name} engine")

    mk = profile.machine_key()
    with db.connected() as con:
        if ctx is not None:
            if ctx <= 0:
                return err("--ctx must be a positive integer")
            _row = db.get_characterization(con, mk, key, model)
            if (msg := _ctx_gate_msg(ctx, _row.get("safe_context") if _row else None, model)):
                return err(msg)
            safe = ctx
            ceiling_measured_at = None       # explicit --ctx, not a stored ceiling — nothing to age
        else:
            row = db.get_characterization(con, mk, key, model)  # keyed by ENGINE KEY, not backend
            if not row or row.get("safe_context") is None:
                return err(f"no measured ceiling for {model} — run: ara characterize {model} "
                           f"(or pass --ctx N)")
            safe = row["safe_context"]
            ceiling_measured_at = row.get("measured_at")

    stale_ceiling = _stale_ceiling_note(c, model, ceiling_measured_at, as_json=as_json)
    items = benchmark.load_probe(use_case)
    n = len(items)
    if not as_json and not assume_yes and sys.stdin.isatty():
        if use_case == "coding":
            c.emit(c.style("warn", "  warning: the coding benchmark EXECUTES model-generated "
                                   "Python in a subprocess (NOT a security sandbox)"))
        scope = f"{n} prompts" if repeat == 1 else f"{n} prompts × {repeat} runs"
        if not _confirm(f"Benchmark {model} on {use_case} ({scope})? "
                        f"loads the model at ≤{safe} ctx"):
            c.emit(c.style("dim", "  skipped."))
            return 0

    prompts = [benchmark.prompt_for(use_case, it) for it in items]
    # Default max_tokens is the backend's own (256); --max-tokens lifts it so thinking models
    # aren't truncated mid-reasoning (the campaign sets ≥512). Omit the kwarg when unset.
    bench_kw = {} if max_tokens is None else {"max_tokens": max_tokens}
    # Pre-fetch weights so wcx/wmx benchmark uncached models on demand (like the GGUF engines),
    # instead of the worker refusing "model not found in HF cache" (#109).
    progress = (not as_json) and sys.stderr.isatty()
    if (rc := _prefetch_weights(c, model, bk, key, as_json=as_json, progress=progress)) is not None:
        return rc
    # --repeat N: run the probe set N times (N separate model loads — acceptable v1). Never let a
    # single lucky roll stand in as THE number: score each run independently, store the MEAN as the
    # point estimate, and surface the LO–HI band so a wide spread is visible (pass^k spirit).
    run_scores: list[float] = []
    refused_n = 0
    errored_n = 0
    for _ in range(repeat):
        result = bk.benchmark(model, prompts, max_context=safe, **bench_kw)
        if result.get("refused"):
            # A whole-run refusal on ANY run aborts — no partial band scraped from a failed load.
            return err(f"the engine refused: {result.get('reason', 'no reason given')}")
        results = result.get("results", [])
        refused_n += sum(1 for r in results if r.get("refused"))
        errored_n += sum(1 for r in results if r.get("error"))
        completions = [""] * len(prompts)
        for r in results:
            idx = r.get("prompt_index")
            if isinstance(idx, int) and 0 <= idx < len(completions):
                completions[idx] = r.get("completion", "")
        run_scores.append(benchmark.score_probe_set(use_case, items, completions))

    total = n * repeat                       # total generations attempted across every run
    if prompts and (refused_n + errored_n) == total:
        # No generation anywhere produced a completion (all refused by governance and/or errored
        # mid-generation) — NOT a 0% capability measurement; refuse to store a misleading score.
        return err("every prompt was refused or errored — no measurement taken")
    if refused_n:
        c.emit(c.style("warn", f"  note: {refused_n}/{total} prompts were refused by "
                               f"governance and scored 0 — the result is depressed accordingly"))
    if errored_n:
        c.emit(c.style("warn", f"  note: {errored_n}/{total} prompts errored (engine "
                               f"exception) and scored 0 — the result is depressed accordingly"))

    score = sum(run_scores) / repeat         # MEAN across runs — a better estimate than any one roll
    lo, hi = min(run_scores), max(run_scores)
    low_confidence = n < 100
    source = f"{key} probe={n} ({model})"
    if low_confidence:
        source += f"; low_confidence n={n}"
    if repeat > 1:
        source += f"; repeat={repeat} band={lo * 100:.0f}-{hi * 100:.0f}"
    # Record the quant the score was actually taken at (the quant×capability degradation an
    # imported score hides): prefer the catalog's recorded quant, else derive it from the id.
    with db.connected() as con:
        mrow = db.get_model(con, model)
        quant = mrow.get("quant") if mrow else None
        quant = quant or scoring.quant_key(model)
        db.save_benchmark_result(con, mk, model, use_case, score=score, source=source,
                                 engine_key=key, backend=backend,
                                 base_model=scoring.base_key(model), quant=quant,
                                 benchmark_id=use_case, sample_size=n,
                                 refused_n=refused_n, errored_n=errored_n)
        con.commit()

    if as_json:
        payload: dict = {"model": model, "use_case": use_case, "score": score,
                         "sample_size": n, "engine": key, "stored": True}
        if stale_ceiling:
            payload["stale_ceiling"] = True
        if repeat > 1:
            payload["runs"] = run_scores
            payload["band"] = [lo, hi]
            payload["repeat"] = repeat
        if low_confidence:
            payload["low_confidence"] = True
        if refused_n or errored_n:
            payload["refused"] = refused_n
            payload["errored"] = errored_n
        if quant:
            payload["quant"] = quant
        print(json.dumps(payload))
        return 0
    if repeat > 1:
        score_line = (f"  {use_case}: {score * 100:.0f}% measured here  "
                      f"(mean of {repeat} runs, band {lo * 100:.0f}–{hi * 100:.0f}%, "
                      f"{n} prompts, {model})")
    else:
        score_line = (f"  {use_case}: {score * 100:.0f}% measured here  ({n} prompts, {model})")
    if low_confidence:
        score_line += f" (low-confidence: n={n})"
    if refused_n or errored_n:
        partial = []
        if refused_n:
            partial.append(f"{refused_n} refused")
        if errored_n:
            partial.append(f"{errored_n} errored")
        score_line += f" (partial: {', '.join(partial)})"
    c.emit(c.style("good", score_line))
    c.emit(c.style("dim", f"  stored — ara recommend --use-case {use_case} now shows it"))
    if repeat > 1 and lo == hi:
        # Zero variance under greedy decoding is determinism, not measured robustness — say so
        # honestly rather than let an identical-across-runs band read as evidence of stability.
        c.emit(c.style("warn", f"  note: all {repeat} runs scored identically — decoding is "
                               f"deterministic on this engine; the band is not evidence of "
                               f"stability"))
    if score == 0.0 or score == 1.0:
        c.emit(c.style("warn", "  note: a flat 0%/100% often means a broken probe or "
                               "misconfig — verify before trusting"))
    c.emit()
    return 0


RUN_MAX_TOKENS = 256


def render_run(c: Console, model: str, *, prompt: str | None = None, engine: str | None = None,
               assume_yes: bool = False, as_json: bool = False,
               max_tokens: int = RUN_MAX_TOKENS, flash_attn: bool = True,
               flash_attn_optin: bool = False, kv_quant: str = "f16",
               weight_quant: str = "none", prefill_chunk: int | None = None) -> int:
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
    if not acquire.valid_model_ref(model):
        return err(f"invalid model {model!r} — expected a Hugging Face repo id (org/name) "
                   f"or a local .gguf file path")
    if kv_quant not in _KV_QUANT_CHOICES:
        return err(_kv_quant_error(kv_quant))
    if weight_quant not in _WEIGHT_QUANT_CHOICES:
        return err(f"invalid --weight-quant {weight_quant!r} — choose one of: "
                   f"{', '.join(_WEIGHT_QUANT_CHOICES)}")

    mk = profile.machine_key()
    suffix = "" if engine is None else f" --engine {sel.engine_key}"

    with db.connected() as con:
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
            ceiling_measured_at = row.get("measured_at")
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
                                   hasattr(get_backend(backend), "generate"),
                                   row.get("measured_at"))
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
            safe, backend, _, ceiling_measured_at = runnable[engine_key]

    stale_ceiling = _stale_ceiling_note(c, model, ceiling_measured_at, as_json=as_json)
    lever_err = _unsupported_lever_error(backend, kv_quant=kv_quant, flash_attn=flash_attn,
                                         flash_attn_optin=flash_attn_optin, weight_quant=weight_quant,
                                         prefill_chunk=prefill_chunk)
    if lever_err is not None:
        return err(lever_err)

    engine_ok, engine_pkg = engine_status(backend)
    if not engine_ok:
        return err(f"the {engine_pkg} engine isn't installed — run: ara install{suffix}")
    bk = get_backend(backend)
    if not hasattr(bk, "generate"):
        return err(f"run isn't supported on the {engine_pkg} engine yet")
    hw_err = _weight_quant_hw_error(bk, backend, weight_quant)
    if hw_err is not None:
        return err(hw_err)

    # Consent before load (a courtesy — the ceiling already makes it wall-safe). Interactive only;
    # --yes or a non-tty (scripts/--json) proceed straight to the governed run.
    if not as_json and not assume_yes and sys.stdin.isatty():
        if not _confirm(f"Load {model} on {engine_pkg} and generate (≤ ~{safe} tokens)?"):
            c.emit(c.style("dim", "  skipped."))
            return 0

    if not as_json:
        c.emit(c.style("dim", f"  running {model} on {engine_pkg} … (≤ ~{safe} tokens)"))
    fa_kw = _kv_fa_kwargs(backend, flash_attn=flash_attn, flash_attn_optin=flash_attn_optin,
                          kv_quant=kv_quant, weight_quant=weight_quant, prefill_chunk=prefill_chunk)
    _flash_sdpa_note(c, bk, backend, flash_attn_optin, as_json)
    try:
        result = bk.generate(model, prompt, max_context=safe, max_tokens=max_tokens, **fa_kw)
    except (SystemExit, Exception) as exc:        # engine may refuse/abort/OOM-guard
        return err(f"run failed: {exc}")
    if result.get("refused"):
        return err(f"the {engine_pkg} engine refused: {result.get('reason', 'no reason given')}")

    completion = result.get("completion", "")
    if as_json:
        print(json.dumps({"model": model, "engine": engine_key,
                          "safe_context": safe, "stale_ceiling": stale_ceiling,
                          "completion": completion}, indent=2))
        return 0
    c.emit()
    c.emit(completion)
    c.emit()
    return 0


# --------------------------------------------------------------------------- #
# serve — stand a model up as a governed OpenAI-compatible endpoint on Ollama
# (Decision 2026-06-26; spec 2026-06-26-ara-serve-governed-endpoint)
# --------------------------------------------------------------------------- #
# Ollama runs llama.cpp under the hood, so a safe ceiling measured on the GGUF/llama.cpp-class
# engines transfers; the MLX (wmx) ceiling does NOT (different allocation model — the seam mismatch).
_OLLAMA_CEILING_ENGINES = ("ollama", "cpu", "vulkan", "cuda-gguf")


def _ollama_safe_ceiling(con, mk: str, model: str):
    """The largest *measured* llama.cpp-class safe ceiling for *model* on this machine, as
    ``(safe_context, "measured", measured_at)``, or ``None`` if none is recorded. ``measured_at``
    lets the caller flag a stale ceiling (cache changed since it was measured)."""
    best = None
    for key in _OLLAMA_CEILING_ENGINES:
        row = db.get_characterization(con, mk, key, model)
        if (row and row.get("safe_context") is not None
                and (best is None or row["safe_context"] > best[0])):
            best = (row["safe_context"], "measured", row.get("measured_at"))
    return best


def _ollama_estimated_ceiling(model: str):
    """The engine-free *estimated* safe ceiling for *model*, as ``(est_context, "estimated", None)``,
    or ``None`` when the architecture can't be read or the model doesn't fit.

    Reads the model's architecture from Ollama's own ``/api/show`` (llama.cpp-class, local, no
    network — the honest source for an Ollama-native model HF can't describe) and runs it through
    the analytic estimator (``ara profile``'s math). The result is conservative (wall − margin) and
    labelled ``estimated`` — never reported as measured (Rule #3). This is the fallback that lets
    ``ara serve`` stand a not-yet-characterized model up safely in one command; ``ara characterize``
    tightens it to ``measured`` later. Spec 2026-07-04-ara-serve-one-command-estimated-ceiling."""
    detail = ollama.show(model)
    if not detail:
        return None
    info = detail.get("model_info") or {}
    arch = info.get("general.architecture")
    if not isinstance(arch, str):
        return None
    meta = {
        "n_layers": info.get(f"{arch}.block_count"),
        "kv_heads": info.get(f"{arch}.attention.head_count_kv"),
        "head_dim": info.get(f"{arch}.attention.key_length"),
        "max_context": info.get(f"{arch}.context_length"),
    }
    size = ollama.size_bytes(model)
    weights_gb = size / 1e9 if size else None            # decimal GB — the estimator's unit
    lim = estimate.limits(detect.machine())              # heuristic wall: an estimate stays an estimate
    fit = estimate.model_fit(lim, meta, weights_gb)
    ec = fit.get("est_context")
    return (ec, "estimated", None) if ec else None


def _ollama_max_context(model: str) -> int | None:
    """The model's advertised max context from Ollama's own ``/api/show`` architecture metadata, or
    ``None`` when it can't be read — the hard upper bound for the characterization ramp."""
    detail = ollama.show(model)
    if not detail:
        return None
    info = detail.get("model_info") or {}
    arch = info.get("general.architecture")
    if not isinstance(arch, str):
        return None
    mc = info.get(f"{arch}.context_length")
    return mc if isinstance(mc, int) and mc > 0 else None


def _ollama_ramp_contexts(max_ctx: int) -> list[int]:
    """Ascending probe rungs for the residency ramp: 2048 doubling up to *max_ctx*, plus *max_ctx*
    itself (deduped, sorted). A *max_ctx* below the 2048 floor yields just ``[max_ctx]``."""
    rungs = set()
    n = 2048
    while n < max_ctx:
        rungs.add(n)
        n *= 2
    rungs.add(max_ctx)
    return sorted(rungs)


def _ollama_measure_ceiling(model: str, max_ctx: int, probe: str):
    """Ramp Ollama residency to the largest context *model* loads with NO spill. For each rung
    (ascending), bake a *probe* derived model at that ctx, load it, and read ``/api/ps``: it counts
    only when governance took (``context_length`` == ctx) AND it's fully resident
    (``size_vram >= size``). KV grows monotonically, so the first spill/failure ends the ramp.
    Returns ``(best_ctx | None, points)``. Spec 2026-07-04-characterize-through-ollama-ramp."""
    best, points = None, []
    for ctx in _ollama_ramp_contexts(max_ctx):
        if not ollama.create(probe, model, ctx):     # couldn't bake this rung — stop, keep what fit
            break
        ollama.load(probe)
        entry = _find_loaded(ollama.ps() or [], probe)
        if entry is None or entry.get("context_length") != ctx:   # didn't load / governance slipped
            points.append({"context": ctx, "fit": False})
            break
        size, vram = entry.get("size"), entry.get("size_vram")
        spilled = isinstance(size, int) and isinstance(vram, int) and vram < size
        points.append({"context": ctx, "fit": not spilled, "size": size, "size_vram": vram})
        if spilled:                                   # hit the wall — a higher rung can't recover
            break
        best = ctx
    return best, points


def _render_characterize_ollama(c: Console, model: str, *, as_json: bool) -> int:
    """``ara characterize <model> --engine ollama``: ramp the model's residency through Ollama to
    find and record its true measured safe ceiling (engine ``"ollama"``), instead of relying on
    serve's self-heal which only ever records the one context it served at. Serve stays fast; this
    is where the slow, thorough measurement lives (Will, 2026-07-04). Spec
    2026-07-04-characterize-through-ollama-ramp."""
    def err(msg: str) -> int:
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1

    if ollama.version() is None:
        return err("Ollama isn't serving — start it with `ollama serve` (or set OLLAMA_HOST).")
    names = ollama.tags()
    if names is None:
        return err("couldn't list Ollama models — is the server reachable?")
    if model not in names:
        return err(f"{model} isn't in Ollama — pull it first: ollama pull {model}")
    max_ctx = _ollama_max_context(model)
    if not max_ctx:
        return err(f"couldn't read {model}'s context length from Ollama — can't bound the ramp.")

    probe = _governed_name(model) + "-probe"
    try:
        best, points = _ollama_measure_ceiling(model, max_ctx, probe)
    finally:
        ollama.delete(probe)                          # never leave the throwaway probe behind
    with db.connected() as con:
        db.save_characterization(con, profile.machine_key(), "ollama", model,
                                 safe_context=best, points=points, measured_at=None)
    if as_json:
        print(json.dumps({"model": model, "engine": "ollama", "safe_context": best,
                          "source": "measured", "max_context": max_ctx}))
        return 0
    if best is None:
        c.emit(c.style("warn", f"  no no-spill ceiling for {model} on this box — it spills even at "
                               f"the smallest tested context (recorded)."))
    else:
        c.emit(c.field("measured ceiling", f"~{best} tokens  (ollama, fully resident)"))
    return 0


def _ollama_pick_best(names: list[str]) -> str | None:
    """Zero-arg ``ara serve`` selection: of the models already in the Ollama store, the one whose
    safe ceiling is largest — measured if we have it, else the conservative estimate — so a bare
    ``ara serve`` stands up the model that gives this machine the most usable context. ``None`` when
    nothing in the store fits/estimates. Recommend applied to what you already have; no HF round-trip."""
    mk = profile.machine_key()
    best_name, best_ceiling = None, -1
    with db.connected() as con:
        for n in names:
            found = _ollama_safe_ceiling(con, mk, n) or _ollama_estimated_ceiling(n)
            if found and found[0] is not None and found[0] > best_ceiling:
                best_name, best_ceiling = n, found[0]
    return best_name


def _governed_name(model: str) -> str:
    """A valid derived Ollama model name carrying the governed ceiling — e.g.
    ``qwen3:0.6b`` → ``qwen3-0.6b-ara``. Anything outside ``[a-z0-9._-]`` becomes ``-``."""
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "-" for ch in model.lower())
    return safe + "-ara"


def _find_loaded(entries: list[dict], served: str) -> dict | None:
    """The ``/api/ps`` entry for our derived model (Ollama tags it ``:latest``), or ``None``."""
    return next((m for m in entries if m.get("name") == served
                 or m.get("name", "").startswith(served + ":")), None)


def _free_port() -> int:
    """An OS-assigned free TCP port on localhost (small bind/close race, acceptable for v1)."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _render_serve_mlx(c: Console, model: str, *, engine_key: str, ctx: int | None = None,
                      assume_yes: bool = False, as_json: bool = False,
                      kv_quant: str = "f16") -> int:
    """Stand *model* up on the governed MLX server (wmx-suite), capped at the MEASURED apple
    ceiling (or explicit ``--ctx``), and hand back an OpenAI-compatible endpoint. ARA owns the
    server subprocess, so it stays foreground until Ctrl-C. The wmx ceiling is valid here because
    serve and characterize share the mlx_lm allocation path (seam-mismatch rule, the other way).
    Spec 2026-06-28-recommend-use-case-and-serve-selection."""
    def err(msg: str) -> int:
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1

    mk = profile.machine_key()
    with db.connected() as con:
        if ctx is not None:
            if ctx <= 0:
                return err("--ctx must be a positive integer")
            _row = db.get_characterization(con, mk, engine_key, model)
            if (msg := _ctx_gate_msg(ctx, _row.get("safe_context") if _row else None, model)):
                return err(msg)
            safe, source = ctx, "requested"
            ceiling_measured_at = None       # explicit --ctx, not a stored ceiling
            measured_slope = None            # a-priori gate for an unmeasured override (Rule #1)
        else:
            row = db.get_characterization(con, mk, engine_key, model)  # keyed by engine key, not backend
            if not row or row.get("safe_context") is None:
                return err(f"no measured MLX ceiling for {model} — run: ara characterize {model} "
                           f"(or pass --ctx N).")
            safe, source = row["safe_context"], "measured"
            ceiling_measured_at = row.get("measured_at")
            # Serving the model's OWN measured ceiling: fit the real ramp slope so the pre-load gate
            # predicts with it, not the conservative a-priori prior that would falsely refuse a
            # long-window measured serve (slug 2026-07-02-wmx-serve-measured-provenance-gate).
            measured_slope = _measured_ramp_slope(row)

    _stale_ceiling_note(c, model, ceiling_measured_at, as_json=as_json)
    if not as_json and not assume_yes and sys.stdin.isatty():
        if not _confirm(f"Serve {model} on MLX, governed at ≤{safe} ctx?"):
            c.emit(c.style("dim", "  skipped."))
            return 0

    port = _free_port()
    try:
        proc, url, served_ctx = get_backend("apple").serve(
            model, port=port, max_context=safe, kv_quant=kv_quant,
            measured_slope_gb_per_k=measured_slope)
    except Exception as exc:                       # gate refusal / engine not installed / etc.
        return err(f"couldn't start the MLX server: {exc}")

    endpoint = url.rstrip("/") + "/v1"
    if as_json:
        print(json.dumps({"endpoint": endpoint, "model": model, "served_context": served_ctx,
                          "ceiling_source": source, "openai_base_url": endpoint,
                          "runtime": "mlx"}, indent=2))
    else:
        c.emit()
        c.emit(c.field("serving", f"{model}  (MLX @ {served_ctx} ctx, {source})"))
        c.emit(c.field("endpoint", f"{endpoint}  (OpenAI-compatible)"))
        c.emit(c.field("use it", f"export OPENAI_BASE_URL={endpoint}"))
        c.emit(c.style("dim", "  serving in the foreground — Ctrl-C to stop."))
        c.emit()
    sys.stdout.flush()                             # hand the endpoint to a piped reader BEFORE we block
    # Stay alive to keep serving. Ctrl-C (SIGINT) hits the whole group, but a bare `kill <pid>`
    # (SIGTERM) would terminate us without running cleanup and orphan the child (GPU + port leak);
    # install a handler that terminates the child first.
    import signal

    def _stop(_sig, _frame):
        proc.terminate()
        sys.exit(0)

    old = signal.signal(signal.SIGTERM, _stop)
    try:
        proc.wait()                                # our child IS the server; stay alive to serve
    finally:
        signal.signal(signal.SIGTERM, old)
    return 0


def render_serve(c: Console, model: str | None = None, *, ctx: int | None = None,
                 name: str | None = None, engine: str | None = None,
                 assume_yes: bool = False, as_json: bool = False) -> int:
    """Stand *model* up as a **governed** OpenAI-compatible endpoint on a local Ollama, capped at a
    safe context ceiling, and hand back the connection — then get out of the way (BYO consumer).

    The ceiling is **baked into a derived model** (``<model>-ara``): a plain ``/v1`` request reloads
    the base model at its *default* context, blowing past the safe wall (measured 2026-06-26), so
    governing per-request isn't enough. The ceiling is *measured* (a llama.cpp-class
    characterization), *explicit* (``--ctx``), or — when nothing is measured — a conservative
    engine-free *estimate* from the model's own ``/api/show`` architecture, always labelled by its
    true source and never a silent guess (Rule #1/#3). A missing model is pulled rather than refused,
    so a fresh model serves in one command. After load it verifies the ceiling actually took before
    returning an endpoint. Specs 2026-06-26-ara-serve-governed-endpoint,
    2026-07-04-ara-serve-one-command-estimated-ceiling. ``--engine wmx`` routes to the governed MLX
    server instead (spec 2026-06-28); the Ollama path below is unchanged."""
    auto_selected = model is None    # bare `ara serve` → pick the best-fitting model in the store
    if engine is not None and model is not None:
        key = engines.resolve(engine)
        if key and engines.ENGINES.get(key, {}).get("backend") == "apple":
            return _render_serve_mlx(c, model, engine_key=key, ctx=ctx,
                                     assume_yes=assume_yes, as_json=as_json)

    def err(msg: str) -> int:
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1

    if engine is not None and model is None:
        return err("`ara serve` with no model picks from the Ollama store — pass a model to use "
                   "--engine.")

    # 1. liveness — honest about a server that isn't there (Rule #3)
    if ollama.version() is None:
        return err("Ollama isn't serving — start it with `ollama serve` (or set OLLAMA_HOST).")

    names = ollama.tags()
    if names is None:
        return err("couldn't list Ollama models — is the server reachable?")

    # 2a. zero-arg: recommend among what's already pulled, then serve the best fit
    if model is None:
        if not names:
            return err("no models in Ollama — pull one (`ollama pull <model>`), or name one: "
                       "`ara serve <model>`.")
        model = _ollama_pick_best(names)
        if model is None:
            return err("no model in Ollama fits this machine — pull a smaller / more-quantized "
                       "one, or name one explicitly.")
        if not as_json:
            c.emit(c.style("dim", "  auto-selected ") + c.style("accent", model)
                   + c.style("dim", " (best fit in your Ollama store)"))

    # 2b. named model: ensure it's in the store — pull it if missing (get out of the way)
    if model not in names:
        if not as_json:
            c.emit(c.style("dim", f"  pulling {model} …"))
        if not ollama.pull(model):
            return err(f"couldn't pull {model} into Ollama — check the model name.")
        if not as_json:
            c.emit(c.style("dim", "  pulled."))

    # 3. resolve the safe ceiling — measured, or explicit; never guessed
    if ctx is not None:
        if ctx <= 0:
            return err("--ctx must be a positive integer")
        # Rule #1 gate: an explicit --ctx must not exceed what was MEASURED for this model on
        # this machine (previously this path never consulted the measurement at all).
        with db.connected() as con:
            found = _ollama_safe_ceiling(con, profile.machine_key(), model)
        if (msg := _ctx_gate_msg(ctx, found[0] if found else None, model)):
            return err(msg)
        safe, source = ctx, "requested"
        ceiling_measured_at = None           # explicit --ctx, not a stored ceiling
    else:
        with db.connected() as con:
            found = _ollama_safe_ceiling(con, profile.machine_key(), model)
        # No measurement yet → fall back to a conservative engine-free ESTIMATE (labelled as such,
        # never as measured — Rule #3), so a fresh model still serves safely in one command.
        if found is None:
            found = _ollama_estimated_ceiling(model)
        if found is None:
            return err(f"couldn't determine a safe ceiling for {model} — pass --ctx N, or run "
                       f"`ara characterize {model}` to measure one.")
        safe, source, ceiling_measured_at = found
        if source == "estimated" and not as_json:
            c.emit(c.style("dim", "  ceiling ") + c.style("accent", "estimated")
                   + c.style("dim", " — run ") + c.style("accent", f"ara characterize {model}")
                   + c.style("dim", " for a measured one"))

    _stale_ceiling_note(c, model, ceiling_measured_at, as_json=as_json)
    # consent — serve creates + holds a model in memory
    if not as_json and not assume_yes and sys.stdin.isatty():
        if not _confirm(f"Stand up {model} on Ollama, governed at ≤{safe} ctx?"):
            c.emit(c.style("dim", "  skipped."))
            return 0

    # 4. bake the ceiling into a derived model
    served = name or _governed_name(model)
    if not ollama.create(served, model, safe):
        return err(f"couldn't create the governed model {served!r} on Ollama.")

    # 5. load + verify the ceiling took — never hand back an ungoverned endpoint (Rule #1)
    ollama.load(served)
    entry = _find_loaded(ollama.ps() or [], served)
    if entry is None:
        return err(f"{served} didn't load — Ollama may be out of memory.")
    served_ctx = entry.get("context_length")
    if served_ctx != safe:
        return err(f"governance failed: Ollama served {served_ctx} ctx, not {safe} — refusing.")
    size, vram = entry.get("size"), entry.get("size_vram")
    spilled = isinstance(size, int) and isinstance(vram, int) and vram < size

    # 5b. self-heal: we just loaded this model at `safe` ctx and verified it fits with NO spill —
    # that is an empirical measurement, not a guess. Record it (engine "ollama") so the next serve
    # reads a `measured` ceiling and skips the estimate. Only ever persist observed-good evidence:
    # never a higher untested ceiling (Rule #1), never a measurement we already had (source
    # "estimated" only), never on spill (no clean evidence). Rule #3: labelled measured because we
    # measured it fits.
    recorded_measured = False
    if source == "estimated" and not spilled:
        with db.connected() as con:
            db.save_characterization(con, profile.machine_key(), "ollama", model,
                                     safe_context=safe, points=[{"context": safe, "fit": True}],
                                     measured_at=None)
        recorded_measured = True

    # 6. the handoff — connection info, then ARA exits (the model stays served)
    endpoint = ollama.base_url() + "/v1"
    if as_json:
        print(json.dumps({"endpoint": endpoint, "model": served, "base_model": model,
                          "served_context": safe, "ceiling_source": source, "spilled": spilled,
                          "auto_selected": auto_selected, "recorded_measured": recorded_measured,
                          "openai_base_url": endpoint}, indent=2))
        return 0
    c.emit()
    c.emit(c.field("serving", f"{served}  ({model} @ {safe} ctx, {source})"))
    c.emit(c.field("endpoint", f"{endpoint}  (OpenAI-compatible)"))
    c.emit(c.field("use it", f"export OPENAI_BASE_URL={endpoint}"))
    if spilled:
        c.emit(c.style("warn", "  note: partially offloaded (size_vram < size) — expect it slow."))
    elif recorded_measured:
        c.emit(c.style("dim", "  recorded a measured ceiling — future serves skip the estimate."))
    c.emit()
    return 0


_INSTALL_OK = ("installed", "refreshed", "already")


def render_install(c: Console, *, engine: str = "auto", refresh: bool = False,
                   as_json: bool = False) -> int:
    """Install the matched engine. ``--engine`` is the consent; exit 0 once the
    engine is present (installed, refreshed, or already), nonzero otherwise.

    ``--refresh`` forces a reinstall even when the engine is already present and current — used to
    repair or re-pin an env after an ARA upgrade."""
    key = engines.resolve(engine)
    if key is None:
        if as_json:
            print(json.dumps({"status": "no_match", "engine": engine}))
        else:
            c.emit(c.style("warn", f"  no engine matches '{engine}' on this hardware"))
        return 1

    result = engines.install(key, refresh=refresh)
    if as_json:
        print(json.dumps({"key": result.key, "status": result.status,
                          "detail": result.detail}))
        return 0 if result.status in _INSTALL_OK else 1

    pkg = engines.ENGINES[key]["package"]
    if result.status == "installed":
        c.emit(c.style("good", f"  installed {pkg}"))
    elif result.status == "refreshed":
        c.emit(c.style("good", f"  refreshed {pkg} to the current ARA release"))
    elif result.status == "already":
        c.emit(c.style("dim", f"  {pkg} already installed"))
    elif result.status == "coming_soon":
        c.emit(c.style("warn", f"  {pkg} — coming soon (not installable yet)"))
    else:  # failed
        c.emit(c.style("bad", f"  installing {pkg} failed:"))
        c.emit(c.style("dim", f"  {result.detail}"))
    return 0 if result.status in _INSTALL_OK else 1


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
    with db.connected() as con:
        rows = db.list_characterizations(con, profile.machine_key(), engine_key)
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
    with db.connected() as con:
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
    with db.connected() as con:
        measured = calibration.get_calibration(con, sel.engine_key)
    lim = estimate.limits(m, measured=measured)

    # profile is the persister (detect stays ephemeral): record this analytic snapshot, history kept.
    with db.connected() as con:
        profile.capture(con)

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
def _node_say(c: Console, as_json: bool, msg: str, **fields) -> int:
    """Success line for an `ara node` subcommand: a JSON object under --json, a styled field else."""
    if as_json:
        print(json.dumps({"ok": True, "message": msg, **fields}))
    else:
        c.emit(c.field("ara node", msg))
    return 0


def _node_err(c: Console, as_json: bool, msg: str) -> int:
    """Failure line for an `ara node` subcommand (Rule #3: structured under --json, styled warn else)."""
    print(json.dumps({"error": msg})) if as_json else c.emit(c.style("warn", f"  {msg}"))
    return 1


def render_node(c: Console, rest: list[str], *, token: str | None = None,
                as_json: bool = False) -> int:
    """`ara node <sub>` — run/manage the push-only node daemon (a client that phones home).

    Subcommands: ``enroll <server_url> --token <t>`` (phone home to a coordinator and wait for
    approval) and ``run`` (the push-only work loop); ``install`` (write + enable the systemd --user
    boot unit, whose ExecStart runs ``ara node run``), and ``start``/``stop``/``status``/
    ``uninstall`` (service lifecycle). The systemd path is Linux-only and raises a clear message
    elsewhere. The node holds no inbound socket — it only ever dials out to its coordinator.
    """
    from ara.node import agent, config, enroll, service

    sub = rest[1] if len(rest) > 1 else None

    if sub == "enroll":
        server_url = rest[2] if len(rest) > 2 else None
        if not server_url or not token:
            return _node_err(c, as_json, "usage: ara node enroll <server_url> --token <token>")
        cfg = config.NodeConfig(server_url=server_url, enrollment_token=token)
        config.save(cfg)
        try:
            enroll.enroll_flow(cfg)
        except Exception as exc:                    # surface, don't swallow: enrollment can fail
            return _node_err(c, as_json, f"enrollment failed: {exc}")
        return _node_say(c, as_json, f"enrolled with {server_url}", endpoint=server_url)

    if sub == "run":
        cfg = config.load()
        if cfg is None:
            return _node_err(c, as_json,
                             "not enrolled — run: ara node enroll <server_url> --token <token>")
        agent.run_loop(cfg)                         # blocks: phone-home work loop until stopped
        return _node_say(c, as_json, "run loop exited")

    if sub in ("install", "start", "stop", "status", "uninstall"):
        try:
            if sub == "install":
                service.install()
                return _node_say(c, as_json, "installed + started (systemd --user)")
            if sub == "status":
                out = service.status()
                print(json.dumps({"status": out})) if as_json else c.emit(out.rstrip())
                return 0
            getattr(service, sub)()            # start | stop | uninstall
            return _node_say(c, as_json, f"{sub} ok")
        except RuntimeError as exc:             # the systemd path is Linux-only
            return _node_err(c, as_json, str(exc))

    return _node_err(c, as_json,
                     "usage: ara node {enroll|run|install|start|stop|status|uninstall}")


_DOCTOR_TABLES = ("calibrations", "characterizations", "profiles", "benchmark_results")


def render_doctor(c: Console, *, rekey: bool = False, as_json: bool = False) -> int:
    """``ara doctor``: this machine's identity (``machine_key``) and the count of stored records
    keyed to it, so a user can see at a glance whether ARA still recognises the box. With
    ``--rekey``, first migrate any lingering legacy (byte-exact) machine_keys to the versioned
    GiB-rounded form and report how many rows moved (the manual counterpart to the automatic
    one-time migration). Rows under *other* keys are counted separately, never folded in (Rule #3).
    Spec 2026-07-04-machine-key-stabilization."""
    mk = profile.machine_key()
    with db.connected() as con:
        rekeyed = db._rekey_legacy(con) if rekey else None
        counts = {t: con.execute(f"SELECT COUNT(*) FROM {t} WHERE machine_key=?",  # noqa: S608
                                 (mk,)).fetchone()[0] for t in _DOCTOR_TABLES}
        other = sum(con.execute(f"SELECT COUNT(*) FROM {t} WHERE machine_key<>?",  # noqa: S608
                                (mk,)).fetchone()[0] for t in _DOCTOR_TABLES)
    if as_json:
        out: dict = {"machine_key": mk, "counts": counts, "other_keys_rows": other}
        if rekey:
            out["rekeyed_rows"] = rekeyed
        print(json.dumps(out))
        return 0
    if rekey:
        c.emit(c.style("good" if rekeyed else "dim",
                       f"  rekeyed {rekeyed} legacy row(s) to the versioned machine_key format"))
    c.emit(c.style("accent", "  machine  ") + c.style("dim", mk))
    for name, n in counts.items():
        c.emit(f"    {name:<18} {n}")
    if other:
        c.emit(c.style("dim", f"    ({other} row(s) under other machine keys — not this box)"))
    return 0


def main() -> int:
    """CLI entry. Front-door honesty guard (Rule #3): an exception that escapes a command under
    ``--json`` becomes a structured ``{"error": ...}`` instead of a raw traceback a JSON consumer
    can't parse. Without ``--json``, an :class:`~ara.engine_env.EngineEnvError` (the common
    engine-env failure — a broken/missing env, a dead worker) prints a friendly one-line diagnostic
    instead of a raw traceback; any other exception still propagates. KeyboardInterrupt / SystemExit
    are not caught."""
    try:
        return _main_impl()
    except Exception as exc:   # noqa: BLE001 — deliberate front-door honesty guard
        if isinstance(exc, MeasurementBusy):   # a concurrent measurement holds the lock — say so
            print(json.dumps({"error": str(exc)})) if "--json" in sys.argv[1:] \
                else Console.from_env().emit(Console.from_env().style("warn", f"  {exc}"))
            return 1
        if "--json" in sys.argv[1:]:
            print(json.dumps({"error": f"ara failed: {exc}"}))
            return 1
        if isinstance(exc, EngineEnvError):
            c = Console.from_env()
            c.emit(c.style("bad", f"  engine env problem: {exc}"))
            c.emit(c.style("dim", "  check the GPU driver / toolchain and retry: ara install"))
            return 1
        raise


def _main_impl() -> int:
    argv = sys.argv[1:]
    if "--version" in argv:
        print(_ara_version())
        return 0
    wants_help = "-h" in argv or "--help" in argv
    verbose = "--verbose" in argv or "-v" in argv
    as_json = "--json" in argv
    assume_yes = "--yes" in argv or "-y" in argv
    exec_consent = "--exec-consent" in argv      # explicit opt-in to model-code execution (benchmark)
    refresh = "--refresh" in argv                 # `install --refresh`: force reinstall of a present engine
    rekey = "--rekey" in argv                     # `doctor --rekey`: migrate legacy machine_keys
    flash_attn = "--no-flash-attn" not in argv   # vulkan engine: FA on by default, this disables it
    flash_attn_optin = "--flash-attn" in argv    # cuda engine: SDPA by default, this opts into FA2
    chunked_prefill = "--chunked-prefill" in argv  # cuda engine: opt into chunked prefill (def 512)

    # --model / --engine / --token / --include / --exclude / --kv-quant take values; pull them first.
    model: str | None = None
    engine: str | None = None
    token: str | None = None
    kv_quant: str = "f16"
    weight_quant: str = "none"
    prefill_chunk_val: int | None = None         # explicit --prefill-chunk N (overrides the default)
    serve_ctx: int | None = None                 # `serve --ctx N`: explicit governed context
    serve_name: str | None = None                # `serve --name X`: derived served-model name
    use_case: str | None = None                  # `recommend --use-case X`: capability dimension
    max_tokens_val: int | None = None            # `benchmark --max-tokens N`: lift the generation cap
    repeat_val: int = 1                          # `benchmark --repeat N`: runs for the variance band
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
        if a == "--kv-quant":
            kv_quant = argv[i + 1] if i + 1 < len(argv) else ""
            skip = True
            continue
        if a.startswith("--kv-quant="):
            kv_quant = a.split("=", 1)[1]
            continue
        if a == "--weight-quant":
            weight_quant = argv[i + 1] if i + 1 < len(argv) else ""
            skip = True
            continue
        if a.startswith("--weight-quant="):
            weight_quant = a.split("=", 1)[1]
            continue
        if a == "--prefill-chunk":
            prefill_chunk_val = _int_or_none(argv[i + 1] if i + 1 < len(argv) else "")
            skip = True
            continue
        if a.startswith("--prefill-chunk="):
            prefill_chunk_val = _int_or_none(a.split("=", 1)[1])
            continue
        if a == "--ctx":
            serve_ctx = _int_or_none(argv[i + 1] if i + 1 < len(argv) else "")
            skip = True
            continue
        if a.startswith("--ctx="):
            serve_ctx = _int_or_none(a.split("=", 1)[1])
            continue
        if a == "--max-tokens":
            max_tokens_val = _int_or_none(argv[i + 1] if i + 1 < len(argv) else "")
            skip = True
            continue
        if a.startswith("--max-tokens="):
            max_tokens_val = _int_or_none(a.split("=", 1)[1])
            continue
        if a == "--repeat":
            # Bad/non-integer value → 0, which render_benchmark rejects (the positive-integer gate).
            p = _int_or_none(argv[i + 1] if i + 1 < len(argv) else "")
            repeat_val = p if p is not None else 0
            skip = True
            continue
        if a.startswith("--repeat="):
            p = _int_or_none(a.split("=", 1)[1])
            repeat_val = p if p is not None else 0
            continue
        if a == "--name":
            serve_name = argv[i + 1] if i + 1 < len(argv) else None
            skip = True
            continue
        if a.startswith("--name="):
            serve_name = a.split("=", 1)[1] or None
            continue
        if a == "--use-case":
            use_case = argv[i + 1] if i + 1 < len(argv) else None
            skip = True
            continue
        if a.startswith("--use-case="):
            use_case = a.split("=", 1)[1] or None
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
        if a in ("--verbose", "-v", "--json", "--yes", "-y", "--exec-consent", "--refresh",
                 "--rekey", "--no-flash-attn", "--flash-attn",
                 "--chunked-prefill", "-h", "--help"):
            continue
        rest.append(a)
    c = Console.from_env(verbose=verbose)

    # Resolve the chunked-prefill lever: an explicit size wins; a bare --chunked-prefill uses the
    # default; neither → off (single-shot). One source of truth passed down to the cuda worker.
    prefill_chunk = prefill_chunk_val if prefill_chunk_val is not None else (
        _DEFAULT_PREFILL_CHUNK if chunked_prefill else None)

    def _arg_err(msg: str) -> int:
        """Usage / dispatch error: structured JSON under --json, styled warn otherwise (Rule #3)."""
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("warn", f"  {msg}"))
        return 1

    if wants_help:                          # `ara <cmd> --help` / `-h` → that command's usage
        return render_help(c, rest[0] if rest else None, as_json=as_json)
    if not rest:
        render_landing(c)
        return 0

    cmd = rest[0]
    # Section filtering is shared across the recon commands; build the predicate once.
    want = (_resolve_want(cmd, include, exclude, c, as_json=as_json)
            if (include or exclude) else None)

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
            return _arg_err("usage: ara search <query>")
        return render_search(c, " ".join(rest[1:]), as_json=as_json)

    if cmd == "characterize":
        if len(rest) < 2:
            return _arg_err("usage: ara characterize <model>")
        # Hold the machine's measurement lock: a concurrent characterize/benchmark would read this
        # one's memory footprint into its own reading and store a corrupted ceiling (Rule #1, G9).
        with locking.measurement_lock():
            return render_characterize(c, rest[1], engine=engine, as_json=as_json,
                                       flash_attn=flash_attn, flash_attn_optin=flash_attn_optin,
                                       kv_quant=kv_quant, weight_quant=weight_quant,
                                       prefill_chunk=prefill_chunk)

    if cmd == "profile":
        return render_profile(c, as_json=as_json, model=model, engine=engine)

    if cmd == "recommend":
        return render_recommend(c, as_json=as_json, use_case=use_case)

    if cmd == "run":
        if len(rest) < 2:
            return _arg_err("usage: ara run <model> <prompt>")
        return render_run(c, rest[1], prompt=" ".join(rest[2:]) or None,
                          engine=engine, assume_yes=assume_yes, as_json=as_json,
                          flash_attn=flash_attn, flash_attn_optin=flash_attn_optin,
                          kv_quant=kv_quant, weight_quant=weight_quant,
                          prefill_chunk=prefill_chunk)

    if cmd == "serve":
        # no model → zero-arg: recommend among the Ollama store, then serve the best fit
        return render_serve(c, rest[1] if len(rest) >= 2 else None, ctx=serve_ctx,
                            name=serve_name, engine=engine, assume_yes=assume_yes, as_json=as_json)

    if cmd == "benchmark":
        if len(rest) < 2 or use_case is None:
            return _arg_err("usage: ara benchmark <model> "
                            "--use-case <coding|reasoning|agentic|extraction|rag>")
        with locking.measurement_lock():        # same measurement lock as characterize (Rule #1, G9)
            return render_benchmark(c, rest[1], use_case=use_case, engine=engine, ctx=serve_ctx,
                                    max_tokens=max_tokens_val, repeat=repeat_val,
                                    assume_yes=assume_yes,
                                    exec_consent=exec_consent, as_json=as_json)

    if cmd == "install":
        # engine from a positional (`ara install wmx`), else --engine, else the auto-matched one.
        return render_install(c, engine=rest[1] if len(rest) > 1 else (engine or "auto"),
                              refresh=refresh, as_json=as_json)

    if cmd == "uninstall":
        return render_uninstall(c, engine=rest[1] if len(rest) > 1 else (engine or "auto"),
                                as_json=as_json)

    if cmd == "hf":
        return render_hf(c, rest[1] if len(rest) > 1 else None, token=token, as_json=as_json)

    if cmd == "node":
        return render_node(c, rest, token=token, as_json=as_json)

    if cmd == "doctor":
        return render_doctor(c, rekey=rekey, as_json=as_json)

    if as_json:
        return _arg_err(f"'{rest[0]}' isn't built yet — ARA is an early scaffold.")
    c.emit(c.style("warn", f"  '{rest[0]}' isn't built yet — ARA is an early scaffold."))
    c.emit(
        c.style("dim", "  run ") + c.style("accent", "ara")
        + c.style("dim", " with no arguments to see the planned commands.")
    )
    return 1


if __name__ == "__main__":   # pragma: no cover — so `python -m ara.cli ...` works (node wiring shells out this way)
    sys.exit(main())
