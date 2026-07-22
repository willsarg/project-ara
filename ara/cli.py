# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""ARA command-line front door.

``ara`` with no arguments renders the landing screen. The full command roster —
detect recon, live ARA activity, models/search, profile/characterize/recommend,
governed run/serve, benchmark, hub/node (fleet), install/uninstall, hf auth, doctor — is
dispatched below; an unrecognized command falls through to a clear error.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
from collections.abc import Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path

import click

from ara import (acquire, activity, apps, benchmark, catalog, db, detect, engine_audit,
                 engine_identity, engines, estimate, hub,
                 hub_server,
                 hf_auth, locking, mlx, ollama, ollama_evidence, profile, calibration, pythons, scoring, serialize,
                 staleness, versions)
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
    "python": ("interpreters",),
}
_SECTION_ALIASES = {
    "gpu": "accelerator", "app": "apps", "framework": "frameworks", "engine": "engines",
    "model": "models", "lib": "libraries", "libs": "libraries", "library": "libraries",
    "ready": "readiness",
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
# landing
# --------------------------------------------------------------------------- #
def _landing_hardware(chip: str, backend: str, mem_gb: float | None,
                      gpu_cores: int | None, gpu_name: str | None) -> list[str]:
    """The 'this machine:' tokens — chip, memory, GPU — in plain user terms only (never an
    internal backend/engine key). Memory is the *unified* pool on Apple (the shared wall ARA
    governs) and plain 'RAM' elsewhere; the GPU is shown by core count on Apple, or by name for a
    discrete card. Ordered: chip first, then whatever hardware we could read."""
    tokens = [chip]
    if mem_gb:
        tokens.append(f"{mem_gb:.0f} GB {'unified memory' if backend == 'apple' else 'RAM'}")
    if backend == "apple" and gpu_cores:
        tokens.append(f"{gpu_cores}-core GPU")
    elif gpu_name:
        tokens.append(gpu_name)
    return tokens


def _guidance_engine(backend: str | None = None) -> str:
    """Choose the concrete engine shown in human first-run guidance.

    Automatic install intentionally covers only detected Apple/NVIDIA acceleration. When no
    supported accelerated lane is detected, guidance names the portable CPU lane explicitly.
    """
    return engines.for_backend(backend or detect.backend_name()) or "cpu"


def _install_guidance_command(engine_key: str) -> str:
    """The install command a newcomer should copy for *engine_key*."""
    return "ara install" if engine_key in ("mlx", "cuda") else f"ara install --engine {engine_key}"


def _emit_first_model_path(c: Console, engine_key: str | None = None) -> None:
    """Explain how an empty catalog becomes one measured, runnable smoke model."""
    key = engine_key or _guidance_engine()
    model = engines.ENGINES[key]["smoke_model"]
    c.emit(c.style("dim", "  no models cached yet"))
    c.emit(c.style("dim", "  start: ") + c.style("accent", _install_guidance_command(key)))
    c.emit(c.style("dim", "  then:  ")
           + c.style("accent", f"ara characterize {model} --engine {key}"))
    c.emit(c.style("dim", "         downloads a tiny demo model if needed, then measures its "
                           "safe context ceiling"))
    c.emit(c.style("dim", "  run:   ")
           + c.style("accent", f'ara run {model} "Explain local AI simply"'))
    c.emit(c.style("dim", "         after characterization succeeds"))


def render_landing(c: Console) -> None:
    chip = detect.chip_name()
    backend = detect.backend_name()
    accelerated = backend in ("apple", "cuda")
    engine_key = _guidance_engine(backend)
    smoke_model = engines.ENGINES[engine_key]["smoke_model"]

    acc = detect.accelerator(chip)
    tokens = _landing_hardware(
        chip, backend, detect._memory_gb()[0],
        acc.cores if backend == "apple" else None,
        acc.name if backend in ("cuda", "vulkan") else None,
    )

    c.emit(
        "  " + c.style("accent", "ara")
        + c.style("dim", "  —  AI Runs Anywhere: run local models on whatever hardware you've got")
    )
    c.emit()
    line = c.style("dim", "  this machine: ") + c.style("metric", tokens[0])
    for tok in tokens[1:]:
        line += c.style("dim", " · ") + c.style("metric", tok)
    c.emit(line)
    c.emit()
    c.emit(c.section("  FIRST RUN"))
    c.emit(_cmd(c, "ara detect", "inspect this machine — read-only recon"))
    c.emit(c.style("dim", "                  --python · --apps · --runtime · --models · --json"))
    install_command = _install_guidance_command(engine_key).removeprefix("ara ")
    c.emit(_cmd(c, f"ara {install_command}", "install the engine for this machine"))
    c.emit(_cmd(c, f"ara characterize {smoke_model} --engine {engine_key}",
                "download a tiny demo model and measure its safe context ceiling"))
    c.emit(_cmd(c, f'ara run {smoke_model} "Explain local AI simply"',
                "generate one completion under that measured ceiling"))
    c.emit()
    c.emit(c.section("  EXPLORE"))
    c.emit(_cmd(c, "ara status", "show what ARA is doing right now"))
    c.emit(_cmd(c, "ara models search", "find models on the Hugging Face Hub"))
    c.emit(_cmd(c, "ara detect --models", "catalog models physically cached on this machine"))
    c.emit(_cmd(c, "ara profile", "estimate this machine's capability (analytic — no engine)"))
    c.emit(_cmd(c, "ara models recommend", "catalog models that fit, ranked by usable context"))
    c.emit(_cmd(c, "ara serve <model>", "stand it up safely on Ollama + hand back the endpoint"))
    c.emit(_cmd(c, "ara benchmark <model>", "run a capability probe and store the measured score"))
    c.emit(_cmd(c, "ara hub", "host the coordinator that ARA nodes phone home to"))
    c.emit(_cmd(c, "ara node <sub>", "run ARA as a push-only daemon that phones home (enroll/run/install/…)"))
    c.emit()
    if not accelerated:
        c.emit(c.style("dim", "  no accelerator auto-selected — using the portable CPU path "
                              "(llama.cpp); "
                              "install with ") + c.style("accent", "ara install --engine cpu"))
    c.emit(c.style("dim", "  start with ") + c.style("accent", "ara detect"))


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
        c.emit(c.field("pythons", str(n_py),
                       "interpreters on this machine — run: ara detect --python"))
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


def _det_engines(c: Console, m, *, show_absent: bool = False) -> None:
    engines = [rt for rt in m.runtimes if rt.kind == "engine"]
    c.emit(c.section("  ENGINES") + c.style("dim", "  (third-party launchers found on this system)"))
    for rt in engines:
        if rt.present:
            val = f"{rt.name} {rt.version}" if rt.version else rt.name
            if rt.requires:  # installed, but can't accelerate on this hardware
                c.emit(c.field("·", val, f"installed · {rt.requires}", value_role="warn"))
            elif rt.serving is False:  # a server runtime that's installed but not up
                if rt.endpoint_scope == "unknown":
                    detail = "installed · invalid Ollama endpoint configuration"
                elif rt.endpoint_scope in {"remote", "cloud"}:
                    detail = f"installed · {rt.endpoint_scope} endpoint unreachable"
                else:
                    detail = "installed · local endpoint not serving — run `ollama serve`"
                if rt.endpoint is not None:
                    detail += f" · {rt.endpoint}"
                c.emit(c.field("·", val, detail, value_role="warn"))
            elif rt.serving is True:
                detail = "serving"
                if rt.endpoint_scope is not None:
                    detail += f" · {rt.endpoint_scope}"
                if rt.endpoint is not None:
                    detail += f" · {rt.endpoint}"
                c.emit(c.field("·", val, detail, value_role="good"))
            else:
                c.emit(c.field("·", val, "found", value_role="good"))
        elif c.verbose or show_absent:
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

    if present_fw:
        libs = " · ".join(f"{rt.name} {rt.version}".strip() for rt in present_fw)
        c.emit(c.style("dim", "  Your default python has AI frameworks:"))
        c.emit("      " + c.style("accent", m.framework_python))
        c.emit("      " + c.style("good", libs))
    else:
        if m.framework_python:
            c.emit(c.style("dim", "  Your default python has no AI frameworks:"))
            c.emit("      " + c.style("accent", m.framework_python))
        else:
            c.emit(c.style("dim", "  no separate user Python found"))
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
            c.emit(c.style("dim", "  Run ") + c.style("accent", "ara detect --python")
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
    # Detect shows a per-category summary; `ara detect --apps` has the full list with versions.
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
           + c.style("accent", "ara detect --apps")
           + c.style("dim", " for the full list with versions"))
    c.emit()


def _det_ara(c: Console, m) -> None:
    backend_hint = "auto-picked for this hardware"
    if not m.accelerated:
        backend_hint = "portable CPU path — no accelerator auto-selected"
        if any(g.usable_backend == "vulkan" for g in getattr(m, "gpus", [])):
            backend_hint += (
                "; Vulkan acceleration available — optional: ara install --engine vulkan")
    install_key = _guidance_engine(m.backend)
    c.emit(c.section("  ARA"))
    c.emit(c.field(
        "backend", m.backend,
        backend_hint,
        value_role="good",
    ))
    c.emit(c.field(
        "engine", f"{m.engine} {'ready' if m.engine_ready else 'not installed'}",
        None if m.engine_ready else f"install: {_install_guidance_command(install_key)}",
        value_role="good" if m.engine_ready else "warn",
    ))
    c.emit(c.field("hf cli",
                   ("not found" if not m.hf_cli
                    else f"present {m.hf_cli_version}" if m.hf_cli_version else "present"),
                   "the hf command" if m.hf_cli else
                   "missing from ARA's environment — source checkout: uv sync --frozen",
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


def _mlx_runtime_detail(m) -> dict:
    """Observed Apple-only MLX ecosystem detail for the common runtime report."""
    interps = mlx.scan()
    return {
        "source": "read-only user ecosystem probes",
        "gpu": {"name": m.accel.name, "cores": m.accel.cores},
        "mlx_community_models": mlx.mlx_community_model_count(),
        "lmstudio_mlx_runtimes": mlx.lmstudio_mlx_runtimes(),
        "interpreters": [
            {"path": item.path, "origin": item.origin, "version": item.version,
             "packages": item.packages}
            for item in interps
        ],
    }


def render_runtime(c: Console, *, as_json: bool = False, want=None) -> None:
    """Cross-platform runtime/backend inventory; MLX ecosystem detail is Darwin-only.

    This is recon over :func:`detect.machine` plus the existing read-only MLX probes. It never
    resolves, imports, installs, or loads a hardware engine.
    """
    del want  # Facet reports one fixed inventory; include/exclude belong to the full detect report.
    m = detect.machine()
    apple_mlx = m.system == "Darwin" and m.accel.kind == "apple"
    detail = _mlx_runtime_detail(m) if apple_mlx else None
    if as_json:
        payload = {
            "system": m.system,
            "backend_selection": {
                "name": m.backend,
                "source": "observed hardware selection",
            },
            "ara_engine": {
                "name": m.engine,
                "ready": m.engine_ready,
                "source": "ARA isolated engine environment",
            },
            "user_environment": {
                "source": "user environment",
                "runtimes": [asdict(runtime) for runtime in m.runtimes],
            },
        }
        if detail is not None:
            payload["mlx_ecosystem"] = detail
        print(json.dumps(payload, indent=2))
        return

    c.emit()
    c.emit(c.section("  RUNTIME"))
    c.emit(c.field("backend selection", m.backend, "observed hardware selection", label_width=19))
    c.emit()
    c.emit(c.section("  ARA ISOLATED ENGINE ENVIRONMENT"))
    c.emit(c.field("engine", m.engine, "ready" if m.engine_ready else "not installed",
                   value_role="good" if m.engine_ready else "warn"))
    c.emit()
    c.emit(c.section("  USER ENVIRONMENT") + c.style("dim", "  (user environment)"))
    c.emit()
    _det_engines(c, m, show_absent=True)
    _det_frameworks(c, m)
    if detail is not None:
        c.emit(c.section("  MLX ECOSYSTEM")
               + c.style("dim", "  (read-only user ecosystem probes · Apple Silicon)"))
        gpu = detail["gpu"]
        cores = f"{gpu['cores']}-core Metal" if gpu["cores"] else "Metal"
        c.emit(c.field("GPU", gpu["name"], cores))
        c.emit(c.field("models", f"{detail['mlx_community_models']} cached",
                       "mlx-community models in the HF cache"))
        runtimes = detail["lmstudio_mlx_runtimes"]
        c.emit(c.field("LM Studio", f"MLX runtime {runtimes[0]}" if runtimes else "not found"))
        packages = sorted({name for item in detail["interpreters"]
                           for name in item["packages"]})
        c.emit(c.field("libraries", " · ".join(packages) if packages else "none found"))
        c.emit()


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
# status (only ARA-owned live activity; never infers unrelated machine state)
# --------------------------------------------------------------------------- #
def _activity_text(item: activity.Activity) -> str:
    if item.kind == "searching":
        return "searching for models"
    if item.kind == "hosting":
        return "hosting the fleet coordinator"
    return f"{item.kind} {item.model or 'a model'}"


def render_status(c: Console, *, as_json: bool = False) -> None:
    activities = activity.snapshot()
    if as_json:
        state = "idle" if not activities else activities[0].kind \
            if len(activities) == 1 else "active"
        public = [{"kind": item.kind,
                   **({"model": item.model} if item.model is not None else {}),
                   **({"pid": item.pid} if item.pid is not None else {}),
                   "started_at": item.started_at,
                   **({"runtime": item.runtime} if item.runtime is not None else {}),
                   **({"served_name": item.served_name}
                      if item.served_name is not None else {}),
                   **({"context": item.context} if item.context is not None else {}),
                   **({"endpoint": item.endpoint} if item.endpoint is not None else {})}
                  for item in activities]
        print(json.dumps({"state": state, "activities": public}, indent=2))
        return
    if not activities:
        c.emit("ARA is idle.")
    elif len(activities) == 1:
        c.emit(f"ARA is {_activity_text(activities[0])}.")
    else:
        c.emit("ARA is active:")
        for item in activities:
            c.emit(f"  {_activity_text(item)}")


def render_hub(c: Console, *, bind: str, port: int, data_dir: Path,
               rebuild: bool) -> int:
    """Build and attach to the Docker-backed fleet coordinator."""
    endpoint = f"http://{bind}:{port}"
    c.emit(c.field("hub", endpoint, "starting in Docker; Ctrl-C stops it"))
    c.emit(c.field("data", str(data_dir), "persistent host storage"))
    try:
        with activity.track("hosting"):
            return hub_server.run(
                bind=bind, port=port, data_dir=data_dir,
                version=_ara_version(), rebuild=rebuild,
            )
    except hub_server.HubError as exc:
        c.emit(c.style("bad", f"  hub failed: {exc}"))
        return 1


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
            c.emit("  " + c.style("dim", "Use a uv-managed project environment, e.g. ")
                   + c.style("accent", "uv add mlx-lm"))
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
    if m.get("engine"):
        c.emit(c.field("engine", m["engine"]))
    c.emit(c.field("device", f"{m['device']} · {_fmt_gb(m['total_gb'], 0)}"))
    c.emit(c.field("crash wall", _fmt_gb(m["wall_gb"], 1),
                   "the hard ceiling — never cross", value_role="bad"))
    c.emit(c.field("safe budget", _fmt_gb(m["safe_budget_gb"], 1),
                   f"wall − {m['margin_gb']:.0f} GB margin", value_role="good"))
    if c.verbose:
        if measured:
            calibrated_at = m.get("calibrated_at") or "unknown"
            c.emit(c.field("provenance", "stored measurement",
                           f"calibrated {calibrated_at}"))
            if m.get("estimated_wall_gb") is not None:
                c.emit(c.field("analytic wall", _fmt_gb(m["estimated_wall_gb"], 1),
                               "before measured correction", label_width=16))
            if m.get("estimated_safe_budget_gb") is not None:
                c.emit(c.field("analytic budget", _fmt_gb(m["estimated_safe_budget_gb"], 1),
                               "before measured correction", label_width=16))
        else:
            c.emit(c.field("provenance", "analytic estimate", "read-only hardware facts"))
    if m.get("headroom_gb") is not None:
        c.emit(c.field("headroom", _fmt_gb(m["headroom_gb"], 1), "free under budget right now"))
    if m["overhead_gb"] is not None:
        gloss = "default estimate" if not m["calibrated"] else \
            f"measured cold-start · calibrated {m.get('calibrated_at') or 'unknown'}"
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
    engine's native units (MLX: decimal GB) — the SAME units the MLX serve gate predicts in — so
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


def _ollama_display_authority(endpoint: ollama.OllamaEndpoint) -> ollama.OllamaRuntimeAuthority:
    """Return attributable local authority, or explicit display-only authority for other scopes."""
    if endpoint.scope == "loopback":
        return ollama.runtime_authority(endpoint)
    return ollama.OllamaRuntimeAuthority(endpoint=endpoint, issue="non_loopback_endpoint")


def _ollama_trained_context(record: ollama.OllamaModel, detail: dict | None) -> int | None:
    """Prefer typed inventory context, then an exact positive ``model_info`` context field."""
    if record.context_length is not None:
        return record.context_length
    info = detail.get("model_info") if isinstance(detail, dict) else None
    if not isinstance(info, dict):
        return None
    values = [value for key, value in info.items()
              if isinstance(key, str) and key.endswith(".context_length")
              and isinstance(value, int) and not isinstance(value, bool) and value > 0]
    return max(values) if values else None


def _ollama_characterization_view(
    assessment: ollama_evidence.CharacterizationAssessment,
    artifact_id: str | None,
) -> list[dict]:
    """Public, target/config-separated view of the current compatibility-schema evidence row."""
    row = assessment.display
    if row is None:
        return []
    config = row.get("config") if isinstance(row.get("config"), dict) else None
    reusable = assessment.reusable is not None
    artifact_matches = artifact_id is not None and row.get("artifact_id") == artifact_id
    return [{
        "safe_context": row.get("safe_context"),
        "measured_at": row.get("measured_at"),
        "artifact_id": row.get("artifact_id"),
        "artifact_matches": artifact_matches,
        "stale": not artifact_matches,
        "target": config.get("endpoint_authority") if config is not None else None,
        "config": config,
        "reusable": reusable,
        "status": "reusable" if reusable else "display_only",
        "reason": assessment.reason,
    }]


def _render_model_detail_ollama(c: Console, model_id: str, *, as_json: bool) -> int:
    """Read-only exact Ollama artifact, API detail, and reusable/display-only evidence."""
    def err(msg: str) -> int:
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1

    endpoint = ollama.endpoint_authority()
    if endpoint.url is None:
        return err("couldn't identify the configured Ollama endpoint")
    models = ollama.inventory()
    if models is None:
        return err("couldn't list Ollama models — is the server reachable?")
    record = ollama.find_model(models, model_id)
    if record is None:
        return err(f"{model_id} isn't in the configured Ollama store")
    detail = ollama.show(record.name)
    authority = _ollama_display_authority(endpoint)
    with ExitStack() as stack:
        scratch = sqlite3.connect(":memory:")
        scratch.row_factory = sqlite3.Row
        scratch.executescript(db.SCHEMA)
        stack.callback(scratch.close)
        con = (stack.enter_context(db.connected_readonly())
               if db._db_path().is_file() else scratch)
        assessment = ollama_evidence.assess_characterization(
            con, profile.machine_key(), record, authority)
    artifact_id = (_OLLAMA_ARTIFACT_PREFIX + record.digest
                   if record.digest is not None else None)
    info = detail.get("model_info") if isinstance(detail, dict) else None
    payload = {
        "model_id": model_id,
        "canonical_name": record.name,
        "runtime": "ollama",
        "endpoint": endpoint.url,
        "endpoint_scope": endpoint.scope,
        "scope": record.scope,
        "remote_model": record.remote_model,
        "remote_host": record.remote_host,
        "artifact_id": artifact_id,
        "digest": record.digest,
        "size_bytes": record.size_bytes,
        "format": record.format,
        "family": record.family,
        "families": list(record.families),
        "parameter_size": record.parameter_size,
        "quantization": record.quantization,
        "capabilities": list(record.capabilities),
        "trained_context": _ollama_trained_context(record, detail),
        "embedding_length": record.embedding_length,
        "show_available": isinstance(detail, dict),
        "model_info": info if isinstance(info, dict) else None,
        "baked_parameters": detail.get("parameters") if isinstance(detail, dict) else None,
        "template": detail.get("template") if isinstance(detail, dict) else None,
        "runtime_authority_issue": authority.issue,
        "characterizations": _ollama_characterization_view(assessment, artifact_id),
    }
    if as_json:
        print(json.dumps(payload, indent=2))
        return 0
    c.emit()
    c.emit(c.section(f"  {record.name}") + c.style("dim", "  (Ollama)"))
    c.emit(c.field("artifact", artifact_id or "unknown"))
    c.emit(c.field("scope", record.scope))
    c.emit(c.field("format", record.format or "unknown"))
    c.emit(c.field("quant", record.quantization or "unknown"))
    c.emit(c.field("parameters", record.parameter_size or "unknown"))
    c.emit(c.field("capabilities", ", ".join(record.capabilities) or "unknown"))
    trained = payload["trained_context"]
    c.emit(c.field("trained context", str(trained) if trained is not None else "unknown"))
    rows = payload["characterizations"]
    if not rows:
        c.emit(c.field("ceiling", "not characterized for this exact artifact"))
    else:
        for row in rows:
            ceiling = row["safe_context"]
            label = f"~{ceiling} tokens" if ceiling is not None else "no safe ceiling"
            label += f" · {row['status'].replace('_', ' ')}"
            if row["reason"] is not None:
                label += f" ({row['reason'].replace('_', ' ')})"
            c.emit(c.field("ollama ceiling", label))
    c.emit()
    return 0


def render_model_detail(c: Console, model_id: str, *, as_json: bool = False,
                        engine: str | None = None) -> int:
    """Detail for one model: architecture (from its HF config) + its safe ceiling here."""
    if engine == "ollama":
        return _render_model_detail_ollama(c, model_id, as_json=as_json)
    meta = catalog.describe(model_id)
    if meta is None:
        if as_json:
            print(json.dumps({"error": f"couldn't describe {model_id}"}))
        else:
            c.emit(c.style("warn", f"  couldn't describe {model_id} — is it downloaded / a valid repo?"))
        return 1
    evidence_model = scoring.durable_model_id(model_id)
    mk = profile.machine_key()
    # Per-engine: a model can be characterized under several engines on one machine (GPU + CPU).
    per_engine = {}  # engine_key -> (safe_context, decode_context, measured_at, config, artifact_id)
    with ExitStack() as stack:
        scratch = sqlite3.connect(":memory:")
        scratch.row_factory = sqlite3.Row
        scratch.executescript(db.SCHEMA)
        stack.callback(scratch.close)
        con = (stack.enter_context(db.connected_readonly())
               if db._db_path().is_file() else scratch)
        for key in engines.ENGINES:
            row = db.get_characterization(con, mk, key, evidence_model)
            if row is not None:
                per_engine[key] = (row["safe_context"], row.get("decode_context"),
                                   row.get("measured_at"), row.get("config"),
                                   row.get("artifact_id"))
    # Best (largest) ceiling, carrying its decode_context AND measured_at so the top-level scalars
    # and the staleness flag all describe the SAME engine — not independent max() picks.
    best_row = max((row for row in per_engine.values() if row[0] is not None),
                   key=lambda row: row[0], default=None)
    best = best_row[0] if best_row else None
    best_decode = best_row[1] if best_row else None
    # Rule #3: a stored ceiling whose cache changed since it was measured isn't authoritative —
    # flag it here just as serve/run do, so no command shows a stale number unqualified.
    best_stale = (best_row is not None
                  and not staleness.artifact_matches(evidence_model, best_row[4]))
    if as_json:
        print(json.dumps({"model_id": model_id, **meta, "safe_context": best,
                          "decode_context": best_decode, "stale_ceiling": best_stale,
                          "engines": {k: sc for k, (sc, _, _, _, _) in per_engine.items()},
                          "engine_configs": {k: cfg for k, (_, _, _, cfg, _) in per_engine.items()},
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
        for key, (sc, dc, at, config, artifact_id) in per_engine.items():
            ceiling_str = f"~{sc} tokens" if sc else "no safe ceiling"
            if sc and dc and dc > sc:
                ceiling_str += f"  · ~{dc} stream-only (est.)"
            if sc and not staleness.artifact_matches(evidence_model, artifact_id):
                ceiling_str += "  · ⚠ stale — re-characterize"
            if config is None:
                ceiling_str += "  · settings unknown — re-characterize"
            elif config:
                ceiling_str += "  · " + _measurement_config_text(config)
            if c.verbose:
                ceiling_str += f"  · measured {at or 'unknown'}"
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
    on the AMD iGPU (`vulkan`), Apple (`mlx`), and NVIDIA (`cuda`) lanes. Flash-attention has
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


def _measurement_config(backend: str, *, flash_attn: bool = True,
                        flash_attn_optin: bool = False, kv_quant: str = "f16",
                        weight_quant: str = "none",
                        prefill_chunk: int | None = None) -> dict:
    """Canonical non-default settings that materially define a measured ceiling."""
    config = {}
    if backend in {"apple", "cuda", "vulkan"} and kv_quant != "f16":
        config["kv_quant"] = kv_quant
    if backend == "cuda":
        if flash_attn_optin:
            config["flash_attn"] = True
        if weight_quant != "none":
            config["weight_quant"] = weight_quant
        if prefill_chunk is not None:
            config["prefill_chunk"] = prefill_chunk
    elif backend == "vulkan" and not flash_attn:
        config["flash_attn"] = False
    return config


def _effective_measurement_config(bk, backend: str, *, flash_attn: bool = True,
                                  flash_attn_optin: bool = False,
                                  kv_quant: str = "f16", weight_quant: str = "none",
                                  prefill_chunk: int | None = None) -> dict:
    """Normalize requested settings to the allocation path the backend will actually use."""
    effective_flash = flash_attn_optin
    if (backend == "cuda" and effective_flash and hasattr(bk, "flash_attn_capable")
            and not bk.flash_attn_capable()):
        effective_flash = False
    return _measurement_config(
        backend, flash_attn=flash_attn, flash_attn_optin=effective_flash,
        kv_quant=kv_quant, weight_quant=weight_quant, prefill_chunk=prefill_chunk)


def _measurement_config_error(row: dict, expected: dict, backend: str,
                              model: str) -> str | None:
    """Refuse a ceiling whose memory-affecting settings do not match this operation."""
    if "config" not in row:  # lightweight test/dynamic callers represent the default this way
        actual = {}
    else:
        actual = row["config"]
    if actual is None:
        if _ENGINE_LEVERS.get(backend):
            return (f"the measured ceiling for {model} predates engine-setting tracking — "
                    f"re-run: ara characterize {model}")
        return None
    if actual != expected:
        return (f"the measured ceiling for {model} used different engine settings "
                f"({actual or 'defaults'}; this operation uses {expected or 'defaults'}) — "
                f"re-run ara characterize with matching settings")
    return None


def _measurement_config_text(config: dict) -> str:
    """Compact, stable disclosure for non-default measurement settings."""
    parts = []
    for key, value in sorted(config.items()):
        flag = key.replace("_", "-")
        rendered = str(value).lower() if isinstance(value, bool) else value
        parts.append(f"{flag}={rendered}")
    return "settings " + ", ".join(parts)


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


def _prefetch_plan(c: Console, model: str, bk, engine_key: str | None,
                   *, as_json: bool,
                   authorized_artifact_id: str | None = None) -> tuple[bool, object | None, int | None]:
    """Run deterministic cache/compatibility/disk gates before live work is claimed."""
    incompatible = engines.engine_for_model(model) not in (None, engine_key)
    if incompatible:
        return False, None, None
    authorized_ref = None
    if authorized_artifact_id is not None:
        if staleness.artifact_matches(model, authorized_artifact_id):
            return False, None, None
        authorized_ref = staleness.authorized_download_ref(model, authorized_artifact_id)
        if authorized_ref is None:
            msg = f"cannot recover the exact authorized artifact for {model}"
            print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
            return False, None, 1
        if not callable(getattr(bk, "download_prepared_model", None)):
            msg = f"the {engine_key} engine cannot recover an exact authorized artifact"
            print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
            return False, None, 1
    cached = getattr(bk, "calibration_model_cached", None)
    if authorized_ref is None and (cached is None or cached(model)):
        return False, None, None
    prepare = getattr(bk, "prepare_download", None)
    try:
        if authorized_ref is not None:
            selector, revision = authorized_ref
            payload = acquire.prepare_download(
                selector, gguf=engine_key in {"cpu", "vulkan", "cuda-gguf"},
                revision=revision)
        else:
            payload = (prepare(model) if prepare is not None else
                       acquire.gguf_size_gb(model)
                       if engine_key in {"cpu", "vulkan", "cuda-gguf"}
                       else acquire.repo_size_gb(model))
    except Exception as exc:
        msg = _fetch_error_msg(model, acquire.classify_repo_error(exc))
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return False, None, 1
    size_gb = payload.size_gb if isinstance(payload, acquire.AcquisitionPlan) else payload
    free_gb = acquire.free_disk_gb()
    if isinstance(payload, acquire.AcquisitionPlan) and free_gb is None:
        msg = f"cannot verify free disk space for downloading {model} — download refused"
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return False, payload, 1
    if (size_gb is not None and free_gb is not None
            and free_gb < size_gb + acquire.DISK_BUFFER_GB):
        msg = (f"not enough disk for {model}: needs ~{size_gb:.1f} GB "
               f"for the download + "
               f"{acquire.DISK_BUFFER_GB:.0f} GB headroom, only {free_gb:.1f} GB free.")
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return False, payload, 1
    return True, payload, None


def _artifact_identity_for_plan(model: str, payload: object | None, *,
                                expected: str | None = None) -> str | None:
    if isinstance(payload, acquire.AcquisitionPlan):
        return staleness.artifact_identity(model, revision=payload.revision)
    current = staleness.artifact_identity(model)
    if current is None and expected and staleness.artifact_matches(model, expected):
        return expected
    return current


def _pinned_model_for_plan(model: str, artifact_id: str, payload: object | None):
    """Bind the verified plan authority to the exact cache path an engine will load."""
    pinned = (staleness.pinned_model_ref(model, artifact_id, revision=payload.revision)
              if isinstance(payload, acquire.AcquisitionPlan)
              else staleness.pinned_model_ref(model, artifact_id))
    if pinned is None:
        raise RuntimeError(f"cannot pin the authorized artifact for {model}")
    return nullcontext(pinned)


def _download_prefetched_weights(c: Console, model: str, bk, payload: object | None,
                                 *, as_json: bool, progress: bool) -> int | None:
    """Perform the actual HF download after the caller has started live tracking."""
    size_gb = payload.size_gb if isinstance(payload, acquire.AcquisitionPlan) else payload
    _hf_hint(c, as_json)        # nudge to `ara hf login` before the (visible) HF rate-limit warning
    if not as_json:
        c.emit(c.style("dim", f"  downloading {model} … ({_fmt_size(size_gb)})"))
    try:
        prepared = getattr(bk, "download_prepared_model", None)
        if isinstance(payload, acquire.AcquisitionPlan) and prepared is not None:
            prepared(payload, progress=progress)
        else:
            bk.download_calibration_model(model, progress=progress)
    except Exception as exc:
        msg = _fetch_error_msg(model, acquire.classify_repo_error(exc))
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1
    return None


def _prefetch_weights(c: Console, model: str, bk, engine_key: str | None,
                      *, as_json: bool, progress: bool,
                      authorized_artifact_id: str | None = None) -> int | None:
    """Ensure a transformers/MLX model's weights are in the HF cache before the engine runs.

    CUDA/MLX fetch full transformer snapshots; GGUF engines fetch only the selected quant before
    authority is pinned. Without prefetch the worker's ``blobs/`` scan also yields
    ``weights_gb≈0`` for uncached transformers models, under-predicting the a-priori memory gate.
    No-op when the model's engine doesn't match *engine_key* or the exact selection is cached.
    Returns 1 (after printing) on a disk-space or fetch error, else None.
    """
    needed, size_gb, rc = _prefetch_plan(
        c, model, bk, engine_key, as_json=as_json,
        authorized_artifact_id=authorized_artifact_id)
    if rc is not None or not needed:
        return rc
    return _download_prefetched_weights(
        c, model, bk, size_gb, as_json=as_json, progress=progress)


def render_characterize(c: Console, model: str, *, engine: str | None = None,
                        as_json: bool = False, flash_attn: bool = True,
                        flash_attn_optin: bool = False, kv_quant: str = "f16",
                        weight_quant: str = "none", prefill_chunk: int | None = None) -> int:
    """Measure a model's safe context ceiling on an engine, and store it.

    Defaults to the detected engine; ``--engine`` overrides it so you can target a non-detected
    backend (e.g. the CPU fallback on a GPU box). ARA owns the result, so it shows up in
    `ara models show` regardless of which engine measured it.

    ``--engine ollama`` routes to a dedicated residency-ramp path (Slice 2): Ollama isn't a registry
    engine, and its model names (``qwen3:0.6b``) aren't HF refs — so it branches before both
    ``resolve_engine`` and ``valid_model_ref``. Spec 2026-07-04-characterize-through-ollama-ramp."""
    if kv_quant not in _KV_QUANT_CHOICES:
        msg = _kv_quant_error(kv_quant)
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1
    if weight_quant not in _WEIGHT_QUANT_CHOICES:
        msg = f"invalid --weight-quant {weight_quant!r} — choose one of: {', '.join(_WEIGHT_QUANT_CHOICES)}"
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1
    if not flash_attn and flash_attn_optin:
        msg = "--flash-attn and --no-flash-attn cannot be used together"
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1
    if engine == "ollama":
        lever_err = _unsupported_lever_error(
            "ollama", kv_quant=kv_quant, flash_attn=flash_attn,
            flash_attn_optin=flash_attn_optin, weight_quant=weight_quant,
            prefill_chunk=prefill_chunk)
        if lever_err is not None:
            print(json.dumps({"error": lever_err})) if as_json else c.emit(
                c.style("bad", f"  {lever_err}"))
            return 1
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
    evidence_model = scoring.durable_model_id(model)
    lever_err = _unsupported_lever_error(sel.backend, kv_quant=kv_quant, flash_attn=flash_attn,
                                         flash_attn_optin=flash_attn_optin, weight_quant=weight_quant,
                                         prefill_chunk=prefill_chunk)
    if lever_err is not None:
        print(json.dumps({"error": lever_err})) if as_json else c.emit(c.style("bad", f"  {lever_err}"))
        return 1
    engine_ok, engine_label = engine_status(sel.backend)
    if not engine_ok:
        if as_json:
            print(json.dumps({"error": f"{engine_label} not installed"}))
        else:
            c.emit(c.style("warn", f"  the {engine_label} isn't installed — run: ")
                   + c.style("accent", f"ara install --engine {sel.engine_key}"))
        return 1
    bk = get_backend(sel.backend)
    hw_err = _weight_quant_hw_error(bk, sel.backend, weight_quant)
    if hw_err is not None:
        print(json.dumps({"error": hw_err})) if as_json else c.emit(c.style("bad", f"  {hw_err}"))
        return 1
    measured_config = _effective_measurement_config(
        bk, sel.backend, flash_attn=flash_attn, flash_attn_optin=flash_attn_optin,
        kv_quant=kv_quant, weight_quant=weight_quant, prefill_chunk=prefill_chunk)
    if c.verbose and not as_json:
        c.emit(c.field("engine", sel.engine_key, engine_label))
        c.emit(c.field("KV cache", kv_quant))
    progress = (not as_json) and sys.stderr.isatty()
    # Deterministic pre-fetch gates run before ARA claims live work. The actual network download
    # stays in the same lifecycle record as calibration and measurement.
    prefetch, prefetch_size, rc = _prefetch_plan(
        c, model, bk, sel.engine_key, as_json=as_json)
    if rc is not None:
        return rc
    # characterize owns calibration: measure + persist the engine baseline once (when none is
    # stored) so the ramp uses the real overhead, not the default. Spec 2026-06-23-capability-pipeline.
    calibration_error = None
    with activity.track("characterizing", model):
        if prefetch and (rc := _download_prefetched_weights(
                c, model, bk, prefetch_size,
                as_json=as_json, progress=progress)) is not None:
            return rc
        with db.connected() as cal_con:
            if hasattr(bk, "calibrate") and calibration.get_calibration(cal_con, sel.engine_key) is None:
                if not as_json:
                    c.emit(c.style("dim", f"  calibrating {sel.engine_key} … (first run on this machine)"))
                cal = bk.calibrate()
                overhead = (cal or {}).get("overhead_gb")
                wall = (cal or {}).get("wall_gb")
                # Honesty (Rule #3): if calibration couldn't run (model missing, worker error), say so —
                # never let the conservative default masquerade as a measurement. The ramp still proceeds
                # safely on the default overhead; we just don't hide that it's a fallback.
                cal_err = (cal or {}).get("calibration_error")
                calibration_error = cal_err
                if cal_err and not as_json:
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
                if wall is not None and not as_json:
                    budget = (cal or {}).get("safe_budget_gb")
                    line = c.field("measured wall", _fmt_gb(wall, 1), label_width=15)
                    if budget is not None:
                        line += "  · " + c.style("dim", f"safe budget {_fmt_gb(budget, 1)}")
                    c.emit(line)
        if not as_json:
            c.emit(c.style("dim", f"  characterizing {model} … (loads the model on the device)"))
        fa_kw = _kv_fa_kwargs(sel.backend, flash_attn=flash_attn, flash_attn_optin=flash_attn_optin,
                              kv_quant=kv_quant, weight_quant=weight_quant, prefill_chunk=prefill_chunk)
        _flash_sdpa_note(c, bk, sel.backend, flash_attn_optin, as_json)
        artifact_id_before = _artifact_identity_for_plan(evidence_model, prefetch_size)
        if artifact_id_before is None:
            msg = f"cannot identify the exact artifact characterized for {model} — result not stored"
            print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
            return 1
        try:
            with _pinned_model_for_plan(
                    evidence_model, artifact_id_before, prefetch_size) as pinned_model:
                result = bk.characterize(pinned_model, progress=progress, **fa_kw)
        except (SystemExit, Exception) as exc:   # engine may refuse/abort/OOM-guard
            msg = f"characterization failed: {exc}"
            # Rule #3 (Honesty): under --json a consumer parses stdout — emit a structured error, never
            # styled text or a traceback that would break the parse.
            if as_json:
                payload = {"error": msg}
                if calibration_error:
                    payload.update(calibration_error=calibration_error, calibration_fallback=True)
                print(json.dumps(payload))
            else:
                c.emit(c.style("bad", f"  {msg}"))
            return 1

    # An engine that couldn't even load the model returns an `error` (not a measurement) — don't
    # persist a misleading null row. Suggest a compatible engine when we can tell cheaply (e.g. a
    # GGUF handed to the torch-based CUDA engine → suggest the CPU/llama.cpp engine).
    if result.get("error"):
        suggest = engines.engine_for_model(model)
        hint = ("  — try " + c.style("accent", f"ara characterize {model} --engine {suggest}")
                if suggest and suggest != sel.engine_key else "")
        if as_json:
            payload = {"error": result["error"]}
            if calibration_error:
                payload.update(calibration_error=calibration_error, calibration_fallback=True)
            print(json.dumps(payload))
        else:
            c.emit(c.style("warn", f"  {engine_label} couldn't load {model}: {result['error']}") + hint)
        return 1

    ceiling = result["safe_context"]
    artifact_id = _artifact_identity_for_plan(evidence_model, prefetch_size)
    if artifact_id != artifact_id_before:
        msg = f"the artifact for {model} changed during characterization — result not stored"
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1
    engine_report = engine_audit.audit_engine(
        sel.engine_key, host_features=detect._cpu_features())
    engine_evidence = engine_audit.characterization_evidence(engine_report)
    with db.connected() as con:
        db.save_characterization(con, profile.machine_key(), sel.engine_key,
                                 evidence_model, safe_context=ceiling, points=result["points"],
                                 decode_context=result.get("decode_context"),
                                 config=measured_config, artifact_id=artifact_id,
                                 characterization_evidence=engine_evidence)
        canonical_model_id = scoring.canonical_model_id(evidence_model)
        if evidence_model != canonical_model_id:
            catalog.remember_variant(
                con, evidence_model, canonical_model_id, quant=scoring.quant_key(evidence_model),
                weights_gb=staleness.artifact_size_gb(evidence_model))
        else:
            catalog.remember(con, evidence_model)

    if as_json:
        out: dict = {"model": model, "engine": sel.engine_key, "safe_context": ceiling,
                     "config": measured_config,
                     "decode_context": result.get("decode_context")}
        if calibration_error:
            out.update(calibration_error=calibration_error, calibration_fallback=True)
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
               + c.style("dim", f"  · {sel.engine_key} · stored (see ara models show {model})"))
        dc = result.get("decode_context")
        if dc and dc > ceiling:
            c.emit(c.style("good", f"  decode ceiling (est.)  ~{dc} tokens")
                   + c.style("dim", "  · grow-by-streaming, not a prompt size"))
        c.emit(c.style("dim", "  next: ")
               + c.style("accent", f'ara run {model} "Explain local AI simply"'))
    else:
        base = result.get("base_gb")
        budget = result.get("budget_gb")
        if base is not None and budget is not None:
            c.emit(c.style("warn",
                           f"  couldn't fit a ceiling on {sel.engine_key} — estimated base {base:.2f} GiB"
                           f" already near {budget:.1f} GiB safe budget"))
        else:
            c.emit(c.style("warn", f"  couldn't fit a ceiling on {sel.engine_key} — "
                                    "the model may be too big or borderline"))
        recovery = {
            "mlx": "a smaller or more heavily pre-quantized MLX model",
            "cuda": "--weight-quant int4 or int8, or a smaller model",
            "cpu": "a smaller or more heavily quantized GGUF model",
            "vulkan": "a smaller or more heavily quantized GGUF model",
            "cuda-gguf": "a smaller or more heavily quantized GGUF model",
        }
        c.emit(c.style("dim", f"  try: {recovery[sel.engine_key]}"))
    c.emit()
    return 0


def render_search(c: Console, query: str, *, as_json: bool = False) -> int:
    """Search the Hugging Face Hub for models (engine-agnostic)."""
    with activity.track("searching"):
        results = hub.search(query)
    if results is None:
        msg = ("couldn't search — check your connection and the hf command; in a source checkout, "
               "repair ARA with `uv sync --frozen`, then `uv run ara models search`")
        if as_json:
            print(json.dumps({"error": msg}))
        else:
            c.emit(c.style("warn", f"  {msg}"))
        return 1
    if as_json:
        print(json.dumps(results, indent=2))
        return 0
    c.emit()
    c.emit(c.section(f"  HUB SEARCH: {query}"))
    if c.verbose:
        c.emit(c.field("source", "hf models list", "sorted by downloads · limit 20"))
    for r in results:
        c.emit("  " + c.style("metric", r["id"])
               + c.style("dim", f"  ↓{r['downloads']} · ♥{r['likes']}"))
    if not results:
        c.emit(c.style("dim", "  no models found"))
    _hf_hint(c, as_json)
    c.emit()
    return 0


def _best_ceilings(
        con) -> dict[str, tuple[int | None, str, int | None, dict | None, str | None]]:
    """Best safe-context per model with engine, decode/config, and artifact authority.

    A model can be characterized under several engines on one machine (GPU + CPU); ``ara models show``
    shows the largest ceiling and which engine reached it. A real ceiling beats a null
    (measured-but-unfit) one; ties favour the detected default engine (considered first)."""
    mk = profile.machine_key()
    default = engines.for_backend(detect.backend_name())
    best: dict[str, tuple[int | None, str, int | None, dict | None, str | None]] = {}
    for key in dict.fromkeys([default, *engines.ENGINES]):
        if key is None:
            continue
        for r in db.list_characterizations(con, mk, key):
            mid, sc = r["model_id"], r["safe_context"]
            cur = best.get(mid)
            if cur is None or (sc is not None and (cur[0] is None or sc > cur[0])):
                best[mid] = (sc, key, r.get("decode_context"), r.get("config"),
                             r.get("artifact_id"))
    return best


def render_models(c: Console, *, as_json: bool = False, want=None) -> None:
    """Read-only cached-model inventory, enriched with any already-stored safe ceilings."""
    cache_con = sqlite3.connect(":memory:")
    cache_con.row_factory = sqlite3.Row
    try:
        cache_con.executescript(db.SCHEMA)
        catalog.scan(cache_con)
        models = catalog.all_models(cache_con)
    finally:
        cache_con.close()

    best: dict[str, tuple[int | None, str, int | None, dict | None, str | None]] = {}
    if db._db_path().is_file():
        with db.connected_readonly() as stored:
            best = _best_ceilings(stored)
            # A cache scan discovers repos, not exact repo:file GGUF variants or loose local files.
            # Merge only durable variants that are still physically present so their exact
            # characterization remains inspectable without resurrecting deleted artifacts.
            present = {model["model_id"] for model in models}
            for durable in catalog.all_models(stored):
                model_id = durable["model_id"]
                is_variant = ":" in model_id or model_id.lower().endswith(".gguf")
                if (is_variant and model_id not in present
                        and staleness.artifact_identity(model_id) is not None):
                    models.append(durable)
                    present.add(model_id)

    if as_json:
        print(json.dumps(
            [{**m,
              "safe_context": best[m["model_id"]][0] if m["model_id"] in best else None,
              "engine": best[m["model_id"]][1] if m["model_id"] in best else None,
              "decode_context": best[m["model_id"]][2] if m["model_id"] in best else None,
              "config": best[m["model_id"]][3] if m["model_id"] in best else None,
              "stale_ceiling": (not staleness.artifact_matches(
                  m["model_id"], best[m["model_id"]][4])
                  if m["model_id"] in best and best[m["model_id"]][0] is not None else False),
              "characterized": m["model_id"] in best} for m in models], indent=2))
        return

    c.emit()
    c.emit(c.section("  MODEL CATALOG"))
    for m in models:
        mid = m["model_id"]
        if mid in best:                           # measured under at least one engine
            ceiling, ekey, decode, config, artifact_id = best[mid]
            tail = f"~{ceiling} tokens ({ekey})" if ceiling else "no safe ceiling"
            if ceiling and decode and decode > ceiling:
                tail = f"~{ceiling} tokens ({ekey}) · ~{decode} stream-only (est.)"
            if config is None:
                tail += " · settings unknown"
            elif config:
                tail += " · " + _measurement_config_text(config)
            if ceiling and not staleness.artifact_matches(mid, artifact_id):
                tail += " · ⚠ stale — re-characterize"
            role = "good" if ceiling else "dim"   # measured-but-unfit mirrors profile's '—'
        else:
            tail, role = "not characterized", "dim"
        c.emit("  " + c.style("metric", mid)
               + c.style("dim", f"  {m['modality'] or '?'}  →  ")
               + c.style(role, tail))
    if not models:
        _emit_first_model_path(c)
    c.emit()
    n_char = sum(1 for m in models if m["model_id"] in best)
    c.emit(c.style("dim", f"  {len(models)} cataloged · {n_char} characterized on this machine"))
    c.emit()


def render_recommend(c: Console, *, as_json: bool = False, use_case: str | None = None,
                     engine: str | None = None) -> int:
    """Analytic recommendations — which cataloged models fit this machine, ranked by the context
    the estimated budget supports (most first), marking those already characterized here.

    Engine-free: reuses ``estimate.limits``/``model_fit`` (profile's math, anti-silo) over the
    catalog (which records each model's on-disk weight). No engine, no model load. Only models
    with a rankable context estimate are listed. With ``use_case``, ranks by a capability score —
    *measured here* (a local benchmark on the actual quant) or *imported* (a published number,
    labelled), never a guess (Rule #3). Stays engine-free either way. Spec
    2026-06-23-capability-pipeline + 2026-06-28-recommend-use-case-and-serve-selection."""
    if engine == "ollama":
        return _render_recommend_ollama(c, as_json=as_json, use_case=use_case)
    if use_case is not None and use_case not in scoring.USE_CASES:
        msg = (f"unknown use-case {use_case!r} — choose one of: "
               f"{', '.join(scoring.USE_CASES)}")
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1
    with ExitStack() as stack:
        cache_con = sqlite3.connect(":memory:")
        cache_con.row_factory = sqlite3.Row
        cache_con.executescript(db.SCHEMA)
        stack.callback(cache_con.close)
        catalog.scan(cache_con)            # ephemeral snapshot of the local cache
        evidence_con = (stack.enter_context(db.connected_readonly())
                        if db._db_path().is_file() else cache_con)
        for durable in catalog.all_models(evidence_con):
            model_id = durable["model_id"]
            if ((":" in model_id or model_id.lower().endswith(".gguf"))
                    and staleness.artifact_identity(model_id) is not None):
                db.upsert_model(
                    cache_con, model_id,
                    **{name: durable.get(name) for name in db._MODEL_COLS})
        # Prefer the measured wall for the detected engine (anti-silo: same grounding as profile).
        default_engine = engines.for_backend(detect.backend_name())
        measured = (calibration.get_calibration(evidence_con, default_engine)
                    if default_engine is not None else None)
        lim = estimate.limits(detect.machine(), measured=measured)
        best = _best_ceilings(evidence_con)  # model_id -> (safe_context, engine, decode, config)

        recs = []
        unrankable = 0                    # weights fit, but we can't read the arch to estimate context
        models = catalog.all_models(cache_con)
        for row in models:
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
            rows = db.list_benchmark_results(evidence_con, profile.machine_key())
            bench_measured = {}
            evidence_warnings = {}
            catalog_quants = {rec["model_id"]: rec["quant"] for rec in recs}
            for row in reversed(rows):
                evidence_key = (row["model_id"], row["use_case"])
                evidence, evidence_warning = scoring.validate_measured_evidence(row)
                if evidence is None:
                    evidence_warnings.setdefault(evidence_key, evidence_warning)
                    continue
                engine_spec = engines.ENGINES.get(row["engine_key"])
                if engine_spec is None or engine_spec.get("backend") != row["backend"]:
                    evidence_warnings.setdefault(
                        evidence_key, "invalid stored benchmark evidence")
                    continue
                if (row["model_id"] in catalog_quants
                        and row.get("quant") != catalog_quants[row["model_id"]]):
                    evidence_warnings.setdefault(
                        evidence_key, "invalid stored benchmark evidence")
                    continue
                if staleness.artifact_identity(row["model_id"]) != row["artifact_id"]:
                    evidence_warnings.setdefault(
                        evidence_key, "cached model changed since benchmark")
                    continue
                bench_measured.setdefault(evidence_key, evidence)
                evidence_warnings.pop(evidence_key, None)
            bench_measured = bench_measured or None
            recs = scoring.rank(recs, use_case, measured=bench_measured,
                                imported=scoring.load_imported())
            for rec in recs:
                rec["evidence_warning"] = evidence_warnings.get(
                    (rec["model_id"], use_case))
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
                                     "probe_context": r["score"].probe_context,
                                     "generation_cap": r["score"].generation_cap,
                                     "repeat_count": r["score"].repeat_count,
                                     "total_generations": r["score"].total_generations,
                                     "run_scores": r["score"].run_scores,
                                     "evidence_warning": r["score"].evidence_warning,
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
    if c.verbose:
        c.emit(c.field(
            "provenance",
            f"wall {lim['basis']} · {default_engine or 'unknown'} · "
            f"{_fmt_gb(lim['safe_budget_gb'], 1)} safe budget",
        ))
        noun = "model" if len(models) == 1 else "models"
        c.emit(c.field("catalog", f"{len(models)} cached {noun} · ephemeral read-only scan"))
    if not recs:
        if not models:
            _emit_first_model_path(c, default_engine or "cpu")
        else:
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
                        denominator = f"/{s.total_generations}" if s.total_generations else ""
                        if s.refused_n:
                            partial.append(f"{s.refused_n}{denominator} refused")
                        if s.errored_n:
                            partial.append(f"{s.errored_n}{denominator} errored")
                        head += f" [partial: {', '.join(partial)}]"
                    if s.sample_size is not None and s.sample_size < 100:
                        head += f" [low-confidence n={s.sample_size}]"
                    if s.inversion:
                        head += f" [quant-inversion: {s.inversion}]"
                    if c.verbose and s.sample_size is not None:
                        repeat_text = (f" × {s.repeat_count} runs"
                                       if s.repeat_count and s.repeat_count > 1 else "")
                        evidence = f"{s.sample_size} prompts{repeat_text}"
                        if s.probe_context is not None:
                            evidence += f"; ctx {s.probe_context}"
                        if s.generation_cap is not None:
                            evidence += f"; max {s.generation_cap}"
                        head += f" [evidence: {evidence}]"
            if r.get("evidence_warning"):
                head += f" [{r['evidence_warning']}]"
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


_OLLAMA_BENCHMARK_METRIC_NAMES = frozenset({
    "prompt_tokens", "completion_tokens", "total_duration_ns", "load_duration_ns",
    "prompt_eval_duration_ns", "eval_duration_ns",
})


def _ollama_backend_for_placement(placement: str) -> str:
    return {
        "cpu": "cpu",
        "unified": "apple",
        "accelerator": "cuda",
        "partial_offload": "cuda",
    }[placement]


def _ollama_benchmark_policy(record: ollama.OllamaModel, max_tokens: int) -> dict:
    thinking = "thinking" in record.capabilities
    return {
        "api": "/api/generate",
        "raw": False,
        "think": thinking,
        "template": "model_default",
        "truncate": False,
        "shift": False,
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": max_tokens,
        "workload_contract": (
            "thinking_completion_v1" if thinking else "completion_v1"),
    }


def _ollama_benchmark_target(
    authority: ollama.OllamaRuntimeAuthority,
    record: ollama.OllamaModel,
    config: dict,
    backend: str,
    artifact_id: str,
) -> dict:
    config_digest = hashlib.sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "runtime": "ollama",
        "model": record.name,
        "artifact_id": artifact_id,
        "endpoint_authority": authority.endpoint.url,
        "runtime_version": authority.server_version,
        "server_instance_id": authority.server_instance_id,
        "backend": backend,
        "placement": config.get("placement"),
        "config_sha256": config_digest,
    }


class _OllamaBenchmarkBackend:
    """Backend-shaped adapter whose execution stays inside ARA's Ollama governance boundary."""

    def __init__(self, endpoint, authority, record, safe, config, artifact_id):
        self.endpoint = endpoint
        self.authority = authority
        self.record = record
        self.safe = safe
        self.config = config
        self.artifact_id = artifact_id

    def benchmark(self, model, prompts, *, max_context, max_tokens=256):
        if model != self.record.name or max_context != self.safe:
            raise _OllamaGovernanceError("Ollama benchmark execution plan changed")
        with _governed_ollama_request(
                self.endpoint, self.authority, self.record, self.safe, self.config,
                self.artifact_id, activity_kind="benchmarking", label="benchmark"):
            return ollama.benchmark_prompts(
                self.record.name, prompts, self.safe, max_tokens,
                think="thinking" in self.record.capabilities)


def _ollama_benchmark_plan(model: str, ctx: int | None) -> tuple[dict | None, str | None]:
    """Resolve exact reusable Ollama evidence into the generic benchmark execution shape."""

    endpoint, endpoint_error = _ollama_governed_endpoint()
    if endpoint_error is not None:
        return None, endpoint_error
    try:
        activity.validate_ollama_serving_fields(
            served_name=_governed_name(model), model=model, context=1,
            endpoint=endpoint.url or ollama.base_url())
    except ValueError as exc:
        return None, f"invalid Ollama model identity: {exc}"
    authority, authority_error = _ollama_runtime_authority(endpoint)
    if authority_error is not None:
        return None, authority_error
    models = ollama.inventory()
    if models is None:
        return None, "couldn't list Ollama models — is the server reachable?"
    record = ollama.find_model(models, model)
    if record is None:
        return None, f"{model} isn't in Ollama — pull it with Ollama and characterize it first"
    if (support_error := ollama.initial_governed_model_error(record)) is not None:
        return None, support_error
    with db.connected() as con:
        assessment = ollama_evidence.assess_characterization(
            con, profile.machine_key(), record, authority)
    if assessment.reusable is None:
        reason = assessment.reason or "incomplete_evidence"
        return None, (f"{record.name}'s Ollama characterization is display-only ({reason}); "
                      "re-characterize it on this server before benchmark")
    row = assessment.reusable
    safe = row["safe_context"]
    if ctx is not None:
        if ctx <= 0:
            return None, "--ctx must be a positive integer"
        if msg := _ctx_gate_msg(ctx, safe, record.name):
            return None, msg
        safe = ctx
    config = row["config"]
    artifact_id = row["artifact_id"]
    placement = config["placement"]
    backend = _ollama_backend_for_placement(placement)
    return {
        "key": "ollama",
        "backend": backend,
        "bk": _OllamaBenchmarkBackend(
            endpoint, authority, record, safe, config, artifact_id),
        "row": row,
        "safe": safe,
        "ceiling_measured_at": row.get("measured_at"),
        "artifact_id": artifact_id,
        "evidence_model": record.name,
        "record": record,
        "authority": authority,
        "config": config,
    }, None


def _native_benchmark_plan(model: str, engine: str | None,
                           ctx: int | None) -> tuple[dict | None, str | None]:
    """Resolve one native-engine benchmark plan without loading weights."""

    auto_engine = engine is None or engine == "auto"
    if auto_engine:
        key = engines.for_backend(detect.backend_name()) or engines.for_hardware()
    else:
        key = engines.resolve(engine)
    if key is None:
        if engine is not None:
            return None, (f"benchmark doesn't support --engine {engine!r} — choose one of: "
                          "auto, ollama, mlx, cuda, cpu, vulkan, cuda-gguf")
        return None, "no benchmark-capable engine matches this machine"
    default_backend = engines.ENGINES.get(key, {}).get("backend") if key else None
    default_bk = get_backend(default_backend) if default_backend else None
    if default_bk is None or not hasattr(default_bk, "benchmark"):
        be_name = default_backend or "none"
        return None, f"benchmark isn't supported on the {be_name} engine"

    mk = profile.machine_key()
    evidence_model = scoring.durable_model_id(model)
    with db.connected() as con:
        if auto_engine:
            candidates = []
            config_errors = []
            unavailable = []
            artifact_mismatch = False
            missing_artifact_authority = False
            for candidate_key in dict.fromkeys([key, *engines.ENGINES]):
                candidate_backend = engines.ENGINES[candidate_key]["backend"]
                candidate_bk = get_backend(candidate_backend)
                if not hasattr(candidate_bk, "benchmark"):
                    continue
                expected_config = _measurement_config(candidate_backend)
                candidate_row = db.get_reusable_characterization_for_engine(
                    con, mk, candidate_key, evidence_model, config=expected_config)
                display_row = db.get_characterization(
                    con, mk, candidate_key, evidence_model)
                if candidate_row is None:
                    if display_row and (config_error := _measurement_config_error(
                            display_row, expected_config, candidate_backend, model)):
                        config_errors.append(config_error)
                    continue
                if candidate_row.get("safe_context") is None:
                    continue
                installed, label = engine_status(candidate_backend)
                if not installed:
                    unavailable.append((candidate_key, label))
                    continue
                if not candidate_row.get("artifact_id"):
                    missing_artifact_authority = True
                    continue
                if not staleness.artifact_matches(
                        evidence_model, candidate_row["artifact_id"]):
                    if staleness.authorized_download_ref(
                            evidence_model, candidate_row["artifact_id"]) is None:
                        artifact_mismatch = True
                        continue
                candidates.append((candidate_row["safe_context"], candidate_key,
                                   candidate_backend, candidate_bk, candidate_row))
            if not candidates:
                if config_errors:
                    return None, config_errors[0]
                if missing_artifact_authority:
                    return None, (f"the measured ceiling for {model} is not bound to an exact "
                                  f"artifact — re-run: ara characterize {model}")
                if artifact_mismatch:
                    return None, (f"the cached artifact for {model} differs from its measured "
                                  f"ceiling — re-run: ara characterize {model}")
                if unavailable:
                    unavailable_key, label = unavailable[0]
                    return None, (f"the {label} isn't installed — run: ara install "
                                  f"--engine {unavailable_key}")
                return None, f"no measured ceiling for {model} — run: ara characterize {model}"
            _, key, backend, bk, row = max(candidates, key=lambda candidate: candidate[0])
        else:
            backend, bk = default_backend, default_bk
            installed, label = engine_status(backend)
            if not installed:
                return None, f"the {label} isn't installed — run: ara install --engine {key}"
            expected_config = _measurement_config(backend)
            row = db.get_reusable_characterization_for_engine(
                con, mk, key, evidence_model, config=expected_config)
            display_row = db.get_characterization(con, mk, key, evidence_model)
            if row is None:
                if display_row is None:
                    return None, (f"no measured ceiling for {model} — run: "
                                  f"ara characterize {model}")
                if display_row and (msg := _measurement_config_error(
                        display_row, expected_config, backend, model)):
                    return None, msg
                return None, (f"no reusable measured ceiling for {model} — run: "
                              f"ara characterize {model}")
            if row.get("safe_context") is None:
                return None, f"no measured ceiling for {model} — run: ara characterize {model}"
        characterized_artifact_id = row.get("artifact_id")
        if not characterized_artifact_id:
            return None, (f"the measured ceiling for {model} is not bound to an exact artifact — "
                          f"re-run: ara characterize {model}")
        if ctx is not None:
            if ctx <= 0:
                return None, "--ctx must be a positive integer"
            if msg := _ctx_gate_msg(ctx, row["safe_context"], model):
                return None, msg
            safe = ctx
        else:
            safe = row["safe_context"]
    return {
        "key": key,
        "backend": backend,
        "bk": bk,
        "row": row,
        "safe": safe,
        "ceiling_measured_at": row.get("measured_at"),
        "artifact_id": characterized_artifact_id,
        "evidence_model": evidence_model,
    }, None


def render_benchmark(c: Console, model: str, *, use_case: str, engine: str | None = None,
                     ctx: int | None = None, max_tokens: int | None = None,
                     repeat: int = 1, assume_yes: bool = False,
                     exec_consent: bool = False, as_json: bool = False) -> int:
    """Run a capability probe set against *model* and store the score as a measured tier result.

    Requires a characterization ceiling; ``--ctx`` may only lower it. Native engines and an
    explicitly selected local Ollama runtime share the same judge-free scoring path."""
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

    ollama_mode = engine == "ollama"
    if ollama_mode:
        plan, plan_error = _ollama_benchmark_plan(model, ctx)
    else:
        plan, plan_error = _native_benchmark_plan(model, engine, ctx)
    if plan_error is not None:
        return err(plan_error)
    key = plan["key"]
    backend = plan["backend"]
    bk = plan["bk"]
    row = plan["row"]
    safe = plan["safe"]
    ceiling_measured_at = plan["ceiling_measured_at"]
    characterized_artifact_id = plan["artifact_id"]
    evidence_model = plan["evidence_model"]
    mk = profile.machine_key()

    stale_ceiling = (
        False
        if ollama_mode else
        _stale_ceiling_note(c, evidence_model, ceiling_measured_at, as_json=as_json)
    )
    items = benchmark.load_probe(use_case)
    n = len(items)
    if n == 0:
        return err(f"the {use_case} probe set is empty — no measurement taken")
    methodology_id = benchmark.methodology_id(use_case, items)
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
    effective_generation_cap = max_tokens if max_tokens is not None else 256
    ollama_policy = None
    if ollama_mode:
        ollama_policy = _ollama_benchmark_policy(
            plan["record"], effective_generation_cap)
    # Pre-fetch weights so CUDA/MLX benchmark uncached models on demand (like the GGUF engines),
    # instead of the worker refusing "model not found in HF cache" (#109).
    progress = (not as_json) and sys.stderr.isatty()
    # One record owns the complete operational lifecycle: an on-demand fetch and every repeat
    # backend call. All deterministic gates above run before ARA claims that work is live.
    if ollama_mode:
        prefetch, prefetch_size, rc = False, None, None
    else:
        prefetch, prefetch_size, rc = _prefetch_plan(
            c, model, bk, key, as_json=as_json,
            authorized_artifact_id=characterized_artifact_id)
    if rc is not None:
        return rc
    activity_scope = nullcontext() if ollama_mode else activity.track("benchmarking", model)
    with activity_scope:
        if prefetch and (rc := _download_prefetched_weights(
                c, model, bk, prefetch_size,
                as_json=as_json, progress=progress)) is not None:
            return rc
        artifact_id = (
            _ollama_artifact_id(evidence_model)
            if ollama_mode else
            _artifact_identity_for_plan(
                evidence_model, prefetch_size, expected=characterized_artifact_id)
        )
        if artifact_id != characterized_artifact_id:
            return err(f"the cached artifact for {model} differs from its measured ceiling — "
                       f"re-run: ara characterize {model}")
        # --repeat N: run the probe set N times (N separate model loads — acceptable v1). Never let a
        # single lucky roll stand in as THE number: score each run independently, store the MEAN as the
        # point estimate, and surface the LO–HI band so a wide spread is visible (pass^k spirit).
        run_scores: list[float] = []
        refused_n = 0
        errored_n = 0
        ollama_runtime_metrics: dict[str, int] = {}
        loaded_model_ref = evidence_model
        try:
            model_scope = (
                nullcontext(evidence_model)
                if ollama_mode else
                _pinned_model_for_plan(evidence_model, artifact_id, prefetch_size)
            )
            with model_scope as pinned_model:
                loaded_model_ref = pinned_model
                for _ in range(repeat):
                    try:
                        result = bk.benchmark(pinned_model, prompts, max_context=safe, **bench_kw)
                    except (SystemExit, Exception) as exc:
                        return err(f"benchmark failed: {exc}")
                    if not isinstance(result, dict):
                        return err("invalid benchmark result: the engine returned a non-object response")
                    reported_context = result.get("context")
                    if (not isinstance(reported_context, int) or isinstance(reported_context, bool)
                            or reported_context != safe):
                        return err(f"invalid benchmark result: the engine reported context "
                                       f"{reported_context!r}, expected {safe}")
                    if ollama_mode and result.get("request_policy") != ollama_policy:
                        return err("invalid benchmark result: Ollama request policy changed")
                    if result.get("refused"):
                        # A whole-run refusal on ANY run aborts — no partial band scraped from a failed load.
                        return err(f"the engine refused: {result.get('reason', 'no reason given')}")
                    results = result.get("results")
                    if not isinstance(results, list) or len(results) != len(prompts):
                        return err("invalid benchmark result: expected exactly one result per prompt")
                    completions = [""] * len(prompts)
                    seen: set[int] = set()
                    for r in results:
                        if not isinstance(r, dict):
                            return err("invalid benchmark result: each prompt result must be an object")
                        idx = r.get("prompt_index")
                        if (not isinstance(idx, int) or isinstance(idx, bool)
                                or not 0 <= idx < len(completions) or idx in seen):
                            return err("invalid benchmark result: prompt indexes must be unique and cover "
                                       "the probe set")
                        seen.add(idx)
                        outcomes = [name for name in ("completion", "refused", "error") if name in r]
                        if len(outcomes) != 1:
                            return err("invalid benchmark result: each prompt needs exactly one completion, "
                                       "refusal, or error")
                        outcome = outcomes[0]
                        if outcome == "completion":
                            if not isinstance(r["completion"], str):
                                return err("invalid benchmark result: completion must be text")
                            if ollama_mode:
                                usage = r.get("usage")
                                if (not isinstance(usage, dict)
                                        or not set(usage).issubset(
                                            _OLLAMA_BENCHMARK_METRIC_NAMES)
                                        or any(
                                        type(value) is not int or value < 0
                                        for value in usage.values())):
                                    return err("invalid benchmark result: Ollama usage metrics "
                                               "must be nonnegative integers")
                                for metric, value in usage.items():
                                    ollama_runtime_metrics[metric] = (
                                        ollama_runtime_metrics.get(metric, 0) + value)
                            completions[idx] = r["completion"]
                        elif outcome == "refused":
                            if r["refused"] is not True:
                                return err("invalid benchmark result: refused must be true")
                            refused_n += 1
                        else:
                            if not isinstance(r["error"], str):
                                return err("invalid benchmark result: error must be text")
                            errored_n += 1
                    run_scores.append(benchmark.score_probe_set(use_case, items, completions))
                    current_artifact_id = (
                        _ollama_artifact_id(evidence_model)
                        if ollama_mode else
                        _artifact_identity_for_plan(
                            evidence_model, prefetch_size, expected=artifact_id)
                    )
                    if current_artifact_id != artifact_id:
                        return err(f"the cached artifact for {model} changed during the benchmark — "
                                   "no measurement taken")
        except (SystemExit, Exception) as exc:
            return err(f"benchmark failed: {exc}")

    total = n * repeat                       # total generations attempted across every run
    if prompts and (refused_n + errored_n) == total:
        # No generation anywhere produced a completion (all refused by governance and/or errored
        # mid-generation) — NOT a 0% capability measurement; refuse to store a misleading score.
        return err("every prompt was refused or errored — no measurement taken")
    if refused_n and not as_json:
        c.emit(c.style("warn", f"  note: {refused_n}/{total} prompts were refused by "
                               f"governance and scored 0 — the result is depressed accordingly"))
    if errored_n and not as_json:
        c.emit(c.style("warn", f"  note: {errored_n}/{total} prompts errored (engine "
                               f"exception) and scored 0 — the result is depressed accordingly"))

    score = sum(run_scores) / repeat         # MEAN across runs — a better estimate than any one roll
    lo, hi = min(run_scores), max(run_scores)
    low_confidence = n < 100
    band_source = f" band={lo * 100:.0f}-{hi * 100:.0f}" if repeat > 1 else ""
    source = (f"{key} probe={n} ctx={safe} max_tokens={effective_generation_cap} "
              f"repeat={repeat}{band_source} ({model})")
    ollama_target = None
    if ollama_mode:
        ollama_target = _ollama_benchmark_target(
            plan["authority"], plan["record"], plan["config"], backend, artifact_id)
        config_digest = ollama_target["config_sha256"]
        policy_digest = hashlib.sha256(json.dumps(
            ollama_policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        source += (
            f" runtime={plan['authority'].server_version}"
            f" instance={plan['authority'].server_instance_id}"
            f" endpoint={plan['config']['endpoint_authority']}"
            f" placement={plan['config']['placement']}"
            " api=/api/generate template=model_default think="
            f"{'true' if 'thinking' in plan['record'].capabilities else 'false'}"
            " truncate=false shift=false temperature=0 seed=0"
            f" config_sha256={config_digest} policy_sha256={policy_digest}"
            " runtime_metrics="
            + json.dumps(ollama_runtime_metrics, sort_keys=True, separators=(",", ":"))
        )
    if low_confidence:
        source += f"; low_confidence n={n}"
    # Record the quant the score was actually taken at (the quant×capability degradation an
    # imported score hides): prefer the catalog's recorded quant, else derive it from the id.
    with db.connected() as con:
        if ollama_mode:
            canonical_model_id = evidence_model
            quant = plan["record"].quantization
            base_model = evidence_model
        else:
            canonical_model_id = scoring.canonical_model_id(evidence_model)
            mrow = db.get_model(con, evidence_model) or db.get_model(con, canonical_model_id)
            quant = (scoring.quant_key(evidence_model) or scoring.quant_key(loaded_model_ref)
                     or (mrow.get("quant") if mrow else None))
            if evidence_model != canonical_model_id:
                catalog.remember_variant(
                    con, evidence_model, canonical_model_id, quant=quant,
                    weights_gb=staleness.artifact_size_gb(evidence_model))
            base_model = scoring.base_key(canonical_model_id)
        db.save_benchmark_result(con, mk, evidence_model, use_case, score=score, source=source,
                                 engine_key=key, backend=backend,
                                 base_model=base_model, quant=quant,
                                 benchmark_id=use_case, max_score=1.0, sample_size=n,
                                 methodology_id=methodology_id,
                                 refused_n=refused_n, errored_n=errored_n,
                                 probe_context=safe, generation_cap=effective_generation_cap,
                                 repeat_count=repeat, total_generations=total,
                                 run_scores=run_scores, artifact_id=artifact_id,
                                 canonical_model_id=canonical_model_id,
                                 target=ollama_target, request_policy=ollama_policy,
                                 runtime_metrics=(ollama_runtime_metrics
                                                  if ollama_mode else None))
        con.commit()

    if as_json:
        payload: dict = {"model": model, "use_case": use_case, "score": score,
                         "sample_size": n, "engine": key, "stored": True}
        if ollama_mode:
            payload["concurrency_scope"] = (
                "ARA governs these requests; outside Ollama clients remain concurrent")
        if c.verbose:
            payload.update(
                backend=backend,
                probe_context=safe,
                generation_cap=effective_generation_cap,
                generation_cap_source=(
                    "explicit" if max_tokens is not None else
                    "ara_default" if ollama_mode else "backend_default"),
                total_generations=total,
                source=source,
            )
            if ollama_mode:
                payload.update(
                    request_policy=ollama_policy,
                    runtime_metrics=ollama_runtime_metrics,
                )
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
    if c.verbose:
        c.emit(c.field("engine", f"{key} ({backend})"))
        c.emit(c.field("probe context", f"{safe} tokens"))
        default_source = "ARA default" if ollama_mode else "backend default"
        generation_cap = (f"{max_tokens} tokens" if max_tokens is not None
                          else f"{effective_generation_cap} tokens ({default_source})")
        c.emit(c.field("generation cap", generation_cap))
        evidence = f"{n} prompts" if repeat == 1 else f"{n} prompts × {repeat} runs"
        c.emit(c.field("evidence", evidence))
        c.emit(c.field("quant", quant or "unknown"))
    if ollama_mode:
        c.emit(c.style(
            "warn", "  note: ARA governs these requests; outside Ollama clients remain concurrent."))
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
    if ollama_mode:
        c.emit(c.style("dim", "  stored as measured capability evidence"))
    else:
        c.emit(c.style("dim", f"  stored — ara models recommend --use-case {use_case} now shows it"))
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


class _OllamaGovernanceError(RuntimeError):
    """One failed invariant in a governed request on the shared Ollama daemon."""


@contextmanager
def _governed_ollama_request(
    endpoint: ollama.OllamaEndpoint,
    authority: ollama.OllamaRuntimeAuthority,
    record: ollama.OllamaModel,
    safe: int,
    config: dict,
    artifact_id: str,
    *,
    activity_kind: str,
    label: str,
):
    """Admit one Ollama operation, warming if needed, and re-attest after it returns."""

    expected_digest = artifact_id.removeprefix(_OLLAMA_ARTIFACT_PREFIX)
    with locking.ollama_setup_lock(endpoint.url, record.name):
        with activity.track(activity_kind, record.name):
            if ollama.runtime_authority(endpoint) != authority:
                raise _OllamaGovernanceError(
                    f"Ollama runtime authority changed before the {label}")
            current_models = ollama.inventory()
            current_record = (
                ollama.find_model(current_models, record.name)
                if current_models is not None else None)
            if current_record is None or current_record.digest != record.digest:
                raise _OllamaGovernanceError(
                    f"the Ollama manifest changed before the {label}")
            probe_target = _ollama_recorded_probe_target(
                config, current_models,
                model_name=record.name, base_artifact_id=artifact_id)
            alternate_targets = ((probe_target,) if probe_target is not None else ())

            processes = ollama.processes()
            residency_error = _ollama_residency_error(
                processes,
                allowed_name=record.name,
                allowed_digest=expected_digest,
                allowed_context=safe,
                alternate_targets=alternate_targets,
            )
            if residency_error is not None:
                raise _OllamaGovernanceError(residency_error)
            resident = bool(processes)
            before = ollama_evidence.capture_memory_snapshot()
            headroom_error = ollama_evidence.live_headroom_refusal_reason(
                before, config, resident=resident)
            if headroom_error is not None:
                raise _OllamaGovernanceError(
                    f"live Ollama admission refused: {headroom_error}")

            if resident:
                live_error = ollama_evidence.live_residency_refusal_reason(
                    before, processes[0], config, safe)
                if live_error is not None:
                    raise _OllamaGovernanceError(
                        "resident Ollama target does not match reusable evidence: "
                        f"{live_error}")

            if not resident:
                warmed = ollama.warm_for_run(record.name, safe)
                if not isinstance(warmed, dict) or warmed.get("done") is not True:
                    raise _OllamaGovernanceError(
                        "Ollama couldn't confirm the governed warm request")
                if ollama.runtime_authority(endpoint) != authority:
                    raise _OllamaGovernanceError(
                        "Ollama runtime authority changed during warmup")
                processes = ollama.processes()
                residency_error = _ollama_residency_error(
                    processes,
                    allowed_name=record.name,
                    allowed_digest=expected_digest,
                    allowed_context=safe,
                    alternate_targets=alternate_targets,
                    require_target=True,
                )
                if residency_error is not None:
                    raise _OllamaGovernanceError(residency_error)
                warm_snapshot = ollama_evidence.capture_memory_snapshot()
                live_error = ollama_evidence.live_residency_refusal_reason(
                    warm_snapshot, processes[0], config, safe)
                if live_error is not None:
                    raise _OllamaGovernanceError(
                        f"Ollama warmup did not match reusable evidence: {live_error}")

            yield

            if ollama.runtime_authority(endpoint) != authority:
                raise _OllamaGovernanceError(
                    f"Ollama runtime authority changed during the {label}; no result shown")
            final_models = ollama.inventory()
            final_record = (
                ollama.find_model(final_models, record.name)
                if final_models is not None else None)
            if final_record is None or final_record.digest != record.digest:
                raise _OllamaGovernanceError(
                    f"the Ollama manifest changed during the {label}; no result shown")
            final_probe_target = _ollama_recorded_probe_target(
                config, final_models,
                model_name=record.name, base_artifact_id=artifact_id)
            final_alternate_targets = (
                (final_probe_target,) if final_probe_target is not None else ())
            final_processes = ollama.processes()
            residency_error = _ollama_residency_error(
                final_processes,
                allowed_name=record.name,
                allowed_digest=expected_digest,
                allowed_context=safe,
                alternate_targets=final_alternate_targets,
                require_target=True,
            )
            if residency_error is not None:
                raise _OllamaGovernanceError(
                    f"Ollama residency changed during the {label}; no result shown: "
                    f"{residency_error}")
            final_snapshot = ollama_evidence.capture_memory_snapshot()
            live_error = ollama_evidence.live_residency_refusal_reason(
                final_snapshot, final_processes[0], config, safe)
            if live_error is not None:
                raise _OllamaGovernanceError(
                    f"Ollama walls changed during the {label}; no result shown: {live_error}")


def _render_run_ollama(
    c: Console,
    model: str,
    *,
    prompt: str | None,
    assume_yes: bool,
    as_json: bool,
    max_tokens: int,
    flash_attn: bool,
    flash_attn_optin: bool,
    kv_quant: str,
    weight_quant: str,
    prefill_chunk: int | None,
) -> int:
    """Run one buffered request through an already-authoritative local Ollama daemon."""

    def err(msg: str) -> int:
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1

    if not prompt or not prompt.strip():
        return err("usage: ara run <model> <prompt>")
    if max_tokens <= 0:
        return err("--max-tokens must be a positive integer")
    if (
        kv_quant != "f16"
        or weight_quant != "none"
        or prefill_chunk is not None
        or not flash_attn
        or flash_attn_optin
    ):
        return err("Ollama run doesn't accept native-engine tuning options; omit KV, weight, "
                   "prefill, and flash-attention flags so the measured daemon config remains "
                   "authoritative")

    endpoint, endpoint_error = _ollama_governed_endpoint()
    if endpoint_error is not None:
        return err(endpoint_error)
    try:
        activity.validate_ollama_serving_fields(
            served_name=_governed_name(model), model=model, context=1,
            endpoint=endpoint.url or ollama.base_url())
    except ValueError as exc:
        return err(f"invalid Ollama model identity: {exc}")

    authority, authority_error = _ollama_runtime_authority(endpoint)
    if authority_error is not None:
        return err(authority_error)
    models = ollama.inventory()
    if models is None:
        return err("couldn't list Ollama models — is the server reachable?")
    record = ollama.find_model(models, model)
    if record is None:
        return err(f"{model} isn't in Ollama — pull and characterize it first")
    if (support_error := ollama.initial_governed_model_error(record)) is not None:
        return err(support_error)

    with db.connected() as con:
        assessment = ollama_evidence.assess_characterization(
            con, profile.machine_key(), record, authority)
    if assessment.reusable is None:
        reason = assessment.reason or "incomplete_evidence"
        return err(f"{record.name}'s Ollama characterization is display-only ({reason}); "
                   f"re-characterize it on this server before run")
    row = assessment.reusable
    safe = row["safe_context"]
    config = row["config"]
    artifact_id = row["artifact_id"]

    if not as_json and not assume_yes and sys.stdin.isatty():
        if not _confirm(f"Run {record.name} through Ollama at ≤{safe} ctx?"):
            c.emit(c.style("dim", "  skipped."))
            return 0

    concurrency_scope = (
        "ARA governs this request; outside Ollama clients remain concurrent")
    if not as_json:
        c.emit(c.style("dim", f"  running {record.name} on Ollama … (≤ ~{safe} tokens)"))
        c.emit(c.style("warn", f"  note: {concurrency_scope}."))

    result = None
    try:
        with _governed_ollama_request(
                endpoint, authority, record, safe, config, artifact_id,
                activity_kind="running", label="run"):
            result = ollama.generate_for_run(
                record.name, prompt, safe, max_tokens)
    except locking.OllamaSetupBusy as exc:
        return err(str(exc))
    except _OllamaGovernanceError as exc:
        return err(str(exc))
    except (SystemExit, Exception) as exc:
        return err(f"Ollama run failed: {exc}")

    if not isinstance(result, dict) or result.get("done") is not True:
        return err("Ollama returned an incomplete completion")
    if isinstance(result.get("error"), str):
        return err(f"Ollama run failed: {result['error']}")
    completion = result.get("response")
    if not isinstance(completion, str):
        return err("Ollama returned an invalid completion")
    thinking = result.get("thinking")
    if thinking is not None and not isinstance(thinking, str):
        return err("Ollama returned invalid thinking metadata")
    stop_reason = result.get("done_reason")
    if stop_reason is not None and not isinstance(stop_reason, str):
        return err("Ollama returned an invalid stop reason")

    def usage_value(key: str) -> int | None:
        value = result.get(key)
        return value if type(value) is int and value >= 0 else None

    placement = config["placement"]
    backend = {
        "cpu": "cpu",
        "unified": "apple",
        "accelerator": "cuda",
        "partial_offload": "cuda",
    }[placement]
    payload = {
        "model": record.name,
        "engine": "ollama",
        "runtime": "ollama",
        "backend": backend,
        "placement": placement,
        "artifact_id": artifact_id,
        "safe_context": safe,
        "completion": completion,
        "stop_reason": stop_reason,
        "usage": {
            "prompt_tokens": usage_value("prompt_eval_count"),
            "completion_tokens": usage_value("eval_count"),
        },
        "evidence": {
            "methodology": config["methodology"],
            "runtime_version": config["runtime_version"],
            "server_instance_id": config["server_instance_id"],
        },
        "concurrency_scope": concurrency_scope,
    }
    if thinking is not None:
        payload["thinking"] = thinking
    if as_json:
        print(json.dumps(payload, indent=2))
        return 0
    c.emit()
    c.emit(completion)
    c.emit()
    return 0


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

    if engine == "ollama":
        return _render_run_ollama(
            c,
            model,
            prompt=prompt,
            assume_yes=assume_yes,
            as_json=as_json,
            max_tokens=max_tokens,
            flash_attn=flash_attn,
            flash_attn_optin=flash_attn_optin,
            kv_quant=kv_quant,
            weight_quant=weight_quant,
            prefill_chunk=prefill_chunk,
        )

    try:
        sel = resolve_engine(engine)
    except UnknownEngine:
        return err(f"unknown engine {engine!r} — try one of: {', '.join(engines.ENGINES)}")
    if not prompt or not prompt.strip():
        return err("usage: ara run <model> <prompt>")
    if max_tokens <= 0:
        return err("--max-tokens must be a positive integer")
    if not acquire.valid_model_ref(model):
        return err(f"invalid model {model!r} — expected a Hugging Face repo id (org/name) "
                   f"or a local .gguf file path")
    evidence_model = scoring.durable_model_id(model)
    if kv_quant not in _KV_QUANT_CHOICES:
        return err(_kv_quant_error(kv_quant))
    if weight_quant not in _WEIGHT_QUANT_CHOICES:
        return err(f"invalid --weight-quant {weight_quant!r} — choose one of: "
                   f"{', '.join(_WEIGHT_QUANT_CHOICES)}")

    mk = profile.machine_key()
    suffix = "" if engine is None else f" --engine {sel.engine_key}"
    requested_config = _effective_measurement_config(
        get_backend(sel.backend), sel.backend, flash_attn=flash_attn,
        flash_attn_optin=flash_attn_optin, kv_quant=kv_quant,
        weight_quant=weight_quant, prefill_chunk=prefill_chunk)

    with db.connected() as con:
        if engine is not None:
            # Pinned: use exactly the named engine — honour the explicit choice, don't second-guess it.
            row = db.get_reusable_characterization_for_engine(
                con, mk, sel.engine_key, evidence_model, config=requested_config)
            display_row = db.get_characterization(con, mk, sel.engine_key, evidence_model)
            if row is None and display_row is None:
                return err(f"{model} isn't characterized on {sel.engine_key} yet — run: "
                           f"ara characterize {model}{suffix}")
            if row is None:
                if msg := _measurement_config_error(
                        display_row, requested_config, sel.backend, model):
                    return err(msg)
                return err(f"the measured ceiling for {model} is not reusable — re-run: "
                           f"ara characterize {model}{suffix}")
            if row.get("safe_context") is None:
                return err(f"{model} was characterized but didn't fit on {sel.engine_key} — "
                           f"too big for this machine")
            engine_key, backend, safe = sel.engine_key, sel.backend, row["safe_context"]
            ceiling_measured_at = row.get("measured_at")
            characterized_artifact_id = row.get("artifact_id")
        else:
            # No --engine: scan every engine this model is characterized under on this machine and pick
            # the largest measured ceiling whose backend can actually run (has `generate`). A model
            # characterized on the CPU fallback runs there even when the detected backend differs.
            # Mirror _best_ceilings' iteration: detected default first so ties favour it. The default
            # is never None here — resolve_engine(None) above would have raised if the detected backend
            # had no engine — so [default, *ENGINES] holds only real keys.
            default = engines.for_backend(detect.backend_name())
            per_engine = {}                  # engine_key -> (safe_context, backend, can_run, time, row)
            for key in dict.fromkeys([default, *engines.ENGINES]):
                backend = engines.ENGINES[key]["backend"]
                candidate_config = _effective_measurement_config(
                    get_backend(backend), backend, flash_attn=flash_attn,
                    flash_attn_optin=flash_attn_optin, kv_quant=kv_quant,
                    weight_quant=weight_quant, prefill_chunk=prefill_chunk)
                row = db.get_reusable_characterization_for_engine(
                    con, mk, key, evidence_model, config=candidate_config)
                display_row = db.get_characterization(con, mk, key, evidence_model)
                if row is None and display_row is None:
                    continue
                selected = row if row is not None else display_row
                per_engine[key] = (selected.get("safe_context"), backend,
                                   hasattr(get_backend(backend), "generate"),
                                   selected.get("measured_at"), selected,
                                   row is not None)
            if not per_engine:
                return err(f"{model} isn't characterized on {sel.engine_key} yet — run: "
                           f"ara characterize {model}")
            fitted = {k: v for k, v in per_engine.items() if v[0] is not None}
            if not fitted:
                return err(f"{model} was characterized but didn't fit on {sel.engine_key} — "
                           f"too big for this machine")
            lever_supported = {
                k: v for k, v in fitted.items()
                if v[2] and _unsupported_lever_error(
                    v[1], kv_quant=kv_quant, flash_attn=flash_attn,
                    flash_attn_optin=flash_attn_optin, weight_quant=weight_quant,
                    prefill_chunk=prefill_chunk) is None
            }
            config_runnable = {
                k: v for k, v in lever_supported.items()
                if v[5] and _measurement_config_error(
                    v[4], _effective_measurement_config(
                        get_backend(v[1]), v[1], flash_attn=flash_attn,
                        flash_attn_optin=flash_attn_optin,
                        kv_quant=kv_quant, weight_quant=weight_quant,
                        prefill_chunk=prefill_chunk), v[1], model) is None
            }
            if not config_runnable:
                mismatches = [
                    _measurement_config_error(
                        v[4], _effective_measurement_config(
                            get_backend(v[1]), v[1], flash_attn=flash_attn,
                            flash_attn_optin=flash_attn_optin,
                            kv_quant=kv_quant, weight_quant=weight_quant,
                            prefill_chunk=prefill_chunk), v[1], model)
                    for v in lever_supported.values()
                ]
                if any(mismatches):
                    return err(next(msg for msg in mismatches if msg is not None))
                if any(not v[5] for v in lever_supported.values()):
                    return err(f"the measured ceiling for {model} is not reusable — re-run: "
                               f"ara characterize {model}")
                lever_errors = [
                    _unsupported_lever_error(
                        v[1], kv_quant=kv_quant, flash_attn=flash_attn,
                        flash_attn_optin=flash_attn_optin, weight_quant=weight_quant,
                        prefill_chunk=prefill_chunk)
                    for v in fitted.values() if v[2]
                ]
                if any(lever_errors):
                    return err(next(msg for msg in lever_errors if msg is not None))
                # Characterized + fits, but only on engine(s) ARA can't run through yet (apple/cuda).
                # Be honest about that — don't masquerade as uncharacterized.
                where = ", ".join(fitted)
                return err(f"{model} is characterized on {where}, but run isn't supported on "
                           f"that engine yet")
            runnable = {}
            missing_artifact_authority = False
            artifact_mismatch = False
            unavailable = []
            for candidate_key, candidate in config_runnable.items():
                candidate_artifact = candidate[4].get("artifact_id")
                if not candidate_artifact:
                    missing_artifact_authority = True
                    continue
                if not staleness.artifact_matches(evidence_model, candidate_artifact):
                    artifact_mismatch = True
                    continue
                installed, label = engine_status(candidate[1])
                if not installed:
                    unavailable.append((candidate_key, label))
                    continue
                runnable[candidate_key] = candidate
            if not runnable:
                if missing_artifact_authority:
                    return err(f"the measured ceiling for {model} is not bound to an exact "
                               f"artifact — re-run: ara characterize {model}")
                if artifact_mismatch:
                    return err(f"the artifact for {model} differs from its measured ceiling — "
                               f"re-run: ara characterize {model}")
                unavailable_key, label = unavailable[0]
                return err(f"the {label} isn't installed — run: ara install "
                           f"--engine {unavailable_key}")
            # Largest ceiling wins; the dict is detected-first, so a strict `>` lets ties favour it.
            engine_key = max(runnable, key=lambda k: runnable[k][0])
            safe, backend, _, ceiling_measured_at, selected_row, _reusable = runnable[engine_key]
            characterized_artifact_id = selected_row.get("artifact_id")

    stale_ceiling = _stale_ceiling_note(
        c, evidence_model, ceiling_measured_at, as_json=as_json)
    lever_err = _unsupported_lever_error(backend, kv_quant=kv_quant, flash_attn=flash_attn,
                                         flash_attn_optin=flash_attn_optin, weight_quant=weight_quant,
                                         prefill_chunk=prefill_chunk)
    if lever_err is not None:
        return err(lever_err)

    engine_ok, engine_label = engine_status(backend)
    if not engine_ok:
        return err(f"the {engine_label} isn't installed — run: ara install{suffix}")
    bk = get_backend(backend)
    if not hasattr(bk, "generate"):
        return err(f"run isn't supported on the {engine_label} yet")
    hw_err = _weight_quant_hw_error(bk, backend, weight_quant)
    if hw_err is not None:
        return err(hw_err)
    if not characterized_artifact_id:
        return err(f"the measured ceiling for {model} is not bound to an exact artifact — "
                   f"re-run: ara characterize {model}")

    # Consent before load (a courtesy — the ceiling already makes it wall-safe). Interactive only;
    # --yes or a non-tty (scripts/--json) proceed straight to the governed run.
    if not as_json and not assume_yes and sys.stdin.isatty():
        if not _confirm(f"Load {model} on {engine_label} and generate (≤ ~{safe} tokens)?"):
            c.emit(c.style("dim", "  skipped."))
            return 0

    if not as_json:
        c.emit(c.style("dim", f"  running {model} on {engine_label} … (≤ ~{safe} tokens)"))
    fa_kw = _kv_fa_kwargs(backend, flash_attn=flash_attn, flash_attn_optin=flash_attn_optin,
                          kv_quant=kv_quant, weight_quant=weight_quant, prefill_chunk=prefill_chunk)
    _flash_sdpa_note(c, bk, backend, flash_attn_optin, as_json)
    try:
        with activity.track("running", model):
            if not staleness.artifact_matches(evidence_model, characterized_artifact_id):
                return err(f"the artifact for {model} differs from its measured ceiling — "
                           f"re-run: ara characterize {model}")
            with _pinned_model_for_plan(
                    evidence_model, characterized_artifact_id, None) as pinned_model:
                result = bk.generate(
                    pinned_model, prompt, max_context=safe, max_tokens=max_tokens, **fa_kw)
            if not staleness.artifact_matches(evidence_model, characterized_artifact_id):
                return err(f"the artifact for {model} changed during the run — no result shown")
    except (SystemExit, Exception) as exc:        # engine may refuse/abort/OOM-guard
        return err(f"run failed: {exc}")
    if not isinstance(result, dict):
        return err("run failed: engine returned an invalid completion")
    reported_context = result.get("context")
    if (not isinstance(reported_context, int) or isinstance(reported_context, bool)
            or reported_context != safe):
        return err(f"run failed: the engine reported context {reported_context!r}, "
                   f"expected {safe}")
    if result.get("refused"):
        return err(f"the {engine_label} refused: {result.get('reason', 'no reason given')}")
    if result.get("error"):
        return err(f"run failed: {result['error']}")

    completion = result.get("completion")
    if not isinstance(completion, str):
        return err("run failed: engine returned an invalid completion")
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
# Ollama measurements are valid only for the exact Ollama manifest that was measured. Even another
# llama.cpp-class runtime can allocate differently, so cross-runtime ceilings never transfer.
_OLLAMA_ARTIFACT_PREFIX = "ollama-manifest-sha256:"
_OLLAMA_DERIVED_POLICY_VERSION = "ollama-derived-v2"


def _ollama_governed_endpoint() -> tuple[ollama.OllamaEndpoint, str | None]:
    """Return configured endpoint authority and any initial local-governance refusal."""
    endpoint = ollama.endpoint_authority()
    if endpoint.scope == "unknown":
        return endpoint, "Ollama's configured endpoint is invalid or ambiguous"
    if endpoint.scope != "loopback":
        return endpoint, ("governed Ollama operations require a local loopback endpoint; "
                          f"the configured endpoint is {endpoint.scope}")
    return endpoint, None


def _ollama_runtime_authority(
    endpoint: ollama.OllamaEndpoint,
) -> tuple[ollama.OllamaRuntimeAuthority, str | None]:
    """Return the attested runtime plus any reason it remains display-only."""
    authority = ollama.runtime_authority(endpoint)
    errors = {
        "listener_unattributed": (
            "ARA couldn't attribute the local Ollama listener; detection remains available, "
            "but governed operations require a directly owned Ollama process"),
        "listener_ambiguous": (
            "the local Ollama port has ambiguous process ownership; refusing governed work"),
        "listener_not_ollama": (
            "the loopback endpoint is owned by another process, not Ollama; refusing a proxy as "
            "runtime authority"),
        "server_unreachable": (
            "Ollama isn't serving on its directly attributed listener"),
        "process_environment_unavailable": (
            "ARA can identify the Ollama server, but cannot read its configured process inputs; "
            "detection remains available, governed operations do not"),
        "parallelism_unknown": (
            "Ollama's configured parallelism is unknown for this exact server version; "
            "detection remains available, governed operations do not"),
        "parallelism_not_one": (
            "initial governed Ollama support requires configured maximum parallelism of one"),
    }
    if authority.issue is not None:
        return authority, errors.get(
            authority.issue, "Ollama runtime authority is incomplete; refusing governed work")
    return authority, None


def _ollama_runtime_authority_error(endpoint: ollama.OllamaEndpoint) -> str | None:
    """Return why a loopback Ollama is display-only, or ``None`` when it is authoritative."""
    return _ollama_runtime_authority(endpoint)[1]


def _ollama_artifact_id(model: str, *, record: ollama.OllamaModel | None = None) -> str | None:
    """Manifest identity from a supplied inventory record, or a fresh snapshot when omitted."""
    digest = record.digest if record is not None else ollama.manifest_digest(model)
    return _OLLAMA_ARTIFACT_PREFIX + digest if digest else None


def _ollama_safe_ceiling(
    con,
    mk: str,
    record: ollama.OllamaModel,
    authority: ollama.OllamaRuntimeAuthority,
):
    """Return a measured ceiling only from the strict reusable-evidence assessment."""

    row = ollama_evidence.assess_characterization(
        con, mk, record, authority).reusable
    if row is None:
        return None
    return row["safe_context"], "measured", row.get("measured_at")


def _ollama_estimated_ceiling(model: str, *, record: ollama.OllamaModel | None = None):
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
    size = record.size_bytes if record is not None else ollama.size_bytes(model)
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


def _ollama_measure_ceiling(model: str, max_ctx: int, probe: str, *,
                            base_artifact_id: str | None = None,
                            provenance: dict | None = None,
                            authority: ollama.OllamaRuntimeAuthority | None = None,
                            model_size_bytes: int | None = None):
    """Ramp Ollama to the largest context proven safe against every physical memory wall.

    Each rung performs one bounded generated token with truncation and shifting disabled, then
    binds Ollama's effective context and placement evidence to before/after physical-wall
    snapshots. CPU, unified memory, full accelerator residency, and partial offload are evaluated
    according to their actual walls; unknown topology fails closed.
    """
    best, points = None, []
    previous_context, previous_digest = None, None
    for ctx in _ollama_ramp_contexts(max_ctx):
        before = ollama_evidence.capture_memory_snapshot()
        if model_size_bytes is not None:
            preflight_reason = ollama_evidence.preflight_refusal_reason(
                before, model_size_bytes)
            if preflight_reason is not None:
                points.append(ollama_evidence.failed_characterization_point(
                    ctx, preflight_reason))
                break
        if authority is not None and ollama.runtime_authority(authority.endpoint) != authority:
            raise RuntimeError("Ollama runtime authority changed before measurement")
        if (base_artifact_id is not None
                and _ollama_artifact_id(model) != base_artifact_id):
            raise RuntimeError("base manifest changed before probe creation")
        if not ollama.create(probe, model, ctx):     # couldn't bake this rung — stop, keep what fit
            break
        probe_artifact_id = None
        if base_artifact_id is not None or provenance is not None:
            if provenance is not None:
                provenance["created"] = True
            probe_artifact_id = _ollama_artifact_id(probe)
            if probe_artifact_id is None:
                raise RuntimeError("created probe manifest could not be identified")
            if provenance is not None:
                provenance["artifact_id"] = probe_artifact_id
            if (base_artifact_id is not None
                    and _ollama_artifact_id(model) != base_artifact_id):
                raise RuntimeError("base manifest changed during probe creation")
        if not ollama.probe_generate(probe, ctx):
            points.append(ollama_evidence.failed_characterization_point(
                ctx, "generation_failed"))
            break
        if authority is not None and ollama.runtime_authority(authority.endpoint) != authority:
            raise RuntimeError("Ollama runtime authority changed during measurement")
        processes = ollama.processes()
        after = ollama_evidence.capture_memory_snapshot()
        expected_digest = (probe_artifact_id.removeprefix(_OLLAMA_ARTIFACT_PREFIX)
                           if probe_artifact_id is not None else None)
        residency_error = _ollama_residency_error(
            processes, allowed_name=probe, allowed_digest=expected_digest,
            allowed_context=ctx, require_target=True)
        if residency_error is not None:
            previous_runner_remains = (
                previous_context is not None
                and _ollama_residency_error(
                    processes, allowed_name=probe, allowed_digest=previous_digest,
                    allowed_context=previous_context, require_target=True) is None
            )
            if processes == [] or previous_runner_remains:
                points.append(ollama_evidence.failed_characterization_point(
                    ctx, "effective_residency_unverified"))
                break
            raise RuntimeError(residency_error)
        process = processes[0]
        point = ollama_evidence.characterization_point(before, after, process, ctx)
        points.append(point)
        if not point["fit"]:                         # hit/failed to verify the wall — stop safely
            break
        best = ctx
        previous_context, previous_digest = ctx, expected_digest
    return best, points


def _ollama_characterization_config(
    authority: ollama.OllamaRuntimeAuthority,
    record: ollama.OllamaModel,
    best: int | None,
    points: list[dict],
) -> dict:
    """Persist the runtime, scheduler, placement, and wall evidence behind one ceiling."""

    best_point = next(
        (point for point in reversed(points)
         if point.get("fit") and point.get("requested_context") == best),
        None,
    )
    configured = dict(authority.configured_inputs)
    return {
        "methodology": "ollama-physical-walls-v1",
        "runtime": "ollama",
        "runtime_version": authority.server_version,
        "endpoint_authority": authority.endpoint.url,
        "server_instance_id": authority.server_instance_id,
        "format": record.format.casefold() if record.format is not None else None,
        "capability": "completion" if "completion" in record.capabilities else None,
        "configured_inputs": configured,
        "configured_num_parallel": authority.configured_num_parallel,
        "configured_num_parallel_authority": authority.configured_num_parallel_authority,
        "effective_num_parallel": 1,
        "effective_num_parallel_authority": "configured_maximum_is_one",
        "requested_context": best,
        "effective_per_request_context": (
            best_point.get("effective_per_request_context") if best_point else None),
        "placement": best_point.get("placement") if best_point else "unknown",
        "resident_total_bytes": (
            best_point.get("resident_total_bytes") if best_point else None),
        "resident_accelerator_bytes": (
            best_point.get("resident_accelerator_bytes") if best_point else None),
        "applicable_walls": best_point.get("applicable_walls", []) if best_point else [],
        "system_memory_delta_bytes": (
            best_point.get("system_memory_delta_bytes") if best_point else None),
        "accelerator_memory_delta_bytes": (
            best_point.get("accelerator_memory_delta_bytes") if best_point else None),
        "system_margin_bytes": (
            best_point.get("system_margin_bytes") if best_point else None),
        "accelerator_margin_bytes": (
            best_point.get("accelerator_margin_bytes") if best_point else None),
        "configured_kv_cache_type": configured.get("OLLAMA_KV_CACHE_TYPE", "unknown"),
        "effective_kv_cache_type": "unknown",
        "configured_flash_attention": configured.get("OLLAMA_FLASH_ATTENTION", "unknown"),
        "effective_flash_attention": "unknown",
        "configured_scheduler_spread": configured.get("OLLAMA_SCHED_SPREAD", "unknown"),
        "effective_scheduler_spread": "unknown",
    }


def _cleanup_ollama_probe(probe: str, expected_artifact_id: str) -> str | None:
    """Delete an idle probe, or retain its exact manifest while its runner is resident."""
    if _ollama_artifact_id(probe) != expected_artifact_id:
        return "probe manifest identity changed; refused delete"
    processes = ollama.processes()
    if processes is None:
        return "couldn't inspect probe residency; refused delete"
    accepted_names = {probe, probe + ":latest"}
    residents = [process for process in processes if process.name in accepted_names]
    if residents:
        expected_digest = expected_artifact_id.removeprefix(_OLLAMA_ARTIFACT_PREFIX)
        if len(processes) == 1 and residents[0].digest == expected_digest:
            return None
        return "probe residency identity changed; refused delete"
    return _delete_exact_ollama_model(
        probe, label="probe", expected_artifact_id=expected_artifact_id)


def _delete_exact_ollama_model(name: str, *, label: str,
                               expected_artifact_id: str) -> str | None:
    """Delete one exact manifest without changing its potentially shared runner's residency."""
    if _ollama_artifact_id(name) != expected_artifact_id:
        return f"{label} manifest identity changed; refused delete"
    if not ollama.delete(name):
        return f"couldn't delete {label} model"
    return None


def _render_characterize_ollama(c: Console, model: str, *, as_json: bool) -> int:
    """``ara characterize <model> --engine ollama``: ramp the model's residency through Ollama to
    find and record its true measured safe ceiling (engine ``"ollama"``). Serving from an estimated
    ceiling never manufactures measured evidence; this is where the slow, thorough measurement
    lives (Will, 2026-07-04). Spec
    2026-07-04-characterize-through-ollama-ramp."""
    def err(msg: str) -> int:
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1

    endpoint, endpoint_error = _ollama_governed_endpoint()
    if endpoint_error is not None:
        return err(endpoint_error)
    authority, authority_error = _ollama_runtime_authority(endpoint)
    if authority_error:
        return err(authority_error)
    models = ollama.inventory()
    if models is None:
        return err("couldn't list Ollama models — is the server reachable?")
    record = ollama.find_model(models, model)
    if record is None:
        return err(f"{model} isn't in Ollama — pull it first: ollama pull {model}")
    model = record.name
    if (support_error := ollama.initial_governed_model_error(record)) is not None:
        return err(support_error)
    if record.size_bytes is None:
        return err(f"couldn't verify {record.name}'s on-disk size — refusing an unbounded "
                   "Ollama model load.")
    preflight_reason = ollama_evidence.preflight_refusal_reason(
        ollama_evidence.capture_memory_snapshot(), record.size_bytes)
    if preflight_reason is not None:
        return err(f"Ollama model-load preflight refused: {preflight_reason.replace('_', ' ')}")
    artifact_id = _ollama_artifact_id(model, record=record)
    if artifact_id is None:
        return err(f"couldn't identify {model}'s Ollama manifest — refusing to measure mutable "
                   "weights without artifact provenance.")
    max_ctx = _ollama_max_context(model)
    if not max_ctx:
        return err(f"couldn't read {model}'s context length from Ollama — can't bound the ramp.")
    if c.verbose and not as_json:
        c.emit(c.field("engine", "ollama", "external runtime"))
        c.emit(c.field("model limit", f"{max_ctx} tokens", "architecture maximum"))

    probe = _governed_name(
        model, artifact_id=artifact_id, context=max_ctx) + "-probe"
    cleanup_error = None
    measurement_error = None
    provenance: dict = {}
    try:
        with locking.ollama_setup_lock(endpoint.url, probe):
            residency_error = _ollama_residency_error(ollama.processes())
            if residency_error is not None:
                return err(residency_error)
            latest_models = ollama.inventory()
            if latest_models is None:
                return err("couldn't recheck Ollama model names before characterization — "
                           "refusing to risk a probe collision.")
            if ollama.find_model(latest_models, probe) is not None:
                return err(f"Ollama characterization probe {probe!r} already exists — refusing "
                           "to overwrite or delete it.")
            with activity.track("characterizing", model):
                try:
                    best, points = _ollama_measure_ceiling(
                        model, max_ctx, probe, base_artifact_id=artifact_id,
                        provenance=provenance, authority=authority,
                        model_size_bytes=record.size_bytes)
                except (SystemExit, Exception) as exc:
                    measurement_error = exc
                finally:
                    probe_artifact_id = provenance.get("artifact_id")
                    if probe_artifact_id is None and provenance.get("created"):
                        cleanup_error = ("probe ownership could not be proven; ARA refused "
                                         "destructive cleanup")
                    elif probe_artifact_id is not None:
                        cleanup_error = _cleanup_ollama_probe(probe, probe_artifact_id)
    except locking.OllamaSetupBusy as exc:
        return err(str(exc))
    if measurement_error is not None:
        msg = f"Ollama characterization failed: {measurement_error}"
        if cleanup_error:
            msg += f"; probe cleanup also failed: {cleanup_error}"
        return err(msg)
    if cleanup_error:
        return err(f"Ollama probe cleanup failed: {cleanup_error}")
    if _ollama_artifact_id(model) != artifact_id:
        return err(f"{model}'s Ollama manifest changed during measurement — refusing to store a "
                   "ceiling for ambiguous weights.")
    if ollama.runtime_authority(endpoint) != authority:
        return err("Ollama runtime authority changed during measurement — refusing to store "
                   "evidence for an ambiguous daemon configuration.")
    config = _ollama_characterization_config(authority, record, best, points)
    probe_artifact_id = provenance.get("artifact_id")
    if isinstance(probe_artifact_id, str):
        config["characterization_probe_name"] = probe
        config["characterization_probe_artifact_id"] = probe_artifact_id
        config["characterization_probe_context"] = max_ctx
    with db.connected() as con:
        db.save_characterization(con, profile.machine_key(), "ollama", model,
                                 safe_context=best, points=points, measured_at=None,
                                 artifact_id=artifact_id, config=config)
    if as_json:
        print(json.dumps({"model": model, "engine": "ollama", "safe_context": best,
                          "source": "measured", "max_context": max_ctx}))
        return 0
    if best is None:
        c.emit(c.style("warn", f"  no physically safe ceiling for {model} on this box at the "
                               f"smallest tested context (recorded)."))
    else:
        placement = config["placement"].replace("_", " ")
        c.emit(c.field("measured ceiling", f"~{best} tokens  (ollama, {placement})"))
    return 0


def _is_ara_ollama_derived(name: str) -> bool:
    """Recognize ARA's exact content-addressed governed/probe model-name shape."""
    base = name.removesuffix(":latest").removesuffix("-probe")
    parts = base.rsplit("-", 1)
    return (len(parts) == 2 and parts[0].startswith("ara-") and "-ctx" in parts[0]
            and len(parts[1]) == 24 and all(char in "0123456789abcdef" for char in parts[1]))


def _ollama_benchmark_evidence(
    row: dict,
    record: ollama.OllamaModel,
    authority: ollama.OllamaRuntimeAuthority,
    config: dict,
) -> tuple[dict | None, str | None]:
    """Validate one Ollama score against the exact target that would run now."""
    evidence, warning = scoring.validate_measured_evidence(row)
    if evidence is None:
        return None, warning
    if row.get("engine_key") != "ollama" or row.get("quant") != record.quantization:
        return None, "invalid stored Ollama benchmark evidence"
    artifact_id = (_OLLAMA_ARTIFACT_PREFIX + record.digest
                   if record.digest is not None else None)
    if row.get("artifact_id") != artifact_id:
        return None, "Ollama benchmark artifact changed"
    try:
        backend = _ollama_backend_for_placement(config["placement"])
    except (KeyError, TypeError):
        return None, "Ollama benchmark target changed"
    if row.get("backend") != backend:
        return None, "Ollama benchmark target changed"
    expected_target = _ollama_benchmark_target(
        authority, record, config, backend, artifact_id)
    if row.get("target") != expected_target:
        return None, "Ollama benchmark target changed"
    expected_policy = _ollama_benchmark_policy(record, row["generation_cap"])
    if row.get("request_policy") != expected_policy:
        return None, "Ollama benchmark request policy changed"
    metrics = row.get("runtime_metrics")
    if (not isinstance(metrics, dict)
            or not set(metrics).issubset(_OLLAMA_BENCHMARK_METRIC_NAMES)
            or any(type(value) is not int or value < 0 for value in metrics.values())):
        return None, "invalid Ollama benchmark runtime metrics"
    return evidence, None


def _ollama_ranked_models(
    models: list[ollama.OllamaModel],
    authority: ollama.OllamaRuntimeAuthority,
    *,
    use_case: str | None = None,
) -> list[dict]:
    """Central Ollama recommendation result shared by model UX and bare ``serve``.

    Only exact reusable characterization evidence is selection-eligible. Analytic estimates are
    retained as labelled comparison evidence and never promoted into execution authority.
    """
    with ExitStack() as stack:
        scratch = sqlite3.connect(":memory:")
        scratch.row_factory = sqlite3.Row
        scratch.executescript(db.SCHEMA)
        stack.callback(scratch.close)
        con = (stack.enter_context(db.connected_readonly())
               if db._db_path().is_file() else scratch)
        mk = profile.machine_key()
        benchmark_rows: dict[tuple[str, str], list[dict]] = {}
        if use_case is not None:
            for row in db.list_benchmark_results(con, mk):
                benchmark_rows.setdefault(
                    (row["model_id"], row["use_case"]), []).append(row)
        ranked = []
        for record in models:
            if (_is_ara_ollama_derived(record.name)
                    or ollama.initial_governed_model_error(record) is not None):
                continue
            assessment = ollama_evidence.assess_characterization(con, mk, record, authority)
            reusable = assessment.reusable
            safe_context = reusable.get("safe_context") if reusable is not None else None
            estimate_found = _ollama_estimated_ceiling(record.name, record=record)
            estimated_context = estimate_found[0] if estimate_found is not None else None
            if assessment.reusable is not None:
                evidence_status = "reusable"
            elif assessment.display is not None:
                evidence_status = "display_only"
            else:
                evidence_status = "missing"
            item = {
                "model_id": record.name,
                "artifact_id": (_OLLAMA_ARTIFACT_PREFIX + record.digest
                                if record.digest is not None else None),
                "digest": record.digest,
                "size_bytes": record.size_bytes,
                "format": record.format,
                "family": record.family,
                "parameter_size": record.parameter_size,
                "quantization": record.quantization,
                "capabilities": list(record.capabilities),
                "scope": record.scope,
                "safe_context": safe_context,
                "estimated_context": estimated_context,
                "selection_eligible": reusable is not None,
                "evidence_status": evidence_status,
                "evidence_reason": assessment.reason,
            }
            if use_case is not None:
                score = None
                benchmark_warning = None
                candidates = benchmark_rows.get((record.name, use_case), [])
                if not candidates:
                    benchmark_warning = "no stored Ollama benchmark evidence"
                elif assessment.reusable is None:
                    benchmark_warning = "current Ollama characterization is not reusable"
                else:
                    rejected = []
                    for benchmark_row in reversed(candidates):
                        measured, warning = _ollama_benchmark_evidence(
                            benchmark_row, record, authority,
                            assessment.reusable["config"])
                        if measured is not None:
                            score = scoring.score_for(
                                record.name, use_case,
                                measured={(record.name, use_case): measured})
                            benchmark_warning = None
                            break
                        rejected.append(warning)
                    if score is None:
                        benchmark_warning = rejected[0]
                item["score"] = score
                item["evidence_warning"] = benchmark_warning
            ranked.append(item)
    return sorted(
        ranked,
        key=lambda item: (
            use_case is not None and item["score"] is None,
            -(item["score"].value
              if use_case is not None and item["score"] is not None else 0.0),
            not item["selection_eligible"],
            -(item["safe_context"] if item["selection_eligible"]
              else item["estimated_context"] or -1),
            item["model_id"],
        ),
    )


def _render_recommend_ollama(
    c: Console, *, as_json: bool, use_case: str | None,
) -> int:
    """Read-only ranking of supported local Ollama artifacts and their exact evidence."""
    def err(msg: str) -> int:
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1

    if use_case is not None and use_case not in benchmark.USE_CASES:
        return err(f"unknown use-case {use_case!r} — choose one of: "
                   f"{', '.join(benchmark.USE_CASES)}")
    endpoint = ollama.endpoint_authority()
    if endpoint.url is None:
        return err("couldn't identify the configured Ollama endpoint")
    models = ollama.inventory()
    if models is None:
        return err("couldn't list Ollama models — is the server reachable?")
    authority = _ollama_display_authority(endpoint)
    ranked = (_ollama_ranked_models(models, authority, use_case=use_case)
              if use_case is not None else _ollama_ranked_models(models, authority))
    if as_json:
        if use_case is not None:
            ranked = [{**item, "score": (
                asdict(item["score"]) if item["score"] is not None else None)}
                      for item in ranked]
        print(json.dumps(ranked, indent=2))
        return 0
    c.emit()
    sub = ("  (exact reusable evidence first)" if use_case is None else
           f"  (for {use_case} — exact current Ollama measurements only)")
    c.emit(c.section("  RECOMMENDED OLLAMA MODELS") + c.style("dim", sub))
    if not ranked:
        c.emit(c.style("dim", "  no local cached GGUF completion artifacts are supported"))
        c.emit()
        return 0
    for item in ranked:
        if item["selection_eligible"]:
            tail = f"~{item['safe_context']} tok measured · eligible for governed selection"
            role = "good"
        elif item["estimated_context"] is not None:
            tail = (f"~{item['estimated_context']} tok analytic estimate · comparison only"
                    f" · {item['evidence_status'].replace('_', ' ')}")
            role = "accent"
        else:
            tail = f"unrankable · {item['evidence_status'].replace('_', ' ')}"
            role = "dim"
        if item["evidence_reason"] not in (None, "missing"):
            tail += f" ({item['evidence_reason'].replace('_', ' ')})"
        if use_case is not None:
            score = item["score"]
            if score is None:
                head = f"{use_case} unknown"
            else:
                head = f"{use_case} {score.value * 100:.0f}% ({score.tier})"
                if score.refused_n or score.errored_n:
                    partial = []
                    denominator = f"/{score.total_generations}" if score.total_generations else ""
                    if score.refused_n:
                        partial.append(f"{score.refused_n}{denominator} refused")
                    if score.errored_n:
                        partial.append(f"{score.errored_n}{denominator} errored")
                    head += f" [partial: {', '.join(partial)}]"
                if score.sample_size is not None and score.sample_size < 100:
                    head += f" [low-confidence n={score.sample_size}]"
            if item["evidence_warning"]:
                head += f" [{item['evidence_warning']}]"
            tail = f"{head} · {tail}"
        quant = f" [{item['quantization']}]" if item["quantization"] else ""
        c.emit("  " + c.style("metric", item["model_id"])
               + c.style("dim", quant + "  →  ") + c.style(role, tail))
    eligible = sum(1 for item in ranked if item["selection_eligible"])
    measured = (sum(1 for item in ranked if item["score"] is not None)
                if use_case is not None else None)
    c.emit()
    summary = f"  {len(ranked)} supported local · {eligible} certified for selection"
    if measured is not None:
        summary += f" · {measured} reusable {use_case} measurement"
    c.emit(c.style("dim", summary))
    c.emit()
    return 0


def _ollama_pick_best(
    models: list[ollama.OllamaModel],
    authority: ollama.OllamaRuntimeAuthority,
) -> str | None:
    """Return the best central recommendation eligible for automatic governed execution."""
    return next((item["model_id"] for item in _ollama_ranked_models(models, authority)
                 if item["selection_eligible"]), None)


def _governed_name(model: str, *, artifact_id: str | None = None,
                   context: int | None = None) -> str:
    """Return a deterministic Ollama name bound to policy, manifest, and governed context.

    The two-argument form is retained for internal probe-name compatibility only.
    """
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "-" for ch in model.lower())
    if artifact_id is not None and context is not None:
        ascii_safe = "".join(ch if (ch.isascii() and (ch.isalnum() or ch in "._-")) else "-"
                             for ch in model.lower()).strip("-._") or "model"
        digest = hashlib.sha256(
            f"{_OLLAMA_DERIVED_POLICY_VERSION}\0{model}\0{artifact_id}\0{context}".encode(
                "utf-8")).hexdigest()[:24]
        return f"ara-{ascii_safe[:40]}-ctx{context}-{digest}"
    return safe + "-ara"


def _find_loaded(entries: list[dict], served: str, *,
                 expected_context: int | None = None) -> dict | None:
    """The ``/api/ps`` entry for our derived model (Ollama tags it ``:latest``), or ``None``."""
    matches = [m for m in entries if isinstance(m, dict)
               and isinstance(m.get("name"), str)
               and m["name"] in (served, served + ":latest")]
    if expected_context is None:
        return matches[0] if matches else None
    valid = next((m for m in matches
                  if isinstance(m.get("context_length"), int)
                  and not isinstance(m["context_length"], bool)
                  and m["context_length"] > 0
                  and m["context_length"] == expected_context), None)
    return valid if valid is not None else (matches[0] if matches else None)


def _ollama_residency_error(
        processes: list[ollama.OllamaProcess] | None, *,
        allowed_name: str | None = None,
        allowed_digest: str | None = None,
        allowed_context: int | None = None,
        alternate_targets: tuple[tuple[str, str], ...] = (),
        require_target: bool = False) -> str | None:
    """Return a strict admission refusal, or ``None`` for an admissible snapshot.

    Empty residency is admissible before setup. The only non-empty admissible state is one exact,
    fully described target supplied by the caller. A post-load check sets ``require_target`` so an
    empty snapshot also fails closed.
    """
    if processes is None:
        if require_target:
            return "couldn't inspect Ollama residency safely after load"
        return ("couldn't inspect Ollama residency safely — ARA refused to create or load "
                "a model")
    if not processes:
        if require_target:
            return f"expected Ollama model {json.dumps(allowed_name)} is not resident after load"
        return None
    names = ", ".join(json.dumps(process.name) for process in processes)
    refusal = (f"resident Ollama model(s) {names} block strict admission; "
               "ARA did not unload them")
    if len(processes) != 1 or allowed_name is None:
        return refusal
    process = processes[0]
    targets = ((allowed_name, allowed_digest), *alternate_targets)
    verified = (
        any(
            process.name in {name, name + ":latest"} and process.digest == digest
            for name, digest in targets
        )
        and process.effective_context_per_request == allowed_context
        and isinstance(process.size_bytes, int) and not isinstance(process.size_bytes, bool)
        and process.size_bytes > 0
        and isinstance(process.size_vram_bytes, int)
        and not isinstance(process.size_vram_bytes, bool)
        and process.size_vram_bytes >= 0
    )
    return None if verified else refusal


def _ollama_recorded_probe_target(
        config: dict, models: list[ollama.OllamaModel] | None, *,
        model_name: str, base_artifact_id: str) -> tuple[str, str] | None:
    """Return an exact, still-installed characterization probe admitted by stored evidence."""
    name = config.get("characterization_probe_name")
    artifact_id = config.get("characterization_probe_artifact_id")
    context = config.get("characterization_probe_context")
    if (
        not isinstance(name, str)
        or not _is_ara_ollama_derived(name)
        or not isinstance(artifact_id, str)
        or not artifact_id.startswith(_OLLAMA_ARTIFACT_PREFIX)
        or not isinstance(context, int)
        or isinstance(context, bool)
        or context <= 0
        or name != _governed_name(
            model_name, artifact_id=base_artifact_id, context=context) + "-probe"
        or models is None
    ):
        return None
    digest = artifact_id.removeprefix(_OLLAMA_ARTIFACT_PREFIX)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return None
    model = ollama.find_model(models, name)
    if model is None or model.digest != digest:
        return None
    return name, digest


def _free_port() -> int:
    """An OS-assigned free TCP port on localhost (small bind/close race, acceptable for v1)."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stop_mlx_server(proc) -> None:
    """Best-effort terminate, kill, and reap for a child whose serve lifecycle failed."""
    for method in ("terminate", "kill", "wait"):
        try:
            getattr(proc, method)()
        except BaseException:
            pass


def _render_serve_mlx(c: Console, model: str, *, engine_key: str, ctx: int | None = None,
                      assume_yes: bool = False, as_json: bool = False,
                      kv_quant: str = "f16") -> int:
    """Stand *model* up on the governed MLX server, capped at the MEASURED apple
    ceiling (or explicit ``--ctx``), and hand back an OpenAI-compatible endpoint. ARA owns the
    server subprocess, so it stays foreground until Ctrl-C. The MLX ceiling is valid here because
    serve and characterize share the mlx_lm allocation path (seam-mismatch rule, the other way).
    Spec 2026-06-28-recommend-use-case-and-serve-selection."""
    def err(msg: str) -> int:
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1

    mk = profile.machine_key()
    expected_config = _measurement_config("apple", kv_quant=kv_quant)
    with db.connected() as con:
        reusable_row = db.get_reusable_characterization_for_engine(
            con, mk, engine_key, model, config=expected_config)
        display_row = db.get_characterization(con, mk, engine_key, model)
        if ctx is not None:
            if display_row and (msg := _measurement_config_error(
                    display_row, expected_config, "apple", model)):
                return err(msg)
            if (msg := _ctx_gate_msg(
                    ctx, reusable_row.get("safe_context") if reusable_row else None, model)):
                return err(msg)
            safe, source = ctx, "requested"
            ceiling_measured_at = None       # explicit --ctx, not a stored ceiling
            measured_slope = None            # a-priori gate for an unmeasured override (Rule #1)
        else:
            row = reusable_row
            if not row or row.get("safe_context") is None:
                if display_row and (msg := _measurement_config_error(
                        display_row, expected_config, "apple", model)):
                    return err(msg)
                return err(f"no reusable measured MLX ceiling for {model} — run: ara characterize {model} "
                           f"(or pass --ctx N).")
            safe, source = row["safe_context"], "measured"
            ceiling_measured_at = row.get("measured_at")
            # Serving the model's OWN measured ceiling: fit the real ramp slope so the pre-load gate
            # predicts with it, not the conservative a-priori prior that would falsely refuse a
            # long-window measured serve (slug 2026-07-02-wmx-serve-measured-provenance-gate).
            measured_slope = _measured_ramp_slope(row)

    stale_ceiling = _stale_ceiling_note(c, model, ceiling_measured_at, as_json=as_json)
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

    expected_url = f"http://127.0.0.1:{port}"
    if not isinstance(url, str) or url.rstrip("/") != expected_url:
        _stop_mlx_server(proc)
        return err("invalid MLX server ready signal — endpoint did not match the allocated "
                   "localhost port")
    if (not isinstance(served_ctx, int) or isinstance(served_ctx, bool)
            or served_ctx != safe):
        _stop_mlx_server(proc)
        return err(f"MLX governance failed: server reported {served_ctx!r} ctx, not {safe}")

    endpoint = url.rstrip("/") + "/v1"
    if as_json:
        print(json.dumps({"endpoint": endpoint, "model": model, "served_context": served_ctx,
                          "ceiling_source": source, "openai_base_url": endpoint,
                          "runtime": "mlx", "stale_ceiling": stale_ceiling}, indent=2))
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
        try:
            with activity.track("serving", model):
                returncode = proc.wait()            # our child IS the server; stay alive to serve
        except KeyboardInterrupt:
            _stop_mlx_server(proc)
            return 0
        except BaseException:
            _stop_mlx_server(proc)
            raise
    finally:
        signal.signal(signal.SIGTERM, old)
    if returncode not in (None, 0):
        return err(f"MLX server exited with status {returncode}")
    return 0


def render_serve(c: Console, model: str | None = None, *, ctx: int | None = None,
                 name: str | None = None, engine: str | None = None,
                 assume_yes: bool = False, as_json: bool = False) -> int:
    """Stand *model* up as a **governed** OpenAI-compatible endpoint on a local Ollama, capped at a
    safe context ceiling, and hand back the connection — then get out of the way (BYO consumer).

    The ceiling is **baked into a content-addressed derived model**: a plain ``/v1`` request reloads
    the base model at its *default* context, blowing past the safe wall (measured 2026-06-26), so
    governing per-request isn't enough. Handoff requires exact reusable measured evidence;
    ``--ctx`` may lower that ceiling but cannot replace it, and missing models are never pulled
    implicitly. After load ARA verifies the ceiling before returning a point-in-time endpoint
    contract. ``--engine mlx`` routes to the governed MLX server instead (spec 2026-06-28)."""
    def err(msg: str) -> int:
        print(json.dumps({"error": msg})) if as_json else c.emit(c.style("bad", f"  {msg}"))
        return 1

    auto_selected = model is None    # bare `ara serve` → pick the best-fitting model in the store
    if ctx is not None and ctx <= 0:
        return err("--ctx must be a positive integer")
    if engine is not None and model is None:
        return err("`ara serve` with no model picks from the Ollama store — pass a model to use "
                   "--engine.")
    if engine not in (None, "ollama"):
        key = engines.resolve(engine)
        if key and engines.ENGINES.get(key, {}).get("backend") == "apple":
            if name is not None:
                return err("--name applies only to Ollama serving; MLX exposes the model name.")
            return _render_serve_mlx(c, model, engine_key=key, ctx=ctx,
                                     assume_yes=assume_yes, as_json=as_json)
        if engine != "auto":
            return err(f"serve doesn't support --engine {engine!r} — use ollama, mlx, or auto.")
        # Auto uses native MLX on Apple Silicon (handled above), otherwise Ollama. Ollama owns its
        # own CPU/GPU routing, so claiming a CUDA/CPU engine pin here would be dishonest.

    endpoint_authority, endpoint_error = _ollama_governed_endpoint()
    if endpoint_error is not None:
        return err(endpoint_error)

    # Reject unsafe persistent activity fields before touching Ollama or pulling model data. A
    # placeholder positive context is sufficient here because the real ceiling is separately
    # validated before setup; this pass keeps malformed user input side-effect free.
    def validate_identity(candidate: str) -> int | None:
        try:
            activity.validate_ollama_serving_fields(
                served_name=name or _governed_name(candidate), model=candidate,
                context=ctx or 1, endpoint=endpoint_authority.url or ollama.base_url())
        except ValueError as exc:
            return err(f"invalid serving identity: {exc}")
        return None

    if model is not None and (invalid := validate_identity(model)) is not None:
        return invalid

    authority, authority_error = _ollama_runtime_authority(endpoint_authority)
    if authority_error:
        return err(authority_error)

    # 1. liveness was established by the attributed runtime-authority check above.
    models = ollama.inventory()
    if models is None:
        return err("couldn't list Ollama models — is the server reachable?")

    # 2a. zero-arg: recommend among what's already pulled, then serve the best fit
    if model is None:
        if not models:
            return err("no models in Ollama — pull one (`ollama pull <model>`), or name one: "
                       "`ara serve <model>`.")
        model = _ollama_pick_best(models, authority)
        if model is None:
            return err("no model in Ollama fits this machine — pull a smaller / more-quantized "
                       "one, or name one explicitly.")
        if (invalid := validate_identity(model)) is not None:
            return invalid
        if not as_json:
            c.emit(c.style("dim", "  auto-selected ") + c.style("accent", model)
                   + c.style("dim", " (best fit in your Ollama store)"))

    # 2b. named model: serving never acquires weights; installation remains an explicit user act.
    record = ollama.find_model(models, model)
    if record is None:
        return err(f"{model} isn't in the configured Ollama store — install it there explicitly, "
                   f"then run `ara characterize {model} --engine ollama`")

    model = record.name
    if (support_error := ollama.initial_governed_model_error(record)) is not None:
        return err(support_error)
    base_artifact_id = _ollama_artifact_id(model, record=record)
    if base_artifact_id is None:
        return err(f"couldn't identify {model}'s Ollama manifest — refusing to serve mutable "
                   "weights without artifact provenance.")

    # 3. Resolve exact reusable measured evidence. An analytic estimate is comparison-only and an
    # explicit context may lower, but never mint, handoff authority.
    with db.connected() as con:
        found = _ollama_safe_ceiling(con, profile.machine_key(), record, authority)
    if found is None:
        return err(f"no reusable measured ceiling for {model} on this Ollama runtime — run "
                   f"`ara characterize {model} --engine ollama` first")
    if ctx is not None:
        if ctx > found[0]:
            return err(f"--ctx {ctx} exceeds the measured safe ceiling {found[0]} for {model} — refusing "
                       "(Rule #1: never exceed the memory wall).")
        safe, source = ctx, "requested"
        ceiling_measured_at = found[2]
    else:
        safe, source, ceiling_measured_at = found

    stale_ceiling = _stale_ceiling_note(c, model, ceiling_measured_at, as_json=as_json)
    # consent — serve creates + holds a model in memory
    if not as_json and not assume_yes and sys.stdin.isatty():
        if not _confirm(f"Stand up {model} on Ollama, governed at ≤{safe} ctx?"):
            c.emit(c.style("dim", "  skipped."))
            return 0

    # 4. bake the ceiling into a derived model. This is temporary process-owned work until exact
    # verification succeeds; only then can ARA hand off to a persistent Ollama ownership claim.
    served = name or _governed_name(
        model, artifact_id=base_artifact_id, context=safe)
    served_preexisting = ollama.find_model(models, served) is not None
    endpoint_base = endpoint_authority.url
    live_activity = activity.snapshot()
    ownership_activity = activity.ollama_ownership()
    handoff_authority = {
        "runtime_version": authority.server_version,
        "server_instance_id": authority.server_instance_id,
        "configured_inputs": dict(authority.configured_inputs),
        "configured_num_parallel": authority.configured_num_parallel,
        "configured_num_parallel_authority": authority.configured_num_parallel_authority,
    }
    legacy_item = next((item for item in live_activity
                        if item.runtime == "ollama" and item.model == model
                        and item.endpoint == endpoint_base and item.base_artifact_id is None), None)
    if legacy_item is not None:
        return err(f"legacy ARA service {legacy_item.served_name!r} is still live for {model} — "
                   "refusing to load a duplicate; stop and remove that legacy service first.")
    owned_item = next((
        item
        for item in ownership_activity
        if item.runtime == "ollama" and item.served_name == served
        and item.context == safe and item.endpoint == endpoint_base
        and item.model == model and item.base_artifact_id == base_artifact_id
        and item.served_artifact_id is not None
        and item.policy_version == _OLLAMA_DERIVED_POLICY_VERSION
    ), None)
    already_owned = owned_item is not None
    if served_preexisting and not already_owned:
        return err(f"Ollama model {served!r} already exists but is not owned by ARA with this "
                   "exact model, context, and endpoint — refusing to overwrite or unload it.")
    if name is not None and not already_owned:
        return err("a new custom --name cannot be created safely because Ollama has no atomic "
                   "create-if-absent operation; omit --name to use ARA's content-addressed name.")

    served_artifact_id = owned_item.served_artifact_id if owned_item else None
    ownership_recorded = False

    ownership_fields = {
        "served_name": served,
        "model": model,
        "context": safe,
        "endpoint": endpoint_base,
        "base_artifact_id": base_artifact_id,
        "policy_version": _OLLAMA_DERIVED_POLICY_VERSION,
        "runtime_authority": handoff_authority,
    }

    def setup_err(msg: str) -> int:
        if already_owned:
            return err(msg)
        if (served_artifact_id is None
                or _ollama_artifact_id(served) != served_artifact_id):
            return err(msg + "; ownership of the derived manifest is unverified, so ARA did not "
                       "unload or delete it")
        cleanup = _delete_exact_ollama_model(
            served, label="governed", expected_artifact_id=served_artifact_id)
        if cleanup:
            msg += f"; cleanup also failed: {cleanup}"
        else:
            msg += "; deleted the untracked manifest without expiring its runner"
        return err(msg)

    already_live = any(
        item.runtime == "ollama" and item.served_name == served
        and item.context == safe and item.endpoint == endpoint_base
        and item.model == model and item.base_artifact_id == base_artifact_id
        and item.served_artifact_id == served_artifact_id
        for item in live_activity)
    setup_activity = nullcontext() if already_live else activity.track("serving", model)
    create_confirmed = False
    try:
        with locking.ollama_setup_lock(endpoint_base, served):
            with setup_activity:
                allowed_digest = (served_artifact_id.removeprefix(_OLLAMA_ARTIFACT_PREFIX)
                                  if already_owned and served_artifact_id is not None else None)
                residency_error = _ollama_residency_error(
                    ollama.processes(),
                    allowed_name=served if already_owned else None,
                    allowed_digest=allowed_digest,
                    allowed_context=safe if already_owned else None)
                if residency_error is not None:
                    return err(residency_error)
                if not already_owned:
                    latest_models = ollama.inventory()
                    if latest_models is None:
                        return err("couldn't recheck Ollama model names before creating the governed "
                                   "model — refusing to risk a collision.")
                    if ollama.find_model(latest_models, served) is not None:
                        return err(f"Ollama model {served!r} appeared before ARA could create it — "
                                   "refusing to overwrite or unload it.")
                    if not ollama.create(served, model, safe):
                        return err(f"couldn't confirm creation of governed model {served!r}; no "
                                   "load or destructive cleanup was attempted because ownership "
                                   "is unknown.")
                    create_confirmed = True
                    served_artifact_id = _ollama_artifact_id(served)
                    if served_artifact_id is None:
                        return err(f"created {served!r}, but couldn't identify its Ollama manifest; "
                                   "refusing to load or destructively clean up unknown ownership.")
                    if _ollama_artifact_id(model) != base_artifact_id:
                        return setup_err(
                            f"{model}'s Ollama manifest changed during setup — refusing to load it")

                # 5. Load under the daemon's cache/eviction policy. ARA verifies this handoff now;
                # it does not force indefinite residency or claim to govern later cold reloads.
                if _ollama_artifact_id(served) != served_artifact_id:
                    return err(f"governed model {served!r} changed before load — refusing mutable "
                               "identity; ARA did not unload or delete it")
                if ollama.load(served, keep_alive=None) is None:
                    return setup_err(f"couldn't load the governed model {served!r} on Ollama")
                expected_served_digest = served_artifact_id.removeprefix(_OLLAMA_ARTIFACT_PREFIX)
                processes = ollama.processes()
                residency_error = _ollama_residency_error(
                    processes, allowed_name=served, allowed_digest=expected_served_digest,
                    allowed_context=safe, require_target=True)
                if residency_error is not None:
                    return setup_err(residency_error)
                process = processes[0]
                residency_verified = True
                spilled = process.size_vram_bytes < process.size_bytes

                # 6. Exercise the public protocol ARA is about to hand off, then re-attest every
                # authority that makes the point-in-time claim true. The /v1 request itself may
                # load or replace a runner, so the earlier /api/ps observation is not sufficient.
                if not ollama.openai_completion_probe(served):
                    return setup_err(
                        f"{served!r} did not satisfy the OpenAI /v1 completion contract")
                if ollama.runtime_authority(endpoint_authority) != authority:
                    return setup_err(
                        "Ollama runtime changed during OpenAI contract verification — refusing "
                        "stale ownership")
                if _ollama_artifact_id(model) != base_artifact_id:
                    return setup_err(
                        f"{model}'s Ollama manifest changed during OpenAI contract verification")
                if _ollama_artifact_id(served) != served_artifact_id:
                    return err(
                        f"governed model {served!r} changed during OpenAI contract verification; "
                        "ARA did not unload or delete it")
                processes = ollama.processes()
                residency_error = _ollama_residency_error(
                    processes, allowed_name=served, allowed_digest=expected_served_digest,
                    allowed_context=safe, require_target=True)
                if residency_error is not None:
                    return setup_err(residency_error)
                process = processes[0]
                spilled = process.size_vram_bytes < process.size_bytes

            try:
                activity.record_ollama_serving(
                    **ownership_fields, served_artifact_id=served_artifact_id)
                ownership_recorded = True
            except (OSError, ValueError) as exc:
                return setup_err(
                    f"{served} loaded at {safe} ctx, but ARA ownership could not be recorded: "
                    f"{exc}")
    except locking.OllamaSetupBusy as exc:
        return err(str(exc))
    except BaseException:
        if ownership_recorded:
            c.emit(c.style(
                "warn", "  setup was interrupted after ownership handoff; ARA preserved the "
                "exact ownership record and governed manifest"))
        elif (create_confirmed and served_artifact_id is not None
                and _ollama_artifact_id(served) == served_artifact_id):
            cleanup = _delete_exact_ollama_model(
                served, label="governed", expected_artifact_id=served_artifact_id)
            if cleanup:
                c.emit(c.style("warn", f"  interrupted setup cleanup failed: {cleanup}"))
        raise

    # 7. the handoff — connection info, then ARA exits (the model stays served)
    endpoint = endpoint_base + "/v1"
    if as_json:
        print(json.dumps({"endpoint": endpoint, "model": served, "base_model": model,
                          "served_context": safe, "ceiling_source": source, "spilled": spilled,
                          "residency_verified": residency_verified,
                          "openai_contract_verified": True,
                          "base_artifact_id": base_artifact_id,
                          "served_artifact_id": served_artifact_id,
                          "auto_selected": auto_selected, "stale_ceiling": stale_ceiling,
                          "handoff": {
                              "claim": "verified_point_in_time",
                              "policy_version": _OLLAMA_DERIVED_POLICY_VERSION,
                              **handoff_authority,
                              "continuous_admission": False,
                              "future_reload": "ollama_daemon_policy",
                          },
                          "openai_base_url": endpoint}, indent=2))
        return 0
    c.emit()
    c.emit(c.field("serving", f"{served}  ({model} @ {safe} ctx, {source})"))
    c.emit(c.field("endpoint", f"{endpoint}  (OpenAI-compatible)"))
    c.emit(c.field("use it", f"export OPENAI_BASE_URL={endpoint}"))
    c.emit(c.field("validity", "verified now · later reloads follow Ollama daemon policy"))
    if spilled is True:
        c.emit(c.style("warn", "  note: partially offloaded (size_vram < size) — expect it slow."))
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
    verification = None
    if result.status in {"installed", "refreshed"}:
        verification = engine_audit.audit_engine(
            key, host_features=detect._cpu_features())
    if as_json:
        payload = {"key": result.key, "status": result.status, "detail": result.detail}
        if verification is not None:
            payload["verification"] = verification
        print(json.dumps(payload))
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
    if result.status in _INSTALL_OK:
        if verification is not None:
            build = verification["build"]["status"]
            runtime = verification["runtime"]["status"]
            role = "good" if build == runtime == "matched" else "warn"
            c.emit(c.style(role, "  verified installed engine")
                   + c.style("dim", f" · build {build} · runtime {runtime}"))
            for finding in verification["findings"]:
                c.emit(c.style("warn", f"    {finding['code'].replace('_', ' ')}")
                       + c.style("dim", f" — {finding['detail']}"))
        model = engines.ENGINES[key]["smoke_model"]
        c.emit(c.style("dim", "  next: ")
               + c.style("accent", f"ara characterize {model} --engine {key}"))
        c.emit(c.style("dim", "        downloads a tiny demo model if needed, then measures its "
                               "safe context ceiling"))
    return 0 if result.status in _INSTALL_OK else 1


_ENGINE_BACKED_ACTIVITY_KINDS = frozenset({
    "running", "characterizing", "benchmarking", "serving",
})


def _active_engine_work() -> list[activity.Activity]:
    """Observed ARA work that may own an engine env; persistent Ollama serving owns none."""
    return [
        item for item in activity.snapshot()
        if item.kind in _ENGINE_BACKED_ACTIVITY_KINDS
        and (item.kind != "serving" or item.pid is not None)
    ]


def render_uninstall(c: Console, *, engine: str = "auto", as_json: bool = False) -> int:
    """Remove the matched engine. Exit 0 once it's gone (removed or already absent)."""
    key = engines.resolve(engine)
    if key is None:
        if as_json:
            print(json.dumps({"status": "no_match", "engine": engine}))
        else:
            c.emit(c.style("warn", f"  no engine matches '{engine}' on this hardware"))
        return 1

    pkg = engines.ENGINES[key]["package"]
    active = _active_engine_work()
    if active:
        activities = [{"kind": item.kind, "model": item.model} for item in active]
        if as_json:
            print(json.dumps({
                "status": "busy", "engine": key, "activities": activities,
            }))
        else:
            kinds = ", ".join(dict.fromkeys(item.kind for item in active))
            c.emit(c.style("warn", f"  refusing to remove {pkg} while ARA work is active"))
            c.emit(c.style("dim", f"  active: {kinds}"))
            c.emit(c.style("dim", "  wait for ara status to be idle, then retry"))
        return 1

    result = engines.uninstall(key)
    if as_json:
        print(json.dumps({"key": result.key, "status": result.status,
                          "detail": result.detail}))
        return 0 if result.status in ("removed", "absent") else 1

    if result.status == "removed":
        c.emit(c.style("good", f"  removed {pkg}"))
    elif result.status == "absent":
        c.emit(c.style("dim", f"  {pkg} not installed"))
    else:  # failed
        c.emit(c.style("bad", f"  removing {pkg} failed:"))
        c.emit(c.style("dim", f"  {result.detail}"))
    if c.verbose:
        path = engines.engine_env.env_path(engines.ENGINES[key]["backend"])
        c.emit(c.style("dim", f"  environment: {path}"))
        c.emit(c.style(
            "dim",
            "  kept: models, shared uv cache, ARA database/characterizations, and other engines",
        ))
    return 0 if result.status in ("removed", "absent") else 1


def _emit_characterized(c: Console, engine_key: str | None) -> None:
    """Show models ARA has characterized on this machine + engine (from the store)."""
    if engine_key is None or not db._db_path().is_file():
        return
    with db.connected_readonly() as con:
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
    row = None
    if db._db_path().is_file():
        with db.connected_readonly() as con:
            row = catalog.get(con, model)
    weights_gb = row.get("weights_gb") if row else None
    if weights_gb is None:
        weights_gb = catalog._cache_size_gb(model)
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
    ARA's heuristics to estimate the memory budget and (with ``--model``) checks whether a model's
    weights + context window fit the estimate. It never loads an engine or model and never mutates
    ARA's store; ``characterize`` does that to measure and persist the real ceiling.
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
    measured = None
    if db._db_path().is_file():
        with db.connected_readonly() as con:
            measured = calibration.get_calibration(con, sel.engine_key)
    lim = {"engine": sel.engine_key,
           **estimate.limits(m, measured=measured, backend=sel.backend)}

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

    def _success(payload: dict) -> None:
        out = dict(payload)
        if c.verbose:
            out["token_path"] = str(hf_auth._token_path())
        print(json.dumps(out))

    def _verbose_store() -> None:
        if c.verbose:
            c.emit(c.style("dim", f"  token store: {hf_auth._token_path()}"))

    def _operation_error(action: str, exc: Exception) -> int:
        return _err(f"hugging face {action} failed ({type(exc).__name__})")

    if sub == "login":
        # Warn when token comes from --token (visible in both shell history and process listings).
        if token is not None and not as_json:
            c.emit(c.style("warn",
                           "  note: --token is visible in shell history and process lists; "
                           "prefer the hidden prompt"))
        try:
            if token is None:
                token = _read_token(c)
            if not token or not token.strip():
                return _err("no token provided")
            res = hf_auth.set_token(token)
        except Exception as exc:  # token-store and local Hub failures must not traceback
            return _operation_error("login", exc)
        if not res["saved"]:
            msg = ("that token was rejected by the hub"
                   if res["error"] == "invalid" else "no token provided")
            return _err(msg)
        shadowed = hf_auth._env_token_present()
        if as_json:
            _success({**res, "shadowed_by_env": shadowed})
            return 0
        if res["verified"]:
            message = (f"  stored token verified as {res['user']}" if shadowed
                       else f"  logged in as {res['user']}")
            c.emit(c.style("good", message))
        else:
            c.emit(c.style("warn",
                           f"  token saved — couldn't verify ({res['error']})"))
        if shadowed:
            c.emit(c.style("warn", "  an HF token environment variable remains active and "
                           "overrides the stored token"))
        _verbose_store()
        return 0

    if sub == "logout":
        try:
            res = hf_auth.clear_token()
        except Exception as exc:  # token-store failures are an expected CLI boundary
            return _operation_error("logout", exc)
        if as_json:
            _success(res)
            return 0
        if res["removed"]:
            c.emit(c.style("good", "  removed the stored hugging face token"))
        else:
            c.emit(c.style("dim", "  no stored hugging face token to remove"))
        if res["shadowed_by_env"]:
            c.emit(c.style("warn",
                           "  an HF token environment variable is still active — "
                           "ARA can't unset your environment"))
        _verbose_store()
        return 0

    if sub == "status":
        try:
            st = hf_auth.status()
        except Exception as exc:  # local token lookup / OIDC failures must not traceback
            return _operation_error("status", exc)
        if as_json:
            _success(st)
            return 0
        if not st["present"]:
            c.emit(c.style("dim", "  not logged in — run ")
                   + c.style("accent", "ara hf login")
                   + c.style("dim", " (needed for gated models)"))
            _verbose_store()
            return 0
        if st["verified"]:
            c.emit(c.style("good", f"  logged in as {st['user']}"))
            c.emit(c.style("dim", f"  · token from {st['source']}"))
        else:
            c.emit(c.style("warn",
                           f"  token present ({st['source']}) but couldn't verify ({st['error']})"))
        _verbose_store()
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
        if bool(server_url) != bool(token):
            return _node_err(c, as_json, "usage: ara node enroll <server_url> --token <token>")
        try:
            if server_url and token:
                cfg = config.NodeConfig(server_url=server_url, enrollment_token=token)
            else:
                pending = config.load_pending()
                if pending is None:
                    return _node_err(
                        c, as_json, "usage: ara node enroll <server_url> --token <token>")
                cfg = config.NodeConfig(
                    server_url=pending.server_url, enrollment_token=pending.enrollment_token)
            enroll.enroll_flow(cfg)
        except Exception as exc:                    # surface, don't swallow: enrollment can fail
            return _node_err(c, as_json, f"enrollment failed: {exc}")
        return _node_say(c, as_json, f"enrolled with {cfg.server_url}", endpoint=cfg.server_url)

    if sub == "run":
        try:
            cfg = config.load()
        except (OSError, ValueError) as exc:
            return _node_err(c, as_json, f"cannot load node configuration: {exc}")
        if cfg is None or not cfg.session_token:
            return _node_err(c, as_json,
                             "not enrolled — run: ara node enroll <server_url> --token <token>")
        try:
            config.clear_pending()  # finish cleanup if enrollment crashed after saving authority
        except OSError as exc:
            return _node_err(c, as_json, f"cannot clear completed enrollment state: {exc}")
        try:
            agent.run_loop(cfg)                     # blocks: phone-home work loop until stopped
        except (agent.NodeAgentBusy, agent.CoordinatorWorkRejected) as exc:
            return _node_err(c, as_json, str(exc))
        return _node_say(c, as_json, "run loop exited")

    if sub in ("install", "start", "stop", "status", "uninstall"):
        try:
            if sub in {"install", "start"}:
                cfg = config.load()
                if cfg is None or not cfg.session_token:
                    return _node_err(
                        c, as_json,
                        "not enrolled — run: ara node enroll <server_url> --token <token>",
                    )
            if sub == "install":
                service.install()
                return _node_say(
                    c, as_json,
                    "installed + started (systemd --user; starts with the user manager)",
                )
            if sub == "status":
                if as_json:
                    print(json.dumps({"status": service.status_info()}))
                else:
                    c.emit(service.status().rstrip())
                return 0
            getattr(service, sub)()            # start | stop | uninstall
            return _node_say(c, as_json, f"{sub} ok")
        except (RuntimeError, OSError, ValueError) as exc:
            return _node_err(c, as_json, str(exc))

    return _node_err(c, as_json,
                     "usage: ara node {enroll|run|install|start|stop|status|uninstall}")


def _ollama_doctor_report() -> dict:
    """Read-only Ollama endpoint and persistent-ownership health for ``ara doctor``."""
    endpoint = ollama.endpoint_authority()
    claims = activity.ollama_ownership()
    findings: list[dict] = []

    def finding(code: str, detail: str, *, served_name: str | None = None) -> None:
        row = {"code": code, "severity": "warning", "detail": detail}
        if served_name is not None:
            row["served_name"] = served_name
        findings.append(row)

    report = {
        "endpoint": endpoint.url,
        "scope": endpoint.scope,
        "version": None,
        "ownership_claims": len(claims),
        "findings": findings,
    }
    if endpoint.url is None:
        finding("endpoint_invalid", "the configured Ollama endpoint could not be identified")
        return report
    version = ollama.version()
    report["version"] = version
    models = ollama.inventory() if version is not None else None
    if version is None or models is None:
        finding("runtime_unreadable", "the configured Ollama runtime or inventory is unreadable")
        return report
    authority = _ollama_display_authority(endpoint)
    observed_authority = {
        "runtime_version": authority.server_version,
        "server_instance_id": authority.server_instance_id,
        "configured_inputs": dict(authority.configured_inputs),
        "configured_num_parallel": authority.configured_num_parallel,
        "configured_num_parallel_authority": authority.configured_num_parallel_authority,
    }
    if endpoint.scope != "loopback":
        finding("non_loopback_endpoint", "direct governed execution is limited to loopback Ollama")
    for claim in claims:
        name = claim.served_name or "unknown"
        if claim.policy_version is None or claim.runtime_authority is None:
            finding("legacy_ownership",
                    "ownership predates runtime-bound policy evidence", served_name=name)
        else:
            if claim.policy_version != _OLLAMA_DERIVED_POLICY_VERSION:
                finding("policy_stale", "ownership was created under another derived policy",
                        served_name=name)
            if claim.endpoint != endpoint.url:
                finding("endpoint_mismatch", "ownership belongs to another Ollama endpoint",
                        served_name=name)
            if authority.issue is not None or claim.runtime_authority != observed_authority:
                finding("runtime_authority_stale",
                        "the current daemon no longer matches the recorded handoff authority",
                        served_name=name)
        base = ollama.find_model(models, claim.model) if claim.model is not None else None
        current_base = (_OLLAMA_ARTIFACT_PREFIX + base.digest
                        if base is not None and base.digest is not None else None)
        if current_base != claim.base_artifact_id:
            finding("base_artifact_stale", "the base manifest is missing or changed",
                    served_name=name)
        served = ollama.find_model(models, name)
        current_served = (_OLLAMA_ARTIFACT_PREFIX + served.digest
                          if served is not None and served.digest is not None else None)
        if current_served != claim.served_artifact_id:
            finding("served_artifact_stale", "the derived manifest is missing or changed",
                    served_name=name)
    for model in models:
        if not _is_ara_ollama_derived(model.name):
            continue
        artifact_id = (_OLLAMA_ARTIFACT_PREFIX + model.digest
                       if model.digest is not None else None)
        authoritative = any(
            claim.policy_version == _OLLAMA_DERIVED_POLICY_VERSION
            and claim.served_artifact_id == artifact_id
            and ollama.find_model([model], claim.served_name or "") is not None
            for claim in claims)
        if not authoritative:
            finding("unowned_derived_tag", "ARA-derived tag has no current authoritative claim",
                    served_name=model.name)
    return report


_DOCTOR_TABLES = ("calibrations", "characterizations", "profiles", "benchmark_results")


def render_doctor(c: Console, *, rekey: bool = False, engines: bool = False,
                  as_json: bool = False) -> int:
    """``ara doctor``: this machine's identity (``machine_key``) and the count of stored records
    keyed to it, so a user can see at a glance whether ARA still recognises the box. With
    ``--rekey``, first migrate any lingering legacy (byte-exact) machine_keys to the versioned
    GiB-rounded form and report how many rows moved (the manual counterpart to the automatic
    one-time migration). Rows under *other* keys are counted separately, never folded in (Rule #3).
    ``--engines`` explicitly loads each installed engine runtime (but no model), checking the
    installed build and one minimal device operation. Spec 2026-07-04-machine-key-stabilization."""
    mk = profile.machine_key()
    path = db._db_path()
    try:
        with db.connected() as con:
            rekeyed = db._rekey_legacy(con) if rekey else None
            counts = {t: con.execute(f"SELECT COUNT(*) FROM {t} WHERE machine_key=?",  # noqa: S608
                                     (mk,)).fetchone()[0] for t in _DOCTOR_TABLES}
            other = sum(con.execute(f"SELECT COUNT(*) FROM {t} WHERE machine_key<>?",  # noqa: S608
                                    (mk,)).fetchone()[0] for t in _DOCTOR_TABLES)
            schema_version = con.execute("PRAGMA user_version").fetchone()[0]
            characterization_rows = (db.list_characterizations_for_display(con, mk)
                                     if engines else [])
    except (OSError, sqlite3.Error) as exc:
        msg = f"database problem at {path}: {exc}"
        print(json.dumps({"error": msg, "database": str(path)})) if as_json \
            else c.emit(c.style("bad", f"  {msg}"))
        return 1
    ollama_report = _ollama_doctor_report()
    engine_reports = (engine_audit.audit_installed(
        host_features=detect._cpu_features(), characterization_rows=characterization_rows)
        if engines else None)
    if as_json:
        out: dict = {"machine_key": mk, "counts": counts, "other_keys_rows": other,
                     "ollama": ollama_report}
        if rekey:
            out["rekeyed_rows"] = rekeyed
        if engine_reports is not None:
            out["engines"] = engine_reports
        if c.verbose:
            out.update(database=str(path), schema_version=schema_version)
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
    if c.verbose:
        c.emit(f"    {'database':<20} {path}")
        c.emit(f"    {'schema version':<20} {schema_version}")
    if ollama_report["findings"]:
        c.emit(c.section("  OLLAMA"))
        for finding in ollama_report["findings"]:
            name = f" · {finding['served_name']}" if "served_name" in finding else ""
            c.emit(c.style("warn", f"    {finding['code'].replace('_', ' ')}{name}")
                   + c.style("dim", f" — {finding['detail']}"))
    elif c.verbose:
        c.emit(c.style("dim", "    Ollama ownership: no findings"))
    if engine_reports is not None:
        c.emit(c.section("  INSTALLED ENGINES"))
        if not engine_reports:
            c.emit(c.style("dim", "    none installed"))
        for report in engine_reports:
            version = f"  {report['package_version']}" if report["package_version"] else ""
            c.emit(c.style("accent", f"    {report['key']}{version}"))
            for dimension in ("installation", "build", "runtime", "workload"):
                fact = report[dimension]
                role = ("good" if fact["status"] in {"matched", "verified"}
                        else "warn" if fact["status"] in {"mismatch", "stale"} else "dim")
                c.emit(c.style(role, f"      {dimension} {fact['status']}")
                       + c.style("dim", f" — {fact['detail']}"))
            for finding in report["findings"]:
                c.emit(c.style("warn", f"      {finding['code'].replace('_', ' ')}")
                       + c.style("dim", f" — {finding['detail']}"))
    return 0


_HELP_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _mark_json(ctx: click.Context, as_json: bool) -> Console:
    """Record output mode only after Click has accepted a real command invocation."""
    ctx.find_root().meta["as_json"] = as_json
    return Console.from_env(verbose=ctx.params.get("verbose", False))


def _canonical_engine_arg(value: str | None) -> str | None:
    """Map one-release engine aliases while keeping deprecation text off stdout."""
    if value in engine_identity.LEGACY_ENGINE_ALIASES:
        canonical = engine_identity.LEGACY_ENGINE_ALIASES[value]
        print(f"ara: --engine {value} is deprecated; use --engine {canonical}", file=sys.stderr)
        return canonical
    return value


def _warn_deprecated(alias: str, replacement: str) -> None:
    """Emit one-release command-alias guidance without contaminating stdout."""
    print(f"ara: {alias} is deprecated; use {replacement}", file=sys.stderr)


def _engine_callback(_ctx: click.Context, _param: click.Parameter,
                     value: str | None) -> str | None:
    return _canonical_engine_arg(value)


def _csv_values(_ctx: click.Context, _param: click.Parameter,
                values: tuple[str, ...]) -> list[str]:
    return [item for value in values for item in _csv(value)]


_DETECT_FACETS = {
    "python_facet": "python",
    "apps_facet": "apps",
    "runtime_facet": "runtime",
    "models_facet": "models",
}


def _record_detect_facet(ctx: click.Context, param: click.Parameter, value: bool) -> bool:
    """Keep the first facet in Click's parameter-processing order (the argv order)."""
    if value and "detect_facet" not in ctx.meta:
        ctx.meta["detect_facet"] = _DETECT_FACETS[param.name]
    return value


def _json_verbose_options(func):
    func = click.option("--json", "as_json", is_flag=True,
                        help="Emit machine-readable JSON.")(func)
    return click.option("-v", "--verbose", is_flag=True,
                        help="Show additional detail.")(func)


def _recon_options(func):
    func = click.option("--include", multiple=True, callback=_csv_values, metavar="SECTIONS",
                        help="Include sections (repeatable; comma-separated).")(func)
    func = click.option("--exclude", multiple=True, callback=_csv_values, metavar="SECTIONS",
                        help="Exclude sections (repeatable; comma-separated).")(func)
    return _json_verbose_options(func)


def _engine_option(func):
    return click.option("--engine", callback=_engine_callback, metavar="ENGINE",
                        help=("Select an execution engine. Choices: auto, ollama, mlx, cuda, cpu, "
                              "vulkan, cuda-gguf."))(func)


def _run_engine_option(func):
    return click.option(
        "--engine", callback=_engine_callback, metavar="ENGINE",
        help=("Pin ENGINE; omitted selects the compatible characterized engine with the largest "
              "safe ceiling. Choices: auto, mlx, cuda, cpu, vulkan, cuda-gguf, ollama."),
    )(func)


def _serve_engine_option(func):
    return click.option(
        "--engine", callback=_engine_callback, metavar="ENGINE",
        help=("Serve through ollama, mlx, or auto. Omitted uses Ollama; auto uses native MLX on "
              "Apple Silicon and Ollama elsewhere."),
    )(func)


def _characterize_engine_option(func):
    return click.option(
        "--engine", callback=_engine_callback, metavar="ENGINE",
        help=("Measure with ENGINE; defaults to auto. Choices: auto, mlx, cuda, cpu, vulkan, "
              "cuda-gguf, ollama."),
    )(func)


def _characterize_generation_options(func):
    func = click.option(
        "--kv-quant", default="f16", show_default=True, metavar="FORMAT",
        help="KV-cache format (mlx/cuda/vulkan): f16, q8_0, or q4_0.",
    )(func)
    func = click.option(
        "--weight-quant", default="none", show_default=True, metavar="FORMAT",
        help="CUDA weight format: none, int8, int4, or fp8.",
    )(func)
    func = click.option(
        "--prefill-chunk", type=click.IntRange(min=1), metavar="N",
        help="Positive CUDA prefill chunk size.",
    )(func)
    func = click.option(
        "--chunked-prefill", is_flag=True,
        help=f"Enable CUDA chunked prefill with the default size ({_DEFAULT_PREFILL_CHUNK}).",
    )(func)
    func = click.option(
        "--no-flash-attn", is_flag=True,
        help="Disable Vulkan flash attention.",
    )(func)
    return click.option(
        "--flash-attn", is_flag=True,
        help="Request CUDA FlashAttention 2 (Ampere or newer).",
    )(func)


def _generation_options(func):
    func = click.option(
        "--max-tokens", type=click.IntRange(min=1), default=RUN_MAX_TOKENS,
        show_default=True, metavar="N", help="Maximum new tokens to generate.",
    )(func)
    func = click.option(
        "--kv-quant", default="f16", show_default=True, metavar="FORMAT",
        help="KV-cache format (mlx/cuda/vulkan): f16, q8_0, or q4_0.",
    )(func)
    func = click.option(
        "--weight-quant", default="none", show_default=True, metavar="FORMAT",
        help="CUDA weight format: none, int8, int4, or fp8.",
    )(func)
    func = click.option(
        "--prefill-chunk", type=click.IntRange(min=1), metavar="N",
        help="Positive CUDA prefill chunk size.",
    )(func)
    func = click.option(
        "--chunked-prefill", is_flag=True,
        help=f"Enable CUDA chunked prefill with the default size ({_DEFAULT_PREFILL_CHUNK}).",
    )(func)
    func = click.option("--no-flash-attn", is_flag=True,
                        help="Disable Vulkan flash attention.")(func)
    return click.option("--flash-attn", is_flag=True,
                        help="Request CUDA FlashAttention 2 (Ampere or newer).")(func)


def _prefill_chunk(prefill_chunk: int | None, chunked_prefill: bool) -> int | None:
    return prefill_chunk if prefill_chunk is not None else (
        _DEFAULT_PREFILL_CHUNK if chunked_prefill else None)


@click.group(invoke_without_command=True, context_settings=_HELP_SETTINGS)
@click.option("--version", is_flag=True, is_eager=True, help="Show the installed version and exit.")
@click.pass_context
def _click_cli(ctx: click.Context, version: bool) -> int:
    """AI Runs Anywhere: inspect this machine and run local models safely."""
    if version:
        click.echo(_ara_version())
        ctx.exit(0)
    if ctx.invoked_subcommand is None:
        render_landing(Console.from_env())
    return 0


@_click_cli.command("detect", context_settings=_HELP_SETTINGS,
                    epilog="Examples:\n\n\b\n  ara detect --runtime\n  ara detect --runtime --json")
@click.option("--python", "python_facet", is_flag=True, expose_value=False,
              callback=_record_detect_facet, help="Show Python interpreters.")
@click.option("--apps", "apps_facet", is_flag=True, expose_value=False,
              callback=_record_detect_facet, help="Show installed AI/ML apps.")
@click.option("--runtime", "runtime_facet", is_flag=True, expose_value=False,
              callback=_record_detect_facet, help="Show runtime readiness.")
@click.option("--models", "models_facet", is_flag=True, expose_value=False,
              callback=_record_detect_facet, help="Show cached models.")
@_recon_options
@click.pass_context
def _click_detect(ctx: click.Context, include: list[str], exclude: list[str],
                  verbose: bool, as_json: bool) -> int:
    """Inspect this machine without loading an AI engine."""
    c = _mark_json(ctx, as_json)
    want = _resolve_want("detect", include, exclude, c, as_json=as_json) \
        if (include or exclude) else None
    facet = ctx.meta.get("detect_facet")
    renderer = {"python": render_python, "apps": render_apps, "runtime": render_runtime,
                "models": render_models}.get(facet, render_detect)
    renderer(c, as_json=as_json, want=want)
    return 0


def _invoke_recon(ctx: click.Context, name: str, renderer, include: list[str],
                  exclude: list[str], as_json: bool) -> int:
    c = _mark_json(ctx, as_json)
    want = _resolve_want(name, include, exclude, c, as_json=as_json) \
        if (include or exclude) else None
    renderer(c, as_json=as_json, want=want)
    return 0


@_click_cli.command("status", context_settings=_HELP_SETTINGS)
@_json_verbose_options
@click.pass_context
def _click_status(ctx: click.Context, verbose: bool, as_json: bool) -> int:
    """Show what ARA is doing right now."""
    render_status(_mark_json(ctx, as_json), as_json=as_json)
    return 0


@_click_cli.command("hub", context_settings=_HELP_SETTINGS)
@click.option("--bind", default="127.0.0.1", show_default=True, metavar="ADDRESS",
              help="Host address to expose; use a TLS proxy for remote nodes.")
@click.option("--port", type=click.IntRange(min=1, max=65535), default=3000,
              show_default=True, metavar="PORT", help="Host port for the coordinator.")
@click.option("--data-dir", type=click.Path(file_okay=False, path_type=Path),
              default=hub_server.default_data_dir(), show_default=True, metavar="PATH",
              help="Persistent host directory for coordinator state.")
@click.option("--rebuild", is_flag=True,
              help="Rebuild the coordinator image without Docker's build cache.")
def _click_hub(bind: str, port: int, data_dir: Path, rebuild: bool) -> int:
    """Run the fleet coordinator in Docker and attach to its logs."""
    return render_hub(
        Console.from_env(), bind=bind, port=port, data_dir=data_dir, rebuild=rebuild,
    )


@_click_cli.command("python", hidden=True, context_settings=_HELP_SETTINGS)
@_recon_options
@click.pass_context
def _click_python(ctx: click.Context, include: list[str], exclude: list[str],
                  verbose: bool, as_json: bool) -> int:
    """List Python interpreters and their AI libraries."""
    _warn_deprecated("python", "detect --python")
    return _invoke_recon(ctx, "python", render_python, include, exclude, as_json)


@_click_cli.command("apps", hidden=True, context_settings=_HELP_SETTINGS)
@_recon_options
@click.pass_context
def _click_apps(ctx: click.Context, include: list[str], exclude: list[str],
                verbose: bool, as_json: bool) -> int:
    """List installed AI and ML applications."""
    _warn_deprecated("apps", "detect --apps")
    return _invoke_recon(ctx, "apps", render_apps, include, exclude, as_json)


@_click_cli.command("mlx", hidden=True, context_settings=_HELP_SETTINGS)
@_recon_options
@click.pass_context
def _click_mlx(ctx: click.Context, include: list[str], exclude: list[str],
               verbose: bool, as_json: bool) -> int:
    """Inspect MLX ecosystem readiness."""
    _warn_deprecated("mlx", "detect --runtime")
    return _invoke_recon(ctx, "mlx", render_mlx, include, exclude, as_json)


@click.command("MODEL", hidden=True, context_settings=_HELP_SETTINGS)
@_recon_options
@click.pass_context
def _click_legacy_model(ctx: click.Context, include: list[str], exclude: list[str],
                        verbose: bool, as_json: bool) -> int:
    """Route the one-release ``models MODEL`` compatibility spelling."""
    model_id = ctx.info_name or ""
    _warn_deprecated("models MODEL", "models show MODEL")
    c = _mark_json(ctx, as_json)
    if include or exclude:
        _resolve_want("models", include, exclude, c, as_json=as_json)
    return render_model_detail(c, model_id, as_json=as_json)


class _ModelsGroup(click.Group):
    """Resolve unknown model IDs through the one-release detail compatibility route."""

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None or cmd_name == "list":
            return command
        return _click_legacy_model


@_click_cli.group("models", cls=_ModelsGroup, invoke_without_command=True,
                  no_args_is_help=False, context_settings=_HELP_SETTINGS)
@click.pass_context
def _click_models(ctx: click.Context) -> int:
    """Search the Hub, rank cached models, or inspect one cached model."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
    return 0


@_click_models.command("search", context_settings=_HELP_SETTINGS,
                       epilog='Example:\n  ara models search "small instruct model" --json')
@click.argument("query", nargs=-1, required=True)
@_json_verbose_options
@click.pass_context
def _click_models_search(ctx: click.Context, query: tuple[str, ...],
                         verbose: bool, as_json: bool) -> int:
    """Find models on the Hugging Face Hub."""
    return render_search(_mark_json(ctx, as_json), " ".join(query), as_json=as_json)


@_click_models.command("recommend", context_settings=_HELP_SETTINGS)
@click.option("--engine", type=click.Choice(["ollama"], case_sensitive=False),
              help="Rank artifacts in an external runtime store (currently: ollama).")
@click.option("--use-case", metavar="USE_CASE",
              help="Rank by capability evidence: extraction, reasoning, rag, agentic, or coding.")
@_json_verbose_options
@click.pass_context
def _click_models_recommend(ctx: click.Context, engine: str | None, use_case: str | None,
                            verbose: bool, as_json: bool) -> int:
    """Rank cached models by estimated usable context or capability evidence."""
    return render_recommend(_mark_json(ctx, as_json), as_json=as_json, use_case=use_case,
                            engine=engine)


@_click_models.command("show", context_settings=_HELP_SETTINGS)
@click.argument("model")
@click.option("--engine", type=click.Choice(["ollama"], case_sensitive=False),
              help="Inspect MODEL in an external runtime store (currently: ollama).")
@_json_verbose_options
@click.pass_context
def _click_models_show(ctx: click.Context, model: str, engine: str | None,
                       verbose: bool, as_json: bool) -> int:
    """Show cached architecture and this machine's measured ceilings."""
    return render_model_detail(_mark_json(ctx, as_json), model, as_json=as_json, engine=engine)


@_click_cli.command("search", hidden=True, context_settings=_HELP_SETTINGS)
@click.argument("query", nargs=-1, required=True)
@_json_verbose_options
@click.pass_context
def _click_search(ctx: click.Context, query: tuple[str, ...], verbose: bool, as_json: bool) -> int:
    """Find models on the Hugging Face Hub."""
    _warn_deprecated("search", "models search")
    return render_search(_mark_json(ctx, as_json), " ".join(query), as_json=as_json)


@_click_cli.command("characterize", context_settings=_HELP_SETTINGS)
@click.argument("model")
@_characterize_engine_option
@_characterize_generation_options
@_json_verbose_options
@click.pass_context
def _click_characterize(ctx: click.Context, model: str, engine: str | None, kv_quant: str,
                        weight_quant: str, prefill_chunk: int | None,
                        chunked_prefill: bool, no_flash_attn: bool, flash_attn: bool,
                        verbose: bool, as_json: bool) -> int:
    """Safely measure MODEL's real context ceiling by loading it on an engine.

    ARA ramps context within the selected engine's safety boundary, then stores the measured
    ceiling as evidence for later governed operations. Omit tuning options to use ARA's safe
    defaults.
    """
    c = _mark_json(ctx, as_json)
    with locking.measurement_lock():
        return render_characterize(
            c, model, engine=engine, as_json=as_json, flash_attn=not no_flash_attn,
            flash_attn_optin=flash_attn, kv_quant=kv_quant, weight_quant=weight_quant,
            prefill_chunk=_prefill_chunk(prefill_chunk, chunked_prefill),
        )


@_click_cli.command("profile", context_settings=_HELP_SETTINGS)
@click.option("--model", metavar="MODEL",
              help="Estimate whether MODEL fits and its usable context.")
@click.option("--engine", callback=_engine_callback, metavar="ENGINE",
              help="Estimate for ENGINE; defaults to the detected engine.")
@_json_verbose_options
@click.pass_context
def _click_profile(ctx: click.Context, model: str | None, engine: str | None,
                   verbose: bool, as_json: bool) -> int:
    """Estimate this machine's safe memory budget without loading an engine or model."""
    return render_profile(_mark_json(ctx, as_json), as_json=as_json, model=model, engine=engine)


@_click_cli.command("recommend", hidden=True, context_settings=_HELP_SETTINGS)
@click.option("--use-case", help="Rank by a measured capability dimension.")
@_json_verbose_options
@click.pass_context
def _click_recommend(ctx: click.Context, use_case: str | None,
                     verbose: bool, as_json: bool) -> int:
    """Rank cached models that fit this machine."""
    _warn_deprecated("recommend", "models recommend")
    return render_recommend(_mark_json(ctx, as_json), as_json=as_json, use_case=use_case)


@_click_cli.command("run", context_settings=_HELP_SETTINGS,
                    epilog='Example:\n  ara run org/model "Explain this" --json')
@click.argument("model")
@click.argument("prompt", nargs=-1, required=True)
@_run_engine_option
@click.option("-y", "--yes", "assume_yes", is_flag=True, help="Skip confirmation prompts.")
@_generation_options
@_json_verbose_options
@click.pass_context
def _click_run(ctx: click.Context, model: str, prompt: tuple[str, ...], engine: str | None,
               assume_yes: bool, max_tokens: int, kv_quant: str, weight_quant: str,
               prefill_chunk: int | None, chunked_prefill: bool, no_flash_attn: bool,
               flash_attn: bool, verbose: bool, as_json: bool) -> int:
    """Generate one governed completion under MODEL's characterized safe ceiling.

    ARA selects a compatible characterized native engine unless --engine pins one. Pin ollama to
    use the existing local daemon with exact reusable evidence. ARA refuses before loading when
    requested settings do not match the measurement. Omit tuning options to use ARA's safe
    defaults.
    """
    return render_run(
        _mark_json(ctx, as_json), model, prompt=" ".join(prompt) or None, engine=engine,
        assume_yes=assume_yes, as_json=as_json, max_tokens=max_tokens,
        flash_attn=not no_flash_attn,
        flash_attn_optin=flash_attn, kv_quant=kv_quant, weight_quant=weight_quant,
        prefill_chunk=_prefill_chunk(prefill_chunk, chunked_prefill),
    )


@_click_cli.command("serve", context_settings=_HELP_SETTINGS)
@click.argument("model", required=False)
@click.option("--ctx", "serve_ctx", type=click.IntRange(min=1), metavar="N",
              help="Context cap; never above this model's measured or estimated safe bound.")
@click.option("--name", "serve_name", metavar="NAME",
              help="Existing exactly ARA-owned Ollama name to reuse; omit for a new service.")
@_serve_engine_option
@click.option("-y", "--yes", "assume_yes", is_flag=True, help="Skip confirmation prompts.")
@_json_verbose_options
@click.pass_context
def _click_serve(ctx: click.Context, model: str | None, serve_ctx: int | None,
                 serve_name: str | None, engine: str | None, assume_yes: bool,
                 verbose: bool, as_json: bool) -> int:
    """Serve MODEL behind a governed OpenAI-compatible endpoint.

    With no MODEL, ARA selects the best-fitting model already in Ollama. Ollama hands the endpoint
    off and exits; MLX serves in the foreground until stopped.
    """
    return render_serve(_mark_json(ctx, as_json), model, ctx=serve_ctx,
                        name=serve_name or None, engine=engine, assume_yes=assume_yes,
                        as_json=as_json)


_USE_CASE = click.Choice(["coding", "reasoning", "agentic", "extraction", "rag"])


@_click_cli.command(
    "benchmark",
    context_settings=_HELP_SETTINGS,
    short_help="Measure MODEL under its characterized safe ceiling.",
)
@click.argument("model")
@click.option("--use-case", type=_USE_CASE, required=True,
              help="Measured capability category.")
@_engine_option
@click.option("--ctx", "serve_ctx", type=click.IntRange(min=1), metavar="N",
              help="Lower probe context; never above the measured safe ceiling.")
@click.option("--max-tokens", type=click.IntRange(min=1), metavar="N",
              help="Maximum new tokens generated for each probe.")
@click.option("--repeat", "repeat_count", type=click.IntRange(min=1), default=1,
              show_default=True, metavar="N", help="Independent runs used for a variance band.")
@click.option("--exec-consent", is_flag=True,
              help="Authorize execution of coding-probe output.")
@click.option("-y", "--yes", "assume_yes", is_flag=True, help="Skip confirmation prompts.")
@_json_verbose_options
@click.pass_context
def _click_benchmark(ctx: click.Context, model: str, use_case: str, engine: str | None,
                     serve_ctx: int | None, max_tokens: int | None, repeat_count: int,
                     exec_consent: bool, assume_yes: bool, verbose: bool, as_json: bool) -> int:
    """Run a judge-free capability probe against MODEL's actual quant on the selected engine,
    then store the measured capability evidence.

    Requires a prior matching characterization; --ctx may lower, but never replace or exceed,
    that measured safe ceiling. Coding probes execute model-written Python only with
    --exec-consent.
    """
    c = _mark_json(ctx, as_json)
    with locking.measurement_lock():
        return render_benchmark(
            c, model, use_case=use_case, engine=engine, ctx=serve_ctx,
            max_tokens=max_tokens, repeat=repeat_count, assume_yes=assume_yes,
            exec_consent=exec_consent, as_json=as_json,
        )


def _selected_engine(engine_arg: str | None, engine_option: str | None) -> str:
    return _canonical_engine_arg(engine_arg) if engine_arg is not None else (engine_option or "auto")


@_click_cli.command("install", context_settings=_HELP_SETTINGS)
@click.argument("engine_arg", required=False, metavar="[ENGINE]")
@click.option("--engine", "engine_option", callback=_engine_callback, metavar="ENGINE",
              help="Engine to install (also accepted positionally).")
@click.option("--refresh", is_flag=True, help="Reinstall even when already present.")
@_json_verbose_options
@click.pass_context
def _click_install(ctx: click.Context, engine_arg: str | None, engine_option: str | None,
                   refresh: bool, verbose: bool, as_json: bool) -> int:
    """Install an engine on demand.

    Engines: auto, mlx, cuda, cpu, vulkan, cuda-gguf.

    Decision guide: ara install --engine --help
    """
    return render_install(_mark_json(ctx, as_json),
                          engine=_selected_engine(engine_arg, engine_option),
                          refresh=refresh, as_json=as_json)


@_click_cli.command("uninstall", context_settings=_HELP_SETTINGS)
@click.argument("engine_arg", required=False, metavar="[ENGINE]")
@click.option("--engine", "engine_option", callback=_engine_callback, metavar="ENGINE",
              help="Engine to remove (also accepted positionally).")
@_json_verbose_options
@click.pass_context
def _click_uninstall(ctx: click.Context, engine_arg: str | None, engine_option: str | None,
                     verbose: bool, as_json: bool) -> int:
    """Remove an installed engine environment.

    Engines: auto, mlx, cuda, cpu, vulkan, cuda-gguf.

    Keeps models, the shared uv cache, ARA's database and characterizations, and other engines.
    """
    return render_uninstall(_mark_json(ctx, as_json),
                            engine=_selected_engine(engine_arg, engine_option), as_json=as_json)


@_click_cli.command(
    "doctor",
    context_settings=_HELP_SETTINGS,
    short_help="Diagnose ARA's stored identity and records for this machine.",
)
@click.option("--rekey", is_flag=True,
              help="Rewrite legacy machine identity keys in ARA's database.")
@click.option("--engines", "audit_engines", is_flag=True,
              help="Load installed engine runtimes without loading a model, then verify them.")
@_json_verbose_options
@click.pass_context
def _click_doctor(ctx: click.Context, rekey: bool, audit_engines: bool,
                  verbose: bool, as_json: bool) -> int:
    """Show how ARA identifies this machine, count records stored for it, and report records
    under other machine identities."""
    return render_doctor(
        _mark_json(ctx, as_json), rekey=rekey, engines=audit_engines, as_json=as_json)


@_click_cli.group("hf", no_args_is_help=False, context_settings=_HELP_SETTINGS)
def _click_hf() -> None:
    """Manage Hugging Face authentication for gated model access."""


@_click_hf.command("login", context_settings=_HELP_SETTINGS)
@click.option("--token", help="Token to store (visible in shell history and process lists).")
@_json_verbose_options
@click.pass_context
def _click_hf_login(ctx: click.Context, token: str | None,
                    verbose: bool, as_json: bool) -> int:
    """Store and verify a Hugging Face token for gated models."""
    return render_hf(_mark_json(ctx, as_json), "login", token=token, as_json=as_json)


@_click_hf.command("logout", context_settings=_HELP_SETTINGS)
@_json_verbose_options
@click.pass_context
def _click_hf_logout(ctx: click.Context, verbose: bool, as_json: bool) -> int:
    """Remove the locally stored Hugging Face token."""
    return render_hf(_mark_json(ctx, as_json), "logout", as_json=as_json)


@_click_hf.command("status", context_settings=_HELP_SETTINGS)
@_json_verbose_options
@click.pass_context
def _click_hf_status(ctx: click.Context, verbose: bool, as_json: bool) -> int:
    """Check whether a Hugging Face token is active and verified."""
    return render_hf(_mark_json(ctx, as_json), "status", as_json=as_json)


@_click_cli.group("node", no_args_is_help=False, context_settings=_HELP_SETTINGS)
def _click_node() -> None:
    """Run or manage the push-only ARA node daemon."""


@_click_node.command("enroll", context_settings=_HELP_SETTINGS,
                     epilog="Example:\n  ara node enroll https://ara.example --token TOKEN")
@click.argument("server_url", required=False)
@click.option("--token", help="One-time coordinator enrollment token.")
@_json_verbose_options
@click.pass_context
def _click_node_enroll(ctx: click.Context, server_url: str | None, token: str | None,
                       verbose: bool, as_json: bool) -> int:
    """Enroll this node with a coordinator; omit arguments to resume a pending enrollment."""
    rest = ["node", "enroll", *([server_url] if server_url else [])]
    return render_node(_mark_json(ctx, as_json), rest,
                       token=token, as_json=as_json)


def _invoke_node(ctx: click.Context, name: str, as_json: bool) -> int:
    return render_node(_mark_json(ctx, as_json), ["node", name], as_json=as_json)


@_click_node.command("run", context_settings=_HELP_SETTINGS)
@_json_verbose_options
@click.pass_context
def _click_node_run(ctx: click.Context, verbose: bool, as_json: bool) -> int:
    """Run the push-only node work loop."""
    return _invoke_node(ctx, "run", as_json)


@_click_node.command("install", context_settings=_HELP_SETTINGS)
@_json_verbose_options
@click.pass_context
def _click_node_install(ctx: click.Context, verbose: bool, as_json: bool) -> int:
    """Install and start the Linux user service.

    The service starts with this user's systemd manager. Headless boot before login requires
    administrator-configured user lingering; ARA does not enable that policy automatically.
    """
    return _invoke_node(ctx, "install", as_json)


@_click_node.command("start", context_settings=_HELP_SETTINGS)
@_json_verbose_options
@click.pass_context
def _click_node_start(ctx: click.Context, verbose: bool, as_json: bool) -> int:
    """Start the user service."""
    return _invoke_node(ctx, "start", as_json)


@_click_node.command("stop", context_settings=_HELP_SETTINGS)
@_json_verbose_options
@click.pass_context
def _click_node_stop(ctx: click.Context, verbose: bool, as_json: bool) -> int:
    """Stop the user service."""
    return _invoke_node(ctx, "stop", as_json)


@_click_node.command("status", context_settings=_HELP_SETTINGS)
@_json_verbose_options
@click.pass_context
def _click_node_status(ctx: click.Context, verbose: bool, as_json: bool) -> int:
    """Show user-service status."""
    return _invoke_node(ctx, "status", as_json)


@_click_node.command("uninstall", context_settings=_HELP_SETTINGS)
@_json_verbose_options
@click.pass_context
def _click_node_uninstall(ctx: click.Context, verbose: bool, as_json: bool) -> int:
    """Remove the user service."""
    return _invoke_node(ctx, "uninstall", as_json)


def _install_help_request(args: list[str]) -> tuple[str | None, bool] | None:
    """Recognize only the contextual ``install --engine ... --help`` dialect."""
    if not args or args[0] != "install" or not any(a in ("-h", "--help") for a in args[1:]):
        return None
    verbose = any(a in ("-v", "--verbose") for a in args[1:])
    tail = [a for a in args[1:] if a not in ("-h", "--help", "-v", "--verbose")]
    if tail == ["--engine"]:
        return None, verbose
    if len(tail) == 1 and tail[0].startswith("--engine="):
        return tail[0].partition("=")[2], verbose
    if len(tail) == 2 and tail[0] == "--engine":
        return tail[1], verbose
    return None


def _engine_help_entry(key: str, *, verbose: bool) -> None:
    """Render one concrete catalog engine without importing or validating it."""
    engine = engines.ENGINES[key]
    click.echo(f"Engine: {key}")
    click.echo(f"  Purpose: {engine['purpose']}")
    click.echo(f"  Hardware: {engine['hardware']}")
    click.echo(f"  Models: {engine['formats']}")
    click.echo(f"  Installs: {engine['install_summary']}")
    click.echo(f"  Note: {engine['caution']}")
    if verbose:
        plan = engines.install_plan(key)
        source_env = engine.get("source_env")
        click.echo(f"  Backend/env: {plan.backend}")
        click.echo(f"  Python: {plan.python or 'default'}")
        click.echo(f"  Platform: {plan.platform}")
        click.echo(f"  Install arguments: {' '.join(plan.targets)}")
        if source_env:
            click.echo(f"  Source override: {plan.source_override or f'none ({source_env})'}")
        click.echo(f"  Environment schema: {plan.schema or 'built-in worker'}")
    click.echo()


def render_install_engine_help(engine: str | None, *, verbose: bool) -> int:
    """Render all-engine or focused install guidance from catalog and plan data."""
    selected = _canonical_engine_arg(engine)
    if selected is not None and selected not in ("auto", *engines.ENGINES):
        root_ctx = click.Context(_click_cli, info_name="ara")
        install_ctx = click.Context(_click_install, info_name="install", parent=root_ctx)
        choices = ", ".join(("auto", *engines.ENGINES))
        raise click.BadParameter(
            f"{selected!r} is not an engine; choose from {choices}",
            ctx=install_ctx, param_hint="--engine",
        )

    click.echo("ARA engine install guide")
    click.echo()
    if selected is None:
        decision = engines.auto_decision()
        click.echo("Engine: auto")
        click.echo(f"  Selected: {decision.key or 'no automatic match'}")
        click.echo(f"  Why: {decision.reason}")
        click.echo()
        for key in engines.ENGINES:
            _engine_help_entry(key, verbose=verbose)
        return 0

    if selected == "auto":
        decision = engines.auto_decision()
        click.echo("Engine: auto")
        click.echo(f"  Selected: {decision.key or 'no automatic match'}")
        click.echo(f"  Why: {decision.reason}")
        if decision.key is None:
            click.echo("  Choose cpu, vulkan, cuda-gguf, or another engine explicitly.")
            click.echo()
            return 0
        click.echo()
        _engine_help_entry(decision.key, verbose=verbose)
        return 0

    _engine_help_entry(selected, verbose=verbose)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry. Front-door honesty guard (Rule #3): an exception that escapes a command under
    ``--json`` becomes a structured ``{"error": ...}`` instead of a raw traceback a JSON consumer
    can't parse. Without ``--json``, an :class:`~ara.engine_env.EngineEnvError` (the common
    engine-env failure — a broken/missing env, a dead worker) prints a friendly one-line diagnostic
    instead of a raw traceback; any other exception still propagates. KeyboardInterrupt / SystemExit
    are not caught."""
    args = list(sys.argv[1:] if argv is None else argv)
    ctx: click.Context | None = None
    try:
        if (request := _install_help_request(args)) is not None:
            engine, verbose = request
            return render_install_engine_help(engine, verbose=verbose)
        with _click_cli.make_context("ara", args) as ctx:
            result = _click_cli.invoke(ctx)
        return int(result or 0)
    except click.exceptions.Exit as exc:
        return exc.exit_code
    except click.ClickException as exc:
        exc.show(file=sys.stderr)
        return exc.exit_code
    except Exception as exc:   # noqa: BLE001 — deliberate front-door honesty guard
        as_json = bool(ctx and ctx.meta.get("as_json"))
        if isinstance(exc, MeasurementBusy):   # a concurrent measurement holds the lock — say so
            print(json.dumps({"error": str(exc)})) if as_json \
                else Console.from_env().emit(Console.from_env().style("warn", f"  {exc}"))
            return 1
        if as_json:
            print(json.dumps({"error": f"ara failed: {exc}"}))
            return 1
        if isinstance(exc, EngineEnvError):
            c = Console.from_env()
            c.emit(c.style("bad", f"  engine env problem: {exc}"))
            c.emit(c.style("dim", "  check the GPU driver / toolchain and retry: ara install"))
            return 1
        raise


if __name__ == "__main__":   # pragma: no cover — compatibility only; production uses `python -m ara`
    sys.exit(main())
