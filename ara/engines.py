# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The engine catalog and hardware-matched resolution.

ARA's core is engine-free; every hardware engine is installed on demand (`ara install`) into
its **own isolated uv env** under the data dir (``ara/engine_env.py``), never into ARA's venv.
That keeps the core universal, the lock engine-free, and incompatible toolchains (torch-CUDA
vs torch-ROCm, MLX vs llama.cpp) from ever colliding. ARA drives an engine over a subprocess
worker in its env — it never imports one in-process.

Two kinds of engine live here:
  * **external suites** — the two heavyweights that get their own repos (``wmx`` = MLX/Apple,
    ``wcx`` = CUDA); installed from a git ``spec``.
  * **built-in engines** — everything else ships in the ARA repo (only the worker's heavy deps
    install into the env); described by a ``packages`` list and ``builtin: True``.

Read-only resolution (``for_hardware``/``resolve``) does no I/O; install/uninstall delegate to
:mod:`ara.engine_env`.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
from dataclasses import dataclass

from ara import engine_env

# Short, stable handles → how to install each. ``backend`` is both the adapter module name and
# the isolated env name. ``available`` is False for engines whose suite isn't shippable yet.
ENGINES: dict[str, dict] = {
    "wmx": {
        "backend": "apple",
        "package": "wmx-suite",
        "available": True,
        "spec": "git+https://github.com/willsarg/wmx-suite",
        "python": "3.12",          # wmx-suite requires >=3.12
        "model_kinds": ("transformers",),
    },
    "wcx": {
        "backend": "cuda",
        "package": "wcx-suite",
        # Converted to the isolated-env worker model (backends/cuda.py drives wcx-suite's
        # device + measure_one workers out-of-process; nothing torch-shaped loads in ARA).
        "available": True,
        "spec": "git+https://github.com/willsarg/wcx-suite",
        "extras": "cuda",                        # pulls torch + transformers
        # uv auto-detects the GPU and picks the matching CUDA torch wheel (the default
        # PyPI torch on Windows/Linux is CPU-only).
        "pip_args": ["--torch-backend=auto"],
        "python": "3.12",
        "model_kinds": ("transformers",),
    },
    "cpu": {
        "backend": "cpu",
        "package": "llama.cpp",
        "available": True,
        "builtin": True,           # worker ships in ARA; only its deps install into the env
        "packages": ["llama-cpp-python>=0.3", "psutil", "huggingface_hub"],
        "model_kinds": ("gguf",),
        # llama-cpp-python ships NO PyPI wheels, so a stock Windows box (no MSVC) can't build
        # it from source. On Windows pull a prebuilt CPU wheel from the project's own index;
        # `--only-binary` (added in _install_targets) makes that deterministic. Scoped to
        # Windows so Linux/macOS/aarch64 (Pi) keep the source build — the abetlen index has no
        # wheel for them, and those platforms ship a toolchain.
        #   max_version: abetlen's wheels after 0.3.19 are static AVX-512 builds that fault
        #   (illegal instruction, 0xc000001d) on the many x86 CPUs without AVX-512 — e.g. AMD
        #   Zen 1–3. 0.3.19 is the newest AVX2-baseline wheel, so it runs on essentially any
        #   x86-64. Native builds elsewhere pick the host's own ISA, so this cap is Windows-only.
        "wheel_only": {
            "llama-cpp-python": {
                "index": "https://abetlen.github.io/llama-cpp-python/whl/cpu",
                "max_version": "0.3.19",
            },
        },
        "wheel_platforms": ("Windows",),   # the abetlen CPU index serves only Windows wheels
        "python": "3.12",
    },
    "vulkan": {
        "backend": "vulkan",
        "package": "llama.cpp (Vulkan)",
        "available": True,
        "builtin": True,           # worker ships in ARA; only its deps install into the env
        "packages": ["llama-cpp-python>=0.3", "psutil", "huggingface_hub"],
        "model_kinds": ("gguf",),
        # GPU-offload GGUF via llama.cpp's Vulkan backend (opt-in, --engine vulkan). Prebuilt
        # Vulkan wheels exist for x86_64 Linux + Windows on the project's own index (default PyPI
        # ships no llama-cpp-python wheel at all). We MUST force the prebuilt Vulkan wheel from
        # that index: a plain `llama-cpp-python` install hands back the `cpu` engine's CPU-only
        # wheel from uv's cache (same version, no GGML_VULKAN) — verified. `--only-binary` (added
        # in _builtin_targets) makes it deterministic (no silent source-build fallback); pinned to
        # the newest Vulkan wheel. Kept AFTER `cpu` so engine_for_model's GGUF default stays `cpu`.
        "wheel_only": {
            "llama-cpp-python": {
                "index": "https://abetlen.github.io/llama-cpp-python/whl/vulkan",
                "max_version": "0.3.31",
            },
        },
        "wheel_platforms": ("Linux", "Windows"),
        "python": "3.12",
    },
}


def for_hardware() -> str | None:
    """The engine ARA would pick for this machine from light recon, or None.

    Deliberately cheap — no subprocess: Apple Silicon by ``platform``, NVIDIA by a
    bare ``nvidia-smi`` on PATH. This is the resolution behind ``--engine auto``.
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "wmx"
    if shutil.which("nvidia-smi"):
        return "wcx"
    return None


def for_backend(backend: str) -> str | None:
    """The engine key whose backend matches *backend* (e.g. 'cuda' → 'wcx'), or None.
    One place maps hardware backends to engines, shared by detect and the registry."""
    return next((k for k, e in ENGINES.items() if e["backend"] == backend), None)


def resolve(value: str) -> str | None:
    """Map an ``--engine`` value to a concrete engine key, or None if it doesn't
    name one. ``auto`` defers to :func:`for_hardware`; explicit keys pass through."""
    if value == "auto":
        return for_hardware()
    return value if value in ENGINES else None


def is_installed(key: str) -> bool:
    """Is engine *key*'s isolated env present? Cheap — just checks the env's python exists,
    never imports the engine. Unknown keys are simply 'not installed'."""
    engine = ENGINES.get(key)
    return engine is not None and engine_env.exists(engine["backend"])


def source_for(key: str) -> str:
    """The install source for an external engine *key*: its git spec, or a dev override.

    Setting ``ARA_<KEY>_SOURCE`` (e.g. ``ARA_WMX_SOURCE=../wmx-suite``) replaces the
    git URL — lets a developer install from a local checkout instead of cloning."""
    override = os.environ.get(f"ARA_{key.upper()}_SOURCE")
    return override or ENGINES[key]["spec"]


@dataclass(frozen=True)
class InstallResult:
    """Outcome of an install/uninstall attempt — also the shape behind ``--json``."""
    key: str
    status: str       # installed | already | coming_soon | unknown | failed | removed | absent
    detail: str = ""


def _cap_to_wheel_max(req: str, wheels: dict) -> str:
    """Append a wheel-compatible version ceiling to requirement *req* if it's a ``wheel_only``
    package with a ``max_version`` (the newest prebuilt wheel on the engine's index).
    Leaves the requirement's own floor intact (``llama-cpp-python>=0.3`` → ``…>=0.3,<=0.3.19``).
    """
    name = re.match(r"[A-Za-z0-9._-]+", req).group(0)
    spec = wheels.get(name)
    if spec and spec.get("max_version"):
        return f"{req},<={spec['max_version']}"
    return req


def _builtin_targets(engine: dict) -> list[str]:
    """uv-pip args for a builtin engine: its package list, with prebuilt-wheel handling.

    On the platforms an engine lists in ``wheel_platforms``, every ``wheel_only`` package is
    forced to a prebuilt wheel from its index (``--only-binary`` makes that deterministic — no
    silent source-build fallback) and capped at its ``max_version``. This covers two cases: the
    ``cpu`` engine on Windows (no MSVC → can't source-build), and the ``vulkan`` engine on
    x86_64 Linux/Windows (a plain install would resolve to the CPU-only wheel from cache). On any
    other platform the source build picks the host's own ISA, so the list passes through untouched.
    """
    wheels = engine.get("wheel_only") or {}
    if not wheels or platform.system() not in engine.get("wheel_platforms", ()):
        return list(engine["packages"])
    flags: list[str] = []
    for pkg, spec in wheels.items():
        flags += ["--only-binary", pkg, "--extra-index-url", spec["index"]]
    return [*flags, *(_cap_to_wheel_max(req, wheels) for req in engine["packages"])]


def _install_targets(key: str) -> list[str]:
    """The trailing ``uv pip install`` args (pip flags + targets) for engine *key*'s env.

    Built-in engines install a plain package list (with per-platform prebuilt-wheel handling — see
    :func:`_builtin_targets`). External suites install their source: a git/remote ``spec``
    (folding in an ``extras`` group via a PEP 508 direct reference) or a local path installed
    editable (``-e``) for dev. Any ``pip_args`` (e.g. ``--torch-backend=auto`` to fetch the
    right CUDA torch wheel) come first.
    """
    engine = ENGINES[key]
    if engine.get("builtin"):
        return _builtin_targets(engine)
    pip_args = list(engine.get("pip_args", []))
    extras = engine.get("extras")
    suffix = f"[{extras}]" if extras else ""
    source = source_for(key)
    if source.startswith(("git+", "http://", "https://")):
        # PEP 508 direct reference: ``name[extra] @ git+url``
        target = f"{engine['package']}{suffix} @ {source}" if extras else source
        return [*pip_args, target]
    return [*pip_args, "-e", f"{source}{suffix}"]


def install(key: str) -> InstallResult:
    """Install engine *key* into its own isolated uv env. Idempotent and honest:
    never creates an env for an unknown or not-yet-available engine."""
    if key not in ENGINES:
        return InstallResult(key, "unknown")
    engine = ENGINES[key]
    if not engine["available"]:
        return InstallResult(key, "coming_soon", f"{engine['package']} isn't available yet")
    if is_installed(key):
        return InstallResult(key, "already")
    try:
        engine_env.create(engine["backend"], _install_targets(key),
                          python=engine.get("python"))
    except engine_env.EngineEnvError as e:
        return InstallResult(key, "failed", str(e))
    return InstallResult(key, "installed")


def uninstall(key: str) -> InstallResult:
    """Remove engine *key*'s isolated env. No-op when it isn't an engine or isn't installed.
    The shared uv cache and other engines' envs are untouched."""
    if key not in ENGINES:
        return InstallResult(key, "unknown")
    if not is_installed(key):
        return InstallResult(key, "absent")
    engine_env.remove(ENGINES[key]["backend"])
    return InstallResult(key, "removed")


def _is_gguf(ref: str) -> bool:
    """Confident GGUF signal: a .gguf file path, or a repo:file.gguf reference."""
    tail = ref.split(":", 1)[1] if ":" in ref else ref
    return tail.endswith(".gguf")


def engine_for_model(ref: str) -> str | None:
    """The engine key best suited to *ref*, or None when it can't be told cheaply.

    Only a confident GGUF signal classifies; a bare repo id (even one named '...-GGUF')
    returns None so the caller falls back to the engine's own preflight error."""
    if _is_gguf(ref):
        return next((k for k, e in ENGINES.items() if "gguf" in e.get("model_kinds", ())), None)
    return None
