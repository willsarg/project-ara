# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The engine catalog and hardware-matched resolution.

ARA's core is engine-free; every hardware engine is installed on demand (`ara install`) into
its **own isolated uv env** under the data dir (``ara/engine_env.py``), never into ARA's venv.
That keeps the core universal, the lock engine-free, and incompatible toolchains (torch-CUDA
vs torch-ROCm, MLX vs llama.cpp) from ever colliding. ARA drives an engine over a subprocess
worker in its env — it never imports one in-process.

Two kinds of engine live here:
  * **vendored suites** — the MLX and CUDA heavyweights ship under ``ara/_vendor`` and install
    from that exact source into isolated environments.
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
from importlib import metadata
from pathlib import Path

from ara import engine_env, engine_identity


def _ara_version() -> str:
    """The installed project-ara version (also behind ``ara --version``), or a sentinel when running
    from an un-installed source tree with no distribution metadata. Stamped into an engine env at
    install and compared on the next ``ara install`` to detect a stale vendored engine — a newer ARA
    wheel carries newer ``ara/_vendor/*`` source that must reach a box that already has the env."""
    try:
        return metadata.version("project-ara")
    except metadata.PackageNotFoundError:
        return "0+unknown"

# Short, stable handles → how to install each. ``backend`` is both the adapter module name and
# the isolated env name. ``available`` is False for engines whose suite isn't shippable yet.
ENGINES: dict[str, dict] = {
    "mlx": {
        "backend": "apple",
        "package": "ara-engine-mlx",
        "available": True,
        "source_dir": "_vendor/wmx",
        "source_env": "ARA_MLX_SOURCE",
        "legacy_source_env": "ARA_WMX_SOURCE",
        # Vendored: the wmx_suite source ships in ARA's wheel under ara/_vendor/wmx and installs into
        # the isolated `apple` env from there — no git fetch at install time, so a release is
        # reproducible from the wheel alone. Folded 2026-06-30 from wmx-suite@374c47d (the #107
        # single-BOS + turn-end stop fixes). Re-vendor via scripts/vendor_engine.py to bump.
        # ARA_WMX_SOURCE still overrides to a local checkout (editable) for engine dev. MLX +
        # transformers stay engine-env-only — never ARA dependencies.
        "vendored": True,
        "python": "3.12",          # wmx-suite requires >=3.12
        "model_kinds": ("transformers",),
    },
    "cuda": {
        "backend": "cuda",
        "package": "ara-engine-cuda",
        # Converted to the isolated-env worker model (backends/cuda.py drives wcx-suite's
        # device + measure_one workers out-of-process; nothing torch-shaped loads in ARA).
        "available": True,
        "source_dir": "_vendor/wcx",
        "source_env": "ARA_CUDA_SOURCE",
        "legacy_source_env": "ARA_WCX_SOURCE",
        # Vendored (see the wmx note): wcx_suite ships in ARA's wheel under ara/_vendor/wcx and
        # installs into the isolated `cuda` env from there. Folded 2026-06-30 from wcx-suite@3a43f63.
        "vendored": True,
        "extras": "cuda",                        # pulls torch + transformers (into the env, not ARA)
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
        #   max_version: 0.3.19 is the last Windows CPU wheel that actually RUNS via this binding.
        #   The post-0.3.19 /whl/cpu Windows wheels (py3-none-win_amd64) ship a split/runtime-loaded
        #   ggml backend that llama-cpp-python doesn't initialize (upstream #2069, "no backends
        #   loaded"). Verified 2026-06-30 on willw11 (Zen-3, cp312): 0.3.21 imports but fails to
        #   load ANY model (SmolLM2 and gemma-4 both "Failed to load model from file"); 0.3.32's
        #   llama.dll fails to load outright (WinError 127). (Historically these were also flagged
        #   as AVX-512 builds that fault on non-AVX-512 CPUs — 0xc000001d.) Native builds elsewhere
        #   are monolithic + host-ISA, so this cap is Windows-only; Linux/macOS get the latest
        #   llama-cpp-python (verified Mac cpu engine ships 0.3.31, runs gemma-4).
        #   RE-CERTIFY before raising this cap — don't guess. `scripts/certify_llama_cpp_cpu.py`
        #   installs a candidate version the same way (--only-binary + this index), loads a tiny
        #   GGUF, and generates a token; exit 0 = safe to bump. Re-run 2026-07-02 on willw11
        #   re-confirmed 0.3.32 still fails at llama.dll load (WinError 127 → cap stays).
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
    "cuda-gguf": {
        "backend": "cuda_gguf",
        "package": "llama.cpp (CUDA)",
        "available": True,
        "builtin": True,           # worker ships in ARA; only its deps install into the env
        "packages": ["llama-cpp-python>=0.3", "psutil", "huggingface_hub"],
        "model_kinds": ("gguf",),
        # CUDA-offload GGUF via llama.cpp's CUDA build (opt-in, --engine cuda-gguf). Enables the
        # two-wall hybrid path: K layers on NVIDIA VRAM, N-K on CPU RAM, auto-fitted each run.
        # NOT a hardware auto-pick — NVIDIA still auto-picks ``cuda`` (the full-GPU transformers
        # engine). Prebuilt CUDA-124 wheels on the project's own index; we MUST force the prebuilt
        # wheel (same reason as vulkan: a plain install gets the CPU wheel from cache). `--only-binary`
        # (added in _builtin_targets) makes that deterministic. Kept AFTER `vulkan` so the GGUF
        # auto-default stays `cpu`. Linux + Windows only (macOS has no NVIDIA discrete GPU target).
        "wheel_only": {
            "llama-cpp-python": {
                "index": "https://abetlen.github.io/llama-cpp-python/whl/cu124",
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
        return "mlx"
    if shutil.which("nvidia-smi"):
        return "cuda"
    return None


def for_backend(backend: str) -> str | None:
    """The engine key whose backend matches *backend* (e.g. 'cuda' → 'cuda'), or None.
    One place maps hardware backends to engines, shared by detect and the registry."""
    return next((k for k, e in ENGINES.items() if e["backend"] == backend), None)


def resolve(value: str) -> str | None:
    """Map an ``--engine`` value to a concrete engine key, or None if it doesn't
    name one. ``auto`` defers to :func:`for_hardware`; explicit keys pass through."""
    if value == "auto":
        return for_hardware()
    canonical = engine_identity.canonical_engine(value)
    return canonical if canonical in ENGINES else None


def is_installed(key: str) -> bool:
    """Is engine *key*'s isolated env ready? Cheap and engine-free.

    An env is ready when its Python exists and, if the catalog declares an engine-package schema,
    its schema stamp matches. Unknown keys and stale package layouts are not installed-ready.
    """
    engine = ENGINES.get(engine_identity.canonical_engine(key))
    if engine is None or not engine_env.exists(engine["backend"]):
        return False
    schema = engine.get("env_schema")
    return schema is None or engine_env.stamped_schema(engine["backend"]) == schema


def _vendored_source(key: str) -> Path:
    """The directory of engine *key*'s vendored package source — ``ara/_vendor/<key>``, which holds
    the engine's ``pyproject.toml``. This is the path handed to ``uv pip install``: uv builds the
    engine package from it into the isolated env. Ships inside ARA's wheel (no network at install)."""
    engine = ENGINES[key]
    return Path(__file__).resolve().parent / engine["source_dir"]


def source_for(key: str) -> str:
    """The install source for an external engine *key*: a dev override, else the vendored path.

      * ``ARA_<KEY>_SOURCE`` (e.g. ``ARA_WMX_SOURCE=../wmx-suite``) — a local checkout for engine
        development; used verbatim (installed editable by :func:`_install_targets`).
      * otherwise the package source ARA ships under ``ara/_vendor/<key>``, so a release installs the
        exact engine code in the wheel — reproducibly and offline (no git fetch)."""
    engine = ENGINES[key]
    override = os.environ.get(engine["source_env"]) or os.environ.get(engine["legacy_source_env"])
    if override:
        return override
    return str(_vendored_source(key))


@dataclass(frozen=True)
class InstallResult:
    """Outcome of an install/uninstall attempt — also the shape behind ``--json``."""
    key: str
    status: str       # installed | refreshed | already | coming_soon | unknown | failed | removed | absent
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
    :func:`_builtin_targets`). External suites install from a local path, with any ``extras`` group
    appended (e.g. ``[cuda]``): the **vendored** source installed plain (read-only inside ARA's
    wheel), or a dev-override local checkout installed editable (``-e``). Any ``pip_args``
    (e.g. ``--torch-backend=auto`` to fetch the right CUDA torch wheel) come first.
    """
    engine = ENGINES[key]
    if engine.get("builtin"):
        return _builtin_targets(engine)
    pip_args = list(engine.get("pip_args", []))
    extras = engine.get("extras")
    suffix = f"[{extras}]" if extras else ""
    # A dev override (ARA_<KEY>_SOURCE) installs editable so engine edits are live; the vendored
    # default installs plain — its source is read-only inside ARA's wheel.
    target = f"{source_for(key)}{suffix}"
    if os.environ.get(engine["source_env"]) or os.environ.get(engine["legacy_source_env"]):
        return [*pip_args, "-e", target]
    return [*pip_args, target]


def install(key: str, *, refresh: bool = False) -> InstallResult:
    """Install engine *key* into its own isolated uv env. Idempotent and honest:
    never creates an env for an unknown or not-yet-available engine.

    Version-aware: an installed env is *stale* when its stamped ARA version differs from the
    current one (a newer ARA wheel ships newer ``ara/_vendor/*`` engine source) or when it carries
    no stamp at all (built by a pre-stamp ARA). A stale env is torn down and reinstalled so the
    shipped engine code actually reaches the box — reported as ``refreshed``. ``refresh=True`` forces
    that reinstall even when the stamp already matches. Every fresh/refresh install stamps the env
    with the current version."""
    if key not in ENGINES:
        return InstallResult(key, "unknown")
    engine = ENGINES[key]
    if not engine["available"]:
        return InstallResult(key, "coming_soon", f"{engine['package']} isn't available yet")
    current = _ara_version()
    present = engine_env.exists(engine["backend"])
    stale = present and (
        refresh
        or engine_env.stamped_version(engine["backend"]) != current
        or (engine.get("env_schema") is not None
            and engine_env.stamped_schema(engine["backend"]) != engine["env_schema"])
    )
    if present and not stale:
        return InstallResult(key, "already")
    if stale:                       # wipe the old env so the reinstall isn't itself a noop
        engine_env.remove(engine["backend"])
    try:
        engine_env.create(engine["backend"], _install_targets(key),
                          python=engine.get("python"), version=current,
                          schema=engine.get("env_schema"))
    except engine_env.EngineEnvError as e:
        return InstallResult(key, "failed", str(e))
    return InstallResult(key, "refreshed" if stale else "installed")


def uninstall(key: str) -> InstallResult:
    """Remove engine *key*'s isolated env. No-op when it isn't an engine or isn't present.
    The shared uv cache and other engines' envs are untouched."""
    if key not in ENGINES:
        return InstallResult(key, "unknown")
    if not engine_env.exists(ENGINES[key]["backend"]):
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
