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
    },
    "wcx": {
        "backend": "cuda",
        "package": "wcx-suite",
        # Not yet wired to the isolated-env model: backends/cuda.py still imports wcx_suite
        # in-process, but installs now go to the isolated env — so a real install would import
        # nothing. Marked unavailable (honest "coming soon") until it's converted to the worker
        # model AND tested on real NVIDIA hardware. See the cuda-conversion follow-up task.
        "available": False,
        "spec": "git+https://github.com/willsarg/wcx-suite",
        "extras": "cuda",                        # pulls torch + transformers
        # uv auto-detects the GPU and picks the matching CUDA torch wheel (the default
        # PyPI torch on Windows/Linux is CPU-only).
        "pip_args": ["--torch-backend=auto"],
        "python": "3.12",
    },
    "cpu": {
        "backend": "cpu",
        "package": "llama.cpp",
        "available": True,
        "builtin": True,           # worker ships in ARA; only its deps install into the env
        "packages": ["llama-cpp-python>=0.3", "psutil", "huggingface_hub"],
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


def _install_targets(key: str) -> list[str]:
    """The trailing ``uv pip install`` args (pip flags + targets) for engine *key*'s env.

    Built-in engines install a plain package list. External suites install their source:
    a git/remote ``spec`` (folding in an ``extras`` group via a PEP 508 direct reference) or
    a local path installed editable (``-e``) for dev. Any ``pip_args`` (e.g.
    ``--torch-backend=auto`` to fetch the right CUDA torch wheel) come first.
    """
    engine = ENGINES[key]
    if engine.get("builtin"):
        return list(engine["packages"])
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
