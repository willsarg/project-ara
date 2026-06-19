"""The engine catalog and hardware-matched resolution.

ARA's core is engine-free; the hardware-specific suite is installed on demand
(`ara install`), not declared as a dependency. This module is the single source
of truth for *which* engines exist, *what* installs them, and *which one* fits
the current machine — the data behind `--engine {wmx|wcx|auto}`.

Read-only here: nothing in this module installs or imports an engine.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from importlib.util import find_spec

# Short, stable handles → the real package behind each. `available` is False for
# engines whose suite isn't shippable yet (resolvable, but install says so).
ENGINES: dict[str, dict] = {
    "wmx": {
        "backend": "apple",
        "module": "wmx_suite",     # import name (find_spec)
        "package": "wmx-suite",    # distribution name (uninstall)
        "available": True,
        "spec": "git+https://github.com/willsarg/wmx-suite",
    },
    "wcx": {
        "backend": "cuda",
        "module": "wcx_suite",
        "package": "wcx-suite",
        "available": True,
        "spec": "git+https://github.com/willsarg/wcx-suite",
        "extras": "cuda",                        # pulls torch + transformers
        # uv auto-detects the GPU and picks the matching CUDA torch wheel (the default
        # PyPI torch on Windows/Linux is CPU-only).
        "pip_args": ["--torch-backend=auto"],
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
    name one. ``auto`` defers to :func:`for_hardware`; ``wmx``/``wcx`` pass through."""
    if value == "auto":
        return for_hardware()
    return value if value in ENGINES else None


def is_installed(key: str) -> bool:
    """Is the engine *key*'s package importable? Cheap — uses ``find_spec``, never
    imports the engine. Unknown keys are simply 'not installed'."""
    engine = ENGINES.get(key)
    return engine is not None and find_spec(engine["module"]) is not None


def source_for(key: str) -> str:
    """The install source for engine *key*: its git spec, or a dev override.

    Setting ``ARA_<KEY>_SOURCE`` (e.g. ``ARA_WMX_SOURCE=../wmx-suite``) replaces the
    git URL — lets a developer install from a local checkout instead of cloning."""
    override = os.environ.get(f"ARA_{key.upper()}_SOURCE")
    return override or ENGINES[key]["spec"]


@dataclass(frozen=True)
class InstallResult:
    """Outcome of an install/uninstall attempt — also the shape behind ``--json``."""
    key: str
    status: str       # installed | already | coming_soon | unknown | failed
    detail: str = ""


def _install_args(key: str, source: str) -> list[str]:
    """uv-pip args for installing engine *key* from *source*.

    A local path installs editable (``-e``) for dev; a git/remote spec installs plainly.
    An engine's ``extras`` (e.g. wcx's ``[cuda]``) and any ``pip_args`` (e.g.
    ``--torch-backend=auto`` to fetch the right CUDA torch wheel) are folded in.
    """
    engine = ENGINES[key]
    pip_args = list(engine.get("pip_args", []))
    extras = engine.get("extras")
    suffix = f"[{extras}]" if extras else ""
    if source.startswith(("git+", "http://", "https://")):
        # PEP 508 direct reference: ``name[extra] @ git+url``
        target = f"{engine['package']}{suffix} @ {source}" if extras else source
        return ["install", *pip_args, target]
    return ["install", *pip_args, "-e", f"{source}{suffix}"]


def _run_pip(args: list[str]) -> tuple[int, str]:
    """Run ``uv pip <args>``; return (returncode, combined stdout+stderr). Never
    raises — a missing uv or a crash becomes a non-zero code with the message."""
    try:
        proc = subprocess.run(["uv", "pip", *args], capture_output=True, text=True)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except Exception as e:  # uv not found, etc.
        return 1, str(e)


def install(key: str) -> InstallResult:
    """Install engine *key* into the active environment. Idempotent and honest:
    never shells out for an unknown or not-yet-available engine."""
    if key not in ENGINES:
        return InstallResult(key, "unknown")
    engine = ENGINES[key]
    if not engine["available"]:
        return InstallResult(key, "coming_soon", f"{engine['module']} isn't available yet")
    if is_installed(key):
        return InstallResult(key, "already")
    rc, out = _run_pip(_install_args(key, source_for(key)))
    return InstallResult(key, "installed" if rc == 0 else "failed", out)


def uninstall(key: str) -> InstallResult:
    """Remove engine *key*'s package from the active environment. No-op when it
    isn't an engine or isn't installed."""
    if key not in ENGINES:
        return InstallResult(key, "unknown")
    if not is_installed(key):
        return InstallResult(key, "absent")
    rc, out = _run_pip(["uninstall", ENGINES[key]["package"]])
    return InstallResult(key, "removed" if rc == 0 else "failed", out)
