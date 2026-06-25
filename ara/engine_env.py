# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Isolated per-engine environments + the worker IPC seam.

Each hardware engine lives in its own uv environment under the data dir
(``engines/<name>/``), so incompatible toolchains (torch-CUDA vs torch-ROCm) and
Python pins can never collide, and ARA's core stays engine-free. ARA never imports
an engine; it drives one over a subprocess — spawning the env's own ``python`` and
reading a single JSON line back (the same shape ``wmx_suite.probe_worker`` already
emits).

Pure orchestration: no ML library is imported here. The one external boundary is
:func:`_run` (uv / the engine's python), which tests stub.

ARA does **not** implement a link-mode fallback ladder, because uv already owns one.
Verified empirically (uv 0.11.20, ``--link-mode clone`` into a venv on a non-CoW exFAT
volume): uv degrades per-file in a *single* install pass — clone → hardlink → full copy —
succeeds (rc 0), and only prints a stderr warning. So ``clone`` is the right universal
default: optimal (~7 MB/extra env) on a CoW filesystem (APFS/btrfs/XFS), and a correct
full copy elsewhere, with no errors and no multi-GB retry waste. ``create`` passes a single
``link_mode`` and lets uv resolve the rest.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import platformdirs

# Optimal on a CoW filesystem (APFS/btrfs/XFS): ~7 MB/extra env vs a full copy. On a
# non-CoW filesystem uv auto-degrades clone → hardlink → copy in one pass (verified), so
# this stays the universal default; a caller/config may still override it.
DEFAULT_LINK_MODE = "clone"


class EngineEnvError(RuntimeError):
    """An engine env operation (create / worker call) failed."""


def _run(cmd: list[str], *, input: str | None = None,
         stream: bool = False) -> tuple[int, str, str]:
    """Run a command, return (returncode, stdout, stderr). The one external boundary.

    ``stream=False`` (default): capture both stdout and stderr, return all three.
    ``stream=True``: capture stdout only; stderr passes through live to the terminal
    (so HF download bars show). Returns ``(rc, stdout, "")``.
    """
    if stream:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True, input=input)
        return proc.returncode, proc.stdout or "", ""
    proc = subprocess.run(cmd, capture_output=True, text=True, input=input)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def engines_root() -> Path:
    """Where engine envs live — ``ARA_ENGINES_DIR`` if set (tests), else the data dir."""
    override = os.environ.get("ARA_ENGINES_DIR")
    if override:
        return Path(override)
    return Path(platformdirs.user_data_dir("ara")) / "engines"


def env_path(name: str) -> Path:
    """Directory of the engine *name*'s environment."""
    return engines_root() / name


def _is_windows() -> bool:
    """Indirection so tests can flip the OS without monkeypatching ``os.name``
    globally (which would make pathlib try to build a WindowsPath on posix)."""
    return os.name == "nt"


def python_path(name: str) -> Path:
    """The engine *name*'s own interpreter — what ARA spawns the worker with."""
    base = env_path(name)
    if _is_windows():
        return base / "Scripts" / "python.exe"
    return base / "bin" / "python"


def exists(name: str) -> bool:
    """Is engine *name*'s environment present (its python materialized)?"""
    return python_path(name).exists()


def create(name: str, packages: list[str], *, link_mode: str = DEFAULT_LINK_MODE,
           python: str | None = None) -> Path:
    """Create engine *name*'s isolated uv env and install *packages* into it.

    *python* pins the interpreter version (e.g. ``"3.12"``) — some engines require a floor
    (wmx-suite needs ``>=3.12``); omit to take uv's default. Raises :class:`EngineEnvError`
    if the venv or the install fails.
    """
    path = env_path(name)
    engines_root().mkdir(parents=True, exist_ok=True)
    venv_cmd = ["uv", "venv", str(path)]
    if python:
        venv_cmd += ["--python", python]
    rc, _out, err = _run(venv_cmd)
    if rc != 0:
        raise EngineEnvError(f"creating env {name!r} failed: {err.strip()}")
    rc, _out, err = _run(
        ["uv", "pip", "install", "--python", str(python_path(name)),
         "--link-mode", link_mode, *packages]
    )
    if rc != 0:
        # The venv (with a working python) already exists, so a half-built env would make
        # exists()/is_installed() report it as ready forever. Make create atomic: tear the
        # env down on install failure so `ara install` can be retried to actually repair it.
        shutil.rmtree(path, ignore_errors=True)
        raise EngineEnvError(f"installing into {name!r} failed: {err.strip()}")
    return path


def remove(name: str) -> bool:
    """Delete engine *name*'s env. Returns True if it existed, False if absent.

    The shared uv cache and other engine envs are untouched."""
    path = env_path(name)
    if not path.exists():
        return False
    shutil.rmtree(path)
    return True


def run_worker(name: str, args: list[str], *, input: str | None = None,
               stream: bool = False) -> dict:
    """Spawn engine *name*'s python with *args*, return the single JSON object it emits.

    The worker prints one ``{...}`` line to stdout (other lines are ignored as logs),
    mirroring ``wmx_suite.probe_worker``. Raises :class:`EngineEnvError` on a non-zero
    exit or if no JSON line is found.

    ``stream=False`` (default): stderr is captured and included in error messages.
    ``stream=True``: stderr passes through live to the terminal (e.g. so HF download
    bars show during the CPU worker's GGUF fetch); error messages say to check output
    above since stderr wasn't captured.
    """
    cmd = [str(python_path(name)), *args]
    rc, out, err = _run(cmd, input=input, stream=stream)
    if rc != 0:
        if stream:
            raise EngineEnvError(f"worker {name!r} exited {rc} (see output above)")
        raise EngineEnvError(f"worker {name!r} exited {rc}: {err.strip()}")
    line = next((ln for ln in out.splitlines() if ln.lstrip().startswith("{")), None)
    if line is None:
        raise EngineEnvError(f"worker {name!r} emitted no JSON")
    return json.loads(line)
