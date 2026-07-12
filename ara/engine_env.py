# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Isolated per-engine environments + the worker IPC seam.

Each hardware engine lives in its own uv environment under the data dir
(``engines/<name>/``), so incompatible toolchains (torch-CUDA vs torch-ROCm) and
Python pins can never collide, and ARA's core stays engine-free. ARA never imports
an engine; it drives one over a subprocess — spawning the env's own ``python`` and
reading a single JSON line back from the engine's native worker package.

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
import threading
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


def _stamp_path(name: str) -> Path:
    """The version-stamp file inside engine *name*'s env (``.ara-version``): the ARA release that
    installed it. Lets ``ara install`` tell a stale vendored engine (older ARA wheel) from a current
    one, so a new wheel's nested engine source actually reaches a box that already has the env."""
    return env_path(name) / ".ara-version"


def stamped_version(name: str) -> str | None:
    """The ARA version stamped into engine *name*'s env at install, or None if unstamped — no
    stamp file at all, or an env built by a pre-stamp ARA. Callers treat None as 'definitely stale'."""
    try:
        return _stamp_path(name).read_text().strip()
    except OSError:                      # no stamp (missing dir/file) → unknown/stale
        return None


def _schema_stamp_path(name: str) -> Path:
    """The engine-package schema stamp inside engine *name*'s env."""
    return env_path(name) / ".ara-schema"


def stamped_schema(name: str) -> str | None:
    """The package schema stamped into engine *name*'s env, or None when absent."""
    try:
        return _schema_stamp_path(name).read_text().strip()
    except OSError:
        return None


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
           python: str | None = None, version: str | None = None,
           schema: str | None = None, expected_import: str | None = None) -> Path:
    """Create engine *name*'s isolated uv env and install *packages* into it.

    *python* pins the interpreter version (e.g. ``"3.12"``) — some engines require a floor
    (the native MLX engine needs ``>=3.12``); omit to take uv's default. *version* and *schema* are
    stamped into the env (``.ara-version`` and ``.ara-schema``) after a successful install so a
    later ``ara install`` can detect stale source or a stale package layout. *expected_import*,
    when given, must be discoverable by the env's interpreter after installation and before either
    stamp is written. Raises :class:`EngineEnvError` if ``uv`` is missing, the venv/install fails,
    or the installed source does not provide the expected import package.
    """
    if shutil.which("uv") is None:
        raise EngineEnvError(
            "uv not found on PATH — install uv (https://docs.astral.sh/uv/) and retry")
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
    if expected_import is not None:
        rc, _out, _err = _run([
            str(python_path(name)),
            "-c",
            "import importlib.util, sys; "
            "sys.exit(0 if importlib.util.find_spec(sys.argv[1]) is not None else 1)",
            expected_import,
        ])
        if rc != 0:
            shutil.rmtree(path, ignore_errors=True)
            raise EngineEnvError(
                f"installed engine {name!r} does not provide expected import package "
                f"{expected_import!r}")
    if version is not None:               # record the ARA release that built this env (staleness stamp)
        stamp = _stamp_path(name)
        stamp.parent.mkdir(parents=True, exist_ok=True)   # uv made this in prod; be robust regardless
        stamp.write_text(version)
    if schema is not None:
        stamp = _schema_stamp_path(name)
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(schema)
    return path


def remove(name: str) -> bool:
    """Delete engine *name*'s env. Returns True if it existed, False if absent.

    The shared uv cache and other engine envs are untouched."""
    path = env_path(name)
    if not path.exists():
        return False
    shutil.rmtree(path)
    return True


def start_worker_server(name: str, args: list[str], *,
                        ready_timeout: float = 600.0) -> tuple[subprocess.Popen, dict]:
    """Spawn engine *name*'s python with *args* as a long-lived server process.

    Companion to :func:`run_worker`, but the process is **not waited on** — the server
    keeps running after this call returns. Stdout is read line-by-line until the first
    ``{``-prefixed line; that JSON is parsed and returned with the Popen handle.

    On a ``refused`` or ``error`` payload the process is waited and
    :class:`EngineEnvError` is raised with the reason. If the process exits before
    emitting any ``{``-line, :class:`EngineEnvError` is raised with
    ``"exited without a ready signal"``.
    """
    cmd = [str(python_path(name)), *args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    box: dict = {}

    def _read_ready() -> None:
        try:
            for line in proc.stdout:
                if line.lstrip().startswith("{"):
                    box["line"] = line
                    return
        except Exception as exc:                       # reader died (pipe broke, decode, …)
            box["error"] = exc

    # Read the handshake in a thread so a stalled child (e.g. a hung model load) can't hang
    # us forever — ready_timeout bounds the wait. On ANY failure we reap the child in the
    # finally-equivalent except, so we never leak a process holding the GPU + the port.
    reader = threading.Thread(target=_read_ready, daemon=True)
    reader.start()
    reader.join(ready_timeout)
    try:
        if reader.is_alive():
            raise EngineEnvError(
                f"server {name!r} timed out after {ready_timeout:.0f}s waiting for a ready signal")
        if "error" in box:
            raise EngineEnvError(f"server {name!r} failed reading the ready signal: {box['error']}")
        line = box.get("line")
        if line is None:
            raise EngineEnvError(f"server {name!r} exited without a ready signal")
        try:
            ready = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EngineEnvError(f"server {name!r} emitted an invalid ready signal: {exc}")
        if ready.get("error") or ready.get("refused"):
            reason = ready.get("reason") or ready.get("error", "unknown")
            raise EngineEnvError(f"server {name!r} refused: {reason}")
        return proc, ready
    except BaseException:
        for _step in (proc.kill, proc.wait):           # reap; never leak the child
            try:
                _step()
            except Exception:
                pass
        raise


def run_worker(name: str, args: list[str], *, input: str | None = None,
               stream: bool = False) -> dict:
    """Spawn engine *name*'s python with *args*, return the single JSON object it emits.

    The worker prints one ``{...}`` line to stdout (other lines are ignored as logs), matching
    the native engine worker contract. Raises :class:`EngineEnvError` on a non-zero
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
