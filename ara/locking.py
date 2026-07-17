# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Cross-process locks for safety-sensitive ARA operations.

Two ARA measurements running at once on one machine each read the *other's* memory footprint into
their own reading — a Rule #1 hazard (a corrupted-high reading can store a ceiling that exceeds the
real wall). :func:`measurement_lock` makes them mutually exclusive: a non-blocking exclusive OS
advisory lock on a file next to the store. It auto-releases when the holding process exits, so a
crashed measurement never wedges the machine, and it is scoped to a measurement's process lifetime
(``serve`` exits while the model stays served, so serve can't and doesn't hold it — see the spec).

Spec 2026-07-04-measurement-flock-lock.
"""
from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

from ara import db

_BUSY_MSG = ("another ARA measurement (characterize/benchmark) is already running on this machine — "
             "wait for it to finish so the readings don't corrupt each other.")


class MeasurementBusy(RuntimeError):
    """Raised when another measurement already holds the lock."""


class OllamaSetupBusy(RuntimeError):
    """Raised when another ARA process is setting up the same Ollama identity."""


def _lock_path() -> Path:
    """The lock file, alongside ``ara.db`` — so the test ``ARA_DB_PATH`` override isolates it too."""
    return db._db_path().parent / "measurement.lock"


def _ollama_lock_path(endpoint: str) -> Path:
    suffix = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:24]
    return db._db_path().parent / f"ollama-setup-{suffix}.lock"


def _is_windows() -> bool:
    """Platform seam (a function, not a global ``os.name`` patch — patching that would flip pathlib
    to WindowsPath). Tests mock THIS to exercise the Windows branch on a POSIX host."""
    return os.name == "nt"


def _acquire(fd: int) -> bool:
    """Take a non-blocking exclusive advisory lock on *fd*. ``True`` on success, ``False`` if another
    holder already has it. Platform-branched (mockable, not hasattr-gated): POSIX ``fcntl.flock``,
    Windows ``msvcrt.locking``."""
    if _is_windows():
        import msvcrt
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _release(fd: int) -> None:
    if _is_windows():
        import msvcrt
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def measurement_lock():
    """Hold the machine's measurement lock for the duration of the block. Raises
    :class:`MeasurementBusy` immediately (never blocks) if another measurement holds it."""
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if not _acquire(fd):
            raise MeasurementBusy(_BUSY_MSG)
        try:
            yield
        finally:
            _release(fd)
    finally:
        os.close(fd)


@contextmanager
def ollama_setup_lock(endpoint: str, served_name: str):
    """Serialize all Ollama admission and setup on one endpoint across local ARA processes.

    The lock is non-blocking and process-scoped. Ollama clients that do not use ARA remain outside
    this advisory lock. ``served_name`` remains for call compatibility and user-facing context.
    """
    path = _ollama_lock_path(endpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if not _acquire(fd):
            raise OllamaSetupBusy(
                f"another ARA process is using this Ollama endpoint while setting up "
                f"{served_name!r} — retry after that setup finishes."
            )
        try:
            yield
        finally:
            _release(fd)
    finally:
        os.close(fd)
