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
import stat
from contextlib import contextmanager
from pathlib import Path

import platformdirs

from ara import db

_BUSY_MSG = ("another ARA measurement (characterize/benchmark) is already running on this machine — "
             "wait for it to finish so the readings don't corrupt each other.")


class MeasurementBusy(RuntimeError):
    """Raised when another measurement already holds the lock."""


class OllamaSetupBusy(RuntimeError):
    """Raised when another ARA process is setting up the same Ollama identity."""


class StagingBusy(RuntimeError):
    """Raised when another ARA process owns the model-staging lease for this volume."""


def _lock_path() -> Path:
    """The lock file, alongside ``ara.db`` — so the test ``ARA_DB_PATH`` override isolates it too."""
    return db._db_path().parent / "measurement.lock"


def _ollama_lock_path(endpoint: str, served_name: str) -> Path:
    identity = f"{endpoint}\0{served_name}".encode("utf-8")
    suffix = hashlib.sha256(identity).hexdigest()[:24]
    return db._db_path().parent / f"ollama-setup-{suffix}.lock"


def _staging_lock_root() -> Path:
    """Stable user-owned root for volume leases (never the OS-cleaned temporary directory)."""
    return Path(platformdirs.user_data_dir("ara")) / "locks"


def _secure_staging_lock_path(parent: Path) -> Path:
    """Create/validate the durable inode used to lease *parent*'s filesystem volume.

    On POSIX the root becomes read/execute-only after the inode exists. That prevents another
    ordinary ARA process from unlinking the pathname while a holder still has the old inode locked,
    which would otherwise let it recreate and independently lock a replacement inode.
    """
    root = _staging_lock_root()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root_info = root.lstat()
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise RuntimeError("ARA cannot establish a secure staging lock directory")
    if not _is_windows() and root_info.st_uid != os.getuid():
        raise RuntimeError("ARA cannot establish a secure staging lock directory")

    path = root / f"volume-{parent.stat().st_dev}.lock"
    try:
        path_info = path.lstat()
    except FileNotFoundError:
        if not _is_windows():
            root.chmod(0o700)
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(path, flags, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
        path_info = path.lstat()
    if stat.S_ISLNK(path_info.st_mode) or not stat.S_ISREG(path_info.st_mode):
        raise RuntimeError("ARA cannot establish a secure staging lock file")
    if not _is_windows():
        if path_info.st_uid != os.getuid():
            raise RuntimeError("ARA cannot establish a secure staging lock file")
        path.chmod(0o600)
        root.chmod(0o500)
    return path


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
    """Serialize setup of one Ollama model identity across local ARA processes.

    The lock is non-blocking and process-scoped. It closes ARA's same-host collision-check/create
    race; Ollama clients that do not use ARA remain outside this advisory lock.
    """
    path = _ollama_lock_path(endpoint, served_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if not _acquire(fd):
            raise OllamaSetupBusy(
                f"another ARA process is setting up Ollama model {served_name!r} — retry after "
                "that setup finishes."
            )
        try:
            yield
        finally:
            _release(fd)
    finally:
        os.close(fd)


@contextmanager
def staging_lock(parent: Path):
    """Hold the cross-process lease for private model copies on *parent*'s volume.

    The lease spans admission, copy, engine use, and cleanup. A crashed process releases the OS
    lock automatically, allowing its marked stale stage to be reclaimed by the next operation.
    """
    path = _secure_staging_lock_path(parent)
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or (
                not _is_windows() and info.st_uid != os.getuid()):
            raise RuntimeError("ARA cannot establish a secure staging lock file")
        if not _acquire(fd):
            raise StagingBusy(
                "another ARA process is staging or using a governed model on this volume — retry "
                "after it finishes."
            )
        try:
            yield
        finally:
            _release(fd)
    finally:
        os.close(fd)
