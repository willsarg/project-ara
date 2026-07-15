# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Cross-process measurement lock — mutual exclusion so concurrent characterize/benchmark can't
corrupt each other's memory readings (Rule #1). Spec 2026-07-04-measurement-flock-lock."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from ara import locking


def test_lock_excludes_a_concurrent_acquire_then_frees(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "ara.db"))
    with locking.measurement_lock():
        with pytest.raises(locking.MeasurementBusy):
            with locking.measurement_lock():        # second holder on the same machine → refused
                pass
    with locking.measurement_lock():                # released after the first → free again
        pass


def test_lock_file_lands_under_the_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "d" / "ara.db"))
    with locking.measurement_lock():
        assert (tmp_path / "d" / "measurement.lock").exists()


def _fake_msvcrt(locking_fn):
    return types.SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0, locking=locking_fn)


def test_windows_branch_locks_and_unlocks_via_msvcrt(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "ara.db"))
    modes = []
    monkeypatch.setitem(sys.modules, "msvcrt", _fake_msvcrt(lambda fd, mode, n: modes.append(mode)))
    monkeypatch.setattr(locking, "_is_windows", lambda: True)
    with locking.measurement_lock():
        pass
    assert modes == [2, 0]                          # LK_NBLCK to acquire, LK_UNLCK to release


def test_windows_busy_maps_oserror_to_measurement_busy(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "ara.db"))
    def boom(fd, mode, n):
        raise OSError("locked")
    monkeypatch.setitem(sys.modules, "msvcrt", _fake_msvcrt(boom))
    monkeypatch.setattr(locking, "_is_windows", lambda: True)
    with pytest.raises(locking.MeasurementBusy):
        with locking.measurement_lock():
            pass


def _fake_fcntl(flock_fn):
    return types.SimpleNamespace(LOCK_EX=2, LOCK_NB=4, LOCK_UN=8, flock=flock_fn)


def test_posix_branch_locks_and_unlocks_via_fcntl(tmp_path, monkeypatch):
    # Mirror of the msvcrt test so the POSIX fcntl branch is covered by MOCKS on every host — on a
    # real Windows runner the branch is otherwise unreachable (fcntl is POSIX-only), leaving it
    # uncovered and breaking the 100% gate cross-OS.
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "ara.db"))
    ops = []
    monkeypatch.setitem(sys.modules, "fcntl", _fake_fcntl(lambda fd, op: ops.append(op)))
    monkeypatch.setattr(locking, "_is_windows", lambda: False)
    with locking.measurement_lock():
        pass
    assert ops == [2 | 4, 8]                        # LOCK_EX|LOCK_NB to acquire, LOCK_UN to release


def test_posix_busy_maps_oserror_to_measurement_busy(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "ara.db"))
    def boom(fd, op):
        raise OSError("locked")
    monkeypatch.setitem(sys.modules, "fcntl", _fake_fcntl(boom))
    monkeypatch.setattr(locking, "_is_windows", lambda: False)
    with pytest.raises(locking.MeasurementBusy):
        with locking.measurement_lock():
            pass


def test_ollama_setup_lock_excludes_same_identity_but_not_another(
        tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "ara.db"))
    endpoint = "http://127.0.0.1:11434"
    with locking.ollama_setup_lock(endpoint, "ara-model-a"):
        with pytest.raises(locking.OllamaSetupBusy, match="ara-model-a"):
            with locking.ollama_setup_lock(endpoint, "ara-model-a"):
                pass
        with locking.ollama_setup_lock(endpoint, "ara-model-b"):
            pass
    with locking.ollama_setup_lock(endpoint, "ara-model-a"):
        pass


def test_ollama_setup_lock_uses_stable_nonidentity_filename(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "ara.db"))
    with locking.ollama_setup_lock("http://host.invalid:11434/path", "org/private:model"):
        locks = list(tmp_path.glob("ollama-setup-*.lock"))
        assert len(locks) == 1
        assert "host" not in locks[0].name and "private" not in locks[0].name


def test_staging_lock_serializes_the_whole_volume(tmp_path, monkeypatch):
    lock_root = tmp_path / "locks"
    monkeypatch.setattr(locking, "_staging_lock_root", lambda: lock_root)
    first = tmp_path / "models-a"
    second = tmp_path / "models-b"
    first.mkdir()
    second.mkdir()

    with locking.staging_lock(first):
        with pytest.raises(locking.StagingBusy, match="this volume"):
            with locking.staging_lock(second):
                pass
    with locking.staging_lock(second):
        pass
    locks = list(lock_root.glob("*.lock"))
    assert len(locks) == 1
    if sys.platform != "win32":
        lock_root.chmod(0o700)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX directory permissions")
def test_staging_lock_path_cannot_be_unlinked_while_lease_is_live(tmp_path, monkeypatch):
    lock_root = tmp_path / "locks"
    monkeypatch.setattr(locking, "_staging_lock_root", lambda: lock_root)
    models = tmp_path / "models"
    models.mkdir()

    try:
        with locking.staging_lock(models):
            lock_path = next(lock_root.glob("*.lock"))
            with pytest.raises(PermissionError):
                lock_path.unlink()
    finally:
        # pytest must be able to remove its temporary tree after exercising the persistent root.
        lock_root.chmod(0o700)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink semantics")
def test_staging_lock_rejects_a_symlinked_lock_root(tmp_path, monkeypatch):
    real_root = tmp_path / "real"
    real_root.mkdir()
    link_root = tmp_path / "locks"
    link_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setattr(locking, "_staging_lock_root", lambda: link_root)
    models = tmp_path / "models"
    models.mkdir()

    with pytest.raises(RuntimeError, match="secure staging lock directory"):
        with locking.staging_lock(models):
            pass


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX ownership semantics")
def test_staging_lock_rejects_lock_root_owned_by_another_user(tmp_path, monkeypatch):
    lock_root = tmp_path / "locks"
    monkeypatch.setattr(locking, "_staging_lock_root", lambda: lock_root)
    monkeypatch.setattr(locking.os, "getuid", lambda: lock_root.stat().st_uid + 1)
    models = tmp_path / "models"
    models.mkdir()

    with pytest.raises(RuntimeError, match="secure staging lock directory"):
        locking._secure_staging_lock_path(models)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX ownership semantics")
def test_staging_lock_rejects_lock_file_owned_by_another_user(tmp_path, monkeypatch):
    lock_root = tmp_path / "locks"
    monkeypatch.setattr(locking, "_staging_lock_root", lambda: lock_root)
    models = tmp_path / "models"
    models.mkdir()
    lock_path = locking._secure_staging_lock_path(models)
    lock_root.chmod(0o700)
    real_lstat = Path.lstat

    def fake_lstat(path):
        info = real_lstat(path)
        if path == lock_path:
            return types.SimpleNamespace(st_mode=info.st_mode, st_uid=info.st_uid + 1)
        return info

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(RuntimeError, match="secure staging lock file"):
        locking._secure_staging_lock_path(models)


def test_staging_lock_rejects_nonregular_lock_path(tmp_path, monkeypatch):
    lock_root = tmp_path / "locks"
    lock_root.mkdir()
    monkeypatch.setattr(locking, "_staging_lock_root", lambda: lock_root)
    models = tmp_path / "models"
    models.mkdir()
    (lock_root / f"volume-{models.stat().st_dev}.lock").mkdir()

    with pytest.raises(RuntimeError, match="secure staging lock file"):
        locking._secure_staging_lock_path(models)


def test_staging_lock_handles_another_creator_winning_inode_race(tmp_path, monkeypatch):
    lock_root = tmp_path / "locks"
    monkeypatch.setattr(locking, "_staging_lock_root", lambda: lock_root)
    models = tmp_path / "models"
    models.mkdir()
    real_open = locking.os.open
    raced = False

    def racing_open(path, flags, mode=0o777):
        nonlocal raced
        if not raced and flags & locking.os.O_EXCL:
            raced = True
            fd = real_open(path, locking.os.O_RDWR | locking.os.O_CREAT, 0o600)
            locking.os.close(fd)
            raise FileExistsError(path)
        return real_open(path, flags, mode)

    monkeypatch.setattr(locking.os, "open", racing_open)
    path = locking._secure_staging_lock_path(models)
    assert raced is True and path.is_file()
    if sys.platform != "win32":
        lock_root.chmod(0o700)


def test_staging_lock_windows_security_path_uses_open_file_semantics(tmp_path, monkeypatch):
    lock_root = tmp_path / "locks"
    monkeypatch.setattr(locking, "_staging_lock_root", lambda: lock_root)
    monkeypatch.setattr(locking, "_is_windows", lambda: True)
    models = tmp_path / "models"
    models.mkdir()

    assert locking._secure_staging_lock_path(models).is_file()


def test_staging_lock_rejects_invalid_open_inode(tmp_path, monkeypatch):
    lock_file = tmp_path / "lease.lock"
    lock_file.write_bytes(b"")
    monkeypatch.setattr(locking, "_secure_staging_lock_path", lambda _parent: lock_file)
    monkeypatch.setattr(locking.os, "fstat",
                        lambda _fd: types.SimpleNamespace(st_mode=0, st_uid=0))

    with pytest.raises(RuntimeError, match="secure staging lock file"):
        with locking.staging_lock(tmp_path):
            pass


def test_staging_lock_operates_without_no_follow_flag(tmp_path, monkeypatch):
    lock_root = tmp_path / "locks"
    monkeypatch.setattr(locking, "_staging_lock_root", lambda: lock_root)
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.delattr(locking.os, "O_NOFOLLOW", raising=False)

    with locking.staging_lock(models):
        pass
    if sys.platform != "win32":
        lock_root.chmod(0o700)
