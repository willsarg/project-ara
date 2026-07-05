# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Cross-process measurement lock — mutual exclusion so concurrent characterize/benchmark can't
corrupt each other's memory readings (Rule #1). Spec 2026-07-04-measurement-flock-lock."""
from __future__ import annotations

import sys
import types

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
