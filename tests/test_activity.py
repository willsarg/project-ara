# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Contract tests for ARA-owned live activity records."""
from __future__ import annotations

import json
import os
from pathlib import Path

import psutil
import pytest

from ara import activity


@pytest.fixture(autouse=True)
def activity_dir(tmp_path, monkeypatch):
    path = tmp_path / "activity"
    monkeypatch.setenv("ARA_ACTIVITY_DIR", str(path))
    return path


class _Process:
    def __init__(self, *, created: float = 12.5, running: bool = True):
        self.created = created
        self.running = running

    def is_running(self):
        return self.running

    def create_time(self):
        return self.created


def _live(monkeypatch, *, created=12.5, running=True):
    monkeypatch.setattr(activity.psutil, "Process",
                        lambda _pid: _Process(created=created, running=running))


def _record(path: Path, name: str, **overrides) -> Path:
    data = {
        "kind": "running",
        "model": "org/model",
        "pid": 123,
        "process_created_at": 12.5,
        "started_at": 100.0,
    }
    data.update(overrides)
    target = path / name
    target.write_text(json.dumps(data), encoding="utf-8")
    return target


def test_track_atomically_writes_exact_private_record_and_cleans_up(
        activity_dir, monkeypatch):
    monkeypatch.setattr(activity.os, "getpid", lambda: 123)
    monkeypatch.setattr(activity.time, "time", lambda: 100.0)
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    replacements = []
    real_replace = os.replace

    def replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(activity.os, "replace", replace)

    with activity.track("running", "org/model"):
        files = list(activity_dir.glob("*.json"))
        assert len(files) == 1
        assert json.loads(files[0].read_text(encoding="utf-8")) == {
            "kind": "running",
            "model": "org/model",
            "pid": 123,
            "process_created_at": 12.5,
            "started_at": 100.0,
        }
        assert replacements == [(replacements[0][0], files[0])]
        assert replacements[0][0].parent == activity_dir
        assert not replacements[0][0].exists()
        if os.name != "nt":
            assert files[0].stat().st_mode & 0o777 == 0o600

    assert list(activity_dir.iterdir()) == []


def test_failed_atomic_write_removes_partial_temp_file(activity_dir, monkeypatch):
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    monkeypatch.setattr(activity.json, "dump",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(OSError, match="disk full"):
        with activity.track("running", "org/model"):
            pass
    assert list(activity_dir.iterdir()) == []


def test_nested_same_pid_records_coexist_and_snapshot_is_deterministic(
        activity_dir, monkeypatch):
    monkeypatch.setattr(activity.os, "getpid", lambda: 123)
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    times = iter((30.0, 10.0, 20.0))
    monkeypatch.setattr(activity.time, "time", lambda: next(times))

    with activity.track("serving", "org/z"):
        with activity.track("benchmarking", "org/a"):
            with activity.track("running", "org/b"):
                assert len(list(activity_dir.glob("*.json"))) == 3
                assert [(item.kind, item.model) for item in activity.snapshot()] == [
                    ("benchmarking", "org/a"),
                    ("running", "org/b"),
                    ("serving", "org/z"),
                ]


@pytest.mark.parametrize("raised", [Exception, KeyboardInterrupt, SystemExit])
def test_track_removes_its_record_for_every_exit_path(activity_dir, monkeypatch, raised):
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    with pytest.raises(raised):
        with activity.track("characterizing", "org/model"):
            raise raised("stop")
    assert list(activity_dir.iterdir()) == []


@pytest.mark.parametrize("kind", [
    "characterizing", "benchmarking", "searching", "running", "serving",
])
def test_all_and_only_contract_kinds_are_accepted(activity_dir, monkeypatch, kind):
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    with activity.track(kind):
        assert activity.snapshot()[0].kind == kind


def test_public_api_rejects_sensitive_arbitrary_or_unsafe_inputs(monkeypatch):
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    with pytest.raises(ValueError, match="kind"):
        activity.track("idle")
    with pytest.raises(ValueError, match="model"):
        activity.track("running", "org/model\nAuthorization: secret")
    with pytest.raises(TypeError):
        activity.track("running", prompt="secret")
    with pytest.raises(TypeError):
        activity.track("running", metadata={"token": "secret"})


def test_snapshot_ignores_malformed_partial_temp_and_extra_schema(
        activity_dir, monkeypatch):
    activity_dir.mkdir()
    _live(monkeypatch)
    (activity_dir / "broken.json").write_text("{", encoding="utf-8")
    (activity_dir / "partial.json").write_text(json.dumps({"kind": "running"}),
                                                encoding="utf-8")
    _record(activity_dir, "temporary.tmp", kind="serving")
    _record(activity_dir, "extra.json", prompt="secret")
    _record(activity_dir, "bad-kind.json", kind="idle")
    _record(activity_dir, "bad-model.json", model="unsafe\ntext")
    _record(activity_dir, "bad-pid.json", pid=True)
    _record(activity_dir, "bad-created.json", process_created_at="12.5")
    _record(activity_dir, "bad-start.json", started_at=None)

    assert activity.snapshot() == []


@pytest.mark.parametrize("failure", [
    psutil.NoSuchProcess(123),
    psutil.AccessDenied(123),
    psutil.ZombieProcess(123),
])
def test_snapshot_ignores_unknown_or_inaccessible_processes(
        activity_dir, monkeypatch, failure):
    activity_dir.mkdir()
    _record(activity_dir, "record.json")

    def inaccessible(_pid):
        raise failure

    monkeypatch.setattr(activity.psutil, "Process", inaccessible)
    assert activity.snapshot() == []


def test_snapshot_ignores_dead_pid_and_creation_time_mismatch(activity_dir, monkeypatch):
    activity_dir.mkdir()
    _record(activity_dir, "dead.json", pid=1)
    _record(activity_dir, "reused.json", pid=2)

    def process(pid):
        return _Process(running=False) if pid == 1 else _Process(created=99.0)

    monkeypatch.setattr(activity.psutil, "Process", process)
    assert activity.snapshot() == []


def test_snapshot_ignores_record_that_disappears_or_cannot_be_read(activity_dir, monkeypatch):
    activity_dir.mkdir()
    vanishing = _record(activity_dir, "vanishing.json")
    unreadable = _record(activity_dir, "unreadable.json", pid=124)
    real_open = Path.open

    def open_file(self, *args, **kwargs):
        if self == vanishing:
            raise FileNotFoundError
        if self == unreadable:
            raise PermissionError
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    assert activity.snapshot() == []


def test_snapshot_missing_directory_is_strictly_read_only(activity_dir, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("snapshot attempted a mutation")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "unlink", forbidden)
    monkeypatch.setattr(activity.os, "replace", forbidden)
    assert activity.snapshot() == []


def test_snapshot_never_mutates_stale_records_or_touches_db_or_locks(
        activity_dir, monkeypatch):
    activity_dir.mkdir()
    stale = _record(activity_dir, "stale.json")
    mutations = []
    real_open = Path.open

    def open_file(self, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            mutations.append(("write", self))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(Path, "mkdir", lambda *_a, **_k: mutations.append("mkdir"))
    monkeypatch.setattr(Path, "unlink", lambda *_a, **_k: mutations.append("unlink"))
    monkeypatch.setattr(activity.os, "replace", lambda *_a: mutations.append("replace"))
    monkeypatch.setattr("ara.db.connect", lambda *_a, **_k: mutations.append("db"))
    monkeypatch.setattr("ara.locking.measurement_lock",
                        lambda *_a, **_k: mutations.append("lock"))
    _live(monkeypatch, running=False)

    assert activity.snapshot() == []
    assert stale.exists()
    assert mutations == []


def test_default_path_uses_platformdirs(monkeypatch, tmp_path):
    monkeypatch.delenv("ARA_ACTIVITY_DIR")
    monkeypatch.setattr(activity, "user_data_path", lambda appname: tmp_path / appname)
    assert activity.activity_dir() == tmp_path / "ara" / "activity"
