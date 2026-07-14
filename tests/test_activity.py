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

    def replace(source, destination, **kwargs):
        replacements.append((Path(source), Path(destination), kwargs))
        real_replace(source, destination, **kwargs)

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
        assert len(replacements) == 1
        assert replacements[0][0].suffix == ".tmp"
        assert replacements[0][1].name == files[0].name
        assert replacements[0][2]["src_dir_fd"] == replacements[0][2]["dst_dir_fd"]
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


def test_replace_failure_removes_temp_and_never_exposes_final(activity_dir, monkeypatch):
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    monkeypatch.setattr(activity.os, "replace",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            PermissionError("replace denied")))

    with pytest.raises(PermissionError, match="replace denied"):
        with activity.track("running", "org/model"):
            pass
    assert list(activity_dir.iterdir()) == []


def test_temp_cleanup_failure_preserves_atomic_write_error(activity_dir, monkeypatch):
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    monkeypatch.setattr(activity.os, "replace",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            PermissionError("replace denied")))
    real_unlink = activity.os.unlink

    def unlink(path, *args, **kwargs):
        if str(path).endswith(".tmp"):
            raise OSError("cleanup denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(activity.os, "unlink", unlink)
    with pytest.raises(PermissionError, match="replace denied") as caught:
        with activity.track("running", "org/model"):
            pass
    assert any("cleanup denied" in note for note in caught.value.__notes__)
    assert not list(activity_dir.glob("*.json"))


def test_fdopen_failure_closes_descriptor_and_removes_temp(activity_dir, monkeypatch):
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    opened = []
    closed = []
    real_open = os.open
    real_close = os.close

    def open_file(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def close_file(descriptor):
        closed.append(descriptor)
        return real_close(descriptor)

    monkeypatch.setattr(activity.os, "open", open_file)
    monkeypatch.setattr(activity.os, "close", close_file)
    monkeypatch.setattr(activity.os, "fdopen",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("fdopen failed")))

    with pytest.raises(OSError, match="fdopen failed"):
        with activity.track("running", "org/model"):
            pass
    assert sorted(opened) == sorted(closed)
    assert list(activity_dir.iterdir()) == []


def test_descriptor_close_failure_does_not_mask_fdopen_error(activity_dir, monkeypatch):
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    real_close = os.close

    def close_then_fail(descriptor):
        real_close(descriptor)
        raise OSError("close failed")

    monkeypatch.setattr(activity.os, "close", close_then_fail)
    monkeypatch.setattr(activity.os, "fdopen",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("fdopen failed")))
    with pytest.raises(OSError, match="fdopen failed") as caught:
        with activity.track("running", "org/model"):
            pass
    assert any("close failed" in note for note in caught.value.__notes__)
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


@pytest.mark.parametrize("raised", [Exception, KeyboardInterrupt, SystemExit])
def test_cleanup_failure_preserves_original_body_exception(
        activity_dir, monkeypatch, raised):
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    real_unlink = activity.os.unlink

    def unlink(path, *args, **kwargs):
        if str(path).endswith(".json"):
            raise OSError("record cleanup failed")
        return real_unlink(path, *args, **kwargs)

    with pytest.raises(raised, match="original") as caught:
        with activity.track("running", "org/model"):
            monkeypatch.setattr(activity.os, "unlink", unlink)
            raise raised("original")
    assert any("record cleanup failed" in note for note in caught.value.__notes__)


def test_cleanup_failure_without_body_exception_is_surfaced(monkeypatch):
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    real_unlink = activity.os.unlink

    def unlink(path, *args, **kwargs):
        if str(path).endswith(".json"):
            raise OSError("record cleanup failed")
        return real_unlink(path, *args, **kwargs)

    with pytest.raises(OSError, match="record cleanup failed"):
        with activity.track("running", "org/model"):
            monkeypatch.setattr(activity.os, "unlink", unlink)


def test_directory_close_failure_preserves_original_body_exception(monkeypatch):
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    real_close = activity.os.close

    def close_then_fail(descriptor):
        real_close(descriptor)
        raise OSError("directory close failed")

    with pytest.raises(RuntimeError, match="original") as caught:
        with activity.track("running"):
            monkeypatch.setattr(activity.os, "close", close_then_fail)
            raise RuntimeError("original")
    assert any("directory close failed" in note for note in caught.value.__notes__)


def test_directory_close_failure_without_body_exception_is_surfaced(monkeypatch):
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    real_close = activity.os.close

    def close_then_fail(descriptor):
        real_close(descriptor)
        raise OSError("directory close failed")

    with pytest.raises(OSError, match="directory close failed"):
        with activity.track("running"):
            monkeypatch.setattr(activity.os, "close", close_then_fail)


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


@pytest.mark.parametrize("model", [
    "org/model\u2028forged", "org/model\u2029forged", "org/model\u202eforged",
    "org/model\u2066forged", "org/model\ud800forged",
])
def test_track_rejects_unicode_control_format_and_surrogate_models(monkeypatch, model):
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    with pytest.raises(ValueError, match="display-safe"):
        activity.track("running", model)


def test_track_accepts_display_safe_unicode_and_emoji(monkeypatch):
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    with activity.track("running", "模型/🚀"):
        assert activity.snapshot()[0].model == "模型/🚀"


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


@pytest.mark.parametrize("kind", [[], {}, 7, None])
def test_snapshot_ignores_non_string_kind_without_crashing(activity_dir, monkeypatch, kind):
    activity_dir.mkdir()
    _live(monkeypatch)
    _record(activity_dir, "bad-kind.json", kind=kind)
    assert activity.snapshot() == []


@pytest.mark.parametrize("model", [
    "org/model\u2028forged", "org/model\u2029forged", "org/model\u202eforged",
    "org/model\u2066forged", "org/model\ud800forged",
])
def test_snapshot_ignores_unicode_control_format_and_surrogate_models(
        activity_dir, monkeypatch, model):
    activity_dir.mkdir()
    _live(monkeypatch)
    _record(activity_dir, "unsafe-model.json", model=model)
    assert activity.snapshot() == []


def test_snapshot_ignores_json_symlink_outside_registry(activity_dir, monkeypatch, tmp_path):
    activity_dir.mkdir()
    _live(monkeypatch)
    outside = _record(tmp_path, "outside.json")
    link = activity_dir / "linked.json"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    assert activity.snapshot() == []


def test_root_symlink_is_never_read(activity_dir, monkeypatch, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    _record(outside, "forged.json")
    try:
        activity_dir.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: pytest.fail("followed root"))
    assert activity.snapshot() == []


def test_root_symlink_refuses_ephemeral_write(activity_dir, monkeypatch, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        activity_dir.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    with pytest.raises(OSError):
        with activity.track("running", "org/model"):
            pass
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="directory-fd race coverage is POSIX-specific")
def test_snapshot_root_swap_keeps_original_registry_identity(activity_dir, monkeypatch, tmp_path):
    activity_dir.mkdir()
    _record(activity_dir, "real.json", model="org/real")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    _record(replacement, "real.json", model="org/forged")
    displaced = tmp_path / "displaced"
    _live(monkeypatch)
    real_scandir = activity.os.scandir
    swapped = False

    def swapping_scandir(path):
        nonlocal swapped
        iterator = real_scandir(path)
        if not swapped and (path == activity_dir or isinstance(path, int)):
            swapped = True
            activity_dir.rename(displaced)
            replacement.rename(activity_dir)
        return iterator

    monkeypatch.setattr(activity.os, "scandir", swapping_scandir)
    assert [item.model for item in activity.snapshot()] == ["org/real"]


@pytest.mark.skipif(os.name == "nt", reason="directory-fd race coverage is POSIX-specific")
def test_tracker_cleanup_root_swap_owns_original_record(activity_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    displaced = tmp_path / "displaced"
    real_replace = activity.os.replace
    swapped = False

    def swapping_replace(source, target, *args, **kwargs):
        nonlocal swapped
        real_replace(source, target, *args, **kwargs)
        if not swapped:
            swapped = True
            name = Path(target).name
            activity_dir.rename(displaced)
            (replacement / name).write_text("forged", encoding="utf-8")
            replacement.rename(activity_dir)

    monkeypatch.setattr(activity.os, "replace", swapping_replace)
    with activity.track("running", "org/model"):
        pass
    assert list(displaced.glob("*.json")) == []
    assert [path.read_text() for path in activity_dir.glob("*.json")] == ["forged"]


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
    real_os_open = activity.os.open

    def open_readonly(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            mutations.append("os.open")
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(activity.os, "open", open_readonly)
    monkeypatch.setattr(activity.os, "replace", lambda *_a: mutations.append("replace"))
    monkeypatch.setattr(activity.os, "unlink", lambda *_a: mutations.append("os.unlink"))
    monkeypatch.setattr(activity.os, "mkdir", lambda *_a, **_k: mutations.append("os.mkdir"))
    monkeypatch.setattr("ara.db.connect", lambda *_a, **_k: mutations.append("db"))
    monkeypatch.setattr("ara.locking.measurement_lock",
                        lambda *_a, **_k: mutations.append("lock"))
    monkeypatch.setattr("ara.ollama.ps", lambda *_a, **_k: mutations.append("ollama"))
    _live(monkeypatch, running=False)

    assert activity.snapshot() == []
    assert stale.exists()
    assert mutations == []


def test_default_path_uses_platformdirs(monkeypatch, tmp_path):
    monkeypatch.delenv("ARA_ACTIVITY_DIR")
    monkeypatch.setattr(activity, "user_data_path", lambda appname: tmp_path / appname)
    assert activity.activity_dir() == tmp_path / "ara" / "activity"


def test_fallback_track_snapshot_and_cleanup(activity_dir, monkeypatch):
    monkeypatch.setattr(activity, "_USE_DIR_FD", False)
    monkeypatch.setattr(activity.os, "getpid", lambda: 123)
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    with activity.track("running", "org/fallback"):
        assert [item.model for item in activity.snapshot()] == ["org/fallback"]
    assert list(activity_dir.iterdir()) == []


def test_fallback_rejects_root_symlink_for_read_and_write(activity_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(activity, "_USE_DIR_FD", False)
    outside = tmp_path / "fallback-outside"
    outside.mkdir()
    activity_dir.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    assert activity.snapshot() == []
    with pytest.raises(OSError, match="real directory"):
        with activity.track("running"):
            pass


def test_fallback_tracker_detects_root_identity_swap(activity_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(activity, "_USE_DIR_FD", False)
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    displaced = tmp_path / "fallback-displaced"
    replacement = tmp_path / "fallback-replacement"
    replacement.mkdir()
    with pytest.raises(OSError, match="changed during operation"):
        with activity.track("running"):
            activity_dir.rename(displaced)
            replacement.rename(activity_dir)


def test_fallback_failed_write_cleans_temp_and_closes_without_directory_fd(
        activity_dir, monkeypatch):
    monkeypatch.setattr(activity, "_USE_DIR_FD", False)
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    monkeypatch.setattr(
        activity.os, "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("replace denied")),
    )
    with pytest.raises(PermissionError, match="replace denied"):
        with activity.track("running"):
            pass
    assert list(activity_dir.iterdir()) == []


def test_fallback_json_reader_rejects_symlink_record(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    assert activity._json_record(link) is None


def test_fallback_snapshot_identity_swap_fails_closed(activity_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(activity, "_USE_DIR_FD", False)
    activity_dir.mkdir()
    replacement = tmp_path / "snapshot-fallback-replacement"
    replacement.mkdir()
    displaced = tmp_path / "snapshot-fallback-displaced"

    def swap_after_reads(*_args, **_kwargs):
        activity_dir.rename(displaced)
        replacement.rename(activity_dir)
        return []

    monkeypatch.setattr(activity, "_live_ollama_serving", swap_after_reads)
    assert activity.snapshot() == []


def test_secure_cleanup_tolerates_temp_removed_during_failed_replace(
        activity_dir, monkeypatch):
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    real_unlink = activity.os.unlink

    def remove_then_fail(source, _target, **kwargs):
        real_unlink(source, dir_fd=kwargs["src_dir_fd"])
        raise PermissionError("replace denied")

    monkeypatch.setattr(activity.os, "replace", remove_then_fail)
    with pytest.raises(PermissionError, match="replace denied"):
        with activity.track("running"):
            pass


def test_secure_tracker_cleanup_tolerates_record_already_removed(activity_dir, monkeypatch):
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: _Process())
    with activity.track("running"):
        next(activity_dir.glob("*.json")).unlink()
    assert list(activity_dir.iterdir()) == []


def test_record_name_scan_failure_is_empty(activity_dir, monkeypatch):
    activity_dir.mkdir()
    monkeypatch.setattr(
        activity.os, "scandir",
        lambda _fd: (_ for _ in ()).throw(PermissionError("scan denied")),
    )
    assert activity.snapshot() == []
