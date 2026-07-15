# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Adversarial platform-handle and record-open contracts for the activity registry."""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

from ara import activity


class _WinCall:
    def __init__(self, result, callback=None):
        self.result = result
        self.callback = callback

    def __call__(self, *args):
        if self.callback:
            self.callback(*args)
        return self.result


def _fake_kernel32(**calls):
    return types.SimpleNamespace(**calls)


def test_low_level_windows_create_file_success_and_failure(monkeypatch, tmp_path):
    import ctypes

    create = _WinCall(81)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_a, **_k: _fake_kernel32(CreateFileW=create),
                        raising=False)
    assert activity._win_create_file(tmp_path, 1, 2, 3, 4) == 81
    assert create.argtypes and create.restype

    invalid = ctypes.c_void_p(-1).value
    monkeypatch.setattr(
        ctypes, "WinDLL",
        lambda *_a, **_k: _fake_kernel32(CreateFileW=_WinCall(invalid)),
        raising=False,
    )
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    monkeypatch.setattr(ctypes, "WinError", lambda code: OSError(f"winerror {code}"), raising=False)
    with pytest.raises(OSError, match="winerror 5"):
        activity._win_create_file(tmp_path, 1, 2, 3, 4)


def test_low_level_windows_info_and_close_success_and_failure(monkeypatch):
    import ctypes

    def fill_info(_handle, _klass, pointer, _size):
        pointer._obj.FileAttributes = 0x10
        pointer._obj.ReparseTag = 0

    info = _WinCall(1, fill_info)
    close = _WinCall(1)
    monkeypatch.setattr(
        ctypes, "WinDLL",
        lambda *_a, **_k: _fake_kernel32(
            GetFileInformationByHandleEx=info, CloseHandle=close),
        raising=False,
    )
    assert activity._win_file_info(82) == (0x10, 0)
    activity._win_close_handle(82)
    assert info.argtypes and info.restype and close.argtypes and close.restype

    monkeypatch.setattr(
        ctypes, "WinDLL",
        lambda *_a, **_k: _fake_kernel32(
            GetFileInformationByHandleEx=_WinCall(0), CloseHandle=_WinCall(0)),
        raising=False,
    )
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 6, raising=False)
    monkeypatch.setattr(ctypes, "WinError", lambda code: OSError(f"winerror {code}"), raising=False)
    with pytest.raises(OSError, match="winerror 6"):
        activity._win_file_info(82)
    with pytest.raises(OSError, match="winerror 6"):
        activity._win_close_handle(82)


def test_low_level_windows_handle_to_fd(monkeypatch):
    fake = types.SimpleNamespace(open_osfhandle=lambda handle, flags: handle + flags + 1)
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    assert activity._win_handle_to_fd(90) == 90 + os.O_RDONLY + 1


def test_platform_mode_and_guard_fd_branches(monkeypatch, tmp_path):
    monkeypatch.setattr(activity.os, "name", "nt")
    assert activity._platform_mode() == "windows"
    monkeypatch.setattr(activity.os, "name", "posix")
    monkeypatch.setattr(activity, "_POSIX_DIR_FD", True)
    assert activity._platform_mode() == "posix"
    monkeypatch.setattr(activity.os, "name", "other")
    monkeypatch.setattr(activity, "_POSIX_DIR_FD", False)
    assert activity._platform_mode() is None
    assert activity._DirectoryGuard(tmp_path, "windows", 1).dir_fd is None
    assert activity._DirectoryGuard(tmp_path, "posix", 2).dir_fd == 2


def test_windows_directory_handle_uses_no_delete_share_and_no_reparse(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        activity, "_win_create_file",
        lambda *args: calls.append(args) or 41,
    )
    monkeypatch.setattr(
        activity, "_win_file_info",
        lambda _handle: (activity._WIN_FILE_ATTRIBUTE_DIRECTORY, 0),
    )
    assert activity._win_open_directory(tmp_path) == 41
    path, access, share, creation, flags = calls[0]
    assert path == tmp_path
    assert access == activity._WIN_GENERIC_READ
    assert share & activity._WIN_FILE_SHARE_READ
    assert share & activity._WIN_FILE_SHARE_WRITE
    assert not share & activity._WIN_FILE_SHARE_DELETE
    assert creation == activity._WIN_OPEN_EXISTING
    assert flags & activity._WIN_FILE_FLAG_BACKUP_SEMANTICS
    assert flags & activity._WIN_FILE_FLAG_OPEN_REPARSE_POINT


@pytest.mark.parametrize("attributes,tag", [
    (0x10 | 0x400, 1),
    (0x80, 0),
])
def test_windows_directory_rejects_reparse_and_non_directory(
        monkeypatch, tmp_path, attributes, tag):
    closed = []
    monkeypatch.setattr(activity, "_win_create_file", lambda *_args: 42)
    monkeypatch.setattr(activity, "_win_file_info", lambda _handle: (attributes, tag))
    monkeypatch.setattr(activity, "_win_close_handle", lambda handle: closed.append(handle))
    with pytest.raises(OSError):
        activity._win_open_directory(tmp_path)
    assert closed == [42]


def test_windows_record_handle_rejects_reparse_and_closes(monkeypatch, tmp_path):
    calls = []
    closed = []
    monkeypatch.setattr(
        activity, "_win_create_file",
        lambda *args: calls.append(args) or 43,
    )
    monkeypatch.setattr(
        activity, "_win_file_info",
        lambda _handle: (activity._WIN_FILE_ATTRIBUTE_REPARSE_POINT, 0xA000000C),
    )
    monkeypatch.setattr(activity, "_win_close_handle", lambda handle: closed.append(handle))
    with pytest.raises(OSError):
        activity._win_open_record(tmp_path / "record.json")
    _path, access, share, creation, flags = calls[0]
    assert access == activity._WIN_GENERIC_READ
    assert not share & activity._WIN_FILE_SHARE_DELETE
    assert creation == activity._WIN_OPEN_EXISTING
    assert flags & activity._WIN_FILE_FLAG_OPEN_REPARSE_POINT
    assert not flags & activity._WIN_FILE_FLAG_BACKUP_SEMANTICS
    assert closed == [43]


def test_windows_record_handle_accepts_regular_file(monkeypatch, tmp_path):
    monkeypatch.setattr(activity, "_win_create_file", lambda *_args: 44)
    monkeypatch.setattr(activity, "_win_file_info", lambda _handle: (0, 0))
    assert activity._win_open_record(tmp_path / "record.json") == 44


def test_windows_info_failure_closes_handle_without_masking_original(monkeypatch, tmp_path):
    closed = []
    monkeypatch.setattr(activity, "_win_create_file", lambda *_args: 45)
    monkeypatch.setattr(
        activity, "_win_file_info",
        lambda _handle: (_ for _ in ()).throw(KeyboardInterrupt("info interrupted")),
    )
    monkeypatch.setattr(
        activity, "_win_close_handle",
        lambda handle: closed.append(handle) or (_ for _ in ()).throw(OSError("close failed")),
    )
    with pytest.raises(KeyboardInterrupt, match="info interrupted") as caught:
        activity._win_open_directory(tmp_path)
    assert closed == [45]
    assert any("close failed" in note for note in caught.value.__notes__)


def test_windows_tracker_holds_root_until_record_cleanup(tmp_path, monkeypatch):
    root = tmp_path / "activity"
    monkeypatch.setenv("ARA_ACTIVITY_DIR", str(root))
    monkeypatch.setattr(activity, "_platform_mode", lambda: "windows")
    monkeypatch.setattr(activity, "_win_open_directory", lambda _path: 51)
    events = []
    monkeypatch.setattr(activity, "_win_close_handle", lambda handle: events.append(("close", handle)))
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: type(
        "P", (), {"create_time": lambda self: 1.0})())
    real_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        assert events == []
        events.append(("unlink", path.name))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    with activity.track("running"):
        assert list(root.glob("*.json"))
    assert events[0][0] == "unlink"
    assert events[1] == ("close", 51)


def test_windows_tracker_relative_override_stays_bound_across_chdir(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    elsewhere = tmp_path / "elsewhere"
    origin.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(origin)
    monkeypatch.setenv("ARA_ACTIVITY_DIR", "registry")
    monkeypatch.setattr(activity, "_platform_mode", lambda: "windows")
    monkeypatch.setattr(activity, "_win_open_directory", lambda _path: 52)
    monkeypatch.setattr(activity, "_win_close_handle", lambda _handle: None)
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: type(
        "P", (), {"create_time": lambda self: 1.0})())

    assert activity.activity_dir() == origin / "registry"
    with activity.track("running"):
        assert list((origin / "registry").glob("*.json"))
        monkeypatch.chdir(elsewhere)
    assert list((origin / "registry").iterdir()) == []
    assert not (elsewhere / "registry").exists()


def test_windows_persistent_holds_and_closes_serving_then_root(tmp_path, monkeypatch):
    root = tmp_path / "activity"
    monkeypatch.setenv("ARA_ACTIVITY_DIR", str(root))
    monkeypatch.setattr(activity, "_platform_mode", lambda: "windows")
    handles = iter((61, 62))
    monkeypatch.setattr(activity, "_win_open_directory", lambda _path: next(handles))
    closed = []
    monkeypatch.setattr(activity, "_win_close_handle", lambda handle: closed.append(handle))
    path = activity.record_ollama_serving(
        served_name="owned", model="org/model", context=1,
        endpoint="http://127.0.0.1:11434", started_at=1.0,
        base_artifact_id="ollama-manifest-sha256:" + "a" * 64,
        served_artifact_id="ollama-manifest-sha256:" + "b" * 64)
    assert path.exists()
    assert closed == [62, 61]


def test_windows_persistent_relative_override_stays_bound_across_chdir(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    elsewhere = tmp_path / "elsewhere"
    origin.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(origin)
    monkeypatch.setenv("ARA_ACTIVITY_DIR", "registry")
    monkeypatch.setattr(activity, "_platform_mode", lambda: "windows")
    calls = []

    def open_directory(path):
        calls.append(path)
        if len(calls) == 1:
            monkeypatch.chdir(elsewhere)
        return 60 + len(calls)

    monkeypatch.setattr(activity, "_win_open_directory", open_directory)
    monkeypatch.setattr(activity, "_win_close_handle", lambda _handle: None)
    path = activity.record_ollama_serving(
        served_name="owned", model="org/model", context=1,
        endpoint="http://127.0.0.1:11434", started_at=1.0,
        base_artifact_id="ollama-manifest-sha256:" + "a" * 64,
        served_artifact_id="ollama-manifest-sha256:" + "b" * 64,
    )
    assert path.parent == origin / "registry" / "serving"
    assert path.exists()
    assert calls == [origin / "registry", origin / "registry" / "serving"]
    assert not (elsewhere / "registry").exists()


def test_windows_snapshot_relative_override_stays_bound_across_chdir(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    elsewhere = tmp_path / "elsewhere"
    registry = origin / "registry"
    registry.mkdir(parents=True)
    elsewhere.mkdir()
    record = registry / "live.json"
    record.write_text(json.dumps({
        "kind": "running", "pid": 123, "process_created_at": 1.0,
        "started_at": 2.0,
    }), encoding="utf-8")
    monkeypatch.chdir(origin)
    monkeypatch.setenv("ARA_ACTIVITY_DIR", "registry")
    monkeypatch.setattr(activity, "_platform_mode", lambda: "windows")
    opened = []

    def open_directory(path):
        opened.append(path)
        if len(opened) == 1:
            monkeypatch.chdir(elsewhere)
        return 67 + len(opened)

    monkeypatch.setattr(activity, "_win_open_directory", open_directory)
    monkeypatch.setattr(activity, "_win_close_handle", lambda _handle: None)
    monkeypatch.setattr(activity, "_win_open_record", lambda path: 70)
    monkeypatch.setattr(
        activity, "_win_handle_to_fd", lambda _handle: os.open(record, os.O_RDONLY),
    )
    monkeypatch.setattr(activity.psutil, "Process", lambda _pid: type(
        "P", (), {"is_running": lambda self: True, "create_time": lambda self: 1.0})())

    found = activity.snapshot()
    assert [(item.kind, item.pid) for item in found] == [("running", 123)]
    assert opened[0] == registry


def test_windows_existing_serving_directory_and_failed_replace_cleanup(tmp_path, monkeypatch):
    root_path = tmp_path / "activity"
    serving_path = root_path / "serving"
    serving_path.mkdir(parents=True)
    root = activity._DirectoryGuard(root_path, "windows", 63)
    monkeypatch.setattr(activity, "_win_open_directory", lambda _path: 64)
    serving = activity._open_serving(root, create=True)
    monkeypatch.setattr(
        activity.os, "replace",
        lambda *_a, **_k: (_ for _ in ()).throw(PermissionError("replace denied")),
    )
    target = serving_path / "owned.json"
    with pytest.raises(PermissionError, match="replace denied"):
        activity._atomic_write(target, {"value": 1}, guard=serving)
    assert list(serving_path.iterdir()) == []


def test_windows_open_existing_serving_without_create(tmp_path, monkeypatch):
    root_path = tmp_path / "activity"
    serving_path = root_path / "serving"
    serving_path.mkdir(parents=True)
    root = activity._DirectoryGuard(root_path, "windows", 65)
    monkeypatch.setattr(activity, "_win_open_directory", lambda path: 66)
    serving = activity._open_serving(root, create=False)
    assert serving == activity._DirectoryGuard(serving_path, "windows", 66)


def test_windows_record_handle_transfers_to_fd_and_reads_json(tmp_path, monkeypatch):
    root = tmp_path / "activity"
    root.mkdir()
    record = root / "record.json"
    record.write_text(json.dumps({"value": 1}), encoding="utf-8")
    guard = activity._DirectoryGuard(root, "windows", 70)
    monkeypatch.setattr(activity, "_win_open_record", lambda path: 71)
    monkeypatch.setattr(activity, "_win_handle_to_fd", lambda _handle: os.open(record, os.O_RDONLY))
    monkeypatch.setattr(
        activity, "_win_close_handle",
        lambda _handle: pytest.fail("transferred record handle closed twice"),
    )
    assert activity._json_record(record, guard=guard) == {"value": 1}


def test_windows_record_conversion_failure_closes_handle(monkeypatch, tmp_path):
    root = tmp_path / "activity"
    root.mkdir()
    guard = activity._DirectoryGuard(root, "windows", 72)
    closed = []
    monkeypatch.setattr(activity, "_win_open_record", lambda _path: 73)
    monkeypatch.setattr(
        activity, "_win_handle_to_fd",
        lambda _handle: (_ for _ in ()).throw(KeyboardInterrupt("convert interrupted")),
    )
    monkeypatch.setattr(activity, "_win_close_handle", lambda handle: closed.append(handle))
    with pytest.raises(KeyboardInterrupt, match="convert interrupted"):
        activity._json_record(root / "record.json", guard=guard)
    assert closed == [73]


def test_windows_record_conversion_close_failure_notes_original(monkeypatch, tmp_path):
    root = tmp_path / "activity"
    root.mkdir()
    guard = activity._DirectoryGuard(root, "windows", 74)
    monkeypatch.setattr(activity, "_win_open_record", lambda _path: 75)
    monkeypatch.setattr(
        activity, "_win_handle_to_fd",
        lambda _handle: (_ for _ in ()).throw(KeyboardInterrupt("convert interrupted")),
    )
    monkeypatch.setattr(
        activity, "_win_close_handle",
        lambda _handle: (_ for _ in ()).throw(OSError("handle close failed")),
    )
    with pytest.raises(KeyboardInterrupt, match="convert interrupted") as caught:
        activity._json_record(root / "record.json", guard=guard)
    assert any("handle close failed" in note for note in caught.value.__notes__)


def test_record_fstat_oserror_closes_and_suppresses(monkeypatch, tmp_path):
    root = tmp_path / "activity"
    root.mkdir()
    record = root / "record.json"
    record.write_text("{}", encoding="utf-8")
    guard = activity._DirectoryGuard(root, "posix", 71)
    closed = []
    monkeypatch.setattr(activity.os, "open", lambda *_a, **_k: 72)
    monkeypatch.setattr(
        activity.os, "fstat",
        lambda _fd: (_ for _ in ()).throw(OSError("fstat failed")),
    )
    monkeypatch.setattr(activity.os, "close", lambda fd: closed.append(fd))
    assert activity._json_record(record, guard=guard) is None
    assert closed == [72]


def test_record_fstat_close_failure_notes_baseexception(monkeypatch, tmp_path):
    root = tmp_path / "activity"
    root.mkdir()
    record = root / "record.json"
    record.write_text("{}", encoding="utf-8")
    guard = activity._DirectoryGuard(root, "posix", 71)
    target = [72]

    def close_file(descriptor):
        if descriptor in target:
            raise OSError("fstat descriptor close failed")

    monkeypatch.setattr(activity.os, "open", lambda *_a, **_k: 72)
    monkeypatch.setattr(
        activity.os, "fstat",
        lambda _fd: (_ for _ in ()).throw(KeyboardInterrupt("fstat interrupted")),
    )
    monkeypatch.setattr(activity.os, "close", close_file)
    with pytest.raises(KeyboardInterrupt, match="fstat interrupted") as caught:
        activity._json_record(record, guard=guard)
    assert any("fstat descriptor close failed" in note for note in caught.value.__notes__)


def test_missing_record_open_is_suppressed(tmp_path, monkeypatch):
    root = tmp_path / "activity"
    root.mkdir()
    guard = activity._DirectoryGuard(root, "posix", 71)
    monkeypatch.setattr(
        activity.os, "open", lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError()))
    assert activity._json_record(root / "missing.json", guard=guard) is None


def test_windows_record_name_listing_filters_json(tmp_path):
    root = tmp_path / "activity"
    root.mkdir()
    (root / "a.json").write_text("{}", encoding="utf-8")
    (root / "ignore.tmp").write_text("{}", encoding="utf-8")
    guard = activity._DirectoryGuard(root, "windows", 76)
    assert activity._record_names(root, guard=guard) == ["a.json"]


@pytest.mark.parametrize("operation", ["snapshot", "track", "persistent"])
def test_unsupported_platform_fails_closed_before_any_path_mutation(
        tmp_path, monkeypatch, operation):
    root = tmp_path / "activity"
    monkeypatch.setenv("ARA_ACTIVITY_DIR", str(root))
    monkeypatch.setattr(activity, "_platform_mode", lambda: None)
    mutations = []
    monkeypatch.setattr(Path, "mkdir", lambda *_a, **_k: mutations.append("mkdir"))
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: mutations.append("open"))
    monkeypatch.setattr(Path, "unlink", lambda *_a, **_k: mutations.append("unlink"))
    monkeypatch.setattr(activity.os, "open", lambda *_a, **_k: mutations.append("os.open"))
    monkeypatch.setattr(activity.os, "replace", lambda *_a, **_k: mutations.append("replace"))
    if operation == "snapshot":
        assert activity.snapshot() == []
    elif operation == "track":
        monkeypatch.setattr(activity.psutil, "Process", lambda _pid: type(
            "P", (), {"create_time": lambda self: 1.0})())
        with pytest.raises(OSError, match="safe activity registry"):
            with activity.track("running"):
                pass
    else:
        with pytest.raises(OSError, match="safe activity registry"):
            activity.record_ollama_serving(
                served_name="owned", model="org/model", context=1,
                endpoint="http://127.0.0.1:11434", started_at=1.0,
                base_artifact_id="ollama-manifest-sha256:" + "a" * 64,
                served_artifact_id="ollama-manifest-sha256:" + "b" * 64)
    assert mutations == []


def test_posix_record_open_is_nonblocking_and_fifo_is_suppressed(tmp_path, monkeypatch):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO unavailable")
    root = tmp_path / "activity"
    root.mkdir()
    fifo = root / "race.json"
    os.mkfifo(fifo)
    monkeypatch.setenv("ARA_ACTIVITY_DIR", str(root))
    monkeypatch.setattr(activity, "_record_names", lambda *_a, **_k: [fifo.name])
    real_open = activity.os.open
    record_flags = []

    def checked_open(path, flags, *args, **kwargs):
        if path == fifo.name:
            record_flags.append(flags)
            assert flags & os.O_NONBLOCK
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(activity.os, "open", checked_open)
    assert activity.snapshot() == []
    assert record_flags


def test_reader_fdopen_baseexception_closes_descriptor(monkeypatch, tmp_path):
    root = tmp_path / "activity"
    root.mkdir()
    record = root / "record.json"
    record.write_text(json.dumps({"kind": "running"}), encoding="utf-8")
    guard = activity._DirectoryGuard(root, "posix", 71)
    closed = []
    monkeypatch.setattr(activity.os, "open", lambda *_a, **_k: 72)
    monkeypatch.setattr(activity.os, "fstat", lambda _fd: types.SimpleNamespace(st_mode=0o100600))
    monkeypatch.setattr(activity.os, "close", lambda descriptor: closed.append(descriptor))
    monkeypatch.setattr(activity.os, "fdopen", lambda *_a, **_k: (_ for _ in ()).throw(
        KeyboardInterrupt("fdopen interrupted")))
    with pytest.raises(KeyboardInterrupt, match="fdopen interrupted"):
        activity._json_record(record, guard=guard)
    assert closed == [72]


def test_reader_fdopen_close_failure_notes_original(monkeypatch, tmp_path):
    root = tmp_path / "activity"
    root.mkdir()
    record = root / "record.json"
    record.write_text("{}", encoding="utf-8")
    guard = activity._DirectoryGuard(root, "posix", 71)
    target_fd = [72]

    def close_file(descriptor):
        if descriptor in target_fd:
            raise OSError("record close failed")

    monkeypatch.setattr(activity.os, "open", lambda *_a, **_k: 72)
    monkeypatch.setattr(activity.os, "fstat", lambda _fd: types.SimpleNamespace(st_mode=0o100600))
    monkeypatch.setattr(activity.os, "close", close_file)
    monkeypatch.setattr(activity.os, "fdopen", lambda *_a, **_k: (_ for _ in ()).throw(
        KeyboardInterrupt("fdopen interrupted")))
    with pytest.raises(KeyboardInterrupt, match="fdopen interrupted") as caught:
        activity._json_record(record, guard=guard)
    assert any("record close failed" in note for note in caught.value.__notes__)


class _ClosingStream:
    def __init__(self, *, close_error=None):
        self.close_error = close_error
        self.closed = False
        self.descriptor = None

    def write(self, _value):
        return None

    def flush(self):
        return None

    def fileno(self):
        return self.descriptor

    def close(self):
        self.closed = True
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None
        if self.close_error is not None:
            raise self.close_error


def test_writer_body_baseexception_survives_stream_close_failure(tmp_path, monkeypatch):
    root = tmp_path / "activity"
    root.mkdir()
    guard = activity._DirectoryGuard(root, "posix", 71)
    stream = _ClosingStream(close_error=OSError("writer stream close failed"))
    original = KeyboardInterrupt("write interrupted")
    monkeypatch.setattr(
        activity.os, "fdopen",
        lambda descriptor, *_a, **_k: setattr(stream, "descriptor", descriptor) or stream,
    )
    monkeypatch.setattr(activity.os, "open", lambda *_a, **_k: 72)
    monkeypatch.setattr(activity.os, "close", lambda _fd: None)
    monkeypatch.setattr(activity.os, "unlink", lambda *_a, **_k: None)
    monkeypatch.setattr(
        activity.json, "dump", lambda *_a, **_k: (_ for _ in ()).throw(original),
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        activity._atomic_write(root / "record.json", {}, guard=guard)
    assert caught.value is original
    assert stream.closed
    assert any("writer stream close failed" in note for note in original.__notes__)


def test_writer_stream_close_failure_propagates_without_body_error(tmp_path, monkeypatch):
    root = tmp_path / "activity"
    root.mkdir()
    guard = activity._DirectoryGuard(root, "posix", 71)
    stream = _ClosingStream(close_error=OSError("writer stream close failed"))
    monkeypatch.setattr(
        activity.os, "fdopen",
        lambda descriptor, *_a, **_k: setattr(stream, "descriptor", descriptor) or stream,
    )
    monkeypatch.setattr(activity.os, "open", lambda *_a, **_k: 72)
    monkeypatch.setattr(activity.os, "close", lambda _fd: None)
    monkeypatch.setattr(activity.os, "unlink", lambda *_a, **_k: None)
    monkeypatch.setattr(activity.os, "fsync", lambda _fd: None)
    with pytest.raises(OSError, match="writer stream close failed"):
        activity._atomic_write(root / "record.json", {}, guard=guard)
    assert stream.closed


def test_reader_body_baseexception_survives_stream_close_failure(tmp_path, monkeypatch):
    root = tmp_path / "activity"
    root.mkdir()
    guard = activity._DirectoryGuard(root, "posix", 71)
    stream = _ClosingStream(close_error=OSError("reader stream close failed"))
    original = KeyboardInterrupt("read interrupted")
    record = root / "record.json"
    record.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        activity, "_open_record_fd", lambda *_a, **_k: os.open(record, os.O_RDONLY),
    )
    monkeypatch.setattr(
        activity.os, "fdopen",
        lambda descriptor, *_a, **_k: setattr(stream, "descriptor", descriptor) or stream,
    )
    monkeypatch.setattr(activity.os, "close", lambda _fd: None)
    monkeypatch.setattr(
        activity.json, "load", lambda *_a, **_k: (_ for _ in ()).throw(original),
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        activity._json_record(root / "record.json", guard=guard)
    assert caught.value is original
    assert stream.closed
    assert any("reader stream close failed" in note for note in original.__notes__)


def test_reader_stream_close_failure_propagates_without_body_error(tmp_path, monkeypatch):
    root = tmp_path / "activity"
    root.mkdir()
    guard = activity._DirectoryGuard(root, "posix", 71)
    stream = _ClosingStream(close_error=OSError("reader stream close failed"))
    record = root / "record.json"
    record.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        activity, "_open_record_fd", lambda *_a, **_k: os.open(record, os.O_RDONLY),
    )
    monkeypatch.setattr(
        activity.os, "fdopen",
        lambda descriptor, *_a, **_k: setattr(stream, "descriptor", descriptor) or stream,
    )
    monkeypatch.setattr(activity.os, "close", lambda _fd: None)
    monkeypatch.setattr(activity.json, "load", lambda _stream: {})
    with pytest.raises(OSError, match="reader stream close failed"):
        activity._json_record(root / "record.json", guard=guard)
    assert stream.closed


def test_reader_fstat_baseexception_closes_descriptor(monkeypatch, tmp_path):
    root = tmp_path / "activity"
    root.mkdir()
    record = root / "record.json"
    record.write_text("{}", encoding="utf-8")
    guard = activity._DirectoryGuard(root, "posix", 71)
    record_fd = [72]
    closed = []
    monkeypatch.setattr(activity.os, "open", lambda *_a, **_k: 72)
    monkeypatch.setattr(activity.os, "fstat", lambda _fd: (_ for _ in ()).throw(
        KeyboardInterrupt("fstat interrupted")))
    monkeypatch.setattr(
        activity.os, "close",
        lambda fd: closed.append(fd),
    )
    with pytest.raises(KeyboardInterrupt, match="fstat interrupted"):
        activity._json_record(record, guard=guard)
    assert record_fd == closed


def test_persistent_body_exception_closes_serving_then_root_and_preserves_original(
        monkeypatch):
    events = []
    root = _ClosingGuard("root", events, OSError("root close failed"))
    root.path = Path("/activity")
    root.mode = "windows"
    serving = _ClosingGuard("serving", events, OSError("serving close failed"))
    serving.path = Path("/activity/serving")
    serving.mode = "windows"
    monkeypatch.setattr(activity, "_open_root", lambda **_kwargs: root)
    monkeypatch.setattr(activity, "_open_serving", lambda *_args, **_kwargs: serving)
    monkeypatch.setattr(
        activity, "_atomic_write",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt("write interrupted")),
    )
    with pytest.raises(KeyboardInterrupt, match="write interrupted") as caught:
        activity.record_ollama_serving(
            served_name="owned", model="org/model", context=1,
            endpoint="http://127.0.0.1:11434", started_at=1.0,
            base_artifact_id="ollama-manifest-sha256:" + "a" * 64,
            served_artifact_id="ollama-manifest-sha256:" + "b" * 64)
    assert events == ["serving", "root"]
    assert len(caught.value.__notes__) == 2


def test_snapshot_serving_close_failure_closes_root_and_fails_closed(monkeypatch, tmp_path):
    events = []
    root = _ClosingGuard("root", events)
    root.path = tmp_path / "activity"
    root.mode = "windows"
    serving = _ClosingGuard("serving", events, OSError("serving close failed"))
    serving.path = root.path / "serving"
    serving.mode = "windows"
    monkeypatch.setattr(activity, "_open_root", lambda **_kwargs: root)
    monkeypatch.setattr(activity, "_open_serving", lambda *_args, **_kwargs: serving)
    monkeypatch.setattr(activity, "_record_names", lambda *_args, **_kwargs: [])
    assert activity.snapshot() == []
    assert events == ["serving", "root"]


def test_snapshot_root_close_failure_fails_closed(monkeypatch, tmp_path):
    events = []
    root = _ClosingGuard("root", events, OSError("root close failed"))
    root.path = tmp_path / "activity"
    root.mode = "windows"
    monkeypatch.setattr(activity, "_open_root", lambda **_kwargs: root)
    monkeypatch.setattr(activity, "_record_names", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(activity, "_live_ollama_serving", lambda *_args: [])
    assert activity.snapshot() == []
    assert events == ["root"]


def test_serving_reader_baseexception_closes_child_then_root(monkeypatch, tmp_path):
    events = []
    root = _ClosingGuard("root", events)
    root.path = tmp_path / "activity"
    root.mode = "windows"
    serving = _ClosingGuard("serving", events)
    serving.path = root.path / "serving"
    serving.mode = "windows"
    monkeypatch.setattr(activity, "_open_root", lambda **_kwargs: root)
    monkeypatch.setattr(activity, "_open_serving", lambda *_args, **_kwargs: serving)
    monkeypatch.setattr(
        activity, "_record_names",
        lambda *_args, **kwargs: [] if kwargs["guard"] is root else ["x.json"],
    )
    monkeypatch.setattr(
        activity, "_json_record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt("read interrupted")),
    )
    with pytest.raises(KeyboardInterrupt, match="read interrupted"):
        activity.snapshot()
    assert events == ["serving", "root"]


class _ClosingGuard:
    def __init__(self, name, events, error=None):
        self.name = name
        self.events = events
        self.error = error

    def close(self):
        self.events.append(self.name)
        if self.error:
            raise self.error


def test_close_guards_attempts_both_and_notes_body_exception():
    events = []
    original = RuntimeError("body failed")
    activity._close_guards([
        _ClosingGuard("serving", events, OSError("serving close failed")),
        _ClosingGuard("root", events, OSError("root close failed")),
    ], original=original)
    assert events == ["serving", "root"]
    assert len(original.__notes__) == 2


def test_close_guards_without_body_raises_first_and_notes_second():
    events = []
    with pytest.raises(OSError, match="serving close failed") as caught:
        activity._close_guards([
            _ClosingGuard("serving", events, OSError("serving close failed")),
            _ClosingGuard("root", events, OSError("root close failed")),
        ])
    assert events == ["serving", "root"]
    assert any("root close failed" in note for note in caught.value.__notes__)
