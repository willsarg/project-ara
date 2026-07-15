# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Minimal registry of ARA-owned work that is live right now.

Process-bound writers use one atomic JSON file per activity. Governed Ollama serving
uses a separate persistent ownership manifest, corroborated read-only against the exact
live endpoint, name, and context. Readers order both forms deterministically; the final
record-id tie-breaker remains private.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path

import psutil
from platformdirs import user_data_path

_KINDS = frozenset({
    "characterizing", "benchmarking", "searching", "running", "serving", "hosting",
})
_REQUIRED_FIELDS = {"kind", "pid", "process_created_at", "started_at"}
_ALLOWED_FIELDS = _REQUIRED_FIELDS | {"model"}


@dataclass(frozen=True)
class Activity:
    """Display-safe public view of a validated live activity."""

    kind: str
    model: str | None
    pid: int | None
    started_at: float
    runtime: str | None = None
    served_name: str | None = None
    context: int | None = None
    endpoint: str | None = None
    base_artifact_id: str | None = None
    served_artifact_id: str | None = None


def activity_dir() -> Path:
    """Return a lexical absolute path without resolving or following filesystem links."""
    override = os.environ.get("ARA_ACTIVITY_DIR")
    if override:
        path = Path(override).expanduser()
    else:
        path = Path(user_data_path("ara")) / "activity"
    return Path(os.path.abspath(path))


def _validate_kind(kind: str) -> None:
    if not isinstance(kind, str) or kind not in _KINDS:
        raise ValueError(f"kind must be one of: {', '.join(sorted(_KINDS))}")


def _display_safe(value) -> bool:
    return (isinstance(value, str) and bool(value) and len(value) <= 512
            and all(unicodedata.category(char)[0] != "C"
                    and unicodedata.category(char) not in {"Zl", "Zp"}
                    for char in value))


def _validate_model(model: str | None) -> None:
    if model is not None and not _display_safe(model):
        raise ValueError("model must be a non-empty, display-safe string")


def _note_cleanup_failure(original: BaseException, action: str, cleanup: OSError) -> None:
    original.add_note(f"ARA could not {action}: {cleanup}")


_POSIX_DIR_FD = (os.name == "posix" and hasattr(os, "O_DIRECTORY")
                 and hasattr(os, "O_NOFOLLOW") and os.open in os.supports_dir_fd)

_WIN_GENERIC_READ = 0x80000000
_WIN_FILE_SHARE_READ = 0x00000001
_WIN_FILE_SHARE_WRITE = 0x00000002
_WIN_FILE_SHARE_DELETE = 0x00000004
_WIN_OPEN_EXISTING = 3
_WIN_FILE_ATTRIBUTE_DIRECTORY = 0x00000010
_WIN_FILE_ATTRIBUTE_NORMAL = 0x00000080
_WIN_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_WIN_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_WIN_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_WIN_REPLACE_ATTEMPTS = 5
_WIN_REPLACE_RETRY_SECONDS = 0.01


def _platform_mode() -> str | None:
    if os.name == "nt":
        return "windows"
    if _POSIX_DIR_FD:
        return "posix"
    return None


def _win_create_file(path: Path, access: int, share: int, creation: int, flags: int) -> int:
    """Create one Windows handle. Imported lazily so non-Windows never loads Win32 APIs."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                            wintypes.HANDLE)
    create_file.restype = wintypes.HANDLE
    handle = create_file(str(path), access, share, None, creation, flags, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(handle)


def _win_file_info(handle: int) -> tuple[int, int]:
    """Return ``(attributes, reparse_tag)`` for an already-open Windows handle."""
    import ctypes
    from ctypes import wintypes

    class _FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [("FileAttributes", wintypes.DWORD), ("ReparseTag", wintypes.DWORD)]

    info = _FileAttributeTagInfo()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_info = kernel32.GetFileInformationByHandleEx
    get_info.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    get_info.restype = wintypes.BOOL
    if not get_info(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(info.FileAttributes), int(info.ReparseTag)


def _win_close_handle(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close = kernel32.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL
    if not close(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _win_handle_to_fd(handle: int) -> int:
    import msvcrt
    return msvcrt.open_osfhandle(handle, os.O_RDONLY)


def _win_open_checked(path: Path, *, directory: bool) -> int:
    flags = _WIN_FILE_FLAG_OPEN_REPARSE_POINT
    if directory:
        flags |= _WIN_FILE_FLAG_BACKUP_SEMANTICS
    handle = _win_create_file(
        path, _WIN_GENERIC_READ, _WIN_FILE_SHARE_READ | _WIN_FILE_SHARE_WRITE,
        _WIN_OPEN_EXISTING, flags)
    try:
        attributes, tag = _win_file_info(handle)
        is_directory = bool(attributes & _WIN_FILE_ATTRIBUTE_DIRECTORY)
        is_reparse = bool(attributes & _WIN_FILE_ATTRIBUTE_REPARSE_POINT) or tag != 0
        if is_reparse or is_directory != directory:
            raise OSError(f"unsafe Windows activity path: {path}")
    except BaseException as original:
        try:
            _win_close_handle(handle)
        except OSError as cleanup:
            _note_cleanup_failure(original, "close the Windows activity handle", cleanup)
        raise
    return handle


def _win_open_directory(path: Path) -> int:
    return _win_open_checked(path, directory=True)


def _win_open_record(path: Path) -> int:
    return _win_open_checked(path, directory=False)


@dataclass
class _DirectoryGuard:
    path: Path
    mode: str
    value: int

    @property
    def dir_fd(self) -> int | None:
        return self.value if self.mode == "posix" else None

    def close(self) -> None:
        if self.mode == "posix":
            os.close(self.value)
        else:
            _win_close_handle(self.value)


def _close_guards(guards, *, original: BaseException | None = None) -> None:
    """Close every held directory, preserving the first failure or annotating *original*."""
    first = original
    for guard in guards:
        if guard is None:
            continue
        try:
            guard.close()
        except OSError as cleanup:
            if first is None:
                first = cleanup
            else:
                _note_cleanup_failure(first, "close an activity directory", cleanup)
    if original is None and first is not None:
        raise first


def _open_directory(path: Path, *, create: bool) -> _DirectoryGuard:
    mode = _platform_mode()
    if mode is None:
        raise OSError("no safe activity registry primitive is available on this platform")
    if create:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
    if mode == "posix":
        value = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    else:
        value = _win_open_directory(path)
    return _DirectoryGuard(path, mode, value)


def _open_root(*, create: bool, path: Path | None = None) -> _DirectoryGuard:
    return _open_directory(activity_dir() if path is None else path, create=create)


def _open_serving(root: _DirectoryGuard, *, create: bool) -> _DirectoryGuard:
    path = root.path / "serving"
    if root.mode == "posix":
        if create:
            try:
                os.mkdir("serving", mode=0o700, dir_fd=root.value)
            except FileExistsError:
                pass
        value = os.open(
            "serving", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root.value)
        return _DirectoryGuard(path, "posix", value)
    if create:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
    return _DirectoryGuard(path, "windows", _win_open_directory(path))


def _replace_record(source, target, *, guard: _DirectoryGuard) -> None:
    """Replace one record, tolerating transient Windows sharing violations."""
    if guard.mode == "posix":
        os.replace(source, target, src_dir_fd=guard.value, dst_dir_fd=guard.value)
        return
    attempt = 0
    while True:
        try:
            os.replace(source, target)
            return
        except PermissionError:
            attempt += 1
            if attempt >= _WIN_REPLACE_ATTEMPTS:
                raise
            time.sleep(_WIN_REPLACE_RETRY_SECONDS)


def _atomic_write(path: Path, record: dict, *, guard: _DirectoryGuard) -> None:
    """Write atomically while a no-follow/no-reparse directory guard is held.

    The private 0700 registry is per-user state, not an authentication boundary. A malicious
    same-user process able to replace this unique temporary entry can already write a schema-valid
    final record directly. The guarantees here are no symlink/reparse escape or directory swap,
    0600 temporary files, atomic visibility for cooperating ARA writers, malformed-record
    suppression, and accurate live corroboration—not same-user tamper resistance.
    """
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp")
    relative = guard.mode == "posix"
    temporary_arg = temporary.name if relative else temporary
    target_arg = path.name if relative else path
    open_kwargs = {"dir_fd": guard.value} if relative else {}
    descriptor = os.open(temporary_arg, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                         0o600, **open_kwargs)
    try:
        try:
            stream = os.fdopen(descriptor, "w", encoding="utf-8")
        except BaseException as original:
            try:
                os.close(descriptor)
            except OSError as cleanup:
                _note_cleanup_failure(original, "close the activity descriptor", cleanup)
            raise
        try:
            json.dump(record, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException as original:
            try:
                stream.close()
            except OSError as cleanup:
                _note_cleanup_failure(original, "close the activity stream", cleanup)
            raise
        stream.close()
        _replace_record(temporary_arg, target_arg, guard=guard)
    except BaseException as original:
        try:
            if relative:
                try:
                    os.unlink(temporary_arg, dir_fd=guard.value)
                except FileNotFoundError:
                    pass
            else:
                temporary.unlink(missing_ok=True)
        except OSError as cleanup:
            _note_cleanup_failure(original, "remove the partial activity record", cleanup)
        raise


class _Tracker:
    def __init__(self, kind: str, model: str | None):
        self._kind = kind
        self._model = model
        self._path = activity_dir() / f"{uuid.uuid4().hex}.json"
        self._guard: _DirectoryGuard | None = None

    def __enter__(self) -> None:
        pid = os.getpid()
        record = {
            "kind": self._kind,
            "pid": pid,
            "process_created_at": psutil.Process(pid).create_time(),
            "started_at": time.time(),
        }
        if self._model is not None:
            record["model"] = self._model
        self._guard = _open_root(create=True, path=self._path.parent)
        try:
            _atomic_write(self._path, record, guard=self._guard)
        except BaseException as original:
            _close_guards([self._guard], original=original)
            self._guard = None
            raise

    def __exit__(self, _exc_type, exc, _traceback) -> bool:
        cleanup_error: OSError | None = None
        try:
            if self._guard.mode == "windows":
                self._path.unlink(missing_ok=True)
            else:
                try:
                    os.unlink(self._path.name, dir_fd=self._guard.value)
                except FileNotFoundError:
                    pass
        except OSError as cleanup:
            if exc is None:
                cleanup_error = cleanup
            else:
                _note_cleanup_failure(exc, "remove the finished activity record", cleanup)
        finally:
            original = exc or cleanup_error
            _close_guards([self._guard], original=original)
            self._guard = None
        if cleanup_error is not None:
            raise cleanup_error
        return False


def track(kind: str, model: str | None = None) -> _Tracker:
    """Return a context manager that owns exactly one live activity record."""
    _validate_kind(kind)
    _validate_model(model)
    return _Tracker(kind, model)


_SERVING_FIELDS = {
    "runtime", "served_name", "model", "context", "endpoint", "started_at",
    "base_artifact_id", "served_artifact_id",
}
_LEGACY_SERVING_FIELDS = {
    "runtime", "served_name", "model", "context", "endpoint", "started_at",
}
_OLLAMA_ARTIFACT_PREFIX = "ollama-manifest-sha256:"


def _valid_ollama_artifact_id(value) -> bool:
    if not isinstance(value, str) or not value.startswith(_OLLAMA_ARTIFACT_PREFIX):
        return False
    digest = value.removeprefix(_OLLAMA_ARTIFACT_PREFIX)
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def _valid_serving_record(record) -> bool:
    return (isinstance(record, dict) and set(record) == _SERVING_FIELDS
            and record.get("runtime") == "ollama"
            and _display_safe(record.get("served_name"))
            and _display_safe(record.get("model"))
            and _valid_ollama_artifact_id(record.get("base_artifact_id"))
            and _valid_ollama_artifact_id(record.get("served_artifact_id"))
            and isinstance(record.get("context"), int)
            and not isinstance(record.get("context"), bool)
            and record["context"] > 0
            and _display_safe(record.get("endpoint"))
            and _number(record.get("started_at")))


def _valid_legacy_serving_record(record) -> bool:
    """One-release read compatibility for records written before manifest provenance.

    Legacy records remain visible only when their exact name/context/endpoint is live. They never
    authorize reuse, cleanup, or a new serving setup.
    """
    return (isinstance(record, dict) and set(record) == _LEGACY_SERVING_FIELDS
            and record.get("runtime") == "ollama"
            and _display_safe(record.get("served_name"))
            and _display_safe(record.get("model"))
            and isinstance(record.get("context"), int)
            and not isinstance(record.get("context"), bool)
            and record["context"] > 0
            and _display_safe(record.get("endpoint"))
            and _number(record.get("started_at")))


def validate_ollama_serving(*, served_name: str, model: str, context: int,
                            endpoint: str, base_artifact_id: str,
                            served_artifact_id: str) -> None:
    """Validate public persistent-ownership fields without touching the filesystem."""
    record = {
        "runtime": "ollama", "served_name": served_name, "model": model,
        "context": context, "endpoint": endpoint, "started_at": 0.0,
        "base_artifact_id": base_artifact_id, "served_artifact_id": served_artifact_id,
    }
    if not _valid_serving_record(record):
        raise ValueError("invalid Ollama serving activity")


def validate_ollama_serving_fields(*, served_name: str, model: str, context: int,
                                   endpoint: str) -> None:
    """Validate side-effect-facing identity fields before artifact IDs are observable."""
    if (not _display_safe(served_name) or not _display_safe(model)
            or not isinstance(context, int) or isinstance(context, bool) or context <= 0
            or not _display_safe(endpoint)):
        raise ValueError("invalid Ollama serving activity")


def record_ollama_serving(*, served_name: str, model: str, context: int,
                          endpoint: str, base_artifact_id: str, served_artifact_id: str,
                          started_at: float | None = None) -> Path:
    """Atomically own one exactly identified governed model on an Ollama endpoint."""
    validate_ollama_serving(
        served_name=served_name, model=model, context=context, endpoint=endpoint,
        base_artifact_id=base_artifact_id, served_artifact_id=served_artifact_id)
    record = {
        "runtime": "ollama",
        "served_name": served_name,
        "model": model,
        "context": context,
        "endpoint": endpoint,
        "base_artifact_id": base_artifact_id,
        "served_artifact_id": served_artifact_id,
        "started_at": time.time() if started_at is None else started_at,
    }
    if not _number(record["started_at"]):
        raise ValueError("invalid Ollama serving activity")
    identity = hashlib.sha256(
        f"{endpoint}\0{served_name}".encode("utf-8")).hexdigest()
    root_path = activity_dir()
    root = _open_root(create=True, path=root_path)
    serving: _DirectoryGuard | None = None
    try:
        serving = _open_serving(root, create=True)
        path = serving.path / f"{identity}.json"
        _atomic_write(path, record, guard=serving)
    except BaseException as original:
        _close_guards([serving, root], original=original)
        raise
    _close_guards([serving, root])
    return path


def _number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _open_record_fd(path: Path, guard: _DirectoryGuard) -> int | None:
    """Open and validate one regular record without following a link or blocking on a FIFO."""
    windows_handle: int | None = None
    try:
        if guard.mode == "posix":
            descriptor = os.open(
                path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=guard.value)
        else:
            windows_handle = _win_open_record(path)
            try:
                descriptor = _win_handle_to_fd(windows_handle)
            except BaseException as original:
                try:
                    _win_close_handle(windows_handle)
                except OSError as cleanup:
                    _note_cleanup_failure(original, "close the activity record", cleanup)
                raise
        try:
            info = os.fstat(descriptor)
        except BaseException as original:
            try:
                os.close(descriptor)
            except OSError as cleanup:
                _note_cleanup_failure(original, "close the activity record", cleanup)
            if isinstance(original, OSError):
                return None
            raise
        if not stat.S_ISREG(info.st_mode):
            os.close(descriptor)
            return None
        return descriptor
    except OSError:
        return None


def _json_record(path: Path, *, guard: _DirectoryGuard):
    descriptor = _open_record_fd(path, guard)
    if descriptor is None:
        return None
    try:
        stream = os.fdopen(descriptor, "r", encoding="utf-8")
    except BaseException as original:
        try:
            os.close(descriptor)
        except OSError as cleanup:
            _note_cleanup_failure(original, "close the activity record", cleanup)
        raise
    try:
        record = json.load(stream)
    except (OSError, UnicodeError, ValueError):
        record = None
    except BaseException as original:
        try:
            stream.close()
        except OSError as cleanup:
            _note_cleanup_failure(original, "close the activity stream", cleanup)
        raise
    stream.close()
    return record


def _read_record(path: Path, *, guard: _DirectoryGuard) -> Activity | None:
    record = _json_record(path, guard=guard)
    if not isinstance(record, dict) or set(record) - _ALLOWED_FIELDS \
            or not _REQUIRED_FIELDS.issubset(record):
        return None
    kind = record["kind"]
    model = record.get("model")
    pid = record["pid"]
    created = record["process_created_at"]
    started = record["started_at"]
    if not isinstance(kind, str) or kind not in _KINDS \
            or (model is not None and not _display_safe(model)):
        return None
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    if not _number(created) or not _number(started):
        return None
    try:
        process = psutil.Process(pid)
        if not process.is_running() or process.create_time() != created:
            return None
    except psutil.Error:
        return None
    return Activity(kind=kind, model=model, pid=pid, started_at=float(started))


def _record_names(directory: Path, *, guard: _DirectoryGuard) -> list[str]:
    """List regular JSON record names without following directory entries."""
    try:
        if guard.mode == "windows":
            return sorted(path.name for path in directory.iterdir()
                          if path.suffix == ".json")
        with os.scandir(guard.value) as entries:
            return sorted(entry.name for entry in entries
                          if entry.name.endswith(".json")
                          and entry.is_file(follow_symlinks=False))
    except OSError:
        return []


def _read_serving_records(root: _DirectoryGuard) -> list[tuple[dict, str]]:
    try:
        serving = _open_serving(root, create=False)
    except OSError:
        return []
    found = []
    try:
        for name in _record_names(serving.path, guard=serving):
            record = _json_record(serving.path / name, guard=serving)
            if _valid_serving_record(record) or _valid_legacy_serving_record(record):
                found.append((record, name))
    except BaseException as original:
        _close_guards([serving], original=original)
        raise
    _close_guards([serving])
    return found


def _live_ollama_serving(root: _DirectoryGuard) -> list[tuple[Activity, str]]:
    records = _read_serving_records(root)
    if not records:
        return []
    # Import lazily: status remains core/engine-free and contacts Ollama only when ARA has a
    # schema-valid ownership claim worth corroborating.
    from ara import ollama

    endpoint = ollama.base_url()
    matching = [(record, name) for record, name in records
                if record["endpoint"] == endpoint]
    if not matching:
        return []
    loaded = ollama.ps()
    if not isinstance(loaded, list):
        return []
    live = []
    for record, name in matching:
        served = record["served_name"]
        legacy = set(record) == _LEGACY_SERVING_FIELDS
        expected_served = None
        if not legacy:
            base_digest = ollama.manifest_digest(record["model"])
            expected_base = _OLLAMA_ARTIFACT_PREFIX + base_digest if base_digest else None
            if expected_base != record["base_artifact_id"]:
                continue
            expected_served = record["served_artifact_id"].removeprefix(_OLLAMA_ARTIFACT_PREFIX)
        entry = next((item for item in loaded if isinstance(item, dict)
                      and isinstance(item.get("name"), str)
                      and item["name"] in (served, served + ":latest")
                      and isinstance(item.get("context_length"), int)
                      and not isinstance(item["context_length"], bool)
                      and item["context_length"] > 0
                      and item["context_length"] == record["context"]
                      and (legacy or item.get("digest") == expected_served)), None)
        if entry is None:
            continue
        live.append((Activity(
            kind="serving", model=record["model"], pid=None,
            started_at=float(record["started_at"]), runtime="ollama",
            served_name=served, context=record["context"], endpoint=endpoint,
            base_artifact_id=record.get("base_artifact_id"),
            served_artifact_id=record.get("served_artifact_id"),
        ), name))
    return live


def snapshot() -> list[Activity]:
    """Read live records without creating, repairing, or deleting anything."""
    directory = activity_dir()
    try:
        root = _open_root(create=False, path=directory)
    except OSError:
        return []
    found: list[tuple[Activity, str]] = []
    try:
        for name in _record_names(directory, guard=root):
            parsed = _read_record(directory / name, guard=root)
            if parsed is not None:
                found.append((parsed, name))
        found.extend(_live_ollama_serving(root))
    except OSError as original:
        _close_guards([root], original=original)
        return []
    except BaseException as original:
        _close_guards([root], original=original)
        raise
    try:
        _close_guards([root])
    except OSError:
        return []
    found.sort(key=lambda pair: (
        pair[0].started_at, pair[0].pid or 0, pair[0].kind,
        pair[0].model or "", pair[1],
    ))
    return [item for item, _record_id in found]
