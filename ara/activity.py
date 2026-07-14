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
from typing import cast

import psutil
from platformdirs import user_data_path

_KINDS = frozenset({
    "characterizing", "benchmarking", "searching", "running", "serving",
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


def activity_dir() -> Path:
    """Return the per-user directory used for live activity records."""
    override = os.environ.get("ARA_ACTIVITY_DIR")
    if override:
        return Path(override).expanduser()
    return Path(user_data_path("ara")) / "activity"


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


_USE_DIR_FD = (os.name != "nt" and hasattr(os, "O_DIRECTORY")
               and hasattr(os, "O_NOFOLLOW") and os.open in os.supports_dir_fd)


def _fallback_identity(path: Path) -> tuple[int, int]:
    """Verify a fallback directory is real and return its stable filesystem identity."""
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise OSError(f"ARA activity directory is not a real directory: {path}")
    return info.st_dev, info.st_ino


def _open_root(*, create: bool) -> tuple[int | None, tuple[int, int] | None]:
    """Open the configured root without following it; create only for writer calls."""
    path = activity_dir()
    if create:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
    if _USE_DIR_FD:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        return os.open(path, flags), None
    return None, _fallback_identity(path)


def _same_fallback_directory(path: Path, identity: tuple[int, int]) -> None:
    if _fallback_identity(path) != identity:
        raise OSError(f"ARA activity directory changed during operation: {path}")


def _atomic_write(path: Path, record: dict, *, directory_fd: int | None = None,
                  fallback_identity: tuple[int, int] | None = None) -> None:
    """Write one record atomically, relative to a held directory when available."""
    if directory_fd is None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        identity = cast(tuple[int, int], fallback_identity)
        _same_fallback_directory(path.parent, identity)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp")
    temporary_arg = temporary.name if directory_fd is not None else temporary
    target_arg = path.name if directory_fd is not None else path
    open_kwargs = {"dir_fd": directory_fd} if directory_fd is not None else {}
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
        with stream:
            json.dump(record, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        if directory_fd is None:
            os.replace(temporary_arg, target_arg)
            _same_fallback_directory(path.parent, identity)
        else:
            os.replace(temporary_arg, target_arg,
                       src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    except BaseException as original:
        try:
            if directory_fd is None:
                temporary.unlink(missing_ok=True)
            else:
                try:
                    os.unlink(temporary_arg, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
        except OSError as cleanup:
            _note_cleanup_failure(original, "remove the partial activity record", cleanup)
        raise


class _Tracker:
    def __init__(self, kind: str, model: str | None):
        self._kind = kind
        self._model = model
        self._path = activity_dir() / f"{uuid.uuid4().hex}.json"
        self._directory_fd: int | None = None
        self._fallback_identity: tuple[int, int] | None = None

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
        self._directory_fd, self._fallback_identity = _open_root(create=True)
        try:
            _atomic_write(self._path, record, directory_fd=self._directory_fd,
                          fallback_identity=self._fallback_identity)
        except BaseException as original:
            if self._directory_fd is not None:
                try:
                    os.close(self._directory_fd)
                except OSError as cleanup:
                    _note_cleanup_failure(original, "close the activity directory", cleanup)
                self._directory_fd = None
            raise

    def __exit__(self, _exc_type, exc, _traceback) -> bool:
        cleanup_error: OSError | None = None
        try:
            if self._directory_fd is None:
                identity = cast(tuple[int, int], self._fallback_identity)
                _same_fallback_directory(self._path.parent, identity)
                self._path.unlink(missing_ok=True)
                _same_fallback_directory(self._path.parent, identity)
            else:
                try:
                    os.unlink(self._path.name, dir_fd=self._directory_fd)
                except FileNotFoundError:
                    pass
        except OSError as cleanup:
            if exc is None:
                cleanup_error = cleanup
            else:
                _note_cleanup_failure(exc, "remove the finished activity record", cleanup)
        finally:
            if self._directory_fd is not None:
                try:
                    os.close(self._directory_fd)
                except OSError as close_error:
                    original = exc or cleanup_error
                    if original is None:
                        raise
                    _note_cleanup_failure(
                        original, "close the activity directory", close_error)
                self._directory_fd = None
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
}


def _valid_serving_record(record) -> bool:
    return (isinstance(record, dict) and set(record) == _SERVING_FIELDS
            and record.get("runtime") == "ollama"
            and _display_safe(record.get("served_name"))
            and _display_safe(record.get("model"))
            and isinstance(record.get("context"), int)
            and not isinstance(record.get("context"), bool)
            and record["context"] > 0
            and _display_safe(record.get("endpoint"))
            and _number(record.get("started_at")))


def validate_ollama_serving(*, served_name: str, model: str, context: int,
                            endpoint: str) -> None:
    """Validate public persistent-ownership fields without touching the filesystem."""
    record = {
        "runtime": "ollama", "served_name": served_name, "model": model,
        "context": context, "endpoint": endpoint, "started_at": 0.0,
    }
    if not _valid_serving_record(record):
        raise ValueError("invalid Ollama serving activity")


def record_ollama_serving(*, served_name: str, model: str, context: int,
                          endpoint: str, started_at: float | None = None) -> Path:
    """Atomically own one exactly identified governed model on an Ollama endpoint."""
    validate_ollama_serving(
        served_name=served_name, model=model, context=context, endpoint=endpoint)
    record = {
        "runtime": "ollama",
        "served_name": served_name,
        "model": model,
        "context": context,
        "endpoint": endpoint,
        "started_at": time.time() if started_at is None else started_at,
    }
    if not _number(record["started_at"]):
        raise ValueError("invalid Ollama serving activity")
    identity = hashlib.sha256(
        f"{endpoint}\0{served_name}".encode("utf-8")).hexdigest()
    root_fd, root_identity = _open_root(create=True)
    serving = activity_dir() / "serving"
    serving_fd: int | None = None
    serving_identity: tuple[int, int] | None = None
    try:
        if root_fd is None:
            if serving.is_symlink():
                raise OSError("ARA serving activity directory must not be a symlink")
            serving.mkdir(mode=0o700, exist_ok=True)
            _same_fallback_directory(activity_dir(), root_identity)
            serving_identity = _fallback_identity(serving)
        else:
            try:
                os.mkdir("serving", mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
            serving_fd = os.open(
                "serving", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        path = serving / f"{identity}.json"
        _atomic_write(path, record, directory_fd=serving_fd,
                      fallback_identity=serving_identity)
        return path
    finally:
        if serving_fd is not None:
            os.close(serving_fd)
        if root_fd is not None:
            os.close(root_fd)


def _number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _json_record(path: Path, *, directory_fd: int | None = None):
    """Load one no-follow JSON file relative to a held directory when available."""
    try:
        if directory_fd is None:
            if path.is_symlink() or not path.is_file():
                return None
            with path.open("r", encoding="utf-8") as stream:
                return json.load(stream)
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _read_record(path: Path, *, directory_fd: int | None = None) -> Activity | None:
    record = _json_record(path, directory_fd=directory_fd)
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


def _record_names(directory: Path, *, directory_fd: int | None = None) -> list[str]:
    """List regular JSON record names without following directory entries."""
    try:
        if directory_fd is None:
            return sorted(path.name for path in directory.iterdir()
                          if path.suffix == ".json" and not path.is_symlink()
                          and path.is_file())
        with os.scandir(directory_fd) as entries:
            return sorted(entry.name for entry in entries
                          if entry.name.endswith(".json")
                          and entry.is_file(follow_symlinks=False))
    except OSError:
        return []


def _read_serving_records(directory: Path, *, root_fd: int | None = None,
                          root_identity: tuple[int, int] | None = None) -> list[tuple[dict, str]]:
    serving_fd: int | None = None
    if root_fd is None:
        _same_fallback_directory(
            directory.parent, cast(tuple[int, int], root_identity))
        if directory.is_symlink():
            return []
        try:
            serving_identity = _fallback_identity(directory)
        except OSError:
            return []
    else:
        try:
            serving_fd = os.open(
                "serving", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        except OSError:
            return []
        serving_identity = None
    found = []
    try:
        for name in _record_names(directory, directory_fd=serving_fd):
            record = _json_record(directory / name, directory_fd=serving_fd)
            if _valid_serving_record(record):
                found.append((record, name))
        if serving_identity is not None:
            _same_fallback_directory(directory, serving_identity)
    finally:
        if serving_fd is not None:
            os.close(serving_fd)
    return found


def _live_ollama_serving(directory: Path, *, root_fd: int | None = None,
                         root_identity: tuple[int, int] | None = None
                         ) -> list[tuple[Activity, str]]:
    records = _read_serving_records(
        directory / "serving", root_fd=root_fd, root_identity=root_identity)
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
        entry = next((item for item in loaded if isinstance(item, dict)
                      and isinstance(item.get("name"), str)
                      and item["name"] in (served, served + ":latest")
                      and isinstance(item.get("context_length"), int)
                      and not isinstance(item["context_length"], bool)
                      and item["context_length"] > 0
                      and item["context_length"] == record["context"]), None)
        if entry is None:
            continue
        live.append((Activity(
            kind="serving", model=record["model"], pid=None,
            started_at=float(record["started_at"]), runtime="ollama",
            served_name=served, context=record["context"], endpoint=endpoint,
        ), name))
    return live


def snapshot() -> list[Activity]:
    """Read live records without creating, repairing, or deleting anything."""
    directory = activity_dir()
    try:
        root_fd, root_identity = _open_root(create=False)
    except OSError:
        return []
    found: list[tuple[Activity, str]] = []
    try:
        for name in _record_names(directory, directory_fd=root_fd):
            parsed = _read_record(directory / name, directory_fd=root_fd)
            if parsed is not None:
                found.append((parsed, name))
        found.extend(_live_ollama_serving(
            directory, root_fd=root_fd, root_identity=root_identity))
        if root_identity is not None:
            _same_fallback_directory(directory, root_identity)
    except OSError:
        return []
    finally:
        if root_fd is not None:
            os.close(root_fd)
    found.sort(key=lambda pair: (
        pair[0].started_at, pair[0].pid or 0, pair[0].kind,
        pair[0].model or "", pair[1],
    ))
    return [item for item, _record_id in found]
