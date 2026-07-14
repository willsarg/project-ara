# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Minimal registry of ARA-owned work that is live right now.

Writers use one atomic JSON file per activity. Readers only observe complete live
records and order them by ``(started_at, pid, kind, model, record_id)``. The final
record-id tie-breaker is private and makes simultaneous records deterministic.
"""
from __future__ import annotations

import json
import math
import os
import time
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path

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
    pid: int
    started_at: float


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


def _atomic_write(path: Path, record: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
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
        os.replace(temporary, path)
    except BaseException as original:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup:
            _note_cleanup_failure(original, "remove the partial activity record", cleanup)
        raise


class _Tracker:
    def __init__(self, kind: str, model: str | None):
        self._kind = kind
        self._model = model
        self._path = activity_dir() / f"{uuid.uuid4().hex}.json"

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
        _atomic_write(self._path, record)

    def __exit__(self, _exc_type, exc, _traceback) -> bool:
        try:
            self._path.unlink(missing_ok=True)
        except OSError as cleanup:
            if exc is None:
                raise
            _note_cleanup_failure(exc, "remove the finished activity record", cleanup)
        return False


def track(kind: str, model: str | None = None) -> _Tracker:
    """Return a context manager that owns exactly one live activity record."""
    _validate_kind(kind)
    _validate_model(model)
    return _Tracker(kind, model)


def _number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _read_record(path: Path) -> Activity | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            record = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
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


def snapshot() -> list[Activity]:
    """Read live records without creating, repairing, or deleting anything."""
    directory = activity_dir()
    try:
        paths = sorted(path for path in directory.iterdir()
                       if path.suffix == ".json" and not path.is_symlink()
                       and path.is_file())
    except OSError:
        return []
    found: list[tuple[Activity, str]] = []
    for path in paths:
        parsed = _read_record(path)
        if parsed is not None:
            found.append((parsed, path.name))
    found.sort(key=lambda pair: (
        pair[0].started_at, pair[0].pid, pair[0].kind,
        pair[0].model or "", pair[1],
    ))
    return [item for item, _record_id in found]
