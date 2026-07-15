# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Is a stored characterization ceiling stale for the model's current revision?

ARA remembers each model's fitted ``safe_context`` with the timestamp it was measured. If the
model's cache artifacts are later updated (a re-download, a new quant of the same id), that stored
ceiling was measured against a *different* model — governance should say so rather than silently
trust it (Rule #3).

The staleness test itself (cache mtime vs the stored timestamp) is engine-agnostic and depends
only on the standard HF cache layout. ARA owns that pure, standard-library-only logic here so core
code never imports a nested engine package in-process.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

_HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
_REVISION_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")


def _cache_dir(model_id: str) -> Path:
    default_hub = (Path(os.path.expanduser(os.environ["XDG_CACHE_HOME"]))
                   / "huggingface" / "hub"
                   if os.environ.get("XDG_CACHE_HOME") else _HUB)
    hub = (Path(os.path.expanduser(os.environ["HF_HUB_CACHE"]))
           if os.environ.get("HF_HUB_CACHE") else
           Path(os.path.expanduser(os.environ["HF_HOME"])) / "hub"
           if os.environ.get("HF_HOME") else default_hub)
    return hub / ("models--" + model_id.replace("/", "--"))


def _cache_updated_at(model_id: str) -> float | None:
    """Newest artifact mtime in any locally cached model snapshot."""
    root = _cache_dir(model_id)
    if not root.is_dir():
        return None
    latest: float | None = None
    for dirpath, _, filenames in os.walk(root / "snapshots"):
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                mtime = max(path.lstat().st_mtime, path.stat().st_mtime)
            except OSError:
                continue
            latest = mtime if latest is None else max(latest, mtime)
    return latest


def fit_is_stale(model_id: str, measured_at: str | None) -> bool:
    """Whether cache artifacts are newer than a characterization run."""
    if not measured_at:
        return False
    try:
        measured = datetime.fromisoformat(measured_at)
    except (TypeError, ValueError):
        return False
    if measured.tzinfo is None:
        measured = measured.replace(tzinfo=timezone.utc)
    cache_mtime = _cache_updated_at(model_id)
    if cache_mtime is None:
        return False
    # DB timestamps use second precision; avoid false positives within that second.
    return cache_mtime > measured.timestamp() + 1.0


def artifact_identity(model: str) -> str | None:
    """Identity of the exact local weights selected by *model*, without loading an engine."""
    if not isinstance(model, str):
        return None
    local = Path(model).expanduser()
    if model.lower().endswith(".gguf") and local.is_file():
        try:
            stat = local.stat()
            return (f"local-gguf:{local.resolve()}:{stat.st_dev}:{stat.st_ino}:"
                    f"{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            return None

    repo, separator, filename = model.partition(":")
    repo_id = repo if separator and filename.lower().endswith(".gguf") else model
    root = _cache_dir(repo_id)
    try:
        revision = (root / "refs" / "main").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if not _REVISION_RE.fullmatch(revision):
        return None
    if separator:
        selected = root / "snapshots" / revision / filename
        try:
            if not selected.is_file():
                return None
            blob = selected.resolve(strict=True)
            stat = blob.stat()
        except OSError:
            return None
        return f"hf-gguf:{repo_id}@{revision}:{filename}:{blob.name}:{stat.st_size}"
    if not (root / "snapshots" / revision).is_dir():
        return None
    return f"hf:{repo_id}@{revision}"


def artifact_matches(model: str, expected_artifact_id: str | None) -> bool:
    """Whether *model* still resolves to the exact artifact that authorized stored evidence."""
    return (isinstance(expected_artifact_id, str) and bool(expected_artifact_id)
            and artifact_identity(model) == expected_artifact_id)


def artifact_size_gb(model: str) -> float | None:
    """Exact selected GGUF size for cataloging a quant variant; otherwise unknown."""
    if not isinstance(model, str):
        return None
    local = Path(model).expanduser()
    if model.lower().endswith(".gguf") and local.is_file():
        try:
            return round(local.stat().st_size / 1e9, 3)
        except OSError:
            return None
    repo, separator, filename = model.partition(":")
    if not separator or not filename.lower().endswith(".gguf"):
        return None
    root = _cache_dir(repo)
    try:
        revision = (root / "refs" / "main").read_text(encoding="utf-8").strip()
        return round((root / "snapshots" / revision / filename).stat().st_size / 1e9, 3)
    except (OSError, UnicodeError):
        return None


def ceiling_is_stale(model_id: str, measured_at: str | None) -> bool:
    """True when *model_id*'s HF cache is newer than *measured_at* — the stored ceiling predates
    the current cached files and should be re-characterized.

    Conservative by design: a missing timestamp or an uncached/unknown model returns ``False`` (we
    never nag without evidence), and this is advisory only — callers warn, they do not block, since
    the measured ceiling is still the best number on record until a fresh ``ara characterize`` runs.
    """
    return fit_is_stale(model_id, measured_at)
