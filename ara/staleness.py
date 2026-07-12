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
from datetime import datetime, timezone
from pathlib import Path

_HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))


def _cache_dir(model_id: str) -> Path:
    return _HUB / ("models--" + model_id.replace("/", "--"))


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
    except ValueError:
        return False
    if measured.tzinfo is None:
        measured = measured.replace(tzinfo=timezone.utc)
    cache_mtime = _cache_updated_at(model_id)
    if cache_mtime is None:
        return False
    # DB timestamps use second precision; avoid false positives within that second.
    return cache_mtime > measured.timestamp() + 1.0


def ceiling_is_stale(model_id: str, measured_at: str | None) -> bool:
    """True when *model_id*'s HF cache is newer than *measured_at* — the stored ceiling predates
    the current cached files and should be re-characterized.

    Conservative by design: a missing timestamp or an uncached/unknown model returns ``False`` (we
    never nag without evidence), and this is advisory only — callers warn, they do not block, since
    the measured ceiling is still the best number on record until a fresh ``ara characterize`` runs.
    """
    return fit_is_stale(model_id, measured_at)
