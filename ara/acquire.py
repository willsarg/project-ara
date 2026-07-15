# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Model acquisition — backend-neutral downloads into the HF cache.

``download(repo_id)`` fetches a model; ``repo_size_gb`` / ``free_disk_gb`` back the
pre-download disk check. Uses ``huggingface_hub`` directly, which produces the exact
cache layout that the native MLX engine reads.

No token required for ungated models (e.g. mlx-community/*). Set HF_TOKEN or
HUGGING_FACE_HUB_TOKEN for gated ones, or HF_ENDPOINT for a mirror.
"""
from __future__ import annotations

import os
import re

# Headroom we insist on beyond the raw download, so a fetch never fills the disk
# (unpacking, the snapshot's own .incomplete temp files, normal system churn).
DISK_BUFFER_GB = 2.0

# A well-formed Hugging Face repo id: ``name`` or ``org/name``, each segment starting with an
# alphanumeric. Rejects anything an out-of-process worker's argparse could mis-read as a flag or
# path — a leading ``-``, an ``=``, whitespace, ``..`` traversal, extra slashes. The model is a
# *sink arg* (it becomes argv for the engine worker), so ARA validates its shape before it ever
# leaves the process. Defensive: the value is a local CLI arg, but cheap to get right.
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


def valid_model_id(model: str) -> bool:
    """True if *model* is a well-formed HF repo id (``org/name`` or ``name``), safe to pass as a
    worker argv positional. Rejects flag-like / traversal / malformed values."""
    return isinstance(model, str) and _MODEL_ID_RE.match(model) is not None


def is_local_gguf(model: str) -> bool:
    """True if *model* points at an existing local ``.gguf`` file that's safe as a worker argv
    positional (never flag-like). The engine workers already resolve a ``.gguf`` path directly, so
    this lets the CLI accept loose GGUF files on disk (e.g. a local model library) without
    weakening the repo-id guard. The leading-``-`` ban preserves the argv-injection guarantee — a
    real file path is a safe positional. (Slug: 2026-06-25-local-gguf-cli-support)"""
    return (isinstance(model, str) and model.endswith(".gguf")
            and not model.startswith("-") and os.path.isfile(model))


def valid_repo_gguf_ref(model: str) -> bool:
    """True if *model* is the ``repo:filename.gguf`` quant-selector form — a specific quantization
    inside an HF repo. The engine workers' ``_resolve_gguf`` resolves this directly, and it's how a
    caller pins a quant (essential for quant-ladder benchmarking). Safe as a worker argv positional:
    the repo half is id-validated and the file half must be a non-flag ``.gguf``."""
    if not isinstance(model, str) or ":" not in model:
        return False
    if (is_local_gguf(model) or model.startswith(("/", "./", "../", "~/"))
            or re.match(r"^[A-Za-z]:[\\/]", model)):
        return False
    repo, _, fname = model.partition(":")
    return (valid_model_id(repo) and fname.endswith(".gguf")
            and not fname.startswith("-"))


def valid_model_ref(model: str) -> bool:
    """True if *model* is a usable model reference safe to pass to an engine worker: a well-formed
    HF repo id, a ``repo:filename.gguf`` quant selector, or a local ``.gguf`` file path. The single
    guard the CLI applies before a model becomes worker argv. (Slug: 2026-06-25-local-gguf-cli-support)"""
    return valid_model_id(model) or valid_repo_gguf_ref(model) or is_local_gguf(model)


_REASON_GATED = "gated"
_REASON_NOT_FOUND = "not_found"
_REASON_AUTH = "auth"
_REASON_OFFLINE = "offline"
_REASON_UNKNOWN = "unknown"


def classify_repo_error(exc: BaseException) -> str:
    """Map a Hugging Face (or network) exception to a small honest reason string.

    Returns one of: ``"gated"``, ``"not_found"``, ``"auth"``, ``"offline"``, ``"unknown"``.
    Pure function — safe to call with any exception type, including non-HF ones.
    Imported lazily so this module stays cheap at import time.
    """
    from huggingface_hub.errors import (
        GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError,
        OfflineModeIsEnabled, RepositoryNotFoundError,
    )

    if isinstance(exc, GatedRepoError):
        return _REASON_GATED
    if isinstance(exc, RepositoryNotFoundError):
        return _REASON_NOT_FOUND
    if isinstance(exc, (LocalEntryNotFoundError, OfflineModeIsEnabled)):
        return _REASON_OFFLINE
    if isinstance(exc, ConnectionError):
        return _REASON_OFFLINE
    if isinstance(exc, HfHubHTTPError) and getattr(
            getattr(exc, "response", None), "status_code", None) == 401:
        return _REASON_AUTH
    return _REASON_UNKNOWN


def probe_repo(repo_id: str) -> dict:
    """Probe *repo_id* and return ``{"size_gb": float|None, "reason": str|None}``.

    ``reason`` is None on success; one of the ``classify_repo_error`` strings on failure.
    ``size_gb`` is None when the size can't be read (empty repo or any error).
    Use this when the caller needs to surface *why* a fetch failed (e.g. the CLI).
    ``repo_size_gb`` is still the right call when only the size matters.
    """
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
        total = sum(s.size for s in (info.siblings or []) if s.size)
        return {"size_gb": round(total / 1e9, 3) if total else None, "reason": None}
    except Exception as exc:
        return {"size_gb": None, "reason": classify_repo_error(exc)}


def repo_size_gb(repo_id: str) -> float | None:
    """Total download size of *repo_id* in GB (decimal). None if it can't be read
    (offline, private, or an API hiccup) — callers treat None as 'size unknown'."""
    return probe_repo(repo_id)["size_gb"]


def free_disk_gb() -> float | None:
    """Free space (GB, decimal) on the volume holding the home directory."""
    import shutil
    from pathlib import Path

    try:
        return shutil.disk_usage(Path.home()).free / 1e9
    except Exception:
        return None


def download(repo_id: str, *, progress: bool = False) -> None:
    """Download *repo_id* into the HF cache. Network + disk only, no engine load.

    ``progress=True`` enables HF's native tqdm bars for the duration of this call;
    ``progress=False`` (default) silences them so the caller owns the output.
    The prior bar state is always restored in ``finally`` regardless of which path
    ran or whether the download succeeded.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars

    was_disabled = are_progress_bars_disabled()
    if progress:
        enable_progress_bars()
    else:
        disable_progress_bars()
    try:
        snapshot_download(repo_id)
    finally:
        if was_disabled:
            disable_progress_bars()
        else:
            enable_progress_bars()
