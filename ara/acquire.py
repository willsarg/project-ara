# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Model acquisition — backend-neutral downloads into the HF cache.

``download(repo_id)`` fetches a model; ``repo_size_gb`` / ``free_disk_gb`` back the
pre-download disk check. Uses ``huggingface_hub`` directly, which produces the exact
cache layout that mlx_lm / wmx-suite reads.

No token required for ungated models (e.g. mlx-community/*). Set HF_TOKEN or
HUGGING_FACE_HUB_TOKEN for gated ones, or HF_ENDPOINT for a mirror.
"""
from __future__ import annotations

# Headroom we insist on beyond the raw download, so a fetch never fills the disk
# (unpacking, the snapshot's own .incomplete temp files, normal system churn).
DISK_BUFFER_GB = 2.0


def repo_size_gb(repo_id: str) -> float | None:
    """Total download size of *repo_id* in GB (decimal). None if it can't be read
    (offline, private, or an API hiccup) — callers treat None as 'size unknown'."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
        total = sum(s.size for s in (info.siblings or []) if s.size)
        return round(total / 1e9, 3) if total else None
    except Exception:
        return None


def free_disk_gb() -> float | None:
    """Free space (GB, decimal) on the volume holding the home directory."""
    import shutil
    from pathlib import Path

    try:
        return shutil.disk_usage(Path.home()).free / 1e9
    except Exception:
        return None


def download(repo_id: str) -> None:
    """Download *repo_id* into the HF cache. Network + disk only, no engine load.

    Hugging Face's tqdm progress bars are silenced so the caller owns the output
    (ARA prints its own one-line status); we restore the prior setting afterward.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars

    was_disabled = are_progress_bars_disabled()
    disable_progress_bars()
    try:
        snapshot_download(repo_id)
    finally:
        if not was_disabled:
            enable_progress_bars()
