# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Hugging Face Hub search via the ``hf`` CLI — no Python hub dependency for the query.

Engine-agnostic: unlike the MLX engine's format-specific discovery, this searches the whole Hub
(filter by author/library at the call site if a backend wants to).
"""
from __future__ import annotations

import json
import subprocess


def search(query: str, *, limit: int = 20, author: str | None = None) -> list[dict] | None:
    """Search the Hub for models matching *query*, sorted by downloads.

    Returns a list of ``{id, downloads, likes}`` dicts, ``[]`` on a bad response, or
    ``None`` when the ``hf`` CLI is missing or the search fails.
    """
    cmd = ["hf", "models", "list", "--search", query,
           "--sort", "downloads", "--limit", str(limit), "--json"]
    if author:
        cmd += ["--author", author]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception:        # hf not installed, timeout, etc.
        return None
    if proc.returncode != 0:
        return None
    try:
        raw = json.loads(proc.stdout or "[]")
    except (ValueError, TypeError):
        return []
    return [{"id": m["id"], "downloads": m.get("downloads") or 0,
             "likes": m.get("likes") or 0} for m in raw]
