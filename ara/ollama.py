# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Thin, engine-free client for a local Ollama server.

stdlib only (``urllib`` + ``json``), lazy, with a single patchable seam so ``detect``
can probe liveness without depending on a running server. Tier 1 is liveness only
(``version()``); richer endpoints (tags/ps/show) arrive with later tiers.

See vault spec 2026-06-26-detect-ollama-liveness.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_HOST = "127.0.0.1:11434"


def base_url() -> str:
    """Resolve the Ollama server base URL from ``OLLAMA_HOST`` — accepts ``host:port``,
    ``http://host:port``, or a bare host — defaulting to ``http://127.0.0.1:11434``.
    No trailing slash."""
    host = os.environ.get("OLLAMA_HOST", "").strip() or DEFAULT_HOST
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/")


def _get_json(path: str, timeout: float) -> dict | None:
    """GET ``base_url() + path`` and parse a JSON object. Returns the dict, or ``None`` on
    any transport/parse failure (server down, refused, timeout, non-JSON, non-object).
    The single urllib seam — tests monkeypatch ``urllib.request.urlopen`` here."""
    try:
        with urllib.request.urlopen(base_url() + path, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def version(timeout: float = 0.5) -> str | None:
    """The running server's version via ``GET /api/version``, or ``None`` when the server
    isn't reachable/serving. ``None`` is the canonical 'not serving' signal."""
    data = _get_json("/api/version", timeout)
    if not data:
        return None
    v = data.get("version")
    return v if isinstance(v, str) else None
