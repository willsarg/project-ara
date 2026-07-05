# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Thin, engine-free client for a local Ollama server.

stdlib only (``urllib`` + ``json``), lazy, with two patchable seams (``_get_json`` /
``_post_json``) so callers can probe and drive a local server without depending on one
running. Liveness (``version``) serves ``detect``; the ``serve`` tier adds inventory
(``tags``/``ps``) and the governed-model lifecycle (``create``/``load``).

See vault specs 2026-06-26-detect-ollama-liveness and 2026-06-26-ara-serve-governed-endpoint.
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


def _post_json(path: str, payload: dict, timeout: float) -> dict | None:
    """POST a JSON ``payload`` to ``base_url() + path`` and parse a JSON object response.
    Returns the dict, or ``None`` on any transport/parse failure. The POST counterpart to
    ``_get_json`` — tests monkeypatch ``urllib.request.urlopen`` here (or patch this seam)."""
    try:
        req = urllib.request.Request(
            base_url() + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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


def tags(timeout: float = 2.0) -> list[str] | None:
    """Installed model names via ``GET /api/tags`` (e.g. ``["qwen3:0.6b", ...]``), or ``None``
    when the server is unreachable / the payload is malformed. Malformed entries are skipped."""
    data = _get_json("/api/tags", timeout)
    if not data:
        return None
    models = data.get("models")
    if not isinstance(models, list):
        return None
    return [m["name"] for m in models
            if isinstance(m, dict) and isinstance(m.get("name"), str)]


def ps(timeout: float = 2.0) -> list[dict] | None:
    """Currently-loaded models via ``GET /api/ps`` — each dict carries ``name``,
    ``context_length``, ``size`` and ``size_vram`` (``size_vram < size`` ⇒ partial offload /
    spill). ``None`` when unreachable; non-dict entries are dropped."""
    data = _get_json("/api/ps", timeout)
    if not data:
        return None
    models = data.get("models")
    if not isinstance(models, list):
        return None
    return [m for m in models if isinstance(m, dict)]


def pull(name: str, timeout: float = 600.0) -> bool:
    """Fetch *name* into the local Ollama store via ``POST /api/pull`` (non-streamed). ``True`` on
    success. This is ``serve``'s get-out-of-the-way step — an uninstalled model is pulled rather
    than refused, so ``ara serve <model>`` is one command. (Streamed progress is a follow-up.)"""
    data = _post_json("/api/pull", {"model": name, "stream": False}, timeout)
    return bool(data and data.get("status") == "success")


def show(name: str, timeout: float = 30.0) -> dict | None:
    """Model detail via ``POST /api/show`` — carries ``model_info`` (architecture: ``block_count``,
    ``head_count_kv``, ``key_length``, ``context_length``), read locally from the model with no
    network. Feeds the engine-free *estimated* ceiling for a model ARA hasn't measured — the honest
    source for an Ollama-native model that HF can't describe. ``None`` on failure."""
    return _post_json("/api/show", {"model": name}, timeout)


def size_bytes(name: str, timeout: float = 2.0) -> int | None:
    """On-disk size (bytes) of installed model *name* from ``GET /api/tags``, or ``None`` when the
    server is unreachable / the model isn't listed / the size is malformed. The weights-footprint
    proxy for the analytic estimate (decimal bytes — what the estimator expects)."""
    data = _get_json("/api/tags", timeout)
    if not data or not isinstance(data.get("models"), list):
        return None
    for m in data["models"]:
        if isinstance(m, dict) and m.get("name") == name:
            s = m.get("size")
            return s if isinstance(s, int) else None
    return None


def create(name: str, from_model: str, num_ctx: int, timeout: float = 300.0) -> bool:
    """Create a derived model ``name`` from ``from_model`` with ``num_ctx`` **baked in** as a
    default parameter, via ``POST /api/create``. Baking the ceiling into the model is what makes
    it hold under arbitrary consumers — a plain ``/v1`` request reloads the base model at its
    default context, blowing past the safe wall (measured 2026-06-26). ``True`` on success."""
    data = _post_json("/api/create",
                      {"model": name, "from": from_model,
                       "parameters": {"num_ctx": num_ctx}, "stream": False}, timeout)
    return bool(data and data.get("status") == "success")


def delete(name: str, timeout: float = 30.0) -> bool:
    """Remove model *name* from the local store via ``DELETE /api/delete``. ``True`` on a 2xx.
    Used to clean up the throwaway probe models a characterization ramp bakes. Defensive: any
    transport failure returns ``False`` rather than raising."""
    try:
        req = urllib.request.Request(
            base_url() + "/api/delete", data=json.dumps({"model": name}).encode(),
            headers={"Content-Type": "application/json"}, method="DELETE")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError, TypeError):
        return False


def load(name: str, keep_alive: int = -1, timeout: float = 300.0) -> dict | None:
    """Warm-load ``name`` into memory via an empty-prompt ``POST /api/generate`` (so it appears
    in ``ps`` for verification) and hold it with ``keep_alive`` (``-1`` = until stopped).
    Returns the response dict, or ``None`` on failure."""
    return _post_json("/api/generate",
                      {"model": name, "prompt": "", "stream": False,
                       "keep_alive": keep_alive}, timeout)
