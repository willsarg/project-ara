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

from dataclasses import dataclass
import ipaddress
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_HOST = "127.0.0.1:11434"
_MANIFEST_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class OllamaEndpoint:
    """Normalized Ollama endpoint identity and its execution scope."""

    url: str | None
    scope: str


@dataclass(frozen=True)
class OllamaModel:
    """One installed model from a coherent ``GET /api/tags`` snapshot.

    Optional fields stay ``None`` (or an empty tuple) when an Ollama version omits them or
    returns a malformed value. ``digest`` is accepted only when it is a canonical manifest
    SHA-256, so callers never mistake unverified identity data for evidence.
    """

    name: str
    model: str | None = None
    digest: str | None = None
    size_bytes: int | None = None
    parent_model: str | None = None
    format: str | None = None
    family: str | None = None
    families: tuple[str, ...] = ()
    parameter_size: str | None = None
    quantization: str | None = None
    context_length: int | None = None
    embedding_length: int | None = None
    capabilities: tuple[str, ...] = ()
    remote_model: str | None = None
    remote_host: str | None = None
    scope: str = "local"

    @property
    def aliases(self) -> tuple[str, ...]:
        """Equivalent implicit-``latest`` spelling, when one exists."""
        alias = _latest_alias(self.name)
        return (alias,) if alias is not None else ()


@dataclass(frozen=True)
class OllamaProcess:
    """One resident model from a coherent ``GET /api/ps`` safety snapshot."""

    name: str
    model: str | None = None
    digest: str | None = None
    size_bytes: int | None = None
    size_vram_bytes: int | None = None
    context_length: int | None = None
    expires_at: str | None = None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _positive_int(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _model_scope(row: dict) -> str:
    remote_keys = ("remote_model", "remote_host")
    remote_values = tuple(row.get(key) for key in remote_keys)
    if any(isinstance(value, str) and value for value in remote_values):
        return "cloud"
    return "unknown" if any(key in row for key in remote_keys) else "local"


def _latest_alias(name: str) -> str | None:
    """Return the other spelling in an implicit-``latest`` pair.

    Only the final path component is inspected, so a registry port is not mistaken for a tag.
    """
    leaf = name.rsplit("/", 1)[-1]
    if leaf.endswith(":latest"):
        return name[:-len(":latest")]
    if ":" not in leaf:
        return name + ":latest"
    return None


def base_url() -> str:
    """Resolve the Ollama server base URL from ``OLLAMA_HOST`` — accepts ``host:port``,
    ``http://host:port``, or a bare host — defaulting to ``http://127.0.0.1:11434``.
    No trailing slash."""
    host = os.environ.get("OLLAMA_HOST", "").strip() or DEFAULT_HOST
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/")


def endpoint_authority(url: str | None = None) -> OllamaEndpoint:
    """Normalize an Ollama URL and classify it without contacting the endpoint."""
    raw = base_url() if url is None else url.strip()
    if "://" not in raw:
        raw = "http://" + raw
    try:
        parts = urllib.parse.urlsplit(raw)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return OllamaEndpoint(None, "unknown")
    if (parts.scheme.lower() not in ("http", "https") or not host
            or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment):
        return OllamaEndpoint(None, "unknown")
    host = host.lower()
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = host == "localhost"
    if is_loopback:
        scope = "loopback"
    elif host == "ollama.com" or host.endswith(".ollama.com"):
        scope = "cloud"
    else:
        scope = "remote"
    rendered_host = f"[{host}]" if ":" in host else host
    netloc = rendered_host + (f":{port}" if port is not None else "")
    path = parts.path.rstrip("/")
    normalized = urllib.parse.urlunsplit((parts.scheme.lower(), netloc, path, "", ""))
    return OllamaEndpoint(normalized, scope)


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


def inventory(timeout: float = 2.0) -> list[OllamaModel] | None:
    """Parse one coherent ``GET /api/tags`` snapshot into typed model records.

    ``None`` means the server or root payload could not be trusted. Individual rows without a
    usable name are skipped; malformed optional fields remain explicitly unknown.
    """
    data = _get_json("/api/tags", timeout)
    rows = data.get("models") if data else None
    if not isinstance(rows, list):
        return None
    result: list[OllamaModel] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        details = row.get("details")
        if not isinstance(details, dict):
            details = {}
        digest = row.get("digest")
        result.append(OllamaModel(
            name=name,
            model=_optional_string(row.get("model")),
            digest=(digest if isinstance(digest, str)
                    and _MANIFEST_DIGEST.fullmatch(digest) else None),
            size_bytes=_nonnegative_int(row.get("size")),
            parent_model=_optional_string(details.get("parent_model")),
            format=_optional_string(details.get("format")),
            family=_optional_string(details.get("family")),
            families=_string_tuple(details.get("families")),
            parameter_size=_optional_string(details.get("parameter_size")),
            quantization=_optional_string(details.get("quantization_level")),
            context_length=_positive_int(details.get("context_length")),
            embedding_length=_positive_int(details.get("embedding_length")),
            capabilities=_string_tuple(row.get("capabilities")),
            remote_model=_optional_string(row.get("remote_model")),
            remote_host=_optional_string(row.get("remote_host")),
            scope=_model_scope(row),
        ))
    return result


def find_model(models: list[OllamaModel], name: str) -> OllamaModel | None:
    """Resolve *name* in an inventory, accepting only the implicit ``:latest`` alias."""
    for model in models:
        if model.name == name:
            return model
    alias = _latest_alias(name)
    if alias is not None:
        for model in models:
            if model.name == alias:
                return model
    return None


def initial_governed_model_error(model: OllamaModel) -> str | None:
    """Return why *model* is outside ARA's first governed Ollama cell, if it is."""
    if model.scope == "cloud":
        return f"{model.name} is an Ollama cloud model; ARA's local governor will not execute it"
    if model.scope != "local":
        return f"{model.name} has ambiguous local/remote metadata; refusing local execution"
    if not isinstance(model.format, str) or model.format.casefold() != "gguf":
        found = repr(model.format) if model.format is not None else "unknown"
        return f"{model.name}'s format is {found}; initial Ollama support requires local GGUF"
    if "completion" not in model.capabilities:
        return f"{model.name} does not advertise Ollama's completion capability"
    return None


def tags(timeout: float = 2.0) -> list[str] | None:
    """Installed model names, retained as a compatibility view over :func:`inventory`."""
    models = inventory(timeout)
    return [model.name for model in models] if models is not None else None


def manifest_digest(name: str, timeout: float = 2.0) -> str | None:
    """Return Ollama's manifest SHA-256 for *name*, or ``None`` when it cannot be proven.

    The structured ``digest`` from ``/api/tags`` identifies the complete Ollama manifest (weights
    references, template, and parameters). It is intentionally not described as a weights digest.
    """
    models = inventory(timeout)
    model = find_model(models, name) if models is not None else None
    return model.digest if model is not None else None


def processes(timeout: float = 2.0) -> list[OllamaProcess] | None:
    """Parse one coherent ``GET /api/ps`` snapshot into typed resident-model records."""
    rows = ps(timeout)
    if rows is None:
        return None
    result: list[OllamaProcess] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        digest = row.get("digest")
        result.append(OllamaProcess(
            name=name,
            model=_optional_string(row.get("model")),
            digest=(digest if isinstance(digest, str)
                    and _MANIFEST_DIGEST.fullmatch(digest) else None),
            size_bytes=_nonnegative_int(row.get("size")),
            size_vram_bytes=_nonnegative_int(row.get("size_vram")),
            context_length=_positive_int(row.get("context_length")),
            expires_at=_optional_string(row.get("expires_at")),
        ))
    return result


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
    models = inventory(timeout)
    model = find_model(models, name) if models is not None else None
    return model.size_bytes if model is not None else None


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


def load(name: str, keep_alive: int | None = -1, timeout: float = 300.0) -> dict | None:
    """Warm-load ``name`` into memory via an empty-prompt ``POST /api/generate`` (so it appears
    in ``ps`` for verification). ``-1`` holds it until stopped; ``None`` omits ``keep_alive`` so
    the daemon's configured policy applies. Returns the response dict, or ``None`` on failure."""
    payload = {"model": name, "prompt": "", "stream": False}
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    return _post_json("/api/generate", payload, timeout)
