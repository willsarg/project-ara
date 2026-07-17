# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Thin, engine-free client for an Ollama server.

The HTTP client is stdlib-only and lazy. Local runtime authority additionally uses ARA's
cross-platform ``psutil`` dependency through one patchable socket-owner seam. Liveness
(``version``) serves ``detect``; the ``serve`` tier adds inventory (``tags``/``ps``) and the
governed-model lifecycle (``create``/``load``).

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

import psutil

DEFAULT_HOST = "127.0.0.1:11434"
_MANIFEST_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SEMVER = re.compile(r"(\d+)\.(\d+)\.(\d+)\Z")

# Verified directly against upstream Ollama source on 2026-07-17. The lower bound is the first
# release containing upstream change 20c3266e; the upper bound is the latest released source then
# available. Versions outside this closed interval fail closed until ARA verifies their default.
_PARALLELISM_ONE_MIN_VERSION = (0, 9, 7)
_PARALLELISM_ONE_MAX_VERSION = (0, 32, 1)

# Only configuration that can change runner placement or memory residency is captured. Values are
# read from the attributed daemon process, never ARA's own shell environment.
_MEMORY_CONFIG_KEYS = (
    "OLLAMA_NUM_PARALLEL",
    "OLLAMA_MAX_LOADED_MODELS",
    "OLLAMA_CONTEXT_LENGTH",
    "OLLAMA_FLASH_ATTENTION",
    "OLLAMA_KV_CACHE_TYPE",
    "OLLAMA_SCHED_SPREAD",
    "OLLAMA_GPU_OVERHEAD",
    "OLLAMA_LLM_LIBRARY",
    "OLLAMA_KEEP_ALIVE",
    "OLLAMA_IGPU_ENABLE",
    "OLLAMA_VULKAN",
    "LLAMA_ARG_FIT",
    "LLAMA_ARG_FIT_TARGET",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GGML_VK_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)


@dataclass(frozen=True)
class OllamaEndpoint:
    """Normalized Ollama endpoint identity and its execution scope."""

    url: str | None
    scope: str


@dataclass(frozen=True)
class OllamaListener:
    """One process-owned local TCP listener observed through the OS socket table."""

    pid: int
    create_time: float
    bind_host: str
    process_name: str
    executable: str
    command: tuple[str, ...]
    configured_inputs: tuple[tuple[str, str], ...]
    environment_readable: bool


@dataclass(frozen=True)
class OllamaRuntimeAuthority:
    """Evidence identifying the local server instance and its configured parallelism.

    ``issue is None`` means this authority satisfies the first governed Ollama cell. Other
    results remain useful display evidence, but cannot authorize characterization or execution.
    """

    endpoint: OllamaEndpoint
    server_version: str | None = None
    server_instance_id: str | None = None
    listener_pid: int | None = None
    listener_bind_host: str | None = None
    configured_inputs: tuple[tuple[str, str], ...] = ()
    configured_num_parallel: int | None = None
    configured_num_parallel_authority: str | None = None
    issue: str | None = None


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
    """One resident model from a coherent ``GET /api/ps`` safety snapshot.

    Ollama's ``context_length`` response field is effective context per request, not the runner's
    total allocation. The endpoint does not attest daemon parallelism, so it remains unknown.
    """

    name: str
    model: str | None = None
    digest: str | None = None
    size_bytes: int | None = None
    size_vram_bytes: int | None = None
    effective_context_per_request: int | None = None
    parallelism: None = None
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


def _configured_inputs(
        environment: dict[str, str], *, case_insensitive: bool | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return memory-relevant daemon inputs, respecting OS environment-key semantics."""
    if case_insensitive is None:
        case_insensitive = os.name == "nt"
    normalized = {(key.upper() if case_insensitive else key): value
                  for key, value in environment.items()
                  if isinstance(key, str) and isinstance(value, str)}
    return tuple((key, normalized[key]) for key in _MEMORY_CONFIG_KEYS if key in normalized)


def _local_tcp_listeners(port: int) -> list[OllamaListener]:
    """Read process-owned TCP listeners on *port*, skipping inaccessible processes individually.

    ``psutil.net_connections`` can abort the entire macOS scan when any unrelated process is
    inaccessible. Iterating processes keeps that permission failure local and supplies the same
    portable seam on macOS, Linux, and Windows.
    """
    listeners: list[OllamaListener] = []
    for process in psutil.process_iter():
        try:
            connections_fn = getattr(process, "net_connections", None)
            connections = (connections_fn(kind="tcp") if connections_fn is not None
                           else process.connections(kind="tcp"))
        except (psutil.Error, OSError):
            continue
        for connection in connections:
            local = connection.laddr
            if (connection.status != psutil.CONN_LISTEN
                    or getattr(local, "port", None) != port):
                continue
            try:
                name = process.name()
                executable = process.exe()
                command = tuple(process.cmdline())
                create_time = process.create_time()
                try:
                    environment = _configured_inputs(process.environ())
                    environment_readable = True
                except (psutil.Error, OSError):
                    environment = ()
                    environment_readable = False
            except (psutil.Error, OSError):
                continue
            listeners.append(OllamaListener(
                pid=process.pid,
                create_time=create_time,
                bind_host=str(getattr(local, "ip", "")),
                process_name=name,
                executable=executable,
                command=command,
                configured_inputs=environment,
                environment_readable=environment_readable,
            ))
    return listeners


def _is_loopback_address(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_ollama_listener(listener: OllamaListener) -> bool:
    identities = [listener.process_name, listener.executable]
    if listener.command:
        identities.append(listener.command[0])
    return any(os.path.basename(value).casefold() in {"ollama", "ollama.exe"}
               for value in identities if value)


def _server_instance_id(listener: OllamaListener) -> str:
    executable = listener.executable or (listener.command[0] if listener.command
                                         else listener.process_name)
    return f"{listener.pid}:{listener.create_time:.6f}:{executable}"


def _verified_parallelism_default(server_version: str) -> int | None:
    match = _SEMVER.fullmatch(server_version.strip().removeprefix("v"))
    if match is None:
        return None
    parsed = tuple(int(part) for part in match.groups())
    if _PARALLELISM_ONE_MIN_VERSION <= parsed <= _PARALLELISM_ONE_MAX_VERSION:
        return 1
    return None


def _configured_parallelism(
        listener: OllamaListener, server_version: str,
) -> tuple[int | None, str | None]:
    configured = dict(listener.configured_inputs)
    if "OLLAMA_NUM_PARALLEL" in configured:
        raw = configured["OLLAMA_NUM_PARALLEL"].strip().strip("\"'")
        if not raw.isdecimal():
            return None, None
        return int(raw), "process_environment"
    default = _verified_parallelism_default(server_version)
    return ((default, "exact_version_default") if default is not None else (None, None))


def runtime_authority(endpoint: OllamaEndpoint | None = None) -> OllamaRuntimeAuthority:
    """Attest the directly owned local Ollama listener and its initial governed config.

    This is read-only. Failure is represented as an ``issue`` and deliberately leaves detection
    and display available; callers performing governed work must require ``issue is None``.
    """
    endpoint = endpoint or endpoint_authority()
    if endpoint.scope != "loopback" or endpoint.url is None:
        return OllamaRuntimeAuthority(endpoint=endpoint, issue="endpoint_not_loopback")

    parts = urllib.parse.urlsplit(endpoint.url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    candidates = [candidate for candidate in _local_tcp_listeners(port)
                  if _is_loopback_address(candidate.bind_host)]
    by_pid = {candidate.pid: candidate for candidate in candidates}
    if not by_pid:
        return OllamaRuntimeAuthority(endpoint=endpoint, issue="listener_unattributed")
    if len(by_pid) != 1:
        return OllamaRuntimeAuthority(endpoint=endpoint, issue="listener_ambiguous")

    listener = next(iter(by_pid.values()))
    if not _is_ollama_listener(listener):
        return OllamaRuntimeAuthority(
            endpoint=endpoint, listener_pid=listener.pid,
            listener_bind_host=listener.bind_host, issue="listener_not_ollama")

    instance = _server_instance_id(listener)
    server_version = version()
    common = {
        "endpoint": endpoint,
        "server_version": server_version,
        "server_instance_id": instance,
        "listener_pid": listener.pid,
        "listener_bind_host": listener.bind_host,
        "configured_inputs": listener.configured_inputs,
    }
    if server_version is None:
        return OllamaRuntimeAuthority(**common, issue="server_unreachable")
    if not listener.environment_readable:
        return OllamaRuntimeAuthority(**common, issue="process_environment_unavailable")

    num_parallel, authority = _configured_parallelism(listener, server_version)
    if num_parallel is None:
        issue = "parallelism_unknown"
    elif num_parallel != 1:
        issue = "parallelism_not_one"
    else:
        issue = None
    return OllamaRuntimeAuthority(
        **common,
        configured_num_parallel=num_parallel,
        configured_num_parallel_authority=authority,
        issue=issue,
    )


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
            effective_context_per_request=_positive_int(row.get("context_length")),
            expires_at=_optional_string(row.get("expires_at")),
        ))
    return result


def ps(timeout: float = 2.0) -> list[dict] | None:
    """Currently-loaded models via ``GET /api/ps`` — each dict carries ``name``,
    ``context_length`` (effective per request, not total runner allocation), ``size`` and
    ``size_vram`` (``size_vram < size`` ⇒ partial offload / spill). Parallelism is not attested.
    ``None`` when unreachable; non-dict entries are dropped."""
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


def load(name: str, keep_alive: int | None = None, timeout: float = 300.0) -> dict | None:
    """Warm-load ``name`` into memory via an empty-prompt ``POST /api/generate`` (so it appears
    in ``ps`` for verification). The default omits ``keep_alive`` so the daemon's configured
    cache/eviction policy applies; ``-1`` remains available to explicit callers. Returns the
    response dict, or ``None`` on failure."""
    payload = {"model": name, "prompt": "", "stream": False}
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    return _post_json("/api/generate", payload, timeout)


def probe_generate(name: str, num_ctx: int, timeout: float = 300.0) -> bool:
    """Generate exactly one token at *num_ctx* for a characterization observation.

    Truncation and context shifting are disabled so Ollama must either honor the requested
    context or fail. The daemon's keep-alive policy remains authoritative for the shared runner.
    """
    data = _post_json(
        "/api/generate",
        {
            "model": name,
            "prompt": "ARA",
            "stream": False,
            "truncate": False,
            "shift": False,
            "options": {"num_ctx": num_ctx, "num_predict": 1},
        },
        timeout,
    )
    return bool(data and data.get("done") is True)


def warm_for_run(name: str, num_ctx: int, timeout: float = 300.0) -> dict | None:
    """Warm one governed runner without overriding the daemon's keep-alive policy."""

    return _post_json(
        "/api/generate",
        {
            "model": name,
            "prompt": "",
            "stream": False,
            "truncate": False,
            "shift": False,
            "options": {"num_ctx": num_ctx},
        },
        timeout,
    )


def generate_for_run(
    name: str,
    prompt: str,
    num_ctx: int,
    num_predict: int,
    timeout: float = 300.0,
) -> dict | None:
    """Buffer one native completion under explicit non-rewriting request options."""

    return _post_json(
        "/api/generate",
        {
            "model": name,
            "prompt": prompt,
            "stream": False,
            "truncate": False,
            "shift": False,
            "options": {"num_ctx": num_ctx, "num_predict": num_predict},
        },
        timeout,
    )
