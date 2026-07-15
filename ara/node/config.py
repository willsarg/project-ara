# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The push-only node's on-disk config — where it phones home and the tokens it carries.

A node holds three facts: the coordinator ``server_url``, the one-shot ``enrollment_token`` an
admin handed it, and (once approved) the durable ``session_token`` it auths work with. They live in
the node data dir as ``config.json`` (``ARA_NODE_DIR`` override for tests, else the OS data dir),
written mode 0600 with an owner-only same-directory atomic replacement: a session token is a
credential and must never be world/group-readable, even for the instant between create and chmod.
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import platformdirs

_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


def require_secure_url(url: str) -> None:
    """Fail closed unless *url* is https, or http to a loopback host (localhost/127.0.0.1/[::1]) for
    local dev. A node's enrollment and session tokens ride this URL as bearer credentials, so plain
    http to a remote coordinator would leak them across the network in cleartext (Rule #1)."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if (parsed.scheme == "https" and host) or (parsed.scheme == "http" and host in _LOOPBACK):
        return
    raise ValueError(
        f"insecure coordinator URL {url!r} — use https:// (http:// is allowed only for localhost). "
        f"A node's tokens would otherwise cross the network in cleartext.")


def node_dir() -> Path:
    """The node's state directory — ``ARA_NODE_DIR`` if set (tests), else the OS data dir."""
    override = os.environ.get("ARA_NODE_DIR")
    return Path(override) if override else Path(platformdirs.user_data_dir("ara")) / "node"


@dataclass
class NodeConfig:
    """The node's identity toward one coordinator: where to reach it and the tokens it presents."""

    server_url: str
    enrollment_token: str | None = None
    session_token: str | None = None


@dataclass
class PendingEnrollment:
    """Durable state needed to resume a one-shot enrollment handshake after interruption."""

    server_url: str
    enrollment_token: str
    enrollment_id: str | None = None


def _config_path():
    return node_dir() / "config.json"


def _pending_path():
    return node_dir() / "pending-enrollment.json"


def _fsync_parent(path: Path) -> None:
    """Make a completed rename durable on POSIX; directory handles are not portable to Windows."""
    if os.name == "nt":
        return
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _optional_token(data: dict, key: str, *, context: str) -> str | None:
    value = data.get(key)
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"invalid {context}: {key} must be a nonempty string or null")
    return value


def load() -> NodeConfig | None:
    """The stored config, or None if this node has never been pointed at a coordinator."""
    path = _config_path()
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("server_url"), str):
        raise ValueError("invalid node configuration: server_url must be a nonempty secure URL")
    try:
        require_secure_url(data["server_url"])
        enrollment_token = _optional_token(data, "enrollment_token", context="node configuration")
        session_token = _optional_token(data, "session_token", context="node configuration")
    except ValueError as exc:
        raise ValueError(f"invalid node configuration: {exc}") from exc
    return NodeConfig(data["server_url"], enrollment_token, session_token)


def load_pending() -> PendingEnrollment | None:
    """Return a resumable enrollment handshake, if one was durably started."""
    path = _pending_path()
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("invalid pending enrollment: expected a JSON object")
    server_url = data.get("server_url")
    token = data.get("enrollment_token")
    enrollment_id = data.get("enrollment_id")
    if (not isinstance(server_url, str) or not isinstance(token, str) or not token
            or (enrollment_id is not None
                and (not isinstance(enrollment_id, str) or not enrollment_id))):
        raise ValueError("invalid pending enrollment: missing or malformed fields")
    try:
        require_secure_url(server_url)
    except ValueError as exc:
        raise ValueError(f"invalid pending enrollment: {exc}") from exc
    return PendingEnrollment(server_url, token, enrollment_id)


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise OSError(f"refusing to replace credential-path symlink: {path}")
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def save(config: NodeConfig) -> None:
    """Persist *config* to the node data dir via an owner-only atomic replacement.

    The session token is a credential, so a same-directory temporary file is created 0600 up front
    and replaces the destination only after its contents are durable. This leaves an existing valid
    config intact if writing or replacement fails. On Windows the mode is advisory (ACLs govern)."""
    _write_json_atomic(_config_path(), dataclasses.asdict(config))


def save_pending(pending: PendingEnrollment) -> None:
    """Durably persist a one-shot handshake without replacing an active node config."""
    _write_json_atomic(_pending_path(), dataclasses.asdict(pending))


def clear_pending() -> None:
    """Remove completed enrollment state and durably record its absence."""
    path = _pending_path()
    if path.exists() or path.is_symlink():
        path.unlink()
        _fsync_parent(path)
