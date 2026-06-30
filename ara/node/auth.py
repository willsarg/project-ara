# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node's bearer token — generation, on-disk storage, and constant-time matching.

One token gates every node endpoint (read and action). Possession of the token IS the
authorization to act — the same "the flag is the consent" model the CLI uses, lifted to the API.
The token lives in the node data dir (``ARA_NODE_DIR`` override for tests, else the OS data dir),
mode 0600, generated on ``ara node install``.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

import platformdirs


def node_dir() -> Path:
    """The node's state directory — ``ARA_NODE_DIR`` if set (tests), else the OS data dir."""
    override = os.environ.get("ARA_NODE_DIR")
    return Path(override) if override else Path(platformdirs.user_data_dir("ara")) / "node"


def _token_path() -> Path:
    return node_dir() / "token"


def load_token() -> str | None:
    """The stored token, or None if the node hasn't been given one yet."""
    path = _token_path()
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def _write_token(token: str) -> str:
    path = _token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token, encoding="utf-8")
    path.chmod(0o600)          # owner-only; on Windows this sets the read-only attribute (no raise)
    return token


def ensure_token() -> str:
    """Return the existing token, generating + persisting one on first use (idempotent)."""
    existing = load_token()
    if existing:
        return existing
    return _write_token(secrets.token_urlsafe(32))


def rotate_token() -> str:
    """Replace the token with a fresh one and return it (invalidates the old token)."""
    return _write_token(secrets.token_urlsafe(32))


def token_matches(authorization: str | None) -> bool:
    """True iff an ``Authorization`` header carries ``Bearer <the node token>``.

    Constant-time compare (``secrets.compare_digest``) so a wrong token can't be timed out. False
    when no token is configured, the header is missing/empty, or the scheme isn't bearer.
    """
    token = load_token()
    if not token or not authorization:
        return False
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False
    return secrets.compare_digest(parts[1].strip(), token)
