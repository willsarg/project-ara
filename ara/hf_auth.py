# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""HF token lifecycle — set / clear / inspect the standard Hugging Face token store.

Reads and writes ``huggingface_hub.constants.HF_TOKEN_PATH`` — the same file that
``detect``, ``snapshot_download``, and all engine workers already read, so a token
stored here is immediately usable everywhere without extra plumbing.

All imports from ``huggingface_hub`` are lazy (inside functions) so this module is
cheap at import time and stays engine-free.

Public API:
  set_token(token, *, verify=True) -> dict
  clear_token()                    -> dict
  status()                         -> dict
"""
from __future__ import annotations

import os
from pathlib import Path


# --------------------------------------------------------------------------- #
# patchable seams — monkeypatched in tests to avoid network calls
# --------------------------------------------------------------------------- #

def _whoami(token: str) -> dict:
    """Thin wrapper around huggingface_hub.whoami — exists so tests can monkeypatch."""
    from huggingface_hub import whoami as _hf_whoami
    return _hf_whoami(token=token)


def _get_token() -> str | None:
    """Thin wrapper around huggingface_hub.get_token — exists so tests can monkeypatch."""
    from huggingface_hub import get_token as _hf_get_token
    return _hf_get_token()


def _token_path() -> Path:
    """Return the standard HF token file path. Indirection so tests can monkeypatch."""
    from huggingface_hub.constants import HF_TOKEN_PATH
    return Path(HF_TOKEN_PATH)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _env_token_present() -> bool:
    """True if HF_TOKEN or HUGGING_FACE_HUB_TOKEN is set in the environment."""
    return "HF_TOKEN" in os.environ or "HUGGING_FACE_HUB_TOKEN" in os.environ


def has_token() -> bool:
    """True if any HF token is available — env or the standard token store (what `ara hf login`
    writes). Lets the CLI nudge toward authenticating only when it'd actually help."""
    return _get_token() is not None


def _write_token(path: Path, token: str) -> None:
    """Write *token* to *path* with owner-only permissions — a token is a credential, never
    leave it world-readable. Creates the dir 0o700 and the file 0o600 (atomically at create via
    os.open, then chmod to enforce on a pre-existing file too). chmod is best-effort: on
    filesystems/OSes without POSIX modes (e.g. Windows) it's a harmless no-op, so failures are
    swallowed rather than blocking the user from saving their token."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, token.encode())
    finally:
        os.close(fd)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _classify_whoami_error(exc: BaseException) -> str:
    """Classify a whoami exception into a small honest reason string.

    Uses attribute inspection only — the caller need not construct real HF exceptions.
    Returns one of: ``"invalid"``, ``"offline"``, ``"unknown"``.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return "invalid"
    if isinstance(exc, ConnectionError):
        return "offline"
    name = type(exc).__name__
    if "Connection" in name or "Timeout" in name:
        return "offline"
    return "unknown"


def _whoami_name(token: str) -> tuple[str | None, str | None]:
    """Call whoami; return (username, None) on success or (None, error_code) on failure."""
    try:
        info = _whoami(token)
        return info.get("name"), None
    except Exception as exc:
        return None, _classify_whoami_error(exc)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def set_token(token: str, *, verify: bool = True) -> dict:
    """Store *token* in the standard HF token file.

    Returns ``{"saved": bool, "user": str|None, "verified": bool, "error": str|None}``.

    - Empty token → refused (not written); error="empty".
    - verify=True + 401/403 response → refused (not written); error="invalid".
    - verify=True + offline/unknown → saved with verified=False and the error code.
    - verify=False → saved without calling whoami; user=None, verified=False, error=None.
    """
    token = token.strip()
    if not token:
        return {"saved": False, "user": None, "verified": False, "error": "empty"}

    user: str | None = None
    verified: bool = False
    error: str | None = None

    if verify:
        name, err = _whoami_name(token)
        if err == "invalid":
            return {"saved": False, "user": None, "verified": False, "error": "invalid"}
        if err is None:
            user, verified = name, True
        else:
            # offline or unknown: still save — user explicitly asked
            user, verified, error = None, False, err

    _write_token(_token_path(), token)
    return {"saved": True, "user": user, "verified": verified, "error": error}


def clear_token() -> dict:
    """Remove the stored token file if it exists.

    Returns ``{"removed": bool, "shadowed_by_env": bool}``.
    ``shadowed_by_env`` warns the user that an env-var token is still active.
    """
    p = _token_path()
    removed = False
    if p.exists():
        p.unlink()
        removed = True
    return {"removed": removed, "shadowed_by_env": _env_token_present()}


def status() -> dict:
    """Inspect the current token state without modifying anything.

    Returns ``{"present": bool, "source": "env"|"file"|None,
               "user": str|None, "verified": bool|None, "error": str|None}``.
    Calls whoami to get the username when a token is present.
    Never prints or logs the token value.
    """
    tok = _get_token()
    if not tok:
        return {"present": False, "source": None, "user": None, "verified": None, "error": None}

    source = "env" if _env_token_present() else "file"
    name, err = _whoami_name(tok)
    if err is None:
        return {"present": True, "source": source, "user": name, "verified": True, "error": None}
    return {"present": True, "source": source, "user": None, "verified": False, "error": err}
