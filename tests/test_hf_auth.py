# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""tests for ara/hf_auth.py — TDD gate, 100% branch coverage.

Slug: 2026-06-24-hf-token-auth

Strategy for patching whoami:
  hf_auth._whoami_name() calls `whoami(token=token)` where `whoami` is imported lazily
  inside the function via `from huggingface_hub import whoami`. We expose a module-level
  `_whoami` attribute (the bound function after first import) and monkeypatch THAT, so
  tests never hit the network. The indirection is minimal: one `_whoami` name at module
  scope in hf_auth, patched via monkeypatch.setattr(hf_auth, "_whoami", ...).
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import ara.hf_auth as hf_auth


# --------------------------------------------------------------------------- #
# helpers — small fake exception types
# --------------------------------------------------------------------------- #

class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class _FakeHfHTTPError(Exception):
    """Mimics HfHubHTTPError's .response attribute."""
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.response = _FakeResponse(status_code)


class _FakeConnectionError(ConnectionError):
    pass


class _TimeoutNamedError(Exception):
    """An exception whose class name contains 'Timeout' — classified as offline."""
    pass


_TimeoutNamedError.__name__ = "TimeoutError"
_TimeoutNamedError.__qualname__ = "TimeoutError"


# --------------------------------------------------------------------------- #
# _classify_whoami_error
# --------------------------------------------------------------------------- #

def test_classify_401():
    exc = _FakeHfHTTPError(401)
    assert hf_auth._classify_whoami_error(exc) == "invalid"


def test_classify_403():
    exc = _FakeHfHTTPError(403)
    assert hf_auth._classify_whoami_error(exc) == "invalid"


def test_classify_connection_error():
    exc = _FakeConnectionError("no route to host")
    assert hf_auth._classify_whoami_error(exc) == "offline"


def test_classify_timeout_named():
    exc = _TimeoutNamedError("timed out")
    assert hf_auth._classify_whoami_error(exc) == "offline"


def test_classify_connection_in_name():
    """A class whose name contains 'Connection' is also classified offline."""
    class FakeConnReset(Exception):
        pass
    FakeConnReset.__name__ = "ConnectionResetError"
    FakeConnReset.__qualname__ = "ConnectionResetError"
    exc = FakeConnReset("reset by peer")
    assert hf_auth._classify_whoami_error(exc) == "offline"


def test_classify_plain_exception():
    exc = ValueError("something went wrong")
    assert hf_auth._classify_whoami_error(exc) == "unknown"


# --------------------------------------------------------------------------- #
# _token_path indirection — monkeypatched throughout all tests via tmp_path
# --------------------------------------------------------------------------- #

@pytest.fixture
def token_path(monkeypatch, tmp_path):
    """Point _token_path() at a tmp path so tests never touch ~/.cache/huggingface/token."""
    p = tmp_path / "hf_token"
    monkeypatch.setattr(hf_auth, "_token_path", lambda: p)
    return p


# --------------------------------------------------------------------------- #
# _env_token_present
# --------------------------------------------------------------------------- #

def test_env_token_present_when_hf_token_set(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert hf_auth._env_token_present() is True


def test_env_token_present_when_hugging_face_hub_token_set(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "tok")
    assert hf_auth._env_token_present() is True


def test_env_token_absent(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert hf_auth._env_token_present() is False


def test_has_token_true_when_available(monkeypatch):
    monkeypatch.setattr(hf_auth, "_get_token", lambda: "hf_xyz")
    assert hf_auth.has_token() is True


def test_has_token_false_when_none(monkeypatch):
    monkeypatch.setattr(hf_auth, "_get_token", lambda: None)
    assert hf_auth.has_token() is False


# --------------------------------------------------------------------------- #
# set_token
# --------------------------------------------------------------------------- #

def test_set_token_valid(monkeypatch, token_path):
    monkeypatch.setattr(hf_auth, "_whoami", lambda token: {"name": "alice"})
    res = hf_auth.set_token("hf_valid_token")
    assert res == {"saved": True, "user": "alice", "verified": True, "error": None}
    assert token_path.read_text() == "hf_valid_token"


def test_set_token_401_refused_not_written(monkeypatch, token_path):
    """A 401 from whoami → refused, file NOT written."""
    monkeypatch.setattr(hf_auth, "_whoami", lambda token: (_ for _ in ()).throw(_FakeHfHTTPError(401)))
    res = hf_auth.set_token("hf_bad_token")
    assert res == {"saved": False, "user": None, "verified": False, "error": "invalid"}
    assert not token_path.exists()


def test_set_token_offline_saves_with_verified_false(monkeypatch, token_path):
    """Offline error → save anyway with verified=False."""
    monkeypatch.setattr(hf_auth, "_whoami", lambda token: (_ for _ in ()).throw(_FakeConnectionError()))
    res = hf_auth.set_token("hf_offline_token")
    assert res["saved"] is True
    assert res["verified"] is False
    assert res["error"] == "offline"
    assert token_path.read_text() == "hf_offline_token"


def test_set_token_unknown_error_saves(monkeypatch, token_path):
    """Unknown error → save with verified=False, error='unknown'."""
    monkeypatch.setattr(hf_auth, "_whoami", lambda token: (_ for _ in ()).throw(RuntimeError("surprise")))
    res = hf_auth.set_token("hf_some_token")
    assert res["saved"] is True
    assert res["verified"] is False
    assert res["error"] == "unknown"
    assert token_path.exists()


def test_set_token_empty_refused(monkeypatch, token_path):
    """Empty token → refused without calling whoami, file not written."""
    called = []
    monkeypatch.setattr(hf_auth, "_whoami", lambda t: called.append(t) or {"name": "x"})
    res = hf_auth.set_token("   ")
    assert res == {"saved": False, "user": None, "verified": False, "error": "empty"}
    assert not token_path.exists()
    assert called == []


def test_set_token_verify_false_saves_without_whoami(monkeypatch, token_path):
    """verify=False → save straight to disk, no whoami call."""
    called = []
    monkeypatch.setattr(hf_auth, "_whoami", lambda t: called.append(t) or {"name": "x"})
    res = hf_auth.set_token("hf_noverify", verify=False)
    assert res == {"saved": True, "user": None, "verified": False, "error": None}
    assert token_path.read_text() == "hf_noverify"
    assert called == []


def test_set_token_strips_whitespace(monkeypatch, token_path):
    monkeypatch.setattr(hf_auth, "_whoami", lambda token: {"name": "bob"})
    res = hf_auth.set_token("  hf_padded  ")
    assert token_path.read_text() == "hf_padded"
    assert res["saved"] is True


def test_set_token_creates_parent_dirs(monkeypatch, tmp_path):
    """Parent dirs are created if they don't exist."""
    deep = tmp_path / "a" / "b" / "c" / "token"
    monkeypatch.setattr(hf_auth, "_token_path", lambda: deep)
    monkeypatch.setattr(hf_auth, "_whoami", lambda t: {"name": "carol"})
    res = hf_auth.set_token("hf_deep")
    assert res["saved"] is True
    assert deep.read_text() == "hf_deep"


def test_set_token_writes_owner_only_permissions(monkeypatch, token_path):
    """A token is a credential — the file must be owner read/write only (0o600), parent 0o700."""
    monkeypatch.setattr(hf_auth, "_whoami", lambda token: {"name": "alice"})
    hf_auth.set_token("hf_secret")
    if os.name == "posix":   # POSIX modes are meaningless on Windows
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(token_path.parent.stat().st_mode) == 0o700


def test_set_token_tolerates_chmod_failure(monkeypatch, token_path):
    """chmod is best-effort (a no-op on non-POSIX filesystems); a failure must not block saving."""
    monkeypatch.setattr(hf_auth, "_whoami", lambda token: {"name": "alice"})

    def _boom(*a, **k):
        raise OSError("no chmod here")

    monkeypatch.setattr(hf_auth.os, "chmod", _boom)
    res = hf_auth.set_token("hf_secret")
    assert res["saved"] is True
    assert token_path.read_text() == "hf_secret"


# --------------------------------------------------------------------------- #
# clear_token
# --------------------------------------------------------------------------- #

def test_clear_token_when_file_present(monkeypatch, token_path):
    token_path.write_text("hf_tok")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    res = hf_auth.clear_token()
    assert res == {"removed": True, "shadowed_by_env": False}
    assert not token_path.exists()


def test_clear_token_when_file_absent(monkeypatch, token_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    res = hf_auth.clear_token()
    assert res == {"removed": False, "shadowed_by_env": False}


def test_clear_token_shadowed_by_env(monkeypatch, token_path):
    token_path.write_text("hf_tok")
    monkeypatch.setenv("HF_TOKEN", "env_tok")
    res = hf_auth.clear_token()
    assert res["removed"] is True
    assert res["shadowed_by_env"] is True


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #

def test_status_no_token(monkeypatch, token_path):
    """No file, no env → not present."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(hf_auth, "_get_token", lambda: None)
    res = hf_auth.status()
    assert res == {"present": False, "source": None, "user": None, "verified": None, "error": None}


def test_status_file_verified(monkeypatch, token_path):
    token_path.write_text("hf_valid")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(hf_auth, "_get_token", lambda: "hf_valid")
    monkeypatch.setattr(hf_auth, "_whoami", lambda token: {"name": "alice"})
    res = hf_auth.status()
    assert res == {"present": True, "source": "file", "user": "alice", "verified": True, "error": None}


def test_status_env_source(monkeypatch, token_path):
    """Token from env var → source is 'env'."""
    monkeypatch.setenv("HF_TOKEN", "env_tok")
    monkeypatch.setattr(hf_auth, "_get_token", lambda: "env_tok")
    monkeypatch.setattr(hf_auth, "_whoami", lambda token: {"name": "bob"})
    res = hf_auth.status()
    assert res["source"] == "env"
    assert res["user"] == "bob"
    assert res["verified"] is True


def test_status_offline(monkeypatch, token_path):
    token_path.write_text("hf_tok")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(hf_auth, "_get_token", lambda: "hf_tok")
    monkeypatch.setattr(hf_auth, "_whoami", lambda token: (_ for _ in ()).throw(_FakeConnectionError()))
    res = hf_auth.status()
    assert res["present"] is True
    assert res["verified"] is False
    assert res["error"] == "offline"
    assert res["user"] is None


def test_status_invalid(monkeypatch, token_path):
    token_path.write_text("hf_tok")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(hf_auth, "_get_token", lambda: "hf_tok")
    monkeypatch.setattr(hf_auth, "_whoami", lambda token: (_ for _ in ()).throw(_FakeHfHTTPError(401)))
    res = hf_auth.status()
    assert res["present"] is True
    assert res["verified"] is False
    assert res["error"] == "invalid"


# --------------------------------------------------------------------------- #
# wrapper seam coverage — exercise the real lazy-import bodies (no network needed
# for _token_path and _get_token; _whoami is tested via its stub path above)
# --------------------------------------------------------------------------- #

def test_token_path_real_returns_path():
    """_token_path() real body: imports HF_TOKEN_PATH and wraps it in Path."""
    # Restore real function (not monkeypatched here) — just call it.
    p = hf_auth._token_path.__wrapped__() if hasattr(hf_auth._token_path, "__wrapped__") \
        else hf_auth._token_path()
    assert isinstance(p, Path)


def test_get_token_real_returns_none_or_str(monkeypatch):
    """_get_token() real body: calls huggingface_hub.get_token().
    In CI / clean env there's no token → None. Either way, must not raise.
    """
    # Don't monkeypatch _get_token — call the real implementation directly.
    result = hf_auth._get_token()
    assert result is None or isinstance(result, str)


def test_whoami_real_raises_on_bogus_token():
    """_whoami() real body: calls huggingface_hub.whoami(). A bogus token raises."""
    import pytest
    with pytest.raises(Exception):
        hf_auth._whoami("definitely_not_a_real_token_xyzzy123")
