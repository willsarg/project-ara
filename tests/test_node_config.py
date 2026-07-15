# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The push-only node's on-disk config — round-trip, absent-file, and owner-only mode."""
from __future__ import annotations

import os
import sys
import types

import pytest

from ara.node import config


@pytest.fixture(autouse=True)
def _node_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_NODE_DIR", str(tmp_path / "node"))


def test_node_dir_honors_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_NODE_DIR", str(tmp_path / "override"))
    assert config.node_dir() == tmp_path / "override"


def test_node_dir_defaults_to_os_data_dir(monkeypatch):
    monkeypatch.delenv("ARA_NODE_DIR", raising=False)
    d = config.node_dir()
    assert d.name == "node" and d.parent.name == "ara"   # platformdirs user_data_dir("ara")/node


def test_require_secure_url_accepts_https_and_local_http():
    config.require_secure_url("https://c.example")            # TLS → fine
    config.require_secure_url("http://localhost:8000")        # loopback dev → allowed
    config.require_secure_url("http://127.0.0.1:8000")
    config.require_secure_url("http://[::1]:8000")


@pytest.mark.parametrize("bad", ["http://coordinator.example", "http://10.0.0.5:8000",
                                 "ftp://x", "coordinator.example", "https:///missing-host"])
def test_require_secure_url_rejects_insecure(bad):
    with pytest.raises(ValueError):
        config.require_secure_url(bad)


def test_load_is_none_when_absent():
    assert config.load() is None


@pytest.mark.parametrize("value", [
    [], {}, {"server_url": 7}, {"server_url": ""},
    {"server_url": "http://remote.example"},
    {"server_url": "https://c.example", "enrollment_token": 7},
    {"server_url": "https://c.example", "session_token": ""},
])
def test_load_normalizes_invalid_saved_shapes_to_value_error(value):
    path = config._config_path()
    path.parent.mkdir(parents=True)
    path.write_text(__import__("json").dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid node configuration"):
        config.load()


def test_save_then_load_round_trips():
    cfg = config.NodeConfig(server_url="https://c.example", enrollment_token="ENR",
                            session_token="SES")
    config.save(cfg)
    loaded = config.load()
    assert loaded == cfg
    assert loaded.server_url == "https://c.example"
    assert loaded.enrollment_token == "ENR"
    assert loaded.session_token == "SES"


def test_save_defaults_tokens_to_none():
    config.save(config.NodeConfig(server_url="https://c.example"))
    loaded = config.load()
    assert loaded.enrollment_token is None and loaded.session_token is None


def test_pending_enrollment_round_trips_and_clears_without_touching_active_config():
    active = config.NodeConfig(server_url="https://old.example", session_token="LIVE")
    config.save(active)
    pending = config.PendingEnrollment(
        server_url="https://new.example", enrollment_token="ONE-SHOT", enrollment_id="e1")
    config.save_pending(pending)
    assert config.load_pending() == pending
    assert config.load() == active
    config.clear_pending()
    assert config.load_pending() is None and config.load() == active


@pytest.mark.parametrize("value", [
    [], {}, {"server_url": "https://c.example"},
    {"server_url": "http://remote.example", "enrollment_token": "ENR"},
])
def test_load_pending_rejects_invalid_shape(value):
    path = config._pending_path()
    path.parent.mkdir(parents=True)
    path.write_text(__import__("json").dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid pending enrollment"):
        config.load_pending()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file mode is advisory on Windows")
def test_saved_file_is_owner_only():
    config.save(config.NodeConfig(server_url="https://c.example", session_token="SECRET"))
    mode = os.stat(config._config_path()).st_mode & 0o777
    assert mode == 0o600                       # a session token must never be group/world-readable


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file mode is advisory on Windows")
def test_save_replaces_existing_permissive_file_owner_only():
    path = config._config_path()
    path.parent.mkdir(parents=True)
    path.write_text('{"server_url": "https://old.example"}', encoding="utf-8")
    path.chmod(0o666)

    config.save(config.NodeConfig(server_url="https://new.example", session_token="SECRET"))

    assert config.load().server_url == "https://new.example"
    assert os.stat(path).st_mode & 0o777 == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics differ on Windows")
def test_save_refuses_config_path_symlink_without_touching_target(tmp_path):
    target = tmp_path / "target.json"
    target.write_text('{"server_url": "https://old.example"}', encoding="utf-8")
    path = config._config_path()
    path.parent.mkdir(parents=True)
    path.symlink_to(target)

    with pytest.raises(OSError, match="symlink"):
        config.save(config.NodeConfig(server_url="https://new.example", session_token="SECRET"))

    assert path.is_symlink()
    assert target.read_text(encoding="utf-8") == '{"server_url": "https://old.example"}'


def test_save_replace_failure_preserves_prior_valid_config(monkeypatch):
    old = config.NodeConfig(server_url="https://old.example", session_token="OLD")
    config.save(old)

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(config.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        config.save(config.NodeConfig(server_url="https://new.example", session_token="NEW"))

    assert config.load() == old
    assert list(config._config_path().parent.glob(".config.json.*")) == []


def test_save_degrades_when_fchmod_is_unavailable(monkeypatch):
    monkeypatch.delattr(config.os, "fchmod")
    config.save(config.NodeConfig(server_url="https://c.example"))
    assert config.load().server_url == "https://c.example"


def test_parent_sync_is_a_noop_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(config.os, "name", "nt")
    monkeypatch.setattr(config.os, "open", lambda *_a: pytest.fail("must not open a directory"))
    assert config._fsync_parent(tmp_path / "config.json") is None


def test_clear_pending_is_idempotent_when_absent():
    assert config.clear_pending() is None


def test_clear_session_if_current_is_atomic_and_preserves_newer_authority():
    config.save(config.NodeConfig(server_url="https://c.example", session_token="OLD"))
    assert config.clear_session_if_current("https://c.example", "OLD") is True
    assert config.load().session_token is None

    config.save(config.NodeConfig(server_url="https://c.example", session_token="NEW"))
    assert config.clear_session_if_current("https://c.example", "OLD") is False
    assert config.load().session_token == "NEW"


def test_credential_lock_uses_windows_locking_protocol(tmp_path, monkeypatch):
    calls = []
    fake = types.SimpleNamespace(
        LK_LOCK=1, LK_UNLCK=2,
        locking=lambda fd, operation, size: calls.append((fd, operation, size)),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(config, "_is_windows", lambda: True)
    monkeypatch.setenv("ARA_NODE_DIR", str(tmp_path / "node"))
    with config._credential_lock():
        calls.append("inside")
    assert calls[0][1:] == (fake.LK_LOCK, 1)
    assert calls[1] == "inside"
    assert calls[2][1:] == (fake.LK_UNLCK, 1)
