# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The push-only node's on-disk config — round-trip, absent-file, and owner-only mode."""
from __future__ import annotations

import os
import sys

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
                                 "ftp://x", "coordinator.example"])
def test_require_secure_url_rejects_insecure(bad):
    with pytest.raises(ValueError):
        config.require_secure_url(bad)


def test_load_is_none_when_absent():
    assert config.load() is None


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


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file mode is advisory on Windows")
def test_saved_file_is_owner_only():
    config.save(config.NodeConfig(server_url="https://c.example", session_token="SECRET"))
    mode = os.stat(config._config_path()).st_mode & 0o777
    assert mode == 0o600                       # a session token must never be group/world-readable
