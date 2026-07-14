# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""`ara node <sub>` CLI — enroll, run, the systemd lifecycle, and error paths."""
from __future__ import annotations

import io
import json

import pytest

import ara.cli as cli
from ara.node import agent, config, enroll, service


@pytest.fixture(autouse=True)
def _node_dir(tmp_path, monkeypatch):
    """Point the node data dir at a tmp sandbox so `enroll` never writes the real config."""
    monkeypatch.setenv("ARA_NODE_DIR", str(tmp_path / "node"))


@pytest.fixture
def con():
    """A Console writing styled text to a StringIO (asserted directly); `print(json)` still goes to
    stdout (capsys). Mirrors test_cli.py's make_console pattern."""
    buf = io.StringIO()
    return cli.Console(color=False, stream=buf), buf


def _node(cb, *args, **kw):
    c, _buf = cb
    return cli.render_node(c, ["node", *args], **kw)


# --- enroll (phone home) ---
def test_enroll_writes_config_and_runs_flow(con, monkeypatch):
    seen = {}
    monkeypatch.setattr(enroll, "enroll_flow", lambda cfg: seen.setdefault("cfg", cfg))
    assert _node(con, "enroll", "https://c.example", token="ENR") == 0
    saved = config.load()
    assert saved.server_url == "https://c.example" and saved.enrollment_token == "ENR"
    assert seen["cfg"].server_url == "https://c.example"       # the flow ran against the saved config
    assert "enrolled with https://c.example" in con[1].getvalue()


def test_enroll_json_carries_endpoint(con, capsys, monkeypatch):
    monkeypatch.setattr(enroll, "enroll_flow", lambda cfg: None)
    assert _node(con, "enroll", "https://c.example", token="ENR", as_json=True) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["endpoint"] == "https://c.example"


def test_enroll_without_server_url_is_usage_error(con):
    assert _node(con, "enroll", token="ENR") == 1
    assert "usage: ara node enroll" in con[1].getvalue()


def test_enroll_without_token_is_usage_error(con):
    assert _node(con, "enroll", "https://c.example") == 1        # no --token
    assert "usage: ara node enroll" in con[1].getvalue()


def test_enroll_failure_is_surfaced(con, monkeypatch):
    def _boom(cfg):
        raise RuntimeError("coordinator refused")
    monkeypatch.setattr(enroll, "enroll_flow", _boom)
    assert _node(con, "enroll", "https://c.example", token="ENR") == 1
    assert "enrollment failed: coordinator refused" in con[1].getvalue()


# --- run (work loop) ---
def test_run_requires_enrollment(con):
    assert _node(con, "run") == 1                               # no config on disk yet
    assert "not enrolled" in con[1].getvalue()


def test_run_invokes_the_agent_loop(con, monkeypatch):
    config.save(config.NodeConfig(server_url="https://c.example", session_token="SES"))
    seen = {}
    monkeypatch.setattr(agent, "run_loop", lambda cfg: seen.setdefault("cfg", cfg))
    assert _node(con, "run") == 0
    assert seen["cfg"].session_token == "SES"
    assert "run loop exited" in con[1].getvalue()


# --- install / status / lifecycle ---
def test_install_reports_success(con, monkeypatch):
    called = []
    monkeypatch.setattr(service, "install", lambda: called.append(True))
    assert _node(con, "install") == 0
    assert called == [True]                        # install takes no host/port — push-only
    assert "installed" in con[1].getvalue()


def test_install_json_reports_ok(con, capsys, monkeypatch):
    monkeypatch.setattr(service, "install", lambda: None)
    assert _node(con, "install", as_json=True) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and "installed" in out["message"]


def test_status_text(con, monkeypatch):
    monkeypatch.setattr(service, "status", lambda: "Active: running\n")
    assert _node(con, "status") == 0
    assert "Active: running" in con[1].getvalue()


def test_status_json(con, capsys, monkeypatch):
    monkeypatch.setattr(service, "status", lambda: "Active: running\n")
    assert _node(con, "status", as_json=True) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "Active: running\n"}


@pytest.mark.parametrize("sub", ["start", "stop", "uninstall"])
def test_lifecycle_commands(con, monkeypatch, sub):
    called = []
    monkeypatch.setattr(service, sub, lambda: called.append(sub))
    assert _node(con, sub) == 0
    assert called == [sub] and f"{sub} ok" in con[1].getvalue()


def test_non_linux_runtime_error_is_clean(con, monkeypatch):
    def _nope():
        raise RuntimeError("systemd is Linux-only")
    monkeypatch.setattr(service, "install", _nope)
    assert _node(con, "install") == 1
    assert "Linux-only" in con[1].getvalue()


@pytest.mark.parametrize("sub,exc", [
    ("install", PermissionError("mkdir denied")),
    ("install", OSError("write failed")),
    ("install", ValueError("unit rendering failed")),
    ("uninstall", OSError("unlink failed")),
])
@pytest.mark.parametrize("as_json", [False, True])
def test_service_filesystem_and_rendering_errors_are_honest(
        con, capsys, monkeypatch, sub, exc, as_json):
    monkeypatch.setattr(
        service, sub, lambda: (_ for _ in ()).throw(exc),
    )
    assert _node(con, sub, as_json=as_json) == 1
    if as_json:
        assert str(exc) in json.loads(capsys.readouterr().out)["error"]
    else:
        assert str(exc) in con[1].getvalue()


@pytest.mark.parametrize("raised", [KeyboardInterrupt, SystemExit])
def test_service_base_exceptions_still_propagate(con, monkeypatch, raised):
    monkeypatch.setattr(
        service, "install", lambda: (_ for _ in ()).throw(raised("stop")),
    )
    with pytest.raises(raised, match="stop"):
        _node(con, "install")


@pytest.mark.parametrize("sub", ["start", "stop"])
def test_service_failure_is_actionable_text(con, monkeypatch, sub):
    monkeypatch.setattr(
        service, sub,
        lambda: (_ for _ in ()).throw(RuntimeError("systemctl: daemon message")),
    )
    assert _node(con, sub) == 1
    assert "systemctl: daemon message" in con[1].getvalue()


@pytest.mark.parametrize("sub", ["start", "stop"])
def test_service_failure_is_actionable_json(con, capsys, monkeypatch, sub):
    monkeypatch.setattr(
        service, sub,
        lambda: (_ for _ in ()).throw(RuntimeError("systemctl: daemon message")),
    )
    assert _node(con, sub, as_json=True) == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": "systemctl: daemon message",
    }


# --- usage ---
def test_no_subcommand_is_usage_error(con):
    assert _node(con) == 1
    assert "usage: ara node" in con[1].getvalue()


def test_unknown_subcommand_json_usage_error(con, capsys):
    assert _node(con, "bogus", as_json=True) == 1
    assert "usage: ara node" in json.loads(capsys.readouterr().out)["error"]
