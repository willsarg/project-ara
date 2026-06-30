# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""`ara server <sub>` CLI — serve, migrate, the systemd lifecycle, and error paths.

Mirrors tests/test_cli_node.py exactly. render_server is IN the coverage gate (it lives in
ara/cli.py); ara.server.service is omitted (a separate Django runtime), so every branch here is
exercised against a mocked service module — no real uvicorn/systemctl/django ever runs.
"""
from __future__ import annotations

import io
import json

import pytest

import ara.cli as cli
from ara.server import service


@pytest.fixture
def con():
    """A Console writing styled text to a StringIO (asserted directly); `print(json)` still goes to
    stdout (capsys). Mirrors test_cli_node.py's pattern."""
    buf = io.StringIO()
    return cli.Console(color=False, stream=buf), buf


def _server(cb, *args, host="127.0.0.1", **kw):
    c, _buf = cb
    return cli.render_server(c, ["server", *args], host=host, port=8474, **kw)


# --- serve (foreground) ---
def test_serve_banner_and_launch(con, monkeypatch):
    seen = {}
    monkeypatch.setattr(service, "serve", lambda h, p: seen.update(host=h, port=p))
    assert _server(con, "serve") == 0
    out = con[1].getvalue()
    assert "http://127.0.0.1:8474" in out and "localhost only" in out   # localhost by default
    assert seen == {"host": "127.0.0.1", "port": 8474}


def test_serve_json_skips_banner(con, monkeypatch):
    monkeypatch.setattr(service, "serve", lambda h, p: None)
    assert _server(con, "serve", as_json=True) == 0
    assert con[1].getvalue() == ""               # no banner under --json (loopback → no warning either)


def test_serve_warns_when_exposed(con, monkeypatch):
    monkeypatch.setattr(service, "serve", lambda h, p: None)
    assert _server(con, "serve", host="0.0.0.0") == 0
    assert "exposing the dashboard" in con[1].getvalue()    # the dashboard leak is loud on expose


def test_serve_missing_extra_is_actionable(con, monkeypatch):
    def _no_uvicorn(h, p):
        raise ImportError("no uvicorn")
    monkeypatch.setattr(service, "serve", _no_uvicorn)
    assert _server(con, "serve") == 1
    assert "project-ara[server]" in con[1].getvalue()


# --- migrate ---
def test_migrate_text(con, monkeypatch):
    called = []
    monkeypatch.setattr(service, "migrate", lambda: called.append(True))
    assert _server(con, "migrate") == 0
    assert called == [True] and "migrate ok" in con[1].getvalue()


def test_migrate_json(con, capsys, monkeypatch):
    monkeypatch.setattr(service, "migrate", lambda: None)
    assert _server(con, "migrate", as_json=True) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["message"] == "migrate ok"


def test_migrate_missing_extra_is_actionable(con, monkeypatch):
    def _no_django():
        raise ImportError("no django")
    monkeypatch.setattr(service, "migrate", _no_django)
    assert _server(con, "migrate") == 1
    assert "project-ara[server]" in con[1].getvalue()


# --- install / status / lifecycle ---
def test_install_reports_endpoint(con, monkeypatch):
    seen = {}
    monkeypatch.setattr(service, "install", lambda h, p: seen.update(host=h, port=p))
    assert _server(con, "install") == 0
    assert seen == {"host": "127.0.0.1", "port": 8474}
    assert "installed" in con[1].getvalue()


def test_install_warns_when_exposed(con, monkeypatch):
    monkeypatch.setattr(service, "install", lambda h, p: None)
    assert _server(con, "install", host="0.0.0.0") == 0
    assert "exposing the dashboard" in con[1].getvalue()


def test_install_json_carries_endpoint(con, capsys, monkeypatch):
    monkeypatch.setattr(service, "install", lambda h, p: None)
    assert _server(con, "install", as_json=True) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["endpoint"] == "http://127.0.0.1:8474"


def test_status_text(con, monkeypatch):
    monkeypatch.setattr(service, "status", lambda: "Active: running\n")
    assert _server(con, "status") == 0
    assert "Active: running" in con[1].getvalue()


def test_status_json(con, capsys, monkeypatch):
    monkeypatch.setattr(service, "status", lambda: "Active: running\n")
    assert _server(con, "status", as_json=True) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "Active: running\n"}


@pytest.mark.parametrize("sub", ["start", "stop", "uninstall"])
def test_lifecycle_commands(con, monkeypatch, sub):
    called = []
    monkeypatch.setattr(service, sub, lambda: called.append(sub))
    assert _server(con, sub) == 0
    assert called == [sub] and f"{sub} ok" in con[1].getvalue()


def test_non_linux_runtime_error_is_clean(con, monkeypatch):
    def _nope(h, p):
        raise RuntimeError("systemd is Linux-only")
    monkeypatch.setattr(service, "install", _nope)
    assert _server(con, "install") == 1
    assert "Linux-only" in con[1].getvalue()


# --- usage ---
def test_no_subcommand_is_usage_error(con):
    assert _server(con) == 1
    assert "usage: ara server" in con[1].getvalue()


def test_unknown_subcommand_json_usage_error(con, capsys):
    assert _server(con, "bogus", as_json=True) == 1
    assert "usage: ara server" in json.loads(capsys.readouterr().out)["error"]


# --- createadmin ---
def test_createadmin_default_username(con, monkeypatch):
    seen = {}
    monkeypatch.setattr(service, "create_admin",
                        lambda u: seen.update(u=u) or {"username": u, "password": "pw", "created": True})
    assert _server(con, "createadmin") == 0
    assert seen["u"] == "admin"                       # defaults to 'admin'
    out = con[1].getvalue()
    assert "admin" in out and "pw" in out             # prints the generated password once


def test_createadmin_explicit_username(con, monkeypatch):
    monkeypatch.setattr(service, "create_admin",
                        lambda u: {"username": u, "password": "pw", "created": True})
    assert _server(con, "createadmin", "alice") == 0
    assert "alice" in con[1].getvalue()


def test_createadmin_json(con, capsys, monkeypatch):
    monkeypatch.setattr(service, "create_admin",
                        lambda u: {"username": "admin", "password": "s3cret", "created": True})
    assert _server(con, "createadmin", as_json=True) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["username"] == "admin" and out["password"] == "s3cret"


def test_createadmin_missing_extra(con, monkeypatch):
    def _boom(u):
        raise ImportError("no django")
    monkeypatch.setattr(service, "create_admin", _boom)
    assert _server(con, "createadmin") == 1
    assert "project-ara[server]" in con[1].getvalue()


# --- addnode ---
def test_addnode_registers(con, monkeypatch):
    seen = {}
    monkeypatch.setattr(service, "add_node",
                        lambda n, u, t: seen.update(n=n, u=u, t=t) or {"name": n, "base_url": u})
    assert _server(con, "addnode", "rog", "http://192.168.1.50:8473", "TOK") == 0
    assert seen == {"n": "rog", "u": "http://192.168.1.50:8473", "t": "TOK"}
    assert "rog" in con[1].getvalue()


def test_addnode_json(con, capsys, monkeypatch):
    monkeypatch.setattr(service, "add_node", lambda n, u, t: {"name": n, "base_url": u})
    assert _server(con, "addnode", "rog", "http://x", "T", as_json=True) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["node"] == "rog" and out["url"] == "http://x"


def test_addnode_insufficient_args(con):
    assert _server(con, "addnode", "rog") == 1
    assert "usage: ara server addnode" in con[1].getvalue()


def test_addnode_missing_extra(con, monkeypatch):
    def _boom(n, u, t):
        raise ImportError("no django")
    monkeypatch.setattr(service, "add_node", _boom)
    assert _server(con, "addnode", "rog", "http://x", "T") == 1
    assert "project-ara[server]" in con[1].getvalue()
