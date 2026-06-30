# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""`ara node <sub>` CLI — token, serve, the systemd lifecycle, and error paths."""
from __future__ import annotations

import io
import json

import pytest

import ara.cli as cli
from ara.node import auth, service


@pytest.fixture
def con():
    """A Console writing styled text to a StringIO (asserted directly); `print(json)` still goes to
    stdout (capsys). Mirrors test_cli.py's make_console pattern."""
    buf = io.StringIO()
    return cli.Console(color=False, stream=buf), buf


def _node(cb, *args, **kw):
    c, _buf = cb
    return cli.render_node(c, ["node", *args], host="0.0.0.0", port=8473, **kw)


# --- token ---
def test_token_prints_existing(con, monkeypatch):
    monkeypatch.setattr(auth, "ensure_token", lambda: "TOK")
    assert _node(con, "token") == 0
    assert "TOK" in con[1].getvalue()


def test_token_rotate_replaces(con, monkeypatch):
    monkeypatch.setattr(auth, "rotate_token", lambda: "NEW")
    assert _node(con, "token", "rotate") == 0
    assert "NEW" in con[1].getvalue()


def test_token_json(con, capsys, monkeypatch):
    monkeypatch.setattr(auth, "ensure_token", lambda: "TOK")
    assert _node(con, "token", as_json=True) == 0
    assert json.loads(capsys.readouterr().out) == {"token": "TOK"}


# --- serve (foreground) ---
def test_serve_banner_and_launch(con, monkeypatch):
    monkeypatch.setattr(auth, "ensure_token", lambda: "TOK")
    seen = {}
    monkeypatch.setattr(service, "serve", lambda h, p: seen.update(host=h, port=p))
    assert _node(con, "serve") == 0
    out = con[1].getvalue()
    assert "http://0.0.0.0:8473" in out and "TOK" in out
    assert seen == {"host": "0.0.0.0", "port": 8473}


def test_serve_json_skips_banner(con, capsys, monkeypatch):
    monkeypatch.setattr(auth, "ensure_token", lambda: "TOK")
    monkeypatch.setattr(service, "serve", lambda h, p: None)
    assert _node(con, "serve", as_json=True) == 0
    assert con[1].getvalue() == ""               # no banner under --json


def test_serve_missing_extra_is_actionable(con, monkeypatch):
    monkeypatch.setattr(auth, "ensure_token", lambda: "TOK")
    def _no_uvicorn(h, p):
        raise ImportError("no uvicorn")
    monkeypatch.setattr(service, "serve", _no_uvicorn)
    assert _node(con, "serve") == 1
    assert "project-ara[node]" in con[1].getvalue()


# --- install / status / lifecycle ---
def test_install_reports_endpoint_and_token(con, monkeypatch):
    monkeypatch.setattr(auth, "ensure_token", lambda: "TOK")
    seen = {}
    monkeypatch.setattr(service, "install", lambda h, p: seen.update(host=h, port=p))
    assert _node(con, "install") == 0
    assert seen == {"host": "0.0.0.0", "port": 8473}
    assert "installed" in con[1].getvalue()


def test_install_json_carries_endpoint_and_token(con, capsys, monkeypatch):
    monkeypatch.setattr(auth, "ensure_token", lambda: "TOK")
    monkeypatch.setattr(service, "install", lambda h, p: None)
    assert _node(con, "install", as_json=True) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["endpoint"] == "http://0.0.0.0:8473" and out["token"] == "TOK"


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
    def _nope(h, p):
        raise RuntimeError("systemd is Linux-only")
    monkeypatch.setattr(auth, "ensure_token", lambda: "TOK")
    monkeypatch.setattr(service, "install", _nope)
    assert _node(con, "install") == 1
    assert "Linux-only" in con[1].getvalue()


# --- usage ---
def test_no_subcommand_is_usage_error(con):
    assert _node(con) == 1
    assert "usage: ara node" in con[1].getvalue()


def test_unknown_subcommand_json_usage_error(con, capsys):
    assert _node(con, "bogus", as_json=True) == 1
    assert "usage: ara node" in json.loads(capsys.readouterr().out)["error"]
