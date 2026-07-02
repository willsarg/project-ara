# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Node liveness via sd_notify — real datagram to $NOTIFY_SOCKET on systemd, no-op off it."""
from __future__ import annotations

import socket

import pytest

from ara.node import health


class FakeSock:
    """Records the datagram sent (payload, address) and whether it was closed."""

    def __init__(self):
        self.sent: tuple[bytes, str] | None = None
        self.closed = False

    def sendto(self, data, addr):
        self.sent = (data, addr)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_socket(monkeypatch):
    """Patch socket.socket to a recording FakeSock; assert it's built AF_UNIX/SOCK_DGRAM."""
    created = {}

    def _factory(family, type_):
        assert family == socket.AF_UNIX and type_ == socket.SOCK_DGRAM
        created["sock"] = FakeSock()
        return created["sock"]

    monkeypatch.setattr(health.socket, "socket", _factory)
    return created


# --- sd_notify ---
def test_sd_notify_noop_when_socket_unset(monkeypatch, fake_socket):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    assert health.sd_notify("READY=1") is False
    assert "sock" not in fake_socket                       # never even opened a socket


def test_sd_notify_sends_to_path_socket(monkeypatch, fake_socket):
    monkeypatch.setenv("NOTIFY_SOCKET", "/run/systemd/notify")
    assert health.sd_notify("WATCHDOG=1") is True
    sock = fake_socket["sock"]
    assert sock.sent == (b"WATCHDOG=1", "/run/systemd/notify")
    assert sock.closed is True                             # released even on the happy path


def test_sd_notify_maps_abstract_namespace_prefix(monkeypatch, fake_socket):
    monkeypatch.setenv("NOTIFY_SOCKET", "@/org/freedesktop/systemd")
    assert health.sd_notify("STATUS=busy") is True
    data, addr = fake_socket["sock"].sent
    assert addr == "\0/org/freedesktop/systemd"            # leading @ → NUL
    assert data == b"STATUS=busy"


def test_sd_notify_closes_socket_even_on_send_error(monkeypatch, fake_socket):
    monkeypatch.setenv("NOTIFY_SOCKET", "/run/systemd/notify")

    def _boom(self, data, addr):
        raise OSError("no route")

    monkeypatch.setattr(FakeSock, "sendto", _boom)
    with pytest.raises(OSError):
        health.sd_notify("READY=1")
    assert fake_socket["sock"].closed is True              # finally: still closed


# --- ready / heartbeat / status wrappers ---
def test_ready_sends_ready(monkeypatch):
    seen = []
    monkeypatch.setattr(health, "sd_notify", lambda s: seen.append(s) or True)
    assert health.ready() is True and seen == ["READY=1"]


def test_heartbeat_sends_watchdog(monkeypatch):
    seen = []
    monkeypatch.setattr(health, "sd_notify", lambda s: seen.append(s) or True)
    assert health.heartbeat() is True and seen == ["WATCHDOG=1"]


def test_heartbeat_is_false_off_systemd(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    assert health.heartbeat() is False                     # no socket → no-op


def test_status_prints_line_and_notifies(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(health, "sd_notify", lambda s: seen.append(s) or True)
    assert health.status("warming up") is True
    assert seen == ["STATUS=warming up"]
    assert "ara-node status: warming up" in capsys.readouterr().out    # journald/syslog off systemd
