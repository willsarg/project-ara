# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Optional node liveness signals via systemd's ``sd_notify`` protocol.

ARA's installed user unit is deliberately ``Type=simple`` with no watchdog: legitimate model jobs
can block the polling loop for hours. If a supervising environment independently supplies
``NOTIFY_SOCKET``, these helpers can still emit readiness, heartbeat, and status datagrams. Without
that socket every notify is a deliberate no-op; :func:`status` still prints a structured line so
journald or another log collector can capture it.
"""
from __future__ import annotations

import os
import socket


def sd_notify(state: str) -> bool:
    """Send *state* (e.g. ``"READY=1"``, ``"WATCHDOG=1"``, ``"STATUS=…"``) to systemd's notify
    socket. Returns True if it was sent, False (no-op) when ``NOTIFY_SOCKET`` is unset (off systemd).

    Handles the abstract-namespace socket form: a leading ``@`` maps to a NUL byte."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return False
    family = getattr(socket, "AF_UNIX", None)
    if family is None:
        # No Unix-domain sockets on this host (e.g. Windows) — sd_notify is a Linux/systemd
        # mechanism, so degrade to a no-op rather than crash. NOTIFY_SOCKET is normally unset
        # off systemd anyway; this covers the case where it is set but AF_UNIX doesn't exist.
        return False
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    sock = socket.socket(family, socket.SOCK_DGRAM)
    try:
        sock.sendto(state.encode("utf-8"), addr)
    finally:
        sock.close()
    return True


def ready() -> bool:
    """Optionally announce startup completion (``READY=1``); no-op without ``NOTIFY_SOCKET``."""
    return sd_notify("READY=1")


def heartbeat() -> bool:
    """Optionally emit ``WATCHDOG=1``; ARA's installed unit does not configure a watchdog."""
    return sd_notify("WATCHDOG=1")


def status(msg: str) -> bool:
    """Publish a human-readable status line: ``STATUS=<msg>`` to systemd (shown as the unit's
    Status), and always a structured line to stdout so journald/syslog capture it off systemd.
    Returns whether the systemd notify fired."""
    print(f"ara-node status: {msg}", flush=True)
    return sd_notify(f"STATUS={msg}")
