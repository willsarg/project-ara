# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Node liveness via systemd's ``sd_notify`` protocol — a no-op off systemd.

The push-only node has no inbound endpoint to probe, so liveness is a heartbeat the node *emits*.
On a ``Type=notify`` systemd unit that heartbeat is ``sd_notify(WATCHDOG=1)`` on the
``$NOTIFY_SOCKET`` datagram socket; ``READY=1`` announces startup and ``STATUS=…`` publishes a
human-readable state line. Off systemd (``NOTIFY_SOCKET`` unset) every notify is a deliberate no-op
returning False, so the agent loop can call these unconditionally without a platform branch —
:func:`status` still prints a structured line so journald/syslog pick it up either way.
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
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.sendto(state.encode("utf-8"), addr)
    finally:
        sock.close()
    return True


def ready() -> bool:
    """Announce startup completion to systemd (``READY=1``) — required by ``Type=notify``. No-op
    off systemd."""
    return sd_notify("READY=1")


def heartbeat() -> bool:
    """Pet systemd's watchdog (``WATCHDOG=1``). No-op (returns False) off systemd."""
    return sd_notify("WATCHDOG=1")


def status(msg: str) -> bool:
    """Publish a human-readable status line: ``STATUS=<msg>`` to systemd (shown as the unit's
    Status), and always a structured line to stdout so journald/syslog capture it off systemd.
    Returns whether the systemd notify fired."""
    print(f"ara-node status: {msg}", flush=True)
    return sd_notify(f"STATUS={msg}")
