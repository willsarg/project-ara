# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Node liveness — a STUB that no-ops off systemd.

The push-only node has no inbound endpoint to probe, so liveness is a heartbeat the node emits.
On a systemd install that becomes an ``sd_notify(WATCHDOG=1)`` (a later phase); here — and anywhere
off systemd — :func:`heartbeat` is a deliberate no-op and :func:`status` is a trivial snapshot, so
the agent loop can call it unconditionally without a platform branch.
"""
from __future__ import annotations


def heartbeat() -> None:
    """Emit a liveness beat. No-op off systemd (sd_notify lands in a later phase)."""
    return None


def status() -> dict:
    """A trivial liveness snapshot — enough for a test/log to see the node is alive."""
    return {"service": "ara-node", "status": "ok"}
