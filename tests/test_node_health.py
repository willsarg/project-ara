# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node health stub — a no-op heartbeat and a trivial status snapshot."""
from __future__ import annotations

from ara.node import health


def test_heartbeat_is_a_noop():
    assert health.heartbeat() is None


def test_status_reports_alive():
    assert health.status() == {"service": "ara-node", "status": "ok"}
