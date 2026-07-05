# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Enrollment: introduce this node to the coordinator and wait for an admin to approve it.

The node POSTs its self-description (authed by the one-shot enrollment token) and lands PENDING;
an admin approves it in the dashboard, at which point the poll returns ``active`` with a durable
``session_token``. :func:`enroll_flow` drives that handshake, stores the session token into the
config, and saves. The poll is bounded (``max_polls``) with an injectable ``sleep`` so tests never
actually wait — and so a never-approved node fails loudly instead of spinning forever.
"""
from __future__ import annotations

import time

from ara.node import capabilities, config as config_mod
from ara.node.client import NodeClient


def enroll_flow(config: config_mod.NodeConfig, *, client: NodeClient | None = None,
                sleep=time.sleep, poll_interval: float = 2.0,
                max_polls: int = 30) -> config_mod.NodeConfig:
    """Enroll *config*'s node and block until approved, then persist its session token.

    Raises ``TimeoutError`` if approval doesn't arrive within ``max_polls`` polls, or ``ValueError``
    up front if the coordinator URL isn't secure (never send a token over cleartext http)."""
    config_mod.require_secure_url(config.server_url)
    client = client or NodeClient(config.server_url, config.enrollment_token)
    response = client.enroll(capabilities.self_description())
    enrollment_id = response["enrollment_id"]
    for _ in range(max_polls):
        poll = client.poll_approval(enrollment_id)
        if poll.get("status") == "active":
            config.session_token = poll["session_token"]
            config_mod.save(config)
            return config
        sleep(poll_interval)
    raise TimeoutError("enrollment was not approved within the polling window")
