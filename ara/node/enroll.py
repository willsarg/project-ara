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
    pending = config_mod.load_pending()
    if (pending is None or pending.server_url != config.server_url
            or pending.enrollment_token != config.enrollment_token):
        if not isinstance(config.enrollment_token, str) or not config.enrollment_token:
            raise ValueError("a one-time enrollment token is required")
        pending = config_mod.PendingEnrollment(config.server_url, config.enrollment_token)
        config_mod.save_pending(pending)  # durable before the one-shot token crosses the network
    enrollment_id = pending.enrollment_id
    if enrollment_id is None:
        response = client.enroll(capabilities.self_description())
        enrollment_id = response.get("enrollment_id") if isinstance(response, dict) else None
        if not isinstance(enrollment_id, str) or not enrollment_id:
            raise ValueError("invalid enrollment response from coordinator")
        pending.enrollment_id = enrollment_id
        config_mod.save_pending(pending)
    for _ in range(max_polls):
        poll = client.poll_approval(enrollment_id)
        if isinstance(poll, dict) and poll.get("status") == "active":
            session_token = poll.get("session_token")
            if not isinstance(session_token, str) or not session_token:
                raise ValueError("invalid approval response from coordinator")
            config.session_token = session_token
            config.enrollment_token = None
            config_mod.save(config)
            config_mod.clear_pending()
            return config
        sleep(poll_interval)
    raise TimeoutError("enrollment was not approved within the polling window")
