# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Enrollment handshake — enroll, poll to approval, persist the session token, bounded waiting."""
from __future__ import annotations

import pytest

from ara.node import capabilities, config, enroll


@pytest.fixture(autouse=True)
def _node_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_NODE_DIR", str(tmp_path / "node"))
    # self-description probes the host; stub it so tests don't touch real detect/psutil.
    monkeypatch.setattr(capabilities, "self_description", lambda: {"machine_key": "m"})


class FakeClient:
    """An enroll/poll stub: returns a fixed enroll id, then walks a scripted poll sequence."""

    def __init__(self, polls):
        self._polls = list(polls)
        self.enrolled_with = None
        self.poll_count = 0

    def enroll(self, self_desc):
        self.enrolled_with = self_desc
        return {"enrollment_id": "e1", "status": "pending"}

    def poll_approval(self, enrollment_id):
        assert enrollment_id == "e1"
        self.poll_count += 1
        return self._polls.pop(0)


def _cfg():
    return config.NodeConfig(server_url="https://c.example", enrollment_token="ENR")


def test_active_immediately_stores_and_saves_session_token():
    slept = []
    fake = FakeClient([{"status": "active", "session_token": "SES"}])
    cfg = enroll.enroll_flow(_cfg(), client=fake, sleep=lambda s: slept.append(s))
    assert cfg.session_token == "SES"
    assert config.load().session_token == "SES"            # persisted to disk
    assert fake.enrolled_with == {"machine_key": "m"}
    assert slept == []                                     # approved on the first poll → never slept


def test_waits_through_pending_then_approves():
    slept = []
    fake = FakeClient([{"status": "pending"}, {"status": "active", "session_token": "SES"}])
    enroll.enroll_flow(_cfg(), client=fake, sleep=lambda s: slept.append(s), poll_interval=1.5)
    assert fake.poll_count == 2 and slept == [1.5]         # slept once between the two polls


def test_times_out_when_never_approved():
    slept = []
    fake = FakeClient([{"status": "pending"}] * 5)
    with pytest.raises(TimeoutError):
        enroll.enroll_flow(_cfg(), client=fake, sleep=lambda s: slept.append(s), max_polls=3)
    assert fake.poll_count == 3 and len(slept) == 3
    assert config.load() is None                           # nothing persisted on failure


def test_builds_a_default_client_when_none_injected(monkeypatch):
    fake = FakeClient([{"status": "active", "session_token": "SES"}])
    seen = {}

    def _factory(server_url, token):
        seen.update(server_url=server_url, token=token)
        return fake

    monkeypatch.setattr(enroll, "NodeClient", _factory)
    enroll.enroll_flow(_cfg())                             # no client kwarg → default NodeClient path
    assert seen == {"server_url": "https://c.example", "token": "ENR"}
    assert config.load().session_token == "SES"
