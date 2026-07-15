# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Enrollment handshake — enroll, poll to approval, persist the session token, bounded waiting."""
from __future__ import annotations

import sys
import types

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


def test_enroll_flow_refuses_insecure_url_before_sending_token():
    """A bearer token would cross the network in cleartext over http:// — enroll must fail closed
    BEFORE the first request, never handing the enrollment token to an insecure coordinator."""
    fake = FakeClient([])
    cfg = config.NodeConfig(server_url="http://coordinator.example", enrollment_token="ENR")
    with pytest.raises(ValueError):
        enroll.enroll_flow(cfg, client=fake)
    assert fake.enrolled_with is None            # token was never sent


def test_enroll_flow_requires_a_real_one_shot_token():
    with pytest.raises(ValueError, match="one-time enrollment token"):
        enroll.enroll_flow(
            config.NodeConfig(server_url="https://c.example"),
            client=FakeClient([]),
        )


def test_concurrent_enrollment_cannot_start_a_stale_handshake():
    with enroll._enrollment_lease():
        with pytest.raises(enroll.EnrollmentBusy, match="already owns"):
            enroll.enroll_flow(
                _cfg(), client=FakeClient([{"status": "active", "session_token": "STALE"}]),
            )
    assert config.load() is None


def test_enrollment_lease_uses_windows_locking_protocol(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(
        LK_NBLCK=1, LK_UNLCK=2,
        locking=lambda fd, operation, size: calls.append((fd, operation, size)),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(enroll, "_is_windows", lambda: True)
    with enroll._enrollment_lease():
        calls.append("inside")
    assert calls[0][1:] == (fake.LK_NBLCK, 1)
    assert calls[1] == "inside"
    assert calls[2][1:] == (fake.LK_UNLCK, 1)


def test_enrollment_lease_classifies_windows_contention(monkeypatch):
    fake = types.SimpleNamespace(
        LK_NBLCK=1, LK_UNLCK=2,
        locking=lambda _fd, operation, _size: (
            (_ for _ in ()).throw(OSError("busy")) if operation == 1 else None),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(enroll, "_is_windows", lambda: True)
    with pytest.raises(enroll.EnrollmentBusy):
        with enroll._enrollment_lease():
            pytest.fail("busy lease must not enter")


def test_enrollment_lease_uses_posix_locking_and_optional_permissions(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(
        LOCK_EX=1, LOCK_NB=2, LOCK_UN=4,
        flock=lambda fd, operation: calls.append((fd, operation)),
    )
    monkeypatch.setitem(sys.modules, "fcntl", fake)
    monkeypatch.setattr(enroll, "_is_windows", lambda: False)
    monkeypatch.setattr(enroll.os, "O_NOFOLLOW", 0, raising=False)
    monkeypatch.setattr(enroll.os, "fchmod", lambda fd, mode: calls.append((fd, mode)),
                        raising=False)
    with enroll._enrollment_lease():
        calls.append("inside")
    assert any(call == "inside" for call in calls)
    assert any(isinstance(call, tuple) and call[1] == fake.LOCK_EX | fake.LOCK_NB
               for call in calls)
    assert any(isinstance(call, tuple) and call[1] == fake.LOCK_UN for call in calls)
    assert any(isinstance(call, tuple) and call[1] == 0o600 for call in calls)


def test_enrollment_lease_classifies_posix_contention(monkeypatch):
    fake = types.SimpleNamespace(
        LOCK_EX=1, LOCK_NB=2, LOCK_UN=4,
        flock=lambda _fd, operation: (
            (_ for _ in ()).throw(OSError("busy"))
            if operation == 3 else None),
    )
    monkeypatch.setitem(sys.modules, "fcntl", fake)
    monkeypatch.setattr(enroll, "_is_windows", lambda: False)
    with pytest.raises(enroll.EnrollmentBusy):
        with enroll._enrollment_lease():
            pytest.fail("busy lease must not enter")


def test_enrollment_lease_supports_platform_without_optional_open_features(monkeypatch):
    monkeypatch.delattr(enroll.os, "O_NOFOLLOW", raising=False)
    monkeypatch.delattr(enroll.os, "fchmod", raising=False)
    with enroll._enrollment_lease():
        pass


def test_active_immediately_stores_and_saves_session_token():
    slept = []
    fake = FakeClient([{"status": "active", "session_token": "SES"}])
    cfg = enroll.enroll_flow(_cfg(), client=fake, sleep=lambda s: slept.append(s))
    assert cfg.session_token == "SES"
    assert cfg.enrollment_token is None                    # consumed one-shot secret is discarded
    assert config.load().session_token == "SES"            # persisted to disk
    assert config.load().enrollment_token is None
    assert config.load_pending() is None
    assert fake.enrolled_with == {"machine_key": "m"}
    assert slept == []                                     # approved on the first poll → never slept


def test_waits_through_pending_then_approves():
    slept = []
    fake = FakeClient([{"status": "pending"}, {"status": "active", "session_token": "SES"}])
    enroll.enroll_flow(_cfg(), client=fake, sleep=lambda s: slept.append(s), poll_interval=1.5)
    assert fake.poll_count == 2 and slept == [1.5]         # slept once between the two polls


def test_waits_through_a_non_active_status_then_approves():
    slept = []
    # any status that isn't "active" (here a transient "provisioning") keeps polling, not crashing.
    fake = FakeClient([{"status": "provisioning"}, {"status": "active", "session_token": "SES"}])
    cfg = enroll.enroll_flow(_cfg(), client=fake, sleep=lambda s: slept.append(s), poll_interval=0.5)
    assert cfg.session_token == "SES" and fake.poll_count == 2 and slept == [0.5]


def test_times_out_when_never_approved():
    slept = []
    fake = FakeClient([{"status": "pending"}] * 5)
    with pytest.raises(TimeoutError):
        enroll.enroll_flow(_cfg(), client=fake, sleep=lambda s: slept.append(s), max_polls=3)
    assert fake.poll_count == 3 and len(slept) == 3
    assert config.load() is None                           # active config stays untouched
    pending = config.load_pending()
    assert pending.enrollment_token == "ENR" and pending.enrollment_id == "e1"


def test_resumes_durably_saved_pending_enrollment_without_reposting():
    config.save_pending(config.PendingEnrollment(
        server_url="https://c.example", enrollment_token="ENR", enrollment_id="e1"))
    fake = FakeClient([{"status": "active", "session_token": "SES"}])
    fake.enroll = lambda _desc: pytest.fail("must resume the saved enrollment handle")
    cfg = enroll.enroll_flow(_cfg(), client=fake, sleep=lambda _s: None)
    assert cfg.session_token == "SES" and config.load_pending() is None


def test_saves_one_shot_token_before_first_enrollment_request(monkeypatch):
    fake = FakeClient([])
    def fail_after_observing_pending(_desc):
        pending = config.load_pending()
        assert pending.enrollment_token == "ENR" and pending.enrollment_id is None
        raise RuntimeError("response lost")
    fake.enroll = fail_after_observing_pending
    with pytest.raises(RuntimeError, match="response lost"):
        enroll.enroll_flow(_cfg(), client=fake)


def test_save_failure_after_approval_keeps_resumable_pending_state(monkeypatch):
    fake = FakeClient([{"status": "active", "session_token": "SES"}])
    monkeypatch.setattr(config, "save",
                        lambda _cfg: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        enroll.enroll_flow(_cfg(), client=fake)
    assert config.load_pending().enrollment_id == "e1"


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


def test_rejects_malformed_enrollment_handle():
    fake = FakeClient([])
    fake.enroll = lambda _desc: {"status": "pending", "enrollment_id": ""}
    with pytest.raises(ValueError, match="invalid enrollment response"):
        enroll.enroll_flow(_cfg(), client=fake)


@pytest.mark.parametrize("poll", [
    {"status": "active"}, {"status": "active", "session_token": ""},
    {"status": "active", "session_token": 7},
])
def test_rejects_active_approval_without_real_session_token(poll):
    fake = FakeClient([poll])
    with pytest.raises(ValueError, match="invalid approval response"):
        enroll.enroll_flow(_cfg(), client=fake)
    assert config.load() is None
