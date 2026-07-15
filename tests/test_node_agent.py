# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node agent loop — dispatch a job to ARA's wiring, report done/failed, bounded iteration."""
from __future__ import annotations

import json
import os
import sys
import types

import httpx
import pytest

from ara.node import agent, capabilities, config


@pytest.fixture(autouse=True)
def _stub_env(monkeypatch):
    """environment() probes the host and the loop emits sd_notify; pin both so results are
    schema-shaped, deterministic, and quiet (no real socket/stdout). Individual tests re-patch the
    health hooks when they want to assert the wiring."""
    monkeypatch.setattr(capabilities, "environment",
                        lambda: {"platform": "linux", "accel": "cpu",
                                 "containerized": False, "wall_source": "physical"})
    monkeypatch.setattr(agent.health, "ready", lambda: None)
    monkeypatch.setattr(agent.health, "heartbeat", lambda: None)
    monkeypatch.setattr(agent.health, "status", lambda msg: None)


@pytest.fixture(autouse=True)
def _isolate_node_dir(tmp_path, monkeypatch):
    """The durable result spool lives under node_dir(); point it at a throwaway dir for every test."""
    monkeypatch.setenv("ARA_NODE_DIR", str(tmp_path / "node"))


def _spool_files(tmp_path):
    d = tmp_path / "node" / "results"
    return sorted(d.glob("*.json")) if d.exists() else []


def _accepted_files(tmp_path):
    d = tmp_path / "node" / "accepted"
    return sorted(d.glob("*.json")) if d.exists() else []


def _status_error(code: int) -> httpx.HTTPStatusError:
    """A synthetic httpx.HTTPStatusError carrying *code* — mirrors what raise_for_status raises."""
    request = httpx.Request("GET", "https://c.example/api/work")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError(f"HTTP {code}", request=request, response=response)


class FakeClient:
    """Serves a scripted queue of get_work results and records posted results."""

    def __init__(self, jobs):
        self._jobs = list(jobs)
        self.posted = []
        self.acked = []

    def get_work(self, wait):
        return self._jobs.pop(0)

    def post_result(self, job_id, payload):
        self.posted.append((job_id, payload))

    def ack_work(self, job_id):
        self.acked.append(job_id)


def _cfg(session_token="SES"):
    return config.NodeConfig(server_url="https://c.example", session_token=session_token)


# --- default_runner dispatch ---
def test_default_runner_routes_workers_providers_and_rejects_unknown(monkeypatch):
    monkeypatch.setattr(agent.wiring, "default_workers", lambda: {
        "run": lambda args: {"ran": args},
        "serve": lambda args: pytest.fail("serve is not a wire-contract action"),
    })
    monkeypatch.setattr(agent.wiring, "default_providers", lambda: {
        "detect": lambda: {"detected": True},
        "profile": lambda: pytest.fail("profile is not a wire-contract action"),
    })
    run = agent.default_runner()
    assert run("run", {"x": 1}) == {"ran": {"x": 1}}       # action verb → worker
    assert run("detect", {}) == {"detected": True}         # read verb → provider
    with pytest.raises(ValueError):
        run("nope", {})


# --- result shaping ---
def test_result_payload_done_for_plain_result():
    assert agent._result_payload({"ok": 1})["status"] == "done"


def test_result_payload_failed_when_worker_returns_error():
    payload = agent._result_payload({"error": "boom"})
    assert payload["status"] == "failed" and payload["error"] == "boom"
    assert "result" not in payload


@pytest.mark.parametrize("error", ["", "   ", None])
def test_result_payload_normalizes_missing_failure_reason(error):
    payload = agent._result_payload({"error": error})
    assert payload["status"] == "failed"
    assert payload["error"] == "node worker failed without an error message"


def test_result_payload_failed_retains_actionable_stderr():
    payload = agent._result_payload({"error": "boom", "stderr": "daemon detail"})
    assert payload["status"] == "failed"
    assert payload["error"] == "boom\nstderr: daemon detail"
    assert "stderr" not in payload


@pytest.mark.parametrize("stderr", ["", "   ", None, 7])
def test_result_payload_ignores_non_actionable_stderr(stderr):
    payload = agent._result_payload({"error": "boom", "stderr": stderr})
    assert "stderr" not in payload


# --- run_loop ---
def test_one_iteration_runs_job_and_posts_done():
    fake = FakeClient([{"id": "j1", "kind": "run", "args": {"model": "m"}}])
    n = agent.run_loop(_cfg(), client=fake, runner=lambda k, a: {"ok": k}, max_iterations=1)
    assert n == 1
    job_id, payload = fake.posted[0]
    assert job_id == "j1" and payload["status"] == "done" and payload["result"] == {"ok": "run"}
    assert payload["environment"]["platform"] == "linux"
    assert fake.acked == ["j1"]


def test_second_run_loop_cannot_share_one_node_state_directory():
    with agent._agent_lease():
        with pytest.raises(agent.NodeAgentBusy, match="already owns"):
            agent.run_loop(
                _cfg(), client=FakeClient([None]), runner=lambda _k, _a: {}, max_iterations=1,
            )


def test_agent_lease_uses_windows_locking_protocol(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(
        LK_NBLCK=1, LK_UNLCK=2,
        locking=lambda fd, operation, size: calls.append((fd, operation, size)),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(agent, "_is_windows", lambda: True)
    with agent._agent_lease():
        calls.append("inside")
    assert calls[0][1:] == (fake.LK_NBLCK, 1)
    assert calls[1] == "inside"
    assert calls[2][1:] == (fake.LK_UNLCK, 1)


def test_agent_lease_classifies_windows_lock_contention(monkeypatch):
    fake = types.SimpleNamespace(
        LK_NBLCK=1, LK_UNLCK=2,
        locking=lambda _fd, operation, _size: (
            (_ for _ in ()).throw(OSError("busy")) if operation == 1 else None),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(agent, "_is_windows", lambda: True)
    with pytest.raises(agent.NodeAgentBusy):
        with agent._agent_lease():
            pytest.fail("busy lease must not enter")


def test_agent_lease_supports_platform_without_optional_open_features(monkeypatch):
    monkeypatch.delattr(agent.os, "O_NOFOLLOW")
    monkeypatch.delattr(agent.os, "fchmod")
    with agent._agent_lease():
        pass


def test_job_is_durable_and_acknowledged_before_runner(tmp_path):
    fake = FakeClient([{"id": "j1", "kind": "run", "args": {}}])
    def runner(_kind, _args):
        assert fake.acked == ["j1"]
        assert len(_accepted_files(tmp_path)) == 1
        return {"ok": True}
    agent.run_loop(_cfg(), client=fake, runner=runner, max_iterations=1)
    assert _accepted_files(tmp_path) == []


def test_accepted_job_replays_after_process_restart(tmp_path):
    job = {"id": "j1", "kind": "run", "args": {"model": "m"}}
    agent._journal_job(job)
    fake = FakeClient([])
    agent.run_loop(_cfg(), client=fake, runner=lambda k, a: {"recovered": a["model"]},
                   max_iterations=1)
    assert fake.acked == ["j1"]
    assert fake.posted[0][1]["result"] == {"recovered": "m"}
    assert _accepted_files(tmp_path) == []


def test_process_interrupt_after_ack_leaves_job_for_restart(tmp_path):
    job = {"id": "j1", "kind": "run", "args": {}}
    first = FakeClient([job])
    with pytest.raises(KeyboardInterrupt):
        agent.run_loop(_cfg(), client=first,
                       runner=lambda k, a: (_ for _ in ()).throw(KeyboardInterrupt()),
                       max_iterations=1)
    assert first.acked == ["j1"] and len(_accepted_files(tmp_path)) == 1

    recovered = FakeClient([])
    agent.run_loop(_cfg(), client=recovered, runner=lambda k, a: {"ok": True},
                   max_iterations=1)
    assert recovered.posted and _accepted_files(tmp_path) == []


def test_ack_transport_failure_keeps_job_without_executing(tmp_path):
    client = RaisingClient(job={"id": "j1", "kind": "run", "args": {}},
                           ack_error=httpx.ConnectError("connection refused"))
    slept = []
    agent.run_loop(_cfg(), client=client,
                   runner=lambda k, a: pytest.fail("unacknowledged work must not execute"),
                   max_iterations=1, sleep=slept.append, reauth_backoff=4.0)
    assert len(_accepted_files(tmp_path)) == 1 and slept == [4.0]


@pytest.mark.parametrize("status", [408, 429, 500])
def test_ack_retryable_http_failure_keeps_job_without_executing(tmp_path, status):
    client = RaisingClient(job={"id": "j1", "kind": "run", "args": {}},
                           ack_error=_status_error(status))
    agent.run_loop(_cfg(), client=client,
                   runner=lambda k, a: pytest.fail("unacknowledged work must not execute"),
                   max_iterations=1, sleep=lambda _s: None)
    assert len(_accepted_files(tmp_path)) == 1


def test_ack_401_invalidates_session_and_keeps_accepted_job(tmp_path):
    cfg = _cfg()
    client = RaisingClient(job={"id": "j1", "kind": "run", "args": {}},
                           ack_error=_status_error(401))
    assert agent.run_loop(cfg, client=client,
                          runner=lambda k, a: pytest.fail("must not execute"),
                          max_iterations=1) == 1
    assert cfg.session_token is None and len(_accepted_files(tmp_path)) == 1


def test_ack_permanent_rejection_quarantines_accepted_job(tmp_path):
    client = RaisingClient(job={"id": "j1", "kind": "run", "args": {}},
                           ack_error=_status_error(404))
    agent.run_loop(_cfg(), client=client,
                   runner=lambda k, a: pytest.fail("must not execute"), max_iterations=1)
    assert _accepted_files(tmp_path) == []
    assert len(list((tmp_path / "node" / "accepted" / "quarantine").iterdir())) == 1


@pytest.mark.parametrize("value", [
    [], {"version": 2, "job": {}},
    {"version": 1, "job": {"id": "", "kind": "run", "args": {}}},
    {"version": 1, "job": {"id": "j1", "kind": "serve", "args": {}}},
])
def test_invalid_accepted_job_is_quarantined(tmp_path, value):
    directory = tmp_path / "node" / "accepted"
    directory.mkdir(parents=True)
    (directory / "bad.json").write_text(json.dumps(value), encoding="utf-8")
    agent.run_loop(_cfg(), client=FakeClient([None]), runner=lambda k, a: {},
                   max_iterations=1, sleep=lambda _s: None)
    assert _accepted_files(tmp_path) == []
    assert len(list((directory / "quarantine").iterdir())) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics differ on Windows")
def test_accepted_job_symlink_is_quarantined(tmp_path):
    directory = tmp_path / "node" / "accepted"
    directory.mkdir(parents=True)
    target = tmp_path / "target.json"
    target.write_text("evidence", encoding="utf-8")
    (directory / "link.json").symlink_to(target)
    agent.run_loop(_cfg(), client=FakeClient([None]), runner=lambda k, a: {},
                   max_iterations=1, sleep=lambda _s: None)
    assert (directory / "quarantine" / "link.json").is_symlink()


def test_invalid_accepted_job_survives_quarantine_failure(tmp_path, monkeypatch):
    directory = tmp_path / "node" / "accepted"
    directory.mkdir(parents=True)
    bad = directory / "bad.json"
    bad.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(agent, "_quarantine_spool",
                        lambda _path: (_ for _ in ()).throw(OSError("rename denied")))
    later = {"id": "later", "kind": "run", "args": {}}
    client = FakeClient([later])
    agent.run_loop(
        _cfg(), client=client,
        runner=lambda _k, _a: pytest.fail("later work must not pass an unresolved journal"),
        max_iterations=1, sleep=lambda _s: None,
    )
    assert bad.exists()
    assert client.acked == [] and client._jobs == [later]


def test_completed_spool_suppresses_accepted_job_reexecution_when_post_retries(tmp_path):
    agent._journal_job({"id": "j1", "kind": "run", "args": {}})
    agent._spool_result("j1", {"status": "done", "result": {"ok": True}})
    client = RaisingClient(post_error=_status_error(503))
    agent.run_loop(_cfg(), client=client,
                   runner=lambda k, a: pytest.fail("durably completed work must not rerun"),
                   max_iterations=1, sleep=lambda _s: None)
    assert _accepted_files(tmp_path) == [] and len(_spool_files(tmp_path)) == 1


def test_accepted_recovery_skips_job_with_durable_completion(tmp_path):
    agent._journal_job({"id": "j1", "kind": "run", "args": {}})
    agent._spool_result("j1", {"status": "done", "result": {"ok": True}})

    assert agent._next_accepted_job() is None
    assert _accepted_files(tmp_path) == []


def test_worker_error_dict_is_reported_failed():
    fake = FakeClient([{"id": "j1", "kind": "run", "args": {}}])
    agent.run_loop(_cfg(), client=fake, runner=lambda k, a: {"error": "nope"}, max_iterations=1)
    assert fake.posted[0][1]["status"] == "failed"


@pytest.mark.parametrize("stdout", [
    "", "not json", json.dumps([]), json.dumps({}), json.dumps({"warning": "partial"}),
])
def test_real_wiring_nonzero_result_is_always_reported_failed(monkeypatch, stdout):
    class Proc:
        returncode = 5
        stderr = "actionable stderr\n"

        def __init__(self):
            self.stdout = stdout

    monkeypatch.setattr(agent.wiring.subprocess, "run", lambda *_a, **_k: Proc())
    result = agent.default_runner()("run", {"model": "m"})
    payload = agent._result_payload(result)
    assert payload["status"] == "failed"
    assert "exited 5" in payload["error"]
    assert payload["error"].endswith("stderr: actionable stderr")
    assert "stderr" not in payload


def test_runner_exception_is_reported_failed():
    fake = FakeClient([{"id": "j1", "kind": "run", "args": {}}])

    def boom(kind, args):
        raise RuntimeError("kaboom")

    agent.run_loop(_cfg(), client=fake, runner=boom, max_iterations=1)
    _, payload = fake.posted[0]
    assert payload["status"] == "failed" and "kaboom" in payload["error"]
    assert payload["environment"]["wall_source"] == "physical"


def test_no_work_sleeps_and_continues_without_posting():
    slept = []
    fake = FakeClient([None])
    n = agent.run_loop(_cfg(), client=fake, runner=lambda k, a: {}, max_iterations=1,
                       sleep=lambda s: slept.append(s), poll_gap=0.25)
    assert n == 1 and fake.posted == [] and slept == [0.25]


def test_run_loop_requires_a_fresh_explicit_enrollment_when_session_is_missing():
    fake = FakeClient([None])
    with pytest.raises(ValueError, match="re-enrollment required"):
        agent.run_loop(_cfg(session_token=None), client=fake, runner=lambda k, a: {},
                       max_iterations=1, sleep=lambda s: None)


def test_builds_default_client_and_runner_when_none_injected(monkeypatch):
    fake = FakeClient([None])
    seen = {}
    monkeypatch.setattr(agent, "NodeClient",
                        lambda url, token: seen.setdefault("client", fake) or fake)
    monkeypatch.setattr(agent, "default_runner", lambda: seen.setdefault("runner", True))
    agent.run_loop(_cfg(), max_iterations=1, sleep=lambda s: None)
    assert seen["client"] is fake and seen["runner"] is True


# --- liveness wiring ---
def test_heartbeat_and_status_fire_each_iteration_after_ready(monkeypatch):
    beats = {"ready": 0, "heartbeat": 0, "status": []}
    monkeypatch.setattr(agent.health, "ready", lambda: beats.__setitem__("ready", beats["ready"] + 1))
    monkeypatch.setattr(agent.health, "heartbeat",
                        lambda: beats.__setitem__("heartbeat", beats["heartbeat"] + 1))
    monkeypatch.setattr(agent.health, "status", lambda msg: beats["status"].append(msg))
    fake = FakeClient([None, None])
    agent.run_loop(_cfg(), client=fake, runner=lambda k, a: {}, max_iterations=2, sleep=lambda s: None)
    assert beats["ready"] == 1                              # READY=1 announced once, before the loop
    assert beats["heartbeat"] == 2 and len(beats["status"]) == 2  # optional signal + status


# --- revoked session / coordinator outage handling ---
class RaisingClient:
    """get_work / post_result raise a scripted error (or behave) so the 401 path is exercisable."""

    def __init__(self, *, get_work_error=None, job=None, post_error=None, ack_error=None):
        self._get_work_error = get_work_error
        self._job = job
        self._post_error = post_error
        self._ack_error = ack_error
        self.posted = []
        self.acked = []

    def get_work(self, wait):
        if self._get_work_error is not None:
            raise self._get_work_error
        return self._job

    def post_result(self, job_id, payload):
        if self._post_error is not None:
            raise self._post_error
        self.posted.append((job_id, payload))

    def ack_work(self, job_id):
        if self._ack_error is not None:
            raise self._ack_error
        self.acked.append(job_id)


def test_get_work_401_invalidates_session_and_stops_for_explicit_reenrollment(monkeypatch):
    statuses = []
    monkeypatch.setattr(agent.health, "status", statuses.append)
    cfg = _cfg()
    client = RaisingClient(get_work_error=_status_error(401))
    n = agent.run_loop(cfg, client=client, runner=lambda k, a: {}, max_iterations=1,
                       sleep=lambda s: None)
    assert n == 1 and cfg.session_token is None
    assert config.load().session_token is None
    assert any("re-enrollment required" in status for status in statuses)


def test_late_401_from_old_loop_does_not_clobber_newly_enrolled_session():
    old = _cfg(session_token="OLD")
    config.save(config.NodeConfig(server_url=old.server_url, session_token="NEW"))
    agent._invalidate_session(old)
    assert old.session_token is None
    assert config.load().session_token == "NEW"


@pytest.mark.parametrize("error", [
    _status_error(500), httpx.ConnectError("connection refused"),
    ValueError("invalid work response from coordinator"),
])
def test_get_work_transient_or_invalid_response_backs_off_without_crashing(error):
    client = RaisingClient(get_work_error=error)
    slept = []
    assert agent.run_loop(
        _cfg(), client=client, runner=lambda k, a: {}, max_iterations=1,
        sleep=slept.append, reauth_backoff=7.0,
    ) == 1
    assert slept == [7.0]


@pytest.mark.parametrize("status", [400, 403, 404, 409, 422])
def test_get_work_permanent_rejection_stops_instead_of_retrying(status, monkeypatch):
    statuses = []
    monkeypatch.setattr(agent.health, "status", statuses.append)
    with pytest.raises(agent.CoordinatorWorkRejected, match=f"HTTP {status}"):
        agent.run_loop(
            _cfg(), client=RaisingClient(get_work_error=_status_error(status)),
            runner=lambda _k, _a: {}, max_iterations=2, sleep=lambda _seconds: None,
        )
    assert statuses[-1].endswith(f"HTTP {status}")


def test_post_result_401_invalidates_session_and_keeps_spool(monkeypatch, tmp_path):
    client = RaisingClient(job={"id": "j1", "kind": "run", "args": {}},
                           post_error=_status_error(401))
    cfg = _cfg()
    n = agent.run_loop(cfg, client=client, runner=lambda k, a: {"ok": 1}, max_iterations=1,
                       sleep=lambda s: None)
    assert n == 1 and cfg.session_token is None
    assert len(_spool_files(tmp_path)) == 1                 # result retained for new enrollment


def test_post_result_non_401_error_spools_and_does_not_crash(tmp_path):
    """A non-401 report failure (5xx / network) must NOT crash the agent or lose the finished
    result — the result stays durably spooled for a later retry (the old bug: it raised)."""
    client = RaisingClient(job={"id": "j1", "kind": "run", "args": {}},
                           post_error=_status_error(503))
    slept = []
    n = agent.run_loop(_cfg(), client=client, runner=lambda k, a: {"ok": 1}, max_iterations=1,
                       sleep=lambda s: slept.append(s), reauth_backoff=5.0)
    assert n == 1                                           # loop survived, no exception
    spooled = _spool_files(tmp_path)
    assert len(spooled) == 1                                # finished work kept on disk
    envelope = json.loads(spooled[0].read_text(encoding="utf-8"))
    assert envelope["job_id"] == "j1" and envelope["payload"]["status"] == "done"
    assert 5.0 in slept                                     # bounded backoff after the failure


@pytest.mark.parametrize("status", [400, 403, 404, 409, 422])
def test_permanent_result_rejection_is_quarantined_without_retry_loop(tmp_path, status):
    client = RaisingClient(job={"id": "j1", "kind": "run", "args": {}},
                           post_error=_status_error(status))
    slept = []
    assert agent.run_loop(_cfg(), client=client, runner=lambda k, a: {"ok": 1},
                          max_iterations=1, sleep=slept.append) == 1
    assert _spool_files(tmp_path) == []
    quarantined = list((tmp_path / "node" / "results" / "quarantine").iterdir())
    assert len(quarantined) == 1
    assert json.loads(quarantined[0].read_text(encoding="utf-8"))["job_id"] == "j1"
    assert slept == []


@pytest.mark.parametrize("status", [408, 429])
def test_retryable_result_status_keeps_live_spool_and_backs_off(tmp_path, status):
    client = RaisingClient(job={"id": "j1", "kind": "run", "args": {}},
                           post_error=_status_error(status))
    slept = []
    agent.run_loop(_cfg(), client=client, runner=lambda k, a: {"ok": 1},
                   max_iterations=1, sleep=slept.append, reauth_backoff=3.0)
    assert len(_spool_files(tmp_path)) == 1 and slept == [3.0]


def test_finished_result_spool_is_cleaned_up_on_successful_post(tmp_path):
    client = FakeClient([{"id": "j1", "kind": "run", "args": {}}, None])
    agent.run_loop(_cfg(), client=client, runner=lambda k, a: {"ok": 1}, max_iterations=2,
                   sleep=lambda s: None)
    assert client.posted and client.posted[0][0] == "j1"
    assert _spool_files(tmp_path) == []                     # delivered → no leftover spool


def test_spooled_result_from_prior_crash_is_flushed(tmp_path):
    """A result left on disk by an earlier crash/failure is re-POSTed on the next loop, then removed
    — so finished work is delivered even across a restart (crash-safety)."""
    d = tmp_path / "node" / "results"
    d.mkdir(parents=True)
    (d / "old.json").write_text(json.dumps({"status": "done", "result": {"x": 1}}))
    client = FakeClient([None])                             # no new work this iteration
    agent.run_loop(_cfg(), client=client, runner=lambda k, a: {}, max_iterations=1,
                   sleep=lambda s: None)
    assert client.posted == [("old", {"status": "done", "result": {"x": 1}})]   # flushed
    assert _spool_files(tmp_path) == []                     # removed after delivery


def test_current_spool_envelope_from_prior_crash_is_flushed(tmp_path):
    payload = {"status": "done", "result": {"x": 1}}
    agent._spool_result("current-job", payload)
    client = FakeClient([None])
    agent.run_loop(_cfg(), client=client, runner=lambda k, a: {}, max_iterations=1,
                   sleep=lambda s: None)
    assert client.posted == [("current-job", payload)]
    assert _spool_files(tmp_path) == []


def test_post_network_error_spools_and_does_not_crash(tmp_path):
    """A transport error (not an HTTP status) on report must also spool + survive, not crash."""
    client = RaisingClient(job={"id": "j1", "kind": "run", "args": {}},
                           post_error=httpx.ConnectError("connection refused"))
    n = agent.run_loop(_cfg(), client=client, runner=lambda k, a: {"ok": 1}, max_iterations=1,
                       sleep=lambda s: None)
    assert n == 1 and len(_spool_files(tmp_path)) == 1


def test_flush_keeps_result_when_post_still_fails(tmp_path):
    """A spooled result whose re-POST still fails stays on disk for the next attempt (not dropped)."""
    d = tmp_path / "node" / "results"
    d.mkdir(parents=True)
    (d / "old.json").write_text(json.dumps({"status": "done", "result": {"x": 1}}))
    client = RaisingClient(post_error=_status_error(503))   # flush post fails; get_work → None
    agent.run_loop(_cfg(), client=client, runner=lambda k, a: {}, max_iterations=1,
                   sleep=lambda s: None)
    assert len(_spool_files(tmp_path)) == 1                 # still spooled


def test_undelivered_result_blocks_polling_and_execution_of_later_work(tmp_path):
    """A completed job must remain the node's only responsibility until its result is delivered.

    Otherwise a coordinator outage can let a later job execute and finish before the earlier
    result reaches the coordinator, breaking the end-to-end FIFO contract.
    """
    class ResultOutageClient(FakeClient):
        def post_result(self, job_id, payload):
            raise _status_error(503)

    jobs = [
        {"id": "j1", "kind": "run", "args": {}},
        {"id": "j2", "kind": "run", "args": {}},
    ]
    client = ResultOutageClient(jobs)
    executed = []

    agent.run_loop(
        _cfg(), client=client,
        runner=lambda kind, args: executed.append((kind, args)) or {"ok": True},
        max_iterations=2, sleep=lambda _seconds: None,
    )

    assert executed == [("run", {})]
    assert client.acked == ["j1"]
    assert client._jobs == [jobs[1]]
    assert len(_spool_files(tmp_path)) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics differ on Windows")
def test_unavailable_results_directory_blocks_ack_and_execution(tmp_path):
    agent._journal_job({"id": "j1", "kind": "run", "args": {}})
    results = tmp_path / "node" / "results"
    results.symlink_to(tmp_path / "missing-results")
    client = FakeClient([])

    agent.run_loop(
        _cfg(), client=client,
        runner=lambda _kind, _args: pytest.fail("work must not run without durable result storage"),
        max_iterations=2, sleep=lambda _seconds: None,
    )

    assert client.acked == []
    assert len(_accepted_files(tmp_path)) == 1


def test_failed_atomic_result_probe_blocks_ack_and_execution(tmp_path, monkeypatch):
    agent._journal_job({"id": "j1", "kind": "run", "args": {}})
    real_write = agent._write_json_atomic

    def fail_probe(path, value):
        if path.name.startswith(".write-probe-"):
            raise PermissionError("result directory is not writable")
        return real_write(path, value)

    monkeypatch.setattr(agent, "_write_json_atomic", fail_probe)
    client = FakeClient([])

    agent.run_loop(
        _cfg(), client=client,
        runner=lambda _kind, _args: pytest.fail("work must not run after a failed write probe"),
        max_iterations=1, sleep=lambda _seconds: None,
    )

    assert client.acked == []
    assert len(_accepted_files(tmp_path)) == 1


def test_flush_replays_results_in_persisted_write_order(tmp_path):
    """Opaque hash filenames and tied mtimes must not decide current-envelope replay order."""
    job_ids = ["first candidate", "second candidate"]
    by_filename = sorted(job_ids, key=lambda job_id: agent._spool_path(job_id).name)
    older, newer = reversed(by_filename)  # make filename order intentionally contradict age
    agent._spool_result(older, {"status": "done", "result": {"order": 1}})
    agent._spool_result(newer, {"status": "done", "result": {"order": 2}})
    os.utime(agent._spool_path(older), ns=(1_000_000_000, 1_000_000_000))
    os.utime(agent._spool_path(newer), ns=(1_000_000_000, 1_000_000_000))
    client = FakeClient([])

    agent._flush_spool(client, _cfg())

    assert [job_id for job_id, _payload in client.posted] == [older, newer]


def test_flush_preserves_tied_legacy_results_when_fifo_order_is_unknown(tmp_path, monkeypatch):
    d = tmp_path / "node" / "results"
    d.mkdir(parents=True)
    for job_id in ("legacy-a", "legacy-b"):
        path = d / f"{job_id}.json"
        path.write_text(json.dumps({"status": "done", "result": {"job": job_id}}))
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    statuses = []
    monkeypatch.setattr(agent.health, "status", statuses.append)
    client = FakeClient([])

    agent._flush_spool(client, _cfg())

    assert client.posted == []
    assert len(_spool_files(tmp_path)) == 2
    assert statuses == ["result spool order is ambiguous; preserving evidence and blocking work"]


def test_flush_orders_mixed_envelopes_by_completion_time_not_format(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "_last_spool_order", 0)
    monkeypatch.setattr(agent.time, "time_ns", lambda: 1_000_000_000)
    agent._spool_result("current-first", {"status": "done", "result": {"order": 1}})
    legacy = tmp_path / "node" / "results" / "legacy-second.json"
    legacy.write_text(json.dumps({"status": "done", "result": {"order": 2}}))
    os.utime(legacy, ns=(2_000_000_000, 2_000_000_000))
    client = FakeClient([])

    agent._flush_spool(client, _cfg())

    assert [job_id for job_id, _payload in client.posted] == [
        "current-first", "legacy-second",
    ]


def test_previous_envelope_version_remains_deliverable(tmp_path):
    payload = {"status": "done", "result": {"ok": True}}
    path = agent._spool_path("v1-job")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"version": 1, "job_id": "v1-job", "payload": payload}))
    client = FakeClient([])

    agent._flush_spool(client, _cfg())

    assert client.posted == [("v1-job", payload)]


@pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics differ on Windows")
def test_flush_quarantines_dangling_symlink_before_ordering(tmp_path):
    d = tmp_path / "node" / "results"
    d.mkdir(parents=True)
    (d / "dangling.json").symlink_to(tmp_path / "missing.json")
    client = FakeClient([])

    agent._flush_spool(client, _cfg())

    assert client.posted == []
    assert (d / "quarantine" / "dangling.json").is_symlink()


def test_flush_quarantines_permanently_rejected_result_and_continues(tmp_path):
    d = tmp_path / "node" / "results"
    d.mkdir(parents=True)
    (d / "old.json").write_text(json.dumps({"status": "done", "result": {"x": 1}}))
    client = RaisingClient(post_error=_status_error(404))
    agent.run_loop(_cfg(), client=client, runner=lambda k, a: {}, max_iterations=1,
                   sleep=lambda s: None)
    assert _spool_files(tmp_path) == []
    assert len(list((d / "quarantine").iterdir())) == 1


def test_flush_quarantines_malformed_accepted_path_before_posting(tmp_path):
    payload = {"status": "done", "result": {"ok": True}}
    agent._spool_result("j1", payload)
    accepted = agent._accepted_path("j1")
    accepted.mkdir(parents=True)
    client = FakeClient([])

    agent._flush_spool(client, _cfg())

    assert client.posted == [("j1", payload)]
    assert not accepted.exists()
    assert len(list((accepted.parent / "quarantine").iterdir())) == 1


def test_flush_posts_but_retains_spool_when_accepted_cleanup_is_impossible(
        tmp_path, monkeypatch):
    payload = {"status": "done", "result": {"ok": True}}
    agent._spool_result("j1", payload)
    accepted = agent._accepted_path("j1")
    accepted.mkdir(parents=True)
    monkeypatch.setattr(agent, "_quarantine_spool",
                        lambda _path: (_ for _ in ()).throw(OSError("rename denied")))
    statuses = []
    monkeypatch.setattr(agent.health, "status", statuses.append)
    client = FakeClient([])

    agent._flush_spool(client, _cfg())

    assert client.posted == [("j1", payload)]
    assert len(_spool_files(tmp_path)) == 1
    assert statuses and "could not retire completed job journal" in statuses[0]


def test_immediate_post_retains_spool_when_accepted_cleanup_is_impossible(
        tmp_path, monkeypatch):
    client = FakeClient([{"id": "j1", "kind": "run", "args": {}}])
    monkeypatch.setattr(agent, "_quarantine_spool",
                        lambda _path: (_ for _ in ()).throw(OSError("rename denied")))

    def runner(_kind, _args):
        accepted = agent._accepted_path("j1")
        accepted.unlink()
        accepted.mkdir()
        return {"ok": True}

    agent.run_loop(_cfg(), client=client, runner=runner, max_iterations=1)

    assert client.posted and client.posted[0][0] == "j1"
    assert len(_spool_files(tmp_path)) == 1


def test_permanent_flush_rejection_retains_spool_when_accepted_cleanup_is_impossible(
        tmp_path, monkeypatch):
    agent._spool_result("j1", {"status": "done", "result": {"ok": True}})
    agent._accepted_path("j1").mkdir(parents=True)
    monkeypatch.setattr(agent, "_quarantine_spool",
                        lambda _path: (_ for _ in ()).throw(OSError("rename denied")))

    agent._flush_spool(RaisingClient(post_error=_status_error(404)), _cfg())

    assert len(_spool_files(tmp_path)) == 1


def test_permanent_immediate_rejection_retains_spool_when_accepted_cleanup_is_impossible(
        tmp_path, monkeypatch):
    client = RaisingClient(job={"id": "j1", "kind": "run", "args": {}},
                           post_error=_status_error(404))
    monkeypatch.setattr(agent, "_quarantine_spool",
                        lambda _path: (_ for _ in ()).throw(OSError("rename denied")))

    def runner(_kind, _args):
        accepted = agent._accepted_path("j1")
        accepted.unlink()
        accepted.mkdir()
        return {"ok": True}

    agent.run_loop(_cfg(), client=client, runner=runner, max_iterations=1)

    assert len(_spool_files(tmp_path)) == 1


@pytest.mark.parametrize("phase", ["flush", "ack", "post"])
def test_terminal_rejection_preserves_live_evidence_when_quarantine_fails(
        tmp_path, monkeypatch, phase):
    if phase == "flush":
        d = tmp_path / "node" / "results"
        d.mkdir(parents=True)
        (d / "old.json").write_text(json.dumps({"status": "done"}), encoding="utf-8")
        client = RaisingClient(post_error=_status_error(404))
    elif phase == "ack":
        client = RaisingClient(job={"id": "j1", "kind": "run", "args": {}},
                               ack_error=_status_error(404))
    else:
        client = RaisingClient(job={"id": "j1", "kind": "run", "args": {}},
                               post_error=_status_error(404))
    monkeypatch.setattr(agent, "_quarantine_spool",
                        lambda _path: (_ for _ in ()).throw(OSError("rename denied")))
    agent.run_loop(_cfg(), client=client, runner=lambda k, a: {"ok": True},
                   max_iterations=1, sleep=lambda _s: None)
    assert _spool_files(tmp_path) or _accepted_files(tmp_path)


def test_flush_quarantines_and_preserves_corrupt_spool_file(tmp_path):
    """A corrupt spool file can't be delivered, but evidence must be preserved for inspection."""
    d = tmp_path / "node" / "results"
    d.mkdir(parents=True)
    (d / "bad.json").write_text("{not json")
    client = FakeClient([None])
    agent.run_loop(_cfg(), client=client, runner=lambda k, a: {}, max_iterations=1,
                   sleep=lambda s: None)
    assert client.posted == [] and _spool_files(tmp_path) == []
    quarantined = list((d / "quarantine").iterdir())
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "{not json"


@pytest.mark.parametrize("value", [
    [],
    {"status": "done"},
    {"version": 1, "job_id": 7, "payload": {}},
    {"version": 1, "job_id": "j1", "payload": []},
    {"version": 2, "order": 0, "job_id": "j1", "payload": {}},
])
def test_flush_quarantines_invalid_spool_shapes(tmp_path, value):
    d = tmp_path / "node" / "results"
    d.mkdir(parents=True)
    (d / "bad name.json").write_text(json.dumps(value), encoding="utf-8")
    agent.run_loop(_cfg(), client=FakeClient([None]), runner=lambda k, a: {},
                   max_iterations=1, sleep=lambda s: None)
    assert not list(d.glob("*.json"))
    assert len(list((d / "quarantine").iterdir())) == 1


def test_flush_quarantines_envelope_with_mismatched_filename(tmp_path):
    d = tmp_path / "node" / "results"
    d.mkdir(parents=True)
    (d / "wrong.json").write_text(json.dumps({
        "version": 1, "job_id": "j1", "payload": {"status": "done"},
    }), encoding="utf-8")
    agent.run_loop(_cfg(), client=FakeClient([None]), runner=lambda k, a: {},
                   max_iterations=1, sleep=lambda s: None)
    assert len(list((d / "quarantine").iterdir())) == 1


def test_flush_quarantine_collision_preserves_both_files(tmp_path):
    d = tmp_path / "node" / "results"
    quarantine = d / "quarantine"
    quarantine.mkdir(parents=True)
    (quarantine / "bad.json").write_text("older", encoding="utf-8")
    (d / "bad.json").write_text("{broken", encoding="utf-8")
    agent.run_loop(_cfg(), client=FakeClient([None]), runner=lambda k, a: {},
                   max_iterations=1, sleep=lambda s: None)
    assert sorted(p.name for p in quarantine.iterdir()) == ["bad.json", "bad.json.1"]


@pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics differ on Windows")
def test_flush_quarantines_symlink_without_chmodding_its_target(tmp_path):
    d = tmp_path / "node" / "results"
    d.mkdir(parents=True)
    target = tmp_path / "target.json"
    target.write_text("evidence", encoding="utf-8")
    mode = os.stat(target).st_mode & 0o777
    (d / "link.json").symlink_to(target)
    agent.run_loop(_cfg(), client=FakeClient([None]), runner=lambda k, a: {},
                   max_iterations=1, sleep=lambda s: None)
    quarantined = d / "quarantine" / "link.json"
    assert quarantined.is_symlink() and target.read_text(encoding="utf-8") == "evidence"
    assert os.stat(target).st_mode & 0o777 == mode


def test_flush_survives_quarantine_filesystem_failure(tmp_path, monkeypatch):
    d = tmp_path / "node" / "results"
    d.mkdir(parents=True)
    bad = d / "bad.json"
    bad.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(agent, "_quarantine_spool",
                        lambda _path: (_ for _ in ()).throw(OSError("rename denied")))
    assert agent.run_loop(_cfg(), client=FakeClient([None]), runner=lambda k, a: {},
                          max_iterations=1, sleep=lambda s: None) == 1
    assert bad.exists()


def test_flush_401_invalidates_session_and_stops_with_spool_intact(tmp_path):
    d = tmp_path / "node" / "results"
    d.mkdir(parents=True)
    old = d / "old.json"
    old.write_text(json.dumps({"status": "done"}), encoding="utf-8")
    cfg = _cfg()
    assert agent.run_loop(cfg, client=RaisingClient(post_error=_status_error(401)),
                          runner=lambda k, a: {}, max_iterations=1,
                          sleep=lambda s: None) == 1
    assert cfg.session_token is None and old.exists()


def test_spool_filename_is_deterministic_and_not_derived_from_job_id(tmp_path):
    job_id = "../../escape/SECRET job"
    payload = {"status": "done", "result": {"x": 1}}

    agent._spool_result(job_id, payload)

    files = _spool_files(tmp_path)
    assert files == [agent._spool_path(job_id)]
    assert files[0].parent == tmp_path / "node" / "results"
    assert "escape" not in files[0].name and "SECRET" not in files[0].name
    envelope = json.loads(files[0].read_text(encoding="utf-8"))
    assert isinstance(envelope.pop("order"), int)
    assert envelope == {"version": 2, "job_id": job_id, "payload": payload}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file mode is advisory on Windows")
def test_spool_file_is_owner_only(tmp_path):
    agent._spool_result("j1", {"status": "done"})
    assert os.stat(agent._spool_path("j1")).st_mode & 0o777 == 0o600


def test_spool_replace_failure_preserves_prior_result(tmp_path, monkeypatch):
    old = {"status": "done", "result": {"attempt": 1}}
    agent._spool_result("j1", old)

    monkeypatch.setattr(os, "replace",
                        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")))
    with pytest.raises(OSError, match="replace failed"):
        agent._spool_result("j1", {"status": "done", "result": {"attempt": 2}})

    envelope = json.loads(agent._spool_path("j1").read_text(encoding="utf-8"))
    assert envelope["payload"] == old


def test_spool_write_degrades_when_fchmod_is_unavailable(monkeypatch):
    monkeypatch.delattr(agent.os, "fchmod")
    agent._spool_result("j1", {"status": "done"})
    assert agent._spool_path("j1").exists()


def test_spool_parent_sync_is_a_noop_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(agent.os, "name", "nt")
    monkeypatch.setattr(agent.os, "open", lambda *_a: pytest.fail("must not open a directory"))
    assert agent._fsync_parent(tmp_path / "result.json") is None


@pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics differ on Windows")
def test_private_directory_rejects_symlink_to_existing_directory(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)

    with pytest.raises(OSError, match="must not be a symlink"):
        agent._ensure_private_directory(link)


def test_private_directory_rejects_non_directory_after_creation_check():
    fake_path = types.SimpleNamespace(
        mkdir=lambda **_kwargs: None,
        is_symlink=lambda: False,
        is_dir=lambda: False,
    )

    with pytest.raises(OSError, match="is not a directory"):
        agent._ensure_private_directory(fake_path)
