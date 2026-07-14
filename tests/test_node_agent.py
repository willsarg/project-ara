# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node agent loop — dispatch a job to ARA's wiring, report done/failed, bounded iteration."""
from __future__ import annotations

import json

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

    def get_work(self, wait):
        return self._jobs.pop(0)

    def post_result(self, job_id, payload):
        self.posted.append((job_id, payload))


def _cfg(session_token="SES"):
    return config.NodeConfig(server_url="https://c.example", session_token=session_token)


# --- default_runner dispatch ---
def test_default_runner_routes_workers_providers_and_rejects_unknown(monkeypatch):
    monkeypatch.setattr(agent.wiring, "default_workers",
                        lambda: {"run": lambda args: {"ran": args}})
    monkeypatch.setattr(agent.wiring, "default_providers",
                        lambda: {"detect": lambda: {"detected": True}})
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


def test_result_payload_failed_retains_actionable_stderr():
    payload = agent._result_payload({"error": "boom", "stderr": "daemon detail"})
    assert payload["status"] == "failed"
    assert payload["error"] == "boom"
    assert payload["stderr"] == "daemon detail"


@pytest.mark.parametrize("stderr", ["", None, 7])
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
    result = agent.default_runner()("status", {})
    payload = agent._result_payload(result)
    assert payload["status"] == "failed"
    assert "exited 5" in payload["error"]
    assert payload["stderr"] == "actionable stderr"


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


def test_enrolls_first_when_no_session_token(monkeypatch):
    called = {}
    monkeypatch.setattr(agent.enroll, "enroll_flow",
                        lambda cfg: called.setdefault("cfg", cfg))
    fake = FakeClient([None])
    agent.run_loop(_cfg(session_token=None), client=fake, runner=lambda k, a: {},
                   max_iterations=1, sleep=lambda s: None)
    assert "cfg" in called                                  # missing session → enroll_flow ran first


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
    assert beats["heartbeat"] == 2 and len(beats["status"]) == 2   # watchdog + status every iteration


# --- 401 re-enrollment (session token revoked/expired) ---
class RaisingClient:
    """get_work / post_result raise a scripted error (or behave) so the 401 path is exercisable."""

    def __init__(self, *, get_work_error=None, job=None, post_error=None):
        self._get_work_error = get_work_error
        self._job = job
        self._post_error = post_error
        self.posted = []

    def get_work(self, wait):
        if self._get_work_error is not None:
            raise self._get_work_error
        return self._job

    def post_result(self, job_id, payload):
        if self._post_error is not None:
            raise self._post_error
        self.posted.append((job_id, payload))


def test_get_work_401_reenrolls_and_rebuilds_client(monkeypatch):
    reenrolled = []
    monkeypatch.setattr(agent.enroll, "enroll_flow", lambda cfg: reenrolled.append(cfg))
    fresh = FakeClient([None])
    built = {}
    monkeypatch.setattr(agent, "NodeClient",
                        lambda url, token: built.setdefault("args", (url, token)) or fresh)
    cfg = _cfg()
    client = RaisingClient(get_work_error=_status_error(401))
    n = agent.run_loop(cfg, client=client, runner=lambda k, a: {}, max_iterations=1,
                       sleep=lambda s: None)
    assert n == 1 and reenrolled == [cfg]                  # 401 → re-ran enroll_flow
    assert cfg.session_token is None                       # dropped the revoked token before re-enroll
    assert built["args"] == (cfg.server_url, cfg.session_token)   # rebuilt the client fresh


def test_get_work_non_401_error_propagates():
    client = RaisingClient(get_work_error=_status_error(500))
    with pytest.raises(httpx.HTTPStatusError):
        agent.run_loop(_cfg(), client=client, runner=lambda k, a: {}, max_iterations=1,
                       sleep=lambda s: None)


def test_post_result_401_reenrolls_and_delivers(monkeypatch, tmp_path):
    """A 401 on report means the token was revoked — re-enroll and DELIVER the result to the fresh
    client. It must never be silently dropped (the old bug: `continue` discarded finished work)."""
    reenrolled = []
    monkeypatch.setattr(agent.enroll, "enroll_flow", lambda cfg: reenrolled.append(cfg))
    fresh = FakeClient([None])
    monkeypatch.setattr(agent, "NodeClient", lambda url, token: fresh)
    client = RaisingClient(job={"id": "j1", "kind": "run", "args": {}},
                           post_error=_status_error(401))
    n = agent.run_loop(_cfg(), client=client, runner=lambda k, a: {"ok": 1}, max_iterations=1,
                       sleep=lambda s: None)
    assert n == 1 and reenrolled                            # re-enrolled after the 401
    assert fresh.posted and fresh.posted[0][0] == "j1"      # result delivered, NOT discarded
    assert _spool_files(tmp_path) == []                     # spool cleaned up on success


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
    assert len(spooled) == 1 and spooled[0].stem == "j1"   # finished work kept on disk
    assert 5.0 in slept                                     # bounded backoff after the failure


def test_reauth_returns_none_when_enrollment_token_already_consumed(monkeypatch):
    """The enrollment token is single-use — re-running enroll_flow after the coordinator burned it
    (the exact situation on a session revoke) raises. `_reauth` must ABSORB that and return None so
    callers keep work spooled + back off, instead of letting it crash the agent — Fix #1."""
    def _raise(cfg):
        raise _status_error(401)
    monkeypatch.setattr(agent.enroll, "enroll_flow", _raise)
    assert agent._reauth(_cfg()) is None


def test_get_work_401_survives_failed_reauth(monkeypatch):
    """Revoked session + already-consumed enrollment token → enroll_flow raises. The loop must back
    off and keep running, never crash — otherwise systemd `Restart=on-failure` crash-loops it (the
    bug: `_reauth` propagated the enroll HTTPStatusError straight out of run_loop) — Fix #1."""
    def _raise(cfg):
        raise _status_error(401)
    monkeypatch.setattr(agent.enroll, "enroll_flow", _raise)
    slept = []
    client = RaisingClient(get_work_error=_status_error(401))
    n = agent.run_loop(_cfg(), client=client, runner=lambda k, a: {}, max_iterations=1,
                       sleep=lambda s: slept.append(s), reauth_backoff=5.0)
    assert n == 1                                          # survived, no exception
    assert 5.0 in slept                                    # bounded backoff, no busy-loop


def test_post_401_failed_reauth_keeps_spool_and_survives(tmp_path, monkeypatch):
    """Report 401 → reauth, but enroll_flow raises (consumed token). `_try_post` must report
    not-delivered so the finished result stays spooled for a later retry — never raise/lose it — Fix #1."""
    def _raise(cfg):
        raise _status_error(401)
    monkeypatch.setattr(agent.enroll, "enroll_flow", _raise)
    client = RaisingClient(job={"id": "j1", "kind": "run", "args": {}},
                           post_error=_status_error(401))
    n = agent.run_loop(_cfg(), client=client, runner=lambda k, a: {"ok": 1}, max_iterations=1,
                       sleep=lambda s: None)
    assert n == 1                                          # loop survived
    spooled = _spool_files(tmp_path)
    assert len(spooled) == 1 and spooled[0].stem == "j1"   # finished work kept, not lost


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


def test_post_401_then_reauth_client_also_fails_keeps_spool(tmp_path, monkeypatch):
    """Token refreshed on 401 but the server is still erroring — the retry fails too; the result
    must stay spooled (not lost, not crashed on)."""
    monkeypatch.setattr(agent.enroll, "enroll_flow", lambda cfg: None)
    monkeypatch.setattr(agent, "NodeClient",
                        lambda url, token: RaisingClient(post_error=_status_error(503)))
    client = RaisingClient(job={"id": "j1", "kind": "run", "args": {}},
                           post_error=_status_error(401))
    n = agent.run_loop(_cfg(), client=client, runner=lambda k, a: {"ok": 1}, max_iterations=1,
                       sleep=lambda s: None)
    assert n == 1 and len(_spool_files(tmp_path)) == 1      # kept for a later retry


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


def test_flush_skips_and_drops_corrupt_spool_file(tmp_path):
    """A corrupt spool file can't be delivered — it's dropped, not retried forever or crashed on."""
    d = tmp_path / "node" / "results"
    d.mkdir(parents=True)
    (d / "bad.json").write_text("{not json")
    client = FakeClient([None])
    agent.run_loop(_cfg(), client=client, runner=lambda k, a: {}, max_iterations=1,
                   sleep=lambda s: None)
    assert client.posted == [] and _spool_files(tmp_path) == []
