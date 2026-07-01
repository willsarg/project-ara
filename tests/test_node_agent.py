# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node agent loop — dispatch a job to ARA's wiring, report done/failed, bounded iteration."""
from __future__ import annotations

import pytest

from ara.node import agent, capabilities, config


@pytest.fixture(autouse=True)
def _stub_env(monkeypatch):
    """environment() probes the host; pin it so results are schema-shaped and deterministic."""
    monkeypatch.setattr(capabilities, "environment",
                        lambda: {"platform": "linux", "accel": "cpu",
                                 "containerized": False, "wall_source": "physical"})


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
