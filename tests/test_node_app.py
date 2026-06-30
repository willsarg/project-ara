# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node FastAPI app — auth gate, read endpoints, and the job submit/poll contract."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from ara import db
from ara.node import app as node_app
from ara.node import auth, jobs


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_NODE_DIR", str(tmp_path / "node"))
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "jobs.db"))
    token = auth.ensure_token()
    runner = jobs.JobRunner({"run": lambda a: {"completion": a.get("prompt", "")}},
                            spawn=lambda fn: fn())            # synchronous → jobs finish instantly
    providers = {
        "status": lambda: {"running": []},
        "detect": lambda: {"cpu": "fake"},
        "profile": lambda: {"verdict": "ok"},
        "models": lambda: {"models": []},
    }
    application = node_app.create_app(runner, providers, version="9.9.9")
    c = TestClient(application)
    c.headers.update({"Authorization": f"Bearer {token}"})    # default to authed
    return c


def test_health_is_open_and_reports_version(client):
    r = client.get("/health", headers={"Authorization": ""})   # no auth needed for liveness
    assert r.status_code == 200
    assert r.json() == {"service": "ara-node", "status": "ok", "version": "9.9.9"}


@pytest.mark.parametrize("path", ["/status", "/detect", "/profile", "/models", "/jobs"])
def test_protected_endpoints_401_without_token(client, path):
    r = client.get(path, headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_read_endpoints_return_provider_data(client):
    assert client.get("/status").json() == {"running": []}
    assert client.get("/detect").json() == {"cpu": "fake"}
    assert client.get("/profile").json() == {"verdict": "ok"}
    assert client.get("/models").json() == {"models": []}


def test_submit_job_returns_202_and_runs_it(client):
    r = client.post("/jobs", json={"kind": "run", "args": {"prompt": "hi"}})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    job = client.get(f"/jobs/{jid}").json()
    assert job["status"] == "done"
    assert job["result"] == {"completion": "hi"}              # result_json decoded for the client


def test_submit_unknown_kind_is_400(client):
    r = client.post("/jobs", json={"kind": "nope", "args": {}})
    assert r.status_code == 400


def test_submit_requires_auth(client):
    r = client.post("/jobs", json={"kind": "run", "args": {}}, headers={"Authorization": ""})
    assert r.status_code == 401


def test_get_missing_job_is_404(client):
    assert client.get("/jobs/ghost").status_code == 404


def test_list_jobs(client):
    client.post("/jobs", json={"kind": "run", "args": {"prompt": "a"}})
    rows = client.get("/jobs").json()["jobs"]
    assert len(rows) == 1 and rows[0]["kind"] == "run"


def test_long_poll_returns_immediately_when_done(client):
    jid = client.post("/jobs", json={"kind": "run", "args": {}}).json()["job_id"]
    t0 = time.monotonic()
    r = client.get(f"/jobs/{jid}?wait=5")                     # already done → must not wait 5s
    assert r.json()["status"] == "done" and (time.monotonic() - t0) < 2


def test_long_poll_times_out_on_a_running_job(client):
    con = db.connect()                                        # a job wedged in 'running'
    db.create_job(con, "stuck", "run", "{}")
    db.update_job(con, "stuck", status="running")
    r = client.get("/jobs/stuck?wait=0.2")
    assert r.json()["status"] == "running"                    # returns current state after timeout
