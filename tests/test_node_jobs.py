# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node JobRunner — submit a kind, run it in the background, persist the outcome."""
from __future__ import annotations

import json
import time

import pytest

from ara import db
from ara.node import jobs


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "jobs.db"))


def _con():
    return db.connect()


def test_submit_runs_worker_and_records_done():
    seen = {}
    runner = jobs.JobRunner({"echo": lambda args: {"echoed": args["x"]}},
                            spawn=lambda fn: fn())                  # synchronous for determinism
    jid = runner.submit("echo", {"x": 42})
    job = db.get_job(_con(), jid)
    assert job["kind"] == "echo" and job["status"] == "done"
    assert json.loads(job["result_json"]) == {"echoed": 42}
    assert job["started_at"] and job["finished_at"]                # both stamped
    assert job["error"] is None


def test_submit_unknown_kind_raises_and_creates_no_job():
    runner = jobs.JobRunner({"echo": lambda a: a}, spawn=lambda fn: fn())
    with pytest.raises(ValueError):
        runner.submit("nope", {})
    assert db.list_jobs(_con()) == []


def test_worker_exception_marks_job_failed():
    def boom(args):
        raise RuntimeError("kaboom")

    runner = jobs.JobRunner({"characterize": boom}, spawn=lambda fn: fn())
    jid = runner.submit("characterize", {"model": "m"})
    job = db.get_job(_con(), jid)
    assert job["status"] == "failed"
    assert "kaboom" in job["error"]
    assert job["result_json"] is None and job["finished_at"]


def test_default_spawn_runs_in_a_background_thread():
    # Exercise the real thread path (no injected spawn): the job completes asynchronously.
    runner = jobs.JobRunner({"slow": lambda a: {"ok": True}})
    jid = runner.submit("slow", {})
    for _ in range(200):                                           # poll up to ~2s for completion
        if db.get_job(_con(), jid)["status"] == "done":
            break
        time.sleep(0.01)
    assert db.get_job(_con(), jid)["status"] == "done"
