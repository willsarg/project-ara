# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node job store (db.py jobs table) — create / get / update / list."""
from __future__ import annotations

import json

import pytest

from ara import db


@pytest.fixture
def con(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "jobs.db"))
    return db.connect()


def test_create_and_get_job_roundtrip(con):
    db.create_job(con, "j1", "characterize", json.dumps({"model": "gemma-3-1b"}))
    job = db.get_job(con, "j1")
    assert job["id"] == "j1"
    assert job["kind"] == "characterize"
    assert json.loads(job["args_json"]) == {"model": "gemma-3-1b"}
    assert job["status"] == "queued"          # new jobs start queued
    assert job["created_at"]                  # stamped
    assert job["result_json"] is None and job["error"] is None and job["finished_at"] is None


def test_get_missing_job_is_none(con):
    assert db.get_job(con, "nope") is None


def test_update_job_transitions_and_partial_updates(con):
    db.create_job(con, "j1", "run", "{}")
    db.update_job(con, "j1", status="running", started_at="2026-06-30T00:00:00+00:00")
    j = db.get_job(con, "j1")
    assert j["status"] == "running" and j["started_at"] == "2026-06-30T00:00:00+00:00"
    # a later partial update must not clobber fields it doesn't set (kind/args/started_at survive)
    db.update_job(con, "j1", status="done", result_json='{"safe_context": 4096}',
                  finished_at="2026-06-30T00:05:00+00:00")
    j = db.get_job(con, "j1")
    assert j["status"] == "done"
    assert json.loads(j["result_json"]) == {"safe_context": 4096}
    assert j["finished_at"] == "2026-06-30T00:05:00+00:00"
    assert j["started_at"] == "2026-06-30T00:00:00+00:00"   # untouched by the second update
    assert j["kind"] == "run"


def test_update_unknown_job_is_noop(con):
    db.update_job(con, "ghost", status="done")     # must not raise
    assert db.get_job(con, "ghost") is None


def test_update_with_no_recognised_fields_is_noop(con):
    db.create_job(con, "j1", "run", "{}")
    db.update_job(con, "j1", bogus="x")            # nothing recognised → no write, no error
    assert db.get_job(con, "j1")["status"] == "queued"


def test_list_jobs_newest_first_and_respects_limit(con):
    for i, ts in enumerate(["2026-06-30T00:00:01+00:00", "2026-06-30T00:00:02+00:00",
                            "2026-06-30T00:00:03+00:00"]):
        db.create_job(con, f"j{i}", "benchmark", "{}", created_at=ts)
    rows = db.list_jobs(con)
    assert [r["id"] for r in rows] == ["j2", "j1", "j0"]    # newest first
    assert [r["id"] for r in db.list_jobs(con, limit=2)] == ["j2", "j1"]
