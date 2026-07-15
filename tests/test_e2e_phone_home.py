# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""End-to-end phone-home loop: real Python node client <-> real containerized coordinator over HTTP.

Layer e2e of the testing architecture (spec 2026-07-01-ara-testing-architecture). The unit suite
mocks the network and the wire fixtures are static; THIS proves the two languages actually talk at
runtime: the node's httpx client enrolls, polls, claims work, and posts a result against the real
Next.js coordinator's route handlers, exercising session-token auth, the one-time enrollment-token
poll, the IDOR guard, and the atomic work claim end to end.

Grey-box by necessity: the node-facing steps (enroll/poll/work/result) are pure HTTP and fully
black-box. The ADMIN steps (issue token, approve, enqueue) are Next.js *server actions* with no
external REST seam, so they're seeded by exec-ing the coordinator's own better-sqlite3 inside the
container (documented setup coupling — the admin *logic* is unit-tested in phone-home.lib; here the
seed just puts the coordinator in the state the wire test needs). Marked ``e2e``: excluded from the
hermetic gate, opt-in via ``pytest -m e2e --no-cov``. Requires docker + docker compose.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from ara.node import capabilities
from ara.node.client import NodeClient

pytestmark = pytest.mark.e2e

COMPOSE_FILE = str(Path(__file__).resolve().parent.parent / "docker-compose.e2e.yml")
BASE_URL = "http://127.0.0.1:34714"
SERVICE = "coordinator"


def _compose(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "compose", "-f", COMPOSE_FILE, *args],
                          capture_output=True, text=True, check=check)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _db_js(js: str) -> str:
    """Run a snippet against the coordinator's own better-sqlite3 inside the container and return
    stdout. `db` is the open handle; the snippet prints any result itself."""
    wrapper = (
        "const db=require('better-sqlite3')(process.env.ARA_COORDINATOR_DB);"
        f"{js}"
    )
    proc = _compose("exec", "-T", SERVICE, "node", "-e", wrapper)
    return proc.stdout.strip()


@pytest.fixture(scope="module")
def coordinator():
    # --build so a stale image can't mask a Dockerfile regression; --wait blocks on the healthcheck.
    _compose("up", "-d", "--build", "--wait")
    try:
        # Ensure the schema exists before we seed: a throwaway unauthenticated request opens the DB
        # (open() runs the CREATE TABLE IF NOT EXISTS statements). /api/work with no token -> 401.
        for _ in range(30):
            try:
                httpx.get(f"{BASE_URL}/api/work", headers={"Authorization": "Bearer nope"}, timeout=5)
                break
            except httpx.HTTPError:
                time.sleep(1)
        yield BASE_URL
    finally:
        _compose("down", "-v", check=False)


def test_phone_home_enroll_approve_work_result(coordinator):
    # 1) Admin seeds a one-time enrollment token (only its sha256 is stored — same as the real path).
    enroll_token = secrets.token_urlsafe(32)
    _db_js(
        "db.prepare('INSERT INTO enrollment_tokens (token_hash) VALUES (?)')"
        f".run({json.dumps(_sha256_hex(enroll_token))});"
    )

    # 2) The REAL node client enrolls over HTTP -> lands PENDING (real /api/enroll route).
    client = NodeClient(coordinator, enroll_token)
    resp = client.enroll(capabilities.self_description())
    enrollment_id = resp["enrollment_id"]
    assert resp["status"] == "pending"

    # 3) Admin approves — replicate approveAgent's mint (status active + session token hash + the
    #    one-poll plaintext). The node will receive this exact plaintext on its next poll.
    session_token = secrets.token_urlsafe(32)
    _db_js(
        "db.prepare(\"UPDATE agents SET status='active', session_token_hash=?, "
        "pending_session_token=? WHERE enrollment_id=?\")"
        f".run({json.dumps(_sha256_hex(session_token))}, {json.dumps(session_token)}, "
        f"{json.dumps(enrollment_id)});"
    )

    # 4) The node polls and receives the session token (real /api/enroll/[id]: IDOR guard + one-time
    #    hand-off that NULLs pending_session_token after this read).
    poll = client.poll_approval(enrollment_id)
    assert poll["status"] == "active"
    assert poll["session_token"] == session_token

    agent_id = int(_db_js(
        "process.stdout.write(String(db.prepare('SELECT id FROM agents WHERE enrollment_id=?')"
        f".get({json.dumps(enrollment_id)}).id));"
    ))

    # 5) Admin enqueues a job for this agent.
    job_id = "e2e-job-1"
    _db_js(
        "db.prepare(\"INSERT INTO work (id, agent_id, kind, args_json, status) "
        "VALUES (?, ?, 'detect', '{}', 'queued')\")"
        f".run({json.dumps(job_id)}, {agent_id});"
    )

    # 6) The node receives an atomic offer, journals it durably, then acknowledges before execution.
    authed = NodeClient(coordinator, session_token)
    job = authed.get_work(wait=5)
    assert job is not None and job["id"] == job_id and job["kind"] == "detect"
    authed.ack_work(job_id)  # durable local acceptance authorizes execution

    # 7) The node posts a result (real /api/work/[id]/result: session auth + record). Shape matches
    #    result.request; environment is the real node-produced shape.
    authed.post_result(job_id, {
        "status": "done",
        "result": {"ok": True, "kind": job["kind"]},
        "environment": capabilities.environment(),
    })

    # 8) The coordinator recorded the outcome.
    row = json.loads(_db_js(
        "const r=db.prepare('SELECT status, result_json FROM work WHERE id=?')"
        f".get({json.dumps(job_id)});"
        "process.stdout.write(JSON.stringify(r));"
    ))
    assert row["status"] == "done"
    # recordResult stores result_json = JSON.stringify(payload.result), so result_json IS the
    # result object the node sent (not wrapped) — {"ok": true, "kind": "detect"}.
    assert json.loads(row["result_json"]) == {"ok": True, "kind": "detect"}

    client.close()
    authed.close()
