# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node agent loop — phone home, pull a job, run it, report the result. Repeat.

This is the push-only node's whole life: ensure it's enrolled, then long-poll ``GET /api/work``,
run each dispatched job by reusing ARA's existing node wiring (:mod:`ara.node.wiring` — the same
CLI-shell-out workers the pull-model app uses, so every safety gate comes along), and POST the
outcome back as a ``result.request``. The loop is bounded (``max_iterations``) and its collaborators
(client, runner, sleep) are injectable so a test can run exactly one iteration without a socket, a
subprocess, or a wall-clock wait.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import httpx

from ara.node import capabilities, enroll, health, wiring
from ara.node.client import NodeClient


def default_runner() -> Callable[[str, dict], dict]:
    """A job dispatcher over ARA's real verbs: action verbs via the workers, ``detect`` via the
    read providers. Raises ``ValueError`` for a kind this node can't run (reported as failed)."""
    workers = wiring.default_workers()
    providers = wiring.default_providers()

    def _run(kind: str, args: dict) -> dict:
        if kind in workers:
            return workers[kind](args)
        if kind in providers:
            return providers[kind]()
        raise ValueError(f"unknown job kind: {kind!r}")

    return _run


def _result_payload(result: dict) -> dict:
    """Shape a worker's return into a ``result.request``. A worker signals failure by returning an
    ``{"error": ...}`` dict (the wiring convention), which becomes a ``failed`` result."""
    env = capabilities.environment()
    if isinstance(result, dict) and "error" in result:
        return {"status": "failed", "error": str(result["error"]), "environment": env}
    return {"status": "done", "result": result, "environment": env}


def _is_unauthorized(exc: httpx.HTTPStatusError) -> bool:
    """A 401 means our session token was revoked or expired — time to re-enroll."""
    return exc.response.status_code == 401


def _reauth(config) -> NodeClient:
    """Session token rejected (401): drop it, re-run the enroll handshake for a fresh one, and
    rebuild the client around it."""
    config.session_token = None
    enroll.enroll_flow(config)
    return NodeClient(config.server_url, config.session_token)


def run_loop(config, *, client: NodeClient | None = None,
             runner: Callable[[str, dict], dict] | None = None, wait: float = 20.0,
             max_iterations: int | None = None, sleep=time.sleep, poll_gap: float = 0.0) -> int:
    """Run the phone-home work loop, returning the number of poll iterations performed.

    Ensures the node is enrolled (a session token present), then for each iteration long-polls for a
    job, runs it, and reports the outcome. Emits an sd_notify heartbeat + status each iteration so
    systemd's watchdog is fed (and journald sees liveness off systemd). A 401 on either call means
    the session token was revoked/expired: the node re-enrolls for a fresh token, rebuilds the
    client, and carries on rather than crashing. ``max_iterations`` bounds the loop (None = forever,
    the production default); ``client``/``runner``/``sleep`` are injectable for tests."""
    if not config.session_token:
        enroll.enroll_flow(config)
    client = client or NodeClient(config.server_url, config.session_token)
    runner = runner or default_runner()
    health.ready()
    count = 0
    while max_iterations is None or count < max_iterations:
        count += 1
        health.heartbeat()
        health.status(f"polling for work (iteration {count})")
        try:
            job = client.get_work(wait)
        except httpx.HTTPStatusError as exc:
            if _is_unauthorized(exc):
                client = _reauth(config)
                continue
            raise
        if job is None:
            sleep(poll_gap)
            continue
        try:
            payload = _result_payload(runner(job["kind"], job.get("args") or {}))
        except Exception as exc:  # noqa: BLE001 — any run failure becomes a reported failed result
            payload = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                       "environment": capabilities.environment()}
        try:
            client.post_result(job["id"], payload)
        except httpx.HTTPStatusError as exc:
            if _is_unauthorized(exc):
                client = _reauth(config)
                continue
            raise
    return count
