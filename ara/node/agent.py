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

import json
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from ara.node import capabilities, config as config_mod, enroll, health, wiring
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


def _results_dir() -> Path:
    """Where finished-but-unacknowledged results are spooled (under the node's state dir)."""
    return config_mod.node_dir() / "results"


def _spool_path(job_id: str) -> Path:
    return _results_dir() / f"{job_id}.json"


def _spool_result(job_id: str, payload: dict) -> None:
    """Persist a finished result to disk BEFORE any network attempt, so a report failure or a crash
    can never lose completed work (Rule #1). Same job → same outcome, so overwrite is fine."""
    _results_dir().mkdir(parents=True, exist_ok=True)
    _spool_path(job_id).write_text(json.dumps(payload), encoding="utf-8")


def _try_post(client: NodeClient, config, job_id: str, payload: dict):
    """POST one result. On a 401 (revoked token) re-enroll once and retry with the fresh client.
    Returns ``(delivered, client)`` and NEVER raises on a report failure — the caller keeps the
    result spooled for a later retry instead of crashing or discarding it."""
    try:
        client.post_result(job_id, payload)
        return True, client
    except httpx.HTTPStatusError as exc:
        if not _is_unauthorized(exc):
            return False, client
        client = _reauth(config)                    # token revoked → refresh, then retry once
    except httpx.HTTPError:
        return False, client
    try:
        client.post_result(job_id, payload)
        return True, client
    except httpx.HTTPError:
        return False, client


def _flush_spool(client: NodeClient, config) -> NodeClient:
    """Re-deliver any durably-spooled results (left by an earlier failure or a crash), removing each
    on success — so finished work survives a restart. A corrupt spool file is dropped, not retried
    forever or crashed on."""
    d = _results_dir()
    if not d.exists():
        return client
    for f in sorted(d.glob("*.json")):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            f.unlink(missing_ok=True)
            continue
        delivered, client = _try_post(client, config, f.stem, payload)
        if delivered:
            f.unlink(missing_ok=True)
    return client


def run_loop(config, *, client: NodeClient | None = None,
             runner: Callable[[str, dict], dict] | None = None, wait: float = 20.0,
             max_iterations: int | None = None, sleep=time.sleep, poll_gap: float = 0.0,
             reauth_backoff: float = 5.0) -> int:
    """Run the phone-home work loop, returning the number of poll iterations performed.

    Ensures the node is enrolled (a session token present), then for each iteration long-polls for a
    job, runs it, and reports the outcome. Emits an sd_notify heartbeat + status each iteration so
    systemd's watchdog is fed (and journald sees liveness off systemd). A 401 on either call means
    the session token was revoked/expired: the node re-enrolls for a fresh token, rebuilds the
    client, and carries on rather than crashing — then sleeps ``reauth_backoff`` so a server that
    persistently 401s can't busy-loop (a 401 returns immediately, with none of the long-poll's
    natural pacing). ``max_iterations`` bounds the loop (None = forever, the production default);
    ``client``/``runner``/``sleep`` are injectable for tests."""
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
        client = _flush_spool(client, config)   # first, deliver anything left by a prior failure/crash
        try:
            job = client.get_work(wait)
        except httpx.HTTPStatusError as exc:
            if _is_unauthorized(exc):
                client = _reauth(config)
                sleep(reauth_backoff)          # backoff: a persistent 401 mustn't busy-loop
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
        # Durable BEFORE the network: spool the finished result, then post. Only remove it once the
        # server has acknowledged it — so a 401/5xx/crash retries later instead of losing the work.
        _spool_result(job["id"], payload)
        delivered, client = _try_post(client, config, job["id"], payload)
        if delivered:
            _spool_path(job["id"]).unlink(missing_ok=True)
        else:
            sleep(reauth_backoff)              # bounded backoff; the result stays spooled for retry
    return count
