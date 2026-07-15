# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node agent loop — phone home, pull a job, run it, report the result. Repeat.

This is the push-only node's whole life: require enrollment, long-poll ``GET /api/work``, durably
journal and acknowledge each offer, then run it through ARA's existing node wiring
(:mod:`ara.node.wiring` — the same
CLI-shell-out workers the pull-model app uses, so every safety gate comes along), and POST the
outcome back as a ``result.request``. The loop is bounded (``max_iterations``) and its collaborators
(client, runner, sleep) are injectable so a test can run exactly one iteration without a socket, a
subprocess, or a wall-clock wait.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from ara.node import capabilities, config as config_mod, health, wiring
from ara.node.client import NodeClient, WIRE_JOB_KINDS


def default_runner() -> Callable[[str, dict], dict]:
    """A job dispatcher over ARA's real verbs: action verbs via the workers, ``detect`` via the
    read providers. Raises ``ValueError`` for a kind this node can't run (reported as failed)."""
    workers = wiring.default_workers()
    providers = wiring.default_providers()
    action_kinds = frozenset({"run", "characterize", "benchmark"})

    def _run(kind: str, args: dict) -> dict:
        if kind in action_kinds and kind in workers:
            return workers[kind](args)
        if kind == "detect" and kind in providers:
            return providers[kind]()
        raise ValueError(f"unknown job kind: {kind!r}")

    return _run


def _result_payload(result: dict) -> dict:
    """Shape a worker's return into a ``result.request``. A worker signals failure by returning an
    ``{"error": ...}`` dict (the wiring convention), which becomes a ``failed`` result."""
    env = capabilities.environment()
    if isinstance(result, dict) and "error" in result:
        error = str(result["error"])
        stderr = result.get("stderr")
        if isinstance(stderr, str) and stderr:
            error = f"{error}\nstderr: {stderr}"
        return {"status": "failed", "error": error, "environment": env}
    return {"status": "done", "result": result, "environment": env}


def _is_unauthorized(exc: httpx.HTTPStatusError) -> bool:
    """Return whether the coordinator rejected the node's current session."""
    return exc.response.status_code == 401


class _ReenrollmentRequired(RuntimeError):
    """The current session was rejected and a fresh explicit enrollment is required."""


class _TerminalResultRejection(RuntimeError):
    """The coordinator permanently rejected a completed result; retrying cannot repair it."""


class _TerminalJobRejection(RuntimeError):
    """The coordinator permanently rejected an accepted-job acknowledgement."""


def _invalidate_session(config) -> None:
    """Persist the rejected session as unusable; one-shot enrollment tokens are never retried."""
    rejected_server = config.server_url
    rejected_session = config.session_token
    config.session_token = None
    config.enrollment_token = None
    config_mod.clear_session_if_current(rejected_server, rejected_session)


def _results_dir() -> Path:
    """Where finished-but-unacknowledged results are spooled (under the node's state dir)."""
    return config_mod.node_dir() / "results"


def _accepted_dir() -> Path:
    """Jobs durably accepted from the coordinator but not yet converted into result spools."""
    return config_mod.node_dir() / "accepted"


def _spool_path(job_id: str) -> Path:
    digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
    return _results_dir() / f"{digest}.json"


def _accepted_path(job_id: str) -> Path:
    digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
    return _accepted_dir() / f"{digest}.json"


def _fsync_parent(path: Path) -> None:
    """Make a completed rename durable on POSIX; directory handles are not portable to Windows."""
    if os.name == "nt":
        return
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, value: dict) -> None:
    """Write JSON through a same-directory owner-only temporary, then atomically replace *path*."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _spool_result(job_id: str, payload: dict) -> None:
    """Persist a finished result to disk BEFORE any network attempt, so a report failure or a crash
    can never lose completed work (Rule #1). Same job → same outcome, so overwrite is fine."""
    envelope = {"version": 1, "job_id": job_id, "payload": payload}
    _write_json_atomic(_spool_path(job_id), envelope)


def _journal_job(job: dict) -> None:
    """Persist an offered job before acknowledging or executing it."""
    envelope = {"version": 1, "job": job}
    _write_json_atomic(_accepted_path(job["id"]), envelope)


_SAFE_LEGACY_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _read_spooled_result(path: Path) -> tuple[str, dict]:
    """Read a current envelope or a legacy payload whose filename contains a safe job ID."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("spooled result must be a JSON object")
    if set(value) == {"version", "job_id", "payload"} and value["version"] == 1:
        job_id = value["job_id"]
        payload = value["payload"]
        if not isinstance(job_id, str) or not isinstance(payload, dict):
            raise ValueError("invalid spooled result envelope")
        if path != _spool_path(job_id):
            raise ValueError("spooled result filename does not match its job ID")
        return job_id, payload
    if not _SAFE_LEGACY_JOB_ID.fullmatch(path.stem):
        raise ValueError("unsafe legacy spool filename")
    return path.stem, value


def _quarantine_spool(path: Path) -> None:
    """Move an unreadable spool entry out of the retry lane without destroying evidence."""
    quarantine = path.parent / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = quarantine / path.name
    index = 1
    while destination.exists() or destination.is_symlink():
        destination = quarantine / f"{path.name}.{index}"
        index += 1
    os.replace(path, destination)
    _fsync_parent(destination)
    if not destination.is_symlink():
        destination.chmod(0o600)


def _read_accepted_job(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"version", "job"} or value["version"] != 1:
        raise ValueError("invalid accepted-job envelope")
    job = value["job"]
    if (not isinstance(job, dict) or set(job) != {"id", "kind", "args"}
            or not isinstance(job["id"], str) or not job["id"]
            or job["kind"] not in WIRE_JOB_KINDS
            or not isinstance(job["args"], dict)
            or path != _accepted_path(job["id"])):
        raise ValueError("invalid accepted job")
    return job


def _next_accepted_job() -> tuple[dict, Path] | None:
    """Return the oldest recoverable accepted job, preserving malformed entries as evidence."""
    directory = _accepted_dir()
    if not directory.exists():
        return None
    for path in sorted(directory.glob("*.json")):
        try:
            if path.is_symlink():
                raise ValueError("accepted-job path is a symlink")
            job = _read_accepted_job(path)
        except (OSError, ValueError):
            try:
                _quarantine_spool(path)
            except OSError:
                pass
            continue
        if _spool_path(job["id"]).exists():
            path.unlink(missing_ok=True)  # execution completed before the earlier interruption
            continue
        return job, path
    return None


def _try_ack(client: NodeClient, config, job_id: str) -> bool:
    """Acknowledge a durably journaled offer; classify retryable and permanent failures."""
    try:
        client.ack_work(job_id)
        return True
    except httpx.HTTPStatusError as exc:
        if _is_unauthorized(exc):
            _invalidate_session(config)
            raise _ReenrollmentRequired from exc
        if exc.response.status_code in {408, 429} or exc.response.status_code >= 500:
            return False
        raise _TerminalJobRejection(
            f"coordinator permanently rejected job {job_id!r} acknowledgement "
            f"with HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError:
        return False


def _try_post(client: NodeClient, config, job_id: str, payload: dict):
    """POST one result without losing it on failure.

    Transient failures return ``False`` so the caller keeps the spool. A 401 invalidates the
    rejected session and raises ``_ReenrollmentRequired``: enrollment tokens are one-shot, so only
    a new explicit ``ara node enroll`` can establish the next session.
    """
    try:
        client.post_result(job_id, payload)
        return True, client
    except httpx.HTTPStatusError as exc:
        if _is_unauthorized(exc):
            _invalidate_session(config)
            raise _ReenrollmentRequired from exc
        if exc.response.status_code in {408, 429} or exc.response.status_code >= 500:
            return False, client
        raise _TerminalResultRejection(
            f"coordinator permanently rejected result {job_id!r} "
            f"with HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError:
        return False, client


def _flush_spool(client: NodeClient, config) -> NodeClient:
    """Re-deliver any durably-spooled results (left by an earlier failure or a crash), removing each
    on success — so finished work survives a restart. Corrupt spool files are quarantined for
    inspection rather than retried forever or destroyed."""
    d = _results_dir()
    if not d.exists():
        return client
    for f in sorted(d.glob("*.json")):
        try:
            if f.is_symlink():
                raise ValueError("spool path is a symlink")
            job_id, payload = _read_spooled_result(f)
        except (OSError, ValueError):
            try:
                _quarantine_spool(f)
            except OSError:
                pass
            continue
        try:
            delivered, client = _try_post(client, config, job_id, payload)
        except _TerminalResultRejection as exc:
            health.status(str(exc))
            try:
                _quarantine_spool(f)
            except OSError:
                pass
            continue
        if delivered:
            _accepted_path(job_id).unlink(missing_ok=True)
            f.unlink(missing_ok=True)
    return client


def run_loop(config, *, client: NodeClient | None = None,
             runner: Callable[[str, dict], dict] | None = None, wait: float = 20.0,
             max_iterations: int | None = None, sleep=time.sleep, poll_gap: float = 0.0,
             reauth_backoff: float = 5.0) -> int:
    """Run the phone-home work loop, returning the number of poll iterations performed.

    Requires an active session, then for each iteration recovers accepted work or long-polls for a
    fresh offer, journals and acknowledges it before execution, and durably spools the outcome. A
    401 invalidates the rejected session and stops cleanly for an explicit fresh
    enrollment; a one-shot enrollment token is never reused. Coordinator transport, 5xx, and
    malformed-response failures back off and retry. ``max_iterations`` bounds the loop (None =
    forever, the production default); ``client``/``runner``/``sleep`` are injectable for tests."""
    if not config.session_token:
        raise ValueError("re-enrollment required — run: ara node enroll <server_url> --token <token>")
    client = client or NodeClient(config.server_url, config.session_token)
    runner = runner or default_runner()
    health.ready()
    count = 0
    while max_iterations is None or count < max_iterations:
        count += 1
        health.heartbeat()
        health.status(f"polling for work (iteration {count})")
        try:
            client = _flush_spool(client, config)  # deliver work left by a prior failure/crash
        except _ReenrollmentRequired:
            health.status("re-enrollment required — session rejected while reporting a result")
            return count
        accepted = _next_accepted_job()
        if accepted is None:
            try:
                job = client.get_work(wait)
            except httpx.HTTPStatusError as exc:
                if _is_unauthorized(exc):
                    _invalidate_session(config)
                    health.status("re-enrollment required — coordinator rejected the session")
                    return count
                health.status(f"coordinator unavailable: {exc}")
                sleep(reauth_backoff)
                continue
            except (httpx.HTTPError, ValueError) as exc:
                health.status(f"coordinator unavailable: {exc}")
                sleep(reauth_backoff)
                continue
            if job is None:
                sleep(poll_gap)
                continue
            _journal_job(job)
            accepted_path = _accepted_path(job["id"])
        else:
            job, accepted_path = accepted
        try:
            acknowledged = _try_ack(client, config, job["id"])
        except _ReenrollmentRequired:
            health.status("re-enrollment required — coordinator rejected the session")
            return count
        except _TerminalJobRejection as exc:
            health.status(str(exc))
            try:
                _quarantine_spool(accepted_path)
            except OSError:
                pass
            continue
        if not acknowledged:
            sleep(reauth_backoff)
            continue
        try:
            payload = _result_payload(runner(job["kind"], job.get("args") or {}))
        except Exception as exc:  # noqa: BLE001 — any run failure becomes a reported failed result
            payload = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                       "environment": capabilities.environment()}
        # Durable BEFORE the network: spool the finished result, then post. Only remove it once the
        # server has acknowledged it — so a 401/5xx/crash retries later instead of losing the work.
        _spool_result(job["id"], payload)
        accepted_path.unlink(missing_ok=True)
        try:
            delivered, client = _try_post(client, config, job["id"], payload)
        except _ReenrollmentRequired:
            health.status("re-enrollment required — coordinator rejected the session")
            return count
        except _TerminalResultRejection as exc:
            health.status(str(exc))
            try:
                _quarantine_spool(_spool_path(job["id"]))
            except OSError:
                pass
            continue
        if delivered:
            _spool_path(job["id"]).unlink(missing_ok=True)
        else:
            sleep(reauth_backoff)              # bounded backoff; the result stays spooled for retry
    return count
