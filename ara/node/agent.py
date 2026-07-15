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
from contextlib import contextmanager
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
        raw_error = result["error"]
        error = str(raw_error).strip() if raw_error is not None else ""
        if not error:
            error = "node worker failed without an error message"
        stderr = result.get("stderr")
        if isinstance(stderr, str) and stderr.strip():
            error = f"{error}\nstderr: {stderr.strip()}"
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


class _AcceptedJournalBlocked(RuntimeError):
    """An older accepted-job journal cannot be recovered or safely quarantined."""


class NodeAgentBusy(RuntimeError):
    """Another node loop already owns this state directory."""


class CoordinatorWorkRejected(RuntimeError):
    """The coordinator permanently rejected the node's work-poll request."""


def _is_windows() -> bool:
    return os.name == "nt"


def _try_lock_agent(fd: int) -> bool:
    """Try to lease the node state directory without blocking."""
    if _is_windows():
        import msvcrt
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock_agent(fd: int) -> None:
    if _is_windows():
        import msvcrt
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def _agent_lease():
    """Ensure only one process can poll and execute work from one node state directory."""
    path = config_mod.node_dir() / "agent.lock"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        if not _try_lock_agent(fd):
            raise NodeAgentBusy(
                "another ARA node run loop already owns this node state directory")
        try:
            yield
        finally:
            _unlock_agent(fd)
    finally:
        os.close(fd)


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


def _ensure_private_directory(path: Path) -> None:
    """Create a state directory without accepting a symlink or non-directory in its place."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise OSError(f"ARA node state directory must not be a symlink: {path}")
    if not path.is_dir():
        raise OSError(f"ARA node state path is not a directory: {path}")


def _write_json_atomic(path: Path, value: dict) -> None:
    """Write JSON through a same-directory owner-only temporary, then atomically replace *path*."""
    _ensure_private_directory(path.parent)
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


def _verify_result_storage() -> None:
    """Exercise the real atomic-write path before ARA acknowledges or executes another job."""
    probe = _results_dir() / f".write-probe-{os.getpid()}-{time.time_ns()}"
    try:
        _write_json_atomic(probe, {"probe": True})
    finally:
        probe.unlink(missing_ok=True)


_last_spool_order = 0


def _persisted_spool_order() -> int:
    highest = 0
    paths = list(_results_dir().glob("*.json")) + list(_accepted_dir().glob("*.json"))
    for path in paths:
        try:
            if path.is_symlink():
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(value, dict):
            continue
        order = value.get("order")
        completion = value.get("completion")
        if isinstance(completion, dict):
            order = completion.get("order")
        if isinstance(order, int) and not isinstance(order, bool) and order > highest:
            highest = order
    return highest


def _next_spool_order() -> int:
    """Return a process-monotonic completion order suitable for durable spool replay."""
    global _last_spool_order  # noqa: PLW0603 — the single leased agent owns this sequence
    _last_spool_order = max(time.time_ns(), _last_spool_order + 1, _persisted_spool_order() + 1)
    return _last_spool_order


def _new_result_envelope(job_id: str, payload: dict) -> dict:
    return {
        "version": 2,
        "order": _next_spool_order(),
        "job_id": job_id,
        "payload": payload,
    }


def _write_result_envelope(envelope: dict) -> None:
    _write_json_atomic(_spool_path(envelope["job_id"]), envelope)


def _spool_result(job_id: str, payload: dict) -> None:
    """Persist a finished result to disk BEFORE any network attempt, so a report failure or a crash
    can never lose completed work (Rule #1). Same job → same outcome, so overwrite is fine."""
    _write_result_envelope(_new_result_envelope(job_id, payload))


def _journal_job(job: dict) -> None:
    """Persist an offered job before acknowledging or executing it."""
    envelope = {"version": 2, "order": _next_spool_order(), "job": job}
    _write_json_atomic(_accepted_path(job["id"]), envelope)


def _journal_completion(path: Path, job: dict, result_envelope: dict) -> None:
    """Atomically convert an accepted job into durable completed work before result spooling."""
    _write_json_atomic(path, {"version": 2, "job": job, "completion": result_envelope})


_SAFE_LEGACY_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _read_spooled_result(path: Path) -> tuple[str, dict, int | None]:
    """Read a current envelope or a legacy payload whose filename contains a safe job ID."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("spooled result must be a JSON object")
    current = set(value) == {"version", "order", "job_id", "payload"} and value["version"] == 2
    previous = set(value) == {"version", "job_id", "payload"} and value["version"] == 1
    if current or previous:
        job_id = value["job_id"]
        payload = value["payload"]
        if not isinstance(job_id, str) or not isinstance(payload, dict):
            raise ValueError("invalid spooled result envelope")
        order = value.get("order")
        if current and (not isinstance(order, int) or isinstance(order, bool) or order < 1):
            raise ValueError("invalid spooled result order")
        if path != _spool_path(job_id):
            raise ValueError("spooled result filename does not match its job ID")
        return job_id, payload, order
    if not _SAFE_LEGACY_JOB_ID.fullmatch(path.stem):
        raise ValueError("unsafe legacy spool filename")
    return path.stem, value, None


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
        destination.chmod(0o700 if destination.is_dir() else 0o600)


def _retire_accepted(path: Path) -> bool:
    """Remove a completed job journal, quarantining malformed paths while preserving evidence."""
    try:
        path.unlink(missing_ok=True)
        return True
    except OSError:
        try:
            _quarantine_spool(path)
            return True
        except OSError as exc:
            health.status(f"could not retire completed job journal {path.name}: {exc}")
            return False


def _read_accepted_job(path: Path) -> tuple[dict, int | None]:
    value = json.loads(path.read_text(encoding="utf-8"))
    legacy = isinstance(value, dict) and set(value) == {"version", "job"} and value["version"] == 1
    current = (isinstance(value, dict) and set(value) == {"version", "order", "job"}
               and value["version"] == 2)
    if not legacy and not current:
        raise ValueError("invalid accepted-job envelope")
    job = value["job"]
    order = value.get("order")
    if current and (not isinstance(order, int) or isinstance(order, bool) or order < 1):
        raise ValueError("invalid accepted-job order")
    if (not isinstance(job, dict) or set(job) != {"id", "kind", "args"}
            or not isinstance(job["id"], str) or not job["id"]
            or job["kind"] not in WIRE_JOB_KINDS
            or not isinstance(job["args"], dict)
            or path != _accepted_path(job["id"])):
        raise ValueError("invalid accepted job")
    return job, order


def _next_accepted_job() -> tuple[dict, Path] | None:
    """Return the oldest recoverable accepted job, preserving malformed entries as evidence."""
    directory = _accepted_dir()
    if not directory.exists():
        return None
    entries = []
    order_counts: dict[int, int] = {}
    for path in directory.glob("*.json"):
        try:
            modified_ns = path.lstat().st_mtime_ns
            if path.is_symlink():
                raise ValueError("accepted-job path is a symlink")
            job, order = _read_accepted_job(path)
        except (OSError, ValueError):
            try:
                _quarantine_spool(path)
            except OSError as exc:
                raise _AcceptedJournalBlocked(
                    f"could not quarantine accepted-job journal {path.name}: {exc}"
                ) from exc
            continue
        order_key = order if order is not None else modified_ns
        order_counts[order_key] = order_counts.get(order_key, 0) + 1
        entries.append((order_key, path.name, job, path))
    for order_key, _name, job, path in sorted(entries):
        if order_counts[order_key] > 1:
            raise _AcceptedJournalBlocked(
                "accepted-job order is ambiguous; preserving evidence and blocking work"
            )
        if _spool_path(job["id"]).exists():
            path.unlink(missing_ok=True)  # execution completed before the earlier interruption
            continue
        return job, path
    return None


def _recover_completed_journals() -> None:
    """Materialize completed accepted journals without ever invoking their runners again."""
    directory = _accepted_dir()
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # malformed pending entries are handled by _next_accepted_job
        if not isinstance(value, dict) or set(value) != {"version", "job", "completion"}:
            continue
        if value["version"] != 2:
            continue
        job = value["job"]
        completion = value["completion"]
        if (not isinstance(job, dict) or set(job) != {"id", "kind", "args"}
                or not isinstance(job["id"], str) or not job["id"]
                or job["kind"] not in WIRE_JOB_KINDS or not isinstance(job["args"], dict)
                or path != _accepted_path(job["id"])
                or not isinstance(completion, dict)
                or set(completion) != {"version", "order", "job_id", "payload"}
                or completion["version"] != 2 or completion["job_id"] != job["id"]
                or not isinstance(completion["order"], int)
                or isinstance(completion["order"], bool) or completion["order"] < 1
                or not isinstance(completion["payload"], dict)):
            continue
        spool_path = _spool_path(job["id"])
        if spool_path.exists() or spool_path.is_symlink():
            try:
                existing = json.loads(spool_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise _AcceptedJournalBlocked(
                    f"completion WAL conflicts with unreadable result spool {spool_path.name}"
                ) from exc
            if existing != completion:
                raise _AcceptedJournalBlocked(
                    f"completion WAL conflicts with existing result spool {spool_path.name}"
                )
        else:
            _write_result_envelope(completion)
        if not _retire_accepted(path):
            raise _AcceptedJournalBlocked(
                f"could not retire completed accepted-job journal {path.name}"
            )


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
    if d.is_symlink():
        return client
    if not d.exists():
        return client
    entries = []
    order_counts: dict[int, int] = {}
    blocked = False
    for f in d.glob("*.json"):
        try:
            modified_ns = f.lstat().st_mtime_ns
            if f.is_symlink():
                raise ValueError("spool path is a symlink")
            job_id, payload, order = _read_spooled_result(f)
        except (OSError, ValueError):
            try:
                _quarantine_spool(f)
            except OSError as exc:
                health.status(f"could not quarantine unreadable result spool {f.name}: {exc}")
                blocked = True
            continue
        # Versioned order is authoritative. Legacy entries predate ordered envelopes, so their
        # persisted write time is only usable when unique; a tie is honestly unknowable.
        order_key = order if order is not None else modified_ns
        order_counts[order_key] = order_counts.get(order_key, 0) + 1
        entries.append((order_key, f.name, f, job_id, payload))
    if blocked:
        return client
    for order_key, _name, f, job_id, payload in sorted(entries):
        if order_counts[order_key] > 1:
            health.status("result spool order is ambiguous; preserving evidence and blocking work")
            break
        # The completion spool is durable proof that execution already finished. Its older
        # accepted-job journal is now redundant even if result delivery remains unavailable.
        accepted_retired = _retire_accepted(_accepted_path(job_id))
        try:
            delivered, client = _try_post(client, config, job_id, payload)
        except _TerminalResultRejection as exc:
            health.status(str(exc))
            if accepted_retired:
                try:
                    _quarantine_spool(f)
                except OSError as quarantine_exc:
                    health.status(
                        f"could not quarantine rejected result spool {f.name}: {quarantine_exc}")
                    return client
                continue
            break
        if delivered and accepted_retired:
            f.unlink(missing_ok=True)
        elif delivered:
            break
        else:
            # Do not report a later completion ahead of the oldest transient failure. Leaving the
            # remaining files untouched also lets the loop recognize that it must not accept work.
            break
    return client


def run_loop(config, *, client: NodeClient | None = None,
             runner: Callable[[str, dict], dict] | None = None, wait: float = 20.0,
             max_iterations: int | None = None, sleep=time.sleep, poll_gap: float = 0.0,
             reauth_backoff: float = 5.0) -> int:
    """Run one process-exclusive node loop for this state directory."""
    with _agent_lease():
        return _run_loop(
            config, client=client, runner=runner, wait=wait, max_iterations=max_iterations,
            sleep=sleep, poll_gap=poll_gap, reauth_backoff=reauth_backoff,
        )


def _run_loop(config, *, client: NodeClient | None = None,
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
            _recover_completed_journals()
            client = _flush_spool(client, config)  # deliver work left by a prior failure/crash
        except _AcceptedJournalBlocked as exc:
            health.status(str(exc))
            sleep(reauth_backoff)
            continue
        except OSError as exc:
            health.status(f"completed result recovery unavailable: {exc}")
            sleep(reauth_backoff)
            continue
        except _ReenrollmentRequired:
            health.status("re-enrollment required — session rejected while reporting a result")
            return count
        if any(_results_dir().glob("*.json")):
            health.status("waiting to deliver a completed result before accepting more work")
            sleep(reauth_backoff)
            continue
        try:
            accepted = _next_accepted_job()
        except _AcceptedJournalBlocked as exc:
            health.status(str(exc))
            sleep(reauth_backoff)
            continue
        if accepted is None:
            try:
                job = client.get_work(wait)
            except httpx.HTTPStatusError as exc:
                if _is_unauthorized(exc):
                    _invalidate_session(config)
                    health.status("re-enrollment required — coordinator rejected the session")
                    return count
                if exc.response.status_code in {408, 429} or exc.response.status_code >= 500:
                    health.status(f"coordinator unavailable: {exc}")
                    sleep(reauth_backoff)
                    continue
                message = (
                    "coordinator permanently rejected the work poll "
                    f"with HTTP {exc.response.status_code}"
                )
                health.status(message)
                raise CoordinatorWorkRejected(message) from exc
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
            _verify_result_storage()
        except OSError as exc:
            health.status(f"result spool unavailable; refusing work: {exc}")
            sleep(reauth_backoff)
            continue
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
        # First atomically turn the accepted journal into a completion WAL. If the separate result
        # directory disappears after execution, restart recovery reports this payload without
        # invoking the runner again.
        result_envelope = _new_result_envelope(job["id"], payload)
        completion_journaled = False
        result_spooled = False
        while not completion_journaled and not result_spooled:
            try:
                _journal_completion(accepted_path, job, result_envelope)
                completion_journaled = True
            except OSError:
                try:
                    _write_result_envelope(result_envelope)
                    result_spooled = True
                except OSError as exc:
                    health.status(f"all completion storage unavailable after execution: {exc}")
                    sleep(reauth_backoff)
        while not result_spooled:
            try:
                _write_result_envelope(result_envelope)
                result_spooled = True
            except OSError as exc:
                health.status(
                    f"result spool unavailable after completion; preserving journal: {exc}")
                sleep(reauth_backoff)
        accepted_retired = _retire_accepted(accepted_path)
        try:
            delivered, client = _try_post(client, config, job["id"], payload)
        except _ReenrollmentRequired:
            health.status("re-enrollment required — coordinator rejected the session")
            return count
        except _TerminalResultRejection as exc:
            health.status(str(exc))
            if accepted_retired:
                try:
                    _quarantine_spool(_spool_path(job["id"]))
                except OSError:
                    pass
            continue
        if delivered and accepted_retired:
            _spool_path(job["id"]).unlink(missing_ok=True)
        else:
            sleep(reauth_backoff)              # bounded backoff; the result stays spooled for retry
    return count
