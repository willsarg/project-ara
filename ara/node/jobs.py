# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node's async job runner — submit a kind, run it in a background thread, persist the outcome.

ORPHANED by the pull->push cutover: its only callers were the deleted inbound HTTP ``/jobs`` API
(the removed ``ara.node.app``). The push-only agent (:mod:`ara.node.agent`) runs each dispatched job
synchronously in its work loop instead, so this store is currently unused. Kept as the seam for a
future async local executor — the two-queue model's node-local store (e.g. an ``ara <verb> --detach``
that survives a dropped connection). Decide before shipping: wire it, or delete it (+ ``db.jobs``) as
dead pull-era code. See the Phase-3 note in the migration plan.

The set of runnable ``kind``s is injected (a ``{kind: worker}`` map); tests drive the runner with
trivial fake workers.
"""
from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable

from ara import db

Worker = Callable[[dict], dict]


def _thread_spawn(fn: Callable[[], None]) -> None:
    """Run *fn* on a daemon thread (the production spawn). Tests inject a synchronous spawn."""
    threading.Thread(target=fn, daemon=True).start()


class JobRunner:
    """Submits jobs and runs them in the background, persisting each outcome to the store."""

    def __init__(self, workers: dict[str, Worker], *,
                 spawn: Callable[[Callable[[], None]], None] = _thread_spawn) -> None:
        self._workers = workers
        self._spawn = spawn

    def submit(self, kind: str, args: dict) -> str:
        """Queue *kind* with *args*, return its job id. Raises ``ValueError`` for an unknown kind
        (no job row is created), so a bad request never leaves a phantom job behind."""
        if kind not in self._workers:
            raise ValueError(f"unknown job kind: {kind!r}")
        job_id = uuid.uuid4().hex
        con = db.connect()
        db.create_job(con, job_id, kind, json.dumps(args))
        con.close()
        self._spawn(lambda: self._run(job_id, kind, args))
        return job_id

    def _run(self, job_id: str, kind: str, args: dict) -> None:
        con = db.connect()                       # the worker thread owns its own connection
        db.update_job(con, job_id, status="running", started_at=db._now())
        try:
            result = self._workers[kind](args)
            db.update_job(con, job_id, status="done",
                          result_json=json.dumps(result), finished_at=db._now())
        except Exception as exc:                 # any worker failure becomes a recorded failed job
            db.update_job(con, job_id, status="failed",
                          error=f"{type(exc).__name__}: {exc}", finished_at=db._now())
        finally:
            con.close()
