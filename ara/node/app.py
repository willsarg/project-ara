# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node's FastAPI app — a thin, governed HTTP surface over ARA's verbs.

Read endpoints (`/status`, `/detect`, `/profile`, `/models`) return the same data the CLI's
``--json`` path does, via injected *providers*. Action verbs are submitted as **jobs**
(`POST /jobs` → 202 + ``job_id``) and polled (`GET /jobs/{id}`, optional ``?wait=`` long-poll), so a
long characterize never holds the connection. Every endpoint except ``/health`` requires the node's
bearer token. The runner + providers are injected so this module is engine-free and unit-testable;
the wiring to ARA's real verbs lives in :mod:`ara.node.wiring`.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable

from fastapi import Depends, FastAPI, Header, HTTPException

from ara import db
from ara.node import auth

_TERMINAL = ("done", "failed")


def _job_view(job: dict) -> dict:
    """Client-facing job shape: decode the stored ``result_json`` into a ``result`` object."""
    view = dict(job)
    raw = view.pop("result_json", None)
    view["result"] = json.loads(raw) if raw else None
    return view


def create_app(runner, providers: dict[str, Callable[[], dict]], *, version: str = "?") -> FastAPI:
    """Build the node app from a job *runner* and a map of read-endpoint *providers*."""
    app = FastAPI(title="ARA Node", version=version)

    def _require_token(authorization: str | None = Header(default=None)) -> None:
        if not auth.token_matches(authorization):
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")

    gate = Depends(_require_token)

    @app.get("/health")
    def health() -> dict:                       # open: liveness probe, leaks nothing
        return {"service": "ara-node", "status": "ok", "version": version}

    # No `-> dict` return annotation on the read endpoints: a verb's --json may be a list (e.g.
    # `ara models --json` is an array), and FastAPI would 500 trying to validate a list against dict.
    # `response_model=None` keeps it a transparent pass-through of whatever the verb emitted.
    @app.get("/status", response_model=None)
    def status(_: None = gate):
        return providers["status"]()

    @app.get("/detect", response_model=None)
    def detect(_: None = gate):
        return providers["detect"]()

    @app.get("/profile", response_model=None)
    def profile(_: None = gate):
        return providers["profile"]()

    @app.get("/models", response_model=None)
    def models(_: None = gate):
        return providers["models"]()

    @app.post("/jobs", status_code=202)
    def submit_job(body: dict, _: None = gate) -> dict:
        try:
            job_id = runner.submit(body.get("kind"), body.get("args") or {})
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"job_id": job_id}

    @app.get("/jobs")
    def list_all(_: None = gate) -> dict:
        con = db.connect()
        rows = db.list_jobs(con)
        con.close()
        return {"jobs": [_job_view(r) for r in rows]}

    @app.get("/jobs/{job_id}")
    def get_one(job_id: str, wait: float = 0.0, _: None = gate) -> dict:
        deadline = time.monotonic() + wait      # ``?wait=N`` long-polls up to N seconds
        while True:
            con = db.connect()
            job = db.get_job(con, job_id)
            con.close()
            if job is None:
                raise HTTPException(status_code=404, detail="no such job")
            if job["status"] in _TERMINAL or time.monotonic() >= deadline:
                return _job_view(job)
            time.sleep(0.05)

    return app
