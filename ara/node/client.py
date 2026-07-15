# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node's HTTP client to the coordinator — the one network boundary of the push-only loop.

Every call is node → server, bearer-authed (the enrollment token for enroll, the session token for
work). Wraps ``httpx`` so the rest of the node code (enroll, agent) speaks in dicts, never in HTTP.
The underlying ``httpx.Client`` is injectable so tests drive the whole flow with a stub and never
touch a real socket. Endpoints (pinned by ``contracts/wire``):

- ``POST /api/enroll``            → ``{enrollment_id, status}``
- ``GET  /api/enroll/{id}``       → ``pending`` | ``active`` + ``session_token``
- ``GET  /api/work?wait=N``       → 200 job | 204 (no work in the window)
- ``POST /api/work/{id}/result``  → the run's outcome
"""
from __future__ import annotations

from urllib.parse import quote

import httpx

from ara.node import config as config_mod


class NodeClient:
    """A thin, bearer-authed httpx wrapper over the coordinator's push-only endpoints."""

    def __init__(self, server_url: str, token: str, *, client: httpx.Client | None = None,
                 timeout: float = 30.0) -> None:
        config_mod.require_secure_url(server_url)
        if not isinstance(token, str) or not token:
            raise ValueError("node authentication token is missing")
        self._base = server_url.rstrip("/")
        self._token = token
        self._client = client or httpx.Client(timeout=timeout)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token}"}

    def enroll(self, self_desc: dict) -> dict:
        """POST the self-description; returns ``{enrollment_id, status}`` (lands PENDING)."""
        resp = self._client.post(f"{self._base}/api/enroll", json=self_desc, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def poll_approval(self, enrollment_id: str) -> dict:
        """GET the enrollment's state: ``pending``, or ``active`` carrying the ``session_token``."""
        encoded = quote(enrollment_id, safe="")
        resp = self._client.get(f"{self._base}/api/enroll/{encoded}", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def get_work(self, wait: float) -> dict | None:
        """Long-poll for a job. Returns the job dict, or None on 204 (no work in the window)."""
        resp = self._client.get(f"{self._base}/api/work", params={"wait": wait},
                                headers=self._headers())
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        payload = resp.json()
        job = payload.get("job") if isinstance(payload, dict) else None
        if (not isinstance(job, dict)
                or not isinstance(job.get("id"), str) or not job["id"]
                or not isinstance(job.get("kind"), str) or not job["kind"]
                or not isinstance(job.get("args"), dict)):
            raise ValueError("invalid work response from coordinator")
        return job

    def post_result(self, job_id: str, payload: dict) -> None:
        """POST the run's outcome (``result.request`` shape) for a dispatched job."""
        encoded = quote(job_id, safe="")
        resp = self._client.post(f"{self._base}/api/work/{encoded}/result", json=payload,
                                 headers=self._headers())
        resp.raise_for_status()

    def close(self) -> None:
        """Release the underlying httpx connection pool."""
        self._client.close()
