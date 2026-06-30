# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""A thin httpx client of the node API — the only thing that talks to a node.

Pure and import-light (httpx only, no Django) so it stays unit-testable in isolation. Every call
adds the node's bearer token; each *node* argument just needs ``.base_url`` and ``.token`` (the
:class:`~ara.server.nodes.models.Node` row supplies both). Mirrors the node endpoints in
:mod:`ara.node.app`: read GETs, ``POST /jobs``, and the job list/poll.
"""
from __future__ import annotations

import httpx

_TIMEOUT = 30.0


def _headers(node) -> dict:
    """The bearer header for *node* (empty when no token is configured)."""
    return {"Authorization": f"Bearer {node.token}"} if node.token else {}


def _url(node, path: str) -> str:
    """Join the node's base URL with an endpoint path (tolerating a trailing slash)."""
    return node.base_url.rstrip("/") + path


def get(node, path: str, **params) -> dict | list:
    """GET *path* on *node* and return the decoded JSON (raises on a non-2xx status)."""
    resp = httpx.get(_url(node, path), headers=_headers(node),
                     params=params or None, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def status(node) -> dict | list:
    """The node's ``/status`` payload (the same shape as ``ara status --json``)."""
    return get(node, "/status")


def submit_job(node, kind: str, args: dict | None = None) -> dict:
    """Submit a job (``POST /jobs``) and return ``{"job_id": ...}``."""
    resp = httpx.post(_url(node, "/jobs"), headers=_headers(node),
                      json={"kind": kind, "args": args or {}}, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def list_jobs(node) -> dict:
    """All jobs on the node (``GET /jobs``)."""
    return get(node, "/jobs")


def get_job(node, job_id: str, wait: float = 0.0) -> dict:
    """One job by id (``GET /jobs/{id}``), optionally long-polling up to *wait* seconds."""
    return get(node, f"/jobs/{job_id}", wait=wait)
