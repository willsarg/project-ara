# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node HTTP client — bearer auth, endpoint shapes, and the 204-means-no-work rule.

httpx is mocked with an injected stub client, so the whole surface is exercised without a socket.
"""
from __future__ import annotations

import httpx
import pytest

from ara.node import client as client_mod
from ara.node.client import NodeClient


class FakeResponse:
    def __init__(self, *, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.raised = False

    def json(self):
        return self._payload

    def raise_for_status(self):
        self.raised = True


class FakeHTTPClient:
    """Records requests and replays a queued/looked-up response."""

    def __init__(self, response=None):
        self.response = response or FakeResponse()
        self.calls = []
        self.closed = False

    def post(self, url, json=None, headers=None):
        self.calls.append(("POST", url, json, headers))
        return self.response

    def get(self, url, params=None, headers=None):
        self.calls.append(("GET", url, params, headers))
        return self.response

    def close(self):
        self.closed = True


def _client(response=None):
    fake = FakeHTTPClient(response)
    return NodeClient("https://c.example/", "TOK", client=fake), fake


def test_default_client_is_httpx_and_base_is_trimmed(monkeypatch):
    made = {}
    monkeypatch.setattr(httpx, "Client", lambda timeout: made.setdefault("c", object()))
    nc = NodeClient("https://c.example/", "TOK")           # no injected client → builds httpx.Client
    assert nc._client is made["c"]
    assert nc._base == "https://c.example"                 # trailing slash trimmed


def test_client_refuses_remote_cleartext_even_for_loaded_config():
    with pytest.raises(ValueError, match="insecure coordinator URL"):
        NodeClient("http://coordinator.example", "SECRET", client=FakeHTTPClient())


@pytest.mark.parametrize("token", [None, "", 7])
def test_client_requires_a_nonempty_string_token(token):
    with pytest.raises(ValueError, match="token is missing"):
        NodeClient("https://c.example", token, client=FakeHTTPClient())


def test_enroll_posts_self_description_with_bearer():
    nc, fake = _client(FakeResponse(payload={"enrollment_id": "e1", "status": "pending"}))
    out = nc.enroll({"machine_key": "m"})
    assert out == {"enrollment_id": "e1", "status": "pending"}
    method, url, body, headers = fake.calls[0]
    assert method == "POST" and url == "https://c.example/api/enroll"
    assert body == {"machine_key": "m"} and headers == {"Authorization": "Bearer TOK"}
    assert fake.response.raised


def test_poll_approval_gets_enrollment_state():
    nc, fake = _client(FakeResponse(payload={"status": "active", "session_token": "S"}))
    out = nc.poll_approval("e1")
    assert out["session_token"] == "S"
    assert fake.calls[0][:2] == ("GET", "https://c.example/api/enroll/e1")


def test_poll_approval_percent_encodes_wire_identifier():
    nc, fake = _client(FakeResponse(payload={"status": "pending"}))
    nc.poll_approval("../other enrollment")
    assert fake.calls[0][1] == "https://c.example/api/enroll/..%2Fother%20enrollment"


def test_get_work_returns_job_on_200():
    job = {"id": "j1", "kind": "run", "args": {}}
    nc, fake = _client(FakeResponse(status_code=200, payload={"job": job}))
    assert nc.get_work(20) == job
    method, url, params, _ = fake.calls[0]
    assert method == "GET" and url == "https://c.example/api/work" and params == {"wait": 20}


@pytest.mark.parametrize("payload", [
    None, {}, {"job": None}, {"job": []},
    {"job": {"id": "", "kind": "run", "args": {}}},
    {"job": {"id": "j", "kind": "", "args": {}}},
    {"job": {"id": "j", "kind": "run", "args": []}},
    {"job": {"id": "j", "kind": "serve", "args": {}}},
    {"job": {"id": "j", "kind": "profile", "args": {}}},
])
def test_get_work_rejects_malformed_wire_jobs(payload):
    nc, _fake = _client(FakeResponse(status_code=200, payload=payload))
    with pytest.raises(ValueError, match="invalid work response"):
        nc.get_work(20)


def test_get_work_returns_none_on_204():
    nc, fake = _client(FakeResponse(status_code=204))
    assert nc.get_work(5) is None
    assert fake.response.raised is False                    # 204 short-circuits before raise_for_status


def test_post_result_posts_payload():
    nc, fake = _client()
    nc.post_result("j1", {"status": "done", "result": {}, "environment": {}})
    method, url, body, headers = fake.calls[0]
    assert method == "POST" and url == "https://c.example/api/work/j1/result"
    assert body["status"] == "done" and headers == {"Authorization": "Bearer TOK"}


def test_post_result_percent_encodes_wire_job_id():
    nc, fake = _client()
    nc.post_result("../../outside job", {"status": "done"})
    assert fake.calls[0][1] == (
        "https://c.example/api/work/..%2F..%2Foutside%20job/result")


def test_close_releases_the_pool():
    nc, fake = _client()
    nc.close()
    assert fake.closed is True
