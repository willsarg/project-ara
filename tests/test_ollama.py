# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Thin Ollama client — liveness probe. Spec 2026-06-26-detect-ollama-liveness."""
from ara import ollama


# --------------------------------------------------------------------------- #
# base_url — honors OLLAMA_HOST in its several shapes
# --------------------------------------------------------------------------- #
def test_base_url_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert ollama.base_url() == "http://127.0.0.1:11434"


def test_base_url_host_port_gets_scheme(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.5:11434")
    assert ollama.base_url() == "http://10.0.0.5:11434"


def test_base_url_with_scheme_preserved_and_trimmed(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://box.local:1234/")
    assert ollama.base_url() == "http://box.local:1234"


def test_base_url_bare_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "myhost")
    assert ollama.base_url() == "http://myhost"


# --------------------------------------------------------------------------- #
# _get_json — the single urllib seam
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_get_json_success(monkeypatch):
    monkeypatch.setattr(ollama.urllib.request, "urlopen",
                        lambda url, timeout: _FakeResp(b'{"version": "0.30.10"}'))
    assert ollama._get_json("/api/version", 0.5) == {"version": "0.30.10"}


def test_get_json_none_on_urlerror(monkeypatch):
    def boom(url, timeout):
        raise ollama.urllib.error.URLError("connection refused")
    monkeypatch.setattr(ollama.urllib.request, "urlopen", boom)
    assert ollama._get_json("/api/version", 0.5) is None


def test_get_json_none_on_bad_json(monkeypatch):
    monkeypatch.setattr(ollama.urllib.request, "urlopen",
                        lambda url, timeout: _FakeResp(b'not json'))
    assert ollama._get_json("/api/version", 0.5) is None


def test_get_json_none_on_non_object(monkeypatch):
    monkeypatch.setattr(ollama.urllib.request, "urlopen",
                        lambda url, timeout: _FakeResp(b'[1, 2, 3]'))
    assert ollama._get_json("/api/version", 0.5) is None


# --------------------------------------------------------------------------- #
# version — liveness; None means "not serving / unreachable"
# --------------------------------------------------------------------------- #
def test_version_returns_string(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json", lambda path, timeout: {"version": "0.30.10"})
    assert ollama.version() == "0.30.10"


def test_version_none_when_unreachable(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json", lambda path, timeout: None)
    assert ollama.version() is None


def test_version_none_when_version_not_a_string(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json", lambda path, timeout: {"version": 123})
    assert ollama.version() is None
