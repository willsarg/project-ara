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


# --------------------------------------------------------------------------- #
# _post_json — the POST seam (serve tier)
# --------------------------------------------------------------------------- #
def test_post_json_success_sends_url_and_body(monkeypatch):
    captured = {}

    def fake(req, timeout):
        captured["url"] = req.full_url
        captured["body"] = req.data
        captured["method"] = req.get_method()
        return _FakeResp(b'{"status": "success"}')

    monkeypatch.setattr(ollama.urllib.request, "urlopen", fake)
    assert ollama._post_json("/api/create", {"model": "x"}, 5.0) == {"status": "success"}
    assert captured["url"] == "http://127.0.0.1:11434/api/create"
    assert captured["method"] == "POST"
    assert b'"model"' in captured["body"]


def test_post_json_none_on_urlerror(monkeypatch):
    def boom(req, timeout):
        raise ollama.urllib.error.URLError("refused")
    monkeypatch.setattr(ollama.urllib.request, "urlopen", boom)
    assert ollama._post_json("/api/create", {}, 5.0) is None


def test_post_json_none_on_bad_json(monkeypatch):
    monkeypatch.setattr(ollama.urllib.request, "urlopen",
                        lambda req, timeout: _FakeResp(b'not json'))
    assert ollama._post_json("/api/create", {}, 5.0) is None


def test_post_json_none_on_non_object(monkeypatch):
    monkeypatch.setattr(ollama.urllib.request, "urlopen",
                        lambda req, timeout: _FakeResp(b'"ok"'))
    assert ollama._post_json("/api/create", {}, 5.0) is None


# --------------------------------------------------------------------------- #
# tags — installed model names
# --------------------------------------------------------------------------- #
def test_tags_returns_names(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json",
                        lambda p, t: {"models": [{"name": "a:1"}, {"name": "b:2"}]})
    assert ollama.tags() == ["a:1", "b:2"]


def test_tags_none_when_unreachable(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: None)
    assert ollama.tags() is None


def test_tags_none_when_models_not_a_list(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: {"models": "nope"})
    assert ollama.tags() is None


def test_tags_skips_malformed_entries(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json",
                        lambda p, t: {"models": [{"name": "a"}, {}, {"name": 5}, "x"]})
    assert ollama.tags() == ["a"]


# --------------------------------------------------------------------------- #
# ps — loaded models
# --------------------------------------------------------------------------- #
def test_ps_returns_dicts(monkeypatch):
    rows = [{"name": "a", "context_length": 8192, "size": 10, "size_vram": 10}]
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: {"models": rows})
    assert ollama.ps() == rows


def test_ps_none_when_unreachable(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: None)
    assert ollama.ps() is None


def test_ps_none_when_models_not_a_list(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: {"models": {}})
    assert ollama.ps() is None


def test_ps_drops_non_dict_entries(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: {"models": [{"name": "a"}, "x", 3]})
    assert ollama.ps() == [{"name": "a"}]


# --------------------------------------------------------------------------- #
# create — derived model with num_ctx baked in
# --------------------------------------------------------------------------- #
def test_create_true_on_success(monkeypatch):
    seen = {}

    def fake_post(path, payload, timeout):
        seen.update(path=path, payload=payload)
        return {"status": "success"}

    monkeypatch.setattr(ollama, "_post_json", fake_post)
    assert ollama.create("x-ara", "x", 8192) is True
    assert seen["path"] == "/api/create"
    assert seen["payload"]["from"] == "x"
    assert seen["payload"]["parameters"]["num_ctx"] == 8192


def test_create_false_when_unreachable(monkeypatch):
    monkeypatch.setattr(ollama, "_post_json", lambda p, pl, t: None)
    assert ollama.create("x-ara", "x", 8192) is False


def test_create_false_on_non_success_status(monkeypatch):
    monkeypatch.setattr(ollama, "_post_json", lambda p, pl, t: {"status": "error"})
    assert ollama.create("x-ara", "x", 8192) is False


# --------------------------------------------------------------------------- #
# load — warm-load + hold
# --------------------------------------------------------------------------- #
def test_load_generates_empty_prompt_with_keepalive(monkeypatch):
    seen = {}

    def fake_post(path, payload, timeout):
        seen.update(path=path, payload=payload)
        return {"done": True}

    monkeypatch.setattr(ollama, "_post_json", fake_post)
    assert ollama.load("x-ara", keep_alive=-1) == {"done": True}
    assert seen["path"] == "/api/generate"
    assert seen["payload"]["model"] == "x-ara"
    assert seen["payload"]["keep_alive"] == -1
    assert seen["payload"]["prompt"] == ""
