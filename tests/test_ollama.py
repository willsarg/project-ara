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


def test_endpoint_authority_normalizes_and_classifies_loopback():
    assert ollama.endpoint_authority("HTTP://LOCALHOST:11434/") == ollama.OllamaEndpoint(
        url="http://localhost:11434", scope="loopback")


def test_endpoint_authority_distinguishes_remote_and_cloud():
    assert ollama.endpoint_authority("http://192.168.1.20:11434").scope == "remote"
    assert ollama.endpoint_authority("https://ollama.com/api/") == ollama.OllamaEndpoint(
        url="https://ollama.com/api", scope="cloud")


def test_endpoint_authority_handles_mixed_case_scheme_and_ipv6_loopback():
    assert ollama.endpoint_authority("Http://[::1]:11434/") == ollama.OllamaEndpoint(
        url="http://[::1]:11434", scope="loopback")


def test_endpoint_authority_fails_closed_on_ambiguous_urls():
    for value in (
        "ftp://localhost:11434",
        "http://user:secret@localhost:11434",
        "http://localhost:bad-port",
        "http://localhost:11434?target=remote",
        "http://localhost:11434#fragment",
        "",
    ):
        assert ollama.endpoint_authority(value) == ollama.OllamaEndpoint(
            url=None, scope="unknown")


def test_endpoint_authority_does_not_trust_loopback_lookalikes():
    assert ollama.endpoint_authority("http://127.0.0.2:11434").scope == "loopback"
    assert ollama.endpoint_authority("http://localhost.example:11434").scope == "remote"


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
# inventory / tags — one structured /api/tags snapshot
# --------------------------------------------------------------------------- #
def test_inventory_parses_current_model_shape(monkeypatch):
    digest = "a" * 64
    row = {
        "name": "custom:latest",
        "model": "custom:latest",
        "size": 522_653_783,
        "digest": digest,
        "details": {
            "parent_model": "qwen3:0.6b",
            "format": "gguf",
            "family": "qwen3",
            "families": ["qwen3"],
            "parameter_size": "751.63M",
            "quantization_level": "Q4_K_M",
            "context_length": 40_960,
            "embedding_length": 1_024,
        },
        "capabilities": ["completion", "tools", "thinking"],
    }
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: {"models": [row]})

    models = ollama.inventory()

    assert models == [ollama.OllamaModel(
        name="custom:latest",
        model="custom:latest",
        digest=digest,
        size_bytes=522_653_783,
        parent_model="qwen3:0.6b",
        format="gguf",
        family="qwen3",
        families=("qwen3",),
        parameter_size="751.63M",
        quantization="Q4_K_M",
        context_length=40_960,
        embedding_length=1_024,
        capabilities=("completion", "tools", "thinking"),
    )]
    assert ollama.OllamaModel.__dataclass_params__.frozen is True


def test_inventory_accepts_older_sparse_model_shape(monkeypatch):
    digest = "b" * 64
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: {"models": [{
        "name": "qwen3:0.6b",
        "size": 522_653_767,
        "digest": digest,
        "details": {
            "format": "gguf",
            "family": "qwen3",
            "families": ["qwen3"],
            "parameter_size": "751.63M",
            "quantization_level": "Q4_K_M",
        },
    }]})

    model = ollama.inventory()[0]

    assert model.name == "qwen3:0.6b"
    assert model.model is None
    assert model.context_length is None
    assert model.embedding_length is None
    assert model.capabilities == ()


def test_inventory_parses_remote_model_identity_as_cloud(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: {"models": [{
        "name": "qwen3.5:cloud",
        "remote_model": "qwen3.5",
        "remote_host": "https://ollama.com",
        "capabilities": ["completion"],
    }]})

    model = ollama.inventory()[0]

    assert model.remote_model == "qwen3.5"
    assert model.remote_host == "https://ollama.com"
    assert model.scope == "cloud"


def test_inventory_does_not_misclassify_malformed_remote_metadata_as_local(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: {"models": [{
        "name": "ambiguous",
        "remote_model": 7,
    }]})

    model = ollama.inventory()[0]

    assert model.remote_model is None
    assert model.scope == "unknown"


def test_inventory_treats_malformed_optional_fields_as_unknown(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: {"models": [
        {"name": "valid", "model": 7, "size": True, "digest": "bad",
         "details": {"family": 2, "families": ["qwen", 3],
                     "context_length": True, "embedding_length": -1},
         "capabilities": ["completion", 4]},
        {},
        {"name": 5},
        "bad",
    ]})

    models = ollama.inventory()

    assert len(models) == 1
    model = models[0]
    assert model.model is None
    assert model.digest is None
    assert model.size_bytes is None
    assert model.family is None
    assert model.families == ("qwen",)
    assert model.context_length is None
    assert model.embedding_length is None
    assert model.capabilities == ("completion",)


def test_inventory_none_when_unreachable_or_models_not_list(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: None)
    assert ollama.inventory() is None
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: {"models": "nope"})
    assert ollama.inventory() is None


def test_find_model_matches_exact_and_implicit_latest():
    model = ollama.OllamaModel(name="base:latest")
    tagged = ollama.OllamaModel(name="base:other")
    assert model.aliases == ("base",)
    assert tagged.aliases == ()
    assert ollama.find_model([model], "base") is model
    assert ollama.find_model([model], "base:latest") is model
    assert ollama.find_model([model], "base:other") is None


def test_find_model_treats_registry_port_as_part_of_name():
    model = ollama.OllamaModel(name="registry.local:5000/org/base:latest")
    assert ollama.find_model([model], "registry.local:5000/org/base") is model


def test_initial_governed_model_support_requires_local_gguf_completion():
    supported = ollama.OllamaModel(
        name="local", format="gguf", capabilities=("completion",))
    assert ollama.initial_governed_model_error(supported) is None

    cloud = ollama.OllamaModel(
        name="cloud", format="gguf", capabilities=("completion",), scope="cloud")
    assert "cloud model" in ollama.initial_governed_model_error(cloud)

    ambiguous = ollama.OllamaModel(
        name="ambiguous", format="gguf", capabilities=("completion",), scope="unknown")
    assert "ambiguous" in ollama.initial_governed_model_error(ambiguous)

    non_gguf = ollama.OllamaModel(
        name="native", format="safetensors", capabilities=("completion",))
    assert "requires local GGUF" in ollama.initial_governed_model_error(non_gguf)

    no_completion = ollama.OllamaModel(name="embed", format="gguf", capabilities=("embedding",))
    assert "completion" in ollama.initial_governed_model_error(no_completion)


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


def test_manifest_digest_matches_exact_and_implicit_latest(monkeypatch):
    digest = "a" * 64
    monkeypatch.setattr(
        ollama, "_get_json",
        lambda p, t: {"models": [{"name": "base:latest", "digest": digest}]},
    )
    assert ollama.manifest_digest("base") == digest
    assert ollama.manifest_digest("base:latest") == digest


def test_manifest_digest_treats_registry_port_as_part_of_name(monkeypatch):
    digest = "b" * 64
    monkeypatch.setattr(
        ollama, "_get_json",
        lambda p, t: {"models": [
            {"name": "registry.local:5000/org/base:latest", "digest": digest},
        ]},
    )
    assert ollama.manifest_digest("registry.local:5000/org/base") == digest


def test_manifest_digest_fails_closed_on_malformed_inventory(monkeypatch):
    monkeypatch.setattr(
        ollama, "_get_json",
        lambda p, t: {"models": [
            {"name": "missing"},
            {"name": "short", "digest": "abc"},
            {"name": "upper", "digest": "A" * 64},
            {"name": 7, "digest": "b" * 64},
            "bad",
        ]},
    )
    assert ollama.manifest_digest("missing") is None
    assert ollama.manifest_digest("short") is None
    assert ollama.manifest_digest("upper") is None
    assert ollama.manifest_digest("absent") is None


def test_manifest_digest_none_when_inventory_unreachable(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: None)
    assert ollama.manifest_digest("base") is None


# --------------------------------------------------------------------------- #
# processes / ps — typed safety snapshot plus raw compatibility view
# --------------------------------------------------------------------------- #
def test_processes_parses_current_shape(monkeypatch):
    digest = "c" * 64
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: {"models": [{
        "name": "qwen3:0.6b",
        "model": "qwen3:0.6b",
        "digest": digest,
        "size": 600_000_000,
        "size_vram": 500_000_000,
        "context_length": 8192,
        "expires_at": "2026-07-17T12:00:00Z",
    }]})

    assert ollama.processes() == [ollama.OllamaProcess(
        name="qwen3:0.6b",
        model="qwen3:0.6b",
        digest=digest,
        size_bytes=600_000_000,
        size_vram_bytes=500_000_000,
        context_length=8192,
        expires_at="2026-07-17T12:00:00Z",
    )]


def test_processes_keeps_malformed_optional_fields_unknown(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: {"models": [
        {"name": "valid", "model": 4, "digest": "bad", "size": True,
         "size_vram": -1, "context_length": 0, "expires_at": 7},
        {"name": ""}, {"name": 5}, "bad",
    ]})

    assert ollama.processes() == [ollama.OllamaProcess(name="valid")]


def test_processes_none_when_unreachable_or_root_malformed(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: None)
    assert ollama.processes() is None
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: {"models": {}})
    assert ollama.processes() is None


def test_processes_defensively_drops_non_dict_ps_rows(monkeypatch):
    monkeypatch.setattr(ollama, "ps", lambda _timeout: ["bad", {"name": "valid"}])
    assert ollama.processes() == [ollama.OllamaProcess(name="valid")]


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
    assert ollama.load("x-ara") == {"done": True}
    assert seen["path"] == "/api/generate"
    assert seen["payload"]["model"] == "x-ara"
    assert seen["payload"]["keep_alive"] == -1
    assert seen["payload"]["prompt"] == ""


def test_load_can_defer_keepalive_to_daemon_policy(monkeypatch):
    seen = {}

    def fake_post(path, payload, timeout):
        seen.update(path=path, payload=payload)
        return {"done": True}

    monkeypatch.setattr(ollama, "_post_json", fake_post)
    assert ollama.load("x-ara", keep_alive=None) == {"done": True}
    assert seen["path"] == "/api/generate"
    assert seen["payload"] == {"model": "x-ara", "prompt": "", "stream": False}


# --------------------------------------------------------------------------- #
# pull — fetch a missing model (serve's get-out-of-the-way step)
# Spec 2026-07-04-ara-serve-one-command-estimated-ceiling.
# --------------------------------------------------------------------------- #
def test_pull_success_sends_model_and_returns_true(monkeypatch):
    seen = {}

    def fake_post(path, payload, timeout):
        seen.update(path=path, payload=payload)
        return {"status": "success"}

    monkeypatch.setattr(ollama, "_post_json", fake_post)
    assert ollama.pull("qwen3:0.6b") is True
    assert seen["path"] == "/api/pull"
    assert seen["payload"] == {"model": "qwen3:0.6b", "stream": False}


def test_pull_false_when_unreachable(monkeypatch):
    monkeypatch.setattr(ollama, "_post_json", lambda p, pl, t: None)
    assert ollama.pull("qwen3:0.6b") is False


def test_pull_false_on_non_success_status(monkeypatch):
    monkeypatch.setattr(ollama, "_post_json", lambda p, pl, t: {"status": "error"})
    assert ollama.pull("qwen3:0.6b") is False


# --------------------------------------------------------------------------- #
# show — architecture detail for the engine-free estimated ceiling
# --------------------------------------------------------------------------- #
def test_show_returns_detail_dict(monkeypatch):
    seen = {}

    def fake_post(path, payload, timeout):
        seen.update(path=path, payload=payload)
        return {"model_info": {"general.architecture": "qwen3"}}

    monkeypatch.setattr(ollama, "_post_json", fake_post)
    assert ollama.show("qwen3:0.6b") == {"model_info": {"general.architecture": "qwen3"}}
    assert seen["path"] == "/api/show"
    assert seen["payload"] == {"model": "qwen3:0.6b"}


def test_show_none_when_unreachable(monkeypatch):
    monkeypatch.setattr(ollama, "_post_json", lambda p, pl, t: None)
    assert ollama.show("qwen3:0.6b") is None


# --------------------------------------------------------------------------- #
# size_bytes — on-disk weights footprint proxy from /api/tags
# --------------------------------------------------------------------------- #
def test_size_bytes_finds_matching_entry(monkeypatch):
    rows = [{"name": "other:1", "size": 111}, {"name": "qwen3:0.6b", "size": 522_000_000}]
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: {"models": rows})
    assert ollama.size_bytes("qwen3:0.6b") == 522_000_000


def test_size_bytes_none_when_absent(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: {"models": [{"name": "other:1", "size": 1}]})
    assert ollama.size_bytes("qwen3:0.6b") is None


def test_size_bytes_none_when_unreachable(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json", lambda p, t: None)
    assert ollama.size_bytes("qwen3:0.6b") is None


def test_size_bytes_none_when_size_not_int(monkeypatch):
    monkeypatch.setattr(ollama, "_get_json",
                        lambda p, t: {"models": [{"name": "qwen3:0.6b", "size": "big"}]})
    assert ollama.size_bytes("qwen3:0.6b") is None


# --------------------------------------------------------------------------- #
# delete — DELETE /api/delete (probe-model cleanup for the characterize ramp)
# --------------------------------------------------------------------------- #
class _StatusResp:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_delete_true_on_2xx(monkeypatch):
    captured = {}

    def fake(req, timeout):
        captured["method"] = req.get_method()
        captured["url"] = req.full_url
        return _StatusResp(200)

    monkeypatch.setattr(ollama.urllib.request, "urlopen", fake)
    assert ollama.delete("m-ara-probe") is True
    assert captured["method"] == "DELETE"
    assert captured["url"].endswith("/api/delete")


def test_delete_false_on_error(monkeypatch):
    def boom(req, timeout):
        raise ollama.urllib.error.URLError("refused")
    monkeypatch.setattr(ollama.urllib.request, "urlopen", boom)
    assert ollama.delete("m-ara-probe") is False


def test_delete_false_on_non_2xx(monkeypatch):
    monkeypatch.setattr(ollama.urllib.request, "urlopen",
                        lambda req, timeout: _StatusResp(404))
    assert ollama.delete("missing") is False
