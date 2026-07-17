# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Thin Ollama client — liveness probe. Spec 2026-06-26-detect-ollama-liveness."""
from types import SimpleNamespace

import pytest

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
# local runtime authority — direct listener + exact daemon configuration
# --------------------------------------------------------------------------- #
def _listener(*, pid=42, host="127.0.0.1", name="ollama", executable="/usr/bin/ollama",
              command=("ollama", "serve"), environment=(), environment_readable=True):
    return ollama.OllamaListener(
        pid=pid,
        create_time=1234.5,
        bind_host=host,
        process_name=name,
        executable=executable,
        command=command,
        configured_inputs=environment,
        environment_readable=environment_readable,
    )


def test_runtime_authority_attests_direct_listener_and_exact_version_default(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "9")  # the caller shell is not daemon evidence
    monkeypatch.setattr(ollama, "_local_tcp_listeners", lambda port: [_listener()])
    monkeypatch.setattr(ollama, "version", lambda timeout=0.5: "0.30.10")

    authority = ollama.runtime_authority(
        ollama.OllamaEndpoint("http://127.0.0.1:11434", "loopback"))

    assert authority.server_instance_id == "42:1234.500000:/usr/bin/ollama"
    assert authority.server_version == "0.30.10"
    assert authority.configured_num_parallel == 1
    assert authority.configured_num_parallel_authority == "exact_version_default"
    assert authority.issue is None


def test_runtime_authority_prefers_observed_process_input_over_version_default(monkeypatch):
    listener = _listener(
        environment=(("OLLAMA_NUM_PARALLEL", "2"), ("OLLAMA_KV_CACHE_TYPE", "q8_0")))
    monkeypatch.setattr(ollama, "_local_tcp_listeners", lambda port: [listener])
    monkeypatch.setattr(ollama, "version", lambda timeout=0.5: "99.0.0")

    authority = ollama.runtime_authority(
        ollama.OllamaEndpoint("http://localhost:11434", "loopback"))

    assert authority.configured_inputs == listener.configured_inputs
    assert authority.configured_num_parallel == 2
    assert authority.configured_num_parallel_authority == "process_environment"
    assert authority.issue == "parallelism_not_one"


@pytest.mark.parametrize("version", ["0.8.0", "0.9.7-dev", "0.32.2", "not-a-version"])
def test_runtime_authority_fails_closed_when_unset_default_is_not_verified(
        monkeypatch, version):
    monkeypatch.setattr(ollama, "_local_tcp_listeners", lambda port: [_listener()])
    monkeypatch.setattr(ollama, "version", lambda timeout=0.5: version)

    authority = ollama.runtime_authority(
        ollama.OllamaEndpoint("http://127.0.0.1:11434", "loopback"))

    assert authority.configured_num_parallel is None
    assert authority.configured_num_parallel_authority is None
    assert authority.issue == "parallelism_unknown"


@pytest.mark.parametrize("raw", ["", "invalid", "-1", "1.5"])
def test_runtime_authority_fails_closed_on_invalid_explicit_parallelism(monkeypatch, raw):
    listener = _listener(environment=(("OLLAMA_NUM_PARALLEL", raw),))
    monkeypatch.setattr(ollama, "_local_tcp_listeners", lambda port: [listener])
    monkeypatch.setattr(ollama, "version", lambda timeout=0.5: "0.30.10")

    authority = ollama.runtime_authority(
        ollama.OllamaEndpoint("http://127.0.0.1:11434", "loopback"))

    assert authority.configured_num_parallel is None
    assert authority.issue == "parallelism_unknown"


def test_runtime_authority_rejects_proxy_ambiguous_and_wildcard_listeners(monkeypatch):
    monkeypatch.setattr(ollama, "version", lambda *_a, **_k: pytest.fail("contacted non-Ollama"))
    endpoint = ollama.OllamaEndpoint("http://127.0.0.1:11434", "loopback")

    for listeners, issue in (
        ([_listener(name="caddy", executable="/usr/bin/caddy", command=("caddy",))],
         "listener_not_ollama"),
        ([_listener(), _listener(pid=43)], "listener_ambiguous"),
        ([_listener(host="0.0.0.0")], "listener_unattributed"),
        ([], "listener_unattributed"),
    ):
        monkeypatch.setattr(ollama, "_local_tcp_listeners", lambda port, rows=listeners: rows)
        assert ollama.runtime_authority(endpoint).issue == issue


def test_runtime_authority_reports_inaccessible_process_environment(monkeypatch):
    monkeypatch.setattr(
        ollama, "_local_tcp_listeners", lambda port: [_listener(environment_readable=False)])
    monkeypatch.setattr(ollama, "version", lambda timeout=0.5: "0.30.10")

    authority = ollama.runtime_authority(
        ollama.OllamaEndpoint("http://127.0.0.1:11434", "loopback"))

    assert authority.server_instance_id is not None
    assert authority.issue == "process_environment_unavailable"


def test_runtime_authority_rejects_non_loopback_or_unreachable_server(monkeypatch):
    monkeypatch.setattr(
        ollama, "_local_tcp_listeners", lambda port: pytest.fail("inspected remote processes"))
    assert ollama.runtime_authority(
        ollama.OllamaEndpoint("http://box.local:11434", "remote")).issue == "endpoint_not_loopback"

    monkeypatch.setattr(ollama, "_local_tcp_listeners", lambda port: [_listener()])
    monkeypatch.setattr(ollama, "version", lambda timeout=0.5: None)
    assert ollama.runtime_authority(
        ollama.OllamaEndpoint("http://127.0.0.1:11434", "loopback")).issue == "server_unreachable"


def test_local_tcp_listeners_isolates_process_failures_and_reads_whitelisted_config(monkeypatch):
    listening = SimpleNamespace(
        status=ollama.psutil.CONN_LISTEN,
        laddr=SimpleNamespace(ip="127.0.0.1", port=11434),
    )
    wrong_status = SimpleNamespace(status="ESTABLISHED", laddr=listening.laddr)
    wrong_port = SimpleNamespace(
        status=ollama.psutil.CONN_LISTEN,
        laddr=SimpleNamespace(ip="127.0.0.1", port=9999),
    )

    class Process:
        def __init__(self, pid, connections, *, fail_connections=False, fail_identity=False,
                     fail_environment=False):
            self.pid = pid
            self._connections = connections
            self.fail_connections = fail_connections
            self.fail_identity = fail_identity
            self.fail_environment = fail_environment

        def net_connections(self, kind):
            assert kind == "tcp"
            if self.fail_connections:
                raise ollama.psutil.AccessDenied(self.pid)
            return self._connections

        def name(self):
            if self.fail_identity:
                raise ollama.psutil.NoSuchProcess(self.pid)
            return "ollama"

        def exe(self):
            return "/opt/ollama"

        def cmdline(self):
            return ["ollama", "serve"]

        def create_time(self):
            return 10.25

        def environ(self):
            if self.fail_environment:
                raise ollama.psutil.AccessDenied(self.pid)
            return {
                "OLLAMA_NUM_PARALLEL": "1",
                "OLLAMA_KV_CACHE_TYPE": "q8_0",
                "UNRELATED_SECRET": "not captured",
                3: "ignored",
                "OLLAMA_CONTEXT_LENGTH": 7,
            }

    class LegacyProcess(Process):
        net_connections = None

        def connections(self, kind):
            assert kind == "tcp"
            return self._connections

    processes = [
        Process(1, [], fail_connections=True),
        Process(2, [wrong_status, wrong_port]),
        Process(3, [listening], fail_identity=True),
        Process(4, [listening], fail_environment=True),
        Process(5, [listening]),
        LegacyProcess(6, [listening]),
    ]
    monkeypatch.setattr(ollama.psutil, "process_iter", lambda: processes)

    listeners = ollama._local_tcp_listeners(11434)

    assert [listener.pid for listener in listeners] == [4, 5, 6]
    assert listeners[0].environment_readable is False
    assert listeners[1].configured_inputs == (
        ("OLLAMA_NUM_PARALLEL", "1"), ("OLLAMA_KV_CACHE_TYPE", "q8_0"))


def test_configured_inputs_normalizes_keys_only_for_windows_semantics():
    environment = {"ollama_num_parallel": "1", "OLLAMA_VULKAN": "true"}
    assert ollama._configured_inputs(environment, case_insensitive=False) == (
        ("OLLAMA_VULKAN", "true"),)
    assert ollama._configured_inputs(environment, case_insensitive=True) == (
        ("OLLAMA_NUM_PARALLEL", "1"), ("OLLAMA_VULKAN", "true"))


def test_listener_identity_fallbacks_and_non_ip_bind():
    command_only = _listener(name="", executable="", command=("ollama.exe",))
    name_only = _listener(name="ollama", executable="", command=())
    unrelated = _listener(name="caddy", executable="", command=())

    assert ollama._is_ollama_listener(command_only)
    assert ollama._is_ollama_listener(name_only)
    assert not ollama._is_ollama_listener(unrelated)
    assert ollama._server_instance_id(command_only).endswith(":ollama.exe")
    assert ollama._server_instance_id(name_only).endswith(":ollama")
    assert not ollama._is_loopback_address("localhost")


@pytest.mark.parametrize(("url", "expected_port"), [
    ("http://127.0.0.1", 80),
    ("https://127.0.0.1", 443),
])
def test_runtime_authority_uses_scheme_default_port_and_implicit_endpoint(
        monkeypatch, url, expected_port):
    endpoint = ollama.OllamaEndpoint(url, "loopback")
    monkeypatch.setattr(ollama, "endpoint_authority", lambda: endpoint)
    seen = []
    monkeypatch.setattr(
        ollama, "_local_tcp_listeners",
        lambda port: seen.append(port) or [_listener()])
    monkeypatch.setattr(ollama, "version", lambda timeout=0.5: "v0.30.10")

    assert ollama.runtime_authority().issue is None
    assert seen == [expected_port]


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
        "num_parallel": 4,
        "expires_at": "2026-07-17T12:00:00Z",
    }]})

    process = ollama.processes()[0]
    assert process.name == "qwen3:0.6b"
    assert process.model == "qwen3:0.6b"
    assert process.digest == digest
    assert process.size_bytes == 600_000_000
    assert process.size_vram_bytes == 500_000_000
    assert process.effective_context_per_request == 8192
    assert process.parallelism is None
    assert process.expires_at == "2026-07-17T12:00:00Z"


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
def test_load_generates_empty_prompt_under_daemon_keepalive_policy(monkeypatch):
    seen = {}

    def fake_post(path, payload, timeout):
        seen.update(path=path, payload=payload)
        return {"done": True}

    monkeypatch.setattr(ollama, "_post_json", fake_post)
    assert ollama.load("x-ara") == {"done": True}
    assert seen["path"] == "/api/generate"
    assert seen["payload"]["model"] == "x-ara"
    assert seen["payload"] == {"model": "x-ara", "prompt": "", "stream": False}


def test_load_can_explicitly_request_indefinite_keepalive(monkeypatch):
    seen = {}

    def fake_post(path, payload, timeout):
        seen.update(path=path, payload=payload)
        return {"done": True}

    monkeypatch.setattr(ollama, "_post_json", fake_post)
    assert ollama.load("x-ara", keep_alive=-1) == {"done": True}
    assert seen["path"] == "/api/generate"
    assert seen["payload"]["keep_alive"] == -1


# --------------------------------------------------------------------------- #
# probe_generate — bounded characterization request with no context rewriting
# --------------------------------------------------------------------------- #
def test_probe_generate_requests_one_token_without_truncation_or_shift(monkeypatch):
    seen = {}

    def fake_post(path, payload, timeout):
        seen.update(path=path, payload=payload, timeout=timeout)
        return {"done": True, "response": "x"}

    monkeypatch.setattr(ollama, "_post_json", fake_post)

    assert ollama.probe_generate("x-ara", 8192) is True
    assert seen == {
        "path": "/api/generate",
        "payload": {
            "model": "x-ara",
            "prompt": "ARA",
            "stream": False,
            "truncate": False,
            "shift": False,
            "options": {"num_ctx": 8192, "num_predict": 1},
        },
        "timeout": 300.0,
    }


def test_probe_generate_fails_closed_on_missing_or_incomplete_response(monkeypatch):
    monkeypatch.setattr(ollama, "_post_json", lambda p, payload, timeout: None)
    assert ollama.probe_generate("x-ara", 4096) is False

    monkeypatch.setattr(ollama, "_post_json", lambda p, payload, timeout: {"done": False})
    assert ollama.probe_generate("x-ara", 4096) is False


# --------------------------------------------------------------------------- #
# governed run — warm + buffered native generation
# --------------------------------------------------------------------------- #
def test_warm_for_run_uses_the_governed_runner_options_without_keepalive(monkeypatch):
    seen = {}

    def fake_post(path, payload, timeout):
        seen.update(path=path, payload=payload, timeout=timeout)
        return {"done": True}

    monkeypatch.setattr(ollama, "_post_json", fake_post)

    assert ollama.warm_for_run("qwen3:0.6b", 8192) == {"done": True}
    assert seen == {
        "path": "/api/generate",
        "payload": {
            "model": "qwen3:0.6b",
            "prompt": "",
            "stream": False,
            "truncate": False,
            "shift": False,
            "options": {"num_ctx": 8192},
        },
        "timeout": 300.0,
    }


def test_generate_for_run_buffers_with_explicit_governed_options(monkeypatch):
    seen = {}
    response = {"done": True, "response": "hello"}

    def fake_post(path, payload, timeout):
        seen.update(path=path, payload=payload, timeout=timeout)
        return response

    monkeypatch.setattr(ollama, "_post_json", fake_post)

    assert ollama.generate_for_run("qwen3:0.6b", "Hi", 8192, 256) is response
    assert seen == {
        "path": "/api/generate",
        "payload": {
            "model": "qwen3:0.6b",
            "prompt": "Hi",
            "stream": False,
            "truncate": False,
            "shift": False,
            "options": {"num_ctx": 8192, "num_predict": 256},
        },
        "timeout": 300.0,
    }


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
