# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node's enroll-time self-description — schema conformance, cross-OS container-honest env, and
the characterized-model capability advertisement.

Host probing is mocked (like conftest): we drive ``profile.machine_key``, ``detect``, ``psutil`` and
the cgroup/container filesystem so every path runs deterministically on any CI host, and validate
the result against the pinned wire contract so a schema drift breaks this test.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from ara import hardware
from ara.node import capabilities

_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "contracts" / "wire" / "schema"


def _registry() -> Registry:
    pairs = [
        (json.loads(p.read_text())["$id"], Resource.from_contents(json.loads(p.read_text())))
        for p in sorted(_SCHEMA_DIR.glob("*.schema.json"))
    ]
    return Registry().with_resources(pairs)


_REGISTRY = _registry()


def _validate(instance: dict, schema_id: str) -> None:
    schema = _REGISTRY.contents(schema_id)
    errors = list(Draft202012Validator(schema, registry=_REGISTRY).iter_errors(instance))
    assert not errors, [e.message for e in errors]


def test_capability_schema_describes_canonical_ara_engine_identity():
    schema = _REGISTRY.contents("https://ara.dev/wire/capability.json")
    assert schema["properties"]["engine"]["description"] == "Canonical ARA engine identity."


def test_ollama_capability_schema_requires_exact_node_target_authority():
    cap = {
        "kind": "serve_model", "id": "qwen3:0.6b", "engine": "ollama",
        "evidence": "characterized",
    }
    with pytest.raises(AssertionError):
        _validate(cap, "https://ara.dev/wire/capability.json")


@pytest.fixture
def env_io(monkeypatch):
    """Deterministic host I/O for the env probes: bare-metal Linux, 32 GiB, no cgroup, no container.

    Returns a mutable ``state`` — a test tweaks ``files`` (path→text), ``exists`` (marker paths),
    ``phys`` (physical RAM bytes) or ``system`` to drive a specific scenario."""
    state = {"files": {}, "exists": set(), "phys": 32 * 1024**3, "system": "Linux"}
    monkeypatch.setattr(capabilities, "_read_text", lambda path: state["files"].get(path))
    monkeypatch.setattr(capabilities, "_path_exists", lambda path: path in state["exists"])
    # The cgroup wall now reads through the shared hardware boundary — mock it from the same state.
    monkeypatch.setattr(hardware, "_read_cgroup_file", lambda path: state["files"].get(path))
    monkeypatch.setattr(capabilities.psutil, "virtual_memory",
                        lambda: types.SimpleNamespace(total=state["phys"]))
    monkeypatch.setattr(capabilities.platform, "system", lambda: state["system"])
    return state


@pytest.fixture
def stub_host(monkeypatch):
    """Force a known accelerator + machine_key so env labelling is deterministic."""
    def _stub(*, accel_kind: str = "nvidia"):
        monkeypatch.setattr(capabilities.profile, "machine_key", lambda: "chip|GPU|16|Linux")
        monkeypatch.setattr(capabilities.detect, "chip_name", lambda: "chip")
        monkeypatch.setattr(capabilities.detect, "accelerator",
                            lambda chip: types.SimpleNamespace(kind=accel_kind))
    return _stub


# --------------------------------------------------------------------------- #
# filesystem boundary helpers
# --------------------------------------------------------------------------- #
def test_read_text_reads_existing_and_none_for_missing(tmp_path):
    p = tmp_path / "f"
    p.write_text("hello")
    assert capabilities._read_text(str(p)) == "hello"
    assert capabilities._read_text(str(tmp_path / "nope")) is None       # OSError → None


def test_path_exists_boundary(tmp_path):
    assert capabilities._path_exists(str(tmp_path)) is True
    assert capabilities._path_exists(str(tmp_path / "nope")) is False


# --------------------------------------------------------------------------- #
# effective_wall / is_cgroup_bound — the label reads the shared hardware cgroup helper
# --------------------------------------------------------------------------- #
def test_effective_wall_is_physical_without_cgroup(env_io):
    env_io["phys"] = 16 * 1024**3
    assert capabilities.effective_wall() == 16 * 1024**3
    assert capabilities.is_cgroup_bound() is False


def test_effective_wall_binds_to_cgroup_below_physical(env_io):
    env_io["phys"] = 16 * 1024**3
    env_io["files"][hardware._CGROUP_V2] = str(4 * 1024**3)
    assert capabilities.effective_wall() == 4 * 1024**3
    assert capabilities.is_cgroup_bound() is True


def test_cgroup_limit_at_or_above_physical_does_not_bind(env_io):
    env_io["phys"] = 16 * 1024**3
    env_io["files"][hardware._CGROUP_V2] = str(64 * 1024**3)
    assert capabilities.effective_wall() == 16 * 1024**3
    assert capabilities.is_cgroup_bound() is False


def test_effective_wall_ignores_cgroup_off_linux(env_io):
    env_io["system"] = "Darwin"
    env_io["phys"] = 16 * 1024**3
    env_io["files"][hardware._CGROUP_V2] = str(1 * 1024**3)               # would bind IF read
    assert capabilities.effective_wall() == 16 * 1024**3
    assert capabilities.is_cgroup_bound() is False


# --------------------------------------------------------------------------- #
# containerized / virtualization_layer
# --------------------------------------------------------------------------- #
def test_containerized_via_dockerenv(env_io):
    env_io["exists"].add("/.dockerenv")
    assert capabilities._containerized(False) is True


def test_containerized_via_cgroup_lineage_proc1(env_io):
    env_io["files"]["/proc/1/cgroup"] = "1:name=systemd:/docker/abc123"
    assert capabilities._containerized(False) is True


def test_containerized_via_cgroup_lineage_self_kubepods(env_io):
    env_io["files"]["/proc/self/cgroup"] = "0::/kubepods/pod-xyz"
    assert capabilities._containerized(False) is True


def test_containerized_cgroup_lineage_without_marker_falls_through(env_io):
    env_io["files"]["/proc/1/cgroup"] = "1:cpu:/user.slice"
    env_io["files"]["/proc/self/cgroup"] = "0::/user.slice"
    assert capabilities._containerized(False) is False
    assert capabilities._containerized(True) is True                      # falls to cgroup_binds


def test_virtualization_layer_wsl2(env_io):
    env_io["files"]["/proc/version"] = "Linux version 5.15 Microsoft-standard WSL2"
    assert capabilities._virtualization_layer() == "wsl2"


def test_virtualization_layer_docker(env_io):
    env_io["exists"].add("/.dockerenv")
    assert capabilities._virtualization_layer() == "docker"


def test_virtualization_layer_none_bare_metal(env_io):
    assert capabilities._virtualization_layer() is None


# --------------------------------------------------------------------------- #
# environment() — mapping + container-honesty, schema-valid
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("system,expected", [
    ("Linux", "linux"), ("Darwin", "darwin"), ("Windows", "windows"), ("Plan9", "unknown"),
])
def test_environment_platform_mapping(stub_host, env_io, system, expected):
    stub_host()
    env_io["system"] = system
    assert capabilities.environment()["platform"] == expected


@pytest.mark.parametrize("kind,expected", [
    ("apple", "metal"), ("nvidia", "nvidia"), ("none", "cpu"), ("weird", "unknown"),
])
def test_environment_accel_mapping(stub_host, env_io, kind, expected):
    stub_host(accel_kind=kind)
    assert capabilities.environment()["accel"] == expected


def test_environment_bare_metal_is_physical_and_schema_valid(stub_host, env_io):
    stub_host()
    env = capabilities.environment()
    assert env["containerized"] is False and env["wall_source"] == "physical"
    assert env["virtualization_layer"] is None
    _validate(env, "https://ara.dev/wire/environment.json")


def test_environment_container_honest_cgroup_wall(stub_host, env_io):
    stub_host()
    env_io["phys"] = 16 * 1024**3
    env_io["files"][hardware._CGROUP_V2] = str(4 * 1024**3)               # squeezed under the host
    env = capabilities.environment()
    assert env["wall_source"] == "cgroup"
    assert env["containerized"] is True                                   # a binding limit implies it
    _validate(env, "https://ara.dev/wire/environment.json")


def test_environment_wsl2_layer_surfaced(stub_host, env_io):
    stub_host()
    env_io["files"]["/proc/version"] = "Linux microsoft-standard-WSL2"
    env = capabilities.environment()
    assert env["virtualization_layer"] == "wsl2"
    _validate(env, "https://ara.dev/wire/environment.json")


# --------------------------------------------------------------------------- #
# advertised_capabilities() — characterized models from ARA's store
# --------------------------------------------------------------------------- #
def test_advertised_capabilities_empty_when_none(monkeypatch):
    monkeypatch.setattr(capabilities.profile, "machine_key", lambda: "m")
    assert capabilities.advertised_capabilities() == []


def test_advertised_capabilities_from_characterizations(monkeypatch):
    monkeypatch.setattr(capabilities.profile, "machine_key", lambda: "m")
    monkeypatch.setattr(capabilities.config, "node_identity", lambda: "node1")
    con = capabilities.db.connect()
    capabilities.db.save_characterization(con, "m", "cuda", "org/model-a",
                                          safe_context=4096, points=[], config={},
                                          artifact_id="hf:org/model-a@revision-a")
    capabilities.db.save_characterization(con, "m", "mlx", "org/model-b",
                                          safe_context=2048, points=[], config={},
                                          artifact_id="hf:org/model-b@revision-b")
    capabilities.db.save_characterization(con, "other", "cuda", "org/model-z",
                                          safe_context=1, points=[], config={},
                                          artifact_id="hf:org/model-z@revision-z")  # other machine → excluded
    con.close()
    caps = capabilities.advertised_capabilities()
    assert [(cap["id"], cap["runtime"], cap["backend"]) for cap in caps] == [
        ("org/model-a", "torch", "cuda"),
        ("org/model-b", "mlx", "apple"),
    ]
    assert all(cap["authority"].startswith("node-target:v1:") for cap in caps)
    node1_authorities = [cap["authority"] for cap in caps]
    for cap in caps:
        _validate(cap, "https://ara.dev/wire/capability.json")
    monkeypatch.setattr(capabilities.config, "node_identity", lambda: "node2")
    assert [cap["authority"] for cap in capabilities.advertised_capabilities()] != node1_authorities


def test_advertised_capabilities_only_include_exact_reusable_rows(monkeypatch):
    monkeypatch.setattr(capabilities.profile, "machine_key", lambda: "m")
    monkeypatch.setattr(capabilities.config, "node_identity", lambda: "node1")
    monkeypatch.setattr(capabilities.db, "list_reusable_characterizations", lambda con, mk: [
        {"logical_model_id": "exact", "legacy_engine": "ollama", "runtime": "ollama",
         "backend": "apple", "artifact_id": "ollama-manifest-sha256:" + "a" * 64,
         "config_key": "cfg:v1:{}", "safe_context": 2048},
        {"logical_model_id": "unfit", "legacy_engine": "ollama", "runtime": "ollama",
         "backend": "apple", "artifact_id": "ollama-manifest-sha256:" + "b" * 64,
         "config_key": "cfg:v1:{}", "safe_context": None},
        {"logical_model_id": "zero", "legacy_engine": "ollama", "runtime": "ollama",
         "backend": "apple", "artifact_id": "ollama-manifest-sha256:" + "c" * 64,
         "config_key": "cfg:v1:{}", "safe_context": 0},
    ])
    caps = capabilities.advertised_capabilities()
    assert [cap["id"] for cap in caps] == ["exact"]
    _validate(caps[0], "https://ara.dev/wire/capability.json")


def test_ollama_execution_authority_is_node_scoped_and_rejects_target_drift(monkeypatch):
    cap = {
        "kind": "serve_model", "id": "qwen3:0.6b", "engine": "ollama",
        "evidence": "characterized", "runtime": "ollama", "backend": "apple",
        "artifact_id": "ollama-manifest-sha256:" + "a" * 64,
        "config_key": "cfg:v1:{}", "safe_context": 2048,
        "authority": "node-target:v1:" + "f" * 64,
    }
    monkeypatch.setattr(capabilities, "advertised_capabilities", lambda: [cap])
    args = {"model": "qwen3:0.6b", "engine": "ollama", "target_authority": cap["authority"]}
    capabilities.require_execution_authority("run", args)
    capabilities.require_execution_authority("benchmark", args)
    for changed in [
        {**args, "model": "other"},
        {**args, "model": None},
        {**args, "target_authority": None},
        {**args, "target_authority": "node-target:v1:" + "0" * 64},
    ]:
        with pytest.raises(ValueError, match="node-scoped runtime authority"):
            capabilities.require_execution_authority("run", changed)


def test_execution_authority_is_only_required_for_governed_remote_ollama_work():
    capabilities.require_execution_authority("detect", {})
    capabilities.require_execution_authority("characterize", {"engine": "ollama"})
    capabilities.require_execution_authority("run", {"engine": "cpu"})


# --------------------------------------------------------------------------- #
# self_description() — full payload against the enroll.request contract
# --------------------------------------------------------------------------- #
def test_self_description_conforms_to_enroll_request(stub_host, env_io, monkeypatch):
    stub_host()
    monkeypatch.setattr(capabilities.config, "node_identity", lambda: "node1_testbox")
    monkeypatch.setattr(capabilities.platform, "node", lambda: "test-box")
    monkeypatch.setattr(capabilities.platform, "machine", lambda: "x86_64")
    desc = capabilities.self_description()
    assert desc["machine_key"] == "node1_testbox"
    assert desc["identity"] == {"hostname": "test-box", "os": "Linux", "arch": "x86_64"}
    assert desc["capabilities"] == []                                     # isolated db → none yet
    _validate(desc, "https://ara.dev/wire/enroll.request.json")


def test_self_description_advertises_characterized_models(stub_host, env_io, monkeypatch):
    stub_host()
    monkeypatch.setattr(capabilities.config, "node_identity", lambda: "node1_testbox")
    monkeypatch.setattr(capabilities.platform, "node", lambda: "test-box")
    monkeypatch.setattr(capabilities.platform, "machine", lambda: "x86_64")
    con = capabilities.db.connect()
    capabilities.db.save_characterization(con, "chip|GPU|16|Linux", "cuda", "org/m",
                                          safe_context=8192, points=[], config={},
                                          artifact_id="hf:org/m@revision")
    con.close()
    desc = capabilities.self_description()
    assert len(desc["capabilities"]) == 1
    cap = desc["capabilities"][0]
    assert {key: cap[key] for key in (
        "kind", "id", "engine", "evidence", "runtime", "backend", "artifact_id",
        "safe_context",
    )} == {
        "kind": "serve_model", "id": "org/m", "engine": "cuda",
        "evidence": "characterized", "runtime": "torch", "backend": "cuda",
        "artifact_id": "hf:org/m@revision", "safe_context": 8192,
    }
    assert cap["authority"].startswith("node-target:v1:")
    _validate(desc, "https://ara.dev/wire/enroll.request.json")


def test_self_description_falls_back_when_host_fields_empty(stub_host, env_io, monkeypatch):
    stub_host()
    monkeypatch.setattr(capabilities.config, "node_identity", lambda: "node1_testbox")
    monkeypatch.setattr(capabilities.platform, "node", lambda: "")
    monkeypatch.setattr(capabilities.platform, "machine", lambda: "")
    desc = capabilities.self_description()
    assert desc["identity"]["hostname"] == "unknown"       # empty hostname would break minLength:1
    assert desc["identity"]["arch"] == "unknown"
    _validate(desc, "https://ara.dev/wire/enroll.request.json")
