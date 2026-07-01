# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node's enroll-time self-description — schema conformance + cross-OS env labelling.

Host probing is mocked (like conftest): we drive ``profile.machine_key`` and ``detect`` so both the
Apple and non-Apple paths run on any CI host, and we validate the result against the pinned wire
contract so a drift in the schema breaks this test.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

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


@pytest.fixture
def stub_host(monkeypatch):
    """Force a known accelerator + machine_key so env labelling is deterministic."""
    def _stub(*, accel_kind: str = "nvidia"):
        monkeypatch.setattr(capabilities.profile, "machine_key", lambda: "chip|GPU|16|Linux")
        monkeypatch.setattr(capabilities.detect, "chip_name", lambda: "chip")
        monkeypatch.setattr(capabilities.detect, "accelerator",
                            lambda chip: types.SimpleNamespace(kind=accel_kind))
    return _stub


@pytest.mark.parametrize("system,expected", [
    ("Linux", "linux"), ("Darwin", "darwin"), ("Windows", "windows"), ("Plan9", "linux"),
])
def test_environment_platform_mapping(stub_host, monkeypatch, system, expected):
    stub_host()
    monkeypatch.setattr(capabilities.platform, "system", lambda: system)
    assert capabilities.environment()["platform"] == expected


@pytest.mark.parametrize("kind,expected", [
    ("apple", "metal"), ("nvidia", "nvidia"), ("none", "cpu"), ("weird", "unknown"),
])
def test_environment_accel_mapping(stub_host, kind, expected):
    stub_host(accel_kind=kind)
    assert capabilities.environment()["accel"] == expected


def test_environment_is_a_physical_wall_and_schema_valid(stub_host):
    stub_host()
    env = capabilities.environment()
    assert env["containerized"] is False and env["wall_source"] == "physical"
    _validate(env, "https://ara.dev/wire/environment.json")


def test_self_description_conforms_to_enroll_request(stub_host, monkeypatch):
    stub_host()
    monkeypatch.setattr(capabilities.platform, "system", lambda: "Linux")
    monkeypatch.setattr(capabilities.platform, "node", lambda: "test-box")
    monkeypatch.setattr(capabilities.platform, "machine", lambda: "x86_64")
    desc = capabilities.self_description()
    assert desc["machine_key"] == "chip|GPU|16|Linux"
    assert desc["identity"] == {"hostname": "test-box", "os": "Linux", "arch": "x86_64"}
    assert desc["capabilities"] == []
    _validate(desc, "https://ara.dev/wire/enroll.request.json")


def test_self_description_falls_back_when_host_fields_empty(stub_host, monkeypatch):
    stub_host()
    monkeypatch.setattr(capabilities.platform, "node", lambda: "")
    monkeypatch.setattr(capabilities.platform, "machine", lambda: "")
    desc = capabilities.self_description()
    assert desc["identity"]["hostname"] == "unknown"      # empty hostname would break minLength:1
    assert desc["identity"]["arch"] == "unknown"
    _validate(desc, "https://ara.dev/wire/enroll.request.json")
