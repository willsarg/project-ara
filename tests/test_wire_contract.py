# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node<->server wire contract: validate every golden fixture against its schema.

Cross-language anti-drift backbone — the coordinator's vitest suite
(``coordinator/test/contracts.test.ts``) validates the SAME fixtures against the SAME
schemas with ajv. If the two ever disagree, the contract has drifted.

This lives outside ``ara/`` on purpose: it exercises the shared ``contracts/wire`` artifact,
not ARA source, so it carries no coverage obligation of its own.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

_WIRE = Path(__file__).resolve().parent.parent / "contracts" / "wire"
_SCHEMA_DIR = _WIRE / "schema"
_FIXTURE_DIR = _WIRE / "fixtures"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _registry() -> Registry:
    """A referencing Registry keyed by each schema's ``$id`` so ``$ref`` resolves."""
    pairs = [
        (schema["$id"], Resource.from_contents(schema))
        for schema in (_load(p) for p in sorted(_SCHEMA_DIR.glob("*.schema.json")))
    ]
    return Registry().with_resources(pairs)


_REGISTRY = _registry()
_MANIFEST = _load(_FIXTURE_DIR / "manifest.json")


@pytest.mark.parametrize("case", _MANIFEST["cases"], ids=lambda c: c["fixture"])
def test_fixture_matches_contract(case: dict) -> None:
    schema = _REGISTRY.contents(case["schema"])
    validator = Draft202012Validator(schema, registry=_REGISTRY)
    errors = list(validator.iter_errors(_load(_FIXTURE_DIR / case["fixture"])))
    if case["valid"]:
        assert not errors, f"{case['fixture']} should be VALID: {[e.message for e in errors]}"
    else:
        assert errors, f"{case['fixture']} should be INVALID but validated clean"


def test_manifest_covers_every_fixture() -> None:
    """No fixture is left unreferenced by the manifest (catches orphaned/renamed cases)."""
    referenced = {c["fixture"] for c in _MANIFEST["cases"]}
    on_disk = {p.name for p in _FIXTURE_DIR.glob("*.json")} - {"manifest.json"}
    assert referenced == on_disk, f"manifest vs fixtures mismatch: {referenced ^ on_disk}"


def test_every_schema_is_valid_metaschema() -> None:
    """Each schema is itself a well-formed 2020-12 schema (catches typos in the contract)."""
    for path in _SCHEMA_DIR.glob("*.schema.json"):
        Draft202012Validator.check_schema(_load(path))


def test_enrollment_fixtures_use_an_obviously_synthetic_identity() -> None:
    """Public wire examples must not identify a maintainer's real fleet hosts."""
    paths = _FIXTURE_DIR.glob("enroll.request.*.json")
    for path in paths:
        fixture = _load(path)
        assert fixture["identity"]["hostname"].startswith("example-")
        machine_key = fixture.get("machine_key")
        assert machine_key is None or machine_key.startswith("example-")
