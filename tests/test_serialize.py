# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""serialize.py — domain→JSON-ready interchange seam (the node identity fleet mode ships).

Spec 2026-06-23-capability-pipeline (Slice 1)."""
from __future__ import annotations

import dataclasses

import pytest

from ara import detect, serialize


def test_machine_matches_detect_json_shape():
    """serialize.machine(m) == asdict(m) + the `accelerated` @property — the detect --json shape."""
    m = detect.machine()
    d = serialize.machine(m)
    expected = {**dataclasses.asdict(m), "accelerated": m.accelerated}
    expected["engine"] = {"wmx": "mlx", "wmx-suite": "mlx", "wcx": "cuda",
                          "wcx-suite": "cuda"}.get(expected["engine"], expected["engine"])
    assert d == expected
    assert isinstance(d["accelerated"], bool)


@pytest.mark.parametrize(("legacy", "canonical"), [("wmx", "mlx"), ("wcx", "cuda")])
def test_machine_canonicalizes_legacy_engine_identity(legacy, canonical):
    m = dataclasses.replace(detect.machine(), engine=legacy)
    assert serialize.machine(m)["engine"] == canonical


def test_profile_record_includes_durable_capability():
    """The durable capability fields are present (chip, accel, ram_total, backend, runtimes...)."""
    m = detect.machine()
    rec = serialize.profile_record(m)
    for key in ("system", "os_version", "arch", "chip", "cpu_physical", "cpu_logical",
                "cpu_features", "ram_total_gb", "swap_gb", "accel", "gpus", "board",
                "backend", "engine", "engine_ready", "runtimes"):
        assert key in rec, f"missing durable field: {key}"
    # nested domain objects come through as dicts/lists (asdict-expanded), JSON-ready
    assert isinstance(rec["accel"], dict)
    assert isinstance(rec["gpus"], list)
    assert isinstance(rec["board"], dict)
    assert isinstance(rec["runtimes"], list)


@pytest.mark.parametrize(("legacy", "canonical"), [("wmx", "mlx"), ("wcx", "cuda")])
def test_profile_record_canonicalizes_engine_without_changing_backend(legacy, canonical):
    m = dataclasses.replace(detect.machine(), engine=legacy, backend="apple" if legacy == "wmx" else "cuda")
    rec = serialize.profile_record(m)
    assert rec["engine"] == canonical
    assert rec["backend"] == ("apple" if legacy == "wmx" else "cuda")


def test_profile_record_excludes_live_transient_fields():
    """The live/transient fields are absent — including them would cause false drift."""
    m = detect.machine()
    rec = serialize.profile_record(m)
    for key in ("ram_available_gb", "disk_free_gb", "power", "model_stores", "apps",
                "hf_token", "hf_cli", "hf_cli_version", "python_version",
                "framework_python", "memory", "storage"):
        assert key not in rec, f"live/transient field leaked into projection: {key}"


def test_profile_record_drops_volatile_serving_from_runtimes():
    """A runtime's `serving` state (e.g. ollama up/down) is LIVE — it must not enter the durable
    projection, or starting/stopping ollama would trip false drift. Spec 2026-06-26-detect-ollama-liveness."""
    m = detect.machine()
    rec = serialize.profile_record(m)
    for r in rec["runtimes"]:
        assert "serving" not in r, "live `serving` leaked into the durable runtime projection"
