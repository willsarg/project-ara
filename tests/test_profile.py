# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""profile.py — this machine's identity (machine_key) + persisted capability profile.

Spec 2026-06-23-capability-pipeline (Slice 1)."""
from __future__ import annotations

import json
import types

from ara import db, detect, profile, serialize


def test_machine_key_is_composite_and_stable(monkeypatch):
    monkeypatch.setattr(profile.detect, "chip_name", lambda: "TestCPU")
    monkeypatch.setattr(profile.detect, "accelerator",
                        lambda chip: types.SimpleNamespace(name="TestGPU"))
    monkeypatch.setattr(profile.psutil, "virtual_memory",
                        lambda: types.SimpleNamespace(total=34359738368))
    monkeypatch.setattr(profile.platform, "system", lambda: "Linux")
    k = profile.machine_key()
    assert k == "TestCPU|TestGPU|34359738368|Linux"
    assert profile.machine_key() == k          # stable across calls


def test_capture_persists_machine_and_projection(store, monkeypatch):
    """capture() persists BOTH the lossless machine blob and the curated projection, and
    returns that record."""
    m = detect.machine()
    monkeypatch.setattr(profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(profile.detect, "machine", lambda: m)   # fixed snapshot, no live churn
    d = profile.capture(store)
    # the curated projection: durable fields present, live ones absent
    proj = d["projection"]
    assert "chip" in proj and "backend" in proj and "ram_total_gb" in proj
    assert "ram_available_gb" not in proj and "disk_free_gb" not in proj and "apps" not in proj
    # the lossless machine blob: full detect --json shape (live fields allowed)
    assert d["machine"] == serialize.machine(m)
    saved = db.get_latest_profile(store, "mkey")          # persisted, keyed by machine_key
    # the stored JSON is the returned record (compare through one JSON round-trip: the store
    # normalises tuples→lists, so round-trip both sides to compare like-for-like)
    assert json.loads(saved["profile_json"]) == json.loads(json.dumps(d, default=str))


def test_capture_projection_is_stable_no_false_drift(store, monkeypatch):
    """Two immediate captures on an unchanged machine produce a byte-identical PROJECTION."""
    monkeypatch.setattr(profile, "machine_key", lambda: "mkey")
    first = profile.capture(store)
    second = profile.capture(store)
    assert first["projection"] == second["projection"]   # no false drift in the durable view
    rows = db.list_profiles(store, "mkey")
    a = json.loads(rows[0]["profile_json"])["projection"]
    b = json.loads(rows[1]["profile_json"])["projection"]
    assert a == b
