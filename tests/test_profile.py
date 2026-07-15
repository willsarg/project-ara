# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""profile.py — this machine's identity (machine_key) + persisted capability profile.

Spec 2026-06-23-capability-pipeline (Slice 1)."""
from __future__ import annotations

import dataclasses
import json
import types

from ara import db, detect, profile, serialize


def _pin_machine(monkeypatch, *, total: int, chip="TestCPU", accel="TestGPU", os_="Linux"):
    monkeypatch.setattr(profile.detect, "chip_name", lambda: chip)
    monkeypatch.setattr(profile.detect, "accelerator",
                        lambda c: types.SimpleNamespace(name=accel))
    monkeypatch.setattr(profile.psutil, "virtual_memory",
                        lambda: types.SimpleNamespace(total=total))
    monkeypatch.setattr(profile.platform, "system", lambda: os_)


def test_machine_key_is_versioned_gib_rounded_and_stable(monkeypatch):
    _pin_machine(monkeypatch, total=34359738368)          # 32 * 2**30 exactly
    k = profile.machine_key()
    assert k == "ara1|TestCPU|TestGPU|32|Linux"           # versioned + GiB-rounded, not byte-exact
    assert profile.machine_key() == k                     # stable across calls


def test_machine_key_absorbs_reboot_ram_drift(monkeypatch):
    """A few-MB reboot-to-reboot RAM drift must NOT change the key (the Rule #1 data-loss bug)."""
    _pin_machine(monkeypatch, total=34359738368)
    k1 = profile.machine_key()
    _pin_machine(monkeypatch, total=34359738368 - 8_000_000)   # ~8 MB lower after a reboot
    assert profile.machine_key() == k1


def test_rekey_legacy_key_transforms_and_guards():
    # legacy 4-field byte-exact key -> versioned GiB-rounded key
    assert (profile.rekey_legacy_key("TestCPU|TestGPU|34359738368|Linux")
            == "ara1|TestCPU|TestGPU|32|Linux")
    # already versioned -> None (nothing to do)
    assert profile.rekey_legacy_key("ara1|TestCPU|TestGPU|32|Linux") is None
    # not exactly 4 fields, or non-int RAM -> None (can't transform safely, never corrupt)
    assert profile.rekey_legacy_key("TestCPU|TestGPU|Linux") is None
    assert profile.rekey_legacy_key("TestCPU|TestGPU|notanint|Linux") is None


def test_capture_persists_machine_and_projection(store, monkeypatch, sample_machine):
    """capture() persists BOTH the lossless machine blob and the curated projection, and
    returns that record."""
    m = dataclasses.replace(sample_machine, engine="wmx", backend="apple")
    monkeypatch.setattr(profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(profile.detect, "machine", lambda: m)   # fixed snapshot, no live churn
    d = profile.capture(store)
    # the curated projection: durable fields present, live ones absent
    proj = d["projection"]
    assert "chip" in proj and "backend" in proj and "ram_total_gb" in proj
    assert proj["engine"] == "mlx" and proj["backend"] == "apple"
    assert "ram_available_gb" not in proj and "disk_free_gb" not in proj and "apps" not in proj
    # the lossless machine blob: full detect --json shape (live fields allowed)
    assert d["machine"] == serialize.machine(m)
    assert d["machine"]["engine"] == "mlx" and d["machine"]["backend"] == "apple"
    saved = db.get_latest_profile(store, "mkey")          # persisted, keyed by machine_key
    # the stored JSON is the returned record (compare through one JSON round-trip: the store
    # normalises tuples→lists, so round-trip both sides to compare like-for-like)
    assert json.loads(saved["profile_json"]) == json.loads(json.dumps(d, default=str))


def test_capture_projection_is_stable_no_false_drift(store, monkeypatch, sample_machine):
    """Two immediate captures on an unchanged machine produce a byte-identical PROJECTION."""
    monkeypatch.setattr(profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(profile.detect, "machine", lambda: sample_machine)
    first = profile.capture(store)
    second = profile.capture(store)
    assert first["projection"] == second["projection"]   # no false drift in the durable view
    rows = db.list_profiles(store, "mkey")
    a = json.loads(rows[0]["profile_json"])["projection"]
    b = json.loads(rows[1]["profile_json"])["projection"]
    assert a == b
