# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""profile.py — this machine's identity (machine_key) + persisted capability profile.

Spec 2026-06-23-capability-pipeline (Slice 1)."""
from __future__ import annotations

import dataclasses
import json
import types

from ara import db, profile


@dataclasses.dataclass
class _FakeMachine:
    chip: str = "Apple M4 Pro"
    backend: str = "apple"


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


def test_capture_persists_serialized_machine(store, monkeypatch):
    monkeypatch.setattr(profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(profile.detect, "machine", lambda: _FakeMachine())
    d = profile.capture(store)
    assert d["chip"] == "Apple M4 Pro" and d["backend"] == "apple"
    saved = db.get_latest_profile(store, "mkey")          # persisted, keyed by machine_key
    assert json.loads(saved["profile_json"])["backend"] == "apple"
