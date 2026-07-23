# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Canonical identities for characterization methodology."""
from __future__ import annotations

from ara import methodology


def _descriptor(**overrides):
    values = {
        "schedule": [2000, 4000, 8000],
        "repeats": 3,
        "reserve_policy": "recommended-working-set-minus-fixed-reserve",
        "reserve_bytes": 2 * 1024 ** 3,
        "worker_protocol": "measurement:v1",
        "sampling_interval_ms": 50,
        "telemetry_failure_policy": "worker-watchdog",
        "watchdog_stop_rule": "wired-gte-budget",
    }
    values.update(overrides)
    return methodology.characterization_descriptor(**values)


def test_methodology_identity_is_canonical_and_self_describing():
    descriptor = _descriptor()
    reversed_order = {key: descriptor[key] for key in reversed(descriptor)}
    assert methodology.key(descriptor) == methodology.key(reversed_order)
    assert methodology.key(descriptor).startswith("methodology:v1:sha256:")
    assert descriptor["ramp_contract"] == "direct-context:fitted-advisory:v1"
    assert descriptor["bisection"]["minimum_gap_tokens"] == 256
    assert descriptor["bisection"]["maximum_steps"] == 6


def test_every_behavior_dimension_changes_methodology_identity():
    original = _descriptor()
    assert methodology.key(_descriptor(repeats=1)) != methodology.key(original)
    assert methodology.key(_descriptor(schedule=[2000, 4000])) != methodology.key(original)
    assert methodology.key(_descriptor(sampling_interval_ms=25)) != methodology.key(original)
    assert methodology.key(_descriptor(reserve_bytes=1024 ** 3)) != methodology.key(original)


def test_descriptor_is_detached_from_mutable_schedule_input():
    schedule = [2000, 4000]
    descriptor = _descriptor(schedule=schedule)
    schedule.append(8000)
    assert descriptor["schedule"] == [2000, 4000]
