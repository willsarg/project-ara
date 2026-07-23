# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Canonical, auditable characterization methodology identities."""
from __future__ import annotations

import hashlib
import json

from ara.contracts import ramp


def characterization_descriptor(*, schedule: list[int], repeats: int,
                                reserve_policy: str,
                                reserve_bytes: int | dict[str, int],
                                worker_protocol: str, sampling_interval_ms: int,
                                telemetry_failure_policy: str,
                                watchdog_stop_rule: str) -> dict:
    """Return the complete behavior descriptor whose hash scopes reusable evidence."""
    return {
        "schema": "characterization-methodology:v1",
        "ramp_contract": "direct-context:fitted-advisory:v1",
        "schedule": list(schedule),
        "repeats": repeats,
        "prediction_policy": "apriori-then-measured-linear-fit:v1",
        "bisection": {
            "minimum_gap_tokens": ramp.BISECT_MIN_GAP,
            "maximum_steps": ramp.BISECT_MAX_STEPS,
        },
        "reserve_policy": reserve_policy,
        "reserve_bytes": reserve_bytes,
        "worker_protocol": worker_protocol,
        "sampling_interval_ms": sampling_interval_ms,
        "telemetry_failure_policy": telemetry_failure_policy,
        "watchdog_stop_rule": watchdog_stop_rule,
    }


def key(descriptor: dict) -> str:
    """Hash a methodology descriptor with stable JSON ordering."""
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    return "methodology:v1:sha256:" + hashlib.sha256(payload).hexdigest()
