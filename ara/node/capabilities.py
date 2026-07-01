# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""What this node tells the coordinator about itself — the enroll-time self-description.

Reuses ARA's own recon (``profile.machine_key`` for stable identity, ``detect`` for the
accelerator) rather than reinventing host probing, and shapes it to the pinned wire contract
(``enroll.request`` + the shared ``environment`` label). This is STUB-level honesty for the
walking skeleton: ``capabilities`` is advertised empty and the environment is labelled a physical
wall. The container-honest environment (cgroup wall_source, virtualization_layer) is a later phase;
what ships here still validates clean against the schema.
"""
from __future__ import annotations

import platform

from ara import detect, profile

# platform.system() → the environment schema's platform enum (linux | darwin | windows).
_PLATFORMS = {"Linux": "linux", "Darwin": "darwin", "Windows": "windows"}
# detect.Accelerator.kind → the environment schema's accel enum.
_ACCELS = {"apple": "metal", "nvidia": "nvidia", "none": "cpu"}


def environment() -> dict:
    """The shared ``environment`` label for this node (schema: ``environment.json``).

    STUB: every measurement here is treated as a physical (non-container) wall — cgroup-honest
    labelling lands in a later phase. Still schema-valid."""
    accel = detect.accelerator(detect.chip_name())
    return {
        "platform": _PLATFORMS.get(platform.system(), "linux"),
        "accel": _ACCELS.get(accel.kind, "unknown"),
        "containerized": False,
        "wall_source": "physical",
    }


def self_description() -> dict:
    """This node's enrollment payload (schema: ``enroll.request.json``).

    ``machine_key`` is ARA's stable per-machine identity (reused, not reinvented); ``capabilities``
    is an advertised-empty stub for the skeleton. Validates against the wire contract."""
    return {
        "machine_key": profile.machine_key(),
        "identity": {
            "hostname": platform.node() or "unknown",
            "os": platform.system(),
            "arch": platform.machine() or "unknown",
        },
        "capabilities": [],
        "environment": environment(),
    }
