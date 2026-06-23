# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""This machine's identity (``machine_key``) and persisted capability profile.

``machine_key()`` is the stable identity ARA keys every stored record by — built from ARA's own
recon (CPU, accelerator, RAM, OS), so a measurement is reused only on the same machine, and the
engine never has to answer "which machine is this". The persisted *profile* — the serialized
``detect.machine()`` snapshot — is captured here too. See Spec 2026-06-23-capability-pipeline.
"""
from __future__ import annotations

import dataclasses
import json
import platform

import psutil

from ara import db, detect


def machine_key() -> str:
    """Stable identity for this machine: chip · accelerator · total RAM · OS."""
    chip = detect.chip_name()
    accel = detect.accelerator(chip)
    ram_bytes = psutil.virtual_memory().total
    return "|".join([chip, accel.name, str(ram_bytes), platform.system()])


def capture(con) -> dict:
    """Build the current ``Machine`` (already enriched, engine-free), persist it as a profile
    (history kept), and return the dict. The persisted JSON is the interchange format."""
    d = dataclasses.asdict(detect.machine())
    db.save_profile(con, machine_key(), json.dumps(d, default=str))
    return d
