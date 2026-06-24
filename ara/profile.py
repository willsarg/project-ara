# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""This machine's identity (``machine_key``) and persisted capability profile.

``machine_key()`` is the stable identity ARA keys every stored record by — built from ARA's own
recon (CPU, accelerator, RAM, OS), so a measurement is reused only on the same machine, and the
engine never has to answer "which machine is this". The persisted *profile* is the curated
DURABLE capability projection of ``detect.machine()`` (see ``serialize.profile_record``) — stable
across re-runs so drift history shows no false positives. See Spec 2026-06-23-capability-pipeline.
"""
from __future__ import annotations

import json
import platform

import psutil

from ara import db, detect, serialize


def machine_key() -> str:
    """Stable identity for this machine: chip · accelerator · total RAM · OS."""
    chip = detect.chip_name()
    accel = detect.accelerator(chip)
    ram_bytes = psutil.virtual_memory().total
    return "|".join([chip, accel.name, str(ram_bytes), platform.system()])


def capture(con) -> dict:
    """Build the current ``Machine`` (already enriched, engine-free) and persist a profile
    (history kept) holding BOTH the lossless ``machine`` blob and the curated DURABLE
    ``projection``. Returns the stored record. The projection is stable across re-runs on an
    unchanged machine, so drift detection over it shows no false positives; the machine blob is
    the full record (it may carry live fields)."""
    m = detect.machine()
    d = {"machine": serialize.machine(m), "projection": serialize.profile_record(m)}
    db.save_profile(con, machine_key(), json.dumps(d, default=str))
    return d
