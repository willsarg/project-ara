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

KEY_VERSION = "ara1"          # machine_key format tag; bump when the key composition changes
_GIB = 1 << 30


def machine_key() -> str:
    """Stable identity for this machine: version · chip · accelerator · RAM (GiB) · OS.

    RAM is rounded to the nearest binary GiB, NOT byte-exact: total-RAM readings drift by a few MB
    across reboots (kernel updates, BIOS reserve, cgroup limits), and a byte-exact key would mint a
    fresh identity each time — silently orphaning every stored measurement (a Rule #1 data-loss).
    The ``ara1`` prefix versions the format so a legacy byte-exact key is mechanically detectable and
    migratable. Spec 2026-07-04-machine-key-stabilization."""
    chip = detect.chip_name()
    accel = detect.accelerator(chip)
    ram_gib = round(psutil.virtual_memory().total / _GIB)
    return "|".join([KEY_VERSION, chip, accel.name, str(ram_gib), platform.system()])


def rekey_legacy_key(old: str) -> str | None:
    """The versioned, GiB-rounded form of a *legacy* ``chip|accel|ram_bytes|os`` machine_key, or
    ``None`` when *old* is already versioned, isn't exactly four ``|`` fields, or its RAM field
    isn't an integer (can't transform safely → leave it untouched, never corrupt). Pure string
    transform: deterministic and machine-independent, so the migration is correct even for a DB
    copied between machines."""
    if old.startswith(KEY_VERSION + "|"):
        return None
    parts = old.split("|")
    if len(parts) != 4:
        return None
    chip, accel, ram, os_ = parts
    try:
        ram_gib = round(int(ram) / _GIB)
    except ValueError:
        return None
    return "|".join([KEY_VERSION, chip, accel, str(ram_gib), os_])


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
