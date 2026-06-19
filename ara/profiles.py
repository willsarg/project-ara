"""This machine's identity, for keying its stored calibration.

ARA owns this, not the engine: the key is built from ARA's own recon — the CPU, the
accelerator, total RAM, and the OS — so a stored measurement is reused only on the same
machine, and the engine never has to answer "which machine is this".
"""
from __future__ import annotations

import platform

import psutil

from ara import db, detect


def machine_key() -> str:
    """Stable identity for this machine: chip · accelerator · total RAM · OS."""
    chip = detect.chip_name()
    accel = detect.accelerator(chip)
    ram_bytes = psutil.virtual_memory().total
    return "|".join([chip, accel.name, str(ram_bytes), platform.system()])


def save_calibration(con, engine: str, *, fixed_overhead_gb: float,
                     calibrated_at: str | None = None) -> None:
    """Persist this machine's measured overhead for *engine*."""
    db.upsert_machine(con, machine_key(), engine,
                      fixed_overhead_gb=fixed_overhead_gb,
                      calibrated_at=calibrated_at or db._now())


def get_calibration(con, engine: str) -> dict | None:
    """This machine's stored calibration for *engine*, or None if never calibrated."""
    return db.get_machine(con, machine_key(), engine)
