"""Comprehensive hardware recon — OS-specific probes behind pure parsers.

I/O (subprocess/winreg/file reads) is isolated from parsing so parsers are unit-tested against
captured real output on any host. Read-only; engine-free; never escalates privilege.
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

GB = 1024 ** 3  # GiB — defined locally (NOT imported from detect) to avoid a circular import,
                # since detect imports hardware. Matches detect.GB exactly.


@dataclass(frozen=True)
class CpuInfo:
    brand: str | None = None
    vendor: str | None = None
    arch_id: str | None = None
    physical: int | None = None
    logical: int | None = None
    base_mhz: int | None = None
    max_mhz: int | None = None
    l1_kb: int | None = None
    l2_kb: int | None = None
    l3_kb: int | None = None
    features: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryModule:
    slot: str | None = None
    capacity_gb: float | None = None
    speed_mts: int | None = None
    manufacturer: str | None = None
    part_number: str | None = None


@dataclass(frozen=True)
class MemoryInfo:
    total_gb: float | None = None
    available_gb: float | None = None
    swap_gb: float | None = None
    kind: str | None = None
    speed_mts: int | None = None
    slots_used: int | None = None
    slots_total: int | None = None
    modules: list[MemoryModule] = field(default_factory=list)


@dataclass(frozen=True)
class Drive:
    model: str | None = None
    media: str | None = None
    size_gb: float | None = None


@dataclass(frozen=True)
class StorageInfo:
    free_gb: float | None = None
    drives: list[Drive] = field(default_factory=list)


@dataclass(frozen=True)
class BoardInfo:
    board_vendor: str | None = None
    board_model: str | None = None
    bios_version: str | None = None
    bios_date: str | None = None
    system_vendor: str | None = None
    system_model: str | None = None


@dataclass(frozen=True)
class Hardware:
    cpu: CpuInfo
    memory: MemoryInfo
    storage: StorageInfo
    board: BoardInfo


_PLACEHOLDERS = {"system manufacturer", "system product name", "to be filled by o.e.m.",
                 "default string", "o.e.m.", "none", ""}


def _clean(s: str | None) -> str | None:
    if not s:
        return None
    s = s.strip()
    return None if s.lower() in _PLACEHOLDERS else (s or None)


def _gib(n) -> float | None:
    try:
        return round(int(n) / GB, 1)
    except (TypeError, ValueError):
        return None


def _gb_dec(n) -> float | None:
    try:
        return round(int(n) / 1e9, 1)
    except (TypeError, ValueError):
        return None


def _run(cmd: list[str], timeout: float = 3) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


def _pwsh_json(args: list[str]) -> list[dict]:
    """Run a PowerShell expr emitting ConvertTo-Json; ALWAYS return a list (handles the
    single-object-vs-array quirk of PS 5.1). [] on any failure."""
    raw = _run(["powershell", "-NoProfile", "-Command", *args])
    if not raw:
        return []
    try:
        val = json.loads(raw)
    except Exception:
        return []
    return val if isinstance(val, list) else [val]


def _wmi_date(s: str | None) -> str | None:
    m = re.search(r"/Date\((\d+)", s or "")
    if not m:
        return None
    try:
        return datetime.fromtimestamp(int(m.group(1)) / 1000, tz=timezone.utc).date().isoformat()
    except Exception:
        return None
