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


def _run(cmd: list[str], timeout: float = 3, ignore_rc: bool = False) -> str | None:
    """Run *cmd*, return stdout (or None on error/timeout). With ``ignore_rc=True`` keep stdout
    even on a non-zero exit — some tools partial-succeed (e.g. `sysctl` exits 1 when one of many
    requested keys is unknown, yet still prints the keys that exist)."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout if (ignore_rc or out.returncode == 0) else None
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


# ---------------------------------------------------------------------------
# Task 2: CPU detail
# ---------------------------------------------------------------------------

def _sysctl_many(keys: list[str]) -> dict[str, str]:
    """Run `sysctl <keys>` (NOT `-n`) and parse 'key: value' lines → dict.

    `-n` prints values only (no key labels), so they can't be mapped; the plain form prints
    `key: value` for the keys that exist and silently omits absent ones (e.g. hw.cpufrequency on
    Apple Silicon) — note sysctl then exits non-zero, so keep stdout via ignore_rc. {} on failure."""
    raw = _run(["sysctl", *keys], ignore_rc=True)
    if not raw:
        return {}
    result: dict[str, str] = {}
    for line in raw.splitlines():
        if ": " in line:
            k, _, v = line.partition(": ")
            result[k.strip()] = v.strip()
    return result


def _winreg_str(subkey: str, name: str) -> str | None:
    """Read a REG_SZ value from HKLM.  Returns None on non-Windows or any error.
    Imports winreg lazily — the module is Windows-only and crashes on import elsewhere."""
    if platform.system() != "Windows":
        return None
    try:
        import winreg  # noqa: PLC0415 — intentional lazy import
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, subkey) as key:
            val, _ = winreg.QueryValueEx(key, name)
            return str(val).strip() or None
    except Exception:
        return None


def _cpu_macos(sysctl: dict[str, str]) -> "CpuInfo":
    """Parse sysctl output dict → CpuInfo.  Honest about Apple Silicon gaps."""
    brand = sysctl.get("machdep.cpu.brand_string")
    # Apple Silicon has no machdep.cpu.vendor; infer from brand.
    if brand and brand.startswith("Apple"):
        vendor = "Apple"
    else:
        vendor = sysctl.get("machdep.cpu.vendor") or None

    def _int(k: str) -> int | None:
        v = sysctl.get(k)
        try:
            return int(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    physical = _int("hw.physicalcpu")
    logical = _int("hw.logicalcpu")

    # Cache sizes — Apple Silicon has no L3.
    l1i = _int("hw.l1icachesize") or 0
    l1d = _int("hw.l1dcachesize") or 0
    l1_kb = (l1i + l1d) // 1024 if (l1i or l1d) else None
    l2_raw = _int("hw.l2cachesize")
    l2_kb = l2_raw // 1024 if l2_raw is not None else None
    l3_raw = _int("hw.l3cachesize")
    l3_kb = l3_raw // 1024 if l3_raw is not None else None

    # Clock — absent on Apple Silicon.
    freq_hz = _int("hw.cpufrequency")
    clock_mhz = int(freq_hz / 1e6) if freq_hz is not None else None

    # Feature flags — absent on Apple Silicon.
    feat_str = sysctl.get("machdep.cpu.features", "")
    features = feat_str.split() if feat_str else []

    return CpuInfo(
        brand=brand,
        vendor=vendor,
        physical=physical,
        logical=logical,
        base_mhz=clock_mhz,
        max_mhz=clock_mhz,
        l1_kb=l1_kb,
        l2_kb=l2_kb,
        l3_kb=l3_kb,
        features=features,
    )


def _cpu_windows(proc: dict, brand: str | None) -> "CpuInfo":
    """Parse Win32_Processor WMI dict → CpuInfo.
    `brand` is the registry ProcessorNameString (preferred); falls back to _clean(proc['Name'])."""
    effective_brand = brand if brand is not None else _clean(proc.get("Name", ""))
    return CpuInfo(
        brand=effective_brand,
        vendor=proc.get("Manufacturer") or None,
        arch_id=platform.processor() or None,
        physical=proc.get("NumberOfCores"),
        logical=proc.get("NumberOfLogicalProcessors"),
        max_mhz=proc.get("MaxClockSpeed"),
        l2_kb=proc.get("L2CacheSize"),
        l3_kb=proc.get("L3CacheSize"),
        features=[],  # WMI gap — honest, not fabricated
    )


def _linux_cpu_caches() -> dict[str, int]:
    """Read L1i/L1d/L2/L3 cache sizes from /sys/devices/system/cpu/cpu0/cache/.
    Returns a dict with keys 'l1i','l1d','l2','l3' → bytes (int). Missing → absent from dict."""
    caches: dict[str, int] = {}
    base = "/sys/devices/system/cpu/cpu0/cache"
    try:
        import glob
        for idx_dir in sorted(glob.glob(f"{base}/index*")):
            try:
                with open(f"{idx_dir}/level") as f:
                    level = f.read().strip()
                with open(f"{idx_dir}/type") as f:
                    kind = f.read().strip()   # Data, Instruction, Unified
                with open(f"{idx_dir}/size") as f:
                    raw = f.read().strip()    # e.g. "32K" or "256K"
                mult = 1024 if raw.endswith("K") else (1024 * 1024 if raw.endswith("M") else 1)
                size = int(raw.rstrip("KMkm")) * mult
                if level == "1" and kind == "Instruction":
                    caches["l1i"] = size
                elif level == "1" and kind == "Data":
                    caches["l1d"] = size
                elif level == "2":
                    caches["l2"] = size
                elif level == "3":
                    caches["l3"] = size
            except Exception:
                continue
    except Exception:
        pass
    return caches


def _cpu_linux(cpuinfo: str, caches: dict[str, int], logical: int | None) -> "CpuInfo":
    """Parse /proc/cpuinfo text + sysfs caches dict → CpuInfo."""
    brand: str | None = None
    vendor: str | None = None
    features: list[str] = []

    for line in cpuinfo.splitlines():
        if ": " not in line:
            continue
        k, _, v = line.partition(": ")
        k = k.strip()
        v = v.strip()
        if k == "model name" and brand is None:
            brand = v
        elif k == "vendor_id" and vendor is None:
            vendor = v
        elif k == "flags" and not features:
            features = v.split()

    l1i = caches.get("l1i", 0)
    l1d = caches.get("l1d", 0)
    l1_kb = (l1i + l1d) // 1024 if (l1i or l1d) else None
    l2 = caches.get("l2")
    l2_kb = l2 // 1024 if l2 is not None else None
    l3 = caches.get("l3")
    l3_kb = l3 // 1024 if l3 is not None else None

    return CpuInfo(
        brand=brand,
        vendor=vendor,
        logical=logical,
        l1_kb=l1_kb,
        l2_kb=l2_kb,
        l3_kb=l3_kb,
        features=features,
    )


_SYSCTL_CPU_KEYS = [
    "machdep.cpu.brand_string",
    "machdep.cpu.vendor",
    "hw.physicalcpu",
    "hw.logicalcpu",
    "hw.cpufrequency",
    "hw.l1icachesize",
    "hw.l1dcachesize",
    "hw.l2cachesize",
    "hw.l3cachesize",
    "machdep.cpu.features",
]


def cpu_info() -> CpuInfo:
    """Dispatch to the per-OS CPU parser and return a CpuInfo.  Never raises."""
    system = platform.system()
    try:
        if system == "Darwin":
            sysctl = _sysctl_many(_SYSCTL_CPU_KEYS)
            return _cpu_macos(sysctl)
        if system == "Windows":
            import psutil  # already a dep
            rows = _pwsh_json([
                "Get-WmiObject Win32_Processor | ConvertTo-Json -Compress"
            ])
            proc = rows[0] if rows else {}
            brand = _winreg_str(
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                "ProcessorNameString",
            )
            return _cpu_windows(proc, brand=brand)
        if system == "Linux":
            import psutil
            cpuinfo = _run(["cat", "/proc/cpuinfo"]) or ""
            caches = _linux_cpu_caches()
            logical = psutil.cpu_count(logical=True)
            return _cpu_linux(cpuinfo, caches, logical=logical)
    except Exception:
        pass
    return CpuInfo()
