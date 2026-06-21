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
from datetime import datetime, timezone

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


# ---------------------------------------------------------------------------
# Task 3: Memory detail
# ---------------------------------------------------------------------------

_SMBIOS_MEM: dict[int, str] = {
    24: "DDR3",
    26: "DDR4",
    27: "LPDDR",
    28: "LPDDR2",
    29: "LPDDR3",
    30: "LPDDR4",
    34: "DDR5",
    35: "LPDDR5",
}


def _mem_windows(
    modules: list[dict],
    array: dict,
    totals: tuple[float, float, float],
) -> "MemoryInfo":
    """Parse Win32_PhysicalMemory rows + Win32_PhysicalMemoryArray dict → MemoryInfo.

    `totals` is (total_gb, available_gb, swap_gb) from psutil — passed in so the parser
    is pure/testable without live psutil calls.
    """
    total_gb, available_gb, swap_gb = totals
    parsed: list[MemoryModule] = []
    for m in modules:
        parsed.append(MemoryModule(
            slot=m.get("DeviceLocator"),
            capacity_gb=_gib(m.get("Capacity")),
            speed_mts=m.get("ConfiguredClockSpeed"),
            manufacturer=_clean(m.get("Manufacturer", "")),
            part_number=_clean(m.get("PartNumber", "")),
        ))

    kind: str | None = None
    speed_mts: int | None = None
    if modules:
        kind = _SMBIOS_MEM.get(modules[0].get("SMBIOSMemoryType", 0))
        speeds = [m.get("ConfiguredClockSpeed") for m in modules if m.get("ConfiguredClockSpeed")]
        speed_mts = max(speeds) if speeds else None

    slots_total = array.get("MemoryDevices") if array else None

    return MemoryInfo(
        total_gb=total_gb,
        available_gb=available_gb,
        swap_gb=swap_gb,
        kind=kind,
        speed_mts=speed_mts,
        slots_used=len(parsed) if parsed else None,
        slots_total=slots_total,
        modules=parsed,
    )


def _mem_macos(
    spmemory_text: str,
    totals: tuple[float, float, float],
) -> "MemoryInfo":
    """Parse `system_profiler SPMemoryDataType` text → MemoryInfo.

    Apple Silicon is soldered — there are no per-module rows, only totals + kind.
    modules=[] is intentional and honest.
    """
    total_gb, available_gb, swap_gb = totals
    kind: str | None = None

    for line in spmemory_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Type:"):
            kind = _clean(stripped.removeprefix("Type:").strip()) or None
            break

    return MemoryInfo(
        total_gb=total_gb,
        available_gb=available_gb,
        swap_gb=swap_gb,
        kind=kind,
        modules=[],
    )


def _mem_linux(
    meminfo_text: str,
    dmidecode_text: str | None,
    totals: tuple[float, float, float],
) -> "MemoryInfo":
    """Parse /proc/meminfo + optional dmidecode output → MemoryInfo.

    Per-module detail requires root (`dmidecode -t memory`). When dmidecode_text is None
    (non-root or tool absent) modules=[] — honest gap, renderer shows "needs root".
    """
    total_gb, available_gb, swap_gb = totals

    # Parse per-module from dmidecode if we have it.
    modules: list[MemoryModule] = []
    kind: str | None = None
    slots_total: int | None = None
    slots_used: int | None = None
    if dmidecode_text:
        current: dict[str, str] = {}
        all_blocks: list[dict[str, str]] = []
        for line in dmidecode_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("Memory Device") and not stripped.startswith("Memory Device Array"):
                if current:
                    all_blocks.append(current)
                current = {}
            elif ":" in stripped and current is not None:
                k, _, v = stripped.partition(":")
                current[k.strip()] = v.strip()
        if current:
            all_blocks.append(current)

        slots_total = len(all_blocks)
        for block in all_blocks:
            size_str = block.get("Size", "").strip()
            # Skip empty slots: no Size field, "No Module Installed", or "Unknown".
            if not size_str or size_str.lower() in ("no module installed", "unknown"):
                continue
            mod = _parse_dmidecode_module(block)
            modules.append(mod)
            if kind is None:
                kind = block.get("Type")
        slots_used = len(modules) if modules else None
        if slots_total == 0:
            slots_total = None

    return MemoryInfo(
        total_gb=total_gb,
        available_gb=available_gb,
        swap_gb=swap_gb,
        kind=kind,
        slots_used=slots_used,
        slots_total=slots_total,
        modules=modules,
    )


def _parse_dmidecode_module(fields: dict[str, str]) -> "MemoryModule":
    """Turn a dmidecode 'Memory Device' field dict → MemoryModule."""
    slot = fields.get("Locator") or None
    manufacturer = _clean(fields.get("Manufacturer", ""))
    part_number = _clean(fields.get("Part Number", ""))

    # Speed: "Configured Memory Speed" preferred, fallback "Speed".
    speed_str = fields.get("Configured Memory Speed") or fields.get("Speed") or ""
    # "3200 MT/s" → 3200
    speed_mts: int | None = None
    m = re.match(r"(\d+)", speed_str)
    if m:
        speed_mts = int(m.group(1))

    # Capacity: "Size: 8 GB" → 8.0; "Size: 4096 MB" → 4.0; absent/unknown → None.
    capacity_gb: float | None = None
    size_str = fields.get("Size", "").strip()
    size_m = re.match(r"(\d+)\s*(GB|MB)", size_str, re.IGNORECASE)
    if size_m:
        size_val = float(size_m.group(1))
        unit = size_m.group(2).upper()
        capacity_gb = round(size_val / 1024 if unit == "MB" else size_val, 1)

    return MemoryModule(
        slot=slot,
        capacity_gb=capacity_gb,
        speed_mts=speed_mts,
        manufacturer=manufacturer,
        part_number=part_number,
    )


# ---------------------------------------------------------------------------
# Task 4: Storage detail
# ---------------------------------------------------------------------------

def _drives_windows(physicaldisks: list[dict]) -> list["Drive"]:
    """Parse Get-PhysicalDisk rows → list[Drive].

    Media classification (plan spec, verbatim priority order):
      BusType "NVMe"                       → "nvme-ssd"
      MediaType "SSD" + BusType "SATA"     → "sata-ssd"
      MediaType "HDD"                      → "hdd"
      BusType "USB"                        → "usb"
      else                                 → "unknown"
    """
    drives: list[Drive] = []
    for disk in physicaldisks:
        bus = disk.get("BusType", "")
        media_type = disk.get("MediaType", "")
        if bus == "NVMe":
            media = "nvme-ssd"
        elif media_type == "SSD" and bus == "SATA":
            media = "sata-ssd"
        elif media_type == "HDD":
            media = "hdd"
        elif bus == "USB":
            media = "usb"
        else:
            media = "unknown"
        drives.append(Drive(
            model=disk.get("FriendlyName") or None,
            media=media,
            size_gb=_gb_dec(disk.get("Size")),
        ))
    return drives


def _drives_macos(spnvme_text: str) -> list["Drive"]:
    """Parse `system_profiler SPNVMeDataType` text → list[Drive].

    system_profiler emits NVMe drive blocks where each drive is identified by a group of
    indented key-value lines. The fixture (real Apple M4 Pro) shows Capacity before Model:

      Capacity:          500.28 GB (500,277,792,768 bytes)
      Model:                 APPLE SSD AP0512Z

    We scan each block independently: a blank line or deeper-nested heading signals a block
    boundary. Simpler approach: collect all Model + Capacity lines globally then zip them.
    Since system_profiler lists drives sequentially (one Model per drive, one Capacity per
    drive), we pair them positionally after sorting by line number.
    """
    models: list[tuple[int, str]] = []      # (lineno, value)
    capacities: list[tuple[int, float | None]] = []  # (lineno, size_gb)

    for i, line in enumerate(spnvme_text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("Model:"):
            val = stripped.removeprefix("Model:").strip() or None
            if val:
                models.append((i, val))
        elif stripped.startswith("Capacity:"):
            m = re.search(r"\(([0-9,]+)\s+bytes\)", stripped)
            if m:
                raw_bytes = int(m.group(1).replace(",", ""))
                size_gb = _gb_dec(raw_bytes)
            else:
                size_gb = None
            capacities.append((i, size_gb))

    # Pair each Model line with the nearest Capacity line in the same block.
    # Strategy: for each model, find the capacity line closest to it (min abs distance).
    drives: list[Drive] = []
    used_cap_indices: set[int] = set()
    for _lineno, model_val in models:
        best_idx: int | None = None
        best_dist = float("inf")
        for ci, (cap_lineno, _size) in enumerate(capacities):
            if ci in used_cap_indices:
                continue
            dist = abs(cap_lineno - _lineno)
            if dist < best_dist:
                best_dist = dist
                best_idx = ci
        size_gb: float | None = None
        if best_idx is not None:
            size_gb = capacities[best_idx][1]
            used_cap_indices.add(best_idx)
        drives.append(Drive(model=model_val, media="nvme-ssd", size_gb=size_gb))

    return drives


def _drives_linux(lsblk_json: str) -> list["Drive"]:
    """Parse `lsblk -d -b -o NAME,MODEL,SIZE,ROTA,TRAN -J` JSON → list[Drive].

    Media classification:
      ROTA == "1"  → "hdd"
      ROTA == "0", TRAN == "nvme"  → "nvme-ssd"
      ROTA == "0", TRAN == "sata"  → "sata-ssd"
      ROTA == "0", TRAN == "usb"   → "usb"
      else                          → "unknown"
    """
    try:
        data = json.loads(lsblk_json)
    except Exception:
        return []
    drives: list[Drive] = []
    for dev in data.get("blockdevices", []):
        # util-linux >= 2.33 emits rota as a JSON boolean (true/false); older versions emit
        # the string "1"/"0". Normalise to lowercase string so both shapes work.
        rota = str(dev.get("rota", "")).strip().lower()
        is_hdd = rota in ("1", "true")
        tran = (dev.get("tran") or "").strip().lower()
        if is_hdd:
            media = "hdd"
        elif tran == "nvme":
            media = "nvme-ssd"
        elif tran == "sata":
            media = "sata-ssd"
        elif tran == "usb":
            media = "usb"
        else:
            media = "unknown"
        size_raw = dev.get("size")
        drives.append(Drive(
            model=dev.get("model") or None,
            media=media,
            size_gb=_gb_dec(size_raw),
        ))
    return drives


def _disk_free_gb() -> float | None:
    """Return free GiB of the home partition, using GiB (1024**3) to match the pre-existing
    detect._disk_free_gb() convention and keep the disk_free_gb field stable."""
    try:
        import shutil
        from pathlib import Path
        return shutil.disk_usage(Path.home()).free / GB
    except Exception:
        return None


def storage_info() -> "StorageInfo":
    """Dispatch to the per-OS storage parser and return a StorageInfo.  Never raises."""
    system = platform.system()
    free_gb = _disk_free_gb()
    try:
        if system == "Darwin":
            raw = _run(["system_profiler", "SPNVMeDataType"], timeout=15) or ""
            return StorageInfo(free_gb=free_gb, drives=_drives_macos(raw))
        if system == "Windows":
            disks = _pwsh_json(["Get-PhysicalDisk | ConvertTo-Json -Compress"])
            return StorageInfo(free_gb=free_gb, drives=_drives_windows(disks))
        if system == "Linux":
            raw = _run(["lsblk", "-d", "-b", "-o", "NAME,MODEL,SIZE,ROTA,TRAN", "-J"]) or ""
            return StorageInfo(free_gb=free_gb, drives=_drives_linux(raw))
    except Exception:
        pass
    return StorageInfo(free_gb=free_gb)


def _psutil_totals() -> tuple[float, float, float]:
    """Return (total_gb, available_gb, swap_gb) from psutil."""
    import psutil
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    total_gb = round(vm.total / GB, 1)
    available_gb = round(vm.available / GB, 1)
    swap_gb = round(sw.total / GB, 1)
    return total_gb, available_gb, swap_gb


def memory_info() -> "MemoryInfo":
    """Dispatch to the per-OS memory parser and return a MemoryInfo. Never raises."""
    system = platform.system()
    try:
        totals = _psutil_totals()
        if system == "Darwin":
            raw = _run(["system_profiler", "SPMemoryDataType"], timeout=15) or ""
            return _mem_macos(raw, totals)
        if system == "Windows":
            modules = _pwsh_json([
                "Get-CimInstance Win32_PhysicalMemory | ConvertTo-Json -Compress"
            ])
            array_rows = _pwsh_json([
                "Get-CimInstance Win32_PhysicalMemoryArray | ConvertTo-Json -Compress"
            ])
            array = array_rows[0] if array_rows else {}
            return _mem_windows(modules, array, totals)
        if system == "Linux":
            meminfo = _run(["cat", "/proc/meminfo"]) or ""
            dmidecode_text: str | None = None
            if os.geteuid() == 0:
                dmidecode_text = _run(["dmidecode", "-t", "memory"])
            return _mem_linux(meminfo, dmidecode_text, totals)
    except Exception:
        pass
    return MemoryInfo()


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


# ---------------------------------------------------------------------------
# Task 5: Board / firmware detail
# ---------------------------------------------------------------------------

# Path to Linux DMI id sysfs directory — module-level so tests can monkeypatch it.
_DMI_ID_PATH = "/sys/class/dmi/id"

# DMI file names → BoardInfo field mapping
_LINUX_DMI_FIELDS = [
    ("board_vendor", "board_vendor"),
    ("board_name",   "board_model"),
    ("bios_version", "bios_version"),
    ("bios_date",    "bios_date"),
    ("sys_vendor",   "system_vendor"),
    ("product_name", "system_model"),
]


def _read_dmi_file(filename: str) -> str | None:
    """Read a single /sys/class/dmi/id/<filename> file; return None on missing/error."""
    try:
        with open(f"{_DMI_ID_PATH}/{filename}") as f:
            return f.read().strip() or None
    except Exception:
        return None


def _board_linux(dmi: dict[str, str | None]) -> "BoardInfo":
    """Pure parser: turn a dict of DMI file values → BoardInfo.

    The dict keys are the DMI filenames (board_vendor, board_name, bios_version,
    bios_date, sys_vendor, product_name). Each value has already been read from
    /sys/class/dmi/id; missing files map to None. _clean() handles placeholders.
    """
    return BoardInfo(
        board_vendor=_clean(dmi.get("board_vendor")),
        board_model=_clean(dmi.get("board_name")),
        bios_version=_clean(dmi.get("bios_version")),
        bios_date=_clean(dmi.get("bios_date")),
        system_vendor=_clean(dmi.get("sys_vendor")),
        system_model=_clean(dmi.get("product_name")),
    )


def _board_windows(baseboard: dict, bios: dict, system: dict) -> "BoardInfo":
    """Parse Win32_BaseBoard + Win32_BIOS + Win32_ComputerSystem WMI dicts → BoardInfo.

    system_vendor/model run through _clean so custom-build placeholders ("System manufacturer",
    "System Product Name") → None — exactly what willw11 returns.
    """
    return BoardInfo(
        board_vendor=_clean(baseboard.get("Manufacturer")),
        board_model=_clean(baseboard.get("Product")),
        bios_version=_clean(bios.get("SMBIOSBIOSVersion")),
        bios_date=_wmi_date(bios.get("ReleaseDate")),
        system_vendor=_clean(system.get("Manufacturer")),
        system_model=_clean(system.get("Model")),
    )


def _board_macos(sphardware_text: str) -> "BoardInfo":
    """Parse `system_profiler SPHardwareDataType` text → BoardInfo.

    Macs have no separate motherboard concept — board_vendor/board_model = None.
    system_vendor is always "Apple" when we successfully parse hardware info.
    system_model = Model Name line value.
    bios_version = System Firmware Version line value.
    """
    model_name: str | None = None
    firmware_version: str | None = None

    for line in sphardware_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Model Name:"):
            model_name = _clean(stripped.removeprefix("Model Name:").strip())
        elif stripped.startswith("System Firmware Version:"):
            firmware_version = _clean(stripped.removeprefix("System Firmware Version:").strip())

    # Only set system_vendor=Apple if we actually found hardware info
    system_vendor = "Apple" if model_name is not None else None

    return BoardInfo(
        board_vendor=None,
        board_model=None,
        bios_version=firmware_version,
        bios_date=None,   # macOS exposes no BIOS date in SPHardwareDataType
        system_vendor=system_vendor,
        system_model=model_name,
    )


def board_info() -> "BoardInfo":
    """Dispatch to the per-OS board/firmware parser and return a BoardInfo.  Never raises."""
    system = platform.system()
    try:
        if system == "Darwin":
            raw = _run(["system_profiler", "SPHardwareDataType"], timeout=15) or ""
            return _board_macos(raw)
        if system == "Windows":
            baseboard_rows = _pwsh_json([
                "Get-CimInstance Win32_BaseBoard | ConvertTo-Json -Compress"
            ])
            bios_rows = _pwsh_json([
                "Get-CimInstance Win32_BIOS | ConvertTo-Json -Compress"
            ])
            system_rows = _pwsh_json([
                "Get-CimInstance Win32_ComputerSystem | ConvertTo-Json -Compress"
            ])
            baseboard = baseboard_rows[0] if baseboard_rows else {}
            bios = bios_rows[0] if bios_rows else {}
            sys_dict = system_rows[0] if system_rows else {}
            return _board_windows(baseboard, bios, sys_dict)
        if system == "Linux":
            dmi: dict[str, str | None] = {}
            for filename, _ in _LINUX_DMI_FIELDS:
                dmi[filename] = _read_dmi_file(filename)
            return _board_linux(dmi)
    except Exception:
        pass
    return BoardInfo()


# ---------------------------------------------------------------------------
# probe() — bundle all four into a Hardware
# ---------------------------------------------------------------------------

def probe() -> "Hardware":
    """Probe all hardware subsystems and return a bundled Hardware dataclass.

    Each subsystem dispatcher is fail-soft and never raises; worst case you get empty
    dataclasses with all-None fields. Task 6 will embed this into detect.Machine.
    """
    return Hardware(
        cpu=cpu_info(),
        memory=memory_info(),
        storage=storage_info(),
        board=board_info(),
    )


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
