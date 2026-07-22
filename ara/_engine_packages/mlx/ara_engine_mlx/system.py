# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Live Metal memory facts for this machine, expressed in binary GiB.

ARA derives its conservative governance boundary from Metal's recommended working-set
size and keeps an additional policy margin below it. Exact byte values remain available
for durable authority; GiB floats are used only for memory arithmetic and presentation.
"""
from __future__ import annotations

import ctypes
import re
import subprocess
import time
from dataclasses import dataclass

from . import units
from .config import DEFAULT_MARGIN_GB


@dataclass(frozen=True)
class SystemLimits:
    device: str
    memory_size_bytes: int
    recommended_working_set_bytes: int
    max_buffer_length_bytes: int
    total_gb: float
    wall_gb: float          # Metal recommended working-set size in GiB
    max_buffer_gb: float    # largest single allocation Metal allows
    swap_free_gb: float | None
    wired_now_gb: float      # OS-wired memory right now (baseline pressure)

    def safe_threshold_gb(self, margin_gb: float = DEFAULT_MARGIN_GB) -> float:
        """The line ARA never lets predicted peak cross (2 GiB cushion by default)."""
        return self.wall_gb - margin_gb


def device_limits() -> dict:
    import mlx.core as mx

    d = mx.device_info()
    memory_size = int(d.get("memory_size", 0))
    working_set = int(d.get("max_recommended_working_set_size", 0))
    max_buffer = int(d.get("max_buffer_length", 0))
    return {
        "device": str(d.get("device_name", "")),
        "memory_unit": units.MEMORY_UNIT,
        "memory_size_bytes": memory_size,
        "recommended_working_set_bytes": working_set,
        "max_buffer_length_bytes": max_buffer,
        "total_gb": units.bytes_to_gib(memory_size),
        "wall_gb": units.bytes_to_gib(working_set),
        "max_buffer_gb": units.bytes_to_gib(max_buffer),
    }


def macos_major() -> int:
    """Major macOS version (e.g. 15), or 0 if undetectable.

    Part of the per-machine profile key: a major OS bump shifts the ambient
    wired baseline enough to invalidate a stored cold-start overhead.
    """
    import platform
    try:
        ver = platform.mac_ver()[0]
        return int(ver.split(".")[0]) if ver else 0
    except (ValueError, IndexError):
        return 0


# --- Native Mach memory reads (no subprocess) ----------------------------------------
#
# wired/swap are read straight from the kernel via ctypes instead of spawning vm_stat /
# sysctl and parsing their text. host_statistics64(HOST_VM_INFO64).wire_count is *exactly*
# vm_stat's "Pages wired down" — the same source — so this is not a re-derivation; it is
# the upstream value without the fork. That matters for safety, not just speed: spawning a
# subprocess wires its own pages and perturbs the very baseline sample_settled_baseline()
# tries to read (measured: ~1000x faster, zero observer-effect jitter). Any failure or
# implausible value falls back to the text parse, so accuracy can only tighten, never drop.
_LIBSYSTEM = "/usr/lib/libSystem.B.dylib"
_HOST_VM_INFO64 = 4
_KERN_SUCCESS = 0
_PLAUSIBLE_MAX_GB = 2048.0  # any real Mac's wired memory is far below this; guards ABI garbage

_natural_t = ctypes.c_uint
_mach_msg_type_number_t = ctypes.c_uint

_LIB = None   # cached CDLL handle (loaded once)
_HOST = None  # cached mach host port (acquired once — re-acquiring leaks a port ref)


class _vm_statistics64(ctypes.Structure):
    """Mirror of <mach/vm_statistics.h> vm_statistics64_data_t (only wire_count is read)."""
    _fields_ = [
        ("free_count", _natural_t),
        ("active_count", _natural_t),
        ("inactive_count", _natural_t),
        ("wire_count", _natural_t),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", _natural_t),
        ("speculative_count", _natural_t),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", _natural_t),
        ("throttled_count", _natural_t),
        ("external_page_count", _natural_t),
        ("internal_page_count", _natural_t),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


_VM_INFO64_COUNT = ctypes.sizeof(_vm_statistics64) // ctypes.sizeof(ctypes.c_int)


class _xsw_usage(ctypes.Structure):
    """Mirror of <sys/sysctl.h> struct xsw_usage (swap totals, in bytes)."""
    _fields_ = [
        ("xsu_total", ctypes.c_uint64),
        ("xsu_avail", ctypes.c_uint64),
        ("xsu_used", ctypes.c_uint64),
        ("xsu_pagesize", ctypes.c_uint32),
        ("xsu_encrypted", ctypes.c_int),
    ]


def _libsystem():
    """The libSystem handle, loaded once. Default int restype is what we want for the
    kern_return_t / int return codes, so no restype/argtypes setup is needed."""
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(_LIBSYSTEM, use_errno=True)
    return _LIB


def _mach_host():
    """The mach host port, acquired once and cached (each mach_host_self() adds a send
    right; calling it per sample would slowly leak port references)."""
    global _HOST
    if _HOST is None:
        _HOST = _libsystem().mach_host_self()
    return _HOST


def _mach_wired_pages() -> int:
    """Wired page count from the kernel (== vm_stat 'Pages wired down'). Raises OSError."""
    stats = _vm_statistics64()
    count = _mach_msg_type_number_t(_VM_INFO64_COUNT)
    kr = _libsystem().host_statistics64(
        _mach_host(), _HOST_VM_INFO64, ctypes.byref(stats), ctypes.byref(count))
    if kr != _KERN_SUCCESS:
        raise OSError(f"host_statistics64 failed (kr={kr})")
    return stats.wire_count


def _mach_page_size() -> int:
    """VM page size in bytes from the kernel (16384 on Apple silicon). Raises OSError."""
    ps = ctypes.c_ulong(0)
    kr = _libsystem().host_page_size(_mach_host(), ctypes.byref(ps))
    if kr != _KERN_SUCCESS:
        raise OSError(f"host_page_size failed (kr={kr})")
    return ps.value


def _native_wired_gb() -> float:
    return units.bytes_to_gib(_mach_wired_pages() * _mach_page_size())


def _native_swap_free_gb() -> float:
    """Free swap in GiB via sysctlbyname('vm.swapusage'), matching the text parser's units
    (MiB/1024). Raises OSError on failure."""
    xsw = _xsw_usage()
    ln = ctypes.c_size_t(ctypes.sizeof(xsw))
    rc = _libsystem().sysctlbyname(b"vm.swapusage", ctypes.byref(xsw), ctypes.byref(ln),
                                   None, 0)
    if rc != _KERN_SUCCESS:
        raise OSError(f"sysctlbyname(vm.swapusage) failed (rc={rc})")
    return xsw.xsu_avail / (1024 ** 3)


def _plausible_gb(gb: float) -> bool:
    return 0.0 < gb < _PLAUSIBLE_MAX_GB


def _swap_free_gb_sysctl() -> float | None:
    """Fallback: parse `sysctl vm.swapusage` text."""
    try:
        out = subprocess.check_output(["sysctl", "vm.swapusage"]).decode()
    except Exception:
        return None
    m = re.search(r"free = ([\d.]+)([MG])", out)
    if not m:
        return None
    v = float(m.group(1))
    return v / 1024 if m.group(2) == "M" else v


def _wired_gb_vmstat() -> float:
    """Fallback: parse `vm_stat` 'Pages wired down'."""
    out = subprocess.check_output(["vm_stat"]).decode()
    page_size = 4096
    wired_pages = 0
    for line in out.splitlines():
        if "page size of" in line:
            page_size = int(line.split()[-2])
        if "Pages wired down" in line:
            wired_pages = int(line.split()[-1].strip("."))
    return units.bytes_to_gib(wired_pages * page_size)


def swap_free_gb() -> float | None:
    """Free swap in GiB. Native sysctlbyname read, falling back to the `sysctl` text parse."""
    try:
        return _native_swap_free_gb()
    except Exception:
        return _swap_free_gb_sysctl()


def wired_gb() -> float:
    """OS-wired memory in GiB — the metric that actually predicts a crash (MLX's own
    get_peak_memory() undercounts it ~40%, excluding the buffer cache the OS still wires).

    Read natively from the Mach kernel (host_statistics64) to avoid spawning vm_stat, whose
    fork perturbs the baseline being measured; falls back to the vm_stat text parse on any
    failure or implausible value, so the safety floor is never weakened."""
    try:
        gb = _native_wired_gb()
    except Exception:
        return _wired_gb_vmstat()
    return gb if _plausible_gb(gb) else _wired_gb_vmstat()


def sample_settled_baseline(settle: float = 0.5, n: int = 3, interval: float = 0.2) -> float:
    """OS-wired baseline after letting IOGPU settle, taking the MIN of several samples.

    When a Metal worker subprocess exits, macOS does not un-wire its pages synchronously —
    sampling immediately catches an artificially high reading and forces an over-conservative
    early stop. A short settle plus a min-of-samples reads the reclaimed floor instead.
    """
    time.sleep(settle)
    samples = [wired_gb()]
    for _ in range(max(1, n) - 1):
        time.sleep(interval)
        samples.append(wired_gb())
    return min(samples)


def read_limits() -> SystemLimits:
    d = device_limits()
    return SystemLimits(
        device=d["device"],
        memory_size_bytes=d["memory_size_bytes"],
        recommended_working_set_bytes=d["recommended_working_set_bytes"],
        max_buffer_length_bytes=d["max_buffer_length_bytes"],
        total_gb=d["total_gb"],
        wall_gb=d["wall_gb"],
        max_buffer_gb=d["max_buffer_gb"],
        swap_free_gb=swap_free_gb(),
        wired_now_gb=wired_gb(),
    )
