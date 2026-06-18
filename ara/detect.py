"""Recon: observe the machine and pick a backend. Read-only — no profiling.

Everything here just *reads* what's already true about the host (memory, GPU,
disk, what's installed). It never stresses, benchmarks, or loads an ML engine,
so `ara detect` is instant and works on any box — even a bare one with no
backend for its hardware yet.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import psutil

GB = 1024 ** 3


# --------------------------------------------------------------------------- #
# backend choice
# --------------------------------------------------------------------------- #
def backend_name() -> str:
    """Return the backend module name for this machine.

    Maps to a module under ``ara.backends``. ``"unsupported"`` means we have no
    adapter for this hardware yet (detect still reports everything else).
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "apple"
    # later: NVIDIA detection -> "cuda"
    return "unsupported"


# --------------------------------------------------------------------------- #
# low-level reads
# --------------------------------------------------------------------------- #
def _sysctl(key: str) -> str | None:
    """Read a sysctl value, or None if unavailable (non-Darwin or error)."""
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def chip_name() -> str:
    """Best-effort human chip name, e.g. 'Apple M4 Pro'."""
    if platform.system() == "Darwin":
        brand = _sysctl("machdep.cpu.brand_string")
        if brand:
            return brand
    return platform.processor() or platform.machine() or "unknown"


def os_version() -> str:
    """Human OS string, e.g. 'macOS 15.3' or 'Linux'."""
    if platform.system() == "Darwin":
        ver = platform.mac_ver()[0]
        return f"macOS {ver}" if ver else "macOS"
    return platform.system() or "unknown"


def _memory_gb() -> tuple[float | None, float | None]:
    """(total, available-right-now) in GB. A snapshot read, not a benchmark."""
    try:
        vm = psutil.virtual_memory()
        return vm.total / GB, vm.available / GB
    except Exception:
        return None, None


def _cpu_counts() -> tuple[int | None, int | None]:
    """(physical, logical) core counts."""
    try:
        return psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)
    except Exception:
        return None, None


def _disk_free_gb() -> float | None:
    """Free space on the home volume, in GB (where models would land)."""
    try:
        return shutil.disk_usage(Path.home()).free / GB
    except Exception:
        return None


def _hf_cache_present() -> bool:
    candidates = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home) / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    return any(p.exists() for p in candidates)


def _ollama_present() -> bool:
    return shutil.which("ollama") is not None or (Path.home() / ".ollama").exists()


# --------------------------------------------------------------------------- #
# accelerator
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Accelerator:
    kind: str               # "apple" | "nvidia" | "none"
    name: str               # human name
    vram_gb: float | None   # discrete VRAM; None when unified (Apple) or unknown
    api: str | None         # "Metal" | "CUDA" | None


def accelerator(chip: str) -> Accelerator:
    """Identify the GPU by reading what the system already reports — no probing."""
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            first = out.stdout.strip().splitlines()[0]
            name, mem_mib = (x.strip() for x in first.split(","))
            return Accelerator("nvidia", name, round(float(mem_mib) / 1024, 1), "CUDA")
        except Exception:
            pass
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return Accelerator("apple", f"{chip} GPU", None, "Metal")
    return Accelerator("none", "none detected", None, None)


# --------------------------------------------------------------------------- #
# the machine snapshot
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Machine:
    system: str
    os_version: str
    chip: str
    arch: str
    cpu_physical: int | None
    cpu_logical: int | None
    ram_total_gb: float | None
    ram_available_gb: float | None
    accel: Accelerator
    disk_free_gb: float | None
    hf_cache: bool
    ollama: bool
    backend: str
    engine: str
    engine_ready: bool

    @property
    def supported(self) -> bool:
        return self.backend != "unsupported"


def profile() -> Machine:
    """Observe the host and choose a backend. Read-only — no engine import."""
    chip = chip_name()
    backend = backend_name()
    engine = "wmx-suite" if backend == "apple" else backend
    engine_ready = backend == "apple" and find_spec("wmx_suite") is not None
    total, available = _memory_gb()
    physical, logical = _cpu_counts()
    return Machine(
        system=platform.system(),
        os_version=os_version(),
        chip=chip,
        arch=platform.machine() or "unknown",
        cpu_physical=physical,
        cpu_logical=logical,
        ram_total_gb=total,
        ram_available_gb=available,
        accel=accelerator(chip),
        disk_free_gb=_disk_free_gb(),
        hf_cache=_hf_cache_present(),
        ollama=_ollama_present(),
        backend=backend,
        engine=engine,
        engine_ready=engine_ready,
    )
