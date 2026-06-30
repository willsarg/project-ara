# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""GPU memory facts for this machine (NVIDIA / CUDA).

The crash wall is the GPU's **VRAM**. Unlike Apple's unified memory — where crossing the
working-set wall can hard-lock the whole machine — a CUDA over-allocation usually raises an
out-of-memory error and kills the *workload*, not the OS. But on a GPU that also drives the
display, VRAM exhaustion can still freeze the desktop or trip a driver reset (TDR). Same
discipline either way: find each model's safe context ceiling by staying under the VRAM
wall, never probing into it. VRAM is read LIVE per machine and treated as exact.

VRAM is read natively from **NVML** (`pynvml`, in the ``[cuda]`` extra) — the same library
``nvidia-smi`` itself wraps, but without forking a subprocess and parsing its text. That matters
most for the L5 watchdog, which re-reads live VRAM every 0.25 s during a probe: a fork per tick
both perturbs the very VRAM we're measuring and adds latency. ``nvidia-smi`` parsing is kept as a
tested fallback so the core still works with zero deps (or if NVML can't initialise).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

from .config import DEFAULT_MARGIN_GB

MIB_PER_GB = 1024  # nvidia-smi reports MiB; 1 GiB = 1024 MiB
BYTES_PER_GIB = 1024 ** 3  # NVML reports bytes
_PLAUSIBLE_MAX_GB = 2048.0  # a single card past this is an implausible read → distrust it

# NVML init + device-0 handle are cached: nvmlInit is ref-counted and re-fetching the handle each
# 0.25 s watchdog tick would be wasteful. Reset to None only matters for tests (they patch the seams).
_NVML = None
_NVML_HANDLE = None


@dataclass(frozen=True)
class GPULimits:
    device: str
    total_gb: float       # total VRAM on the card
    wall_gb: float        # the crash wall — total VRAM (allocating past free OOMs)
    free_gb: float        # VRAM free right now
    used_gb: float        # VRAM in use right now (display / other processes)

    def safe_threshold_gb(self, margin_gb: float = DEFAULT_MARGIN_GB) -> float:
        """The line we never let predicted peak cross — the wall minus a cushion."""
        return self.wall_gb - margin_gb


def _plausible_gb(gb: float) -> bool:
    """A VRAM figure we'll trust: positive and below an absurd single-card ceiling."""
    return 0.0 < gb < _PLAUSIBLE_MAX_GB


def _pynvml():
    """Import NVML and init it once, returning the module. Raises if unavailable."""
    global _NVML
    if _NVML is None:
        import pynvml
        pynvml.nvmlInit()
        _NVML = pynvml
    return _NVML


def _nvml_handle():
    """The cached device-0 NVML handle."""
    global _NVML_HANDLE
    if _NVML_HANDLE is None:
        _NVML_HANDLE = _pynvml().nvmlDeviceGetHandleByIndex(0)
    return _NVML_HANDLE


def _native_limits() -> GPULimits:
    """The first GPU's VRAM facts straight from NVML — no subprocess. Raises if NVML can't read."""
    nv = _pynvml()
    h = _nvml_handle()
    mem = nv.nvmlDeviceGetMemoryInfo(h)
    name = nv.nvmlDeviceGetName(h)
    if isinstance(name, bytes):            # pynvml < 11.5 returns bytes; newer returns str
        name = name.decode()
    total_gb = mem.total / BYTES_PER_GIB
    return GPULimits(
        device=name,
        total_gb=total_gb,
        wall_gb=total_gb,
        free_gb=mem.free / BYTES_PER_GIB,
        used_gb=mem.used / BYTES_PER_GIB,
    )


def _smi_query() -> str | None:
    """Raw ``nvidia-smi`` VRAM query stdout, or None if it can't be run (fallback path)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return out.stdout
    except Exception:
        return None


def _read_limits_smi() -> GPULimits | None:
    """The first NVIDIA GPU's VRAM facts via ``nvidia-smi`` text — the fallback parser."""
    out = _smi_query()
    if not out or not out.strip():
        return None
    first = out.strip().splitlines()[0]
    try:
        name, total, used, free = (x.strip() for x in first.split(","))
        total_gb = float(total) / MIB_PER_GB
        return GPULimits(
            device=name,
            total_gb=total_gb,
            wall_gb=total_gb,
            free_gb=float(free) / MIB_PER_GB,
            used_gb=float(used) / MIB_PER_GB,
        )
    except (ValueError, IndexError):
        return None


def read_limits() -> GPULimits | None:
    """The first NVIDIA GPU's VRAM facts, or None when there's no CUDA GPU here.

    Native NVML first; falls back to the ``nvidia-smi`` parse on any error or an implausible read
    (Rule #1: the read can only get more accurate, never less safe). When neither works (no GPU),
    returns None.
    """
    try:
        lim = _native_limits()
    except Exception:
        return _read_limits_smi()
    return lim if _plausible_gb(lim.total_gb) else _read_limits_smi()


# FlashAttention-2 needs an Ampere+ GPU (compute capability ≥ 8.0) AND the compiled flash_attn
# package. SDPA is the always-available fused fallback, so FA2 stays an availability-gated opt-in.
_FA2_MIN_MAJOR = 8


def flash_attn_capable() -> bool:
    """True only if this GPU can run FlashAttention-2: arch ≥ Ampere (sm_80) AND ``flash_attn`` is
    importable. Read via NVML (no torch import). Conservatively False if either can't be confirmed
    — better to fall back to SDPA than to claim an unsupported path."""
    import importlib.util
    if importlib.util.find_spec("flash_attn") is None:
        return False
    try:
        major, _ = _pynvml().nvmlDeviceGetCudaComputeCapability(_nvml_handle())
    except Exception:
        return False
    return major >= _FA2_MIN_MAJOR


# Hardware FP8 (E4M3) needs Ada (sm_89) or Hopper (sm_90); pre-Ada GPUs can't load FP8 weights.
_FP8_MIN_CC = (8, 9)


def fp8_capable() -> bool:
    """True only if this GPU has hardware FP8 (compute capability ≥ sm_89). NVML, no torch.
    Conservatively False if it can't be read — FP8 is then rejected rather than crashed."""
    try:
        cc = tuple(_pynvml().nvmlDeviceGetCudaComputeCapability(_nvml_handle()))
    except Exception:
        return False
    return cc >= _FP8_MIN_CC
