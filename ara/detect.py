"""Identify the machine and pick a backend. Stdlib only — no ML imports.

This is core: it must stay importable on any OS without pulling a heavy engine.
"""
from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from importlib.util import find_spec


def backend_name() -> str:
    """Return the backend module name for this machine.

    Maps to a module under ``ara.backends``. ``"unsupported"`` means we have no
    adapter for this hardware yet.
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "apple"
    # later: NVIDIA detection -> "cuda"
    return "unsupported"


def chip_name() -> str:
    """Best-effort human chip name, e.g. 'Apple M4 Pro'."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            name = out.stdout.strip()
            if name:
                return name
        except Exception:
            pass
    return platform.processor() or platform.machine() or "unknown"


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


def total_ram_gb() -> float | None:
    """Total physical/unified memory in GB (Apple-marketed GiB), or None."""
    raw = _sysctl("hw.memsize")
    if raw and raw.isdigit():
        return int(raw) / 1024**3
    return None


def os_version() -> str:
    """Human OS string, e.g. 'macOS 15.3' or 'Linux'."""
    if platform.system() == "Darwin":
        ver = platform.mac_ver()[0]
        return f"macOS {ver}" if ver else "macOS"
    return platform.system() or "unknown"


@dataclass(frozen=True)
class Machine:
    """A cheap, ML-free snapshot of the host and the backend ARA picked for it."""

    system: str
    os_version: str
    chip: str
    arch: str
    ram_gb: float | None
    backend: str
    engine: str
    engine_ready: bool

    @property
    def supported(self) -> bool:
        return self.backend != "unsupported"


def profile() -> Machine:
    """Inspect the host and choose a backend. Pure stdlib — no engine import."""
    backend = backend_name()
    engine = "wmx-suite" if backend == "apple" else backend
    engine_ready = backend == "apple" and find_spec("wmx_suite") is not None
    return Machine(
        system=platform.system(),
        os_version=os_version(),
        chip=chip_name(),
        arch=platform.machine() or "unknown",
        ram_gb=total_ram_gb(),
        backend=backend,
        engine=engine,
        engine_ready=engine_ready,
    )
