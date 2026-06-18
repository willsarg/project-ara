"""Identify the machine and pick a backend. Stdlib only — no ML imports.

This is core: it must stay importable on any OS without pulling a heavy engine.
"""
from __future__ import annotations

import platform
import subprocess


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
