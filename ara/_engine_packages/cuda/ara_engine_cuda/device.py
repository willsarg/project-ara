# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Device limits + CUDA-overhead calibration as JSON workers — for ARA's out-of-process driver.

ARA owns no CUDA runtime dependency: it drives these in the CUDA engine's isolated env and reads back a single JSON
line (mirroring :mod:`ara_engine_cuda.measure_one` and the MLX engine's ``device``). The VRAM wall is read exactly
from nvidia-smi, so — unlike Apple's hidden cold-start overhead — the *budget* needs no
calibration. What calibration measures here is the fixed CUDA-context VRAM cost (cuBLAS/cuDNN),
which the per-context safety gate adds on top of model weights.

    python -m ara_engine_cuda.device limits [--margin G]
    python -m ara_engine_cuda.device calibrate [MODEL] [--margin G]
"""
from __future__ import annotations

import argparse
import json

from . import config, probe, system
from .config import DEFAULT_OVERHEAD_GB

__all__ = ["limits", "calibrate", "DEFAULT_OVERHEAD_GB", "main"]


def limits(margin_gb: float | None = None) -> dict:
    """The VRAM wall + safe budget as engine facts (ARA overlays its stored overhead).

    ``headroom_gb`` subtracts the *live* VRAM already in use (display + other processes), so it
    reflects what's actually free on a shared card right now. ``swap_free_gb`` is None — VRAM has
    no swap.
    """
    s = system.read_limits()
    if s is None:
        return {"error": "no NVIDIA GPU visible to nvidia-smi"}
    margin = config.margin_gb(margin_gb)
    safe = s.safe_threshold_gb(margin)
    return {
        "device": s.device,
        "total_gb": s.total_gb,
        "wall_gb": s.wall_gb,
        "safe_budget_gb": safe,
        "margin_gb": margin,
        "headroom_gb": safe - s.used_gb,
        "swap_free_gb": None,
        # Can this GPU run FlashAttention-2 (Ampere+ and the package present)? Lets ARA tell the
        # user upfront when a --flash-attn opt-in will fall back to SDPA. Cheap (NVML, no torch).
        "flash_attn_capable": system.flash_attn_capable(),
        # Hardware FP8 (Ada/Hopper, sm_89+) — lets ARA reject --weight-quant fp8 on older GPUs.
        "fp8_capable": system.fp8_capable(),
    }


def calibrate(model: str | None = None, margin_gb: float | None = None) -> dict:
    """Measure the CUDA context's fixed VRAM overhead (model-independent), as a dict ARA persists.

    *model* is accepted for argv symmetry with the Apple worker but ignored: the CUDA-context cost
    (torch init + cuBLAS/cuDNN) is the same regardless of which model loads after it. ARA computes
    the effective overhead as ``max(default, measured)``, so both are surfaced.
    """
    cal = probe.calibrate()
    if cal is None:
        return {"error": "calibration failed — CUDA unavailable or missing the [cuda] extra"}
    return {
        "device": cal.device,
        "measured_overhead_gb": cal.overhead_gb,
        "default_overhead_gb": DEFAULT_OVERHEAD_GB,
        "n_points": 1,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Device VRAM limits / CUDA-overhead calibration as JSON.")
    ap.add_argument("mode", choices=["limits", "calibrate"])
    ap.add_argument("model", nargs="?", default=None, help="ignored (worker argv symmetry)")
    ap.add_argument("--margin", type=float, default=None)
    args = ap.parse_args(argv)
    if args.mode == "calibrate":
        result = calibrate(args.model, margin_gb=args.margin)
    else:
        result = limits(margin_gb=args.margin)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
