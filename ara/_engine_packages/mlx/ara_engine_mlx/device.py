# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Device limits + calibration as JSON workers — for ARA's out-of-process engine driver.

ARA owns no MLX runtime dependency: it drives these in the MLX engine's isolated env and reads back a single JSON
line (mirroring :mod:`ara_engine_mlx.measure_one`). Progress/console output goes to **stderr** so
stdout carries only the result object.

    python -m ara_engine_mlx.device limits [--margin G]
    python -m ara_engine_mlx.device calibrate [MODEL] [--margin G]
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config, probe, system, units
from .ui import Console


def limits(margin_gb: float | None = None) -> dict:
    """The Metal working-set limit + safe budget (engine facts only; ARA adds history)."""
    s = system.read_limits()
    margin = config.margin_gb(margin_gb)
    safe = s.safe_threshold_gb(margin)
    return {
        "device": s.device,
        "memory_unit": units.MEMORY_UNIT,
        "memory_size_bytes": s.memory_size_bytes,
        "recommended_working_set_bytes": s.recommended_working_set_bytes,
        "max_buffer_length_bytes": s.max_buffer_length_bytes,
        "safe_budget_bytes": units.gib_to_bytes(safe),
        "total_gb": s.total_gb,
        "wall_gb": s.wall_gb,
        "safe_budget_gb": safe,
        "margin_gb": margin,
        "headroom_gb": safe - s.wired_now_gb,
        "swap_free_gb": s.swap_free_gb,
    }


def calibrate(model: str | None, margin_gb: float | None = None) -> dict:
    """Run the MLX engine's guarded cold-start calibration; return what it measured (ARA persists it).
    The interactive console is routed to stderr so stdout stays JSON-only."""
    return probe.calibrate(model, margin_gb=margin_gb,
                           console=Console.from_args(stream=sys.stderr))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Device limits / calibration as JSON.")
    ap.add_argument("mode", choices=["limits", "calibrate"])
    ap.add_argument("model", nargs="?", default=None,
                    help="explicit model for calibrate (else the MLX engine auto-picks)")
    ap.add_argument("--margin", type=float, default=None)
    args = ap.parse_args(argv)
    if args.mode == "calibrate":
        result = calibrate(args.model, margin_gb=args.margin)
    else:
        result = limits(margin_gb=args.margin)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
