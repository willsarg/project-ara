# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Runtime configuration shared by CLI and programmatic entry points."""
from __future__ import annotations

import math
import os

# VRAM cushion kept under the wall. Smaller than MLX's 2 GB because consumer NVIDIA
# cards are smaller (an 8 GB card can't spare 2 GB) — refined during build-out.
DEFAULT_MARGIN_GB = 1.0
MARGIN_ENV = "ARA_CUDA_MARGIN_GB"
LEGACY_MARGIN_ENV = "WCX_SUITE_MARGIN_GB"

# Conservative floor for the CUDA-context VRAM overhead (cuBLAS/cuDNN + context) used by the
# safety gate when no live calibration is available. Real measurement (device.calibrate) almost
# always exceeds this; the floor only keeps the gate honest on an uncalibrated machine.
DEFAULT_OVERHEAD_GB = 0.6


def margin_gb(value: float | str | None = None) -> float:
    """Return a validated safety margin; an explicit value overrides the environment."""
    raw = (
        os.environ.get(MARGIN_ENV, os.environ.get(LEGACY_MARGIN_ENV, str(DEFAULT_MARGIN_GB)))
        if value is None else value
    )
    try:
        margin = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{MARGIN_ENV} / --margin must be a number") from exc
    if not math.isfinite(margin) or margin < 0:
        raise ValueError(f"{MARGIN_ENV} / --margin must be finite and non-negative")
    return margin
