# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Runtime configuration shared by CLI and programmatic entry points."""
from __future__ import annotations

import math
import os
import sys

DEFAULT_MARGIN_GB = 2.0
MARGIN_ENV = "ARA_MLX_MARGIN_GB"
LEGACY_MARGIN_ENV = "WMX_SUITE_MARGIN_GB"


def margin_gb(value: float | str | None = None) -> float:
    """Return a validated safety margin; an explicit value overrides the environment."""
    if value is not None:
        raw = value
    elif MARGIN_ENV in os.environ:
        raw = os.environ[MARGIN_ENV]
    elif LEGACY_MARGIN_ENV in os.environ:
        print(
            f"ara-engine-mlx: {LEGACY_MARGIN_ENV} is deprecated; use {MARGIN_ENV}",
            file=sys.stderr,
        )
        raw = os.environ[LEGACY_MARGIN_ENV]
    else:
        raw = str(DEFAULT_MARGIN_GB)
    try:
        margin = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{MARGIN_ENV} / --margin must be a number") from exc
    if not math.isfinite(margin) or margin < 0:
        raise ValueError(f"{MARGIN_ENV} / --margin must be finite and non-negative")
    return margin
