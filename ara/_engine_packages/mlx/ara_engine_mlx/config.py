# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Runtime configuration shared by CLI and programmatic entry points."""
from __future__ import annotations

import math
import os

DEFAULT_MARGIN_GB = 2.0
MARGIN_ENV = "ARA_MLX_MARGIN_GB"
LEGACY_MARGIN_ENV = "WMX_SUITE_MARGIN_GB"


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
