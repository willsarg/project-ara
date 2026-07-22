# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Canonical memory units for ARA's engine-free core."""
from __future__ import annotations

import math
from numbers import Real

BYTES_PER_GIB = 1 << 30
MEMORY_UNIT = "GiB"


def bytes_to_gib(value: int) -> float:
    """Return an exact byte count expressed in binary GiB."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("byte count must be a non-negative integer")
    return value / BYTES_PER_GIB


def gib_to_bytes(value: float) -> int:
    """Return a finite non-negative GiB quantity as its nearest byte count."""
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or value < 0):
        raise ValueError("GiB value must be finite non-negative")
    return round(float(value) * BYTES_PER_GIB)
