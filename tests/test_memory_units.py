# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Exact byte/GiB conversions at ARA's public memory boundary."""
from __future__ import annotations

import math

import pytest

from ara import memory_units


def test_bytes_to_gib_uses_binary_units() -> None:
    assert memory_units.bytes_to_gib(1_000_000_000) == pytest.approx(
        0.9313225746154785)
    assert memory_units.bytes_to_gib(1_073_741_824) == 1.0
    assert memory_units.bytes_to_gib(19_069_665_280) == 17.760009765625


def test_gib_to_bytes_preserves_exact_integral_boundaries() -> None:
    assert memory_units.gib_to_bytes(1.0) == 1_073_741_824
    assert memory_units.gib_to_bytes(2.0) == 2_147_483_648
    assert memory_units.gib_to_bytes(17.760009765625) == 19_069_665_280


@pytest.mark.parametrize("value", [-1.0, math.inf, -math.inf, math.nan, True])
def test_gib_to_bytes_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        memory_units.gib_to_bytes(value)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_bytes_to_gib_rejects_non_byte_counts(value) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        memory_units.bytes_to_gib(value)

