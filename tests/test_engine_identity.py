# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Canonical engine identity mappings."""
from __future__ import annotations

import pytest

from ara.engine_identity import canonical_engine


@pytest.mark.parametrize(("raw", "expected"), [
    ("wmx", "mlx"), ("wmx-suite", "mlx"),
    ("wcx", "cuda"), ("wcx-suite", "cuda"),
    ("mlx", "mlx"), ("cuda", "cuda"),
    ("cuda-gguf", "cuda-gguf"), (None, None),
])
def test_canonical_engine_maps_only_exact_legacy_identities(raw, expected):
    assert canonical_engine(raw) == expected
