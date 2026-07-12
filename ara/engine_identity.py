# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Canonical public and persisted engine identities."""
from __future__ import annotations

LEGACY_ENGINE_ALIASES = {"wmx": "mlx", "wcx": "cuda"}
_LEGACY_PACKAGE_LABELS = {"wmx-suite": "mlx", "wcx-suite": "cuda"}


def canonical_engine(value: str | None) -> str | None:
    if value is None:
        return None
    return LEGACY_ENGINE_ALIASES.get(value, _LEGACY_PACKAGE_LABELS.get(value, value))
