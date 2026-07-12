# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Nested engine distribution packaging contracts."""
from __future__ import annotations

import tomllib
from pathlib import Path

from ara._engine_packages.mlx.ara_engine_mlx import config as mlx_config


_ROOT = Path(__file__).resolve().parent.parent


def test_mlx_engine_uses_native_distribution_and_package_identities():
    engine_root = _ROOT / "ara" / "_engine_packages" / "mlx"
    manifest = tomllib.loads((engine_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert manifest["project"]["name"] == "ara-engine-mlx"
    assert manifest["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "ara_engine_mlx",
    ]
    assert manifest["project"]["dependencies"] == ["mlx-lm>=0.31"]
    assert "nonllm" in manifest["project"]["optional-dependencies"]

    for relative_path in (
        "LICENSE",
        "NOTICE",
        "ara_engine_mlx/__init__.py",
        "ara_engine_mlx/device.py",
        "ara_engine_mlx/measure_one.py",
        "ara_engine_mlx/generate.py",
        "ara_engine_mlx/benchmark.py",
        "ara_engine_mlx/serve.py",
    ):
        assert (engine_root / relative_path).is_file(), relative_path


def test_mlx_margin_prefers_the_canonical_environment_variable(monkeypatch):
    monkeypatch.setenv("ARA_MLX_MARGIN_GB", "3.5")
    monkeypatch.setenv("WMX_SUITE_MARGIN_GB", "9.0")

    assert mlx_config.margin_gb() == 3.5


def test_mlx_margin_accepts_the_legacy_environment_variable_for_one_release(monkeypatch):
    monkeypatch.delenv("ARA_MLX_MARGIN_GB", raising=False)
    monkeypatch.setenv("WMX_SUITE_MARGIN_GB", "4.25")

    assert mlx_config.margin_gb() == 4.25
