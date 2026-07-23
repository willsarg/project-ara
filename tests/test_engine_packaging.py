# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Nested engine distribution packaging contracts."""
from __future__ import annotations

import tomllib
from pathlib import Path

from ara._engine_packages.cuda.ara_engine_cuda import config as cuda_config
from ara._engine_packages.mlx.ara_engine_mlx import config as mlx_config


_ROOT = Path(__file__).resolve().parent.parent
_ENGINE_PACKAGES = _ROOT / "ara" / "_engine_packages"


def test_sdist_excludes_coordinator_local_artifacts():
    manifest = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    excludes = set(manifest["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"])

    assert {
        "/coordinator/node_modules",
        "/coordinator/.next",
        "/coordinator/out",
        "/coordinator/*.tsbuildinfo",
        "/coordinator/coverage",
        "/coordinator/data",
        "/coordinator/.env",
        "/coordinator/.env.local",
    } <= excludes


def test_wheel_carries_only_the_coordinator_runtime_build_context():
    manifest = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = manifest["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert force_include == {
        "coordinator/.dockerignore": "ara/_hub_source/.dockerignore",
        "coordinator/Dockerfile": "ara/_hub_source/Dockerfile",
        "coordinator/compose.yaml": "ara/_hub_source/compose.yaml",
        "coordinator/next.config.ts": "ara/_hub_source/next.config.ts",
        "coordinator/package-lock.json": "ara/_hub_source/package-lock.json",
        "coordinator/package.json": "ara/_hub_source/package.json",
        "coordinator/public": "ara/_hub_source/public",
        "coordinator/src": "ara/_hub_source/src",
        "coordinator/tsconfig.json": "ara/_hub_source/tsconfig.json",
    }


def test_retired_revendor_script_is_absent():
    assert not (_ROOT / "scripts" / "vendor_engine.py").exists()


def test_native_engine_roots_retain_manifests_and_legal_sidecars():
    for engine in ("mlx", "cuda"):
        engine_root = _ENGINE_PACKAGES / engine
        for sidecar in ("pyproject.toml", "LICENSE", "NOTICE"):
            assert (engine_root / sidecar).is_file(), f"{engine}/{sidecar}"


def test_mlx_engine_uses_native_distribution_and_package_identities():
    engine_root = _ENGINE_PACKAGES / "mlx"
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


def test_mlx_margin_prefers_the_canonical_environment_variable(monkeypatch, capsys):
    monkeypatch.setenv("ARA_MLX_MARGIN_GB", "3.5")
    monkeypatch.setenv("WMX_SUITE_MARGIN_GB", "9.0")

    assert mlx_config.margin_gb() == 3.5
    assert capsys.readouterr().err == ""


def test_mlx_margin_accepts_the_legacy_environment_variable_for_one_release(monkeypatch, capsys):
    monkeypatch.delenv("ARA_MLX_MARGIN_GB", raising=False)
    monkeypatch.setenv("WMX_SUITE_MARGIN_GB", "4.25")

    assert mlx_config.margin_gb() == 4.25
    assert capsys.readouterr().err == (
        "ara-engine-mlx: WMX_SUITE_MARGIN_GB is deprecated; use ARA_MLX_MARGIN_GB\n")


def test_mlx_explicit_margin_suppresses_legacy_warning(monkeypatch, capsys):
    monkeypatch.setenv("WMX_SUITE_MARGIN_GB", "9.0")
    assert mlx_config.margin_gb(2.25) == 2.25
    assert capsys.readouterr().err == ""


def test_cuda_engine_uses_native_distribution_and_package_identities():
    engine_root = _ENGINE_PACKAGES / "cuda"
    manifest = tomllib.loads((engine_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert manifest["project"]["name"] == "ara-engine-cuda"
    assert manifest["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "ara_engine_cuda",
    ]
    assert manifest["project"]["dependencies"] == []
    assert manifest["project"]["optional-dependencies"]["cuda"] == [
        "torch>=2.4",
        "transformers>=4.45",
        "nvidia-ml-py>=12",
        "hqq>=0.2",
        "bitsandbytes>=0.43",
        "accelerate>=0.30",
    ]
    assert "readme" not in manifest["project"]
    assert "scripts" not in manifest["project"]

    for relative_path in (
        "LICENSE",
        "NOTICE",
        "ara_engine_cuda/__init__.py",
        "ara_engine_cuda/device.py",
        "ara_engine_cuda/measure_one.py",
        "ara_engine_cuda/generate.py",
        "ara_engine_cuda/benchmark.py",
    ):
        assert (engine_root / relative_path).is_file(), relative_path


def test_cuda_margin_prefers_the_canonical_environment_variable(monkeypatch, capsys):
    monkeypatch.setenv("ARA_CUDA_MARGIN_GB", "2.5")
    monkeypatch.setenv("WCX_SUITE_MARGIN_GB", "8.0")

    assert cuda_config.margin_gb() == 2.5
    assert capsys.readouterr().err == ""


def test_cuda_margin_accepts_the_legacy_environment_variable_for_one_release(monkeypatch, capsys):
    monkeypatch.delenv("ARA_CUDA_MARGIN_GB", raising=False)
    monkeypatch.setenv("WCX_SUITE_MARGIN_GB", "3.25")

    assert cuda_config.margin_gb() == 3.25
    assert capsys.readouterr().err == (
        "ara-engine-cuda: WCX_SUITE_MARGIN_GB is deprecated; use ARA_CUDA_MARGIN_GB\n")


def test_cuda_explicit_margin_suppresses_legacy_warning(monkeypatch, capsys):
    monkeypatch.setenv("WCX_SUITE_MARGIN_GB", "9.0")
    assert cuda_config.margin_gb(1.25) == 1.25
    assert capsys.readouterr().err == ""
