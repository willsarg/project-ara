# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The independently installable MLX worker uses the same binary memory contract."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from ara._engine_packages.mlx.ara_engine_mlx import models, probe_worker, system, units


def test_mlx_worker_bytes_to_gib_matches_exact_metal_values() -> None:
    assert units.MEMORY_UNIT == "GiB"
    assert units.bytes_to_gib(25_769_803_776) == 24.0
    assert units.bytes_to_gib(19_069_665_280) == 17.760009765625


def test_mlx_worker_margin_is_exactly_binary() -> None:
    wall_bytes = 19_069_665_280
    margin_bytes = units.gib_to_bytes(2.0)
    assert margin_bytes == 2_147_483_648
    assert units.bytes_to_gib(wall_bytes - margin_bytes) == 15.760009765625


@pytest.mark.parametrize("value", [-0.1, float("inf"), float("nan"), False])
def test_mlx_worker_rejects_invalid_gib_values(value) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        units.gib_to_bytes(value)


def test_device_limits_preserves_exact_bytes_and_derives_gib(monkeypatch) -> None:
    info = {
        "device_name": "Apple M4 Pro",
        "memory_size": 25_769_803_776,
        "max_recommended_working_set_size": 19_069_665_280,
        "max_buffer_length": 9_000_000_000,
    }
    core = SimpleNamespace(device_info=lambda: info)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=core))

    result = system.device_limits()

    assert result == {
        "device": "Apple M4 Pro",
        "memory_unit": "GiB",
        "memory_size_bytes": 25_769_803_776,
        "recommended_working_set_bytes": 19_069_665_280,
        "max_buffer_length_bytes": 9_000_000_000,
        "total_gb": 24.0,
        "wall_gb": 17.760009765625,
        "max_buffer_gb": pytest.approx(8.381903171539307),
    }


def test_mach_wired_memory_is_binary_gib(monkeypatch) -> None:
    monkeypatch.setattr(system, "_mach_wired_pages", lambda: 1_000_000)
    monkeypatch.setattr(system, "_mach_page_size", lambda: 1_000)
    assert system._native_wired_gb() == pytest.approx(0.9313225746154785)


def test_probe_worker_wired_memory_is_binary_gib(monkeypatch) -> None:
    output = (b"Mach Virtual Memory Statistics: (page size of 1000 bytes)\n"
              b"Pages wired down: 1000000.\n")
    monkeypatch.setattr(probe_worker.subprocess, "check_output", lambda _argv: output)
    assert probe_worker._wired_gb() == pytest.approx(0.9313225746154785)


def test_mlx_model_weight_and_kv_estimates_use_binary_gib(monkeypatch) -> None:
    monkeypatch.setattr(models, "_local_snapshot", lambda _model: "/model")
    monkeypatch.setattr(models, "_snapshot_weight_bytes", lambda _path: 1_000_000_000)
    assert models.weights_gb("org/model") == pytest.approx(0.9313225746154785)

    info = models.ModelInfo(
        hf_id="org/model", weights_gb=1.0, n_layers=1, growing_layers=1,
        kv_heads=1, head_dim=1, hidden_size=1, max_context=1,
        cache_type="standard", can_quantize_kv=True, layer_types={},
    )
    assert info.estimated_slope_gb_per_k() == pytest.approx(
        2 * 1 * 1 * 1 * 2 * 1000 / units.BYTES_PER_GIB * models.PREFILL_SPIKE_MULT)
