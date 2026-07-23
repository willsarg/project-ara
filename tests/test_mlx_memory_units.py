# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The independently installable MLX worker uses the same binary memory contract."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from ara._engine_packages.mlx.ara_engine_mlx import (
    device, measure_one, models, probe, probe_worker, system, units,
)


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


def test_safe_budget_preserves_exact_binary_margin(monkeypatch) -> None:
    limits = system.SystemLimits(
        device="Apple M4 Pro",
        memory_size_bytes=25_769_803_776,
        recommended_working_set_bytes=19_069_665_280,
        max_buffer_length_bytes=9_000_000_000,
        total_gb=24.0,
        wall_gb=17.760009765625,
        max_buffer_gb=8.381903171539307,
        swap_free_gb=2.0,
        wired_now_gb=4.0,
    )
    monkeypatch.setattr(device.system, "read_limits", lambda: limits)
    monkeypatch.setattr(device.config, "margin_gb", lambda value: 2.0)

    result = device.limits()

    assert result["safe_budget_bytes"] == 16_922_181_632
    assert result["safe_budget_gb"] == 15.760009765625


def test_mach_wired_memory_is_binary_gib(monkeypatch) -> None:
    monkeypatch.setattr(system, "_mach_wired_pages", lambda: 1_000_000)
    monkeypatch.setattr(system, "_mach_page_size", lambda: 1_000)
    assert system._native_wired_gb() == pytest.approx(0.9313225746154785)


def test_native_vm_snapshot_preserves_exact_kernel_counters(monkeypatch) -> None:
    stats = system._vm_statistics64()
    stats.wire_count = 100
    stats.compressor_page_count = 20
    stats.compressions = 30
    stats.decompressions = 40
    stats.swapins = 50
    stats.swapouts = 60
    stats.throttled_count = 7
    swap = system._xsw_usage()
    swap.xsu_total = 10_000
    swap.xsu_used = 6_000
    swap.xsu_avail = 4_000
    monkeypatch.setattr(system, "_mach_vm_statistics", lambda: stats)
    monkeypatch.setattr(system, "_mach_page_size", lambda: 4096)
    monkeypatch.setattr(system, "_native_swap_usage", lambda: swap)

    snapshot = system.native_vm_snapshot()

    assert snapshot == system.VMSnapshot(
        wired_bytes=409_600, compressor_bytes=81_920,
        compressions=30, decompressions=40, swapins=50, swapouts=60,
        throttled_bytes=28_672, swap_total_bytes=10_000,
        swap_used_bytes=6_000, swap_available_bytes=4_000)


def _snapshot(wired: int, *, compressions=0, swapouts=0) -> system.VMSnapshot:
    return system.VMSnapshot(
        wired_bytes=wired, compressor_bytes=20, compressions=compressions,
        decompressions=0, swapins=0, swapouts=swapouts, throttled_bytes=0,
        swap_total_bytes=1000, swap_used_bytes=100, swap_available_bytes=900)


def test_external_supervisor_samples_host_and_qualifies_peak() -> None:
    values = iter([_snapshot(100), _snapshot(120), _snapshot(150, compressions=2)])
    last = [_snapshot(150, compressions=2)]

    def snapshot():
        try:
            last[0] = next(values)
        except StopIteration:
            pass
        return last[0]

    child = (
        "import json,time; time.sleep(.03); "
        "print(json.dumps({'context': 512, 'status': 'ok', "
        "'mlx_peak_bytes': 10, 'mlx_active_plus_cache_bytes': 8}))")
    result = probe._supervise_process(
        [sys.executable, "-c", child], context=512, abort_wired_bytes=1000,
        timeout=2, sampling_interval=0.005, snapshot_fn=snapshot)

    assert result["status"] == "ok"
    assert result["baseline_wired_bytes"] == 100
    assert result["highest_sampled_host_wired_bytes"] == 150
    assert result["telemetry"]["sample_count"] >= 1
    assert result["telemetry"]["scope"] == "host-wide"
    assert result["telemetry"]["peak_qualification"] == "highest sampled; gaps unobserved"
    assert result["telemetry"]["compression_delta"]["compressions"] == 2


def test_terminate_process_group_falls_back_without_killpg(monkeypatch) -> None:
    calls = []

    class Process:
        pid = 123

        @staticmethod
        def poll():
            return None

        @staticmethod
        def terminate():
            calls.append("terminate")

        @staticmethod
        def kill():
            calls.append("kill")

        @staticmethod
        def wait(timeout=None):
            calls.append(("wait", timeout))
            if timeout is not None:
                raise probe.subprocess.TimeoutExpired("worker", timeout)

    monkeypatch.delattr(probe.os, "killpg", raising=False)

    probe._terminate_process_group(Process())

    assert calls == ["terminate", ("wait", 2), "kill", ("wait", None)]


def test_external_supervisor_kills_process_group_at_boundary(monkeypatch) -> None:
    snapshots = iter([_snapshot(100), _snapshot(120)])
    killed = []
    real_terminate = probe._terminate_process_group

    def terminate(proc):
        killed.append(proc.pid)
        real_terminate(proc)

    monkeypatch.setattr(probe, "_terminate_process_group", terminate)
    result = probe._supervise_process(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        context=512, abort_wired_bytes=110, timeout=2, sampling_interval=0.001,
        snapshot_fn=lambda: next(snapshots))

    assert result["status"] == "aborted"
    assert "boundary" in result["note"]
    assert killed and result["telemetry"]["sample_count"] == 1


def test_external_supervisor_refuses_boundary_before_launch(monkeypatch) -> None:
    launched = []
    monkeypatch.setattr(
        probe.subprocess, "Popen", lambda *_a, **_k: launched.append(True))

    result = probe._supervise_process(
        ["unused"], context=512, abort_wired_bytes=100, timeout=2,
        snapshot_fn=lambda: _snapshot(100))

    assert result["status"] == "aborted"
    assert "already reached" in result["note"]
    assert result["telemetry"]["sample_count"] == 0
    assert not launched


def test_external_supervisor_timeout_reaps_process_group(monkeypatch) -> None:
    killed = []
    real_terminate = probe._terminate_process_group

    def terminate(proc):
        killed.append(proc.pid)
        real_terminate(proc)

    monkeypatch.setattr(probe, "_terminate_process_group", terminate)
    result = probe._supervise_process(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        context=512, abort_wired_bytes=1000, timeout=0.01,
        sampling_interval=0.001, snapshot_fn=lambda: _snapshot(100))

    assert result["status"] == "aborted"
    assert "timed out" in result["note"]
    assert killed


def test_external_supervisor_fails_closed_when_launch_fails(monkeypatch) -> None:
    def fail_launch(*_args, **_kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr(probe.subprocess, "Popen", fail_launch)
    result = probe._supervise_process(
        ["missing"], context=512, abort_wired_bytes=1000, timeout=2,
        snapshot_fn=lambda: _snapshot(100))

    assert result["status"] == "aborted"
    assert "launch failed closed" in result["note"]
    assert result["telemetry"]["sample_count"] == 0


def test_external_supervisor_fails_closed_on_telemetry_loss(monkeypatch) -> None:
    calls = [0]

    def snapshot():
        calls[0] += 1
        if calls[0] > 1:
            raise OSError("mach read failed")
        return _snapshot(100)

    result = probe._supervise_process(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        context=512, abort_wired_bytes=1000, timeout=2,
        sampling_interval=0.001, snapshot_fn=snapshot)

    assert result["status"] == "aborted"
    assert "telemetry failed closed" in result["note"]
    assert result["telemetry"]["sample_count"] == 0


def test_external_supervisor_file_capture_cannot_pipe_deadlock() -> None:
    child = (
        "import json,sys; "
        "sys.stdout.write('x'*1000000); sys.stderr.write('y'*1000000); "
        "print('\\n'+json.dumps({'context':512,'status':'ok',"
        "'mlx_peak_bytes':10,'mlx_active_plus_cache_bytes':8}))")
    result = probe._supervise_process(
        [sys.executable, "-c", child], context=512, abort_wired_bytes=1000,
        timeout=3, sampling_interval=0.001, snapshot_fn=lambda: _snapshot(100))

    assert result["status"] == "ok"


def test_measure_one_preserves_each_repeat_telemetry(monkeypatch) -> None:
    info = models.ModelInfo(
        hf_id="org/model", weights_gb=1.0, n_layers=1, growing_layers=1,
        kv_heads=1, head_dim=1, hidden_size=1, max_context=8192,
        cache_type="standard", can_quantize_kv=True, layer_types={})
    limits = system.SystemLimits(
        device="Apple", memory_size_bytes=32 * units.BYTES_PER_GIB,
        recommended_working_set_bytes=20 * units.BYTES_PER_GIB,
        max_buffer_length_bytes=10 * units.BYTES_PER_GIB,
        total_gb=32, wall_gb=20, max_buffer_gb=10,
        swap_free_gb=2, wired_now_gb=1)
    monkeypatch.setattr(measure_one.models, "describe", lambda _model: info)
    monkeypatch.setattr(measure_one.system, "read_limits", lambda: limits)
    monkeypatch.setattr(measure_one.system, "sample_settled_baseline", lambda: 1.0)
    monkeypatch.setattr(measure_one, "safety_gate", lambda *_a, **_k: None)
    raws = iter([
        {"status": "ok", "baseline_wired_bytes": 1 * units.BYTES_PER_GIB,
         "highest_sampled_host_wired_bytes": 3 * units.BYTES_PER_GIB,
         "mlx_peak_bytes": 100, "mlx_active_plus_cache_bytes": 80,
         "telemetry": {"sample_count": 2}},
        {"status": "ok", "baseline_wired_bytes": 1 * units.BYTES_PER_GIB,
         "highest_sampled_host_wired_bytes": 4 * units.BYTES_PER_GIB,
         "mlx_peak_bytes": 110, "mlx_active_plus_cache_bytes": 90,
         "telemetry": {"sample_count": 3}},
    ])
    monkeypatch.setattr(measure_one, "_spawn_worker", lambda *_a, **_k: next(raws))

    result = measure_one.run(
        "org/model", 4096, margin_gb=2, overhead_gb=1, repeats=2)

    assert result["mem_gb"] == 2.5
    assert [repeat["sample_count"] for repeat in result["telemetry"]["repeats"]] == [2, 3]
    assert result["telemetry"]["mlx_allocator_observations"][1] == {
        "mlx_peak_bytes": 110, "mlx_active_plus_cache_bytes": 90}


def test_measure_one_preserves_prior_telemetry_when_later_repeat_fails(
        monkeypatch) -> None:
    info = models.ModelInfo(
        hf_id="org/model", weights_gb=1.0, n_layers=1, growing_layers=1,
        kv_heads=1, head_dim=1, hidden_size=1, max_context=8192,
        cache_type="standard", can_quantize_kv=True, layer_types={})
    limits = system.SystemLimits(
        device="Apple", memory_size_bytes=32 * units.BYTES_PER_GIB,
        recommended_working_set_bytes=20 * units.BYTES_PER_GIB,
        max_buffer_length_bytes=10 * units.BYTES_PER_GIB,
        total_gb=32, wall_gb=20, max_buffer_gb=10,
        swap_free_gb=2, wired_now_gb=1)
    monkeypatch.setattr(measure_one.models, "describe", lambda _model: info)
    monkeypatch.setattr(measure_one.system, "read_limits", lambda: limits)
    monkeypatch.setattr(measure_one.system, "sample_settled_baseline", lambda: 1.0)
    monkeypatch.setattr(measure_one, "safety_gate", lambda *_a, **_k: None)
    raws = iter([
        {"status": "ok", "baseline_wired_bytes": units.BYTES_PER_GIB,
         "highest_sampled_host_wired_bytes": 2 * units.BYTES_PER_GIB,
         "mlx_peak_bytes": 100, "mlx_active_plus_cache_bytes": 80,
         "telemetry": {"sample_count": 2}},
        {"status": "aborted", "note": "boundary",
         "telemetry": {"sample_count": 3}},
    ])
    monkeypatch.setattr(measure_one, "_spawn_worker", lambda *_a, **_k: next(raws))

    result = measure_one.run(
        "org/model", 4096, margin_gb=2, overhead_gb=1, repeats=2)

    assert result["refused"] is True
    assert result["telemetry"]["repeats"] == [
        {"sample_count": 2}, {"sample_count": 3}]


def test_probe_worker_reports_only_exact_mlx_allocator_counters() -> None:
    mx = SimpleNamespace(
        get_peak_memory=lambda: 1234,
        get_active_memory=lambda: 500,
        get_cache_memory=lambda: 250,
    )

    assert probe_worker._allocator_counters(mx) == {
        "mlx_peak_bytes": 1234,
        "mlx_active_plus_cache_bytes": 750,
    }
    assert not hasattr(probe_worker, "_wired_gb")
    assert not hasattr(probe_worker, "_should_abort")


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
