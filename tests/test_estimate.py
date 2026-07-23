# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""estimate.py — engine-free analytic memory limits + per-model fit.

Spec 2026-06-23-capability-pipeline (Slice 2, Task 2): profile reasons analytically — it
mirrors the engine wall from detect facts (no engine, no model load) and checks a model's
context-window limit against the estimated budget.
"""
from __future__ import annotations

import pytest

from ara import estimate, hardware
from ara.detect import Accelerator, Machine


def _machine(**over) -> Machine:
    base = dict(
        system="Darwin", os_version="macOS 15.0", chip="Apple M4 Pro", arch="arm64",
        cpu_physical=12, cpu_logical=12, cpu_features=[], python_version="3.12.8",
        ram_total_gb=48.0, ram_available_gb=20.0, swap_gb=2.0,
        accel=Accelerator("apple", "Apple M4 Pro GPU", None, "Metal", cores=16),
        disk_free_gb=500.0, physical_memory_bytes=48 * 1024 ** 3,
        backend="apple", engine="mlx",
    )
    base.update(over)
    return Machine(**base)


# --- limits: mirror the wall, engine-free --------------------------------- #
def test_limits_apple_without_live_authority_has_no_current_budget():
    # Engine-free Apple recon knows physical RAM exactly, but it cannot read Metal's current
    # recommendation without crossing the isolated engine seam. Never invent a 75% budget.
    physical_bytes = 25_769_803_776
    lim = estimate.limits(_machine(
        backend="apple", ram_total_gb=24.0, physical_memory_bytes=physical_bytes,
        chip="Apple M4 Pro"))
    assert lim["physical_memory_bytes"] == physical_bytes
    assert lim["total_gb"] == 24.0
    assert lim["wall_gb"] is None
    assert lim["safe_budget_gb"] is None
    assert lim["basis"] == "unknown"
    assert lim["calibrated"] is False
    assert lim["device"] == "Apple M4 Pro"        # non-CUDA branch: device is the chip, not None
    # These are the exact key names of the output contract — a renamed/misrouted key breaks
    # every caller silently unless pinned here.
    assert lim["headroom_gb"] is None             # a live quantity; estimate never fills it
    assert lim["calibrated_at"] is None            # uncalibrated: no measurement timestamp


def test_limits_cpu_uses_full_ram():
    lim = estimate.limits(_machine(backend="cpu", ram_total_gb=32.0, chip="Intel i9-13900K"))
    assert lim["wall_gb"] == 32.0
    assert lim["safe_budget_gb"] == 32.0 - estimate.MARGIN_GB
    assert lim["device"] == "Intel i9-13900K"     # non-CUDA branch: device is the chip, not None


def test_limits_cuda_single_gpu_wall_is_vram():
    m = _machine(backend="cuda",
                 accel=Accelerator("nvidia", "RTX 4090", 24.0, "CUDA", count=1))
    lim = estimate.limits(m)
    assert lim["total_gb"] == 24.0
    assert lim["wall_gb"] == 24.0
    assert lim["device"] == "RTX 4090"


def test_limits_cuda_falsy_count_defaults_to_one_device():
    # `count` can arrive falsy (None/0) from detect facts; the wall must still be a SINGLE
    # device's VRAM (24.0), not multiplied — a falsy count is not "0 devices" or "2 devices".
    m = _machine(backend="cuda",
                 accel=Accelerator("nvidia", "RTX 4090", 24.0, "CUDA", count=None))
    lim = estimate.limits(m)
    assert lim["total_gb"] == 24.0
    assert lim["wall_gb"] == 24.0


# --- CUDA multi-GPU: the analytic wall must match what the engine actually governs --------- #
# The shipped CUDA engine measures ONE device (NVML index 0) and does NOT shard a model across
# GPUs. So on a multi-GPU box the analytic wall must be a *single* card's VRAM — otherwise
# `profile` (basis="estimated") promises a budget `characterize` (basis="measured") will refuse,
# violating Rule #3. `total_gb` still reports the true physical total across all cards; a future
# sharding engine (tensor-parallel) opts back into the summed wall via `sharded=True`.
def test_limits_cuda_multi_gpu_wall_is_single_device_by_default():
    m = _machine(backend="cuda",
                 accel=Accelerator("nvidia", "RTX 4090", 24.0, "CUDA", count=2))
    lim = estimate.limits(m)
    assert lim["total_gb"] == 48.0           # physical truth: 24 GB × 2 cards
    assert lim["wall_gb"] == 24.0            # governable: one device (CUDA measures index 0)
    assert lim["safe_budget_gb"] == 24.0 - estimate.MARGIN_GB


def test_limits_cuda_sharded_sums_across_gpus():
    # A sharding engine spreads one model across all GPUs, so the wall IS the physical sum.
    m = _machine(backend="cuda",
                 accel=Accelerator("nvidia", "RTX 4090", 24.0, "CUDA", count=2))
    lim = estimate.limits(m, sharded=True)
    assert lim["total_gb"] == 48.0
    assert lim["wall_gb"] == 48.0
    assert lim["safe_budget_gb"] == 48.0 - estimate.MARGIN_GB


def test_limits_cuda_single_gpu_sharded_is_noop():
    # One GPU: single-device == sum, so `sharded` changes nothing (regression guard).
    m = _machine(backend="cuda",
                 accel=Accelerator("nvidia", "RTX 4090", 24.0, "CUDA", count=1))
    assert estimate.limits(m)["wall_gb"] == 24.0
    assert estimate.limits(m, sharded=True)["wall_gb"] == 24.0


def test_limits_missing_ram_is_unknown():
    lim = estimate.limits(_machine(backend="cpu", ram_total_gb=None))
    assert lim["wall_gb"] is None
    assert lim["safe_budget_gb"] is None


# --- the Rule #1 gate is cgroup-honest: the wall source is clamped for containers --------- #
def _clamped_ram_total_gb(monkeypatch, *, system, phys_gb, cgroup_gb=None):
    """Drive the core RAM-total source (hardware._psutil_totals) under a mocked host + cgroup, and
    return the total the gate would see. This is the exact value detect feeds Machine.ram_total_gb."""
    import psutil
    monkeypatch.setattr(hardware.platform, "system", lambda: system)
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: type("vm", (), {"total": int(phys_gb * hardware.GB),
                                                "available": int(phys_gb * hardware.GB)})())
    monkeypatch.setattr(psutil, "swap_memory", lambda: type("sw", (), {"total": 0})())
    files = {} if cgroup_gb is None else {hardware._CGROUP_V2: str(int(cgroup_gb * hardware.GB))}
    monkeypatch.setattr(hardware, "_read_cgroup_file", lambda path: files.get(path))
    _physical_bytes, total_gb, _avail, _swap = hardware._psutil_totals()
    return total_gb


def test_gate_wall_reflects_cgroup_limit_below_physical(monkeypatch):
    # Container capped at 8 GiB on a 32 GiB host: the CPU gate must size against 8, not 32 (else OOM).
    total = _clamped_ram_total_gb(monkeypatch, system="Linux", phys_gb=32.0, cgroup_gb=8.0)
    lim = estimate.limits(_machine(backend="cpu", ram_total_gb=total))
    assert lim["total_gb"] == 8.0
    assert lim["wall_gb"] == 8.0
    assert lim["safe_budget_gb"] == 8.0 - estimate.MARGIN_GB


def test_gate_wall_is_physical_without_cgroup_limit(monkeypatch):
    total = _clamped_ram_total_gb(monkeypatch, system="Linux", phys_gb=32.0, cgroup_gb=None)
    lim = estimate.limits(_machine(backend="cpu", ram_total_gb=total))
    assert lim["wall_gb"] == 32.0                                   # no limit binds → physical


def test_gate_wall_is_physical_off_linux(monkeypatch):
    total = _clamped_ram_total_gb(monkeypatch, system="Darwin", phys_gb=32.0, cgroup_gb=8.0)
    lim = estimate.limits(_machine(backend="cpu", ram_total_gb=total))
    assert lim["wall_gb"] == 32.0                                   # no cgroup off Linux → physical


# --- limits: prefer a measured wall when one is supplied ------------------ #
def test_limits_uses_measured_wall_when_supplied():
    # Once ARA has a measured wall for this machine + engine, profile reports the measurement
    # (clearly labelled), not the heuristic. Spec 2026-06-23-capability-pipeline.
    measured = {"wall_gb": 41.3, "safe_budget_gb": 39.3, "calibrated_at": "2026-07-01T00:00:00Z"}
    lim = estimate.limits(_machine(backend="apple", ram_total_gb=48.0), measured=measured)
    assert lim["wall_gb"] == 41.3
    assert lim["safe_budget_gb"] == 39.3
    assert lim["basis"] == "measured"
    assert lim["calibrated"] is True
    # The device/total still come from detect facts; only the wall/budget are the measurement.
    assert lim["total_gb"] == 48.0
    # The calibration timestamp must propagate from `measured` into the result verbatim — not be
    # dropped, nulled, or read from the wrong source key.
    assert lim["calibrated_at"] == "2026-07-01T00:00:00Z"


def test_limits_apple_measurement_does_not_retain_removed_heuristic():
    # An explicitly supplied live measurement can govern, but there is no 75%-of-RAM comparison.
    measured = {"wall_gb": 41.3, "safe_budget_gb": 39.3}
    lim = estimate.limits(_machine(backend="apple", ram_total_gb=48.0), measured=measured)
    assert "estimated_wall_gb" not in lim
    assert "estimated_safe_budget_gb" not in lim


def test_limits_apple_stays_unknown_when_measurement_lacks_wall():
    # A partial/historical row cannot become a current budget at the engine-free seam.
    lim = estimate.limits(_machine(backend="apple", ram_total_gb=48.0),
                          measured={"wall_gb": None, "safe_budget_gb": None})
    assert lim["wall_gb"] is None
    assert lim["safe_budget_gb"] is None
    assert lim["basis"] == "unknown"
    assert lim["calibrated"] is False


def test_limits_apple_no_measured_is_unknown():
    lim = estimate.limits(_machine(backend="apple", ram_total_gb=48.0), measured=None)
    assert lim["basis"] == "unknown"
    assert lim["calibrated"] is False
    assert "estimated_wall_gb" not in lim


# --- model_fit: check the model's context limit --------------------------- #
_META = dict(n_layers=32, kv_heads=8, head_dim=128, max_context=8192)


def test_model_fit_full_window_when_budget_covers_it():
    lim = {"safe_budget_gb": 60.0}
    fit = estimate.model_fit(lim, _META, weights_gb=4.0)
    assert fit["fits"] is True
    assert fit["binding"] == "context_window"
    assert fit["est_context"] == 8192        # capped at the model's window
    assert fit["max_context"] == 8192


def test_model_fit_context_limited_by_memory():
    lim = {"safe_budget_gb": 4.5}             # tight: budget binds before the window
    fit = estimate.model_fit(lim, _META, weights_gb=4.0)
    assert fit["fits"] is True
    assert fit["binding"] == "memory"
    assert 0 < fit["est_context"] < 8192


def test_model_fit_weights_exceed_budget():
    lim = {"safe_budget_gb": 5.0}
    fit = estimate.model_fit(lim, _META, weights_gb=8.0)
    assert fit["fits"] is False
    assert fit["est_context"] is None


def test_model_fit_without_current_budget_is_unknown():
    fit = estimate.model_fit({"safe_budget_gb": None}, _META, weights_gb=4.0)
    assert fit["fits"] is None
    assert fit["est_context"] is None
    assert fit["binding"] is None
    assert fit["reason"] == "no_current_budget"


def test_model_fit_weights_equal_budget_does_not_fit():
    # Strict `<`: weights leave zero headroom for KV cache/activations, so an exact match does
    # NOT fit. (A `<=` mutant would say True here.) The budget is set to the fit's own reported
    # (GiB-converted) weights so the equality is exact regardless of float rounding.
    probe = estimate.model_fit({"safe_budget_gb": 60.0}, _META, weights_gb=5.0)
    lim = {"safe_budget_gb": probe["weights_gb"]}
    fit = estimate.model_fit(lim, _META, weights_gb=5.0)
    assert fit["fits"] is False


def test_model_fit_converts_decimal_weights_to_gib():
    """Weights arrive as decimal GB (on-disk size / 1e9) but budgets are binary GiB — model_fit
    must convert before comparing, or a model that actually fits is refused (and the reported
    footprint is ~7.4% overstated against the wall).

    Slug: 2026-07-02-analytic-units-gib
    """
    # 7.8 decimal GB = 7.264… GiB: fits a 7.5 GiB budget only after conversion.
    fit = estimate.model_fit({"safe_budget_gb": 7.5}, _META, weights_gb=7.8)
    assert fit["fits"] is True
    assert fit["weights_gb"] == pytest.approx(7.8 * 1e9 / estimate.GIB)


def test_model_fit_est_context_uses_gib_weights():
    """The decode-ceiling intercept must be the GiB-converted weights, not the decimal figure.

    Slug: 2026-07-02-analytic-units-gib
    """
    from ara.contracts import ramp

    lim = {"safe_budget_gb": 4.5}
    fit = estimate.model_fit(lim, _META, weights_gb=4.0)
    slope = ramp.analytic_kv_slope_gb_per_k(_META["n_layers"], _META["kv_heads"],
                                            _META["head_dim"])
    expected, _ = ramp.decode_ceiling(4.0 * 1e9 / estimate.GIB, slope, 4.5,
                                      max_context=_META["max_context"])
    assert fit["est_context"] == expected
    # And it must NOT equal the unconverted solve (guards against reverting the conversion).
    wrong, _ = ramp.decode_ceiling(4.0, slope, 4.5, max_context=_META["max_context"])
    assert fit["est_context"] != wrong


def test_model_fit_unknown_dims_gives_no_context_estimate():
    lim = {"safe_budget_gb": 60.0}
    meta = dict(n_layers=None, kv_heads=None, head_dim=None, max_context=8192)
    fit = estimate.model_fit(lim, meta, weights_gb=4.0)
    assert fit["fits"] is True
    assert fit["est_context"] is None         # can't estimate the slope → honest unknown
    assert fit["binding"] is None
