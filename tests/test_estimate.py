# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""estimate.py — engine-free analytic memory limits + per-model fit.

Spec 2026-06-23-capability-pipeline (Slice 2, Task 2): profile reasons analytically — it
mirrors the engine wall from detect facts (no engine, no model load) and checks a model's
context-window limit against the estimated budget.
"""
from __future__ import annotations

from ara import estimate
from ara.detect import Accelerator, Machine


def _machine(**over) -> Machine:
    base = dict(
        system="Darwin", os_version="macOS 15.0", chip="Apple M4 Pro", arch="arm64",
        cpu_physical=12, cpu_logical=12, cpu_features=[], python_version="3.12.8",
        ram_total_gb=48.0, ram_available_gb=20.0, swap_gb=2.0,
        accel=Accelerator("apple", "Apple M4 Pro GPU", None, "Metal", cores=16),
        disk_free_gb=500.0, backend="apple", engine="wmx-suite",
    )
    base.update(over)
    return Machine(**base)


# --- limits: mirror the wall, engine-free --------------------------------- #
def test_limits_apple_mirrors_working_set():
    # Apple unified memory: the safe working set is a fraction of total RAM.
    lim = estimate.limits(_machine(backend="apple", ram_total_gb=48.0))
    assert lim["total_gb"] == 48.0
    assert lim["wall_gb"] == 48.0 * estimate.APPLE_WORKING_SET
    assert lim["safe_budget_gb"] == lim["wall_gb"] - estimate.MARGIN_GB
    assert lim["basis"] == "estimated"
    assert lim["calibrated"] is False


def test_limits_cpu_uses_full_ram():
    lim = estimate.limits(_machine(backend="cpu", ram_total_gb=32.0))
    assert lim["wall_gb"] == 32.0
    assert lim["safe_budget_gb"] == 32.0 - estimate.MARGIN_GB


def test_limits_cuda_uses_total_vram():
    m = _machine(backend="cuda",
                 accel=Accelerator("nvidia", "RTX 4090", 24.0, "CUDA", count=2))
    lim = estimate.limits(m)
    assert lim["total_gb"] == 48.0           # 24 GB × 2 GPUs
    assert lim["wall_gb"] == 48.0
    assert lim["device"] == "RTX 4090"


def test_limits_missing_ram_is_unknown():
    lim = estimate.limits(_machine(backend="cpu", ram_total_gb=None))
    assert lim["wall_gb"] is None
    assert lim["safe_budget_gb"] is None


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
    lim = {"safe_budget_gb": 5.0}             # tight: budget binds before the window
    fit = estimate.model_fit(lim, _META, weights_gb=4.0)
    assert fit["fits"] is True
    assert fit["binding"] == "memory"
    assert 0 < fit["est_context"] < 8192


def test_model_fit_weights_exceed_budget():
    lim = {"safe_budget_gb": 5.0}
    fit = estimate.model_fit(lim, _META, weights_gb=8.0)
    assert fit["fits"] is False
    assert fit["est_context"] is None


def test_model_fit_unknown_dims_gives_no_context_estimate():
    lim = {"safe_budget_gb": 60.0}
    meta = dict(n_layers=None, kv_heads=None, head_dim=None, max_context=8192)
    fit = estimate.model_fit(lim, meta, weights_gb=4.0)
    assert fit["fits"] is True
    assert fit["est_context"] is None         # can't estimate the slope → honest unknown
    assert fit["binding"] is None
