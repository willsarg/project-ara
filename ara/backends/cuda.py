# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""NVIDIA / CUDA backend adapter — drives wcx-suite's VRAM measurement out-of-process.

The CUDA twin of backends/apple.py: it reads the GPU's VRAM wall and runs wcx-suite's crash-safe
probe, but owns **no persistence** — ARA stores and reuses the calibration (see cli.render_profile).
It never imports wcx in-process: every engine call goes through the isolated ``cuda`` env via
:mod:`ara.engine_env`, so nothing torch-shaped loads in ARA's interpreter and the core stays
engine-free at runtime, not just at lock time.

Unlike Apple's hidden cold-start overhead, the VRAM wall is read exactly from nvidia-smi, so the
*budget* needs no calibration. What calibration measures here is the fixed CUDA-context VRAM cost
(cuBLAS/cuDNN), which the per-context safety gate adds on top of the model's weights.
"""
from __future__ import annotations

# Core, engine-free helpers (no wcx) — safe to import at module load and patchable in tests.
from ara import calibration, db, engine_env
from ara.contracts import driver

# The wcx worker modules ARA drives in the isolated cuda env (never imported in-process).
DEVICE_MODULE = "wcx_suite.device"
WORKER_MODULE = "wcx_suite.measure_one"

# Tiny model ARA calibrates/characterizes against — transformers format (torch can't load the
# mlx-community 4-bit build the Apple engine uses).
CALIBRATION_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"

# ARA-owned ramp policy (the engine only measures; ARA decides the schedule + safety margin).
RAMP_SCHEDULE = [2000, 4000, 8000, 16000, 32000, 65536, 131072]
DEFAULT_MARGIN_GB = 1.0      # VRAM cushion below the wall — tighter than Apple's 2 GB (smaller cards)
DEFAULT_OVERHEAD_GB = 0.6    # fallback CUDA-context overhead until calibrated


def safe_limits() -> dict:
    """Read this machine's safe VRAM limits via the wcx worker. Pure read — no model.

    Stateless: returns the budget with no stored overhead (``calibrated=False``). ARA overlays a
    previously-measured overhead from its own store — the engine no longer reads a database.
    """
    facts = engine_env.run_worker("cuda", ["-m", DEVICE_MODULE, "limits"])
    if "error" in facts:
        raise RuntimeError(facts["error"])
    return {
        **facts,
        "overhead_gb": None,        # ARA owns the stored calibration now
        "calibrated": False,
        "calibrated_at": None,
    }


def calibration_model_cached(model: str = CALIBRATION_MODEL) -> bool:
    """Is the calibration model already in the HF cache? (cheap, no load)."""
    from huggingface_hub import try_to_load_from_cache

    try:
        return isinstance(try_to_load_from_cache(model, "config.json"), str)
    except Exception:
        return False


def download_calibration_model(model: str = CALIBRATION_MODEL) -> None:
    """Fetch the calibration model into the HF cache. Network + disk only."""
    from ara import acquire

    acquire.download(model)


def calibrate(model: str = CALIBRATION_MODEL) -> dict:
    """Measure the CUDA-context VRAM overhead via the worker; return fresh limits + what it measured.

    The worker initialises a CUDA context (forcing cuBLAS/cuDNN in) and reads the nvidia-smi delta.
    ARA only invokes it (out-of-process in the cuda env). Surfaces the **effective** overhead
    (clamped to the engine's floor: ``max(default, measured)``) as ``overhead_gb`` so ARA can
    persist it; the raw measurement is in the ``"calibration"`` sub-dict for the caller to show.

    If the worker fails (error dict or exception), returns an uncalibrated result with a
    ``calibration_error`` field (never ``calibrated=True`` for unobserved data — Rule #3).
    The safe default overhead is still in effect via ``_budget_params``; callers can detect the
    condition via ``calibrated=False`` + presence of ``calibration_error``.
    """
    limits = safe_limits()
    try:
        result = engine_env.run_worker("cuda", ["-m", DEVICE_MODULE, "calibrate", model])
    except Exception as exc:
        limits["calibrated"] = False
        limits["overhead_gb"] = None
        limits["calibration_error"] = (
            f"calibration unavailable for {model!r}: {exc}"
        )
        return limits
    if result.get("error"):
        limits["calibrated"] = False
        limits["overhead_gb"] = None
        limits["calibration_error"] = (
            f"calibration unavailable for {model!r}: {result['error']}"
        )
        limits["calibration"] = result
        return limits
    overheads = [v for v in (result.get("measured_overhead_gb"),
                             result.get("default_overhead_gb")) if v is not None]
    limits["overhead_gb"] = max(overheads) if overheads else None
    limits["calibrated"] = True
    limits["calibration"] = result
    return limits


def _budget_params() -> tuple[float, float]:
    """ARA-owned (margin, overhead). Margin is policy; overhead is this machine's stored
    calibration for the wcx engine, or a safe default if uncalibrated."""
    overhead = DEFAULT_OVERHEAD_GB
    stored = calibration.get_calibration(db.connect(), "wcx")
    if stored and stored.get("fixed_overhead_gb") is not None:
        overhead = stored["fixed_overhead_gb"]
    return DEFAULT_MARGIN_GB, overhead


def _worker_argv(model: str, ctx: int, margin: float, overhead: float, *,
                 preflight: bool = False) -> list[str]:
    argv = ["-m", WORKER_MODULE, model, str(ctx),
            "--margin", str(margin), "--overhead", str(overhead)]
    if preflight:
        argv.append("--preflight")
    return argv


def characterize(model: str) -> dict:
    """Measure *model*'s safe VRAM context ceiling on this GPU — the thin path.

    Pure wiring: ARA owns the methodology in the engine-agnostic ``contracts.driver``; this adapter
    only supplies the CUDA specifics — the isolated ``cuda`` env, wcx's self-vetoing ``measure_one``
    worker, the budget params, and the schedule. ARA never imports wcx in-process. Crash-safety is
    layered: the driver gates each rung (L1 ``plan_next`` + L2 actual-footprint check), the engine
    refuses-before-load (L4) and a VRAM watchdog aborts mid-probe (L5). Returns
    ``{model, safe_context, points}``.
    """
    margin, overhead = _budget_params()
    return driver.characterize(
        model,
        preflight=lambda m: engine_env.run_worker(
            "cuda", _worker_argv(m, 0, margin, overhead, preflight=True)),
        measure=lambda m, ctx: engine_env.run_worker(
            "cuda", _worker_argv(m, ctx, margin, overhead)),
        schedule=RAMP_SCHEDULE,
    )


DEFAULT_MAX_TOKENS = 256


def generate(model, prompt, *, max_context, max_tokens=DEFAULT_MAX_TOKENS) -> dict:
    """One-shot CUDA completion, governed: max_context is the characterized safe ceiling, so the
    worker generates under the wall. Out-of-process in the isolated `cuda` env via wcx-suite's
    generate worker; the prompt goes over stdin, never argv. Returns {context, completion} or a
    refusal {refused, reason}. ARA never imports torch in-process."""
    margin, overhead = _budget_params()
    return engine_env.run_worker("cuda",
        ["-m", "wcx_suite.generate", model, str(max_context),
         "--margin", str(margin), "--overhead", str(overhead),
         "--max-tokens", str(max_tokens)],
        input=prompt)
