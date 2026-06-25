# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Vulkan backend — GPU-offload GGUF inference via llama.cpp's Vulkan backend.

ARA's first **GPU-offload-on-shared-memory** engine: runs GGUF models on an AMD APU's
integrated Radeon (RDNA3 780M / gfx1103 today) by offloading every layer (``n_gpu_layers=-1``)
through llama.cpp's Vulkan backend. **Opt-in** via ``--engine vulkan`` — CPU stays the safe
auto-default, because on an APU the iGPU **shares the same memory wall** as the CPU (GPU memory
is carved from system RAM), so it's a prefill/throughput win, not a capacity win.

Contract class: **ramp** (safe context ceiling). Wall source: **system RAM** — read exactly,
like the CPU engine, *not* Apple's hidden cold-start overhead: the GPU pool (GTT) is carved from
the same physical RAM, so the one wall is physical RAM and there's nothing extra to calibrate for
the budget itself. What differs from CPU is purely inside the worker (offload + GTT-sysfs
measurement + honest offload verification) — see ``ara/workers/vulkan_llama.py``.

Built into ARA (only the huge CUDA/MLX suites get their own repos). The worker is a self-contained
script — ``ara/workers/vulkan_llama.py``, which never imports ``ara`` — run by the isolated
``vulkan`` env's own python over ``engine_env``, so llama-cpp-python never enters ARA core's lock.
``characterize`` is pure wiring into the engine-agnostic :func:`ara.contracts.driver.characterize`.
"""
from __future__ import annotations

from pathlib import Path

# Core, engine-free helpers (no llama.cpp) — safe at module load and patchable in tests.
from ara import calibration, db, engine_env
from ara.contracts import driver

# The built-in worker script (ships in the ARA repo, runs in the isolated vulkan env by path).
WORKER = Path(__file__).resolve().parent.parent / "workers" / "vulkan_llama.py"

ENV_NAME = "vulkan"
CALIBRATION_ENGINE = "vulkan"   # profiles key for this machine's stored overhead
RAMP_SCHEDULE = [2000, 4000, 8000, 16000, 32000, 65536, 131072]
DEFAULT_MARGIN_GB = 2.0      # margin CAP (GB); the worker scales it to ~10% of RAM, floor 0.5GB
DEFAULT_OVERHEAD_GB = 1.0    # fallback cold-start overhead until calibrated

# A small GGUF ARA characterizes against in the profile flow (the worker auto-picks its
# smallest quant and downloads it on demand) — the same calibration model as the CPU engine.
CALIBRATION_MODEL = "bartowski/SmolLM2-135M-Instruct-GGUF"


def safe_limits() -> dict:
    """This machine's safe RAM limits via the vulkan worker. Pure read — no model load.

    Like the CPU engine (and unlike Apple's hidden MLX cold-start overhead), the wall is read
    exactly — physical RAM, which the GPU's GTT pool is carved from — so there's nothing to
    calibrate for the budget itself (``calibrated`` True, ``overhead_gb`` None). Characterizing a
    *model*'s context ceiling on the GPU is a separate, optional step.
    """
    facts = engine_env.run_worker(
        ENV_NAME, [str(WORKER), "--limits", "--margin", str(DEFAULT_MARGIN_GB)])
    return {
        **facts,
        "overhead_gb": None,         # the wall is exact — nothing to calibrate
        "calibrated": True,
        "calibrated_at": None,
    }


def calibration_model_cached(model: str = CALIBRATION_MODEL) -> bool:
    """Always True: the worker downloads the smallest GGUF on demand during characterization,
    so the profile flow never blocks on a separate fetch step."""
    return True


def download_calibration_model(model: str = CALIBRATION_MODEL, *,
                               progress: bool = False) -> None:
    """No-op: GGUF acquisition is lazy inside the worker (it pulls only the smallest quant).
    ``progress`` is accepted for interface symmetry but ignored."""


def calibrate(model: str = CALIBRATION_MODEL) -> dict:
    """Characterize *model* on the GPU and attach it to the limits (for the profile flow).

    If characterization fails (model unavailable, no Vulkan offload, worker error), returns an
    uncalibrated result with a ``calibration_error`` field (never ``calibrated=True`` for
    unobserved data — Rule #3). Callers detect via ``calibrated=False`` + ``calibration_error``.
    """
    out = safe_limits()
    char = characterize(model)
    if char.get("error"):
        out["calibrated"] = False
        out["calibration_error"] = (
            f"calibration unavailable for {model!r}: {char['error']}"
        )
    else:
        out["characterization"] = char
    return out


def _budget_params() -> tuple[float, float]:
    """ARA-owned (margin, overhead). Margin is policy; overhead is this machine's stored
    calibration for the vulkan engine, or a safe default if uncalibrated."""
    overhead = DEFAULT_OVERHEAD_GB
    stored = calibration.get_calibration(db.connect(), CALIBRATION_ENGINE)
    if stored and stored.get("fixed_overhead_gb") is not None:
        overhead = stored["fixed_overhead_gb"]
    return DEFAULT_MARGIN_GB, overhead


def _worker_argv(model: str, ctx: int, margin: float, overhead: float, *,
                 preflight: bool = False) -> list[str]:
    argv = [str(WORKER), model, str(ctx),
            "--margin", str(margin), "--overhead", str(overhead)]
    if preflight:
        argv.append("--preflight")
    return argv


def characterize(model: str, *, progress: bool = False) -> dict:
    """Measure *model*'s safe context ceiling on the GPU — the thin path, same driver as CPU/Apple.

    Pure wiring: ARA owns the methodology in the engine-agnostic ``contracts.driver``; this
    adapter supplies only the vulkan specifics — the isolated ``vulkan`` env, the built-in
    ``vulkan_llama`` worker, budget params, and the schedule. Crash-safety is layered: the driver
    gates each rung (L1 + L2), and the worker refuses-before-load (L4) / aborts mid-probe (L5).
    Returns ``{model, safe_context, points}``.

    ``progress=True`` streams the worker's stderr live so HF's native tqdm bars are visible
    during the GGUF fetch that the vulkan worker handles in-process.
    """
    margin, overhead = _budget_params()
    return driver.characterize(
        model,
        preflight=lambda m: engine_env.run_worker(
            ENV_NAME, _worker_argv(m, 0, margin, overhead, preflight=True),
            stream=progress),
        measure=lambda m, ctx: engine_env.run_worker(
            ENV_NAME, _worker_argv(m, ctx, margin, overhead),
            stream=progress),
        schedule=RAMP_SCHEDULE,
    )


DEFAULT_MAX_TOKENS = 256


def generate(model: str, prompt: str, *, max_context: int,
             max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    """One-shot completion on the GPU, governed: ``max_context`` is the characterized safe ceiling,
    so the worker's KV cache is capped under the wall (the worker still self-vetoes, L4/L5).
    Out-of-process in the isolated ``vulkan`` env; the prompt goes over stdin, never argv. Returns
    ``{context, completion}`` or a refusal (``{refused, reason}``)."""
    margin, overhead = _budget_params()
    return engine_env.run_worker(
        ENV_NAME,
        [str(WORKER), model, str(max_context), "--generate",
         "--margin", str(margin), "--overhead", str(overhead),
         "--max-tokens", str(max_tokens)],
        input=prompt)
