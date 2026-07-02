# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""CPU backend — system-RAM inference via llama.cpp (GGUF) — the second real engine.

Contract class: **ramp** (safe context ceiling). Wall source: **physical RAM** (swap is
reported but never counted toward the budget — swapping during inference is catastrophically
slow, so it's not usable headroom). One adapter for *all* CPU-only inference regardless of ISA
— x86 (Intel/AMD), arm64, Raspberry Pi, riscv-when-it-matters. The ISA is metadata reported by
``detect``, not a separate backend, because it doesn't change the wall. This is the universal
fallback: nearly every machine matches it.

It exists to keep the abstraction honest. Unlike Apple (whose engine is the external
``wmx-suite``), llama.cpp is **built into ARA** (see the engine repo-split rule: only the huge
CUDA/MLX suites get their own repos). The worker is a self-contained script —
``ara/workers/cpu_llama.py``, which never imports ``ara`` — run by the isolated ``cpu`` env's
own python over ``engine_env``, so llama-cpp-python never enters ARA core's lock. ``characterize``
is pure wiring into the engine-agnostic :func:`ara.contracts.driver.characterize`; the only
Apple↔CPU differences are the env name, the worker, and the wall source.
"""
from __future__ import annotations

import json
from pathlib import Path

# Core, engine-free helpers (no llama.cpp) — safe at module load and patchable in tests.
from ara import calibration, db, engine_env
from ara.contracts import driver

# The built-in worker script (ships in the ARA repo, runs in the isolated cpu env by path).
WORKER = Path(__file__).resolve().parent.parent / "workers" / "cpu_llama.py"

ENV_NAME = "cpu"
CALIBRATION_ENGINE = "cpu"   # profiles key for this machine's stored overhead
RAMP_SCHEDULE = [2000, 4000, 8000, 16000, 32000, 65536, 131072]
DEFAULT_MARGIN_GB = 2.0      # margin CAP (GB); the worker scales it to ~10% of RAM, floor 0.5GB
DEFAULT_OVERHEAD_GB = 1.0    # fallback cold-start overhead until calibrated

# A small GGUF ARA characterizes against in the profile flow (the worker auto-picks its
# smallest quant and downloads it on demand).
CALIBRATION_MODEL = "bartowski/SmolLM2-135M-Instruct-GGUF"


def safe_limits() -> dict:
    """This machine's safe RAM limits via the cpu worker. Pure read — no model load.

    Like CUDA's VRAM (and unlike Apple's hidden MLX cold-start overhead), the wall is read
    exactly — physical RAM — so there's nothing to calibrate for the budget itself
    (``calibrated`` True, ``overhead_gb`` None). Characterizing a *model*'s context ceiling is a
    separate, optional step.
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
    """Characterize *model* on CPU and attach it to the limits (for the profile flow).

    If characterization fails (model unavailable, worker error), returns an uncalibrated result
    with a ``calibration_error`` field (never ``calibrated=True`` for unobserved data — Rule #3).
    The safe default overhead is still in effect via ``_budget_params``; callers can detect the
    condition via ``calibrated=False`` + presence of ``calibration_error``.
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
    calibration for the cpu engine, or a safe default if uncalibrated."""
    overhead = DEFAULT_OVERHEAD_GB
    with db.connected() as con:
        stored = calibration.get_calibration(con, CALIBRATION_ENGINE)
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
    """Measure *model*'s safe context ceiling on CPU — the thin path, same driver as Apple.

    Pure wiring: ARA owns the methodology in the engine-agnostic ``contracts.driver``; this
    adapter supplies only the CPU specifics — the isolated ``cpu`` env, the built-in
    ``cpu_llama`` worker, budget params, and the schedule. Crash-safety is layered: the driver
    gates each rung (L1 + L2), and the worker refuses-before-load (L4) / aborts mid-probe (L5).
    Returns ``{model, safe_context, points}``.

    ``progress=True`` streams the worker's stderr live so HF's native tqdm bars are visible
    during the GGUF fetch that the CPU worker handles in-process.
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
    """One-shot completion on CPU, governed: ``max_context`` is the characterized safe ceiling,
    so the worker's KV cache is capped under the wall (the worker still self-vetoes, L4/L5).
    Out-of-process in the isolated ``cpu`` env; the prompt goes over stdin, never argv. Returns
    ``{context, completion}`` or a refusal (``{refused, reason}``)."""
    margin, overhead = _budget_params()
    return engine_env.run_worker(
        ENV_NAME,
        [str(WORKER), model, str(max_context), "--generate",
         "--margin", str(margin), "--overhead", str(overhead),
         "--max-tokens", str(max_tokens)],
        input=prompt)


def benchmark(model: str, prompts: list, *, max_context: int,
              max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    """Multi-prompt CPU benchmark, governed: the worker loads the GGUF **once** (KV capped at
    ``max_context`` — the characterized safe ceiling) and iterates the prompt array, enforcing the
    ceiling per item. The prompts go as a JSON array over stdin, never argv. Returns the worker
    dict verbatim: ``{"context": N, "results": [...]}`` or a gate refusal ``{"context": N,
    "refused": true, "reason": "..."}``. ARA never imports llama.cpp in-process; ``max_context``
    is the characterized safe ceiling."""
    margin, overhead = _budget_params()
    return engine_env.run_worker(
        ENV_NAME,
        [str(WORKER), model, str(max_context), "--benchmark",
         "--margin", str(margin), "--overhead", str(overhead),
         "--max-tokens", str(max_tokens)],
        input=json.dumps(prompts))
