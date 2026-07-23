# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""CPU backend — system-RAM inference via llama.cpp (GGUF) — the second real engine.

Contract class: **ramp** (safe context ceiling). Wall source: **physical RAM** (swap is
reported but never counted toward the budget — swapping during inference is catastrophically
slow, so it's not usable headroom). One adapter for *all* CPU-only inference regardless of ISA
— x86 (Intel/AMD), arm64, Raspberry Pi, riscv-when-it-matters. The ISA is metadata reported by
``detect``, not a separate backend, because it doesn't change the wall. This is the universal
fallback: nearly every machine matches it.

It exists to keep the abstraction honest. Unlike Apple (whose native engine is a separately
installed package), llama.cpp is **built into ARA**. The worker is a self-contained script —
``ara/workers/cpu_llama.py``, which never imports ``ara`` — run by the isolated ``cpu`` env's
own python over ``engine_env``, so llama-cpp-python never enters ARA core's lock. ``characterize``
is pure wiring into the engine-agnostic :func:`ara.contracts.driver.characterize`; the only
Apple↔CPU differences are the env name, the worker, and the wall source.
"""
from __future__ import annotations

import json
from pathlib import Path

# Core, engine-free helpers (no llama.cpp) — safe at module load and patchable in tests.
from ara import acquire, calibration, db, engine_env, methodology, staleness
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
    """Whether the exact GGUF selection already has a locally identifiable artifact."""
    return staleness.artifact_identity(model) is not None


def download_calibration_model(model: str = CALIBRATION_MODEL, *,
                               progress: bool = False) -> None:
    """Download only the selected GGUF before governed measurement pins it."""
    acquire.download_gguf(model, progress=progress)


def prepare_download(model: str):
    """Bind GGUF sizing, selection, and download to one immutable Hub revision."""
    return acquire.prepare_download(model, gguf=True)


def download_prepared_model(plan, *, progress: bool = False) -> None:
    acquire.download_prepared(plan, progress=progress)


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


def _budget_params(*, engine_fingerprint: str | None = None) -> tuple[float, float]:
    """ARA-owned (margin, overhead). Margin is policy; overhead is this machine's stored
    calibration for the exact CPU engine build, or a safe default if unscoped."""
    overhead = DEFAULT_OVERHEAD_GB
    stored = None
    if engine_fingerprint is not None:
        with db.connected() as con:
            stored = calibration.get_calibration(
                con, CALIBRATION_ENGINE,
                engine_fingerprint=engine_fingerprint)
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


def characterization_methodology(*, margin_gb: float | None = None) -> dict:
    """Current CPU characterization behavior used to authorize evidence reuse."""
    margin = DEFAULT_MARGIN_GB if margin_gb is None else margin_gb
    return methodology.characterization_descriptor(
        schedule=RAMP_SCHEDULE, repeats=3,
        reserve_policy="physical-ram-minus-scaled-reserve",
        reserve_bytes=round(margin * 1024 ** 3),
        worker_protocol="ara-cpu-llama-measurement:v1",
        sampling_interval_ms=50,
        telemetry_failure_policy="in-worker-system-memory-watchdog:v1",
        watchdog_stop_rule="system-used-gte-budget:v1")


def characterize(model: str, *, progress: bool = False,
                 fixed_overhead_gb: float | None = None,
                 engine_fingerprint: str | None = None) -> dict:
    """Measure *model*'s safe context ceiling on CPU — the thin path, same driver as Apple.

    Pure wiring: ARA owns the methodology in the engine-agnostic ``contracts.driver``; this
    adapter supplies only the CPU specifics — the isolated ``cpu`` env, the built-in
    ``cpu_llama`` worker, budget params, and the schedule. Crash-safety is layered: the driver
    gates each rung (L1 + L2), and the worker refuses-before-load (L4) / aborts mid-probe (L5).
    Returns ``{model, safe_context, points}``.

    ``progress=True`` streams the worker's stderr live so HF's native tqdm bars are visible
    during the GGUF fetch that the CPU worker handles in-process.
    """
    if fixed_overhead_gb is None:
        margin, overhead = (
            _budget_params()
            if engine_fingerprint is None
            else _budget_params(engine_fingerprint=engine_fingerprint)
        )
    else:
        margin, overhead = DEFAULT_MARGIN_GB, fixed_overhead_gb
    return driver.characterize(
        model,
        preflight=lambda m: engine_env.run_worker(
            ENV_NAME, _worker_argv(m, 0, margin, overhead, preflight=True),
            stream=progress),
        measure=lambda m, ctx: engine_env.run_worker(
            ENV_NAME, _worker_argv(m, ctx, margin, overhead),
            stream=progress),
        schedule=RAMP_SCHEDULE,
        methodology_descriptor=characterization_methodology(margin_gb=margin),
    )


DEFAULT_MAX_TOKENS = 256


def generate(model: str, prompt: str, *, max_context: int,
             max_tokens: int = DEFAULT_MAX_TOKENS,
             engine_fingerprint: str | None = None) -> dict:
    """One-shot completion on CPU, governed: ``max_context`` is the characterized safe ceiling,
    so the worker's KV cache is capped under the wall (the worker still self-vetoes, L4/L5).
    Out-of-process in the isolated ``cpu`` env; the prompt goes over stdin, never argv. Returns
    ``{context, completion}`` or a refusal (``{refused, reason}``)."""
    margin, overhead = (
        _budget_params()
        if engine_fingerprint is None
        else _budget_params(engine_fingerprint=engine_fingerprint)
    )
    return engine_env.run_worker(
        ENV_NAME,
        [str(WORKER), model, str(max_context), "--generate",
         "--margin", str(margin), "--overhead", str(overhead),
         "--max-tokens", str(max_tokens)],
        input=prompt)


def benchmark(model: str, prompts: list, *, max_context: int,
              max_tokens: int = DEFAULT_MAX_TOKENS,
              engine_fingerprint: str | None = None) -> dict:
    """Multi-prompt CPU benchmark, governed: the worker loads the GGUF **once** (KV capped at
    ``max_context`` — the characterized safe ceiling) and iterates the prompt array, enforcing the
    ceiling per item. The prompts go as a JSON array over stdin, never argv. Returns the worker
    dict verbatim: ``{"context": N, "results": [...]}`` or a gate refusal ``{"context": N,
    "refused": true, "reason": "..."}``. ARA never imports llama.cpp in-process; ``max_context``
    is the characterized safe ceiling."""
    margin, overhead = (
        _budget_params()
        if engine_fingerprint is None
        else _budget_params(engine_fingerprint=engine_fingerprint)
    )
    return engine_env.run_worker(
        ENV_NAME,
        [str(WORKER), model, str(max_context), "--benchmark",
         "--margin", str(margin), "--overhead", str(overhead),
         "--max-tokens", str(max_tokens)],
        input=json.dumps(prompts))
