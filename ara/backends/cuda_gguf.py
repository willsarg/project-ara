# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""CUDA-GGUF backend — partial GPU offload of a GGUF on NVIDIA, runs in the ``cuda_gguf`` env.

ARA's first **two-wall** engine: offloads K of N layers to discrete VRAM, runs the remaining
N−K on the CPU — so it must stay under both the discrete VRAM wall AND the system-RAM wall at
once. Every other engine governs a single wall. **Opt-in** via ``--engine cuda-gguf`` — NVIDIA
auto-picks ``cuda`` (the full-GPU transformers engine); cuda-gguf is the hybrid fallback when the
model is too large for VRAM alone.

Contract class: **ramp** (safe context ceiling). Two wall sources: **discrete VRAM**
(nvidia-smi) and **system RAM** (psutil) — read exactly, like the CPU engine. No calibrated
overhead: both walls are read at the time of the run and the per-layer split is auto-fitted.
Nothing to calibrate for the budget itself (``calibrated`` True, ``overhead_gb`` None).

Built into ARA (the heavy CUDA/MLX stacks stay in separately installed native packages). The worker is a self-contained
script — ``ara/workers/cuda_gguf_llama.py``, which never imports ``ara`` — run by the isolated
``cuda_gguf`` env's own python over ``engine_env``, so llama-cpp-python never enters ARA core's
lock. ``characterize`` is pure wiring into the engine-agnostic
:func:`ara.contracts.driver.characterize`.

Design: Designs/specs/2026-06-29-cuda-gguf-hybrid-two-wall-engine.md
"""
from __future__ import annotations

import json
from pathlib import Path

# Core, engine-free helpers (no llama.cpp) — safe at module load and patchable in tests.
from ara import acquire, calibration, db, engine_env, methodology, staleness
from ara.contracts import driver

# The built-in worker script (ships in the ARA repo, runs in the isolated cuda_gguf env by path).
WORKER = Path(__file__).resolve().parent.parent / "workers" / "cuda_gguf_llama.py"

ENV_NAME = "cuda_gguf"
CALIBRATION_ENGINE = "cuda_gguf"   # profiles key for this machine's stored calibration
RAMP_SCHEDULE = [2000, 4000, 8000, 16000, 32000, 65536, 131072]
DEFAULT_VRAM_MARGIN_GB = 1.0     # VRAM safety margin (GB); the worker floors at 0.5 GB
DEFAULT_RAM_MARGIN_GB = 2.0      # RAM safety margin (GB); the worker scales to ~10% of RAM

# A small GGUF ARA characterizes against in the profile flow (the worker auto-picks its
# smallest quant and downloads it on demand) — the same calibration model as the CPU engine.
CALIBRATION_MODEL = "bartowski/SmolLM2-135M-Instruct-GGUF"


def safe_limits() -> dict:
    """This machine's safe VRAM + RAM limits via the cuda_gguf worker. Pure read — no model load.

    Like the CPU engine (and unlike Apple's hidden MLX cold-start overhead), both walls are read
    exactly — VRAM from nvidia-smi, RAM from psutil — so there's nothing to calibrate for the
    budget itself (``calibrated`` True, ``overhead_gb`` None). Characterizing a *model*'s context
    ceiling on the hybrid GPU+CPU path is a separate, optional step.
    """
    facts = engine_env.run_worker(
        ENV_NAME, [str(WORKER), "--limits",
                   "--vram-margin", str(DEFAULT_VRAM_MARGIN_GB),
                   "--ram-margin", str(DEFAULT_RAM_MARGIN_GB)])
    return {
        **facts,
        "overhead_gb": None,         # both walls are read exactly — nothing to calibrate
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
    """Characterize *model* on the hybrid GPU+CPU path and attach it to the limits.

    If characterization fails (model unavailable, no CUDA offload, worker error), returns an
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


def _budget_params(*, engine_fingerprint: str | None = None) -> tuple[float, float]:
    """ARA-owned margins; exact-build calibration may override the policy defaults."""
    vram_margin = DEFAULT_VRAM_MARGIN_GB
    ram_margin = DEFAULT_RAM_MARGIN_GB
    stored = None
    if engine_fingerprint is not None:
        with db.connected() as con:
            stored = calibration.get_calibration(
                con, CALIBRATION_ENGINE,
                engine_fingerprint=engine_fingerprint)
    if stored and stored.get("vram_margin_gb") is not None:
        vram_margin = stored["vram_margin_gb"]
    if stored and stored.get("ram_margin_gb") is not None:
        ram_margin = stored["ram_margin_gb"]
    return vram_margin, ram_margin


def _worker_argv(model: str, ctx: int, vram_margin: float, ram_margin: float, *,
                 preflight: bool = False) -> list[str]:
    argv = [str(WORKER), model, str(ctx),
            "--vram-margin", str(vram_margin), "--ram-margin", str(ram_margin)]
    if preflight:
        argv.append("--preflight")
    return argv


def characterization_methodology(*, vram_margin_gb: float | None = None,
                                 ram_margin_gb: float | None = None) -> dict:
    """Current two-wall characterization behavior used to authorize evidence reuse."""
    vram_margin_gb = (DEFAULT_VRAM_MARGIN_GB
                      if vram_margin_gb is None else vram_margin_gb)
    ram_margin_gb = (DEFAULT_RAM_MARGIN_GB
                     if ram_margin_gb is None else ram_margin_gb)
    return methodology.characterization_descriptor(
        schedule=RAMP_SCHEDULE, repeats=3,
        reserve_policy="two-wall-fixed-reserve",
        reserve_bytes={
            "vram": round(vram_margin_gb * 1024 ** 3),
            "ram": round(ram_margin_gb * 1024 ** 3),
        },
        worker_protocol="ara-cuda-gguf-llama-measurement:v2",
        sampling_interval_ms=0,
        telemetry_failure_policy="dimension-bound-two-wall-fail-closed:v2",
        watchdog_stop_rule="either-wall-gte-budget:v1")


def characterize(model: str, *, progress: bool = False,
                 engine_fingerprint: str | None = None) -> dict:
    """Measure *model*'s safe context ceiling on the hybrid GPU+CPU path.

    Pure wiring: ARA owns the methodology in the engine-agnostic ``contracts.driver``; this
    adapter supplies only the cuda_gguf specifics — the isolated ``cuda_gguf`` env, the built-in
    ``cuda_gguf_llama`` worker, budget params, and the schedule. Crash-safety is layered: the
    driver gates each rung (L1 + L2), and the worker refuses-before-load (L4) / aborts mid-probe
    (L5). The shared scalar fit is explicitly bound to absolute system RAM; every successful
    point also persists the independently checked VRAM observation and both budgets with GiB
    units and load-log provenance. Returns ``{model, safe_context, points}``.

    ``progress=True`` streams the worker's stderr live so HF's native tqdm bars are visible.
    ``kv_dtype_bytes`` is fixed at 2.0 (fp16) — the worker does not expose KV quantization.
    """
    vram_margin, ram_margin = (
        _budget_params()
        if engine_fingerprint is None
        else _budget_params(engine_fingerprint=engine_fingerprint)
    )
    return driver.characterize(
        model,
        preflight=lambda m: engine_env.run_worker(
            ENV_NAME, _worker_argv(m, 0, vram_margin, ram_margin, preflight=True),
            stream=progress),
        measure=lambda m, ctx: engine_env.run_worker(
            ENV_NAME, _worker_argv(m, ctx, vram_margin, ram_margin),
            stream=progress),
        schedule=RAMP_SCHEDULE,
        kv_dtype_bytes=2.0,
        methodology_descriptor=characterization_methodology(
            vram_margin_gb=vram_margin, ram_margin_gb=ram_margin),
    )


DEFAULT_MAX_TOKENS = 256


def generate(model: str, prompt: str, *, max_context: int,
             max_tokens: int = DEFAULT_MAX_TOKENS,
             engine_fingerprint: str | None = None) -> dict:
    """One-shot completion on the hybrid GPU+CPU path, governed: ``max_context`` is the
    characterized safe ceiling, so the worker's KV cache is capped under both walls (the worker
    still self-vetoes on both, L4/L5). Out-of-process in the isolated ``cuda_gguf`` env; the
    prompt goes over stdin, never argv. Returns ``{context, gpu_layers, completion}`` or a
    refusal (``{context, refused, reason}``)."""
    vram_margin, ram_margin = (
        _budget_params()
        if engine_fingerprint is None
        else _budget_params(engine_fingerprint=engine_fingerprint)
    )
    argv = [str(WORKER), model, str(max_context), "--generate",
            "--vram-margin", str(vram_margin), "--ram-margin", str(ram_margin),
            "--max-tokens", str(max_tokens)]
    return engine_env.run_worker(ENV_NAME, argv, input=prompt)


def benchmark(model: str, prompts: list, *, max_context: int,
              max_tokens: int = DEFAULT_MAX_TOKENS,
              engine_fingerprint: str | None = None) -> dict:
    """Multi-prompt GPU+CPU benchmark, governed: the worker loads the GGUF **once** (K layers
    offloaded, KV capped at ``max_context`` — the characterized safe ceiling) and iterates the
    prompt array, enforcing both walls per item. The prompts go as a JSON array over stdin, never
    argv. Returns the worker dict verbatim: ``{"context": N, "gpu_layers": K, "results": [...]}``
    or a gate refusal. ARA never imports llama.cpp in-process; ``max_context`` is the characterized
    safe ceiling."""
    vram_margin, ram_margin = (
        _budget_params()
        if engine_fingerprint is None
        else _budget_params(engine_fingerprint=engine_fingerprint)
    )
    argv = [str(WORKER), model, str(max_context), "--benchmark",
            "--vram-margin", str(vram_margin), "--ram-margin", str(ram_margin),
            "--max-tokens", str(max_tokens)]
    return engine_env.run_worker(ENV_NAME, argv, input=json.dumps(prompts))
