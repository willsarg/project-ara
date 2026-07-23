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

import json
from pathlib import Path

# Core, engine-free helpers (no llama.cpp) — safe at module load and patchable in tests.
from ara import acquire, calibration, db, engine_env, methodology, staleness
from ara.contracts import driver
# The worker's KV-byte map is the single source of truth for KV-quant element sizes; its module
# top level is stdlib-only (no llama.cpp), so importing it here is engine-free.
from ara.workers.vulkan_llama import _KV_BYTES

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


def _budget_params(*, engine_fingerprint: str | None = None) -> tuple[float, float]:
    """ARA-owned (margin, overhead). Margin is policy; overhead is this machine's stored
    calibration for the exact Vulkan engine build, or a safe default if unscoped."""
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
                 preflight: bool = False, flash_attn: bool = True,
                 kv_quant: str = "f16") -> list[str]:
    argv = [str(WORKER), model, str(ctx),
            "--margin", str(margin), "--overhead", str(overhead)]
    if preflight:
        argv.append("--preflight")
    if not flash_attn:
        argv.append("--no-flash-attn")
    if kv_quant != "f16":
        argv += ["--kv-quant", kv_quant]
    return argv


def characterization_methodology(*, margin_gb: float | None = None) -> dict:
    """Current Vulkan characterization behavior used to authorize evidence reuse."""
    margin = DEFAULT_MARGIN_GB if margin_gb is None else margin_gb
    return methodology.characterization_descriptor(
        schedule=RAMP_SCHEDULE, repeats=3,
        reserve_policy="physical-ram-minus-scaled-reserve",
        reserve_bytes=round(margin * 1024 ** 3),
        worker_protocol="ara-vulkan-llama-measurement:v1",
        sampling_interval_ms=50,
        telemetry_failure_policy="in-worker-system-memory-watchdog:v1",
        watchdog_stop_rule="system-used-gte-budget:v1")


def characterize(model: str, *, progress: bool = False, flash_attn: bool = True,
                 kv_quant: str = "f16",
                 fixed_overhead_gb: float | None = None,
                 engine_fingerprint: str | None = None) -> dict:
    """Measure *model*'s safe context ceiling on the GPU — the thin path, same driver as CPU/Apple.

    Pure wiring: ARA owns the methodology in the engine-agnostic ``contracts.driver``; this
    adapter supplies only the vulkan specifics — the isolated ``vulkan`` env, the built-in
    ``vulkan_llama`` worker, budget params, and the schedule. Crash-safety is layered: the driver
    gates each rung (L1 + L2), and the worker refuses-before-load (L4) / aborts mid-probe (L5).
    Returns ``{model, safe_context, points}``.

    ``flash_attn`` (default True): measure with Vulkan flash-attention, which ~doubles the context
    ceiling on this memory-bound APU at no quality cost (small prefill penalty). ``kv_quant``
    (default ``"f16"``, lossless): ``"q8_0"`` ~halves KV memory near-losslessly, ``"q4_0"`` ~quarters
    it at a quality cost — a further context lever that forces flash-attention on. Keep both
    consistent with ``generate`` — the ceiling is measured under the same attention/KV path the run
    will use. ``progress=True`` streams the worker's stderr live so HF's native tqdm bars are visible.
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
            ENV_NAME, _worker_argv(m, 0, margin, overhead, preflight=True,
                                   flash_attn=flash_attn, kv_quant=kv_quant),
            stream=progress),
        measure=lambda m, ctx: engine_env.run_worker(
            ENV_NAME, _worker_argv(m, ctx, margin, overhead,
                                   flash_attn=flash_attn, kv_quant=kv_quant),
            stream=progress),
        schedule=RAMP_SCHEDULE,
        kv_dtype_bytes=_KV_BYTES[kv_quant],   # decode-ceiling estimate reflects the KV cache type
        methodology_descriptor=characterization_methodology(margin_gb=margin),
    )


DEFAULT_MAX_TOKENS = 256


def generate(model: str, prompt: str, *, max_context: int,
             max_tokens: int = DEFAULT_MAX_TOKENS, flash_attn: bool = True,
             kv_quant: str = "f16",
             engine_fingerprint: str | None = None) -> dict:
    """One-shot completion on the GPU, governed: ``max_context`` is the characterized safe ceiling,
    so the worker's KV cache is capped under the wall (the worker still self-vetoes, L4/L5).
    Out-of-process in the isolated ``vulkan`` env; the prompt goes over stdin, never argv. Returns
    ``{context, completion}`` or a refusal (``{refused, reason}``).

    ``flash_attn`` (default True) and ``kv_quant`` (default ``"f16"``) should match how *model* was
    characterized; if they don't, the worker's L4/L5 gates still keep the run wall-safe (worst case
    a refusal, never a crash)."""
    margin, overhead = (
        _budget_params()
        if engine_fingerprint is None
        else _budget_params(engine_fingerprint=engine_fingerprint)
    )
    argv = [str(WORKER), model, str(max_context), "--generate",
            "--margin", str(margin), "--overhead", str(overhead),
            "--max-tokens", str(max_tokens)]
    if not flash_attn:
        argv.append("--no-flash-attn")
    if kv_quant != "f16":
        argv += ["--kv-quant", kv_quant]
    return engine_env.run_worker(ENV_NAME, argv, input=prompt)


def benchmark(model: str, prompts: list, *, max_context: int,
              max_tokens: int = DEFAULT_MAX_TOKENS, flash_attn: bool = True,
              kv_quant: str = "f16",
              engine_fingerprint: str | None = None) -> dict:
    """Multi-prompt GPU benchmark, governed: the worker loads the GGUF **once** (offloaded, KV
    capped at ``max_context`` — the characterized safe ceiling) and iterates the prompt array,
    enforcing the ceiling per item. The prompts go as a JSON array over stdin, never argv. Returns
    the worker dict verbatim: ``{"context": N, "results": [...]}`` or a gate refusal
    ``{"context": N, "refused": true, "reason": "..."}``. ARA never imports llama.cpp in-process;
    ``max_context`` is the characterized safe ceiling."""
    margin, overhead = (
        _budget_params()
        if engine_fingerprint is None
        else _budget_params(engine_fingerprint=engine_fingerprint)
    )
    argv = [str(WORKER), model, str(max_context), "--benchmark",
            "--margin", str(margin), "--overhead", str(overhead),
            "--max-tokens", str(max_tokens)]
    if not flash_attn:
        argv.append("--no-flash-attn")
    if kv_quant != "f16":
        argv += ["--kv-quant", kv_quant]
    return engine_env.run_worker(ENV_NAME, argv, input=json.dumps(prompts))
