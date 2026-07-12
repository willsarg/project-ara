# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Apple-Silicon backend adapter — drives wmx-suite's MLX measurement out-of-process.


A lean device oracle, symmetric with backends/cuda.py: it reads the machine's memory wall and
runs wmx-suite's crash-safe calibration, but it owns **no persistence** — ARA stores and reuses
the calibration (see cli.render_profile). It never imports wmx in-process: every engine call
goes through the isolated ``apple`` env via :mod:`ara.engine_env`, so nothing MLX-shaped loads
in ARA's interpreter and the core stays engine-free at runtime, not just at lock time.
"""
from __future__ import annotations

import json

# Core, engine-free helpers (no wmx) — safe to import at module load and patchable in tests.
from ara import calibration, db, engine_env
from ara.contracts import driver

# The wmx worker modules ARA drives in the isolated apple env (never imported in-process).
DEVICE_MODULE = "ara_engine_mlx.device"

# Model ARA calibrates against — smallest SmolLM (MLX 4-bit). Calibration only measures
# fixed memory overhead, so a tiny instruct model is plenty.
CALIBRATION_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"

# wmx denominates memory in DECIMAL GB (bytes / 1e9) — except swap, which it reports in binary
# GiB already. ARA's contract is binary GiB throughout (matching detect, the workers, and wcx),
# so wmx facts are converted at this boundary. Slug 2026-07-02-analytic-units-gib.
_GIB = 1024 ** 3
_DEC_TO_GIB = 1e9 / _GIB


def _facts_to_gib(facts: dict) -> dict:
    """Convert wmx's decimal-GB limit facts to ARA's binary-GiB contract.

    The margin-bearing fields are RE-DERIVED rather than scaled: ``margin_gb`` is an absolute
    policy cushion, and multiplying it by the ≈0.93 conversion factor would silently shrink it
    ~7% (an unsafe direction). So: wall/total convert directly; ``safe_budget_gb`` becomes the
    converted wall minus the unchanged margin; ``headroom_gb`` (= safe − wired) is rebuilt from
    the converted wired footprint. ``swap_free_gb`` is already GiB and passes through.
    """
    wall = facts["wall_gb"] * _DEC_TO_GIB
    margin = facts["margin_gb"]
    safe = wall - margin
    # wmx's headroom = its (decimal) safe budget − wired-now; recover wired, convert, re-derive.
    wired = (facts["safe_budget_gb"] - facts["headroom_gb"]) * _DEC_TO_GIB
    return {
        **facts,
        "total_gb": facts["total_gb"] * _DEC_TO_GIB,
        "wall_gb": wall,
        "safe_budget_gb": safe,
        "headroom_gb": safe - wired,
    }


def safe_limits() -> dict:
    """Read this machine's safe memory limits via the wmx worker. Pure read — no model.

    Stateless: returns the budget with no stored overhead (``calibrated=False``). ARA overlays
    a previously-measured overhead from its own store — the engine no longer reads a database.
    Facts arrive in wmx's decimal GB and leave here in ARA's binary GiB (see ``_facts_to_gib``).
    """
    facts = _facts_to_gib(engine_env.run_worker("apple", ["-m", DEVICE_MODULE, "limits"]))
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


def download_calibration_model(model: str = CALIBRATION_MODEL, *,
                               progress: bool = False) -> None:
    """Fetch the calibration model into the HF cache. Network + disk only."""
    from ara import acquire

    acquire.download(model, progress=progress)


def calibrate(model: str = CALIBRATION_MODEL) -> dict:
    """Run wmx-suite's crash-safe calibration via the worker; return fresh limits + what it
    measured.

    The worker loads the model and watches memory under wmx-suite's predictive safety ramp,
    which aborts before approaching the safe budget. ARA only invokes it (out-of-process in the
    apple env). Surfaces the **effective** cold-start overhead (clamped to the engine's floor:
    ``max(default, measured)``) as ``overhead_gb`` so ARA can persist it; the raw measurement is
    in the ``"calibration"`` sub-dict for the caller to show.

    If the worker fails (error dict or exception), returns an uncalibrated result with a
    ``calibration_error`` field (never ``calibrated=True`` for unobserved data — Rule #3).
    The safe default overhead is still in effect via ``_budget_params``; callers can detect the
    condition via ``calibrated=False`` + presence of ``calibration_error``.
    """
    limits = safe_limits()
    try:
        result = engine_env.run_worker("apple", ["-m", DEVICE_MODULE, "calibrate", model])
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
    # Overheads are wmx-measured (decimal GB) and get subtracted from the GiB wall later —
    # convert them to the wall's units here (Slug 2026-07-02-analytic-units-gib). The raw
    # engine numbers stay visible unconverted in the "calibration" sub-dict.
    overheads = [v * _DEC_TO_GIB for v in (result.get("measured_overhead_gb"),
                                           result.get("default_overhead_gb")) if v is not None]
    limits["overhead_gb"] = max(overheads) if overheads else None
    limits["calibrated"] = True
    limits["calibration"] = result
    return limits


# ARA-owned ramp policy (the engine only measures; ARA decides the schedule + safety margin).
WORKER_MODULE = "ara_engine_mlx.measure_one"
RAMP_SCHEDULE = [2000, 4000, 8000, 16000, 32000, 65536, 131072]
DEFAULT_MARGIN_GB = 2.0      # safety cushion below the wall (ARA policy)
DEFAULT_OVERHEAD_GB = 1.0    # fallback cold-start overhead until calibrated


def _budget_params() -> tuple[float, float]:
    """ARA-owned (margin, overhead). Margin is policy; overhead is this machine's stored
    calibration for the wmx engine, or a safe default if uncalibrated."""
    overhead = DEFAULT_OVERHEAD_GB
    with db.connected() as con:
        stored = calibration.get_calibration(con, "wmx")
    if stored and stored.get("fixed_overhead_gb") is not None:
        overhead = stored["fixed_overhead_gb"]
    return DEFAULT_MARGIN_GB, overhead


# KV-cache quant lever (parity with the Vulkan lane). ARA's cross-engine `--kv-quant`
# {f16,q8_0,q4_0} maps to MLX's integer kv-bits (fp16 = no quant). The effective bytes/elem
# (8-bit/4-bit payload + an fp16 scale+bias per 64-elem group) feeds the KV-aware decode
# estimate so it reflects the cache actually in use — not always fp16.
_MLX_KV_BITS = {"f16": None, "q8_0": 8, "q4_0": 4}
_MLX_KV_BYTES = {"f16": 2.0, "q8_0": 8 / 8 + 2 * 2 / 64, "q4_0": 4 / 8 + 2 * 2 / 64}


def _worker_argv(model: str, ctx: int, margin: float, overhead: float, *,
                 preflight: bool = False, kv_quant: str = "f16") -> list[str]:
    argv = ["-m", WORKER_MODULE, model, str(ctx),
            "--margin", str(margin), "--overhead", str(overhead)]
    if preflight:
        argv.append("--preflight")
    bits = _MLX_KV_BITS[kv_quant]
    if bits is not None:
        argv += ["--kv-bits", str(bits)]
    return argv


def characterize(model: str, *, progress: bool = False, kv_quant: str = "f16") -> dict:
    """Measure *model*'s safe context ceiling on this Mac — the thin path.

    Pure wiring: ARA owns the methodology in the engine-agnostic ``contracts.driver`` (the
    antidote to an Apple-shaped abstraction); this adapter only supplies the Apple specifics —
    the isolated ``apple`` env, wmx's self-vetoing ``measure_one`` worker, the budget params,
    and the schedule. ARA never imports wmx in-process. Crash-safety is layered: the driver
    gates each rung (L1 ``plan_next`` + L2 actual-footprint check), the engine refuses-before-
    load (L4) and a watchdog aborts mid-probe (L5). Returns ``{model, safe_context, points}``.

    ``progress`` is accepted for interface symmetry with the cpu backend but has no effect
    here: the HF download bar already ran in-process during the pre-fetch step.
    """
    margin, overhead = _budget_params()
    return driver.characterize(
        model,
        preflight=lambda m: engine_env.run_worker(
            "apple", _worker_argv(m, 0, margin, overhead, preflight=True, kv_quant=kv_quant)),
        measure=lambda m, ctx: engine_env.run_worker(
            "apple", _worker_argv(m, ctx, margin, overhead, kv_quant=kv_quant)),
        schedule=RAMP_SCHEDULE,
        kv_dtype_bytes=_MLX_KV_BYTES[kv_quant],   # decode-ceiling estimate reflects the cache type
    )


def serve(model: str, *, port: int, max_context: int,
          kv_quant: str = "f16", measured_slope_gb_per_k: float | None = None) -> tuple:
    """Start a governed MLX server for *model* via ara_engine_mlx.serve, out-of-process.

    Spawns the isolated ``apple`` env's python running
    ``python -m ara_engine_mlx.serve <model> <max_context> --margin G --overhead G --port N
    [--kv-bits N] [--measured-slope S]``. Reads stdout until the worker emits its ready JSON
    (``{"ready": true, "url": "...", "context": N}``), then returns
    ``(proc, url, context)`` without waiting — the server keeps running.

    ``measured_slope_gb_per_k`` (ARA fits it from this model's stored characterization when
    serving the measured ceiling) makes the pre-load gate predict with the real ramp slope
    instead of the conservative a-priori one — so a measured long-window serve isn't falsely
    refused (slug 2026-07-02-wmx-serve-measured-provenance-gate). Omit for the a-priori gate.

    ARA never imports MLX in-process; all engine calls go through
    :func:`engine_env.start_worker_server`. The ceiling passed as *max_context* must
    be the characterized safe ceiling for this machine (Rule #1).
    """
    margin, overhead = _budget_params()
    argv = ["-m", "ara_engine_mlx.serve", model, str(max_context),
            "--margin", str(margin), "--overhead", str(overhead),
            "--port", str(port)]
    bits = _MLX_KV_BITS[kv_quant]
    if bits is not None:
        argv += ["--kv-bits", str(bits)]
    if measured_slope_gb_per_k is not None:
        argv += ["--measured-slope", str(measured_slope_gb_per_k)]
    proc, info = engine_env.start_worker_server("apple", argv)
    return proc, info["url"], info["context"]


DEFAULT_MAX_TOKENS = 256


def generate(model, prompt, *, max_context, max_tokens=DEFAULT_MAX_TOKENS,
             kv_quant: str = "f16") -> dict:
    """One-shot MLX completion, governed: max_context is the characterized safe ceiling, so the
    worker generates under the wall. Out-of-process in the isolated `apple` env via wmx-suite's
    generate worker; the prompt goes over stdin, never argv. ``kv_quant`` (default ``"f16"``)
    should match how *model* was characterized. Returns {context, completion} or a refusal
    {refused, reason}. ARA never imports MLX in-process."""
    margin, overhead = _budget_params()
    argv = ["-m", "ara_engine_mlx.generate", model, str(max_context),
            "--margin", str(margin), "--overhead", str(overhead),
            "--max-tokens", str(max_tokens)]
    bits = _MLX_KV_BITS[kv_quant]
    if bits is not None:
        argv += ["--kv-bits", str(bits)]
    return engine_env.run_worker("apple", argv, input=prompt)


def benchmark(model: str, prompts: list, *, max_context: int,
              max_tokens: int = DEFAULT_MAX_TOKENS, kv_quant: str = "f16") -> dict:
    """Multi-prompt MLX benchmark, governed: max_context is the characterized safe ceiling.

    Spawns the isolated ``apple`` env's python running
    ``python -m ara_engine_mlx.benchmark <model> <max_context> --margin G --overhead G
    --max-tokens N [--kv-bits N]`` with the JSON prompt array on stdin. The worker loads
    the model once and iterates over all prompts; per-prompt governance enforces the ceiling
    for each item individually. Returns the worker dict verbatim:
    ``{"context": N, "results": [...]}`` or a gate refusal ``{"context": N, "refused": true,
    "reason": "..."}``. ARA never imports MLX in-process; max_context is the characterized
    safe ceiling.
    """
    margin, overhead = _budget_params()
    argv = ["-m", "ara_engine_mlx.benchmark", model, str(max_context),
            "--margin", str(margin), "--overhead", str(overhead),
            "--max-tokens", str(max_tokens)]
    bits = _MLX_KV_BITS[kv_quant]
    if bits is not None:
        argv += ["--kv-bits", str(bits)]
    return engine_env.run_worker("apple", argv, input=json.dumps(prompts))
