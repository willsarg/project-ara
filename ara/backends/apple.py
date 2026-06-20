"""Apple-Silicon backend adapter — drives wmx-suite's MLX measurement out-of-process.

A lean device oracle, symmetric with backends/cuda.py: it reads the machine's memory wall and
runs wmx-suite's crash-safe calibration, but it owns **no persistence** — ARA stores and reuses
the calibration (see cli.render_profile). It never imports wmx in-process: every engine call
goes through the isolated ``apple`` env via :mod:`ara.engine_env`, so nothing MLX-shaped loads
in ARA's interpreter and the core stays engine-free at runtime, not just at lock time.
"""
from __future__ import annotations

# Core, engine-free helpers (no wmx) — safe to import at module load and patchable in tests.
from ara import db, engine_env, profiles
from ara.contracts import driver

# The wmx worker modules ARA drives in the isolated apple env (never imported in-process).
DEVICE_MODULE = "wmx_suite.device"

# Model ARA calibrates against — smallest SmolLM (MLX 4-bit). Calibration only measures
# fixed memory overhead, so a tiny instruct model is plenty.
CALIBRATION_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"


def safe_limits() -> dict:
    """Read this machine's safe memory limits via the wmx worker. Pure read — no model.

    Stateless: returns the budget with no stored overhead (``calibrated=False``). ARA overlays
    a previously-measured overhead from its own store — the engine no longer reads a database.
    """
    facts = engine_env.run_worker("apple", ["-m", DEVICE_MODULE, "limits"])
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
    """Run wmx-suite's crash-safe calibration via the worker; return fresh limits + what it
    measured.

    The worker loads the model and watches memory under wmx-suite's predictive safety ramp,
    which aborts before approaching the safe budget. ARA only invokes it (out-of-process in the
    apple env). Surfaces the **effective** cold-start overhead (clamped to the engine's floor:
    ``max(default, measured)``) as ``overhead_gb`` so ARA can persist it; the raw measurement is
    in the ``"calibration"`` sub-dict for the caller to show.
    """
    result = engine_env.run_worker("apple", ["-m", DEVICE_MODULE, "calibrate", model])
    limits = safe_limits()
    overheads = [v for v in (result.get("measured_overhead_gb"),
                             result.get("default_overhead_gb")) if v is not None]
    limits["overhead_gb"] = max(overheads) if overheads else None
    limits["calibrated"] = True
    limits["calibration"] = result
    return limits


# ARA-owned ramp policy (the engine only measures; ARA decides the schedule + safety margin).
WORKER_MODULE = "wmx_suite.measure_one"
RAMP_SCHEDULE = [2000, 4000, 8000, 16000, 32000, 65536, 131072]
DEFAULT_MARGIN_GB = 2.0      # safety cushion below the wall (ARA policy)
DEFAULT_OVERHEAD_GB = 1.0    # fallback cold-start overhead until calibrated


def _budget_params() -> tuple[float, float]:
    """ARA-owned (margin, overhead). Margin is policy; overhead is this machine's stored
    calibration for the wmx engine, or a safe default if uncalibrated."""
    overhead = DEFAULT_OVERHEAD_GB
    stored = profiles.get_calibration(db.connect(), "wmx")
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
    """Measure *model*'s safe context ceiling on this Mac — the thin path.

    Pure wiring: ARA owns the methodology in the engine-agnostic ``contracts.driver`` (the
    antidote to an Apple-shaped abstraction); this adapter only supplies the Apple specifics —
    the isolated ``apple`` env, wmx's self-vetoing ``measure_one`` worker, the budget params,
    and the schedule. ARA never imports wmx in-process. Crash-safety is layered: the driver
    gates each rung (L1 ``plan_next`` + L2 actual-footprint check), the engine refuses-before-
    load (L4) and a watchdog aborts mid-probe (L5). Returns ``{model, safe_context, points}``.
    """
    margin, overhead = _budget_params()
    return driver.characterize(
        model,
        preflight=lambda m: engine_env.run_worker(
            "apple", _worker_argv(m, 0, margin, overhead, preflight=True)),
        measure=lambda m, ctx: engine_env.run_worker(
            "apple", _worker_argv(m, ctx, margin, overhead)),
        schedule=RAMP_SCHEDULE,
    )
