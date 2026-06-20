"""Apple-Silicon backend adapter — wraps wmx-suite's MLX measurement.

A lean device oracle, symmetric with backends/cuda.py: it reads the machine's memory wall
and runs wmx-suite's crash-safe calibration, but it owns **no persistence** — ARA stores and
reuses the calibration (see cli.render_profile). Lazy by design: ``wmx_suite`` is imported
inside each call, so nothing MLX-shaped loads until ARA actually runs the engine. ARA's only
wmx-suite imports live here, and only the engine's measurement interface (config / system /
probe / ui) — not its db, profiles, or model catalog.
"""
from __future__ import annotations

# Core, engine-free helpers (no wmx) — safe to import at module load and patchable in tests.
from ara import db, engine_env, profiles
from ara.contracts import ramp, worker


# Model ARA calibrates against — smallest SmolLM (MLX 4-bit). Calibration only measures
# fixed memory overhead, so a tiny instruct model is plenty.
CALIBRATION_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"


def safe_limits() -> dict:
    """Read this machine's safe memory limits via wmx-suite. Pure read — no stress, no model.

    Stateless: returns the budget with no stored overhead (``calibrated=False``). ARA overlays
    a previously-measured overhead from its own store — the engine no longer reads a database.
    """
    from wmx_suite import config, system

    s = system.read_limits()
    margin = config.margin_gb(None)
    safe = s.safe_threshold_gb(margin)
    return {
        "device": s.device,
        "total_gb": s.total_gb,
        "wall_gb": s.wall_gb,
        "safe_budget_gb": safe,
        "margin_gb": margin,
        "headroom_gb": safe - s.wired_now_gb,
        "swap_free_gb": s.swap_free_gb,
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
    """Run wmx-suite's crash-safe calibration; return fresh limits + what it measured.

    Loads the model and watches memory under wmx-suite's predictive safety ramp, which aborts
    before approaching the safe budget. ARA only invokes it. Surfaces the **effective**
    cold-start overhead (clamped to the engine's floor: ``max(default, measured)``) as
    ``overhead_gb`` so ARA can persist it; the raw measurement is in the ``"calibration"``
    sub-dict for the caller to show.
    """
    from wmx_suite import probe
    from wmx_suite.ui import Console as EngineConsole

    result = probe.calibrate(model, margin_gb=None, console=EngineConsole.from_args())
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

    ARA owns the methodology: it asks the engine for a no-load estimate (base/slope/budget),
    then drives the ramp (schedule → fit → ceiling, ``ara.contracts.ramp``) by spawning the
    engine's self-vetoing ``measure_one`` worker in the isolated apple env via ``engine_env``.
    ARA never imports wmx in-process. Crash-safety is checked at every layer: ARA gates each
    rung (L1 ``plan_next`` + L2 re-assert), the engine refuses-before-load (L4) and a watchdog
    aborts mid-probe (L5). Returns ``{model, safe_context, points}``.
    """
    margin, overhead = _budget_params()
    est = engine_env.run_worker(
        "apple", _worker_argv(model, 0, margin, overhead, preflight=True))
    if "error" in est:
        return {"model": model, "safe_context": None, "points": []}

    def measure_fn(ctx: int):
        raw = engine_env.run_worker("apple", _worker_argv(model, ctx, margin, overhead))
        m = worker.parse(raw)
        # L2 (independent of L1's prediction): if the ACTUAL measurement reached the budget,
        # stop escalating and don't trust higher contexts — even though L1 predicted it safe.
        if not m.refused and m.mem_gb is not None and m.mem_gb >= est["budget_gb"]:
            return worker.Measurement(context=ctx, mem_gb=None, refused=True,
                                      reason="ARA L2: measured at/over safe budget")
        return m

    schedule = [c for c in RAMP_SCHEDULE
                if est["max_context"] is None or c <= est["max_context"]]
    res = ramp.run(measure_fn, schedule, est["base_gb"],
                   est["slope_gb_per_k"], est["budget_gb"])
    return {"model": model, "safe_context": res.safe_context,
            "points": [{"context": c, "mem_gb": m} for c, m in res.points]}
