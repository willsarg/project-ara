"""Apple-Silicon backend adapter — wraps wmx-suite's MLX measurement.

A lean device oracle, symmetric with backends/cuda.py: it reads the machine's memory wall
and runs wmx-suite's crash-safe calibration, but it owns **no persistence** — ARA stores and
reuses the calibration (see cli.render_profile). Lazy by design: ``wmx_suite`` is imported
inside each call, so nothing MLX-shaped loads until ARA actually runs the engine. ARA's only
wmx-suite imports live here, and only the engine's measurement interface (config / system /
probe / ui) — not its db, profiles, or model catalog.
"""
from __future__ import annotations


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
