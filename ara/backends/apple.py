"""Apple-Silicon backend adapter — the ONLY module that imports wmx-suite.

Lazy by design: this module isn't imported unless detect picks ``"apple"``, and
the heavy ``wmx_suite`` import happens inside the call that needs it — so even on
a Mac, nothing MLX-shaped loads until ARA actually runs the engine.
"""
from __future__ import annotations


# Model ARA calibrates against — smallest Gemma 4 (Will's pick). Calibration only
# measures memory overhead, so a small instruct model is plenty.
CALIBRATION_MODEL = "mlx-community/gemma-4-e4b-it-4bit"


def safe_limits() -> dict:
    """Read this machine's safe memory limits. Pure read — no stress, no model.

    Always returns usable numbers: a stored calibration refines the overhead,
    but uncalibrated machines still get an estimated budget. Heavy engine imports
    happen here, inside the call — not at module load.
    """
    from wmx_suite import config, db, profiles, system

    con = db.connect()
    s = system.read_limits()
    margin = config.margin_gb(None)
    safe = s.safe_threshold_gb(margin)
    prof = db.get_profile(con, profiles.machine_key())

    return {
        "device": s.device,
        "total_gb": s.total_gb,
        "wall_gb": s.wall_gb,
        "safe_budget_gb": safe,
        "margin_gb": margin,
        "headroom_gb": safe - s.wired_now_gb,
        "swap_free_gb": s.swap_free_gb,
        "overhead_gb": prof["fixed_overhead_gb"] if prof else None,
        "calibrated": prof is not None,
        "calibrated_at": prof["calibrated_at"][:10] if prof else None,
    }


def calibration_model_cached(model: str = CALIBRATION_MODEL) -> bool:
    """Is the calibration model already in the HF cache? (cheap, no load)."""
    from wmx_suite import models

    try:
        return models.describe(model) is not None
    except Exception:
        return False


def download_calibration_model(model: str = CALIBRATION_MODEL) -> None:
    """Fetch the calibration model into the HF cache. Network + disk only."""
    from huggingface_hub import snapshot_download

    snapshot_download(model)


def calibrate(model: str = CALIBRATION_MODEL) -> dict:
    """Run wmx-suite's crash-safe calibration against *model*, return fresh limits.

    Loads the model and watches memory under wmx-suite's predictive safety ramp,
    which aborts before approaching the safe budget. ARA only invokes it.
    """
    from wmx_suite import probe
    from wmx_suite.ui import Console as EngineConsole

    probe.calibrate(model, margin_gb=None, console=EngineConsole.from_args())
    return safe_limits()
