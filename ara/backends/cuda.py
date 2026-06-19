"""NVIDIA / CUDA backend adapter — the ONLY module that imports wcx-suite.

Lazy by design: this module isn't imported unless detect picks ``"cuda"``, and the
``wcx_suite`` import happens inside each function — so even on an NVIDIA box nothing
torch-shaped loads until ARA actually runs the engine.
"""
from __future__ import annotations


# Tiny model ARA characterizes against — transformers format (torch can't load the
# mlx-community 4-bit build the Apple engine uses).
CALIBRATION_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"


def safe_limits() -> dict:
    """This machine's safe VRAM limits via wcx-suite. Pure read — no model load.

    The VRAM wall is read exactly from the device (nvidia-smi), so — unlike Apple's hidden
    cold-start overhead — there's nothing to calibrate for the budget itself (``calibrated``
    is True, ``overhead_gb`` None). Characterizing a *model*'s context ceiling is a separate,
    optional step.
    """
    from wcx_suite import config, system

    lim = system.read_limits()
    if lim is None:
        raise RuntimeError("no NVIDIA GPU visible to nvidia-smi")
    margin = config.margin_gb(None)
    safe = lim.safe_threshold_gb(margin)
    return {
        "device": lim.device,
        "total_gb": lim.total_gb,
        "wall_gb": lim.wall_gb,
        "safe_budget_gb": safe,
        "margin_gb": margin,
        "headroom_gb": safe - lim.used_gb,
        "swap_free_gb": None,        # VRAM has no swap
        "overhead_gb": None,         # the wall is exact — nothing to calibrate
        "calibrated": True,
        "calibrated_at": None,
    }


def calibration_model_cached(model: str = CALIBRATION_MODEL) -> bool:
    """Is the probe model already in the HF cache? (cheap, no load)."""
    from huggingface_hub import try_to_load_from_cache

    try:
        return isinstance(try_to_load_from_cache(model, "config.json"), str)
    except Exception:
        return False


def download_calibration_model(model: str = CALIBRATION_MODEL) -> None:
    """Fetch the probe model into the HF cache. Network + disk only."""
    from ara import acquire

    acquire.download(model)


def characterize(model: str) -> dict:
    """Measure *model*'s safe VRAM context ceiling via wcx-suite's isolated probe (small safe
    ramp, stops before OOM). Returns ``{model, safe_context, points}``; None ceiling if it
    couldn't fit/measure."""
    from wcx_suite import config, probe, system

    lim = system.read_limits()
    budget = lim.safe_threshold_gb(config.margin_gb(None))
    result = probe.characterize(model, budget_gb=budget)
    return {
        "model": model,
        "safe_context": result.safe_context if result else None,
        "points": result.points if result else [],
    }


def calibrate(model: str = CALIBRATION_MODEL) -> dict:
    """Characterize *model* on the GPU and attach it to the limits (for the profile flow)."""
    limits = safe_limits()
    limits["characterization"] = characterize(model)
    return limits
