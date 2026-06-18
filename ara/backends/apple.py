"""Apple-Silicon backend adapter — the ONLY module that imports wmx-suite.

Lazy by design: this module isn't imported unless detect picks ``"apple"``, and
the heavy ``wmx_suite`` import happens inside the call that needs it — so even on
a Mac, nothing MLX-shaped loads until ARA actually runs the engine.
"""
from __future__ import annotations


def machine_profile(*, recalibrate: bool = False) -> dict:
    """Measure (or read) this machine's safe memory limits via wmx-suite.

    Calibrates when this machine has no stored profile (or when *recalibrate*),
    then reads the live limits. Calibration loads a small cached model and watches
    memory — wmx-suite owns that crash-safe measurement; ARA just invokes it.

    Returns a backend-neutral dict for ARA to render. Heavy engine imports happen
    here, inside the call — not at module load.
    """
    from wmx_suite import config, db, probe, profiles, system
    from wmx_suite.ui import Console as EngineConsole

    con = db.connect()
    already = db.get_profile(con, profiles.machine_key()) is not None

    measured = False
    if recalibrate or not already:
        # streams wmx-suite's own live calibration view; may SystemExit if no
        # cached model is available to calibrate against (clean message there).
        probe.calibrate(margin_gb=None, console=EngineConsole.from_args())
        measured = True
        con = db.connect()  # re-read after the profile is stored

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
        "just_measured": measured,
    }
