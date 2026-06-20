"""CPU backend — system-RAM inference via llama.cpp (GGUF) — the second real engine.

Contract class: **ramp** (safe context ceiling). Wall source: system RAM + swap. One adapter
for *all* CPU-only inference regardless of ISA — x86 (Intel/AMD), arm64, Raspberry Pi,
riscv-when-it-matters. The ISA is metadata reported by ``detect``, not a separate backend,
because it doesn't change the wall. This is the universal fallback: nearly every machine
matches it.

It exists to keep the abstraction honest. Unlike Apple (whose engine is the external
``wmx-suite``), llama.cpp is **built into ARA** (see the engine repo-split rule: only the huge
CUDA/MLX suites get their own repos). The worker is a self-contained script —
``ara/workers/cpu_llama.py``, which never imports ``ara`` — run by the isolated ``cpu`` env's
own python over ``engine_env``, so llama-cpp-python never enters ARA core's lock. ``characterize``
is pure wiring into the engine-agnostic :func:`ara.contracts.driver.characterize`; the only
Apple↔CPU differences are the env name, the worker, and the wall source.
"""
from __future__ import annotations

from pathlib import Path

# Core, engine-free helpers (no llama.cpp) — safe at module load and patchable in tests.
from ara import db, engine_env, profiles
from ara.contracts import driver

# The built-in worker script (ships in the ARA repo, runs in the isolated cpu env by path).
WORKER = Path(__file__).resolve().parent.parent / "workers" / "cpu_llama.py"

ENV_NAME = "cpu"
CALIBRATION_ENGINE = "cpu"   # profiles key for this machine's stored overhead
RAMP_SCHEDULE = [2000, 4000, 8000, 16000, 32000, 65536, 131072]
DEFAULT_MARGIN_GB = 2.0      # safety cushion below the wall (ARA policy)
DEFAULT_OVERHEAD_GB = 1.0    # fallback cold-start overhead until calibrated


def _budget_params() -> tuple[float, float]:
    """ARA-owned (margin, overhead). Margin is policy; overhead is this machine's stored
    calibration for the cpu engine, or a safe default if uncalibrated."""
    overhead = DEFAULT_OVERHEAD_GB
    stored = profiles.get_calibration(db.connect(), CALIBRATION_ENGINE)
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


def characterize(model: str) -> dict:
    """Measure *model*'s safe context ceiling on CPU — the thin path, same driver as Apple.

    Pure wiring: ARA owns the methodology in the engine-agnostic ``contracts.driver``; this
    adapter supplies only the CPU specifics — the isolated ``cpu`` env, the built-in
    ``cpu_llama`` worker, budget params, and the schedule. Crash-safety is layered: the driver
    gates each rung (L1 + L2), and the worker refuses-before-load (L4) / aborts mid-probe (L5).
    Returns ``{model, safe_context, points}``.
    """
    margin, overhead = _budget_params()
    return driver.characterize(
        model,
        preflight=lambda m: engine_env.run_worker(
            ENV_NAME, _worker_argv(m, 0, margin, overhead, preflight=True)),
        measure=lambda m, ctx: engine_env.run_worker(
            ENV_NAME, _worker_argv(m, ctx, margin, overhead)),
        schedule=RAMP_SCHEDULE,
    )
