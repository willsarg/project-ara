# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Safe 2D (batch x seq_len) OS-wired memory-surface benchmark for encoder embeddings.

Mirrors probe.py's "extrapolate, never crash" approach for a NON-causal encoder:
  * one isolated subprocess per grid cell (probe_worker_embeddings)
  * the predictive gate runs in THIS (parent) process BEFORE spawning each cell
  * per-row ramp; skip the rest of a row once a cell is predicted to breach the wall
  * the gate trusts a fit of REAL measured high-water peaks; a conservative analytic prior
    is used only for cold start (before MIN_FIT_POINTS cells exist)
"""
from __future__ import annotations

import json
import subprocess
import sys

from . import config, db, profiles
from .system import read_limits, sample_settled_baseline

DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32]
DEFAULT_SEQS = [128, 256, 512, 1024, 2048, 4096, 8192]

MIN_FIT_POINTS = 4  # 3-param fit needs >0 DoF; 4th measured cell (1,1024) is trivially safe
PRED_SAFETY = 1.25
MODEL_BASE_SEED_GB = 1.0  # weight-residency seed so cold-start/pre-flight aren't zero
CELL_TIMEOUT_S = 300  # kill a wedged cell subprocess rather than block the sweep forever

# ModernBERT-base architecture constants (verified from config.json).
NUM_LAYERS = 22
HIDDEN_SIZE = 768
NUM_HEADS = 12

# Cold-start OVER-estimate (sum over all layers as if global): a safe upper bound that only
# ever gates the first tiny cells, since the sweep ramps seq from the smallest value.
A_COLD = NUM_LAYERS * HIDDEN_SIZE * 2 / 1e9   # GB per (batch*seq)
B_COLD = NUM_LAYERS * NUM_HEADS * 2 / 1e9     # GB per (batch*seq^2)

# Physical one-layer FLOOR (lower bound: >=1 layer's attention + residual resident at peak).
A_FLOOR = HIDDEN_SIZE * 2 / 1e9
B_FLOOR = NUM_HEADS * 2 / 1e9


def _default_event(_event: dict) -> None:
    pass


def _det3(m: list[list[float]]) -> float:
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _solve3(mat: list[list[float]], rhs: list[float]) -> tuple[float, float, float] | None:
    """Solve a 3x3 linear system by Cramer's rule. Returns None if exactly singular.
    (Relative ill-conditioning is screened by the caller, `_fit_cab`.)"""
    det = _det3(mat)
    if det == 0.0:
        return None
    out = []
    for i in range(3):
        mi = [row[:] for row in mat]
        for j in range(3):
            mi[j][i] = rhs[j]
        out.append(_det3(mi) / det)
    return (out[0], out[1], out[2])


def _fit_cab(points: list[tuple[float, float, float]]) -> tuple[float, float, float] | None:
    """Least-squares delta = c + a*x1 + b*x2 (intercept + linear + quadratic).

    points: (x1=batch*seq, x2=batch*seq^2, delta). Returns (c, a, b), or None when there
    are <3 points or the normal-equation system is singular (e.g. a degenerate grid with
    only one distinct (x1, x2)). The intercept c keeps fixed model residency out of the
    slope terms, which is what prevents through-origin extrapolation blow-up.
    """
    n = len(points)
    if n < 3:
        return None
    sx1 = sx2 = sy = sx11 = sx12 = sx22 = sx1y = sx2y = 0.0
    for x1, x2, y in points:
        sx1 += x1
        sx2 += x2
        sy += y
        sx11 += x1 * x1
        sx12 += x1 * x2
        sx22 += x2 * x2
        sx1y += x1 * y
        sx2y += x2 * y
    mat = [[float(n), sx1, sx2],
           [sx1, sx11, sx12],
           [sx2, sx12, sx22]]
    # Relative conditioning gate: the normal-matrix diagonal entries (n, Σx1², Σx2²) span
    # many orders of magnitude (x2 = batch*seq^2 ~ 1e9), so a fixed epsilon is meaningless.
    # Compare |det| to the product of the diagonals (which bounds it for this symmetric
    # PSD matrix); a near-singular system falls far below it. None → caller uses the safe
    # cold over-estimate.
    ref = mat[0][0] * mat[1][1] * mat[2][2]
    if ref <= 0.0 or abs(_det3(mat)) < 1e-12 * ref:
        return None
    rhs = [sy, sx1y, sx2y]
    return _solve3(mat, rhs)


def _coeffs(points: list[tuple[float, float, float]],
            stored: tuple[float, float, float] | None) -> tuple[float, float, float]:
    """Raw gate coefficients (c, a, b), before floors/clamps. Priority:
      1. in-run 3-param fit once we have >= MIN_FIT_POINTS points (ground truth for the
         current machine state),
      2. a stored calibration profile (seeds the early cells, and also used if the in-run
         fit is singular),
      3. the cold sum-over-layers OVER-estimate (safe fallback when we can't trust a fit).
    """
    if len(points) >= MIN_FIT_POINTS:
        fit = _fit_cab(points)
        if fit is not None:
            return fit
    if stored is not None:
        return stored
    # Cold intercept is 0.0: fixed residency is supplied separately by MODEL_BASE_SEED_GB
    # (the model_base clamp in sweep), not modeled here.
    return (0.0, A_COLD, B_COLD)


def _run_cell(py: str, model: str, batch: int, seq: int, repeats: int, margin: float) -> dict:
    cmd = [py, "-m", "wmx_suite.probe_worker_embeddings",
           "--model", model, "--batch", str(batch), "--seq", str(seq),
           "--repeats", str(repeats), "--margin", str(margin)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=CELL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"status": "error",
                "note": f"worker timed out after {CELL_TIMEOUT_S}s at batch {batch} seq {seq}"}
    line = next((l for l in out.stdout.splitlines() if l.startswith("{")), None)
    if not line:
        return {"status": "error",
                "note": f"no result (stderr: {out.stderr.strip()[-200:]})"}
    return json.loads(line)


def sweep(con, run_id: int, model: str, batches=None, seqs=None, repeats: int = 3,
          margin_gb: float | None = None, *, mlx_version: str | None = None,
          ignore_profile: bool = False, on_event=None, persist: bool = True) -> dict:
    """Run the batch x seq memory-surface sweep, gating each cell before it is spawned.

    Args:
        con: open DB connection (or None when persist is False, e.g. in tests).
        run_id: id from db.start_embeddings_run; measurements are stored under it.
        model: HF model id passed to the worker.
        batches, seqs: grid axes (default DEFAULT_BATCHES / DEFAULT_SEQS); seqs is sorted.
        repeats: forward passes per cell (passed to the worker).
        margin_gb: safety cushion; resolved via config.margin_gb.
        mlx_version: MLX version string used as part of the profile key; when None,
            no profile is loaded and no profile is upserted at the end.
        ignore_profile: force a clean recalibration — run the sweep without seeding the
            gate from any stored profile, then OVERWRITE the stored profile with this
            run's freshly measured fit (still upserts at the end if enough points were
            collected).
        on_event: optional callback invoked with one dict per event —
            {"event": "cell_done"|"row_skipped"|"error"|"preflight_abort", ...}.
        persist: when True and con is not None, each cell is written to the DB.

    Returns a summary dict: model, run_id, n_cells_measured, n_cells_skipped
    (plus "aborted"/"error" keys when the sweep stops early).
    """
    batches = batches or DEFAULT_BATCHES
    seqs = sorted(seqs or DEFAULT_SEQS)
    margin = config.margin_gb(margin_gb)
    on_event = on_event or _default_event
    py = sys.executable

    limits = read_limits()
    threshold = limits.safe_threshold_gb(margin)

    points: list[tuple[float, float, float]] = []  # (x1, x2, delta)
    model_base = MODEL_BASE_SEED_GB
    stored = None
    if not ignore_profile and mlx_version is not None and con is not None:
        stored = profiles.embedding_coeffs(con, model, mlx_version)
    smallest_unsafe_seq: float = float("inf")  # for monotonic pruning across batches
    n_measured = 0
    n_skipped = 0

    # Pre-flight: if even a tiny cell can't fit given current pressure, abort the sweep.
    live = sample_settled_baseline()
    if live + model_base >= threshold:
        on_event({"event": "preflight_abort",
                  "note": (f"host pressure {live:.2f} GB + model seed {model_base} GB "
                           f">= threshold {threshold:.2f} GB")})
        return {"model": model, "run_id": run_id, "n_cells_measured": 0,
                "n_cells_skipped": 0, "aborted": True}

    for batch in batches:
        for seq in seqs:
            # Monotonic pruning: memory grows with batch at fixed seq.
            if seq >= smallest_unsafe_seq:
                on_event({"event": "row_skipped", "batch": batch, "seq": seq,
                          "predicted_gb": None})
                n_skipped += 1
                continue

            live_base = sample_settled_baseline()
            c, a, b = _coeffs(points, stored)
            # model_base = fixed residency (fitted/stored intercept), clamped to the seed
            # and monotonic non-decreasing so a later fit can't shrink it unsafely.
            model_base = max(model_base, MODEL_BASE_SEED_GB, c)
            a = max(A_FLOOR, a)
            b = max(B_FLOOR, b)
            x1, x2 = batch * seq, batch * seq * seq
            predicted = live_base + model_base + PRED_SAFETY * (a * x1 + b * x2)
            if predicted >= threshold:
                on_event({"event": "row_skipped", "batch": batch, "seq": seq,
                          "predicted_gb": round(predicted, 3)})
                n_skipped += 1
                smallest_unsafe_seq = min(smallest_unsafe_seq, seq)
                break  # skip the rest of this (ascending-seq) row

            result = _run_cell(py, model, batch, seq, repeats, margin)
            if result.get("status") != "rung_done":
                on_event({"event": "error", "batch": batch, "seq": seq,
                          "note": result.get("note", "worker error")})
                return {"model": model, "run_id": run_id,
                        "n_cells_measured": n_measured, "n_cells_skipped": n_skipped,
                        "error": result.get("note")}

            delta = max(0.0, result["os_wired_gb"] - live_base)
            points.append((x1, x2, delta))
            n_measured += 1

            if persist and con is not None:
                db.add_embeddings_measurement(
                    con, run_id, batch_size=batch, seq_len=seq,
                    os_wired_gb=result["os_wired_gb"], peak_gb=result["peak_gb"],
                    throughput_tps=result["throughput_tps"], latency_ms=result["latency_ms"],
                )
            on_event({"event": "cell_done", "batch": batch, "seq": seq,
                      "os_wired_gb": result["os_wired_gb"], "peak_gb": result["peak_gb"],
                      "throughput_tps": result["throughput_tps"],
                      "latency_ms": result["latency_ms"]})

    if (persist and con is not None and mlx_version is not None
            and len(points) >= MIN_FIT_POINTS):
        fit = _fit_cab(points)
        if fit is not None:
            c, a, b = fit
            profiles.upsert_embedding_coeffs(con, model, mlx_version,
                                             coef_intercept_gb=c, coef_linear=a,
                                             coef_quad=b, n_points=len(points))

    return {"model": model, "run_id": run_id, "n_cells_measured": n_measured,
            "n_cells_skipped": n_skipped}
