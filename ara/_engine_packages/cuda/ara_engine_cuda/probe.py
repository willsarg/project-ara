# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Safe GPU probes: spawn the worker subprocess, parse its one-line JSON, never crash.

All real GPU work lives in ``probe_worker`` (a separate process) so that an out-of-memory
or a driver fault kills the worker, not the suite. This module only orchestrates and reads
back results — it never imports torch or touches CUDA itself.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Calibration:
    device: str
    overhead_gb: float    # fixed VRAM the CUDA context costs before any model weights


def _fit_line(points: list[tuple[int, float]]) -> tuple[float, float] | None:
    """Least-squares (slope, intercept) of VRAM vs context, or None if undetermined."""
    n = len(points)
    if n < 2:
        return None
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    denom = sum((x - mx) ** 2 for x, _ in points)
    if denom == 0:        # all points at the same context
        return None
    slope = sum((x - mx) * (y - my) for x, y in points) / denom
    return slope, my - slope * mx


def fit_ceiling(points: list[tuple[int, float]], budget_gb: float) -> int | None:
    """The largest context whose extrapolated VRAM stays within *budget_gb*.

    *points* are measured (context, total_vram_gb) samples taken safely below the wall;
    KV cache grows linearly with context, so a line extrapolates the ceiling. Returns None
    when VRAM doesn't grow with context (not context-bound) or the fit is undetermined.
    """
    line = _fit_line(points)
    if line is None:
        return None
    slope, intercept = line
    if slope <= 0:
        return None
    return max(0, int((budget_gb - intercept) / slope))


def _run_worker(args: list[str], timeout: float = 120) -> dict | None:
    """Run the probe worker in a subprocess; return its parsed JSON, or None on any failure."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ara_engine_cuda.probe_worker", *args],
            capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    out = (proc.stdout or "").strip()
    if not out:
        return None
    try:
        return json.loads(out.splitlines()[-1])
    except (ValueError, IndexError):
        return None


def measure_once(hf_id: str, ctx: int, *, abort_gb: float, kv_bits: int | None = None,
                 prefer_flash: bool = False, weight_quant: str = "none",
                 chunk: int | None = None, timeout: float = 600) -> dict:
    """Probe *hf_id* at one context in the isolated GPU worker; normalise to the driver shape.

    The worker speaks ``{ok, ...}``; ARA's :mod:`ara_engine_cuda.measure_one` (and the MLX contract it
    mirrors) speaks ``{status, ...}``. This is the one adapter between them, so the worker stays
    uniform. *abort_gb* is the absolute safe budget the worker's watchdog (L5) must not let live
    VRAM reach. *kv_bits* (when set) measures with a quantized KV cache. Returns
    ``{status: "ok", used_gb, baseline_gb}`` or ``{status: "error", note}``.
    """
    # probe_worker dispatches positionally (fn(*argv[1:])), so kv_bits rides as a 4th positional;
    # --flash-attn is a named flag the worker strips (so it never shifts the kv_bits position).
    args = ["measure", hf_id, str(ctx), str(abort_gb)]
    if kv_bits is not None:
        args.append(str(kv_bits))
    if prefer_flash:
        args.append("--flash-attn")
    if weight_quant and weight_quant != "none":
        args += ["--weight-quant", weight_quant]
    if chunk is not None:
        args += ["--prefill-chunk", str(chunk)]
    data = _run_worker(args, timeout=timeout)
    if data is None:
        return {"status": "error", "note": "no output from probe worker"}
    if not data.get("ok"):
        return {"status": "error", "note": data.get("error", "probe failed")}
    return {"status": "ok", "used_gb": data["used_gb"], "baseline_gb": data["baseline_gb"]}


def calibrate(timeout: float = 120) -> Calibration | None:
    """Measure the CUDA context's fixed VRAM overhead on this machine, or None if it can't."""
    data = _run_worker(["calibrate"], timeout=timeout)
    if not data or not data.get("ok"):
        return None
    return Calibration(device=data["device"], overhead_gb=float(data["overhead_gb"]))


# A small ramp of safe contexts to probe — well below any 8 GB wall — then extrapolate.
DEFAULT_RAMP = (512, 2048)
# The tiny default probe model (transformers format — NOT the mlx-community 4-bit one).
DEFAULT_PROBE_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"


@dataclass(frozen=True)
class Characterization:
    device: str
    model: str
    points: list[tuple[int, float]]   # measured (context, total_vram_gb), safely below the wall
    safe_context: int | None          # extrapolated ceiling under the budget (None = not context-bound)


def characterize(model: str, *, budget_gb: float,
                 contexts: tuple[int, ...] = DEFAULT_RAMP,
                 kv_bits: int | None = None, prefer_flash: bool = False,
                 weight_quant: str = "none", chunk: int | None = None,
                 timeout: float = 600) -> Characterization | None:
    """Probe *model* at a few safe contexts on the GPU, then extrapolate the safe ceiling.

    *kv_bits* (when set) measures with a quantized KV cache; *prefer_flash* opts into
    FlashAttention-2; *weight_quant* loads the weights quantized. Returns None if the worker
    couldn't load/measure (e.g. the model OOMs)."""
    ramp = ",".join(str(c) for c in contexts)
    args = ["characterize", model, ramp]
    if kv_bits is not None:
        args.append(str(kv_bits))   # positional 3rd arg (worker dispatches positionally)
    if prefer_flash:
        args.append("--flash-attn")
    if weight_quant and weight_quant != "none":
        args += ["--weight-quant", weight_quant]
    if chunk is not None:
        args += ["--prefill-chunk", str(chunk)]
    data = _run_worker(args, timeout=timeout)
    if not data or not data.get("ok"):
        return None
    points = [(int(p["context"]), float(p["used_gb"])) for p in data["points"]]
    return Characterization(device=data["device"], model=model, points=points,
                            safe_context=fit_ceiling(points, budget_gb))
