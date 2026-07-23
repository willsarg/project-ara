# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Safe characterizer: find a model's context ceiling by extrapolation, never by crashing.

Strategy:
  1. Pre-flight: estimate base footprint from weights.
       * estimate >= Metal working-set limit -> HARD REFUSE (hopeless; e.g. the 27B). Never probe.
       * estimate >= threshold  -> borderline. Refuse UNLESS allow_min_probe, in which
         case run ONE supervised 512-token probe (deep in the safe zone) to replace the
         blind guess with ground truth, then decide.
       * estimate <  threshold  -> proceed with the ramp.
  2. Ramp context upward through safe rungs, one isolated subprocess each.
  3. Fit the model's DELTA over its own launch baseline: delta = os_wired - baseline.
     This isolates the model's footprint from background noise (browsers, IDEs, etc.).
  4. Before every rung, predict its ABSOLUTE peak = live_baseline + (model_base + slope*c)
     and STOP if it would cross the safe threshold.
  5. Solve for the safe ceiling and boundary projection against a reference baseline.
"""
from __future__ import annotations

import json
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from dataclasses import dataclass

from . import config, models, profiles, system, units
from .system import SystemLimits, read_limits, sample_settled_baseline

DEFAULT_RAMP = [2048, 8192, 16384, 32768, 49152, 65536, 98304, 131072]
CALIBRATE_RAMP = [512, 2048]  # two small safe rungs; the c->0 intercept gives the base
MIN_PROBE_CTX = 512  # supervised calibration probe — deep in the safe zone
DEFAULT_REPEATS = 3  # N-repeat median per rung, to smooth prefill-transient sampling jitter

# Speed presets trade fit granularity for fewer cold model loads (loads = rungs x
# repeats; each load is the dominant cost). They only change WHICH contexts are probed
# and how many times — never the pre-flight safety gate, which still measures each rung
# and refuses to launch any rung predicted to breach the threshold.
SPEED_PRESETS = {
    # quick's speedup comes from repeats=1, not fewer rungs: the pre-flight gate already
    # prunes high rungs, and a mid-dense ramp keeps the fit spanning the super-linear
    # memory bend. Cutting rungs instead would bias the ceiling optimistically (unsafe).
    "quick": {"ramp": [2048, 8192, 16384, 32768, 65536, 131072], "repeats": 1},
    "standard": {"ramp": DEFAULT_RAMP, "repeats": DEFAULT_REPEATS},
    "full": {"ramp": [2048, 4096, 8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072],
             "repeats": DEFAULT_REPEATS},
}
DEFAULT_SPEED = "standard"


def resolve_speed(speed: str, repeats: int | None = None) -> tuple[list[int], int]:
    """Resolve a speed preset to (ramp, repeats).

    The preset sets the repeats default; an explicit ``repeats`` overrides it while the
    ramp still comes from the preset. Raises ValueError on an unknown preset name.
    """
    try:
        preset = SPEED_PRESETS[speed]
    except KeyError:
        raise ValueError(f"unknown speed preset: {speed!r} "
                         f"(choose from {', '.join(SPEED_PRESETS)})")
    return list(preset["ramp"]), preset["repeats"] if repeats is None else repeats
# rough base-footprint estimate (GiB): weights resident + fixed overhead, on top of the
# live system baseline. Calibrated loosely on Gemma/Qwen; refined as more models run.
# profiles.py is the source of truth for these cold-start defaults; aliased here for
# backward references (tests / readers). Do not redefine — change them in profiles.py.
RESIDENT_FACTOR = profiles.DEFAULT_RESIDENT_FACTOR
FIXED_OVERHEAD_GB = profiles.DEFAULT_FIXED_OVERHEAD_GB


@dataclass
class Fit:
    model_base_gb: float      # delta intercept — model's own footprint at c->0 (invariant)
    slope_gb_per_k: float
    r2: float
    ref_baseline_gb: float
    threshold_gb: float
    safe_ceiling_ctx: int
    hard_wall_ctx: int
    n_points: int


def _linfit(xs_k: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares y = a + b*x. Returns (a, b, r2). x in thousands of tokens."""
    n = len(xs_k)
    mx_ = sum(xs_k) / n
    my = sum(ys) / n
    sxx = sum((x - mx_) ** 2 for x in xs_k)
    sxy = sum((x - mx_) * (y - my) for x, y in zip(xs_k, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx_
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs_k, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    return a, b, r2


def _solve_ctx(model_base: float, slope_per_k: float, ref_baseline: float,
               target_gb: float) -> int:
    """Context (tokens) where ref_baseline + model_base + slope*c == target."""
    if slope_per_k <= 0:
        return 0
    headroom = target_gb - ref_baseline - model_base
    return int(max(0.0, headroom / slope_per_k) * 1000)


def estimate_base_gb(info: models.ModelInfo, limits: SystemLimits, overhead_gb: float) -> float:
    """Pre-flight guess of ABSOLUTE base (context->0) OS-wired footprint, before any probe.

    The caller supplies the cold-start overhead (a prior calibration, or the default) — pure
    arithmetic, no db. The resident factor is held fixed (profiles.DEFAULT_RESIDENT_FACTOR).
    """
    os_baseline = max(limits.wired_now_gb, 2.5)
    return os_baseline + info.weights_gb * profiles.DEFAULT_RESIDENT_FACTOR + overhead_gb


def _terminate_process_group(proc) -> None:
    """Terminate, then kill, the worker's entire isolated process group and reap it."""
    if proc.poll() is not None:
        proc.wait()
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _telemetry_summary(baseline, samples, sample_times, interval: float) -> dict:
    """Summarize strict host-wide samples without claiming an unsampled transient peak."""
    final = samples[-1] if samples else baseline
    gaps = [(b - a) * 1000 for a, b in zip(sample_times, sample_times[1:])]
    highest = max(
        baseline.wired_bytes,
        max((sample.wired_bytes for sample in samples), default=baseline.wired_bytes),
    )
    return {
        "schema": "macos-native-vm-telemetry:v1",
        "scope": "host-wide",
        "sampling_interval_ms": round(interval * 1000, 3),
        "sample_count": len(samples),
        "max_sample_gap_ms": round(max(gaps), 3) if gaps else 0.0,
        "peak_qualification": "highest sampled; gaps unobserved",
        "baseline": asdict(baseline),
        "final": asdict(final),
        "highest_sampled_host_wired_bytes": highest,
        "compression_delta": {
            "compressor_bytes": final.compressor_bytes - baseline.compressor_bytes,
            "compressions": final.compressions - baseline.compressions,
            "decompressions": final.decompressions - baseline.decompressions,
        },
        "swap_delta": {
            "swapins": final.swapins - baseline.swapins,
            "swapouts": final.swapouts - baseline.swapouts,
            "swap_used_bytes": final.swap_used_bytes - baseline.swap_used_bytes,
        },
        "throttled_bytes_delta": final.throttled_bytes - baseline.throttled_bytes,
    }


def _supervise_process(cmd: list[str], *, context: int, abort_wired_bytes: int | None,
                       timeout: float, sampling_interval: float = 0.05,
                       snapshot_fn=system.native_vm_snapshot) -> dict:
    """Run one model worker under strict external native VM telemetry."""
    try:
        baseline = snapshot_fn()
    except Exception as exc:
        return {"context": context, "status": "aborted",
                "note": f"telemetry failed closed before launch: {exc}"}
    if abort_wired_bytes is not None and baseline.wired_bytes >= abort_wired_bytes:
        telemetry = _telemetry_summary(baseline, [], [], sampling_interval)
        return {
            "context": context,
            "status": "aborted",
            "note": ("baseline host wired memory already reached ARA boundary: "
                     f"{baseline.wired_bytes} >= {abort_wired_bytes} bytes"),
            "baseline_wired_bytes": baseline.wired_bytes,
            "highest_sampled_host_wired_bytes": baseline.wired_bytes,
            "telemetry": telemetry,
        }
    samples = []
    sample_times = []
    reason = None
    with (tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file,
          tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file):
        try:
            proc = subprocess.Popen(
                cmd, stdout=stdout_file, stderr=stderr_file, text=True,
                start_new_session=True)
        except OSError as exc:
            return {
                "context": context,
                "status": "aborted",
                "note": f"probe worker launch failed closed: {exc}",
                "baseline_wired_bytes": baseline.wired_bytes,
                "highest_sampled_host_wired_bytes": baseline.wired_bytes,
                "telemetry": _telemetry_summary(
                    baseline, [], [], sampling_interval),
            }
        started = time.monotonic()
        while True:
            if time.monotonic() - started >= timeout:
                reason = f"probe worker timed out after {timeout:.0f}s (hung load?)"
                _terminate_process_group(proc)
                break
            try:
                sample = snapshot_fn()
            except Exception as exc:
                reason = f"telemetry failed closed: {exc}"
                _terminate_process_group(proc)
                break
            samples.append(sample)
            sample_times.append(time.monotonic())
            if (abort_wired_bytes is not None
                    and sample.wired_bytes >= abort_wired_bytes):
                reason = ("sampled host wired memory reached ARA boundary: "
                          f"{sample.wired_bytes} >= {abort_wired_bytes} bytes")
                _terminate_process_group(proc)
                break
            if proc.poll() is not None:
                break
            time.sleep(sampling_interval)
        telemetry = _telemetry_summary(
            baseline, samples, sample_times, sampling_interval)
        if reason is not None or not samples:
            return {
                "context": context, "status": "aborted",
                "note": reason or "telemetry failed closed: zero successful samples",
                "baseline_wired_bytes": baseline.wired_bytes,
                "highest_sampled_host_wired_bytes": telemetry[
                    "highest_sampled_host_wired_bytes"],
                "telemetry": telemetry,
            }
        stdout_file.seek(0)
        stderr_file.seek(0)
        line = next((line for line in stdout_file if line.startswith("{")), None)
        if line is None:
            return {
                "context": context, "status": "load_error",
                "note": _summarize_worker_error(stderr_file.read()),
                "baseline_wired_bytes": baseline.wired_bytes,
                "highest_sampled_host_wired_bytes": telemetry[
                    "highest_sampled_host_wired_bytes"],
                "telemetry": telemetry,
            }
        try:
            result = json.loads(line)
        except json.JSONDecodeError as exc:
            return {"context": context, "status": "load_error",
                    "note": f"worker emitted invalid JSON: {exc}", "telemetry": telemetry}
        result.update(
            baseline_wired_bytes=baseline.wired_bytes,
            highest_sampled_host_wired_bytes=telemetry[
                "highest_sampled_host_wired_bytes"],
            baseline_wired_gb=units.bytes_to_gib(baseline.wired_bytes),
            os_wired_gb=units.bytes_to_gib(
                telemetry["highest_sampled_host_wired_bytes"]),
            telemetry=telemetry,
        )
        if "mlx_peak_bytes" in result:
            result["mlx_peak_gb"] = units.bytes_to_gib(result["mlx_peak_bytes"])
        if "mlx_active_plus_cache_bytes" in result:
            result["mlx_true_gb"] = units.bytes_to_gib(
                result["mlx_active_plus_cache_bytes"])
        return result


def _run_worker(py: str, hf_id: str, ctx: int, kv_bits, abort_wired_gb=None,
                timeout: float = 600) -> dict:
    cmd = [py, "-m", "ara_engine_mlx.probe_worker", hf_id, str(ctx)]
    if kv_bits is not None:
        cmd += ["--kv-bits", str(kv_bits)]
    abort_wired_bytes = (units.gib_to_bytes(abort_wired_gb)
                         if abort_wired_gb is not None else None)
    return _supervise_process(
        cmd, context=ctx, abort_wired_bytes=abort_wired_bytes, timeout=timeout)


def _summarize_worker_error(stderr: str) -> str:
    """Turn a worker's raw traceback into one human-readable line.

    A model-load failure can dump thousands of characters of weight-key names;
    we extract the headline instead of echoing the whole list.
    """
    err = (stderr or "").strip()
    if not err:
        return "no result (worker produced no output)"
    # Checkpoint/architecture mismatch (e.g. some gemma-4 4-bit builds carry
    # weights for kv-shared layers that this mlx_lm version doesn't define).
    m = re.search(r"(\d+)\s+parameters not in model", err)
    if m:
        return (f"load failed — this checkpoint has {m.group(1)} weight tensors "
                f"that your installed mlx_lm doesn't expect (a quant/build "
                f"mismatch). Try a different build (e.g. an -OptiQ- variant) or "
                f"upgrade mlx_lm.")
    # Otherwise show the exception summary (last non-empty traceback line), capped.
    last = next((ln.strip() for ln in reversed(err.splitlines()) if ln.strip()), "")
    return f"load failed — {last[:200]}" if last else "no result"


def _measure_rung(py: str, hf_id: str, ctx: int, kv_bits, repeats: int, *, verbose, log):
    """Run a rung `repeats` times in fresh processes; return the MEDIAN high-water.

    Each prefill can land its peak between sampler windows (±~1GiB jitter), so a single run
    makes ceilings look erratic. The median of N isolated runs is the textbook smoother.
    Returns a dict with median os_wired/delta/mlx_peak, or the first failure dict / None.
    """
    abss, deltas, peaks = [], [], []
    for _ in range(max(1, repeats)):
        m = _run_worker(py, hf_id, ctx, kv_bits)
        if m is None or m.get("status") != "ok":
            return m  # propagate error — stop the ramp
        abss.append(m["os_wired_gb"])
        deltas.append(m["os_wired_gb"] - m["baseline_wired_gb"])
        peaks.append(m["mlx_peak_gb"])
    return {
        "status": "ok", "context": ctx, "repeats": len(abss),
        "os_wired_gb": round(statistics.median(abss), 3),
        "delta": round(statistics.median(deltas), 3),
        "mlx_peak_gb": round(statistics.median(peaks), 3),
        "spread_gb": round(max(abss) - min(abss), 3),
    }


class _NullRecorder:
    """Default sink for characterize's stream — records nothing. Callers that want the run
    persisted pass a db-backed recorder (so probe stays free of the database)."""
    def upsert_model(self, info_dict): ...
    def start_run(self, hf_id, **kw): return None
    def add_measurement(self, ctx, **kw): ...
    def save_fit(self, fit_dict): ...


def characterize(hf_id: str, *, margin_gb: float | None = None, ramp=None,
                 allow_min_probe: bool = False, repeats: int = DEFAULT_REPEATS,
                 worker_python: str | None = None, verbose=True, console=None,
                 prior_overhead_gb: float | None = None, recorder=None,
                 kv_bits: int | None = None) -> dict:
    ramp = ramp or DEFAULT_RAMP
    margin_gb = config.margin_gb(margin_gb)
    if prior_overhead_gb is None:
        prior_overhead_gb = profiles.DEFAULT_FIXED_OVERHEAD_GB
    if recorder is None:
        recorder = _NullRecorder()
    limits = read_limits()
    threshold = limits.safe_threshold_gb(margin_gb)
    wall = limits.wall_gb
    sweep_baseline = limits.wired_now_gb  # reference baseline for stored ceiling numbers
    info = models.describe(hf_id)
    if info is None:
        raise SystemExit(f"model not found in HF cache: {hf_id}")
    if not info.is_causal:
        raise SystemExit(f"[characterize] REFUSED: Model {hf_id} is not a supported causal language model.")

    # fp16 by default (conservative + lossless); quant is opt-in. Forced fp16 for cache types
    # that can't quantize (RotatingKVCache crashes past the quant threshold).
    kv_bits = kv_bits if info.can_quantize_kv else None
    py = worker_python or sys.executable

    import mlx.core as mx
    recorder.upsert_model(info.as_dict())
    run_id = recorder.start_run(
        hf_id, kv_bits=kv_bits, kv_group_size=64, quantized_kv_start=5000,
        mlx_version=mx.__version__, wall_gb=wall, safe_threshold_gb=threshold,
    )

    # Presentation: render through the ui schema when a console is available
    # (CLI) or verbose with no console (default to stdout). con_out=None → quiet
    # (tests pass verbose=False). probe stays decision-only; the view formats.
    if console is None and verbose:
        from .ui import Console
        console = Console.from_args()
    con_out = console
    from .views import characterize as _view

    def log(*a):
        # Incidental low-level progress; rendered dim. Milestones use the view.
        if con_out is not None:
            con_out.emit(con_out.style("dim", "  " + " ".join(str(x) for x in a)))

    est = estimate_base_gb(info, limits, prior_overhead_gb)
    if con_out is not None:
        _view.render_header(con_out, {
            "model": hf_id, "cache_type": info.cache_type,
            "kv_mode": ("fp16" if kv_bits is None else f"{kv_bits}-bit"),
            "wall_gb": wall, "safe_budget_gb": threshold,
            "baseline_gb": sweep_baseline, "est_gb": est, "weights_gb": info.weights_gb,
        })

    xs_k: list[float] = []   # context in thousands of tokens
    ys: list[float] = []     # DELTA over launch baseline (model's own footprint)

    # ---- pre-flight gate -------------------------------------------------
    def _refuse(reason: str, kind: str):
        recorder.add_measurement(0, status="skipped", note=reason)
        if con_out is not None:
            _view.render_refusal(con_out, {
                "model": hf_id, "kind": kind, "est_gb": est,
                "threshold_gb": threshold, "wall_gb": wall})
        return {"hf_id": hf_id, "refused": True, "reason": reason,
                "est_base_gb": est, "threshold_gb": threshold, "wall_gb": wall,
                "run_id": run_id}

    if est >= wall:
        return _refuse(f"estimated base {est:.2f}GiB >= Metal working-set limit "
                       f"{wall:.2f}GiB — ARA cannot prove a safe launch. Never probe.",
                       "hopeless")
    if est >= threshold:
        if not allow_min_probe:
            return _refuse(f"estimated base {est:.2f}GiB >= safe threshold {threshold:.2f}GiB "
                           f"(but < wall {wall:.2f}GiB). Borderline — re-run with --min-probe "
                           f"to measure the true base with a supervised 512-token probe.",
                           "borderline")
        # supervised minimal probe: deep in the safe zone, replaces the blind guess
        if con_out is not None:
            _view.render_note(con_out, f"borderline — running a supervised "
                                       f"{MIN_PROBE_CTX}-token calibration probe…")
        m = _run_worker(py, hf_id, MIN_PROBE_CTX, kv_bits)
        if not m or m.get("status") != "ok":
            return _refuse(f"min-probe failed: {(m or {}).get('note', 'no output')}",
                           "borderline")
        true_abs = m["os_wired_gb"]
        true_delta = round(true_abs - m["baseline_wired_gb"], 3)
        recorder.add_measurement(MIN_PROBE_CTX, mlx_peak_gb=m["mlx_peak_gb"],
                           mlx_true_gb=m["mlx_true_gb"], os_wired_gb=true_abs,
                           note=f"min-probe; delta={true_delta}")
        if con_out is not None:
            _view.render_rung(con_out, {"ctx": MIN_PROBE_CTX, "os_wired_gb": true_abs,
                                        "delta_gb": true_delta, "peak_gb": m["mlx_peak_gb"],
                                        "repeats": 1, "spread_gb": 0.0})
        if true_abs >= threshold:
            return _refuse(f"measured base {true_abs:.2f}GiB >= threshold {threshold:.2f}GiB "
                           f"— genuinely too tight, not just a pessimistic estimate.",
                           "borderline")
        xs_k.append(MIN_PROBE_CTX / 1000)
        ys.append(true_delta)

    # ---- safe ramp -------------------------------------------------------
    fit: Fit | None = None

    def _refit():
        nonlocal fit
        if len(xs_k) >= 2:
            a, b, r2 = _linfit(xs_k, ys)
            fit = Fit(
                model_base_gb=round(a, 3), slope_gb_per_k=round(b, 5), r2=round(r2, 4),
                ref_baseline_gb=round(sweep_baseline, 3), threshold_gb=threshold,
                safe_ceiling_ctx=_solve_ctx(a, b, sweep_baseline, threshold),
                hard_wall_ctx=_solve_ctx(a, b, sweep_baseline, wall),
                n_points=len(xs_k),
            )

    _refit()  # in case the min-probe seeded a point (still need >=2 to fit)
    for ctx in ramp:
        if ctx > (info.max_context or ctx):
            break
        if xs_k and ctx <= xs_k[-1] * 1000:
            continue  # already covered by the min-probe seed
        # predict this rung against a SETTLED live baseline before launching it
        # (settle neutralizes un-reclaimed wired pages from the just-exited worker)
        if fit is not None:
            live_base = sample_settled_baseline()
            pred = live_base + fit.model_base_gb + fit.slope_gb_per_k * (ctx / 1000)
            if pred >= threshold:
                if con_out is not None:
                    _view.render_stop(con_out, {"ctx": ctx, "predicted_gb": pred,
                                                "safe_budget_gb": threshold})
                break

        m = _measure_rung(py, hf_id, ctx, kv_bits, repeats, verbose=verbose, log=log)
        if m is None:
            recorder.add_measurement(ctx, status="error", note="no json output")
            if con_out is not None and not xs_k:
                _view.render_failure(con_out, {"model": hf_id, "note": "no output from probe worker"})
            break
        if m.get("status") != "ok":
            recorder.add_measurement(ctx, status=m.get("status"), note=m.get("note"))
            # If the very first rung failed to load, surface a clean failure.
            if con_out is not None and not xs_k:
                _view.render_failure(con_out, {"model": hf_id, "note": m.get("note")})
            break  # a crash/NYI here means production can't go here either

        delta = m["delta"]
        recorder.add_measurement(ctx, mlx_peak_gb=m["mlx_peak_gb"],
                           os_wired_gb=m["os_wired_gb"],
                           note=f"median of {m['repeats']}; delta={delta}; spread={m['spread_gb']}")
        xs_k.append(ctx / 1000)
        ys.append(delta)
        if con_out is not None:
            _view.render_rung(con_out, {"ctx": ctx, "os_wired_gb": m["os_wired_gb"],
                                        "delta_gb": delta, "peak_gb": m["mlx_peak_gb"],
                                        "repeats": m["repeats"], "spread_gb": m["spread_gb"]})
        _refit()

    result = {"hf_id": hf_id, "refused": False, "run_id": run_id,
              "threshold_gb": threshold, "wall_gb": wall}
    if fit is not None:
        recorder.save_fit(fit.__dict__)
        result["fit"] = fit.__dict__
        if con_out is not None:
            _view.render_summary(con_out, {
                "model": hf_id, "safe_ctx": fit.safe_ceiling_ctx,
                "hard_wall_ctx": fit.hard_wall_ctx, "r2": fit.r2,
                "n_points": fit.n_points})
    elif con_out is not None and xs_k:
        # Measured some points but couldn't fit (need >=2).
        _view.render_failure(con_out, {
            "model": hf_id,
            "note": "couldn't measure enough safe points to fit a curve "
                    "(need at least 2). Try again, or the model may be borderline."})
    return result


class _CalibrationLoadFailed(Exception):
    """A calibration candidate failed to LOAD (vs a memory/fit abort).

    Carries a clean human reason + the original message, so the caller can either
    fall back to the next auto-picked candidate or surface guidance.
    """
    def __init__(self, reason: str, test_msg: str):
        super().__init__(reason)
        self.reason = reason
        self.test_msg = test_msg


def _calibration_candidates() -> list[str]:
    """Causal mlx-community models in the HF cache, smallest weights first."""
    out: list[tuple[float, str]] = []
    for hf_id in models.scan_cache():
        info = models.describe(hf_id)
        if info is None or not info.is_causal:
            continue
        out.append((info.weights_gb, hf_id))
    out.sort()
    return [hf_id for _, hf_id in out]


def _pick_calibration_model() -> str:
    """Smallest causal mlx-community model in the HF cache (by on-disk weight size)."""
    candidates = _calibration_candidates()
    if not candidates:
        raise SystemExit(
            "[calibrate] no causal mlx-community model found in the HF cache. "
            "Download a small one (e.g. a 0.5-1.5B mlx-community model) or pass --model."
        )
    return candidates[0]


def _calibrate_one(hf_id: str, *, margin_gb: float, repeats: int, prior_overhead_gb: float,
                   worker_python: str | None, verbose: bool, con_out) -> dict:
    """Calibrate using ONE model. Returns the result dict.

    Raises ``_CalibrationLoadFailed`` if the model can't be loaded (so an
    auto-pick caller can fall back to the next candidate). Raises ``SystemExit``
    on a memory/fit abort — a different model wouldn't fix those.
    """
    from .views import calibrate as _view
    info = models.describe(hf_id)
    if info is None:
        raise SystemExit(f"[calibrate] model not found in HF cache: {hf_id}")
    if not info.is_causal:
        raise SystemExit(f"[calibrate] {hf_id} is not a supported causal language model.")

    limits = read_limits()
    threshold = limits.safe_threshold_gb(margin_gb)
    kv_bits = None  # calibration measures fixed cold-start overhead; fp16 is the safe baseline
    py = worker_python or sys.executable

    def log(*a):
        if con_out is not None:
            con_out.emit(con_out.style("dim", "  " + " ".join(str(x) for x in a)))

    def _abort(test_msg: str, reason: str, kind: str = "fit"):
        if con_out is not None:
            _view.render_abort(con_out, {"reason": reason, "kind": kind})
            raise SystemExit(1)
        raise SystemExit(test_msg)

    import mlx.core as mx

    est = estimate_base_gb(info, limits, prior_overhead_gb)
    if est >= threshold:
        _abort(
            f"[calibrate] estimated base {est:.2f}GiB >= threshold {threshold:.2f}GiB — "
            f"machine too loaded or model too large to calibrate safely. Free memory or "
            f"pass a smaller --model.",
            f"estimated load ({est:.2f} GiB) is at/over the safe budget "
            f"({threshold:.2f} GiB) — the machine is too loaded, or this model is too "
            f"big, to calibrate safely.", kind="memory")

    factor, overhead = profiles.DEFAULT_RESIDENT_FACTOR, prior_overhead_gb
    model_base = info.weights_gb * factor + overhead
    slope = info.estimated_slope_gb_per_k(kv_bits)

    if con_out is not None:
        _view.render_header(con_out, {
            "model": hf_id, "weights_gb": info.weights_gb, "threshold_gb": threshold,
            "kv_mode": ("fp16" if kv_bits is None else f"{kv_bits}-bit")})

    xs_k: list[float] = []
    ys: list[float] = []
    for ctx in CALIBRATE_RAMP:
        if ctx > (info.max_context or ctx):
            continue
        live_base = sample_settled_baseline()
        predicted = live_base + model_base + slope * (ctx / 1000)
        if predicted >= threshold:
            _abort(
                f"[calibrate] predicted {predicted:.2f}GiB (live {live_base:.2f} + model "
                f"{model_base:.2f} + slope {slope:.4f}*{ctx/1000:.1f}k) >= threshold "
                f"{threshold:.2f}GiB before rung {ctx}; aborting (free memory and retry).",
                f"predicted {predicted:.2f} GiB at {ctx:,} tok would reach the safe budget "
                f"({threshold:.2f} GiB) before measuring — aborting before any risk.",
                kind="memory")
        m = _measure_rung(py, hf_id, ctx, kv_bits, repeats, verbose=verbose, log=log)
        if m is None or m.get("status") != "ok":
            note = (m or {}).get("note", "no output")
            raise _CalibrationLoadFailed(
                reason=f"probe at {ctx:,} tok failed: {note}",
                test_msg=f"[calibrate] rung {ctx} failed: {note}")
        xs_k.append(ctx / 1000)
        ys.append(m["delta"])
        if con_out is not None:
            _view.render_rung(con_out, {"ctx": ctx, "delta_gb": m["delta"],
                                        "repeats": m["repeats"]})

    if len(xs_k) < 2:
        _abort("[calibrate] need >=2 successful rungs to fit the base intercept.",
               "couldn't measure enough probe rungs to fit (need at least 2).")

    intercept, _slope, _r2 = _linfit(xs_k, ys)
    measured_overhead = round(intercept - profiles.DEFAULT_RESIDENT_FACTOR * info.weights_gb, 3)
    fixed_overhead = max(profiles.DEFAULT_FIXED_OVERHEAD_GB, measured_overhead)

    # Pure measurement — return the result; the CALLER persists (the calibration command to its
    # db, ARA to its own store). probe no longer writes the profile.
    return {
        "hf_id": hf_id, "machine_key": profiles.machine_key(), "intercept_gb": round(intercept, 3),
        "measured_overhead_gb": measured_overhead, "fixed_overhead_gb": fixed_overhead,
        "default_overhead_gb": profiles.DEFAULT_FIXED_OVERHEAD_GB, "n_points": len(xs_k),
        "mlx_version": mx.__version__,
    }


def calibrate(model: str | None = None, *, margin_gb: float | None = None,
              repeats: int = DEFAULT_REPEATS, worker_python: str | None = None,
              verbose: bool = True, console=None, prior_overhead_gb: float | None = None) -> dict:
    """Measure this machine's cold-start FIXED_OVERHEAD_GB and return it (the caller persists).

    *prior_overhead_gb* seeds the pre-flight base estimate (a previous calibration, or the
    default when None) — probe reads no database.

    With an explicit ``model`` we calibrate that one (and surface a clean error if
    it won't load). With no model we auto-pick the smallest cached causal model
    and, if it fails to LOAD, fall back to the next-smallest — so a single broken
    checkpoint in the cache doesn't sink MLX calibration.
    """
    margin_gb = config.margin_gb(margin_gb)
    if prior_overhead_gb is None:
        prior_overhead_gb = profiles.DEFAULT_FIXED_OVERHEAD_GB
    if console is None and verbose:
        from .ui import Console
        console = Console.from_args()
    con_out = console
    from .views import calibrate as _view

    explicit = model is not None
    if explicit:
        candidates = [model]
    else:
        candidates = _calibration_candidates()
        if not candidates:
            raise SystemExit(
                "[calibrate] no causal mlx-community model found in the HF cache. "
                "Download a small one (e.g. a 0.5-1.5B mlx-community model) or pass --model."
            )

    last: _CalibrationLoadFailed | None = None
    for hf_id in candidates:
        try:
            return _calibrate_one(hf_id, margin_gb=margin_gb, repeats=repeats,
                                  prior_overhead_gb=prior_overhead_gb,
                                  worker_python=worker_python, verbose=verbose,
                                  con_out=con_out)
        except _CalibrationLoadFailed as exc:
            last = exc
            if explicit:
                # User chose this model — don't silently substitute another.
                if con_out is not None:
                    _view.render_abort(con_out, {"reason": exc.reason, "kind": "load"})
                    raise SystemExit(1) from exc
                raise SystemExit(exc.test_msg) from exc
            # Auto-pick: this candidate didn't load — try the next-smallest.
            if con_out is not None:
                short = hf_id.split("/", 1)[-1]
                con_out.emit(con_out.style(
                    "dim", f"  {short} didn't load — trying the next cached model…"))
                con_out.emit()
            continue

    # Auto-pick exhausted every candidate.
    reason = ("none of the cached models could be loaded for calibration"
              + (f" (last: {last.reason})" if last else "")
              + ". Download a known-good small mlx-community model, or pass --model.")
    if con_out is not None:
        _view.render_abort(con_out, {"reason": reason, "kind": "load"})
        raise SystemExit(1)
    raise SystemExit(f"[calibrate] {reason}")
