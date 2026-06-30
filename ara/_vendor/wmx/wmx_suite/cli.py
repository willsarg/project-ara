# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Command-line entry point for the suite.

    uv run wmx-suite system                 # show the machine's memory wall + swap
    uv run wmx-suite health                  # live snapshot: pressure + per-model go/no-go
    uv run wmx-suite characterize <hf_id>   # safe probe -> fitted context ceiling
    uv run wmx-suite list                    # characterized models + ceilings from the DB
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import pty
import re
import signal
import struct
import subprocess
import sys
import termios
import threading
from statistics import median

from mlx_lm.utils import load_tokenizer

from . import config, db, launcher, models, probe, profiles
from .system import read_limits, sample_settled_baseline
from .ui import Console
from .views import health as view_health
from .views import landing as view_landing
from .views import list as view_list
from .views import system as view_system
from .views import run_messages as view_run
from .views import calibrate as view_calibrate

# Default Console for the `run` fast path; main() replaces it per-invocation.
CONSOLE = Console.from_args()


def _configured_margin(value=None) -> float:
    try:
        margin = config.margin_gb(value)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if margin == 0:
        print("[warning] margin is 0 GB — the safe threshold equals the crash wall, with "
              "no cushion. Crossing it can hard-lock the machine.", file=sys.stderr)
    return margin


def _stream_worker(
    cmd: list[str],
    on_line,
    *,
    capture_stderr: bool = True,
) -> tuple[int, str]:
    """Spawn *cmd* and stream its stdout line-by-line via *on_line(line)*.

    Stderr is drained concurrently on a background thread so a large stderr
    payload (e.g. MLX/Metal warnings or a traceback) can never fill the OS
    pipe buffer and deadlock the parent.

    If *on_line* returns a truthy value for a given line, *_stream_worker*
    immediately terminates the child process (SIGTERM), stops consuming stdout,
    and returns.  This is used by handlers to abort on a genuine ``error``
    event so the worker cannot continue processing further rungs.

    Returns ``(returncode, stderr_text)``.  If *capture_stderr* is False the
    subprocess inherits the terminal's stderr (no deadlock risk, empty string
    returned for stderr_text).

    The caller is responsible for handling KeyboardInterrupt around this call
    if it wants to terminate the child cleanly.
    """
    stderr_pipe = subprocess.PIPE if capture_stderr else None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr_pipe, text=True)

    # Drain stderr on a background thread so it never fills (~64 KB limit).
    stderr_buf: list[str] = []
    if capture_stderr:
        def _drain_stderr():
            assert proc.stderr is not None
            for chunk in iter(lambda: proc.stderr.read(4096), ""):
                stderr_buf.append(chunk)
            proc.stderr.close()

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if on_line(line):
                # on_line signalled an abort: terminate the child so no further
                # rungs are processed or persisted, then drain remaining stdout
                # to avoid an OS pipe buffer deadlock before we close it.
                proc.terminate()
                for _ in proc.stdout:
                    pass  # drain without processing
                break
    finally:
        proc.stdout.close()

    if capture_stderr:
        stderr_thread.join()

    proc.wait()
    stderr_text = "".join(stderr_buf).strip() if capture_stderr else ""
    return proc.returncode, stderr_text


def _strip_prefix(model_id: str) -> str:
    return model_id.split("/", 1)[-1] if "/" in model_id else model_id


def cmd_system(args):
    s = read_limits()
    margin = _configured_margin()
    safe = s.safe_threshold_gb(margin)
    con = db.connect()
    prof = db.get_profile(con, profiles.machine_key())
    swap = s.swap_free_gb
    data = {
        "device": s.device,
        "total_gb": s.total_gb,
        "wall_gb": s.wall_gb,
        "safe_budget_gb": safe,
        "wired_gb": s.wired_now_gb,
        "free_headroom_gb": safe - s.wired_now_gb,
        "max_buffer_gb": s.max_buffer_gb,
        "swap_free_gb": swap,
        "swap_warn": swap is not None and swap < 2.0,
        "margin_gb": margin,
        "margin_source": ("WMX_SUITE_MARGIN_GB"
                          if os.environ.get("WMX_SUITE_MARGIN_GB") else "default"),
        "wall_bytes": int(round(s.wall_gb * 1e9)),
        "wired_sample": "median of 3 @ 0.2s",
        "calibrated": prof is not None,
        "cal_model_short": _strip_prefix(prof["model_id"]) if prof else "",
        "cal_overhead_gb": prof["fixed_overhead_gb"] if prof else 0.0,
        "cal_date": (prof["calibrated_at"][:10] if prof else ""),
    }
    view_system.render(args.console, data)


def _safe_ctx_str(pred) -> str:
    """Human label for a model's safe-context verdict (caller-side policy)."""
    if pred.breaches_wall:
        return "over budget — won't load"
    if pred.safe_ctx < launcher.MIN_USEFUL_CTX:
        return f"only ~{pred.safe_ctx:,} tok (too small)"
    return f"~{int(round(pred.safe_ctx, -2)):,} tokens"


def cmd_health(args):
    """Live 'can I run things safely right now?' snapshot: system pressure + per-model go/no-go."""
    margin = _configured_margin(args.margin)
    s = read_limits()
    threshold = s.safe_threshold_gb(margin)
    live_base = sample_settled_baseline()  # same baseline `run` uses, sampled once

    con = db.connect()
    key = profiles.machine_key()
    calibrated = db.get_profile(con, key) is not None
    rows = con.execute(
        "SELECT DISTINCT m.hf_id, m.max_context FROM models m "
        "JOIN probe_runs r ON r.hf_id = m.hf_id JOIN fits f ON f.run_id = r.id "
        "ORDER BY m.hf_id"
    ).fetchall()

    model_rows = []
    for r in rows:
        fit = db.latest_fit(con, r["hf_id"])
        if not fit or not fit.get("slope_gb_per_k"):
            continue
        pred = launcher.predict(
            model_base_gb=float(fit["model_base_gb"]),
            slope_gb_per_k=float(fit["slope_gb_per_k"]),
            live_base_gb=live_base, threshold_gb=threshold,
            wall_gb=s.wall_gb, model_max=r["max_context"],
        )
        go = (not pred.breaches_wall) and pred.safe_ctx >= launcher.MIN_USEFUL_CTX
        model_rows.append({
            "name": _strip_prefix(r["hf_id"]),
            "loads_gb": pred.base_abs_gb,
            "spare_gb": pred.headroom_gb,
            "safe_ctx": pred.safe_ctx,
            "safe_ctx_str": _safe_ctx_str(pred),
            "ok": go,
            "base_gb": float(fit["model_base_gb"]),
            "slope_gb_per_k": float(fit["slope_gb_per_k"]),
            "safe_cap_tok": pred.safe_ctx,
        })

    swap = s.swap_free_gb
    data = {
        "wall_gb": s.wall_gb,
        "safe_budget_gb": threshold,
        "free_now_gb": threshold - live_base,
        "swap_free_gb": swap,
        "swap_warn": swap is not None and swap < 2.0,
        "margin_gb": margin,
        "baseline_gb": live_base,
        "baseline_sample": "median of 3",
        "models": model_rows,
    }
    view_health.render(args.console, data)
    if not calibrated:
        dev, ram, osv = key
        args.console.emit()
        args.console.emit(args.console.style(
            "warn",
            f"No calibration profile for {dev} / {ram / 1e9:.0f}GB / macOS {osv}; "
            "cold-start estimates use fallback priors — run 'wmx-suite calibrate'."))


class _DbRecorder:
    """Persists characterize's measurement stream to the suite db (the probe→CLI seam,
    so probe.characterize itself stays db-free)."""
    def __init__(self, con):
        self.con = con
        self.run_id = None

    def upsert_model(self, info_dict):
        db.upsert_model(self.con, info_dict)

    def start_run(self, hf_id, **kw):
        self.run_id = db.start_run(self.con, hf_id, **kw)
        return self.run_id

    def add_measurement(self, ctx, **kw):
        db.add_measurement(self.con, self.run_id, ctx, **kw)

    def save_fit(self, fit_dict):
        db.save_fit(self.con, self.run_id, fit_dict)


def cmd_characterize(args):
    margin = _configured_margin(args.margin)
    ramp, repeats = probe.resolve_speed(args.speed, repeats=args.repeats)
    con = db.connect()
    probe.characterize(models.resolve_hf_id(args.hf_id), margin_gb=margin,
                       allow_min_probe=args.min_probe, repeats=repeats, ramp=ramp,
                       console=args.console, kv_bits=args.kv_bits,
                       prior_overhead_gb=profiles.cold_start_constants(con)[1],
                       recorder=_DbRecorder(con))


def cmd_calibrate(args):
    """Measure this machine's cold-start overhead and store a per-machine profile."""
    margin = _configured_margin(args.margin)
    model = models.resolve_hf_id(args.model) if args.model else args.model
    con = db.connect()
    prior = profiles.cold_start_constants(con)[1]   # prior overhead seeds the pre-flight estimate
    result = probe.calibrate(model, margin_gb=margin, console=args.console,
                             prior_overhead_gb=prior)
    # probe measures; the CLI persists the per-machine profile.
    db.upsert_profile(con, result["machine_key"],
                      resident_factor=profiles.DEFAULT_RESIDENT_FACTOR,
                      fixed_overhead_gb=result["fixed_overhead_gb"], model_id=result["hf_id"],
                      n_points=result["n_points"], mlx_version=result["mlx_version"])
    dev, ram, osv = result["machine_key"]
    view_calibrate.render_summary(args.console, {
        "machine": f"{dev} / {ram / 1e9:.0f}GB / macOS {osv}",
        "model": result["hf_id"],
        "n_points": result["n_points"],
        "measured_overhead_gb": result["measured_overhead_gb"],
        "default_overhead_gb": result["default_overhead_gb"],
        "fixed_overhead_gb": result["fixed_overhead_gb"],
        "changed": result["fixed_overhead_gb"] > result["default_overhead_gb"],
    })


def _fit_quality(r2: float) -> str:
    if r2 >= 0.99:
        return "good"
    if r2 >= 0.95:
        return "ok"
    return "poor"


def cmd_list(args):
    con = db.connect()
    rows = db.latest_fits(con)
    if not rows:
        view_run.render_no_models(args.console, {})
        return
    speeds = db.gen_speeds(con)
    model_rows = []
    for r in rows:
        s = speeds.get(r["hf_id"])
        model_rows.append({
            "hf_id": r["hf_id"],
            "loads_gb": r["model_base_gb"],
            "safe_ctx": r["safe_ceiling_ctx"],
            "speed_tps": median(s) if s else None,
            "fit": _fit_quality(r["r2"]),
            "tight": r["safe_ceiling_ctx"] < 8192,
            "stale": models.fit_is_stale(r["hf_id"], r["characterized_at"]),
            "slope_gb_per_k": r["slope_gb_per_k"],
            "r2": r["r2"],
            "hard_wall_ctx": r["hard_wall_ctx"],
            "n_runs": len(s) if s else 0,
        })
    view_list.render(args.console, {"models": model_rows})
    stale = [m["hf_id"] for m in model_rows if m["stale"]]
    if stale:
        args.console.emit()
        for hf_id in stale:
            args.console.emit(args.console.style(
                "warn", f"⚠ {_strip_prefix(hf_id)}: fit may be stale — "
                        "re-run 'wmx-suite characterize'."))



RUN_HELP = """usage: wmx-suite run [--margin GB] [--force] [--dry-run] [--co-run-kokoro] -- <mlx_lm.generate args>

Safely launch mlx_lm.generate. Picks kv-bits by cache type, caps --max-kv-size from the
measured ceiling, and refuses if the run would breach the wall.
The passthrough args must include --model <hf_id>.

  --margin GB         safety cushion under the wall (default: WMX_SUITE_MARGIN_GB or 2.0)
  --force             launch even if the planner refuses (may crash the machine)
  --dry-run           print the plan, do not launch
  --no-log            do not record generation speed (bare exec passthrough)
  --co-run-kokoro     subtract Kokoro static active overhead from the safe ceiling
"""

_PROMPT_RE = re.compile(r"Prompt:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec")
_GEN_RE = re.compile(r"Generation:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec")
_PEAK_RE = re.compile(r"Peak memory:\s*([\d.]+)\s*GB")


def _record_generation(text: str, model_id: str, max_kv_size: int) -> None:
    """Parse mlx_lm's tok/s lines from captured output and store them. Best-effort —
    a cancelled or errored run has no generation stats, so nothing is logged."""
    gm = _GEN_RE.search(text)
    if not gm:
        return
    pm, km = _PROMPT_RE.search(text), _PEAK_RE.search(text)
    try:
        con = db.connect()
        db.log_generation(
            con, model_id,
            prompt_tokens=int(pm.group(1)) if pm else None,
            prompt_tps=float(pm.group(2)) if pm else None,
            gen_tokens=int(gm.group(1)),
            gen_tps=float(gm.group(2)),
            peak_gb=float(km.group(1)) if km else None,
            max_kv_size=max_kv_size,
        )
        print(f"[run] logged {gm.group(1)} tok @ {float(gm.group(2)):.1f} tok/s", file=sys.stderr)
    except Exception as e:  # logging must never break a run
        print(f"[run] (speed log failed: {e})", file=sys.stderr)


def _exec_logged(argv: list[str], model_id: str, max_kv_size: int) -> None:
    """Run mlx_lm.generate under a PTY so output streams live to the terminal unchanged,
    while we capture a copy to parse its tok/s stats. A PTY (not a plain pipe) is required:
    Python block-buffers stdout when it isn't a tty, which would batch the token stream."""
    master, slave = pty.openpty()
    if sys.stdout.isatty():  # match the child's window size so tqdm renders correctly
        try:
            cols, rows = os.get_terminal_size(sys.stdout.fileno())
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        except Exception:
            pass
    proc = subprocess.Popen(["mlx_lm.generate"] + argv,
                            stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    captured = bytearray()
    try:
        while True:
            try:
                data = os.read(master, 4096)
            except OSError:  # EIO once the child exits and closes its end
                break
            if not data:
                break
            os.write(1, data)  # tee live to our stdout, byte-for-byte
            captured.extend(data)
    except KeyboardInterrupt:
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass
    finally:
        os.close(master)
    rc = proc.wait()
    _record_generation(captured.decode(errors="replace"), model_id, max_kv_size)
    sys.exit(rc)


def cmd_run_raw(run_args: list[str]):
    """Parse leading suite flags, then treat the remainder as mlx_lm.generate passthrough.

    Done manually (not argparse) because argparse.REMAINDER mishandles optionals that
    precede the positional, and we want `run --model X ...` to work as a drop-in.
    """
    if run_args and run_args[0] in ("-h", "--help"):
        print(RUN_HELP); return
    margin, force, dry, log = None, False, False, True
    co_run_kokoro = False
    rest: list[str] = []
    i = 0
    while i < len(run_args):
        a = run_args[i]
        # Everything after a literal `--` is passthrough verbatim (escape hatch).
        if a == "--":
            rest.extend(run_args[i + 1:]); break
        # Suite-only flags are recognized ANYWHERE in the arg list (not just
        # before the model), so `run --model X ... --dry-run` works. These names
        # are not mlx_lm.generate flags, so extracting them is unambiguous.
        if a == "--margin":
            if i + 1 >= len(run_args):
                raise SystemExit("[run] --margin requires a value")
            margin = run_args[i + 1]; i += 2; continue
        if a.startswith("--margin="):
            margin = a.split("=", 1)[1]; i += 1; continue
        if a == "--force":
            force = True; i += 1; continue
        if a == "--dry-run":
            dry = True; i += 1; continue
        if a == "--no-log":
            log = False; i += 1; continue
        if a == "--co-run-kokoro":
            co_run_kokoro = True; i += 1; continue
        rest.append(a); i += 1  # passthrough to mlx_lm.generate
    _run(rest, margin=margin, force=force, dry_run=dry, log=log, co_run_kokoro=co_run_kokoro)


def _run(rest: list[str], *, margin: float | str | None, force: bool,
         dry_run: bool, log: bool = True, co_run_kokoro: bool = False):
    """Crash-safe launch: plan a launch, then exec mlx_lm.generate."""
    model_id = None
    model_idx = None  # index in rest whose value we rewrite after resolution
    for i, a in enumerate(rest):
        if a == "--model" and i + 1 < len(rest):
            model_id = rest[i + 1]; model_idx = i + 1
            break
        if a.startswith("--model="):
            model_id = a.split("=", 1)[1]; model_idx = i
            break
    if model_id is None:
        raise SystemExit("[run] --model is required")
    # Accept short names (e.g. gemma-4-e4b-it-4bit); rewrite the passthrough so
    # mlx_lm.generate receives the full org/name id too.
    resolved = models.resolve_hf_id(model_id)
    if resolved != model_id:
        rest[model_idx] = (f"--model={resolved}" if rest[model_idx].startswith("--model=")
                           else resolved)
        model_id = resolved

    margin_gb = _configured_margin(margin)
    if co_run_kokoro:
        kokoro_overhead = 0.85  # default fallback
        try:
            con = db.connect()
            latest_base = db.get_latest_kokoro_baseline(con)
            if latest_base and latest_base.get("overhead_gb") is not None:
                kokoro_overhead = float(latest_base["overhead_gb"])
        except Exception:
            pass
        margin_gb += kokoro_overhead

    # KV defaults to fp16; --kv-bits {8,4} opts into quant. Plan for the chosen precision so a
    # q4 fit backs a q4 run (build_argv does the authoritative validation below).
    try:
        requested_kv_bits = launcher._single_int_option(rest, "--kv-bits")
    except launcher.LaunchArgumentError:
        requested_kv_bits = None
    p = launcher.plan(model_id, margin_gb=margin_gb, kv_bits=requested_kv_bits)
    if p.get("error"):
        raise SystemExit(f"[run] {p['error']}")

    if p["source"] == "estimated" and not force:
        if sys.stdin.isatty():
            print(f"[run] Model {model_id} has not been characterized yet.", file=sys.stderr)
            try:
                ans = input("[run] Characterize it now to find its safe context ceiling? [y/N]: ").strip().lower()
            except KeyboardInterrupt:
                print(file=sys.stderr)
                raise SystemExit("[run] Aborted.")
            if ans in ("y", "yes"):
                print(f"[run] Running characterization for {model_id}...", file=sys.stderr)
                try:
                    _con = db.connect()
                    probe.characterize(
                        model_id, margin_gb=margin_gb, allow_min_probe=True,
                        console=Console(color=CONSOLE.color, verbose=CONSOLE.verbose,
                                        stream=sys.stderr),
                        prior_overhead_gb=profiles.cold_start_constants(_con)[1],
                        recorder=_DbRecorder(_con))
                    # Re-plan with the newly saved fit
                    p = launcher.plan(model_id, margin_gb=margin_gb, kv_bits=requested_kv_bits)
                    if p.get("error"):
                        raise SystemExit(f"[run] Re-planning failed: {p['error']}")
                except Exception as e:
                    raise SystemExit(f"[run] Characterization failed: {e}")
            else:
                raise SystemExit("[run] Aborted. Run characterize first or pass --force to run with estimated limits.")
        else:
            raise SystemExit("[run] REFUSED: Model is uncharacterized and shell is non-interactive. Run characterize first or pass --force.")

    # Run diagnostics render to stderr (keeps the model's stdout clean), with the
    # same color/verbose policy as the rest of the CLI. Built at call time so
    # captured streams (tests) and the real stderr both work.
    console = Console(color=CONSOLE.color, verbose=CONSOLE.verbose, stream=sys.stderr)
    kv_mode = (f"{p['kv_bits']}-bit" if p["kv_bits"] is not None
               else "fp16 (not quantizable)" if not p.get("can_quantize", True)
               else "fp16")
    if p.get("fit_stale"):
        console.emit(console.style(
            "warn", "fit may be stale — consider re-running 'wmx-suite characterize'."))
    if p["source"] == "estimated" and p.get("cold_start_profile") == "default":
        console.emit(console.style(
            "warn", "using fallback cold-start constants (Apple M4 Pro testbed); "
                    "run 'wmx-suite calibrate' to tune them for this machine."))

    if p.get("refuse"):
        view_run.render_refusal(console, {
            "model": model_id,
            "needs_gb": p["base_abs_gb"],
            "budget_gb": p["threshold_gb"],
            "wall_gb": p["wall_gb"],
            "live_base_gb": p["live_base_gb"],
            "model_base_gb": p["model_base_gb"],
            "slope_gb_per_k": p["slope_gb_per_k"],
            "safe_cap_tok": p.get("max_kv_size", 0) or 0,
            "source": p["source"],
            "cache_type": p["cache_type"],
            "kv_mode": kv_mode,
        })
        if not force:
            sys.exit(2)
        console.emit(console.style(
            "warn", "--force given; proceeding against safety advice."))

    try:
        argv = launcher.build_argv(rest, p, force=force)
    except launcher.LaunchArgumentError as exc:
        raise SystemExit(f"[run] REFUSED: {exc}") from exc
    effective_cap = launcher.effective_max_kv_size(rest, p)
    prompt_check = None
    try:
        tokenizer = load_tokenizer(
            model_id,
            tokenizer_config_extra=launcher.tokenizer_config(rest),
        )
        prompt_check = launcher.check_prompt(rest, p, tokenizer)
    except launcher.LaunchArgumentError as exc:
        if not force:
            raise SystemExit(f"[run] REFUSED: {exc}") from exc
        print(f"[run] WARNING: prompt preflight bypassed: {exc}", file=sys.stderr)
    except Exception as exc:
        if not force:
            raise SystemExit(
                f"[run] REFUSED: prompt tokenization failed: {exc}; "
                "pass --force to bypass prompt preflight"
            ) from exc
        print(f"[run] WARNING: prompt tokenization failed and was bypassed: {exc}",
              file=sys.stderr)
    if not p.get("max_kv_size_enforced", True):
        print("[run] WARNING: this model's custom MLX cache does not enforce "
              "--max-kv-size", file=sys.stderr)
    if prompt_check is not None:
        print(f"[run] prompt {prompt_check.tokens:,} tokens "
              f"({prompt_check.tokens / prompt_check.cap:.0%} of cap)",
              file=sys.stderr)
        if prompt_check.tokens > prompt_check.cap:
            message = (f"prompt is {prompt_check.tokens:,} tokens, above "
                       f"the {prompt_check.cap:,}-token cap")
            if not force:
                raise SystemExit(f"[run] REFUSED: {message}")
            print(f"[run] WARNING: {message}; --force is overriding this refusal",
                  file=sys.stderr)
        if prompt_check.warn:
            print(f"[run] WARNING: prompt exceeds "
                  f"{launcher.PROMPT_WARNING_FRACTION:.0%} of the context cap",
                  file=sys.stderr)
    view_run.render_plan(console, {
        "model": model_id,
        "source": p["source"],
        "cache_type": p["cache_type"],
        "kv_mode": kv_mode,
        "live_base_gb": p["live_base_gb"],
        "model_base_gb": p["model_base_gb"],
        "budget_gb": p["threshold_gb"],
        "wall_gb": p["wall_gb"],
        "slope_gb_per_k": p["slope_gb_per_k"],
        "max_kv_size": effective_cap,
        "model_max": p["model_max"],
        "max_kv_size_enforced": p.get("max_kv_size_enforced", True),
    })
    print(f"[run] exec: mlx_lm.generate {' '.join(argv)}\n", file=sys.stderr)
    if dry_run:
        print("[run] --dry-run: not launching.", file=sys.stderr)
        return
    if log:
        _exec_logged(argv, model_id, effective_cap)
    else:
        os.execvp("mlx_lm.generate", ["mlx_lm.generate"] + argv)


def _build_global_parser() -> argparse.ArgumentParser:
    """Parent parser providing the global --verbose/--no-color flags.

    Passed as ``parents=[...]`` to every subcommand so the flags are accepted
    after any command (e.g. ``wmx-suite system --verbose``).
    """
    g = argparse.ArgumentParser(add_help=False)
    g.add_argument("--verbose", "-v", action="store_true",
                   help="show the power-user appendix (raw numbers) on each command")
    g.add_argument("--no-color", action="store_true",
                   help="never emit ANSI color, even on a TTY")
    return g


def _build_parser() -> argparse.ArgumentParser:
    gp = _build_global_parser()
    ap = argparse.ArgumentParser(prog="wmx-suite")
    sub = ap.add_subparsers(dest="cmd", required=False, parser_class=lambda **kw:
                            argparse.ArgumentParser(parents=[gp], **kw))
    sub.add_parser("system").set_defaults(func=cmd_system)
    p = sub.add_parser("health")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_health)
    p = sub.add_parser("characterize"); p.add_argument("hf_id")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.add_argument("--min-probe", action="store_true",
                   help="for borderline models, run a supervised 512-token probe to measure "
                        "the true base instead of refusing on the pessimistic estimate")
    p.add_argument("--speed", choices=list(probe.SPEED_PRESETS),
                   default=probe.DEFAULT_SPEED,
                   help="how thorough the ramp is: 'quick' (fewer rungs, faster), "
                        "'standard' (default), or 'full' (denser ramp, finer fit)")
    p.add_argument("--repeats", type=int, default=None,
                   help="isolated runs per context rung; the median high-water is used "
                        "(smooths prefill-transient jitter). Overrides the --speed preset.")
    p.add_argument("--kv-bits", type=int, default=None,
                   help="KV-cache quantization bits (8 or 4) to characterize at; omit for fp16 "
                        "(default). Ignored for non-quantizable (sliding-window) models.")
    p.set_defaults(func=cmd_characterize)
    sub.add_parser("list").set_defaults(func=cmd_list)
    p = sub.add_parser("calibrate", help="measure this machine's cold-start overhead constant")
    p.add_argument("--model", default=None,
                   help="model to calibrate with (default: smallest cached causal model)")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_calibrate)
    # `run` is intercepted before argparse (see below) so it can pass arbitrary flags
    # through to mlx_lm.generate; this stub only makes it show up in `--help`.
    sub.add_parser("run", help="safely launch mlx_lm.generate: picks kv-bits by cache "
                               "type, caps --max-kv-size from the measured ceiling, "
                               "refuses if it would breach the wall")
    
    p = sub.add_parser("benchmark-kokoro", help="Benchmark Kokoro TTS performance")
    p.add_argument("--model", default="mlx-community/Kokoro-82M-bf16",
                   help="HuggingFace model ID or path")
    p.add_argument("--voice", default="af_heart",
                   help="Voice name to use")
    p.add_argument("--lengths", default="10,50,100,200,500,1000,2000,3000",
                   help="Comma-separated character lengths to sweep")
    p.add_argument("--repeats", type=int, default=3,
                   help="Number of trials per length")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_benchmark_kokoro)

    p = sub.add_parser("benchmark-kokoro-ttfa", help="Benchmark Kokoro TTS TTFA latency")
    p.add_argument("--model", default="mlx-community/Kokoro-82M-bf16",
                   help="HuggingFace model ID or path")
    p.add_argument("--voice", default="af_heart",
                   help="Voice name to use")
    p.add_argument("--lengths", default="10,50,100,200,500,1000,2000,3000",
                   help="Comma-separated character lengths to sweep")
    p.add_argument("--repeats", type=int, default=3,
                   help="Number of trials per length")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_benchmark_kokoro_ttfa)

    p = sub.add_parser("benchmark-kokoro-batch", help="Benchmark Kokoro TTS batch performance (concurrency vs throughput)")
    p.add_argument("--model", default="mlx-community/Kokoro-82M-bf16",
                   help="HuggingFace model ID or path")
    p.add_argument("--voice", default="af_heart",
                   help="Voice name to use")
    p.add_argument("--batch-sizes", default="1,2,4,8,16",
                   help="Comma-separated batch sizes to sweep")
    p.add_argument("--repeats", type=int, default=3,
                   help="Number of trials per batch size")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_benchmark_kokoro_batch)

    p = sub.add_parser("benchmark-kokoro-voice", help="Benchmark Kokoro TTS voice switching performance")
    p.add_argument("--model", default="mlx-community/Kokoro-82M-bf16",
                   help="HuggingFace model ID or path")
    p.add_argument("--voice-a", default="af_heart",
                   help="Voice name A")
    p.add_argument("--voice-b", default="am_adam",
                   help="Voice name B")
    p.add_argument("--repeats", type=int, default=5,
                   help="Number of trials")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_benchmark_kokoro_voice)

    p = sub.add_parser("benchmark-kokoro-cache", help="Benchmark Kokoro TTS voice cache memory overhead")
    p.add_argument("--model", default="mlx-community/Kokoro-82M-bf16",
                   help="HuggingFace model ID or path")
    p.add_argument("--cache-sizes", default="0,1,2,4,8,16,24,32",
                   help="Comma-separated cache sizes to sweep")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_benchmark_kokoro_cache)

    p = sub.add_parser("benchmark-kokoro-baseline", help="Benchmark Kokoro TTS static active RAM overhead")
    p.add_argument("--model", default="mlx-community/Kokoro-82M-bf16",
                   help="HuggingFace model ID or path")
    p.add_argument("--voice", default="af_heart",
                   help="Voice name")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_benchmark_kokoro_baseline)

    p = sub.add_parser("benchmark-embeddings",
                       help="Benchmark ModernBERT embeddings batch x seq memory scaling")
    p.add_argument("--model", default="mlx-community/nomicai-modernbert-embed-base-bf16",
                   help="HuggingFace model ID or path")
    p.add_argument("--batches", default="1,2,4,8,16,32",
                   help="Comma-separated batch sizes to sweep")
    p.add_argument("--seqs", default="128,256,512,1024,2048,4096,8192",
                   help="Comma-separated sequence lengths to sweep")
    p.add_argument("--repeats", type=int, default=3,
                   help="Forward passes per cell (median timing, max memory)")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.add_argument("--ignore-profile", action="store_true",
                   help="ignore any stored calibration profile (cold start); still re-fits")
    p.set_defaults(func=cmd_benchmark_embeddings)
    return ap


def _main_argparse(argv=None):
    ap = _build_parser()
    args = ap.parse_args(argv)
    args.console = Console.from_args(
        no_color=getattr(args, "no_color", False),
        verbose=getattr(args, "verbose", False),
    )
    args.func(args)


def _strip_global_flags(argv: list[str]) -> tuple[list[str], bool, bool]:
    """Pull --verbose/-v/--no-color out of *argv* (for the `run` fast path).

    `run` is intercepted before argparse so it can forward arbitrary flags to
    mlx_lm.generate; the global UX flags must not leak into that passthrough.
    Returns ``(remaining_argv, verbose, no_color)``.
    """
    remaining: list[str] = []
    verbose = no_color = False
    for a in argv:
        if a in ("--verbose", "-v"):
            verbose = True
        elif a == "--no-color":
            no_color = True
        else:
            remaining.append(a)
    return remaining, verbose, no_color


def cmd_landing(console: Console) -> None:
    """Front door: shown for `wmx-suite` with no subcommand.

    Gathers a live machine status line and renders the grouped command
    overview. ``console`` carries the --verbose/--no-color policy. Degrades
    gracefully if the machine can't be read (shows the screen with zeros).
    """
    try:
        s = read_limits()
        margin = _configured_margin()
        safe = s.safe_threshold_gb(margin)
        con = db.connect()
        n_ready = con.execute(
            "SELECT COUNT(DISTINCT m.hf_id) FROM models m "
            "JOIN probe_runs r ON r.hf_id = m.hf_id JOIN fits f ON f.run_id = r.id"
        ).fetchone()[0]
        calibrated = db.get_profile(con, profiles.machine_key()) is not None
        data = {
            "device": s.device,
            "free_gb": safe - s.wired_now_gb,
            "safe_budget_gb": safe,
            "models_ready": n_ready,
            "calibrated": calibrated,
        }
    except Exception:
        data = {"device": "unknown", "free_gb": 0.0, "safe_budget_gb": 0.0,
                "models_ready": 0, "calibrated": False}
    view_landing.render(console, data)


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "run":
        run_args, verbose, no_color = _strip_global_flags(argv[1:])
        # Run diagnostics go to stderr (model output owns stdout); color policy
        # follows stderr's TTY. _run reads CONSOLE's color/verbose policy.
        global CONSOLE
        CONSOLE = Console.from_args(stream=sys.stderr, no_color=no_color, verbose=verbose)
        return cmd_run_raw(run_args)
    # Front door: no subcommand present (empty, or only global/help flags).
    rest, verbose, no_color = _strip_global_flags(argv)
    if not rest or rest[0] in ("-h", "--help"):
        console = Console.from_args(no_color=no_color, verbose=verbose)
        return cmd_landing(console)
    return _main_argparse(argv)


# Benchmark handlers live in cli_benchmarks.py; re-export them so `cli.cmd_benchmark_*`
# and the parser registration keep resolving. Imported at the bottom to avoid a cycle
# (cli_benchmarks reaches back into this module for shared helpers).
from .cli_benchmarks import (  # noqa: E402
    cmd_benchmark_embeddings,
    cmd_benchmark_kokoro,
    cmd_benchmark_kokoro_ttfa,
    cmd_benchmark_kokoro_batch,
    cmd_benchmark_kokoro_voice,
    cmd_benchmark_kokoro_cache,
    cmd_benchmark_kokoro_baseline,
)


if __name__ == "__main__":
    main()
