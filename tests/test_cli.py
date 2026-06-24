# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""cli.py — formatters, arg parsing/dispatch, and the render_* surfaces."""
from __future__ import annotations

import json
import sys
import types

import pytest

import ara.cli as cli
from ara.detect import Accelerator, Machine, ModelStore, Runtime
from ara.hardware import (BoardInfo, CpuInfo, Drive, MemoryInfo, MemoryModule, StorageInfo)


def _raise_input(exc):
    """A fake builtins.input that raises — for EOF/Ctrl-C at the prompt."""
    def _f(prompt=""):
        raise exc
    return _f


@pytest.fixture
def stub_pythons(monkeypatch):
    """Stub pythons.count()/discover() so render_detect never touches the real
    filesystem. Call with the count and the interpreter list you want surfaced."""
    def _stub(count=1, discover=()):
        monkeypatch.setattr(cli.pythons, "count", lambda: count)
        monkeypatch.setattr(cli.pythons, "discover", lambda probe=True: list(discover))
    return _stub


# --------------------------------------------------------------------------- #
# formatters
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("v,dec,out", [
    (None, 0, "unknown"),
    (12.0, 0, "12 GB"),
    (12.34, 1, "12.3 GB"),
])
def test_fmt_gb(v, dec, out):
    assert cli._fmt_gb(v, dec) == out


@pytest.mark.parametrize("gb,out", [
    (None, "size unknown"),
    (0.5, "~500 MB"),
    (1.0, "~1.0 GB"),   # boundary: gb == 1 takes the GB branch (gb < 1 is exclusive)
    (2.5, "~2.5 GB"),
])
def test_fmt_size(gb, out):
    assert cli._fmt_size(gb) == out


@pytest.mark.parametrize("secs,out", [
    (5, "5s"), (59, "59s"),
    (60, "1m"),       # boundary: 60s rolls over to minutes (s < 60 is exclusive)
    (90, "1m"), (3600, "1h"),
    (86400, "1d"),    # boundary: 86400s rolls over to days (s < 86400 is exclusive)
    (90000, "1d"),
])
def test_fmt_uptime(secs, out):
    assert cli._fmt_uptime(secs) == out


@pytest.mark.parametrize("gb,out", [
    (0.5, "512 MB"),   # binary MB under a gigabyte
    (1.0, "1.0 GB"),   # boundary: gb == 1 takes the GB branch (gb < 1 is exclusive)
    (2.0, "2.0 GB"),
])
def test_fmt_mem(gb, out):
    assert cli._fmt_mem(gb) == out


# --------------------------------------------------------------------------- #
# main(): arg parsing + dispatch
# --------------------------------------------------------------------------- #
def _capture_dispatch(monkeypatch):
    """Replace the render_* entry points with recorders; return the record dict."""
    rec = {}
    monkeypatch.setattr(cli, "render_landing", lambda c: rec.update(landing=True))
    monkeypatch.setattr(cli, "render_detect", lambda c, as_json=False, want=None: rec.update(detect=as_json, detect_want=want))
    monkeypatch.setattr(cli, "render_status", lambda c, as_json=False, want=None: rec.update(status=as_json))
    monkeypatch.setattr(cli, "render_python", lambda c, as_json=False, want=None: rec.update(python=as_json))
    monkeypatch.setattr(cli, "render_apps", lambda c, as_json=False, want=None: rec.update(apps=as_json))
    monkeypatch.setattr(cli, "render_mlx", lambda c, as_json=False, want=None: rec.update(mlx=as_json))
    monkeypatch.setattr(cli, "render_models", lambda c, as_json=False, want=None: rec.update(models=as_json))
    monkeypatch.setattr(cli, "render_characterize",
                        lambda c, m, engine=None, as_json=False:
                        (rec.update(characterize=m, characterize_engine=engine) or 0))
    monkeypatch.setattr(cli, "render_profile",
                        lambda c, **kw: (rec.update(profile=kw) or 0))
    monkeypatch.setattr(cli, "render_recommend",
                        lambda c, as_json=False: (rec.update(recommend=as_json) or 0))
    monkeypatch.setattr(cli, "render_run",
                        lambda c, model, **kw: (rec.update(run={"model": model, **kw}) or 0))
    monkeypatch.setattr(cli, "render_install", lambda c, **kw: (rec.update(install=kw) or 0))
    monkeypatch.setattr(cli, "render_uninstall", lambda c, **kw: (rec.update(uninstall=kw) or 0))
    return rec


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["ara", *argv])
    return cli.main()


def test_main_no_args_shows_landing(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, []) == 0
    assert rec == {"landing": True}


def test_main_help_shows_landing(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["-h"]) == 0
    assert rec.get("landing") is True


def test_main_detect(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["detect"])
    assert rec["detect"] is False


def test_main_detect_json(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["detect", "--json"])
    assert rec["detect"] is True


def test_main_status(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["status"])
    assert rec["status"] is False


def test_main_profile_model_separate_value(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["profile", "--model", "org/repo"])
    assert rec["profile"]["model"] == "org/repo"


def test_main_profile_model_flag_as_last_arg(monkeypatch):
    # boundary: `--model` with no following value must yield None, not IndexError.
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["profile", "--model"]) == 0
    assert rec["profile"]["model"] is None


def test_main_profile_model_equals_form(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["profile", "--model=org/repo"])
    assert rec["profile"]["model"] == "org/repo"


def test_main_profile_passes_json(monkeypatch):
    # profile is engine-free analytic now — it takes --json/--model/--engine, no calibrate flags.
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["profile", "--json"])
    assert rec["profile"]["as_json"] is True


def test_main_unknown_command_returns_1(monkeypatch, capsys):
    _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["frobnicate"]) == 1
    assert "isn't built yet" in capsys.readouterr().out


def test_main_verbose_flag_sets_console(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "render_detect",
                        lambda c, as_json=False, want=None: captured.update(verbose=c.verbose))
    _run_main(monkeypatch, ["detect", "--verbose"])
    assert captured["verbose"] is True


# --------------------------------------------------------------------------- #
# render_landing
# --------------------------------------------------------------------------- #
def test_render_landing_supported(make_console, monkeypatch):
    monkeypatch.setattr(cli.detect, "chip_name", lambda: "Apple M4 Pro")
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "wmx-suite"))
    c, buf = make_console()
    cli.render_landing(c)
    out = buf.getvalue()
    assert "ara" in out and "Apple M4 Pro" in out
    assert "GETTING STARTED" in out
    assert "detect" in out and "status" in out and "profile" in out
    assert "mlx" in out                       # MLX view shown on Apple
    assert "CPU fallback" not in out


def test_render_landing_cpu_fallback_notes_no_gpu(make_console, monkeypatch):
    monkeypatch.setattr(cli.detect, "chip_name", lambda: "Intel i7")
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cpu")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (False, "llama.cpp"))
    c, buf = make_console()
    cli.render_landing(c)
    out = buf.getvalue()
    assert "no GPU backend detected" in out and "ara install --engine cpu" in out
    assert "mlx" not in out                    # MLX view is Apple-only


def test_cmd_long_name_keeps_gap_before_gloss(make_console):
    """A command label longer than the alignment column must not collide with its gloss."""
    c, _ = make_console()  # color off → plain text
    row = cli._cmd(c, "characterize <model>", "measure a model's safe context ceiling here")
    assert "<model>measure" not in row     # the bug: label runs straight into the gloss
    assert "<model>  measure" in row       # at least a two-space gap


def test_cmd_short_name_stays_column_aligned(make_console):
    """Short labels still align to the fixed command column (no regression)."""
    c, _ = make_console()
    row = cli._cmd(c, "detect", "inspect this machine")
    assert "detect" + " " * 10 + "inspect" in row   # 6-char name padded to the 16-col gutter


# --------------------------------------------------------------------------- #
# render_detect
# --------------------------------------------------------------------------- #
def _machine(**over) -> Machine:
    base = dict(
        system="Darwin", os_version="macOS 15.0", chip="Apple M4 Pro", arch="arm64",
        cpu_physical=12, cpu_logical=12, cpu_features=["NEON", "BF16"],
        python_version="3.12.8", ram_total_gb=48.0, ram_available_gb=20.0, swap_gb=2.0,
        accel=Accelerator("apple", "Apple M4 Pro GPU", None, "Metal", cores=16),
        disk_free_gb=500.0,
        runtimes=[Runtime("MLX", True, "0.18", kind="engine", accels=("apple",), usable=True),
                  Runtime("vLLM", True, "0.5", kind="engine", accels=("nvidia",), usable=False),
                  Runtime("PyTorch", True, "2.1", kind="framework")],
        framework_python="/usr/bin/python3",
        model_stores=[ModelStore("HF cache", True, 3, 12.0),
                      ModelStore("Ollama", True, 0, 0.0)],
        hf_token=True, power="AC power", backend="apple", engine="wmx-suite",
        engine_ready=False,
    )
    base.update(over)
    return Machine(**base)


def test_render_detect_text(make_console, monkeypatch, stub_pythons):
    stub_pythons(count=3)
    monkeypatch.setattr(cli.detect, "machine", lambda: _machine())
    c, buf = make_console()
    cli.render_detect(c)
    out = buf.getvalue()
    for section in ("SYSTEM", "MEMORY", "ACCELERATOR", "STORAGE", "ENGINES",
                    "FRAMEWORKS", "MODELS", "ARA"):
        assert section in out
    assert "Apple M4 Pro" in out
    assert "Metal" in out
    assert "MLX" in out                  # engine
    assert "PyTorch" in out              # framework on the default python
    assert "needs CUDA" in out          # vLLM unusable reason rendered
    assert "/usr/bin/python3" in out    # default interpreter shown under FRAMEWORKS
    assert "interpreters on this machine" in out  # count > 1 → pointer to `ara python`
    assert "3 models" in out


def test_render_detect_nvidia_accel(make_console, monkeypatch, stub_pythons):
    stub_pythons(count=1)
    m = _machine(accel=Accelerator("nvidia", "RTX 4090", 24.0, "CUDA", count=1,
                                   compute="8.9", cuda_version="12.4", driver_version="550.00"))
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console()
    cli.render_detect(c)
    out = buf.getvalue()
    assert "RTX 4090" in out and "24 GB VRAM" in out and "SM 8.9" in out
    assert "CUDA 12.4" in out and "driver 550.00" in out
    assert "interpreters on this machine" not in out  # count == 1 → no pointer line
    assert "(x" not in out  # single GPU → no "(xN)" multiplicity suffix


def test_render_detect_multi_gpu_shows_count(make_console, monkeypatch, stub_pythons):
    stub_pythons(count=1)
    m = _machine(accel=Accelerator("nvidia", "RTX 4090", 24.0, "CUDA", count=2,
                                   compute="8.9", cuda_version="550"))
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console()
    cli.render_detect(c)
    out = buf.getvalue()
    assert "(x2)" in out  # count > 1 → multiplicity shown


def test_render_detect_cpu_without_features_and_no_python_version(
        make_console, monkeypatch, stub_pythons):
    # cpu present but no SIMD features (skip the features append), and no python_version
    # at all (skip the whole python row).
    stub_pythons(count=1)
    m = _machine(cpu_features=[], python_version=None)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=False)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "12 cores" in out                  # cpu row emitted, no features tail
    assert "your default python3" not in out  # python row skipped entirely
    assert "ARA's python" not in out


def test_render_detect_json(monkeypatch, capsys):
    monkeypatch.setattr(cli.detect, "machine", lambda: _machine())
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_detect(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["chip"] == "Apple M4 Pro"
    assert payload["accel"]["api"] == "Metal"
    assert payload["backend"] == "apple"


# --------------------------------------------------------------------------- #
# render_status
# --------------------------------------------------------------------------- #
def _proc(**over):
    from ara.status import Proc
    base = dict(pid=1234, label="Ollama", detail="llama3", rss_gb=2.0,
                uptime_s=120.0, gpu_mb=None, port=11434)
    base.update(over)
    return Proc(**base)


def _app(**over):
    from ara.status import AppProc
    base = dict(label="Claude", n_procs=8, rss_gb=1.2, uptime_s=300.0)
    base.update(over)
    return AppProc(**base)


def test_render_status_with_processes(make_console, monkeypatch):
    monkeypatch.setattr(cli.status, "scan", lambda: [_proc(), _proc(pid=5, label="vLLM", rss_gb=4.0)])
    monkeypatch.setattr(cli.status, "scan_apps", lambda: [])
    c, buf = make_console()
    cli.render_status(c)
    out = buf.getvalue()
    assert "RUNNING AI/ML" in out
    assert "Ollama" in out and "vLLM" in out
    assert "pid 1234" in out and ":11434" in out
    assert "total" in out and "2 processes" in out


def test_render_status_empty(make_console, monkeypatch):
    monkeypatch.setattr(cli.status, "scan", lambda: [])
    monkeypatch.setattr(cli.status, "scan_apps", lambda: [])
    c, buf = make_console()
    cli.render_status(c)
    assert "nothing running right now" in buf.getvalue()


def test_render_status_shows_ai_apps(make_console, monkeypatch):
    monkeypatch.setattr(cli.status, "scan", lambda: [])
    monkeypatch.setattr(cli.status, "scan_apps",
                        lambda: [_app(), _app(label="Claude Code", n_procs=1, rss_gb=0.2)])
    c, buf = make_console()
    cli.render_status(c)
    out = buf.getvalue()
    assert "AI APPS" in out
    assert "Claude" in out and "Claude Code" in out
    assert "8 procs" in out and "1 proc" in out


def test_render_status_ai_apps_empty(make_console, monkeypatch):
    monkeypatch.setattr(cli.status, "scan", lambda: [_proc()])
    monkeypatch.setattr(cli.status, "scan_apps", lambda: [])
    c, buf = make_console()
    cli.render_status(c)
    assert "no AI apps running" in buf.getvalue()


def test_render_status_json(monkeypatch, capsys):
    monkeypatch.setattr(cli.status, "scan", lambda: [_proc()])
    monkeypatch.setattr(cli.status, "scan_apps", lambda: [_app()])
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_status(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["workloads"][0]["label"] == "Ollama" and payload["workloads"][0]["port"] == 11434
    assert payload["apps"][0]["label"] == "Claude" and payload["apps"][0]["n_procs"] == 8


# --------------------------------------------------------------------------- #
# render_profile — every branch, via a fake backend
# --------------------------------------------------------------------------- #
def _limits(calibrated=False, **over):
    base = dict(
        device="Apple M4 Pro", total_gb=48.0, wall_gb=40.0, safe_budget_gb=36.0,
        margin_gb=4.0, headroom_gb=28.0, overhead_gb=6.0, swap_free_gb=2.0,
        calibrated=calibrated, calibrated_at="2026-06-18" if calibrated else None,
        basis="estimated",
    )
    base.update(over)
    return base


class FakeBackend:
    CALIBRATION_MODEL = "org/calib"

    def __init__(self, limits, cached=True):
        self._limits = limits
        self._cached = cached
        self.safe_limits_exc = None
        self.calibrate_exc = None
        self.calibrate_result = _limits(calibrated=True,
                                        calibration={"measured_overhead_gb": 5.0,
                                                     "default_overhead_gb": 6.0, "n_points": 4})
        self.downloaded = []

    def safe_limits(self):
        if self.safe_limits_exc:
            raise self.safe_limits_exc
        return dict(self._limits)

    def calibration_model_cached(self, model):
        return self._cached

    def download_calibration_model(self, model):
        self.downloaded.append(model)

    def calibrate(self, model=None):   # real backends default model=CALIBRATION_MODEL
        if self.calibrate_exc:
            raise self.calibrate_exc
        return self.calibrate_result


def _wire_profile(monkeypatch, set_platform, machine=None):
    """Wire render_profile engine-free (Spec 2026-06-23-capability-pipeline, Slice 2 Task 2):
    a stubbed Machine + machine_key on Apple. profile makes NO engine call — there is
    deliberately no backend wired here."""
    set_platform("Darwin", "arm64")  # resolve_engine(None) -> apple/wmx
    monkeypatch.setattr(cli.detect, "machine", lambda: machine if machine is not None else _machine())
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")


def test_profile_never_loads_an_engine(make_console, monkeypatch, set_platform, store):
    # The defining property: profile is analytic. If it reached for a backend, this would blow up.
    _wire_profile(monkeypatch, set_platform)
    monkeypatch.setattr(cli, "get_backend",
                        lambda *a, **k: pytest.fail("profile loaded an engine"))
    c, buf = make_console()
    assert cli.render_profile(c) == 0
    out = buf.getvalue()
    assert "SAFE LIMITS" in out and "estimated" in out
    assert "ara characterize" in out                              # points at the empirical step
    assert cli.db.get_latest_profile(store, "mkey") is not None   # profile is the persister


def test_profile_estimated_budget_mirrors_wall(make_console, monkeypatch, set_platform):
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    c, buf = make_console()
    assert cli.render_profile(c) == 0
    out = buf.getvalue()
    assert "36.0 GB" in out          # Apple working set: 0.75 × 48
    assert "34.0 GB" in out          # safe budget: wall − 2 GB margin


def test_profile_json_emits_estimate(monkeypatch, set_platform, capsys):
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_profile(c, as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["basis"] == "estimated"
    assert payload["calibrated"] is False
    assert payload["safe_budget_gb"] == 48.0 * 0.75 - 2.0


def test_profile_reports_measured_wall_after_calibration(make_console, monkeypatch, set_platform, store):
    # Spec 2026-06-23-capability-pipeline: once a measured wall is stored for the detected engine,
    # profile reports the MEASURED numbers (labelled), not the heuristic.
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    cli.calibration.save_calibration(store, "wmx", fixed_overhead_gb=1.7,
                                     wall_gb=41.3, safe_budget_gb=39.3)
    c, buf = make_console()
    assert cli.render_profile(c) == 0
    out = buf.getvalue()
    assert "41.3 GB" in out and "39.3 GB" in out      # the measured wall + budget
    assert "not calibrated" not in out                # not an estimate anymore


def test_profile_json_reports_measured_basis(monkeypatch, set_platform, capsys, store):
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    cli.calibration.save_calibration(store, "wmx", fixed_overhead_gb=1.7,
                                     wall_gb=41.3, safe_budget_gb=39.3)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_profile(c, as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["basis"] == "measured"
    assert payload["calibrated"] is True
    assert payload["wall_gb"] == 41.3 and payload["safe_budget_gb"] == 39.3


def test_profile_uncalibrated_stays_estimated(monkeypatch, set_platform, capsys, store):
    # No stored wall → profile must STILL say estimated (no fabrication).
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_profile(c, as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["basis"] == "estimated" and payload["calibrated"] is False


def test_profile_model_fits_full_window(make_console, monkeypatch, set_platform):
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.catalog, "describe",
                        lambda m: dict(n_layers=32, kv_heads=8, head_dim=128, max_context=8192))
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: 4.0)
    c, buf = make_console()
    assert cli.render_profile(c, model="org/small") == 0
    out = buf.getvalue()
    assert "MODEL FIT: org/small" in out
    assert "full 8192 ctx" in out


def test_profile_model_context_limited(make_console, monkeypatch, set_platform):
    # A tiny budget binds before the model's window → context-limited verdict.
    _wire_profile(monkeypatch, set_platform, _machine(backend="cpu", ram_total_gb=8.0))
    monkeypatch.setattr(cli.catalog, "describe",
                        lambda m: dict(n_layers=32, kv_heads=8, head_dim=128, max_context=131072))
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: 4.0)
    c, buf = make_console()
    assert cli.render_profile(c, model="org/big-ctx") == 0
    assert "context-limited" in buf.getvalue()


def test_profile_model_wont_fit(make_console, monkeypatch, set_platform):
    _wire_profile(monkeypatch, set_platform, _machine(backend="cpu", ram_total_gb=8.0))
    monkeypatch.setattr(cli.catalog, "describe",
                        lambda m: dict(n_layers=32, kv_heads=8, head_dim=128, max_context=8192))
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: 20.0)   # weights > budget
    c, buf = make_console()
    assert cli.render_profile(c, model="org/huge") == 0
    assert "won't fit" in buf.getvalue()


def test_profile_model_fits_unknown_architecture(make_console, monkeypatch, set_platform):
    # Describable + fits, but missing dims → no slope → "fits" with an honest unknown-context note.
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.catalog, "describe",
                        lambda m: dict(n_layers=None, kv_heads=None, head_dim=None, max_context=8192))
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: 4.0)
    c, buf = make_console()
    assert cli.render_profile(c, model="org/odd") == 0
    assert "context estimate unavailable" in buf.getvalue()


def test_profile_model_undescribable(make_console, monkeypatch, set_platform):
    _wire_profile(monkeypatch, set_platform)
    monkeypatch.setattr(cli.catalog, "describe", lambda m: None)
    c, buf = make_console()
    assert cli.render_profile(c, model="org/mystery") == 0
    assert "couldn't describe org/mystery" in buf.getvalue()


def test_profile_model_uses_cataloged_weight_no_network(make_console, monkeypatch, set_platform):
    # profile --model and recommend must compute identically: a cataloged model's weight comes
    # from the local catalog (catalog.get → weights_gb), never a network repo_size_gb call.
    # Spec 2026-06-23-capability-pipeline.
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.catalog, "describe",
                        lambda m: dict(n_layers=32, kv_heads=8, head_dim=128, max_context=8192))
    monkeypatch.setattr(cli.catalog, "get",
                        lambda con, m: {"model_id": m, "weights_gb": 4.0})
    monkeypatch.setattr(cli.acquire, "repo_size_gb",
                        lambda m: pytest.fail("repo_size_gb hit the network for a cataloged model"))
    c, buf = make_console()
    assert cli.render_profile(c, model="org/small") == 0
    assert "full 8192 ctx" in buf.getvalue()        # the 4.0 GB cataloged weight drove the fit


def test_profile_model_falls_back_to_network_when_uncataloged(make_console, monkeypatch, set_platform):
    # No catalog weight (model not cataloged, or weights_gb None) → fall back to repo_size_gb.
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.catalog, "describe",
                        lambda m: dict(n_layers=32, kv_heads=8, head_dim=128, max_context=8192))
    monkeypatch.setattr(cli.catalog, "get", lambda con, m: None)
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    called = []
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: called.append(m) or 4.0)
    c, buf = make_console()
    assert cli.render_profile(c, model="org/fresh") == 0
    assert called == ["org/fresh"]                   # network fallback fired
    assert "full 8192 ctx" in buf.getvalue()


def test_profile_model_json_includes_fit(monkeypatch, set_platform, capsys):
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.catalog, "describe",
                        lambda m: dict(n_layers=32, kv_heads=8, head_dim=128, max_context=8192))
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: 4.0)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_profile(c, as_json=True, model="org/small") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_fit"]["fits"] is True
    assert payload["model_fit"]["max_context"] == 8192


def test_main_profile_passes_engine(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["profile", "--engine", "wmx"])
    assert rec["profile"]["engine"] == "wmx"


def test_profile_unknown_engine_errors(make_console, monkeypatch):
    c, buf = make_console()
    assert cli.render_profile(c, engine="bogus") == 1
    assert "unknown engine" in buf.getvalue().lower()


def test_emit_characterized_shows_stored_models(make_console, store, monkeypatch):
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    cli.db.save_characterization(store, "mkey", "wcx", "org/SmolLM", safe_context=16000, points=[],
                                 decode_context=None)
    cli.db.save_characterization(store, "mkey", "wcx", "org/Unbound", safe_context=None, points=[],
                                 decode_context=None)
    c, buf = make_console()
    cli._emit_characterized(c, "wcx")
    out = buf.getvalue()
    assert "CHARACTERIZED" in out and "SmolLM" in out and "16000" in out
    assert "—" in out               # the None-ceiling model


def test_emit_characterized_empty_shows_nothing(make_console, store, monkeypatch):
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    c, buf = make_console()
    cli._emit_characterized(c, "wcx")
    assert buf.getvalue() == ""


def test_emit_characterized_none_engine_key(make_console):
    c, buf = make_console()
    cli._emit_characterized(c, None)
    assert buf.getvalue() == ""


# --------------------------------------------------------------------------- #
# _emit_limits helper
# --------------------------------------------------------------------------- #
def test_emit_limits_omits_overhead_when_none(make_console):
    c, buf = make_console()
    cli._emit_limits(c, _limits(calibrated=False, overhead_gb=None))
    out = buf.getvalue()
    assert "SAFE LIMITS" in out and "estimated" in out
    assert "overhead" not in out


def test_emit_limits_omits_swap_when_none(make_console):
    c, buf = make_console()
    cli._emit_limits(c, _limits(calibrated=True, swap_free_gb=None))
    out = buf.getvalue()
    assert "SAFE LIMITS" in out
    assert "swap" not in out


def test_emit_limits_measured_reads_as_measured(make_console):
    # A measured profile must NOT read as estimated. Spec 2026-06-23-capability-pipeline.
    c, buf = make_console()
    cli._emit_limits(c, _limits(calibrated=True, basis="measured"))
    out = buf.getvalue()
    assert "measured" in out
    assert "estimated" not in out and "not calibrated" not in out


def test_emit_limits_estimated_says_estimated(make_console):
    c, buf = make_console()
    cli._emit_limits(c, _limits(calibrated=False, basis="estimated"))
    assert "not calibrated" in buf.getvalue()


# --------------------------------------------------------------------------- #
# render_detect verbose + CPU-fallback branches
# --------------------------------------------------------------------------- #
def test_render_detect_verbose_and_cpu_fallback(monkeypatch, make_console, stub_pythons):
    # `accelerated` is a property — drive it via backend="cpu" (no GPU-class adapter).
    stub_pythons(count=1, discover=[])
    m = _machine(
        backend="cpu", engine="llama.cpp", engine_ready=False,
        cpu_logical=24, hf_token=False,
        accel=Accelerator("none", "none detected", None, None),
        runtimes=[Runtime("Ollama", False, None, kind="engine"),
                  Runtime("PyTorch", False, None, kind="framework")],
        model_stores=[ModelStore("HF cache", False)],
    )
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "physical" in out and "logical" in out      # verbose cpu line
    assert "CPU fallback — no GPU backend detected" in out   # _det_ara backend hint
    assert "install: ara install" in out               # engine install hint (not gated)
    # verbose lists absent engine and absent store as "not found"
    assert out.count("not found") >= 2


def test_render_detect_minimal_non_verbose(make_console, monkeypatch, stub_pythons):
    # Drive every "skip the optional line" branch: missing cpu/features, no available
    # RAM, no swap, an nvidia GPU with none of its detail bits, empty/absent stores,
    # and no AI frameworks anywhere (discover → []).
    stub_pythons(count=1, discover=[])
    m = _machine(
        cpu_physical=None, cpu_logical=None, cpu_features=[],
        ram_available_gb=None, swap_gb=0.0,
        accel=Accelerator("nvidia", "Mystery GPU", None, "CUDA", count=1,
                          compute=None, cuda_version=None),
        runtimes=[Runtime("MLX", False, None, kind="engine", accels=("apple",), usable=False),
                  Runtime("PyTorch", False, None, kind="framework")],
        framework_python=None,
        model_stores=[ModelStore("HF cache", False)],
    )
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=False)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "Mystery GPU" in out and "VRAM" not in out    # nvidia, but no detail bits
    assert "none detected" in out                         # no engines present, non-verbose
    assert "has no AI frameworks" in out                  # default python is bare
    assert "ARA's env (no separate user python)" in out   # framework_python is None
    assert "None found in any interpreter" in out         # discover surfaced nothing
    assert "HF cache" not in out                           # absent store hidden (non-verbose)


def test_render_detect_frameworks_surfaced_from_other_interpreter(
        make_console, monkeypatch, stub_pythons):
    # Default python is bare, but another interpreter has the AI stack → surface it.
    from ara.pythons import Interpreter
    other = Interpreter(
        path="/opt/homebrew/bin/python3.12", real="/opt/homebrew/Cellar/python3.12",
        origin="Homebrew", version="3.12.4", is_default=False,
        ai_libs={"torch": "2.1.0", "transformers": "4.40.0"},
    )
    stub_pythons(count=2, discover=[other])
    m = _machine(runtimes=[Runtime("PyTorch", False, None, kind="framework")],
                 framework_python="/usr/bin/python3")
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console()
    cli.render_detect(c)
    out = buf.getvalue()
    assert "has no AI frameworks" in out
    assert "But you've got them in" in out
    assert "Homebrew 3.12.4" in out
    assert "torch 2.1.0" in out and "transformers 4.40.0" in out
    assert "ara python" in out


def test_render_status_gpu_and_no_port(make_console, monkeypatch):
    monkeypatch.setattr(cli.status, "scan",
                        lambda: [_proc(gpu_mb=8192.0, port=None, detail=None)])
    c, buf = make_console()
    cli.render_status(c)
    out = buf.getvalue()
    assert "8192 MB GPU" in out
    assert ":" not in out.split("Ollama")[-1].split("\n")[0].replace("pid", "")


# --------------------------------------------------------------------------- #
# _confirm — the interactive y/N prompt
# --------------------------------------------------------------------------- #
def test_confirm_accepts_yes(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "  Yes ")
    assert cli._confirm("proceed?") is True


def test_confirm_rejects_other(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "nope")
    assert cli._confirm("proceed?") is False


def test_confirm_false_on_eof(monkeypatch):
    monkeypatch.setattr("builtins.input", _raise_input(EOFError()))
    assert cli._confirm("proceed?") is False


def test_confirm_false_on_keyboard_interrupt(monkeypatch):
    monkeypatch.setattr("builtins.input", _raise_input(KeyboardInterrupt()))
    assert cli._confirm("proceed?") is False


# --------------------------------------------------------------------------- #
# python (render_python) + dispatch
# --------------------------------------------------------------------------- #
def _interp(**over):
    from ara.pythons import Interpreter
    base = dict(path="/usr/bin/python3", real="/usr/bin/python3", origin="macOS system",
                version="3.9.6", is_default=False, externally_managed=False, ai_libs={})
    base.update(over)
    return Interpreter(**base)


def test_main_python_dispatch(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    monkeypatch.setattr(cli, "render_python", lambda c, as_json=False, want=None: rec.update(python=as_json))
    _run_main(monkeypatch, ["python", "--json"])
    assert rec["python"] is True


def test_render_python_text(make_console, monkeypatch):
    ints = [
        _interp(path="/opt/homebrew/bin/python3.12", real="/opt/homebrew/bin/python3.12",
                origin="Homebrew", version="3.12.4", is_default=True,
                ai_libs={"torch": "2.1.0", "transformers": None}),
        _interp(path="/usr/bin/python3", origin="macOS system", version="3.9.6"),
    ]
    monkeypatch.setattr(cli.pythons, "discover", lambda probe=True: ints)
    c, buf = make_console()
    cli.render_python(c)
    out = buf.getvalue()
    assert "PYTHON INTERPRETERS" in out
    assert "Homebrew" in out and "macOS system" in out      # origin group headers
    assert "torch 2.1.0" in out                              # present lib (None one hidden)
    assert "transformers" not in out                         # version None → not shown
    assert "no AI libraries" in out                          # the bare interpreter
    assert "2 interpreters · 1 with AI libraries" in out     # summary
    assert "managed" in out                                  # macOS system carries a caution
    assert "your default python3" in out                     # legend


def test_render_python_help_text_is_windows_aware(make_console, monkeypatch):
    monkeypatch.setattr(cli.os, "name", "nt")
    # os.name='nt' breaks pathlib on posix; _tilde only needs Path.home() → stub it.
    monkeypatch.setattr(cli, "Path", types.SimpleNamespace(home=lambda: r"C:\Users\dev"))
    ints = [_interp(path=r"C:\Python312\python.exe", real=r"C:\Python312\python.exe",
                    origin="python.org", version="3.12.5")]
    monkeypatch.setattr(cli.pythons, "discover", lambda probe=True: ints)
    c, buf = make_console()
    cli.render_python(c)
    out = buf.getvalue()
    assert "pyenv-win" in out and "the Store" in out   # Windows homes, not macOS
    assert "Homebrew" not in out


def test_render_python_shows_symlink_real_path(make_console, monkeypatch):
    ints = [_interp(path="/usr/local/bin/python3", real="/opt/homebrew/Cellar/python@3.12/3.12.4/bin/python3.12",
                    origin="Homebrew", version="3.12.4")]
    monkeypatch.setattr(cli.pythons, "discover", lambda probe=True: ints)
    c, buf = make_console()
    cli.render_python(c)
    out = buf.getvalue()
    assert "→" in out and "python@3.12" in out  # symlink target surfaced


def test_render_python_json(monkeypatch, capsys):
    ints = [_interp(ai_libs={"torch": "2.1.0"})]
    monkeypatch.setattr(cli.pythons, "discover", lambda probe=True: ints)
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_python(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["origin"] == "macOS system"
    assert payload[0]["ai_libs"] == {"torch": "2.1.0"}


def test_render_python_groups_consecutive_same_origin(make_console, monkeypatch):
    # two Homebrew interpreters in a row → the origin header prints once, not twice.
    ints = [
        _interp(path="/opt/homebrew/bin/python3.12", origin="Homebrew", version="3.12.4"),
        _interp(path="/opt/homebrew/bin/python3.11", origin="Homebrew", version="3.11.9"),
    ]
    monkeypatch.setattr(cli.pythons, "discover", lambda probe=True: ints)
    c, buf = make_console()
    cli.render_python(c)
    out = buf.getvalue()
    assert out.count("  Homebrew\n") == 1   # header emitted once for the group
    assert "3.12.4" in out and "3.11.9" in out


# =========================================================================== #
# section filtering (--include / --exclude)
# =========================================================================== #
def test_csv_splits_and_trims():
    assert cli._csv("a, b ,,c") == ["a", "b", "c"]
    assert cli._csv("") == []


def test_section_filter_include_is_whitelist():
    pred = cli._section_filter(["system"], [])
    assert pred("system") is True and pred("memory") is False


def test_section_filter_exclude_is_blacklist():
    pred = cli._section_filter([], ["memory"])
    assert pred("memory") is False and pred("system") is True


def test_section_filter_neither_allows_all():
    pred = cli._section_filter([], [])
    assert pred("anything") is True


def test_resolve_want_aliases(make_console):
    c, _ = make_console()
    pred = cli._resolve_want("detect", ["gpu"], [], c)   # gpu → accelerator
    assert pred("accelerator") is True and pred("system") is False


def test_resolve_want_unknown_section_warns(make_console):
    c, buf = make_console()
    cli._resolve_want("detect", ["bogus"], [], c)
    out = buf.getvalue()
    assert "unknown section" in out and "valid:" in out


def test_resolve_want_command_without_sections_warns(make_console):
    c, buf = make_console()
    assert cli._resolve_want("profile", ["x"], [], c) is None
    assert "don't apply to 'profile'" in buf.getvalue()


# =========================================================================== #
# render_apps
# =========================================================================== #
def _capp(**over):
    from ara.apps import App
    base = dict(label="X", category="runner", in_app=False, cask=False, formula=False,
                version=None, brew_recorded=None, cask_token=None, installed_at=1.0)
    base.update(over)
    return App(**base)


def test_render_apps_text_with_drift_and_duplicate(make_console, monkeypatch):
    inv = [
        _capp(label="LM Studio", category="runner", cask=True, in_app=True,
              version="0.3.5", brew_recorded="0.3.0", cask_token="lm-studio"),  # clueless drift
        _capp(label="ollama", category="runner", cask=True, formula=True,
              version="0.1.2", cask_token="ollama"),                            # duplicate
    ]
    monkeypatch.setattr(cli.apps, "scan", lambda: inv)
    monkeypatch.setattr(cli.versions, "cask_auto_updates", lambda: {})  # no auto_updates known
    c, buf = make_console()
    cli.render_apps(c)
    out = buf.getvalue()
    assert "AI/ML APPS" in out and "model runners" in out
    assert "LM Studio 0.3.5" in out
    assert "self-updated past brew" in out and "clobber" in out   # clueless drift
    assert "likely duplicate" in out                              # ollama dup


def test_render_apps_drift_with_auto_updates_is_benign(make_console, monkeypatch):
    inv = [_capp(label="Claude", category="assistant", cask=True, in_app=True,
                 version="2.0", brew_recorded="1.0", cask_token="claude")]
    monkeypatch.setattr(cli.apps, "scan", lambda: inv)
    monkeypatch.setattr(cli.versions, "cask_auto_updates", lambda: {"claude": True})
    c, buf = make_console()
    cli.render_apps(c)
    out = buf.getvalue()
    # benign drift: "brew defers", not the clueless "will clobber" gloss (the footer
    # legend's "can clobber" is always present, so assert on the per-app wording).
    assert "brew defers" in out and "will clobber it" not in out


def test_render_apps_empty(make_console, monkeypatch):
    monkeypatch.setattr(cli.apps, "scan", lambda: [])
    monkeypatch.setattr(cli.versions, "cask_auto_updates", lambda: {})
    c, buf = make_console()
    cli.render_apps(c)
    assert "none detected" in buf.getvalue()


def test_render_apps_want_filters_category(make_console, monkeypatch):
    inv = [_capp(label="Ollama", category="runner", cask=True, cask_token="ollama"),
           _capp(label="Cursor", category="coding", cask=True, cask_token="cursor")]
    monkeypatch.setattr(cli.apps, "scan", lambda: inv)
    monkeypatch.setattr(cli.versions, "cask_auto_updates", lambda: {})
    c, buf = make_console()
    cli.render_apps(c, want=lambda k: k == "coding")
    out = buf.getvalue()
    assert "Cursor" in out and "Ollama" not in out


def test_render_apps_json(monkeypatch, capsys):
    inv = [_capp(label="LM Studio", cask=True, in_app=True, version="0.3.5",
                 brew_recorded="0.3.0", cask_token="lm-studio")]
    monkeypatch.setattr(cli.apps, "scan", lambda: inv)
    monkeypatch.setattr(cli.versions, "cask_auto_updates", lambda: {"lm-studio": False})
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_apps(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["label"] == "LM Studio" and payload[0]["drift"] is True
    assert payload[0]["auto_updates"] is False


# =========================================================================== #
# render_mlx
# =========================================================================== #
def test_render_mlx_non_apple(make_console, monkeypatch):
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "unsupported")
    c, buf = make_console()
    cli.render_mlx(c)
    assert "Apple-Silicon only" in buf.getvalue()


def _mlx_setup(monkeypatch, interps=(), runtimes=(), models=0):
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli.detect, "chip_name", lambda: "Apple M4 Pro")
    monkeypatch.setattr(cli.detect, "accelerator",
                        lambda chip: Accelerator("apple", "Apple M4 Pro GPU", None, "Metal", cores=16))
    monkeypatch.setattr(cli.mlx, "scan", lambda: list(interps))
    monkeypatch.setattr(cli.mlx, "lmstudio_mlx_runtimes", lambda: list(runtimes))
    monkeypatch.setattr(cli.mlx, "mlx_community_model_count", lambda: models)


def test_render_mlx_readiness(make_console, monkeypatch):
    _mlx_setup(monkeypatch, runtimes=["0.3.1", "0.2.0"], models=5)
    c, buf = make_console()
    cli.render_mlx(c)
    out = buf.getvalue()
    assert "READINESS" in out and "Apple M4 Pro GPU" in out
    assert "5 cached" in out
    assert "MLX runtime 0.3.1" in out and "+1 older" in out


def test_render_mlx_no_runtime_no_libs(make_console, monkeypatch):
    _mlx_setup(monkeypatch, interps=[], runtimes=[], models=0)
    c, buf = make_console()
    cli.render_mlx(c)
    out = buf.getvalue()
    assert "no MLX runtime" in out
    assert "No MLX packages installed" in out


def test_render_mlx_libraries_with_managed_caution(make_console, monkeypatch):
    from ara.mlx import MlxInterpreter
    mi = MlxInterpreter(path="/opt/homebrew/bin/python3", origin="Homebrew", version="3.12.4",
                        externally_managed=True, packages={"mlx": "0.18", "mlx-lm": "0.20"})
    _mlx_setup(monkeypatch, interps=[mi], models=2)
    c, buf = make_console()
    cli.render_mlx(c)
    out = buf.getvalue()
    assert "Homebrew 3.12.4" in out
    assert "managed by Homebrew" in out          # manager_of caution
    assert "mlx 0.18" in out and "mlx-lm 0.20" in out
    assert "not installed:" in out               # the missing-modalities line


def test_render_mlx_want_libraries_only(make_console, monkeypatch):
    _mlx_setup(monkeypatch, interps=[], models=1)
    c, buf = make_console()
    cli.render_mlx(c, want=lambda k: k == "libraries")
    out = buf.getvalue()
    assert "READINESS" not in out and "LIBRARIES" in out


def test_render_mlx_json(monkeypatch, capsys):
    from ara.mlx import MlxInterpreter
    mi = MlxInterpreter(path="/x", origin="venv", version="3.12", packages={"mlx": "0.18"})
    _mlx_setup(monkeypatch, interps=[mi], runtimes=["0.3.1"], models=3)
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_mlx(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["apple_silicon"] is True and payload["mlx_community_models"] == 3
    assert payload["interpreters"][0]["packages"] == {"mlx": "0.18"}


# =========================================================================== #
# _det_apps summary (inside render_detect)
# =========================================================================== #
def test_det_apps_summary(make_console, monkeypatch):
    m = _machine(apps=[
        _capp(label="LM Studio", category="runner"),
        _capp(label="Ollama", category="runner"),
        _capp(label="GPT4All", category="runner"),
        _capp(label="Jan", category="runner"),
    ])
    stub = lambda count=1, discover=(): None
    monkeypatch.setattr(cli.pythons, "count", lambda: 1)
    c, buf = make_console()
    cli._det_apps(c, m)
    out = buf.getvalue()
    assert "AI/ML APPS" in out and "model runners" in out
    assert "(+1 more)" in out   # 4 runners, top 3 shown


def test_det_apps_empty(make_console):
    m = _machine(apps=[])
    c, buf = make_console()
    cli._det_apps(c, m)
    assert "none detected" in buf.getvalue()


# =========================================================================== #
# main dispatch: apps / mlx + include/exclude parsing
# =========================================================================== #
def test_main_apps_dispatch(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["apps", "--json"])
    assert rec["apps"] is True


def test_main_mlx_dispatch(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["mlx"])
    assert rec["mlx"] is False


def test_main_include_builds_want(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["detect", "--include", "system,memory"])
    want = rec["detect_want"]
    assert want is not None and want("system") and not want("accelerator")


def test_main_include_equals_form(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["detect", "--include=system"])
    assert rec["detect_want"]("system") and not rec["detect_want"]("memory")


def test_main_exclude_builds_want(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["detect", "--exclude=memory"])
    want = rec["detect_want"]
    assert not want("memory") and want("system")


def test_main_no_filter_means_no_want(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["detect"])
    assert rec["detect_want"] is None


# =========================================================================== #
# remaining want-filter / branch corners
# =========================================================================== #
def test_resolve_want_no_filter_returns_none_quietly(make_console):
    c, buf = make_console()
    assert cli._resolve_want("profile", [], [], c) is None   # no include/exclude → no warn
    assert buf.getvalue() == ""


def test_render_detect_want_filters_sections(make_console, monkeypatch, stub_pythons):
    stub_pythons(count=1)
    monkeypatch.setattr(cli.detect, "machine", lambda: _machine())
    c, buf = make_console()
    cli.render_detect(c, want=lambda k: k == "system")
    out = buf.getvalue()
    assert "SYSTEM" in out and "MEMORY" not in out and "ACCELERATOR" not in out


def test_render_status_want_excludes_processes(make_console, monkeypatch):
    monkeypatch.setattr(cli.status, "scan", lambda: [_proc()])
    monkeypatch.setattr(cli.status, "scan_apps", lambda: [])
    c, buf = make_console()
    cli.render_status(c, want=lambda k: k != "processes")   # workloads section filtered out
    out = buf.getvalue()
    assert "RUNNING AI/ML" not in out and "AI APPS" in out  # apps section still shows


def test_render_status_want_excludes_apps(make_console, monkeypatch):
    monkeypatch.setattr(cli.status, "scan", lambda: [_proc()])
    monkeypatch.setattr(cli.status, "scan_apps", lambda: [_app()])
    c, buf = make_console()
    cli.render_status(c, want=lambda k: k != "apps")        # apps section filtered out
    out = buf.getvalue()
    assert "RUNNING AI/ML" in out and "AI APPS" not in out


def test_render_python_want_excludes_interpreters(make_console, monkeypatch):
    monkeypatch.setattr(cli.pythons, "discover", lambda probe=True: [])
    c, buf = make_console()
    cli.render_python(c, want=lambda k: k != "interpreters")
    assert buf.getvalue() == ""


def test_det_apps_single_item_no_more_suffix(make_console, monkeypatch):
    m = _machine(apps=[_capp(label="Cursor", category="coding")])
    monkeypatch.setattr(cli.pythons, "count", lambda: 1)
    c, buf = make_console()
    cli._det_apps(c, m)
    out = buf.getvalue()
    assert "Cursor" in out and "more)" not in out   # 1 item → no "(+N more)"


def test_render_mlx_readiness_only(make_console, monkeypatch):
    _mlx_setup(monkeypatch, interps=[], models=1)
    c, buf = make_console()
    cli.render_mlx(c, want=lambda k: k == "readiness")
    out = buf.getvalue()
    assert "READINESS" in out and "LIBRARIES" not in out


def test_render_mlx_unmanaged_interp_with_all_packages(make_console, monkeypatch):
    from ara.mlx import MlxInterpreter
    # a venv (unmanaged → no caution) holding at least one package from every group
    full = {pkgs[0]: "1.0" for _label, pkgs in cli.mlx.GROUPS}
    mi = MlxInterpreter(path="/venv/bin/python", origin="venv", version="3.12",
                        externally_managed=False, packages=full)
    _mlx_setup(monkeypatch, interps=[mi], models=0)
    c, buf = make_console()
    cli.render_mlx(c)
    out = buf.getvalue()
    assert "managed by" not in out        # venv → no caution (570->573 false branch)
    assert "not installed:" not in out    # every group covered (581->585 false branch)


# --------------------------------------------------------------------------- #
# ara install / ara uninstall — engine bootstrap commands
# --------------------------------------------------------------------------- #
def test_render_install_installs_resolved_engine(make_console, monkeypatch):
    monkeypatch.setattr(cli.engines, "resolve", lambda v: "wmx")
    monkeypatch.setattr(cli.engines, "install",
                        lambda k: cli.engines.InstallResult("wmx", "installed", "ok"))
    c, buf = make_console()
    rc = cli.render_install(c, engine="auto")
    assert rc == 0
    assert "wmx-suite" in buf.getvalue()
    assert "installed" in buf.getvalue().lower()


def _stub_install(monkeypatch, key, status, detail=""):
    monkeypatch.setattr(cli.engines, "resolve", lambda v: key)
    monkeypatch.setattr(cli.engines, "install",
                        lambda k: cli.engines.InstallResult(k, status, detail))


def test_render_install_already_present_is_success(make_console, monkeypatch):
    _stub_install(monkeypatch, "wmx", "already")
    c, buf = make_console()
    assert cli.render_install(c, engine="wmx") == 0
    assert "already" in buf.getvalue().lower()


def test_render_install_coming_soon_exits_nonzero(make_console, monkeypatch):
    _stub_install(monkeypatch, "wcx", "coming_soon", "wcx_suite isn't available yet")
    c, buf = make_console()
    assert cli.render_install(c, engine="wcx") == 1
    assert "coming soon" in buf.getvalue().lower()


def test_render_install_failed_shows_detail_and_exits_nonzero(make_console, monkeypatch):
    _stub_install(monkeypatch, "wmx", "failed", "git clone exploded")
    c, buf = make_console()
    assert cli.render_install(c, engine="wmx") == 1
    assert "git clone exploded" in buf.getvalue()


def test_render_install_no_hardware_match_exits_nonzero(make_console, monkeypatch):
    monkeypatch.setattr(cli.engines, "resolve", lambda v: None)
    c, buf = make_console()
    assert cli.render_install(c, engine="auto") == 1
    assert "no engine" in buf.getvalue().lower()


def test_render_install_json(monkeypatch, capsys):
    _stub_install(monkeypatch, "wmx", "installed", "ok")
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_install(c, engine="wmx", as_json=True)
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "installed" and out["key"] == "wmx" and rc == 0


def _stub_uninstall(monkeypatch, key, status, detail=""):
    monkeypatch.setattr(cli.engines, "resolve", lambda v: key)
    monkeypatch.setattr(cli.engines, "uninstall",
                        lambda k: cli.engines.InstallResult(k, status, detail))


def test_render_uninstall_removes_engine(make_console, monkeypatch):
    _stub_uninstall(monkeypatch, "wmx", "removed")
    c, buf = make_console()
    assert cli.render_uninstall(c, engine="wmx") == 0
    assert "removed" in buf.getvalue().lower() and "wmx-suite" in buf.getvalue()


def test_render_uninstall_absent_is_success(make_console, monkeypatch):
    _stub_uninstall(monkeypatch, "wmx", "absent")
    c, buf = make_console()
    assert cli.render_uninstall(c, engine="wmx") == 0
    assert "not installed" in buf.getvalue().lower()


def test_render_uninstall_failed_exits_nonzero(make_console, monkeypatch):
    _stub_uninstall(monkeypatch, "wmx", "failed", "permission denied")
    c, buf = make_console()
    assert cli.render_uninstall(c, engine="wmx") == 1
    assert "permission denied" in buf.getvalue()


def test_render_uninstall_no_match_exits_nonzero(make_console, monkeypatch):
    monkeypatch.setattr(cli.engines, "resolve", lambda v: None)
    c, buf = make_console()
    assert cli.render_uninstall(c, engine="auto") == 1
    assert "no engine" in buf.getvalue().lower()


def test_render_uninstall_json(monkeypatch, capsys):
    _stub_uninstall(monkeypatch, "wmx", "removed")
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_uninstall(c, engine="wmx", as_json=True)
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "removed" and rc == 0


def test_main_install_defaults_to_auto(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["install"]) == 0
    assert rec["install"] == {"engine": "auto", "as_json": False}


def test_main_install_with_engine_flag(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["install", "--engine", "wmx"])
    assert rec["install"]["engine"] == "wmx"


def test_main_install_engine_equals_form(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["install", "--engine=wcx", "--json"])
    assert rec["install"] == {"engine": "wcx", "as_json": True}


def test_main_uninstall_with_engine_flag(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["uninstall", "--engine", "wmx"])
    assert rec["uninstall"]["engine"] == "wmx"


def test_render_install_no_match_json(monkeypatch, capsys):
    monkeypatch.setattr(cli.engines, "resolve", lambda v: None)
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_install(c, engine="auto", as_json=True)
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "no_match" and rc == 1


def test_render_uninstall_no_match_json(monkeypatch, capsys):
    monkeypatch.setattr(cli.engines, "resolve", lambda v: None)
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_uninstall(c, engine="auto", as_json=True)
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "no_match" and rc == 1


def test_render_landing_lists_install_command(make_console, monkeypatch):
    monkeypatch.setattr(cli.detect, "chip_name", lambda: "Apple M4 Pro")
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (False, "wmx-suite"))
    c, buf = make_console()
    cli.render_landing(c)
    assert "install the engine" in buf.getvalue()


# --------------------------------------------------------------------------- #
# ara models — the catalog view (scan HF cache + list with characterization)
# --------------------------------------------------------------------------- #
def test_render_models_lists_catalog(make_console, store, monkeypatch):
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/A", "modality": "text"},
                                     {"model_id": "org/B", "modality": "text"}])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, e: [{"model_id": "org/A", "safe_context": 16000,
                                             "decode_context": None}])
    c, buf = make_console()
    cli.render_models(c)
    out = buf.getvalue()
    assert "MODEL CATALOG" in out
    assert "org/A" in out and "16000" in out
    assert "org/B" in out and "not characterized" in out
    assert "2 cataloged" in out


def test_render_models_distinguishes_measured_no_ceiling(make_console, store, monkeypatch):
    """A model ARA measured but couldn't fit (row with safe_context=None) reads as
    characterized-with-no-ceiling — not lumped with never-measured models. Keeps
    `ara models` consistent with `ara profile`'s CHARACTERIZED list (which shows '—')."""
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/Fits", "modality": "text"},
                                     {"model_id": "org/NoCeiling", "modality": "text"},
                                     {"model_id": "org/Never", "modality": "text"}])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, e: [{"model_id": "org/Fits", "safe_context": 16000,
                                             "decode_context": None},
                                            {"model_id": "org/NoCeiling", "safe_context": None,
                                             "decode_context": None}])
    c, buf = make_console()
    cli.render_models(c)
    lines = {ln.split()[0]: ln for ln in buf.getvalue().splitlines() if "org/" in ln}
    assert "16000" in lines["org/Fits"]
    assert "no safe ceiling" in lines["org/NoCeiling"]
    assert "not characterized" not in lines["org/NoCeiling"]
    assert "not characterized" in lines["org/Never"]      # the only never-measured one
    assert "2 characterized on this machine" in buf.getvalue()


def test_render_models_empty_and_no_engine(make_console, store, monkeypatch):
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models", lambda con: [])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "unsupported")  # engine_key None
    c, buf = make_console()
    cli.render_models(c)
    assert "empty" in buf.getvalue()


def test_render_models_json(monkeypatch, capsys, store):
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/A", "modality": "text"}])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, e: [{"model_id": "org/A", "safe_context": 9000,
                                             "decode_context": None}])
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_models(c, as_json=True)
    data = json.loads(capsys.readouterr().out)
    assert data[0]["safe_context"] == 9000


def test_render_models_json_has_characterized_flag(monkeypatch, capsys, store):
    """Models JSON carries a `characterized` flag so a null ceiling that was *measured*
    (no fit) is distinguishable from one that was *never measured*."""
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/Fits", "modality": "text"},
                                     {"model_id": "org/NoCeiling", "modality": "text"},
                                     {"model_id": "org/Never", "modality": "text"}])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, e: [{"model_id": "org/Fits", "safe_context": 16000,
                                             "decode_context": None},
                                            {"model_id": "org/NoCeiling", "safe_context": None,
                                             "decode_context": None}])
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_models(c, as_json=True)
    data = {d["model_id"]: d for d in json.loads(capsys.readouterr().out)}
    assert data["org/Fits"].get("characterized") is True
    assert data["org/NoCeiling"].get("characterized") is True
    assert data["org/NoCeiling"]["safe_context"] is None
    assert data["org/Never"].get("characterized") is False


def test_render_models_best_fit_across_engines(make_console, store, monkeypatch):
    """A model characterized under two engines shows the LARGER ceiling + which engine reached
    it — the winbox case (GPU 3500 vs CPU 8192 for one model → 8192, cpu)."""
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/L", "modality": "text"}])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")     # default engine = wcx
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    per_engine = {"wcx": [{"model_id": "org/L", "safe_context": 3500, "decode_context": None}],
                  "cpu": [{"model_id": "org/L", "safe_context": 8192, "decode_context": None}]}
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, e: per_engine.get(e, []))
    c, buf = make_console()
    cli.render_models(c)
    line = next(ln for ln in buf.getvalue().splitlines() if "org/L" in ln)
    assert "8192" in line and "(cpu)" in line and "3500" not in line


def test_main_models_dispatch(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["models", "--json"])
    assert rec["models"] is True


# --------------------------------------------------------------------------- #
# ara recommend — analytic: catalog models that fit, ranked by context
# Spec 2026-06-23-capability-pipeline (Slice 3)
# --------------------------------------------------------------------------- #
def _model_row(model_id, *, weights_gb=4.0, max_context=8192, **over):
    base = dict(model_id=model_id, modality="text", params=None, quant=None,
                n_layers=32, hidden_size=4096, kv_heads=8, head_dim=128,
                weights_gb=weights_gb, max_context=max_context, updated_at="t")
    base.update(over)
    return base


def _wire_recommend(monkeypatch, set_platform, models, machine=None):
    set_platform("Darwin", "arm64")
    monkeypatch.setattr(cli.detect, "machine",
                        lambda: machine if machine is not None
                        else _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models", lambda con: models)
    monkeypatch.setattr(cli.db, "list_characterizations", lambda con, mk, e: [])


def test_recommend_ranks_fits_by_context(make_console, monkeypatch, set_platform):
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Small", max_context=8192),
                     _model_row("org/Big", max_context=131072)])
    c, buf = make_console()
    assert cli.render_recommend(c) == 0
    out = buf.getvalue()
    assert "RECOMMENDED MODELS" in out
    assert out.index("org/Big") < out.index("org/Small")   # most context first
    assert "full window" in out                            # both window-bound on a 48 GB Mac


def test_recommend_excludes_models_that_dont_fit(make_console, monkeypatch, set_platform):
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Fits", weights_gb=4.0),
                     _model_row("org/TooBig", weights_gb=200.0)])
    c, buf = make_console()
    assert cli.render_recommend(c) == 0
    out = buf.getvalue()
    assert "org/Fits" in out and "org/TooBig" not in out


def test_recommend_memory_bound_label(make_console, monkeypatch, set_platform):
    # A tiny CPU budget binds before a huge window → ranked, but not "full window".
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/BigCtx", weights_gb=4.0, max_context=131072)],
                    machine=_machine(backend="cpu", ram_total_gb=8.0))
    c, buf = make_console()
    assert cli.render_recommend(c) == 0
    out = buf.getvalue()
    assert "org/BigCtx" in out and "tok est." in out
    assert "full window" not in out


def test_recommend_marks_characterized(make_console, monkeypatch, set_platform):
    _wire_recommend(monkeypatch, set_platform, [_model_row("org/Known")])
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, e: [{"model_id": "org/Known", "safe_context": 12000,
                                             "decode_context": None}] if e == "wmx" else [])
    c, buf = make_console()
    assert cli.render_recommend(c) == 0
    assert "characterized" in buf.getvalue()


def test_recommend_none_fit(make_console, monkeypatch, set_platform):
    _wire_recommend(monkeypatch, set_platform, [_model_row("org/TooBig", weights_gb=500.0)])
    c, buf = make_console()
    assert cli.render_recommend(c) == 0
    assert "nothing in the catalog fits" in buf.getvalue()


def test_recommend_uses_measured_wall(make_console, monkeypatch, set_platform, store):
    # Spec 2026-06-23-capability-pipeline: after the detected engine is calibrated, recommend ranks
    # against the MEASURED budget, not the heuristic. A measured wall that's tighter than the
    # heuristic must actually bind the fit — proving the measurement drove the math.
    captured = {}
    real_limits = cli.estimate.limits

    def spy_limits(machine, measured=None):
        captured["measured"] = measured
        return real_limits(machine, measured=measured)

    _wire_recommend(monkeypatch, set_platform, [_model_row("org/Small", max_context=8192)])
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.estimate, "limits", spy_limits)
    cli.calibration.save_calibration(store, "wmx", fixed_overhead_gb=1.7,
                                     wall_gb=41.3, safe_budget_gb=39.3)
    c, buf = make_console()
    assert cli.render_recommend(c) == 0
    assert captured["measured"] is not None
    assert captured["measured"]["wall_gb"] == 41.3        # the stored measurement was passed in


def test_recommend_json(monkeypatch, set_platform, capsys):
    _wire_recommend(monkeypatch, set_platform, [_model_row("org/Big", max_context=131072)])
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_recommend(c, as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["model_id"] == "org/Big"
    assert payload[0]["fits"] is True and "est_context" in payload[0]


def test_main_recommend_dispatch(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["recommend", "--json"])
    assert rec["recommend"] is True


# --------------------------------------------------------------------------- #
# ara run — governed one-shot inference (Spec 2026-06-23-capability-pipeline, Slice 4)
# --------------------------------------------------------------------------- #
_CHAR = {"model_id": "org/m", "safe_context": 8192, "decode_context": None}


def _ok_generate(*a, **k):
    return {"completion": "ok"}


def _wire_run(monkeypatch, *, engine_ok=True, generate=_ok_generate, characterization=None,
              isatty=False):
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cpu")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (engine_ok, "llama.cpp"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, m: characterization)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: isatty))
    bk = types.SimpleNamespace()
    if generate is not None:
        bk.generate = generate
    monkeypatch.setattr(cli, "get_backend", lambda b=None: bk)


def test_run_refuses_uncharacterized(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization=None)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi") == 1
    out = buf.getvalue()
    assert "isn't characterized" in out and "ara characterize org/m" in out


def test_run_refuses_when_no_safe_ceiling(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization={"model_id": "org/m", "safe_context": None})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi") == 1
    assert "didn't fit" in buf.getvalue()


def test_run_engine_not_installed(make_console, monkeypatch):
    _wire_run(monkeypatch, engine_ok=False, characterization=_CHAR)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi") == 1
    assert "isn't installed" in buf.getvalue()


def test_run_unsupported_engine(make_console, monkeypatch):
    # A backend with no generate method (apple/cuda until their repos add the verb).
    _wire_run(monkeypatch, characterization=_CHAR, generate=None)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi") == 1
    assert "isn't supported" in buf.getvalue()


def test_run_generates_capped_at_ceiling(make_console, monkeypatch):
    seen = {}

    def gen(model, prompt, *, max_context, max_tokens):
        seen.update(model=model, prompt=prompt, max_context=max_context)
        return {"context": max_context, "completion": "the answer is 42"}

    _wire_run(monkeypatch, characterization=_CHAR, generate=gen, isatty=False)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="meaning?") == 0
    assert "the answer is 42" in buf.getvalue()
    assert seen["max_context"] == 8192 and seen["prompt"] == "meaning?"   # governed ceiling


def test_run_confirm_declined_skips(make_console, monkeypatch):
    called = []
    _wire_run(monkeypatch, characterization=_CHAR,
              generate=lambda *a, **k: called.append(1) or {"completion": "x"}, isatty=True)
    monkeypatch.setattr(cli, "_confirm", lambda q: False)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi") == 0
    assert "skipped" in buf.getvalue() and called == []


def test_run_yes_skips_confirm(make_console, monkeypatch):
    def no_ask(q):
        raise AssertionError("should not prompt with --yes")
    _wire_run(monkeypatch, characterization=_CHAR,
              generate=lambda *a, **k: {"completion": "ok"}, isatty=True)
    monkeypatch.setattr(cli, "_confirm", no_ask)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", assume_yes=True) == 0
    assert "ok" in buf.getvalue()


def test_run_confirm_accepted_runs(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization=_CHAR,
              generate=lambda *a, **k: {"completion": "done"}, isatty=True)
    monkeypatch.setattr(cli, "_confirm", lambda q: True)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi") == 0
    assert "done" in buf.getvalue()


def test_run_worker_refused(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization=_CHAR,
              generate=lambda *a, **k: {"refused": True, "reason": "memory pressure"})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", assume_yes=True) == 1
    assert "memory pressure" in buf.getvalue()


def test_run_failure(make_console, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("worker died")
    _wire_run(monkeypatch, characterization=_CHAR, generate=boom)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", assume_yes=True) == 1
    assert "run failed" in buf.getvalue()


def test_run_json(monkeypatch, capsys):
    _wire_run(monkeypatch, characterization=_CHAR,
              generate=lambda *a, **k: {"completion": "hello"})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_run(c, "org/m", prompt="hi", as_json=True, assume_yes=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["completion"] == "hello" and payload["safe_context"] == 8192


def test_run_usage_without_prompt(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization=_CHAR)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt=None) == 1
    assert "usage" in buf.getvalue()


def test_run_unknown_engine(make_console):
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="bogus") == 1
    assert "unknown engine" in buf.getvalue().lower()


# Cross-engine selection: with no --engine, run scans every engine this model is characterized
# under on this machine and picks the largest safe_context whose backend can actually generate —
# not just the detected engine. Spec 2026-06-23-capability-pipeline.
def _wire_run_cross(monkeypatch, *, detected, chars, supports, engine_ok=True, isatty=False):
    """chars: {engine_key: characterization|None}; supports: {backend: bool} (has .generate)."""
    monkeypatch.setattr(cli.detect, "backend_name", lambda: detected)
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (engine_ok, f"{b} pkg"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, m: chars.get(e))

    def backend(b=None):
        bk = types.SimpleNamespace()
        if supports.get(b):
            bk.generate = lambda model, prompt, *, max_context, max_tokens: {
                "engine_backend": b, "max_context": max_context, "completion": f"ran on {b}"}
        return bk
    monkeypatch.setattr(cli, "get_backend", backend)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: isatty))


def test_run_picks_characterized_engine_across_backends(make_console, monkeypatch):
    # Detected apple (no run support), but the model is characterized on cpu → run on cpu.
    _wire_run_cross(
        monkeypatch, detected="apple",
        chars={"cpu": {"model_id": "org/m", "safe_context": 4096}},
        supports={"cpu": True, "apple": False})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", assume_yes=True) == 0
    assert "ran on cpu" in buf.getvalue()


def test_run_picks_largest_safe_context_engine(monkeypatch, capsys):
    # Characterized on two run-capable engines → pick the largest safe_context.
    _wire_run_cross(
        monkeypatch, detected="cpu",
        chars={"cpu": {"model_id": "org/m", "safe_context": 4096},
               "wcx": {"model_id": "org/m", "safe_context": 16000}},
        supports={"cpu": True, "cuda": True})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_run(c, "org/m", prompt="hi", as_json=True, assume_yes=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"] == "wcx" and payload["safe_context"] == 16000


def test_run_engine_override_pins_named_engine(make_console, monkeypatch):
    # --engine pins exactly that engine even if another engine has a bigger ceiling.
    _wire_run_cross(
        monkeypatch, detected="cpu",
        chars={"cpu": {"model_id": "org/m", "safe_context": 4096},
               "wcx": {"model_id": "org/m", "safe_context": 16000}},
        supports={"cpu": True, "cuda": True})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="cpu", assume_yes=True) == 0
    assert "ran on cpu" in buf.getvalue()        # pinned to cpu (4096), not the bigger wcx


def test_run_characterized_only_on_unsupported_engine(make_console, monkeypatch):
    # Characterized on apple alone, whose backend can't generate yet → honest "not supported",
    # NOT a silent "uncharacterized" refusal.
    _wire_run_cross(
        monkeypatch, detected="apple",
        chars={"wmx": {"model_id": "org/m", "safe_context": 8192}},
        supports={"apple": False, "cpu": True})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", assume_yes=True) == 1
    out = buf.getvalue()
    assert "wmx" in out and "isn't supported" in out
    assert "ara characterize" not in out         # it IS characterized — don't point at characterize


def test_run_uncharacterized_on_every_engine_refuses(make_console, monkeypatch):
    # Characterized nowhere → refuse, pointing at characterize.
    _wire_run_cross(monkeypatch, detected="apple", chars={}, supports={"cpu": True})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", assume_yes=True) == 1
    assert "isn't characterized" in buf.getvalue() and "ara characterize org/m" in buf.getvalue()


def test_run_pinned_refuses_uncharacterized(make_console, monkeypatch):
    # --engine pins exactly that engine: uncharacterized THERE refuses, pointing at characterize.
    _wire_run(monkeypatch, characterization=None)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="cpu") == 1
    out = buf.getvalue()
    assert "isn't characterized on cpu" in out and "ara characterize org/m --engine cpu" in out


def test_run_pinned_refuses_when_no_safe_ceiling(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization={"model_id": "org/m", "safe_context": None})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="cpu") == 1
    assert "didn't fit on cpu" in buf.getvalue()


def test_run_pinned_unsupported_engine(make_console, monkeypatch):
    # --engine pins an engine whose backend can't generate yet → honest "isn't supported".
    _wire_run(monkeypatch, characterization=_CHAR, generate=None)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="cpu") == 1
    assert "isn't supported on the" in buf.getvalue()


def test_main_run_dispatch(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["run", "org/m", "hello", "world"])
    assert rec["run"]["model"] == "org/m"
    assert rec["run"]["prompt"] == "hello world"


def test_main_run_usage_no_model(make_console, monkeypatch):
    monkeypatch.setattr("sys.argv", ["ara", "run"])
    assert cli.main() == 1


# --------------------------------------------------------------------------- #
# ara search — Hub search (engine-agnostic)
# --------------------------------------------------------------------------- #
def test_render_search_lists_results(make_console, monkeypatch):
    monkeypatch.setattr(cli.hub, "search",
                        lambda q: [{"id": "org/Smol", "downloads": 1000, "likes": 5}])
    c, buf = make_console()
    assert cli.render_search(c, "smol") == 0
    out = buf.getvalue()
    assert "HUB SEARCH: smol" in out and "org/Smol" in out and "1000" in out


def test_render_search_empty(make_console, monkeypatch):
    monkeypatch.setattr(cli.hub, "search", lambda q: [])
    c, buf = make_console()
    assert cli.render_search(c, "zzz") == 0
    assert "no models found" in buf.getvalue()


def test_render_search_hf_missing(make_console, monkeypatch):
    monkeypatch.setattr(cli.hub, "search", lambda q: None)
    c, buf = make_console()
    assert cli.render_search(c, "x") == 1
    assert "hf CLI" in buf.getvalue()


def test_render_search_json(monkeypatch, capsys):
    monkeypatch.setattr(cli.hub, "search", lambda q: [{"id": "a", "downloads": 1, "likes": 0}])
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_search(c, "a", as_json=True) == 0
    assert json.loads(capsys.readouterr().out)[0]["id"] == "a"


def test_main_search_dispatch(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "render_search",
                        lambda c, q, as_json=False: (seen.update(q=q, json=as_json) or 0))
    _run_main(monkeypatch, ["search", "smol", "lm", "--json"])
    assert seen == {"q": "smol lm", "json": True}


def test_main_search_no_query(monkeypatch, capsys):
    assert _run_main(monkeypatch, ["search"]) == 1
    assert "usage: ara search" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# ara models <id> — single-model detail (wmx's `show`)
# --------------------------------------------------------------------------- #
def _meta(**over):
    base = dict(modality="text", n_layers=30, hidden_size=576, kv_heads=3,
                head_dim=64, max_context=8192, quant="mlx-4bit")
    base.update(over)
    return base


def test_model_detail_full(make_console, monkeypatch):
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: {"safe_context": 16000, "decode_context": None})
    c, buf = make_console()
    assert cli.render_model_detail(c, "org/Smol") == 0
    out = buf.getvalue()
    assert "org/Smol" in out and "3 heads × 64 dim" in out
    assert "8192" in out and "mlx-4bit" in out and "16000" in out


def test_model_detail_sparse_no_engine(make_console, monkeypatch):
    monkeypatch.setattr(cli.catalog, "describe",
                        lambda mid: _meta(modality=None, n_layers=None, kv_heads=None,
                                          head_dim=None, max_context=None, quant=None))
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "unsupported")  # engine_key None
    c, buf = make_console()
    assert cli.render_model_detail(c, "x") == 0
    out = buf.getvalue()
    assert "?" in out and "none" in out and "not characterized" in out


def test_model_detail_not_found(make_console, monkeypatch):
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: None)
    c, buf = make_console()
    assert cli.render_model_detail(c, "nope") == 1
    assert "couldn't describe" in buf.getvalue()


def test_model_detail_json(monkeypatch, capsys):
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: {"safe_context": 9000, "decode_context": None})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_model_detail(c, "org/A", as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["model_id"] == "org/A" and data["safe_context"] == 9000
    assert data["decode_context"] is None


def test_model_detail_measured_no_ceiling(make_console, monkeypatch):
    """`ara models <id>` for a measured-but-unfit model reads 'no safe ceiling',
    not 'not characterized' — consistent with `ara models` and `ara profile`."""
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: {"safe_context": None, "decode_context": None})
    c, buf = make_console()
    assert cli.render_model_detail(c, "org/Unfit") == 0
    out = buf.getvalue()
    assert "no safe ceiling" in out
    assert "not characterized" not in out


def test_model_detail_json_characterized_flag(monkeypatch, capsys):
    """Detail JSON flags a measured-but-unfit model as characterized with a null ceiling."""
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: {"safe_context": None, "decode_context": None})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_model_detail(c, "org/Unfit", as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["safe_context"] is None
    assert data.get("characterized") is True
    assert data.get("decode_context") is None


def test_model_detail_json_uncharacterized_flag(monkeypatch, capsys):
    """Detail JSON flags a never-measured model as not characterized."""
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "unsupported")  # engine_key None → ch None
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_model_detail(c, "x", as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["safe_context"] is None
    assert data.get("characterized") is False
    assert data.get("decode_context") is None


def test_main_models_id_dispatch(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "render_model_detail",
                        lambda c, mid, as_json=False: (seen.update(mid=mid) or 0))
    _run_main(monkeypatch, ["models", "org/Smol"])
    assert seen["mid"] == "org/Smol"


# --------------------------------------------------------------------------- #
# ara characterize <model> — measure + store a model's ceiling (any engine)
# --------------------------------------------------------------------------- #
def _wire_characterize(monkeypatch, *, backend="apple", engine_ok=True, characterize=None):
    monkeypatch.setattr(cli.detect, "backend_name", lambda: backend)
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (engine_ok, "wmx-suite"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    if characterize is not None:
        monkeypatch.setattr(cli, "get_backend",
                            lambda b=None: types.SimpleNamespace(
                                characterize=characterize,
                                calibration_model_cached=lambda m: True,   # skip pre-fetch
                                download_calibration_model=lambda m: None,
                            ))


def test_render_characterize_persists_and_shows(make_console, store, monkeypatch):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 20000,
                                               "decode_context": None, "points": [[512, 1.4]]})
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Model") == 0
    assert "20000" in buf.getvalue()
    row = cli.db.get_characterization(store, "mkey", "wmx", "org/Model")
    assert row["safe_context"] == 20000 and row["points"] == [[512, 1.4]]


def test_characterize_self_calibrates_when_uncalibrated(make_console, store, monkeypatch):
    # Spec 2026-06-23-capability-pipeline (Slice 2): characterize owns calibration — it measures +
    # persists the engine baseline once when none is stored, before the ramp.
    calls = []
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "wmx-suite"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(
        characterize=lambda m: {"model": m, "safe_context": 9000, "decode_context": None, "points": []},
        calibration_model_cached=lambda m: True,
        download_calibration_model=lambda m: None,
        calibrate=lambda: (calls.append("cal") or {"overhead_gb": 1.7,
                                                    "wall_gb": 41.3, "safe_budget_gb": 39.3}),
    ))
    c, _ = make_console()
    assert cli.render_characterize(c, "org/M") == 0
    assert calls == ["cal"]                                            # calibrated once, before ramp
    row = cli.db.get_calibration(store, "mkey", "wmx")
    assert row["fixed_overhead_gb"] == 1.7                             # persisted
    # The measured wall + budget ride alongside so profile/recommend can report reality.
    assert row["wall_gb"] == 41.3 and row["safe_budget_gb"] == 39.3


def test_characterize_persists_measured_wall_when_overhead_none(make_console, store, monkeypatch):
    # CPU/CUDA read an EXACT wall, so calibrate returns overhead_gb=None. The measured wall must
    # still be stored — otherwise profile/recommend report a perpetual estimate on the very engines
    # `run` works on. Spec 2026-06-23-capability-pipeline.
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cpu")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "llama.cpp"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(
        characterize=lambda m: {"model": m, "safe_context": 9000, "decode_context": None, "points": []},
        calibration_model_cached=lambda m: True,
        download_calibration_model=lambda m: None,
        calibrate=lambda: {"overhead_gb": None, "wall_gb": 30.0, "safe_budget_gb": 28.0},
    ))
    c, _ = make_console()
    assert cli.render_characterize(c, "org/M") == 0
    row = cli.db.get_calibration(store, "mkey", "cpu")
    assert row is not None                                    # persisted despite overhead None
    assert row["wall_gb"] == 30.0 and row["safe_budget_gb"] == 28.0


def test_characterize_skips_calibration_when_already_calibrated(make_console, store, monkeypatch):
    # Already calibrated → characterize does NOT recalibrate (idempotent).
    calls = []
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "wmx-suite"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    cli.calibration.save_calibration(store, "wmx", fixed_overhead_gb=2.0)   # already calibrated
    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(
        characterize=lambda m: {"model": m, "safe_context": 5000, "decode_context": None, "points": []},
        calibration_model_cached=lambda m: True,
        download_calibration_model=lambda m: None,
        calibrate=lambda: (calls.append("cal") or {"overhead_gb": 9.9}),
    ))
    c, _ = make_console()
    assert cli.render_characterize(c, "org/M") == 0
    assert calls == []                                                 # not recalibrated
    assert cli.db.get_calibration(store, "mkey", "wmx")["fixed_overhead_gb"] == 2.0   # unchanged


def test_render_characterize_no_ceiling(make_console, store, monkeypatch):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": None,
                                               "decode_context": None, "points": []})
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Big") == 0
    assert "couldn't fit" in buf.getvalue()
    assert cli.db.get_characterization(store, "mkey", "wmx", "org/Big")["safe_context"] is None


def test_render_characterize_engine_not_installed(make_console, monkeypatch):
    _wire_characterize(monkeypatch, engine_ok=False)
    c, buf = make_console()
    assert cli.render_characterize(c, "x") == 1
    assert "ara install" in buf.getvalue()


def test_render_characterize_engine_error(make_console, monkeypatch):
    def boom(m):
        raise RuntimeError("OOM guard tripped")
    _wire_characterize(monkeypatch, characterize=boom)
    c, buf = make_console()
    assert cli.render_characterize(c, "x") == 1
    assert "characterization failed" in buf.getvalue()


def test_render_characterize_json(monkeypatch, capsys, store):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 9000,
                                               "decode_context": None, "points": []})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(c, "org/M", as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["safe_context"] == 9000
    assert "decode_context" in data


def test_render_characterize_engine_flag_overrides_detected_backend(make_console, store, monkeypatch):
    # winbox's case: a GPU is detected (cuda), but `--engine cpu` must run on the CPU backend
    # and store under the cpu engine key — never silently fall through to the detected GPU.
    seen = {}
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli, "engine_status",
                        lambda b=None: (seen.update(status_backend=b) or (True, "llama.cpp")))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)

    def fake_get_backend(b=None):
        seen["backend"] = b
        return types.SimpleNamespace(
            characterize=lambda m: {"model": m, "safe_context": 8192, "points": [[2000, 0.2]]},
            calibration_model_cached=lambda m: True,   # skip pre-fetch in this test
            download_calibration_model=lambda m: None,
        )

    monkeypatch.setattr(cli, "get_backend", fake_get_backend)
    c, _buf = make_console()
    assert cli.render_characterize(c, "org/G", engine="cpu") == 0
    assert seen["backend"] == "cpu"          # ran on the CPU backend, not the detected cuda
    assert seen["status_backend"] == "cpu"   # install check targeted the CPU engine
    assert cli.db.get_characterization(store, "mkey", "cpu", "org/G")["safe_context"] == 8192


def test_render_characterize_unknown_engine_errors(make_console, monkeypatch):
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    c, buf = make_console()
    assert cli.render_characterize(c, "x", engine="bogus") == 1
    assert "unknown engine" in buf.getvalue().lower()


def _error_characterize(model):
    # the driver shape when an engine couldn't even LOAD the model (preflight error)
    return {"model": model, "safe_context": None, "points": [], "error": "no transformers config"}


def test_render_characterize_skips_persist_on_engine_error(make_console, store, monkeypatch):
    # An engine that can't load the model returns `error` (not a measurement): don't persist a
    # misleading null row, and suggest a compatible engine when we can tell (a .gguf → cpu).
    _wire_characterize(monkeypatch, characterize=_error_characterize)   # default backend apple→wmx
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Model.gguf") == 1
    out = buf.getvalue()
    assert "couldn't load" in out
    assert "--engine cpu" in out                       # suggested the GGUF-capable engine
    assert cli.db.get_characterization(store, "mkey", "wmx", "org/Model.gguf") is None   # not stored


def test_render_characterize_engine_error_json(monkeypatch, capsys, store):
    _wire_characterize(monkeypatch, characterize=_error_characterize)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(c, "org/Model.gguf", as_json=True) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "no transformers config"


def test_render_characterize_engine_error_no_suggestion(make_console, store, monkeypatch):
    # A non-GGUF ref on the wrong engine: we can't cheaply tell which engine fits → no hint.
    _wire_characterize(monkeypatch, characterize=_error_characterize)
    c, buf = make_console()
    assert cli.render_characterize(c, "org/PlainModel") == 1
    out = buf.getvalue()
    assert "couldn't load" in out and "--engine" not in out


def test_main_characterize_dispatch(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["characterize", "org/Model"])
    assert rec["characterize"] == "org/Model"
    assert rec["characterize_engine"] is None    # no --engine → detected backend


def test_main_characterize_passes_engine(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["characterize", "org/Model", "--engine", "cpu"])
    assert rec["characterize"] == "org/Model"
    assert rec["characterize_engine"] == "cpu"


def test_main_characterize_no_model(monkeypatch, capsys):
    assert _run_main(monkeypatch, ["characterize"]) == 1
    assert "usage: ara characterize" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# render_characterize — pre-fetch block (task #47)
# --------------------------------------------------------------------------- #
def _wire_characterize_bk(monkeypatch, bk, *, backend="apple", engine_ok=True,
                          size_gb=4.0, free_gb=50.0):
    """Wire render_characterize with a FakeBackend and stubbed acquire functions."""
    monkeypatch.setattr(cli.detect, "backend_name", lambda: backend)
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (engine_ok, "wmx-suite"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    monkeypatch.setattr(cli, "get_backend", lambda b=None: bk)
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: size_gb)
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: free_gb)


def _fake_bk_characterize(model):
    return {"model": model, "safe_context": 16000, "decode_context": None, "points": [[1024, 1.2]]}


def test_render_characterize_prefetch_uncached_transformers(make_console, store, monkeypatch):
    # Uncached transformers model on a compatible engine → download fired, then characterize runs.
    bk = FakeBackend(_limits(), cached=False)
    bk.calibrate_result = None  # self-calibrate runs but returns None → nothing persisted
    # Give the FakeBackend a characterize method
    bk.characterize = _fake_bk_characterize
    _wire_characterize_bk(monkeypatch, bk)
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Model") == 0
    assert bk.downloaded == ["org/Model"]            # download was called
    assert "downloading" in buf.getvalue()            # status line emitted
    assert "16000" in buf.getvalue()                  # characterize result shown
    row = cli.db.get_characterization(store, "mkey", "wmx", "org/Model")
    assert row["safe_context"] == 16000               # result persisted


def test_render_characterize_prefetch_already_cached(make_console, store, monkeypatch):
    # Already-cached model → download NOT called; characterize still runs.
    bk = FakeBackend(_limits(), cached=True)
    bk.characterize = _fake_bk_characterize
    _wire_characterize_bk(monkeypatch, bk)
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Model") == 0
    assert bk.downloaded == []                        # no download
    assert "16000" in buf.getvalue()


def test_render_characterize_prefetch_incompatible_engine(make_console, store, monkeypatch):
    # A .gguf model on the wmx (apple) engine: engine_for_model returns "cpu" != "wmx"
    # → incompatible=True → download NOT called; existing flow proceeds (engine error path).
    bk = FakeBackend(_limits(), cached=False)
    bk.characterize = _fake_bk_characterize
    _wire_characterize_bk(monkeypatch, bk)
    c, buf = make_console()
    # "org/model.gguf" → engine_for_model returns "cpu"; sel.engine_key is "wmx" → incompatible
    assert cli.render_characterize(c, "org/model.gguf") == 0
    assert bk.downloaded == []                        # download skipped (incompatible)


def test_render_characterize_prefetch_insufficient_disk(make_console, monkeypatch):
    # Not enough disk → error emitted, returns 1, characterize NOT called.
    bk = FakeBackend(_limits(), cached=False)
    bk.characterize = _fake_bk_characterize
    # size_gb=10, free_gb=5, DISK_BUFFER_GB=2 → 5 < 10+2 → shortfall
    _wire_characterize_bk(monkeypatch, bk, size_gb=10.0, free_gb=5.0)
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Big") == 1
    assert "not enough disk" in buf.getvalue()
    assert bk.downloaded == []                        # characterize never reached


def test_render_characterize_prefetch_insufficient_disk_json(monkeypatch, capsys):
    # --json variant of the insufficient-disk branch.
    bk = FakeBackend(_limits(), cached=False)
    bk.characterize = _fake_bk_characterize
    _wire_characterize_bk(monkeypatch, bk, size_gb=10.0, free_gb=5.0)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(c, "org/Big", as_json=True) == 1
    out = json.loads(capsys.readouterr().out)
    assert "error" in out and "disk" in out["error"]


# --------------------------------------------------------------------------- #
# decode-safe ceiling display (task #48b)
# --------------------------------------------------------------------------- #
def test_render_characterize_decode_shown_when_greater(make_console, store, monkeypatch):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 20000,
                                               "decode_context": 25000, "points": []})
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Model") == 0
    out = buf.getvalue()
    assert "25000" in out and "decode" in out and "est." in out


def test_render_characterize_decode_hidden_when_none(make_console, store, monkeypatch):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 20000,
                                               "decode_context": None, "points": []})
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Model") == 0
    assert "decode" not in buf.getvalue()


def test_render_characterize_decode_hidden_when_not_greater(make_console, store, monkeypatch):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 20000,
                                               "decode_context": 15000, "points": []})
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Model") == 0
    assert "decode" not in buf.getvalue()


def test_render_characterize_decode_persisted(store, monkeypatch):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 20000,
                                               "decode_context": 25000, "points": []})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(c, "org/Model") == 0
    row = cli.db.get_characterization(store, "mkey", "wmx", "org/Model")
    assert row["decode_context"] == 25000


def test_render_characterize_json_includes_decode_context(monkeypatch, capsys, store):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 9000,
                                               "decode_context": 12000, "points": []})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(c, "org/M", as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert "decode_context" in data and data["decode_context"] == 12000


def test_render_characterize_json_decode_context_none(monkeypatch, capsys, store):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 9000,
                                               "decode_context": None, "points": []})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(c, "org/M", as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert "decode_context" in data and data["decode_context"] is None


def test_render_models_decode_gloss_when_greater(make_console, store, monkeypatch):
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/A", "modality": "text"}])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, e: [{"model_id": "org/A", "safe_context": 16000,
                                             "decode_context": 20000}])
    c, buf = make_console()
    cli.render_models(c)
    out = buf.getvalue()
    assert "20000" in out and "stream-only" in out


def test_render_models_decode_hidden_when_not_greater(make_console, store, monkeypatch):
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/A", "modality": "text"}])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, e: [{"model_id": "org/A", "safe_context": 16000,
                                             "decode_context": 8000}])
    c, buf = make_console()
    cli.render_models(c)
    assert "decode" not in buf.getvalue()


def test_render_models_json_includes_decode_context(monkeypatch, capsys, store):
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/A", "modality": "text"}])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, e: [{"model_id": "org/A", "safe_context": 16000,
                                             "decode_context": 20000}])
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_models(c, as_json=True)
    data = json.loads(capsys.readouterr().out)
    assert data[0].get("decode_context") == 20000


def test_model_detail_per_engine_decode_gloss(make_console, monkeypatch):
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: {"safe_context": 16000, "decode_context": 20000})
    c, buf = make_console()
    assert cli.render_model_detail(c, "org/Smol") == 0
    out = buf.getvalue()
    assert "20000" in out and "stream-only" in out


def test_model_detail_per_engine_decode_hidden_when_not_greater(make_console, monkeypatch):
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: {"safe_context": 16000, "decode_context": 8000})
    c, buf = make_console()
    assert cli.render_model_detail(c, "org/Smol") == 0
    assert "decode" not in buf.getvalue()


def test_model_detail_json_has_decode_context(monkeypatch, capsys):
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: {"safe_context": 9000, "decode_context": 12000})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_model_detail(c, "org/A", as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert data.get("decode_context") == 12000
    assert all(isinstance(v, int) for v in data["engines"].values())


def test_model_detail_json_decode_context_paired_with_best_safe_engine(monkeypatch, capsys):
    # M1: top-level decode_context must come from the same engine that has the highest
    # safe_context — NOT a global max across engines. Here wcx has safe_context=16000/decode=18000
    # and cpu has safe_context=8000/decode=25000. Top-level decode_context must be 18000, not 25000.
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    _per_engine = {"wcx": {"safe_context": 16000, "decode_context": 18000},
                   "cpu": {"safe_context": 8000, "decode_context": 25000}}
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: _per_engine.get(e))
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_model_detail(c, "org/A", as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["safe_context"] == 16000
    assert data["decode_context"] == 18000   # paired with wcx (best safe), not cpu's 25000


def test_emit_characterized_decode_gloss_when_greater(make_console, store, monkeypatch):
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    cli.db.save_characterization(store, "mkey", "wcx", "org/Model",
                                 safe_context=16000, points=[], decode_context=20000)
    c, buf = make_console()
    cli._emit_characterized(c, "wcx")
    out = buf.getvalue()
    assert "20000" in out and "stream-only" in out


def test_emit_characterized_decode_hidden_when_not_greater(make_console, store, monkeypatch):
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    cli.db.save_characterization(store, "mkey", "wcx", "org/Model",
                                 safe_context=16000, points=[], decode_context=8000)
    c, buf = make_console()
    cli._emit_characterized(c, "wcx")
    assert "decode" not in buf.getvalue()


# =========================================================================== #
# Task 7: hardware verbose detail blocks + JSON nesting
# =========================================================================== #

def _apple_cpu() -> CpuInfo:
    """Apple M4 Pro–shaped CpuInfo: no clock/L3; L1+L2 only; no vendor features."""
    return CpuInfo(
        brand="Apple M4 Pro", vendor="Apple", arch_id="arm64",
        physical=12, logical=12,
        base_mhz=None, max_mhz=None,
        l1_kb=192, l2_kb=4096, l3_kb=None,
        features=[],
    )


def _windows_cpu() -> CpuInfo:
    """Ryzen 9 5900X–shaped CpuInfo: full clocks + L2/L3; no features (WMI gap)."""
    return CpuInfo(
        brand="AMD Ryzen 9 5900X 12-Core Processor", vendor="AuthenticAMD", arch_id="AMD64",
        physical=12, logical=24,
        base_mhz=None, max_mhz=3701,
        l1_kb=None, l2_kb=6144, l3_kb=65536,
        features=[],
    )


def _windows_memory() -> MemoryInfo:
    """winbox-shaped MemoryInfo: 4×DDR4 8 GB DIMMs, 4/4 slots used."""
    modules = [
        MemoryModule(slot=f"DIMM_A{i}", capacity_gb=8.0, speed_mts=3400,
                     manufacturer="G-Skill", part_number="F4-3200C14-8GFX")
        for i in range(1, 5)
    ]
    return MemoryInfo(
        total_gb=32.0, available_gb=20.0, swap_gb=0.0,
        kind="DDR4", speed_mts=3400,
        slots_used=4, slots_total=4,
        modules=modules,
    )


def _apple_memory() -> MemoryInfo:
    """Apple-shaped MemoryInfo: kind/speed present but no slots/modules (soldered)."""
    return MemoryInfo(
        total_gb=24.0, available_gb=18.0, swap_gb=0.0,
        kind="LPDDR5", speed_mts=None,
        slots_used=None, slots_total=None,
        modules=[],
    )


def _windows_storage() -> StorageInfo:
    """winbox-shaped StorageInfo: NVMe SSD + HDD."""
    return StorageInfo(
        free_gb=500.0,
        drives=[
            Drive(model="Samsung SSD 990 EVO 1TB", media="nvme-ssd", size_gb=1000.2),
            Drive(model="ST2000DM008-2FR102", media="hdd", size_gb=2000.4),
        ],
    )


def _apple_storage() -> StorageInfo:
    """Apple-shaped StorageInfo: one NVMe drive."""
    return StorageInfo(
        free_gb=200.0,
        drives=[Drive(model="APPLE SSD AP0512Z", media="nvme-ssd", size_gb=500.3)],
    )


def _windows_board() -> BoardInfo:
    """winbox-shaped BoardInfo: ASUS ROG STRIX; system_* → None (custom PC)."""
    return BoardInfo(
        board_vendor="ASUSTeK COMPUTER INC.", board_model="ROG STRIX X470-F GAMING",
        bios_version="6042", bios_date="2022-04-28",
        system_vendor=None, system_model=None,
    )


def _apple_board() -> BoardInfo:
    """Mac-shaped BoardInfo: system_vendor/model from SPHardwareDataType; board_* → None."""
    return BoardInfo(
        board_vendor=None, board_model=None,
        bios_version="13822.81.10", bios_date=None,
        system_vendor="Apple", system_model="MacBook Pro",
    )


# --------------------------------------------------------------------------- #
# verbose CPU block
# --------------------------------------------------------------------------- #

def test_verbose_cpu_detail_apple_silicon(make_console, monkeypatch, stub_pythons):
    """Apple Silicon: vendor shown, no clock/L3; L1+L2 shown; features empty → skipped."""
    stub_pythons(count=1)
    m = _machine(cpu=_apple_cpu(), memory=_apple_memory(), storage=_apple_storage(),
                 board=_apple_board())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "vendor" in out and "Apple" in out
    assert "threads" in out and "12" in out
    assert "L1" in out and "L2" in out
    assert "L3" not in out             # Apple Silicon has no L3
    assert "clocks" not in out         # no clock data on Apple Silicon
    assert "features" not in out       # empty features list → line skipped


def test_verbose_cpu_detail_windows_ryzen(make_console, monkeypatch, stub_pythons):
    """Ryzen: max clock shown; L2+L3 shown; no features (WMI gap) → skipped."""
    stub_pythons(count=1)
    m = _machine(cpu=_windows_cpu(), memory=_windows_memory(), storage=_windows_storage(),
                 board=_windows_board(), chip="AMD Ryzen 9 5900X 12-Core Processor",
                 cpu_physical=12, cpu_logical=24,
                 accel=Accelerator("none", "none detected", None, None),
                 backend="cpu", engine="llama.cpp", engine_ready=False)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "AuthenticAMD" in out        # vendor
    assert "24" in out                  # logical threads
    assert "max 3701 MHz" in out        # max clock
    assert "L2 6144 KB" in out
    assert "L3 65536 KB" in out
    assert "features" not in out        # empty features → skipped


def test_verbose_cpu_detail_with_features(make_console, monkeypatch, stub_pythons):
    """CPU with features list → features line rendered."""
    stub_pythons(count=1)
    cpu = CpuInfo(brand="Intel Core i9", vendor="GenuineIntel", logical=16,
                  features=["AVX-512", "AVX2"])
    m = _machine(cpu=cpu)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "features" in out and "AVX-512" in out and "AVX2" in out


def test_verbose_cpu_detail_base_clock_only(make_console, monkeypatch, stub_pythons):
    """Base clock with no max → shows 'base N MHz' only."""
    stub_pythons(count=1)
    cpu = CpuInfo(brand="FakeCPU", base_mhz=2400)
    m = _machine(cpu=cpu)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "base 2400 MHz" in out
    assert "max" not in out.split("clocks")[-1].split("\n")[0]


def test_verbose_cpu_all_none_no_detail_lines(make_console, monkeypatch, stub_pythons):
    """All CpuInfo fields None → no vendor/threads/clocks/L1/L2/L3/features lines."""
    stub_pythons(count=1)
    m = _machine(cpu=CpuInfo())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    # None of the optional sub-fields should appear (use specific indented labels)
    assert "  vendor" not in out
    assert "  threads" not in out
    assert "  clocks" not in out
    assert "  cache" not in out
    assert "  features" not in out


def test_non_verbose_no_cpu_detail(make_console, monkeypatch, stub_pythons):
    """Non-verbose: CPU detail block not shown."""
    stub_pythons(count=1)
    m = _machine(cpu=_windows_cpu())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=False)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "AuthenticAMD" not in out
    assert "max 3701 MHz" not in out


# --------------------------------------------------------------------------- #
# verbose MEMORY detail block
# --------------------------------------------------------------------------- #

def test_verbose_memory_detail_windows_4modules(make_console, monkeypatch, stub_pythons):
    """Windows 4-DIMM system: kind, speed, slots 4/4, 4 module rows."""
    stub_pythons(count=1)
    m = _machine(memory=_windows_memory(), cpu=_windows_cpu(), storage=_windows_storage(),
                 board=_windows_board())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "DDR4" in out
    assert "3400 MT/s" in out
    assert "4 / 4 used" in out
    assert out.count("module") == 4      # one row per DIMM
    assert "G-Skill" in out
    assert "F4-3200C14-8GFX" in out


def test_verbose_memory_detail_apple_no_modules(make_console, monkeypatch, stub_pythons):
    """Apple Silicon: kind shown, no slots, no modules → '(not reported)' line."""
    stub_pythons(count=1)
    m = _machine(memory=_apple_memory(), cpu=_apple_cpu(), storage=_apple_storage(),
                 board=_apple_board())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "LPDDR5" in out
    assert "not reported on this system" in out
    assert "slots" not in out            # no slot count (soldered; slots_used/total both None)


def test_verbose_memory_kind_none_skipped(make_console, monkeypatch, stub_pythons):
    """kind=None → kind line not rendered."""
    stub_pythons(count=1)
    mem = MemoryInfo(total_gb=16.0, kind=None, modules=[])
    m = _machine(memory=mem)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    assert "kind" not in buf.getvalue()


def test_verbose_memory_partial_slots(make_console, monkeypatch, stub_pythons):
    """slots_used=2 but slots_total=None → renders '2 / ? used'."""
    stub_pythons(count=1)
    mem = MemoryInfo(total_gb=16.0, slots_used=2, slots_total=None, modules=[
        MemoryModule(slot="A1", capacity_gb=8.0, speed_mts=3200),
        MemoryModule(slot="A2", capacity_gb=8.0, speed_mts=3200),
    ])
    m = _machine(memory=mem)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "2 / ? used" in out


def test_non_verbose_no_memory_detail(make_console, monkeypatch, stub_pythons):
    """Non-verbose: memory detail block not shown."""
    stub_pythons(count=1)
    m = _machine(memory=_windows_memory())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=False)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "DDR4" not in out
    assert "DIMM" not in out


# --------------------------------------------------------------------------- #
# verbose STORAGE detail block
# --------------------------------------------------------------------------- #

def test_verbose_storage_detail_drives(make_console, monkeypatch, stub_pythons):
    """Storage drives listed in verbose mode."""
    stub_pythons(count=1)
    m = _machine(storage=_windows_storage(), cpu=_windows_cpu(), memory=_windows_memory(),
                 board=_windows_board())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "Samsung SSD 990 EVO 1TB" in out
    assert "nvme-ssd" in out
    assert "1000 GB" in out
    assert "ST2000DM008-2FR102" in out
    assert "hdd" in out


def test_verbose_storage_drive_none_fields_skipped(make_console, monkeypatch, stub_pythons):
    """Drive with model=None, media=None: only size shown."""
    stub_pythons(count=1)
    storage = StorageInfo(free_gb=100.0, drives=[Drive(model=None, media=None, size_gb=500.0)])
    m = _machine(storage=storage)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "drive" in out
    assert "500 GB" in out


def test_verbose_storage_no_drives(make_console, monkeypatch, stub_pythons):
    """No drives in StorageInfo → storage verbose block is silent (no drive rows)."""
    stub_pythons(count=1)
    storage = StorageInfo(free_gb=200.0, drives=[])
    m = _machine(storage=storage)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    assert "  drive" not in buf.getvalue()


def test_non_verbose_no_storage_detail(make_console, monkeypatch, stub_pythons):
    """Non-verbose: drives not listed."""
    stub_pythons(count=1)
    m = _machine(storage=_windows_storage())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=False)
    cli.render_detect(c)
    assert "Samsung SSD" not in buf.getvalue()


# --------------------------------------------------------------------------- #
# verbose BOARD section
# --------------------------------------------------------------------------- #

def test_verbose_board_windows(make_console, monkeypatch, stub_pythons):
    """Windows board: vendor, model, BIOS; system_* None → those lines skipped."""
    stub_pythons(count=1)
    m = _machine(board=_windows_board(), cpu=_windows_cpu(), memory=_windows_memory(),
                 storage=_windows_storage())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "BOARD" in out
    assert "ASUSTeK COMPUTER INC." in out
    assert "ROG STRIX X470-F GAMING" in out
    assert "board vendorASUSTeK" not in out  # label must not butt against value (pad width)
    assert "6042" in out               # bios version
    assert "2022-04-28" in out         # bios date
    assert "system vendor" not in out  # None → skipped
    assert "system model" not in out   # None → skipped


def test_verbose_board_apple(make_console, monkeypatch, stub_pythons):
    """Mac board: system_vendor/model shown; board_* None → those lines skipped."""
    stub_pythons(count=1)
    m = _machine(board=_apple_board(), cpu=_apple_cpu(), memory=_apple_memory(),
                 storage=_apple_storage())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "BOARD" in out
    assert "13822.81.10" in out        # bios version (firmware)
    assert "system vendor Apple" in out  # padded label + value (no run-together)
    assert "MacBook Pro" in out        # system model
    assert "board vendor" not in out   # None → skipped
    assert "board model" not in out    # None → skipped


def test_board_all_none_no_section(make_console, monkeypatch, stub_pythons):
    """BoardInfo with all fields None → BOARD section not rendered at all."""
    stub_pythons(count=1)
    m = _machine(board=BoardInfo())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    assert "BOARD" not in buf.getvalue()


def test_non_verbose_no_board_section(make_console, monkeypatch, stub_pythons):
    """Non-verbose: BOARD section never shown even with board data."""
    stub_pythons(count=1)
    m = _machine(board=_windows_board())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=False)
    cli.render_detect(c)
    assert "BOARD" not in buf.getvalue()


# --------------------------------------------------------------------------- #
# JSON: nested hardware structures present + all existing keys intact
# --------------------------------------------------------------------------- #

def test_render_detect_json_has_cpu_nested(monkeypatch, capsys):
    """--json includes cpu as a nested dict with all CpuInfo fields."""
    monkeypatch.setattr(cli.detect, "machine", lambda: _machine(cpu=_apple_cpu()))
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_detect(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert "cpu" in payload
    assert payload["cpu"]["brand"] == "Apple M4 Pro"
    assert payload["cpu"]["vendor"] == "Apple"
    assert payload["cpu"]["l3_kb"] is None
    assert payload["cpu"]["features"] == []


def test_render_detect_json_has_memory_nested(monkeypatch, capsys):
    """--json includes memory as a nested dict with modules list."""
    monkeypatch.setattr(cli.detect, "machine", lambda: _machine(memory=_windows_memory()))
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_detect(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert "memory" in payload
    assert payload["memory"]["kind"] == "DDR4"
    assert payload["memory"]["slots_used"] == 4
    mods = payload["memory"]["modules"]
    assert len(mods) == 4
    assert mods[0]["manufacturer"] == "G-Skill"


def test_render_detect_json_has_storage_nested(monkeypatch, capsys):
    """--json includes storage as a nested dict with drives list."""
    monkeypatch.setattr(cli.detect, "machine", lambda: _machine(storage=_windows_storage()))
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_detect(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert "storage" in payload
    drives = payload["storage"]["drives"]
    assert len(drives) == 2
    assert drives[0]["model"] == "Samsung SSD 990 EVO 1TB"
    assert drives[0]["media"] == "nvme-ssd"


def test_render_detect_json_has_board_nested(monkeypatch, capsys):
    """--json includes board as a nested dict."""
    monkeypatch.setattr(cli.detect, "machine", lambda: _machine(board=_windows_board()))
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_detect(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert "board" in payload
    assert payload["board"]["board_vendor"] == "ASUSTeK COMPUTER INC."
    assert payload["board"]["bios_version"] == "6042"
    assert payload["board"]["system_vendor"] is None


def test_render_detect_json_existing_keys_intact(monkeypatch, capsys):
    """--json still has the existing flat keys (chip, backend, accel, etc.)."""
    monkeypatch.setattr(cli.detect, "machine", lambda: _machine(cpu=_apple_cpu()))
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_detect(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    # All pre-existing keys must survive
    for key in ("chip", "backend", "accelerated", "accel", "ram_total_gb",
                "disk_free_gb", "os_version", "arch"):
        assert key in payload, f"missing key: {key}"
    assert payload["chip"] == "Apple M4 Pro"
    assert payload["accelerated"] is True


def test_render_detect_json_apple_memory_no_modules(monkeypatch, capsys):
    """Apple-shaped memory: modules=[] in JSON, no slots info."""
    monkeypatch.setattr(cli.detect, "machine", lambda: _machine(memory=_apple_memory()))
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_detect(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["memory"]["modules"] == []
    assert payload["memory"]["slots_used"] is None
    assert payload["memory"]["kind"] == "LPDDR5"


# --------------------------------------------------------------------------- #
# branch corners: None-field skipping inside loops
# --------------------------------------------------------------------------- #

def test_verbose_memory_module_with_some_none_fields(make_console, monkeypatch, stub_pythons):
    """MemoryModule with mixed None/non-None fields covers all field-skip branches."""
    stub_pythons(count=1)
    # mod1: slot+manufacturer+part_number=None, capacity+speed present
    mod1 = MemoryModule(slot=None, capacity_gb=16.0, speed_mts=4800,
                        manufacturer=None, part_number=None)
    # mod2: capacity_gb=None and speed_mts=None → those append branches skipped
    mod2 = MemoryModule(slot="B1", capacity_gb=None, speed_mts=None,
                        manufacturer="Kingston", part_number="KVR32N22D8/16")
    mem = MemoryInfo(total_gb=32.0, slots_used=2, slots_total=4, modules=[mod1, mod2])
    m = _machine(memory=mem)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert out.count("module") == 2    # both modules rendered
    assert "16 GB" in out and "4800 MT/s" in out
    assert "Kingston" in out and "B1" in out
    # None fields must be absent
    assert "DIMM" not in out
    assert "G-Skill" not in out


def test_verbose_storage_drive_all_none_fields_no_row(make_console, monkeypatch, stub_pythons):
    """Drive with all fields None → parts is empty → no row emitted."""
    stub_pythons(count=1)
    storage = StorageInfo(free_gb=100.0, drives=[Drive(model=None, media=None, size_gb=None)])
    m = _machine(storage=storage)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    # The drive exists but all fields None → no "drive" row
    assert "  drive" not in buf.getvalue()


def test_verbose_board_bios_version_none_bios_date_present(make_console, monkeypatch, stub_pythons):
    """Board with bios_version=None but bios_date present → date shown, version line skipped."""
    stub_pythons(count=1)
    board = BoardInfo(board_vendor="ACME", bios_version=None, bios_date="2023-01-01")
    m = _machine(board=board)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "BOARD" in out
    assert "bios date" in out and "2023-01-01" in out
    assert "bios  " not in out   # bios version line skipped (only "bios date" present)


def test_verbose_memory_module_all_none_then_real(make_console, monkeypatch, stub_pythons):
    """First module all-None (parts empty → skipped), second module has data → row shown.
    Covers the 'if parts' False-branch with more iterations remaining in the loop."""
    stub_pythons(count=1)
    empty_mod = MemoryModule(slot=None, capacity_gb=None, speed_mts=None,
                             manufacturer=None, part_number=None)
    real_mod = MemoryModule(slot="A2", capacity_gb=8.0, speed_mts=3200, manufacturer=None,
                            part_number=None)
    mem = MemoryInfo(total_gb=8.0, slots_used=1, slots_total=2, modules=[empty_mod, real_mod])
    m = _machine(memory=mem)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    # Only one module row (the real one)
    assert out.count("  module") == 1
    assert "A2" in out and "8 GB" in out


def test_verbose_memory_no_modules_but_slots_known_no_not_reported(
        make_console, monkeypatch, stub_pythons):
    """modules=[] but slots_used is not None → elif is False, 'not reported' NOT printed."""
    stub_pythons(count=1)
    # Rare case: slot count known but no per-module data (e.g. Linux non-root dmidecode)
    mem = MemoryInfo(total_gb=32.0, slots_used=2, slots_total=4, modules=[])
    m = _machine(memory=mem)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "2 / 4 used" in out
    assert "not reported on this system" not in out
    assert "module" not in out


# --------------------------------------------------------------------------- #
# Task 7: _det_accelerator — present-vs-usable GPU rendering
# --------------------------------------------------------------------------- #
from ara.hardware import GpuInfo  # noqa: E402 — grouped with hardware imports above


def _fake_machine(**over):
    """Thin wrapper over _machine() for accelerator tests that need gpus=."""
    return _machine(**over)


def test_accelerator_amd_vulkan_present_not_usable(make_console):
    # accel.kind none, but a Vulkan-usable AMD iGPU present → reported, engine-coming hint
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(accel=detect.Accelerator("none", "none detected", None, None),
                      gpus=[GpuInfo(vendor="amd", name="AMD Radeon 780M",
                            vram_gb=4.0, integrated=True,
                            compute_runtime="Vulkan 1.4.318 · radv · coopmat",
                            usable_backend="vulkan")])
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "AMD Radeon 780M" in s
    assert "Vulkan 1.4.318 · radv · coopmat" in s
    assert "engine coming" in s
    assert "no GPU detected" not in s


def test_accelerator_empty_still_says_none(make_console):
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(accel=detect.Accelerator("none", "none detected", None, None), gpus=[])
    cli._det_accelerator(c, m)
    assert "no GPU detected" in out.getvalue()


def test_accelerator_nvidia_rich_block_preserved(make_console):
    # existing NVIDIA rich rendering must be unchanged
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(accel=detect.Accelerator("nvidia", "RTX 2070", 8.0, "CUDA",
                      compute="7.5", cuda_version="13.1", driver_version="591"), gpus=[])
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "RTX 2070" in s and "CUDA 13.1" in s


def test_accelerator_rocm_noted_not_usable(make_console):
    # usable_backend=None, but compute_runtime set (ROCm noted) → "not ARA's path" hint
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(accel=detect.Accelerator("none", "none detected", None, None),
                      gpus=[GpuInfo(vendor="amd", name="RX 6700 XT",
                            vram_gb=12.0, integrated=False,
                            compute_runtime="ROCm 6.1",
                            usable_backend=None)])
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "RX 6700 XT" in s
    assert "not ARA's path" in s
    assert "engine coming" not in s


def test_accelerator_no_runtime_detected(make_console):
    # usable_backend=None, compute_runtime=None → "no usable GPU runtime detected"
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(accel=detect.Accelerator("none", "none detected", None, None),
                      gpus=[GpuInfo(vendor="intel", name="Intel UHD 770",
                            vram_gb=None, integrated=True,
                            compute_runtime=None,
                            usable_backend=None)])
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "Intel UHD 770" in s
    assert "no usable GPU runtime detected" in s


def test_accelerator_cuda_usable_backend_has_engine(make_console):
    # usable_backend="cuda" is in _ARA_ENGINE_BACKENDS → hint says "usable" without "coming"
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(accel=detect.Accelerator("none", "none detected", None, None),
                      gpus=[GpuInfo(vendor="nvidia", name="RTX 3080",
                            vram_gb=10.0, integrated=False,
                            compute_runtime="CUDA 12.4",
                            usable_backend="cuda")])
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "RTX 3080" in s
    assert "usable" in s
    assert "engine coming" not in s


def test_accelerator_nvidia_with_other_gpu_not_double_printed(make_console):
    # accel.kind=nvidia + gpus has the same nvidia GPU + an amd GPU
    # nvidia GPU should NOT be double-printed; the amd GPU SHOULD be shown
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(
        accel=detect.Accelerator("nvidia", "RTX 4090", 24.0, "CUDA", count=1,
                                  compute="8.9", cuda_version="12.4", driver_version="550"),
        gpus=[
            GpuInfo(vendor="nvidia", name="RTX 4090", vram_gb=24.0),   # same vendor → skip
            GpuInfo(vendor="amd", name="Radeon Pro 580", vram_gb=8.0,
                    compute_runtime=None, usable_backend=None),          # other → show
        ],
    )
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "RTX 4090" in s            # shown once via the rich block
    assert "Radeon Pro 580" in s      # other GPU shown
    # The RTX 4090 must not appear twice (rich block + gpu_line)
    assert s.count("RTX 4090") == 1


def test_accelerator_apple_rich_block_preserved(make_console):
    # Apple GPU: Metal line unchanged
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(accel=detect.Accelerator("apple", "Apple M4 Pro GPU", None, "Metal",
                                               cores=16), gpus=[])
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "Apple M4 Pro GPU" in s and "Metal" in s


def test_accelerator_gpu_vram_none_omitted(make_console):
    # GpuInfo with vram_gb=None → VRAM token absent from the line
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(accel=detect.Accelerator("none", "none detected", None, None),
                      gpus=[GpuInfo(vendor="intel", name="Intel Arc A770",
                            vram_gb=None, integrated=False,
                            compute_runtime=None, usable_backend=None)])
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "Intel Arc A770" in s
    assert "GB" not in s


def test_accelerator_apple_with_other_gpu_shows_it(make_console):
    # accel.kind=apple + gpus has the apple GPU + a non-apple GPU → non-apple shown
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(
        accel=detect.Accelerator("apple", "Apple M4 Pro GPU", None, "Metal", cores=16),
        gpus=[
            GpuInfo(vendor="apple", name="Apple M4 Pro GPU"),   # same vendor → skip
            GpuInfo(vendor="intel", name="Thunderbolt eGPU", vram_gb=8.0,
                    compute_runtime=None, usable_backend=None),  # other → show
        ],
    )
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "Apple M4 Pro GPU" in s and "Metal" in s
    assert "Thunderbolt eGPU" in s


def test_render_detect_json_has_gpus_key(monkeypatch, capsys):
    # --json must emit a 'gpus' key (Task 6 plumbing, confirmed via asdict)
    monkeypatch.setattr(cli.detect, "machine",
                        lambda: _machine(gpus=[GpuInfo(vendor="amd", name="Radeon 780M",
                                                        vram_gb=4.0, integrated=True)]))
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_detect(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert "gpus" in payload
    assert payload["gpus"][0]["vendor"] == "amd"
