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


def test_main_profile_flags(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["profile", "--recalibrate", "--yes", "--json"])
    kw = rec["profile"]
    assert kw["recalibrate"] is True and kw["assume_yes"] is True and kw["as_json"] is True


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
    monkeypatch.setattr(cli.detect, "profile", lambda: _machine())
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
    c, buf = make_console(verbose=False)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "12 cores" in out                  # cpu row emitted, no features tail
    assert "your default python3" not in out  # python row skipped entirely
    assert "ARA's python" not in out


def test_render_detect_json(monkeypatch, capsys):
    monkeypatch.setattr(cli.detect, "profile", lambda: _machine())
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

    def calibrate(self, model):
        if self.calibrate_exc:
            raise self.calibrate_exc
        return self.calibrate_result


def _wire_profile(monkeypatch, set_platform, bk, *, engine_ok=True, isatty=False):
    set_platform("Darwin", "arm64")  # backend_name() -> "apple"
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (engine_ok, "wmx-suite"))
    monkeypatch.setattr(cli, "get_backend", lambda b=None: bk)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: isatty))


def test_profile_engine_not_installed(make_console, monkeypatch, set_platform):
    _wire_profile(monkeypatch, set_platform, FakeBackend(_limits()), engine_ok=False)
    c, buf = make_console()
    assert cli.render_profile(c) == 1            # no --engine → report the gap
    out = buf.getvalue()
    assert "isn't installed" in out
    assert "ara install" in out      # the engine is installed on demand now,
    assert "uv sync" not in out      # not pulled in by `uv sync`


def test_profile_engine_flag_installs_then_asks_rerun(make_console, monkeypatch, set_platform):
    _wire_profile(monkeypatch, set_platform, FakeBackend(_limits()), engine_ok=False)

    def fake_install(c, *, engine, as_json=False):
        c.emit("  installed wmx-suite")
        return 0   # freshly installed — can't import it in THIS process

    monkeypatch.setattr(cli, "render_install", fake_install)
    c, buf = make_console()
    assert cli.render_profile(c, engine="wmx") == 1   # not measured this run
    out = buf.getvalue()
    assert "installed wmx-suite" in out
    assert "re-run ara profile" in out


def test_profile_engine_flag_install_fails_no_rerun(make_console, monkeypatch, set_platform):
    _wire_profile(monkeypatch, set_platform, FakeBackend(_limits()), engine_ok=False)
    monkeypatch.setattr(cli, "render_install", lambda c, *, engine, as_json=False: 1)
    c, buf = make_console()
    assert cli.render_profile(c, engine="wcx") == 1
    assert "re-run ara profile" not in buf.getvalue()


def test_main_profile_passes_engine(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["profile", "--engine", "wmx"])
    assert rec["profile"]["engine"] == "wmx"


def test_profile_unknown_engine_errors(make_console, monkeypatch):
    c, buf = make_console()
    assert cli.render_profile(c, engine="bogus") == 1
    assert "unknown engine" in buf.getvalue().lower()


# --- persistence wiring: overlay a stored calibration / persist a fresh one ---
def test_overlay_stored_calibration_applies(store, monkeypatch):
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
    cli.profiles.save_calibration(store, "wmx", fixed_overhead_gb=5.5,
                                  calibrated_at="2026-06-18T09:30:00Z")
    m = {"overhead_gb": None, "calibrated": False, "calibrated_at": None}
    cli._overlay_stored_calibration(m, "wmx")
    assert m["overhead_gb"] == 5.5 and m["calibrated"] is True
    assert m["calibrated_at"] == "2026-06-18"


def test_overlay_no_stored_leaves_limits(store, monkeypatch):
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
    m = {"overhead_gb": None, "calibrated": False, "calibrated_at": None}
    cli._overlay_stored_calibration(m, "wmx")     # nothing stored
    assert m["calibrated"] is False


def test_overlay_none_engine_key_is_noop():
    m = {"overhead_gb": None, "calibrated": False}
    cli._overlay_stored_calibration(m, None)
    assert m["calibrated"] is False


def test_persist_saves_overhead(store, monkeypatch):
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
    cli._persist_calibration({"overhead_gb": 1.2}, "wmx")
    assert cli.profiles.get_calibration(store, "wmx")["fixed_overhead_gb"] == 1.2


def test_persist_saves_characterization_and_catalogs(store, monkeypatch):
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
    seen = {}
    monkeypatch.setattr(cli.catalog, "remember", lambda con, mid: seen.setdefault("model", mid))
    m = {"overhead_gb": None,
         "characterization": {"model": "smol", "safe_context": 16000, "points": [[512, 1.4]]}}
    cli._persist_calibration(m, "wcx")
    row = cli.db.get_characterization(store, "mkey", "wcx", "smol")
    assert row["safe_context"] == 16000 and seen["model"] == "smol"


def test_persist_none_engine_key_is_noop(store):
    cli._persist_calibration({"overhead_gb": 1.0}, None)
    assert cli.db.get_machine(store, "any", "wmx") is None


def test_emit_characterized_shows_stored_models(make_console, store, monkeypatch):
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
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
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
    c, buf = make_console()
    cli._emit_characterized(c, "wcx")
    assert buf.getvalue() == ""


def test_emit_characterized_none_engine_key(make_console):
    c, buf = make_console()
    cli._emit_characterized(c, None)
    assert buf.getvalue() == ""


def test_profile_safe_limits_error(make_console, monkeypatch, set_platform):
    bk = FakeBackend(_limits())
    bk.safe_limits_exc = RuntimeError("sysctl exploded")
    _wire_profile(monkeypatch, set_platform, bk)
    c, buf = make_console()
    assert cli.render_profile(c) == 1
    assert "couldn't read limits" in buf.getvalue()


def test_profile_safe_limits_error_json(make_console, monkeypatch, set_platform, capsys):
    # --json must emit a JSON error, not human text, when limits can't be read
    import json as _json
    bk = FakeBackend(_limits())
    bk.safe_limits_exc = RuntimeError("sysctl exploded")
    _wire_profile(monkeypatch, set_platform, bk)
    c, _ = make_console()
    assert cli.render_profile(c, as_json=True) == 1
    assert "error" in _json.loads(capsys.readouterr().out.strip())


def test_profile_json(monkeypatch, set_platform, capsys):
    bk = FakeBackend(_limits(calibrated=True))
    _wire_profile(monkeypatch, set_platform, bk)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_profile(c, as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["safe_budget_gb"] == 36.0


def test_profile_cached_early_return(make_console, monkeypatch, set_platform):
    bk = FakeBackend(_limits(calibrated=True))
    _wire_profile(monkeypatch, set_platform, bk)
    c, buf = make_console()
    assert cli.render_profile(c) == 0
    out = buf.getvalue()
    assert "cached" in out and "--recalibrate" in out


def test_profile_calibrated_without_overhead_skips_recalibrate_hint(
        make_console, monkeypatch, set_platform):
    # The cuda case: the VRAM wall is exact (calibrated) but there's no measured
    # cold-start overhead to redo — so no "recalibrate" hint, just the limits.
    bk = FakeBackend(_limits(calibrated=True, overhead_gb=None))
    _wire_profile(monkeypatch, set_platform, bk)
    c, buf = make_console()
    assert cli.render_profile(c) == 0
    out = buf.getvalue()
    assert "SAFE LIMITS" in out
    assert "--recalibrate" not in out


def test_profile_non_interactive_estimated(make_console, monkeypatch, set_platform):
    bk = FakeBackend(_limits(calibrated=False))
    _wire_profile(monkeypatch, set_platform, bk, isatty=False)
    c, buf = make_console()
    assert cli.render_profile(c) == 0
    assert "estimated" in buf.getvalue()


def test_profile_insufficient_disk(make_console, monkeypatch, set_platform):
    bk = FakeBackend(_limits(calibrated=False), cached=False)
    _wire_profile(monkeypatch, set_platform, bk)
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: 10.0)
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: 5.0)
    c, buf = make_console()
    assert cli.render_profile(c, assume_yes=True) == 1
    assert "not enough disk" in buf.getvalue()


def test_profile_disk_exactly_at_threshold_proceeds(make_console, monkeypatch, set_platform):
    # boundary: free == size + buffer is NOT "insufficient" (the check is strict <).
    bk = FakeBackend(_limits(calibrated=False), cached=False)
    _wire_profile(monkeypatch, set_platform, bk)
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: 10.0)
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: 10.0 + cli.acquire.DISK_BUFFER_GB)
    c, buf = make_console()
    assert cli.render_profile(c, assume_yes=True) == 0   # proceeds to calibrate, no disk error
    assert "not enough disk" not in buf.getvalue()


def test_profile_confirm_declined(make_console, monkeypatch, set_platform):
    bk = FakeBackend(_limits(calibrated=False), cached=True)
    _wire_profile(monkeypatch, set_platform, bk, isatty=True)
    monkeypatch.setattr(cli, "_confirm", lambda q: False)
    c, buf = make_console()
    assert cli.render_profile(c) == 0
    assert "skipped" in buf.getvalue()


def test_profile_calibrate_success(make_console, monkeypatch, set_platform):
    bk = FakeBackend(_limits(calibrated=False), cached=True)
    _wire_profile(monkeypatch, set_platform, bk)
    c, buf = make_console()
    assert cli.render_profile(c, assume_yes=True) == 0
    out = buf.getvalue()
    assert "calibrated." in out
    assert "overhead" in out  # _emit_calibration line rendered


def test_profile_calibrate_failure(make_console, monkeypatch, set_platform):
    bk = FakeBackend(_limits(calibrated=False), cached=True)
    bk.calibrate_exc = RuntimeError("OOM during ramp")
    _wire_profile(monkeypatch, set_platform, bk)
    c, buf = make_console()
    assert cli.render_profile(c, assume_yes=True) == 1
    assert "calibration failed" in buf.getvalue()


def test_profile_explicit_model_bypasses_cache(make_console, monkeypatch, set_platform):
    # Calibrated already, but naming --model forces a re-measure path.
    bk = FakeBackend(_limits(calibrated=True), cached=True)
    _wire_profile(monkeypatch, set_platform, bk)
    c, buf = make_console()
    assert cli.render_profile(c, assume_yes=True, model="org/other") == 0
    assert "calibrated." in buf.getvalue()
    assert "cached" not in buf.getvalue()  # did NOT take the early return


def test_profile_downloads_when_not_cached(make_console, monkeypatch, set_platform):
    bk = FakeBackend(_limits(calibrated=False), cached=False)
    _wire_profile(monkeypatch, set_platform, bk)
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: 0.1)
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: 500.0)
    c, buf = make_console()
    assert cli.render_profile(c, assume_yes=True) == 0
    assert bk.downloaded == ["org/calib"]  # fetched before calibrating
    assert "calibrated." in buf.getvalue()


# --------------------------------------------------------------------------- #
# render_profile end-to-end through the REAL Apple backend on the fake engine
# --------------------------------------------------------------------------- #
def test_profile_calibrate_clean_systemexit(make_console, monkeypatch, set_platform):
    # The engine may sys.exit() after printing its own clean reason → rc 1, no traceback.
    bk = FakeBackend(_limits(calibrated=False), cached=True)
    bk.calibrate_exc = SystemExit(1)
    _wire_profile(monkeypatch, set_platform, bk)
    c, buf = make_console()
    assert cli.render_profile(c, assume_yes=True) == 1


# --------------------------------------------------------------------------- #
# _emit_limits / _emit_calibration helpers
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


@pytest.mark.parametrize("measured,default,phrase", [
    (5.0, 6.0, "lean"),                 # under the default → keep default
    (8.0, 6.0, "more conservative"),    # over the default → tighten
    (6.0, 6.0, "matching the default"), # equal
])
def test_emit_calibration_verdicts(make_console, measured, default, phrase):
    c, buf = make_console()
    m = {"calibration": {"measured_overhead_gb": measured, "default_overhead_gb": default,
                         "n_points": 4, "hf_id": "org/calib-model"}}
    cli._emit_calibration(c, m, "org/calib-model")
    assert phrase in buf.getvalue()


def test_emit_calibration_silent_without_measurements(make_console):
    c, buf = make_console()
    cli._emit_calibration(c, {"calibration": {}}, "org/calib")
    assert buf.getvalue() == ""


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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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


def test_profile_through_real_apple_backend(make_console, monkeypatch, set_platform):
    from ara.backends import apple
    set_platform("Darwin", "arm64")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "wmx-suite"))
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(apple, "calibration_model_cached", lambda *a: True)  # no download
    facts = {"device": "Apple M4 Pro", "total_gb": 48.0, "wall_gb": 40.0,
             "safe_budget_gb": 36.0, "margin_gb": 4.0, "headroom_gb": 28.0,
             "swap_free_gb": 2.0}
    calls = []

    def worker(name, argv):
        calls.append(argv[2])
        return {"measured_overhead_gb": 5.0, "default_overhead_gb": 6.0,
                "n_points": 4} if argv[2] == "calibrate" else dict(facts)

    monkeypatch.setattr(apple, "engine_env",
                        type("E", (), {"run_worker": staticmethod(worker)}))
    c, buf = make_console()
    rc = cli.render_profile(c, assume_yes=True)
    out = buf.getvalue()
    assert rc == 0
    assert "SAFE LIMITS" in out
    assert "calibrated." in out
    assert "calibrate" in calls   # real apple.calibrate drove the device worker


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
    monkeypatch.setattr(cli, "Path", types.SimpleNamespace(home=lambda: r"C:\Users\Will"))
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
    monkeypatch.setattr(cli.detect, "profile", lambda: _machine())
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
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
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
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
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
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
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
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
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
    it — the willw11 case (GPU 3500 vs CPU 8192 for one model → 8192, cpu)."""
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/L", "modality": "text"}])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")     # default engine = wcx
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
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
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
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
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
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
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
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
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
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
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
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
    # willw11's case: a GPU is detected (cuda), but `--engine cpu` must run on the CPU backend
    # and store under the cpu engine key — never silently fall through to the detected GPU.
    seen = {}
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli, "engine_status",
                        lambda b=None: (seen.update(status_backend=b) or (True, "llama.cpp")))
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
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
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    monkeypatch.setattr(cli, "get_backend", lambda b=None: bk)
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: size_gb)
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: free_gb)


def _fake_bk_characterize(model):
    return {"model": model, "safe_context": 16000, "decode_context": None, "points": [[1024, 1.2]]}


def test_render_characterize_prefetch_uncached_transformers(make_console, store, monkeypatch):
    # Uncached transformers model on a compatible engine → download fired, then characterize runs.
    bk = FakeBackend(_limits(), cached=False)
    bk.calibrate_result = None  # not used by characterize path
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
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
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
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
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
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
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
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: {"safe_context": 16000, "decode_context": 20000})
    c, buf = make_console()
    assert cli.render_model_detail(c, "org/Smol") == 0
    out = buf.getvalue()
    assert "20000" in out and "stream-only" in out


def test_model_detail_per_engine_decode_hidden_when_not_greater(make_console, monkeypatch):
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: {"safe_context": 16000, "decode_context": 8000})
    c, buf = make_console()
    assert cli.render_model_detail(c, "org/Smol") == 0
    assert "decode" not in buf.getvalue()


def test_model_detail_json_has_decode_context(monkeypatch, capsys):
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
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
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
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
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
    cli.db.save_characterization(store, "mkey", "wcx", "org/Model",
                                 safe_context=16000, points=[], decode_context=20000)
    c, buf = make_console()
    cli._emit_characterized(c, "wcx")
    out = buf.getvalue()
    assert "20000" in out and "stream-only" in out


def test_emit_characterized_decode_hidden_when_not_greater(make_console, store, monkeypatch):
    monkeypatch.setattr(cli.profiles, "machine_key", lambda: "mkey")
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
    """willw11-shaped MemoryInfo: 4×DDR4 8 GB DIMMs, 4/4 slots used."""
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
    """willw11-shaped StorageInfo: NVMe SSD + HDD."""
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
    """willw11-shaped BoardInfo: ASUS ROG STRIX; system_* → None (custom PC)."""
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "features" in out and "AVX-512" in out and "AVX2" in out


def test_verbose_cpu_detail_base_clock_only(make_console, monkeypatch, stub_pythons):
    """Base clock with no max → shows 'base N MHz' only."""
    stub_pythons(count=1)
    cpu = CpuInfo(brand="FakeCPU", base_mhz=2400)
    m = _machine(cpu=cpu)
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "base 2400 MHz" in out
    assert "max" not in out.split("clocks")[-1].split("\n")[0]


def test_verbose_cpu_all_none_no_detail_lines(make_console, monkeypatch, stub_pythons):
    """All CpuInfo fields None → no vendor/threads/clocks/L1/L2/L3/features lines."""
    stub_pythons(count=1)
    m = _machine(cpu=CpuInfo())
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "2 / ? used" in out


def test_non_verbose_no_memory_detail(make_console, monkeypatch, stub_pythons):
    """Non-verbose: memory detail block not shown."""
    stub_pythons(count=1)
    m = _machine(memory=_windows_memory())
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    assert "  drive" not in buf.getvalue()


def test_non_verbose_no_storage_detail(make_console, monkeypatch, stub_pythons):
    """Non-verbose: drives not listed."""
    stub_pythons(count=1)
    m = _machine(storage=_windows_storage())
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    assert "BOARD" not in buf.getvalue()


def test_non_verbose_no_board_section(make_console, monkeypatch, stub_pythons):
    """Non-verbose: BOARD section never shown even with board data."""
    stub_pythons(count=1)
    m = _machine(board=_windows_board())
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
    c, buf = make_console(verbose=False)
    cli.render_detect(c)
    assert "BOARD" not in buf.getvalue()


# --------------------------------------------------------------------------- #
# JSON: nested hardware structures present + all existing keys intact
# --------------------------------------------------------------------------- #

def test_render_detect_json_has_cpu_nested(monkeypatch, capsys):
    """--json includes cpu as a nested dict with all CpuInfo fields."""
    monkeypatch.setattr(cli.detect, "profile", lambda: _machine(cpu=_apple_cpu()))
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
    monkeypatch.setattr(cli.detect, "profile", lambda: _machine(memory=_windows_memory()))
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
    monkeypatch.setattr(cli.detect, "profile", lambda: _machine(storage=_windows_storage()))
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
    monkeypatch.setattr(cli.detect, "profile", lambda: _machine(board=_windows_board()))
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_detect(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert "board" in payload
    assert payload["board"]["board_vendor"] == "ASUSTeK COMPUTER INC."
    assert payload["board"]["bios_version"] == "6042"
    assert payload["board"]["system_vendor"] is None


def test_render_detect_json_existing_keys_intact(monkeypatch, capsys):
    """--json still has the existing flat keys (chip, backend, accel, etc.)."""
    monkeypatch.setattr(cli.detect, "profile", lambda: _machine(cpu=_apple_cpu()))
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
    monkeypatch.setattr(cli.detect, "profile", lambda: _machine(memory=_apple_memory()))
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    # The drive exists but all fields None → no "drive" row
    assert "  drive" not in buf.getvalue()


def test_verbose_board_bios_version_none_bios_date_present(make_console, monkeypatch, stub_pythons):
    """Board with bios_version=None but bios_date present → date shown, version line skipped."""
    stub_pythons(count=1)
    board = BoardInfo(board_vendor="ACME", bios_version=None, bios_date="2023-01-01")
    m = _machine(board=board)
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
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
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "2 / 4 used" in out
    assert "not reported on this system" not in out
    assert "module" not in out
