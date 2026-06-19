"""cli.py — formatters, arg parsing/dispatch, and the render_* surfaces."""
from __future__ import annotations

import json
import sys
import types

import pytest

import ara.cli as cli
from ara.detect import Accelerator, Machine, ModelStore, Runtime


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
    (2.5, "~2.5 GB"),
])
def test_fmt_size(gb, out):
    assert cli._fmt_size(gb) == out


@pytest.mark.parametrize("secs,out", [
    (5, "5s"), (59, "59s"), (90, "1m"), (3600, "1h"), (90000, "1d"),
])
def test_fmt_uptime(secs, out):
    assert cli._fmt_uptime(secs) == out


@pytest.mark.parametrize("gb,out", [
    (0.5, "512 MB"),   # binary MB under a gigabyte
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
    monkeypatch.setattr(cli, "render_detect", lambda c, as_json=False: rec.update(detect=as_json))
    monkeypatch.setattr(cli, "render_status", lambda c, as_json=False: rec.update(status=as_json))
    monkeypatch.setattr(cli, "render_profile",
                        lambda c, **kw: (rec.update(profile=kw) or 0))
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
                        lambda c, as_json=False: captured.update(verbose=c.verbose))
    _run_main(monkeypatch, ["detect", "--verbose"])
    assert captured["verbose"] is True


# --------------------------------------------------------------------------- #
# render_landing
# --------------------------------------------------------------------------- #
def test_render_landing_supported(make_console, monkeypatch):
    monkeypatch.setattr(cli.detect, "chip_name", lambda: "Apple M4 Pro")
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli, "engine_status", lambda: (True, "wmx-suite"))
    c, buf = make_console()
    cli.render_landing(c)
    out = buf.getvalue()
    assert "ara" in out and "Apple M4 Pro" in out
    assert "GETTING STARTED" in out
    assert "detect" in out and "status" in out and "profile" in out
    assert "no supported backend" not in out


def test_render_landing_unsupported_warns(make_console, monkeypatch):
    monkeypatch.setattr(cli.detect, "chip_name", lambda: "Intel i7")
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "unsupported")
    monkeypatch.setattr(cli, "engine_status", lambda: (False, "unsupported"))
    c, buf = make_console()
    cli.render_landing(c)
    assert "no supported backend" in buf.getvalue()


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
        runtimes=[Runtime("MLX", True, "0.18", accels=("apple",), usable=True),
                  Runtime("vLLM", True, "0.5", accels=("nvidia",), usable=False)],
        model_stores=[ModelStore("HF cache", True, 3, 12.0),
                      ModelStore("Ollama", True, 0, 0.0)],
        hf_token=True, power="AC power", backend="apple", engine="wmx-suite",
        engine_ready=False,
    )
    base.update(over)
    return Machine(**base)


def test_render_detect_text(make_console, monkeypatch):
    monkeypatch.setattr(cli.detect, "profile", lambda: _machine())
    c, buf = make_console()
    cli.render_detect(c)
    out = buf.getvalue()
    for section in ("SYSTEM", "MEMORY", "ACCELERATOR", "STORAGE", "RUNTIMES", "MODELS", "ARA"):
        assert section in out
    assert "Apple M4 Pro" in out
    assert "Metal" in out
    assert "MLX" in out
    assert "needs CUDA" in out          # vLLM unusable reason rendered
    assert "3 models" in out


def test_render_detect_nvidia_accel(make_console, monkeypatch):
    m = _machine(accel=Accelerator("nvidia", "RTX 4090", 24.0, "CUDA", count=1,
                                   compute="8.9", cuda_version="550"))
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
    c, buf = make_console()
    cli.render_detect(c)
    out = buf.getvalue()
    assert "RTX 4090" in out and "24 GB VRAM" in out and "SM 8.9" in out


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


def test_render_status_with_processes(make_console, monkeypatch):
    monkeypatch.setattr(cli.status, "scan", lambda: [_proc(), _proc(pid=5, label="vLLM", rss_gb=4.0)])
    c, buf = make_console()
    cli.render_status(c)
    out = buf.getvalue()
    assert "RUNNING AI/ML" in out
    assert "Ollama" in out and "vLLM" in out
    assert "pid 1234" in out and ":11434" in out
    assert "total" in out and "2 processes" in out


def test_render_status_empty(make_console, monkeypatch):
    monkeypatch.setattr(cli.status, "scan", lambda: [])
    c, buf = make_console()
    cli.render_status(c)
    assert "nothing running right now" in buf.getvalue()


def test_render_status_json(monkeypatch, capsys):
    monkeypatch.setattr(cli.status, "scan", lambda: [_proc()])
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_status(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["label"] == "Ollama" and payload[0]["port"] == 11434


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
    monkeypatch.setattr(cli, "engine_status", lambda: (engine_ok, "wmx-suite"))
    monkeypatch.setattr(cli, "get_backend", lambda: bk)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: isatty))


def test_profile_unsupported_backend(make_console, monkeypatch, set_platform):
    set_platform("Linux", "x86_64")
    c, buf = make_console()
    assert cli.render_profile(c) == 1
    assert "profiling needs an ARA backend" in buf.getvalue()


def test_profile_engine_not_installed(make_console, monkeypatch, set_platform):
    _wire_profile(monkeypatch, set_platform, FakeBackend(_limits()), engine_ok=False)
    c, buf = make_console()
    assert cli.render_profile(c) == 1
    assert "isn't installed" in buf.getvalue()


def test_profile_safe_limits_error(make_console, monkeypatch, set_platform):
    bk = FakeBackend(_limits())
    bk.safe_limits_exc = RuntimeError("sysctl exploded")
    _wire_profile(monkeypatch, set_platform, bk)
    c, buf = make_console()
    assert cli.render_profile(c) == 1
    assert "couldn't read limits" in buf.getvalue()


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
# render_detect verbose + unsupported branches
# --------------------------------------------------------------------------- #
def test_render_detect_verbose_and_unsupported(monkeypatch, make_console):
    # `supported` is a property, not a ctor arg — drive it via backend="unsupported".
    m = _machine(
        backend="unsupported", engine="unsupported", engine_ready=False,
        cpu_logical=24, hf_token=False,
        accel=Accelerator("none", "none detected", None, None),
        runtimes=[Runtime("Ollama", False, None)],
        model_stores=[ModelStore("HF cache", False)],
    )
    monkeypatch.setattr(cli.detect, "profile", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "physical" in out and "logical" in out      # verbose cpu line
    assert "no GPU detected" in out                     # accel none branch
    assert "not found" in out                           # verbose shows absent runtime/store
    assert "no ARA backend for this hardware yet" in out  # unsupported footer


def test_render_status_gpu_and_no_port(make_console, monkeypatch):
    monkeypatch.setattr(cli.status, "scan",
                        lambda: [_proc(gpu_mb=8192.0, port=None, detail=None)])
    c, buf = make_console()
    cli.render_status(c)
    out = buf.getvalue()
    assert "8192 MB GPU" in out
    assert ":" not in out.split("Ollama")[-1].split("\n")[0].replace("pid", "")


def test_profile_through_real_apple_backend(make_console, monkeypatch, set_platform, fake_wmx):
    set_platform("Darwin", "arm64")
    monkeypatch.setattr(cli, "engine_status", lambda: (True, "wmx-suite"))
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    fake_wmx.profile = None                 # uncalibrated → proceeds to calibrate
    fake_wmx.describe_return = {"id": "m"}   # model already cached → no download
    c, buf = make_console()
    rc = cli.render_profile(c, assume_yes=True)
    out = buf.getvalue()
    assert rc == 0
    assert "SAFE LIMITS" in out
    assert "calibrated." in out
    assert fake_wmx.calibrate_calls  # real apple.calibrate hit the fake probe
