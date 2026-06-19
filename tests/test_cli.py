"""cli.py — formatters, arg parsing/dispatch, and the render_* surfaces."""
from __future__ import annotations

import json
import sys
import types

import pytest

import ara.cli as cli
from ara.detect import Accelerator, Machine, ModelStore, Runtime


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
# render_detect verbose + unsupported branches
# --------------------------------------------------------------------------- #
def test_render_detect_verbose_and_unsupported(monkeypatch, make_console, stub_pythons):
    # `supported` is a property, not a ctor arg — drive it via backend="unsupported".
    stub_pythons(count=1, discover=[])
    m = _machine(
        backend="unsupported", engine="unsupported", engine_ready=False,
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
    assert "no GPU detected" in out                     # accel none branch
    # verbose lists absent engine and absent store as "not found"
    assert out.count("not found") >= 2
    assert "no ARA backend for this hardware yet" in out  # unsupported footer


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
    monkeypatch.setattr(cli.status, "scan", lambda: [])
    c, buf = make_console()
    cli.render_status(c, want=lambda k: k != "processes")   # filtered out → early return
    assert buf.getvalue() == ""


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
