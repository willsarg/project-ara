"""engines.py — the engine catalog + hardware-matched resolution (`--engine auto`)."""
from __future__ import annotations

import ara.engines as engines


# --------------------------------------------------------------------------- #
# for_hardware() — the light "what would ARA pick here?" probe behind `auto`
# --------------------------------------------------------------------------- #
def test_for_hardware_picks_wmx_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(engines.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(engines.platform, "machine", lambda: "arm64")
    assert engines.for_hardware() == "wmx"


def test_for_hardware_picks_wcx_when_nvidia_smi_present(monkeypatch):
    monkeypatch.setattr(engines.platform, "system", lambda: "Windows")
    monkeypatch.setattr(engines.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(engines.shutil, "which",
                        lambda n: "C:/Windows/System32/nvidia-smi.exe" if n == "nvidia-smi" else None)
    assert engines.for_hardware() == "wcx"


def test_for_hardware_none_when_no_known_accelerator(monkeypatch):
    monkeypatch.setattr(engines.platform, "system", lambda: "Linux")
    monkeypatch.setattr(engines.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(engines.shutil, "which", lambda n: None)
    assert engines.for_hardware() is None


# --------------------------------------------------------------------------- #
# resolve() — map an --engine value (wmx | wcx | auto) to a concrete engine key
# --------------------------------------------------------------------------- #
def test_resolve_passes_through_explicit_engine():
    assert engines.resolve("wmx") == "wmx"
    assert engines.resolve("wcx") == "wcx"


def test_resolve_auto_uses_hardware_pick(monkeypatch):
    monkeypatch.setattr(engines, "for_hardware", lambda: "wcx")
    assert engines.resolve("auto") == "wcx"


def test_resolve_auto_none_when_no_match(monkeypatch):
    monkeypatch.setattr(engines, "for_hardware", lambda: None)
    assert engines.resolve("auto") is None


def test_resolve_unknown_is_none():
    assert engines.resolve("nonsense") is None


# --------------------------------------------------------------------------- #
# is_installed() — is the engine's module importable? (cheap, no import)
# --------------------------------------------------------------------------- #
def test_is_installed_true_when_module_present(monkeypatch):
    monkeypatch.setattr(engines, "find_spec",
                        lambda m: object() if m == "wmx_suite" else None)
    assert engines.is_installed("wmx") is True


def test_is_installed_false_when_module_absent(monkeypatch):
    monkeypatch.setattr(engines, "find_spec", lambda m: None)
    assert engines.is_installed("wmx") is False


def test_is_installed_false_for_unknown_engine():
    assert engines.is_installed("nonsense") is False


# --------------------------------------------------------------------------- #
# source_for() — the install source, with a dev env-var override
# --------------------------------------------------------------------------- #
def test_source_for_defaults_to_git_spec(monkeypatch):
    monkeypatch.delenv("ARA_WMX_SOURCE", raising=False)
    assert engines.source_for("wmx") == "git+https://github.com/willsarg/wmx-suite"


def test_source_for_uses_env_override(monkeypatch):
    monkeypatch.setenv("ARA_WMX_SOURCE", "../wmx-suite")
    assert engines.source_for("wmx") == "../wmx-suite"


# --------------------------------------------------------------------------- #
# install() — orchestration around `uv pip install` (subprocess injected)
# --------------------------------------------------------------------------- #
def test_install_unknown_engine_reports_unknown():
    r = engines.install("nonsense")
    assert r.status == "unknown"


def test_install_unavailable_engine_is_coming_soon(monkeypatch):
    monkeypatch.setitem(engines.ENGINES["wcx"], "available", False)   # force coming-soon
    called = []
    monkeypatch.setattr(engines, "_run_pip", lambda args: called.append(args) or (0, ""))
    r = engines.install("wcx")
    assert r.status == "coming_soon"
    assert called == []   # never shelled out for a not-yet-available engine


def test_install_wcx_uses_cuda_extra_and_torch_backend(monkeypatch):
    monkeypatch.setattr(engines, "is_installed", lambda k: False)
    monkeypatch.delenv("ARA_WCX_SOURCE", raising=False)
    seen = {}

    def fake_pip(args):
        seen["args"] = args
        return 0, "ok"

    monkeypatch.setattr(engines, "_run_pip", fake_pip)
    engines.install("wcx")
    assert seen["args"] == ["install", "--torch-backend=auto",
                            "wcx-suite[cuda] @ git+https://github.com/willsarg/wcx-suite"]


def test_install_wcx_local_source_is_editable_with_extra(monkeypatch):
    monkeypatch.setattr(engines, "is_installed", lambda k: False)
    monkeypatch.setenv("ARA_WCX_SOURCE", "../wcx-suite")
    seen = {}

    def fake_pip(args):
        seen["args"] = args
        return 0, ""

    monkeypatch.setattr(engines, "_run_pip", fake_pip)
    engines.install("wcx")
    assert seen["args"] == ["install", "--torch-backend=auto", "-e", "../wcx-suite[cuda]"]


def test_install_already_present_is_noop(monkeypatch):
    monkeypatch.setattr(engines, "is_installed", lambda k: True)
    called = []
    monkeypatch.setattr(engines, "_run_pip", lambda args: called.append(args) or (0, ""))
    r = engines.install("wmx")
    assert r.status == "already"
    assert called == []   # already there → don't reinstall


def test_install_runs_uv_pip_install_and_succeeds(monkeypatch):
    monkeypatch.setattr(engines, "is_installed", lambda k: False)
    monkeypatch.delenv("ARA_WMX_SOURCE", raising=False)
    seen = {}

    def fake_pip(args):
        seen["args"] = args
        return 0, "ok"

    monkeypatch.setattr(engines, "_run_pip", fake_pip)
    r = engines.install("wmx")
    assert r.status == "installed"
    assert seen["args"] == ["install", "git+https://github.com/willsarg/wmx-suite"]


def test_install_local_source_is_editable(monkeypatch):
    monkeypatch.setattr(engines, "is_installed", lambda k: False)
    monkeypatch.setenv("ARA_WMX_SOURCE", "../wmx-suite")
    seen = {}

    def fake_pip(args):
        seen["args"] = args
        return 0, ""

    monkeypatch.setattr(engines, "_run_pip", fake_pip)
    engines.install("wmx")
    assert seen["args"] == ["install", "-e", "../wmx-suite"]   # local path → editable


def test_install_reports_failed_on_nonzero(monkeypatch):
    monkeypatch.setattr(engines, "is_installed", lambda k: False)
    monkeypatch.delenv("ARA_WMX_SOURCE", raising=False)
    monkeypatch.setattr(engines, "_run_pip", lambda args: (1, "boom: clone failed"))
    r = engines.install("wmx")
    assert r.status == "failed"
    assert "boom" in r.detail


# --------------------------------------------------------------------------- #
# uninstall() — symmetric: remove the engine's package
# --------------------------------------------------------------------------- #
def test_uninstall_unknown_engine_reports_unknown():
    assert engines.uninstall("nonsense").status == "unknown"


def test_uninstall_absent_engine_is_noop(monkeypatch):
    monkeypatch.setattr(engines, "is_installed", lambda k: False)
    called = []
    monkeypatch.setattr(engines, "_run_pip", lambda args: called.append(args) or (0, ""))
    assert engines.uninstall("wmx").status == "absent"
    assert called == []   # nothing installed → nothing to remove


def test_uninstall_removes_installed_package(monkeypatch):
    monkeypatch.setattr(engines, "is_installed", lambda k: True)
    seen = {}

    def fake_pip(args):
        seen["args"] = args
        return 0, "removed"

    monkeypatch.setattr(engines, "_run_pip", fake_pip)
    r = engines.uninstall("wmx")
    assert r.status == "removed"
    assert seen["args"] == ["uninstall", "wmx-suite"]   # the dist name, not the module


def test_uninstall_reports_failed_on_nonzero(monkeypatch):
    monkeypatch.setattr(engines, "is_installed", lambda k: True)
    monkeypatch.setattr(engines, "_run_pip", lambda args: (1, "pip blew up"))
    r = engines.uninstall("wmx")
    assert r.status == "failed"
    assert "blew up" in r.detail


# --------------------------------------------------------------------------- #
# _run_pip() — the real subprocess boundary (uv pip ...)
# --------------------------------------------------------------------------- #
def test_run_pip_prefixes_uv_pip_and_combines_output(monkeypatch):
    import types as _t
    seen = {}

    def fake_run(cmd, capture_output, text):
        seen["cmd"] = cmd
        return _t.SimpleNamespace(returncode=0, stdout="out\n", stderr="warn\n")

    monkeypatch.setattr(engines.subprocess, "run", fake_run)
    rc, out = engines._run_pip(["install", "x"])
    assert seen["cmd"] == ["uv", "pip", "install", "x"]
    assert rc == 0 and "out" in out and "warn" in out


def test_run_pip_swallows_errors_as_nonzero(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("uv not found")

    monkeypatch.setattr(engines.subprocess, "run", boom)
    rc, out = engines._run_pip(["install", "x"])
    assert rc == 1 and "uv not found" in out
