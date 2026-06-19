"""Shared fixtures for the ARA test suite.

The suite runs with NO real ``wmx_suite`` and NO real host probing: the
system-call boundary (``detect._run``, ``platform``, ``psutil``, the filesystem)
is mocked so both the Apple and non-Apple code paths run on any host.
"""
from __future__ import annotations

import io
import sys
import types

import pytest

from ara.ui import Console


# --------------------------------------------------------------------------- #
# version-lookup caches — brew calls are lru_cached at module level, so results
# leak across tests (and detect.profile()/apps.scan() consume them). Clear before
# AND after every test so each starts from a cold, deterministic cache.
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _clear_version_caches():
    from ara import versions
    cached = (versions.brew_formulae, versions.brew_casks, versions.cask_auto_updates)
    for f in cached:
        f.cache_clear()
    yield
    for f in cached:
        f.cache_clear()


# --------------------------------------------------------------------------- #
# console capture
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _isolate_db(tmp_path_factory, monkeypatch):
    """Point ARA's store at a throwaway db for EVERY test — never the real ~/.ara."""
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path_factory.mktemp("aradb") / "ara.db"))


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fresh on-disk ARA db in a tmp dir (via the ARA_DB_PATH override)."""
    from ara import db
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "ara.db"))
    return db.connect()


@pytest.fixture
def make_console():
    """Factory: build a Console writing to an in-memory buffer.

    Returns ``(console, buf)``; read rendered text with ``buf.getvalue()``.
    Color defaults off so assertions match plain text.
    """
    def _make(color: bool = False, verbose: bool = False):
        buf = io.StringIO()
        return Console(color=color, verbose=verbose, stream=buf), buf

    return _make


@pytest.fixture
def console(make_console):
    """A plain (no-color) console + its buffer, the common case."""
    return make_console()


# --------------------------------------------------------------------------- #
# system-call boundary stubs
# --------------------------------------------------------------------------- #
class RunStub:
    """Stand-in for ``detect._run``: match a command by a substring of its
    space-joined argv and return canned stdout (or None to mean 'failed')."""

    def __init__(self):
        self.responses: dict[str, str | None] = {}
        self.calls: list[list[str]] = []

    def add(self, match: str, output: str | None) -> "RunStub":
        self.responses[match] = output
        return self

    def __call__(self, cmd, timeout=3):
        self.calls.append(cmd)
        joined = " ".join(cmd)
        for match, out in self.responses.items():
            if match in joined:
                return out
        return None


@pytest.fixture
def run_stub(monkeypatch):
    """Patch ``detect._run`` with a configurable RunStub."""
    import ara.detect as detect

    stub = RunStub()
    monkeypatch.setattr(detect, "_run", stub)
    return stub


@pytest.fixture
def set_platform(monkeypatch):
    """Factory to force ``platform.system()`` / ``platform.machine()``."""
    import platform as platform_mod

    def _set(system: str, machine: str):
        monkeypatch.setattr(platform_mod, "system", lambda: system)
        monkeypatch.setattr(platform_mod, "machine", lambda: machine)

    return _set


@pytest.fixture
def fake_home(monkeypatch, tmp_path):
    """Point ``Path.home()`` at a tmp dir and clear HF_HOME so model-store and
    token scans look only inside the sandbox."""
    import pathlib

    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    return tmp_path


# --------------------------------------------------------------------------- #
# the wmx-suite seam — a fake engine injected into sys.modules
# --------------------------------------------------------------------------- #
class FakeWmx:
    """Knobs for the fake wmx_suite, read by its submodule functions."""

    def __init__(self):
        # system.read_limits() shape
        self.device = "Apple M4 Pro"
        self.total_gb = 48.0
        self.wall_gb = 40.0
        self.wired_now_gb = 8.0
        self.swap_free_gb = 2.0
        self.safe = 36.0
        self.margin = 4.0
        # db.get_profile() — None means "uncalibrated"
        self.profile = None
        # models.describe()
        self.describe_return = {"id": "x"}
        self.describe_raises = False
        # probe.calibrate()
        self.calibrate_return = {
            "measured_overhead_gb": 5.0,
            "default_overhead_gb": 6.0,
            "n_points": 4,
            "hf_id": "mlx-community/SmolLM-135M-Instruct-4bit",
        }
        self.calibrate_raises: BaseException | None = None
        self.calibrate_calls: list[str] = []


def _build_fake_wmx_modules(state: FakeWmx) -> dict[str, types.ModuleType]:
    def mod(name: str, **attrs) -> types.ModuleType:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    limits = types.SimpleNamespace(
        device=state.device,
        total_gb=state.total_gb,
        wall_gb=state.wall_gb,
        wired_now_gb=state.wired_now_gb,
        swap_free_gb=state.swap_free_gb,
        safe_threshold_gb=lambda margin: state.safe,
    )

    system = mod("wmx_suite.system", read_limits=lambda: limits)
    config = mod("wmx_suite.config", margin_gb=lambda _x: state.margin)
    profiles = mod("wmx_suite.profiles", machine_key=lambda: "machine-abc")
    db = mod(
        "wmx_suite.db",
        connect=lambda: object(),
        get_profile=lambda con, key: state.profile,
    )

    def _describe(model):
        if state.describe_raises:
            raise RuntimeError("describe blew up")
        return state.describe_return

    models = mod("wmx_suite.models", describe=_describe)

    def _calibrate(model, margin_gb=None, console=None):
        state.calibrate_calls.append(model)
        if state.calibrate_raises is not None:
            raise state.calibrate_raises
        return state.calibrate_return

    probe = mod("wmx_suite.probe", calibrate=_calibrate)

    class _EngineConsole:
        @classmethod
        def from_args(cls):
            return cls()

    ui = mod("wmx_suite.ui", Console=_EngineConsole)

    pkg = mod(
        "wmx_suite",
        system=system, config=config, profiles=profiles, db=db,
        models=models, probe=probe, ui=ui,
    )
    pkg.__path__ = []  # mark as a package so submodule imports resolve
    return {
        "wmx_suite": pkg,
        "wmx_suite.system": system,
        "wmx_suite.config": config,
        "wmx_suite.profiles": profiles,
        "wmx_suite.db": db,
        "wmx_suite.models": models,
        "wmx_suite.probe": probe,
        "wmx_suite.ui": ui,
    }


@pytest.fixture
def fake_wmx(monkeypatch):
    """Inject a fake ``wmx_suite`` package into ``sys.modules`` and return the
    FakeWmx knobs. Tests tweak ``.profile`` / ``.calibrate_raises`` etc."""
    state = FakeWmx()
    for name, module in _build_fake_wmx_modules(state).items():
        monkeypatch.setitem(sys.modules, name, module)
    return state
