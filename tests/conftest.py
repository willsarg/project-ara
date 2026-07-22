# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Shared fixtures for the ARA test suite.

The suite runs with NO real MLX/CUDA engine dependency and NO real host probing: the
system-call boundary (``detect._run``, ``platform``, ``psutil``, the filesystem)
is mocked so both the Apple and non-Apple code paths run on any host.
"""
from __future__ import annotations

import io

import pytest

from ara.detect import Accelerator, Machine, ModelStore, Runtime
from ara.ui import Console


# --------------------------------------------------------------------------- #
# version-lookup caches — brew calls are lru_cached at module level, so results
# leak across tests (and detect.machine()/apps.scan() consume them). Clear before
# AND after every test so each starts from a cold, deterministic cache.
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _clear_version_caches():
    from ara import versions
    cached = (versions.brew_formulae, versions.brew_casks, versions._cask_auto_updates_cached)
    for f in cached:
        f.cache_clear()
    yield
    for f in cached:
        f.cache_clear()


# --------------------------------------------------------------------------- #
# console capture
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _isolate_runtime_state(tmp_path_factory, monkeypatch):
    """Keep every ARA-owned state file under one fresh per-test directory."""
    root = tmp_path_factory.mktemp("ara-state")
    monkeypatch.setenv("ARA_DB_PATH", str(root / "ara.db"))
    monkeypatch.setenv("ARA_ACTIVITY_DIR", str(root / "activity"))
    monkeypatch.setenv("ARA_ENGINES_DIR", str(root / "engines"))


@pytest.fixture
def sample_machine():
    """Deterministic domain snapshot for projection tests that do not exercise recon."""
    return Machine(
        system="Darwin", os_version="macOS 15.0", chip="Apple M4 Pro", arch="arm64",
        cpu_physical=12, cpu_logical=12, cpu_features=["NEON"], python_version="3.12.8",
        ram_total_gb=48.0, ram_available_gb=20.0, swap_gb=2.0,
        accel=Accelerator("apple", "Apple M4 Pro GPU", None, "Metal", cores=16),
        disk_free_gb=500.0,
        runtimes=[Runtime("Ollama", True, "0.6", serving=True)],
        framework_python="/usr/bin/python3",
        model_stores=[ModelStore("HF cache", True, 3, 12.0)],
        hf_token=True, power="AC power", backend="apple", engine="mlx",
        engine_ready=True,
    )


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
