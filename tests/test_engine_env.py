"""Isolated per-engine uv environments + the worker IPC seam.

These tests mock only the subprocess boundary (`engine_env._run`, an unavoidable
external call to uv / the engine's python) and use a tmp engines root via the
ARA_ENGINES_DIR override — mirroring how the db tests use ARA_DB_PATH. No real uv
env is created here; a separate smoke test exercises the real thing.
"""
from __future__ import annotations

import json

import pytest

from ara import engine_env


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class RunSpy:
    """Records commands and replays canned (rc, stdout, stderr) by argv substring."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.inputs: list[str | None] = []
        self.responses: list[tuple[str, tuple[int, str, str]]] = []
        self.default = (0, "", "")

    def add(self, match: str, rc: int, out: str = "", err: str = "") -> "RunSpy":
        self.responses.append((match, (rc, out, err)))
        return self

    def __call__(self, cmd, *, input=None):
        self.calls.append(cmd)
        self.inputs.append(input)
        joined = " ".join(cmd)
        for match, resp in self.responses:
            if match in joined:
                return resp
        return self.default


@pytest.fixture
def engines_root(tmp_path, monkeypatch):
    root = tmp_path / "engines"
    monkeypatch.setenv("ARA_ENGINES_DIR", str(root))
    return root


@pytest.fixture
def run_spy(monkeypatch):
    spy = RunSpy()
    monkeypatch.setattr(engine_env, "_run", spy)
    return spy


# --------------------------------------------------------------------------- #
# _run — the one external boundary (subprocess), stubbed everywhere else
# --------------------------------------------------------------------------- #
def test_run_invokes_subprocess_and_returns_streams(monkeypatch):
    captured = {}

    class P:
        returncode, stdout, stderr = 5, "hello", "oops"

    def fake_run(cmd, *, capture_output, text, input):
        captured["cmd"], captured["input"] = cmd, input
        return P()

    monkeypatch.setattr(engine_env.subprocess, "run", fake_run)
    assert engine_env._run(["uv", "venv"], input="x") == (5, "hello", "oops")
    assert captured == {"cmd": ["uv", "venv"], "input": "x"}


def test_run_coerces_none_streams_to_empty(monkeypatch):
    class P:
        returncode, stdout, stderr = 0, None, None

    monkeypatch.setattr(engine_env.subprocess, "run", lambda *a, **k: P())
    assert engine_env._run(["x"]) == (0, "", "")


# --------------------------------------------------------------------------- #
# path resolution
# --------------------------------------------------------------------------- #
def test_engines_root_honors_override(engines_root):
    assert engine_env.engines_root() == engines_root


def test_engines_root_defaults_to_data_dir(monkeypatch):
    monkeypatch.delenv("ARA_ENGINES_DIR", raising=False)
    root = engine_env.engines_root()
    assert root.name == "engines"
    assert root.parent.name == "ara"  # platformdirs user_data_dir("ara")


def test_env_path_joins_name_under_root(engines_root):
    assert engine_env.env_path("cuda") == engines_root / "cuda"


def test_python_path_posix(engines_root, monkeypatch):
    monkeypatch.setattr(engine_env, "_is_windows", lambda: False)
    assert engine_env.python_path("cpu") == engines_root / "cpu" / "bin" / "python"


def test_python_path_windows(engines_root, monkeypatch):
    monkeypatch.setattr(engine_env, "_is_windows", lambda: True)
    assert engine_env.python_path("cpu") == engines_root / "cpu" / "Scripts" / "python.exe"


def test_is_windows_reads_os_name(monkeypatch):
    monkeypatch.setattr(engine_env.os, "name", "nt")
    assert engine_env._is_windows() is True
    monkeypatch.setattr(engine_env.os, "name", "posix")
    assert engine_env._is_windows() is False


def test_exists_false_when_absent(engines_root):
    assert engine_env.exists("ghost") is False


def test_exists_true_when_python_present(engines_root):
    py = engine_env.python_path("apple")
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n")
    assert engine_env.exists("apple") is True


# --------------------------------------------------------------------------- #
# create
# --------------------------------------------------------------------------- #
def test_create_runs_venv_then_install(engines_root, run_spy):
    path = engine_env.create("cpu", ["llama-cpp-python"])
    assert path == engines_root / "cpu"
    # first call creates the venv, second installs
    assert run_spy.calls[0][:2] == ["uv", "venv"]
    assert run_spy.calls[1][:3] == ["uv", "pip", "install"]
    assert "llama-cpp-python" in run_spy.calls[1]


def test_create_defaults_to_clone_link_mode(engines_root, run_spy):
    engine_env.create("cpu", ["x"])
    install = run_spy.calls[1]
    assert install[install.index("--link-mode") + 1] == "clone"


def test_create_honors_explicit_link_mode(engines_root, run_spy):
    engine_env.create("cpu", ["x"], link_mode="hardlink")
    install = run_spy.calls[1]
    assert install[install.index("--link-mode") + 1] == "hardlink"


def test_create_pins_python_when_requested(engines_root, run_spy):
    engine_env.create("apple", ["wmx-suite"], python="3.12")
    venv = run_spy.calls[0]
    assert venv[venv.index("--python") + 1] == "3.12"


def test_create_omits_python_pin_by_default(engines_root, run_spy):
    engine_env.create("cpu", ["x"])
    assert "--python" not in run_spy.calls[0]


def test_create_raises_when_venv_fails(engines_root, run_spy):
    run_spy.add("uv venv", 1, err="no python found")
    with pytest.raises(engine_env.EngineEnvError, match="no python found"):
        engine_env.create("cpu", ["x"])


def test_create_raises_when_install_fails(engines_root, run_spy):
    run_spy.add("pip install", 1, err="resolution impossible")
    with pytest.raises(engine_env.EngineEnvError, match="resolution impossible"):
        engine_env.create("cpu", ["x"])


# --------------------------------------------------------------------------- #
# remove
# --------------------------------------------------------------------------- #
def test_remove_deletes_existing_env(engines_root):
    env = engine_env.env_path("cuda")
    env.mkdir(parents=True)
    (env / "marker").write_text("x")
    assert engine_env.remove("cuda") is True
    assert not env.exists()


def test_remove_absent_returns_false(engines_root):
    assert engine_env.remove("never") is False


# --------------------------------------------------------------------------- #
# run_worker (the IPC seam)
# --------------------------------------------------------------------------- #
def test_run_worker_parses_json_line(engines_root, run_spy):
    run_spy.add("ipykernel", 0, out="noise\n" + json.dumps({"mem_gb": 7.5}) + "\n")
    result = engine_env.run_worker("apple", ["-m", "ipykernel", "measure"])
    assert result == {"mem_gb": 7.5}


def test_run_worker_spawns_envs_own_python(engines_root, run_spy, monkeypatch):
    monkeypatch.setattr(engine_env, "_is_windows", lambda: False)
    run_spy.add("python", 0, out='{"ok": true}')
    engine_env.run_worker("apple", ["-c", "pass"])
    cmd = run_spy.calls[0]
    assert cmd[0] == str(engines_root / "apple" / "bin" / "python")
    assert cmd[1:] == ["-c", "pass"]


def test_run_worker_passes_stdin(engines_root, run_spy):
    run_spy.add("python", 0, out='{"ok": true}')
    engine_env.run_worker("apple", ["-"], input='{"req": 1}')
    assert run_spy.inputs[0] == '{"req": 1}'


def test_run_worker_raises_on_nonzero(engines_root, run_spy):
    run_spy.add("python", 2, err="traceback: boom")
    with pytest.raises(engine_env.EngineEnvError, match="boom"):
        engine_env.run_worker("apple", ["-c", "x"])


def test_run_worker_raises_when_no_json(engines_root, run_spy):
    run_spy.add("python", 0, out="just logs, no json here")
    with pytest.raises(engine_env.EngineEnvError, match="no JSON"):
        engine_env.run_worker("apple", ["-c", "x"])
