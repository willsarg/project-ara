# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Isolated per-engine uv environments + the worker IPC seam.

These tests mock only the subprocess boundary (`engine_env._run`, an unavoidable
external call to uv / the engine's python) and use a tmp engines root via the
ARA_ENGINES_DIR override — mirroring how the db tests use ARA_DB_PATH. No real uv
env is created here; a separate smoke test exercises the real thing.
"""
from __future__ import annotations

import json
import sys

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

    def __call__(self, cmd, *, input=None, stream=False):
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


def test_run_stream_true_uses_pipe_stdout_and_inherit_stderr(monkeypatch):
    """stream=True: subprocess called with stdout=PIPE, stderr=None (inherits terminal).

    Slug: 2026-06-24-download-progress
    """
    captured = {}

    class P:
        returncode = 0
        stdout = '{"ok": true}\n'

    def fake_run(cmd, *, stdout, stderr, text, input):
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["input"] = input
        return P()

    monkeypatch.setattr(engine_env.subprocess, "run", fake_run)
    rc, out, err = engine_env._run(["python", "worker.py"], stream=True)
    assert captured["stdout"] == engine_env.subprocess.PIPE
    assert captured["stderr"] is None       # passes through live — not captured
    assert captured["input"] is None
    assert rc == 0
    assert out == '{"ok": true}\n'
    assert err == ""                        # always empty on the stream path


def test_run_stream_true_returns_empty_err(monkeypatch):
    """stream=True always returns empty string for err (stderr not captured).

    Slug: 2026-06-24-download-progress
    """
    class P:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(engine_env.subprocess, "run", lambda *a, **k: P())
    rc, out, err = engine_env._run(["x"], stream=True)
    assert err == ""


def test_run_passes_explicit_timeout_to_subprocess(monkeypatch):
    seen = {}

    class P:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(cmd, *, capture_output, text, input, timeout):
        seen["timeout"] = timeout
        return P()

    monkeypatch.setattr(engine_env.subprocess, "run", fake_run)
    assert engine_env._run(["probe"], timeout=7) == (0, "", "")
    assert seen == {"timeout": 7}


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
    engine_env.create("apple", ["ara-engine-mlx"], python="3.12")
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


def test_create_tears_down_half_built_env_on_install_failure(engines_root, run_spy):
    # uv venv made the python (env "exists"), but the pip install then fails. Without cleanup,
    # exists()/is_installed() would report the broken env as ready forever and `ara install`
    # would say "already". create() must remove it so a retry can actually repair the install.
    run_spy.add("pip install", 1, err="boom")
    py = engine_env.python_path("cpu")
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("#!/bin/sh\n")          # simulate the venv's python from the (mocked) uv venv
    assert engine_env.exists("cpu") is True
    with pytest.raises(engine_env.EngineEnvError):
        engine_env.create("cpu", ["x"])
    assert engine_env.exists("cpu") is False          # torn down, not left half-built
    assert not engine_env.env_path("cpu").exists()


# --------------------------------------------------------------------------- #
# uv preflight — a missing uv is a friendly error, not a raw FileNotFoundError
# --------------------------------------------------------------------------- #
def test_create_raises_friendly_when_uv_missing(engines_root, run_spy, monkeypatch):
    # No uv on PATH: create must fail up front with an actionable message and never spawn anything.
    monkeypatch.setattr(engine_env.shutil, "which", lambda name: None)
    with pytest.raises(engine_env.EngineEnvError, match="uv not found on PATH"):
        engine_env.create("cpu", ["x"])
    assert run_spy.calls == []            # bailed before touching the subprocess boundary


# --------------------------------------------------------------------------- #
# version stamp — tells a stale installed engine from the current native package
# --------------------------------------------------------------------------- #
def test_stamped_version_none_when_absent(engines_root):
    assert engine_env.stamped_version("ghost") is None


def test_stamped_version_reads_written_stamp(engines_root):
    env = engine_env.env_path("apple")
    env.mkdir(parents=True)
    (env / ".ara-version").write_text("1.2.3\n")
    assert engine_env.stamped_version("apple") == "1.2.3"


def test_create_writes_stamp_when_version_given(engines_root, run_spy):
    engine_env.create("cpu", ["x"], version="9.9.9")
    assert engine_env.stamped_version("cpu") == "9.9.9"


def test_create_omits_stamp_when_version_none(engines_root, run_spy):
    # Existing callers pass no version → no stamp written (stamped_version stays None).
    engine_env.create("cpu", ["x"])
    assert engine_env.stamped_version("cpu") is None


# --------------------------------------------------------------------------- #
# package schema stamp — tells an old module layout from the engine's current one
# --------------------------------------------------------------------------- #
def test_stamped_schema_none_when_absent(engines_root):
    assert engine_env.stamped_schema("ghost") is None


def test_stamped_schema_reads_written_stamp(engines_root):
    env = engine_env.env_path("apple")
    env.mkdir(parents=True)
    (env / ".ara-schema").write_text("mlx-worker-v2\n")
    assert engine_env.stamped_schema("apple") == "mlx-worker-v2"


def test_create_writes_schema_stamp_when_schema_given(engines_root, run_spy):
    engine_env.create("cpu", ["x"], schema="cpu-worker-v1")
    assert engine_env.stamped_schema("cpu") == "cpu-worker-v1"


def test_create_omits_schema_stamp_when_schema_none(engines_root, run_spy):
    engine_env.create("cpu", ["x"])
    assert engine_env.stamped_schema("cpu") is None


def test_create_does_not_leave_stamps_when_install_fails(engines_root, run_spy):
    run_spy.add("pip install", 1, err="boom")
    env = engine_env.env_path("cpu")
    env.mkdir(parents=True)
    (env / ".ara-version").write_text("old")
    (env / ".ara-schema").write_text("old")

    with pytest.raises(engine_env.EngineEnvError):
        engine_env.create("cpu", ["x"], version="2.0.0", schema="cpu-worker-v2")

    assert not env.exists()


def test_create_verifies_expected_import_package_before_stamping(engines_root, run_spy):
    engine_env.create(
        "apple",
        ["/native/mlx"],
        version="2.0.0",
        schema="ara-engine-mlx:ara_engine_mlx:v1",
        expected_import="ara_engine_mlx",
    )

    verify = run_spy.calls[2]
    assert verify[0] == str(engine_env.python_path("apple"))
    assert verify[1:3] == ["-I", "-c"]
    assert "find_spec" in verify[3]
    assert verify[4] == "ara_engine_mlx"
    assert engine_env.stamped_version("apple") == "2.0.0"
    assert engine_env.stamped_schema("apple") == "ara-engine-mlx:ara_engine_mlx:v1"


def test_python_isolated_mode_ignores_hostile_cwd_and_pythonpath(tmp_path, monkeypatch):
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    (hostile / "hostile_engine.py").write_text("SPOOFED = True\n")
    monkeypatch.chdir(hostile)
    monkeypatch.setenv("PYTHONPATH", str(hostile))
    script = (
        "import importlib.util, sys; "
        "sys.exit(0 if importlib.util.find_spec(sys.argv[1]) is not None else 1)"
    )

    normal, _out, _err = engine_env._run(
        [sys.executable, "-c", script, "hostile_engine"])
    isolated, _out, _err = engine_env._run(
        [sys.executable, "-I", "-c", script, "hostile_engine"])

    assert normal == 0
    assert isolated == 1


def test_create_rejects_legacy_source_without_expected_import_and_removes_env(
        engines_root, run_spy):
    run_spy.add("ara_engine_mlx", 1)
    env = engine_env.env_path("apple")
    env.mkdir(parents=True)
    (env / ".ara-version").write_text("old")
    (env / ".ara-schema").write_text("old")

    with pytest.raises(engine_env.EngineEnvError, match="ara_engine_mlx"):
        engine_env.create(
            "apple",
            ["-e", "../legacy-wmx-suite"],
            version="2.0.0",
            schema="ara-engine-mlx:ara_engine_mlx:v1",
            expected_import="ara_engine_mlx",
        )

    assert not env.exists()
    assert engine_env.stamped_version("apple") is None
    assert engine_env.stamped_schema("apple") is None


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


def test_run_worker_spawns_windows_interpreter(engines_root, run_spy, monkeypatch):
    # Portability: the IPC seam must spawn Scripts\python.exe on Windows, not bin/python.
    monkeypatch.setattr(engine_env, "_is_windows", lambda: True)
    run_spy.add("python.exe", 0, out='{"ok": true}')
    engine_env.run_worker("apple", ["-m", "ara_engine_mlx.device", "limits"])
    cmd = run_spy.calls[0]
    assert cmd[0] == str(engines_root / "apple" / "Scripts" / "python.exe")


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


# --------------------------------------------------------------------------- #
# run_worker stream=True (2026-06-24-download-progress)
# --------------------------------------------------------------------------- #
def test_run_worker_stream_true_passes_stream_to_run(engines_root, monkeypatch):
    """stream=True propagated to _run; stdout JSON still parsed correctly.

    Slug: 2026-06-24-download-progress
    """
    captured = {}

    def fake_run(cmd, *, input=None, stream=False):
        captured["stream"] = stream
        return (0, '{"result": 42}\n', "")

    monkeypatch.setattr(engine_env, "_run", fake_run)
    result = engine_env.run_worker("apple", ["-m", "worker"], stream=True)
    assert captured["stream"] is True
    assert result == {"result": 42}


def test_run_worker_stream_false_passes_stream_to_run(engines_root, monkeypatch):
    """stream=False (default) propagated to _run.

    Slug: 2026-06-24-download-progress
    """
    captured = {}

    def fake_run(cmd, *, input=None, stream=False):
        captured["stream"] = stream
        return (0, '{"ok": true}\n', "")

    monkeypatch.setattr(engine_env, "_run", fake_run)
    engine_env.run_worker("apple", ["-m", "worker"])
    assert captured["stream"] is False


def test_run_worker_stream_true_error_message_says_see_output_above(engines_root, monkeypatch):
    """stream=True nonzero rc: error says 'see output above' (stderr wasn't captured).

    Slug: 2026-06-24-download-progress
    """
    monkeypatch.setattr(engine_env, "_run", lambda cmd, *, input=None, stream=False: (2, "", ""))
    with pytest.raises(engine_env.EngineEnvError, match="see output above"):
        engine_env.run_worker("apple", ["-c", "x"], stream=True)


def test_run_worker_stream_false_error_message_embeds_stderr(engines_root, monkeypatch):
    """stream=False nonzero rc: error message includes the captured stderr text.

    Slug: 2026-06-24-download-progress
    """
    monkeypatch.setattr(engine_env, "_run",
                        lambda cmd, *, input=None, stream=False: (3, "", "traceback here"))
    with pytest.raises(engine_env.EngineEnvError, match="traceback here"):
        engine_env.run_worker("apple", ["-c", "x"], stream=False)


def test_run_worker_stream_true_no_json_still_raises(engines_root, monkeypatch):
    """stream=True with no JSON line in stdout: EngineEnvError 'no JSON' unchanged.

    Slug: 2026-06-24-download-progress
    """
    monkeypatch.setattr(engine_env, "_run",
                        lambda cmd, *, input=None, stream=False: (0, "logs only, no json", ""))
    with pytest.raises(engine_env.EngineEnvError, match="no JSON"):
        engine_env.run_worker("apple", ["-m", "w"], stream=True)


# --------------------------------------------------------------------------- #
# run_python_json — bounded no-model runtime probes used by Doctor
# --------------------------------------------------------------------------- #
def test_run_python_json_uses_isolated_engine_interpreter(engines_root, monkeypatch):
    seen = {}

    def fake_run(cmd, *, input=None, stream=False, timeout=None):
        seen.update(cmd=cmd, timeout=timeout)
        return 0, 'log line\n{"available": true}\n', ""

    monkeypatch.setattr(engine_env, "_run", fake_run)

    assert engine_env.run_python_json("cuda", "print('probe')", timeout=12) == {
        "available": True,
    }
    assert seen == {
        "cmd": [str(engine_env.python_path("cuda")), "-I", "-c", "print('probe')"],
        "timeout": 12,
    }


def test_run_python_json_reports_timeout_as_engine_error(engines_root, monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise engine_env.subprocess.TimeoutExpired(["python"], 5)

    monkeypatch.setattr(engine_env, "_run", fake_run)

    with pytest.raises(engine_env.EngineEnvError, match="timed out after 5 seconds"):
        engine_env.run_python_json("cpu", "pass", timeout=5)


def test_run_python_json_rejects_non_object_payload(engines_root, monkeypatch):
    monkeypatch.setattr(
        engine_env, "_run", lambda *_args, **_kwargs: (0, "[1, 2, 3]\n", ""))

    with pytest.raises(engine_env.EngineEnvError, match="emitted no JSON object"):
        engine_env.run_python_json("cpu", "pass")


def test_run_python_json_reports_nonzero_with_stderr(engines_root, monkeypatch):
    monkeypatch.setattr(
        engine_env, "_run", lambda *_args, **_kwargs: (9, "", "library missing"))

    with pytest.raises(engine_env.EngineEnvError, match="exited 9: library missing"):
        engine_env.run_python_json("cpu", "pass")


def test_run_python_json_rejects_invalid_json_object(engines_root, monkeypatch):
    monkeypatch.setattr(
        engine_env, "_run", lambda *_args, **_kwargs: (0, "{not-json}\n", ""))

    with pytest.raises(engine_env.EngineEnvError, match="emitted invalid JSON"):
        engine_env.run_python_json("cpu", "pass")


# --------------------------------------------------------------------------- #
# start_worker_server — long-lived governed server seam
# --------------------------------------------------------------------------- #

class _FakeServerProc:
    """Minimal Popen stand-in for start_worker_server tests."""

    def __init__(self, lines, rc=0):
        self.stdout = iter(lines)
        self.returncode = rc
        self.waited = False
        self.killed = False

    def wait(self):
        self.waited = True

    def kill(self):
        self.killed = True


def _mock_server_popen(monkeypatch, lines, rc=0):
    """Replace subprocess.Popen with a fake that returns a FakeServerProc."""
    proc = _FakeServerProc(lines, rc)
    captured_cmd = []

    def fake_popen(cmd, *, stdout, text):
        captured_cmd.extend(cmd)
        return proc

    monkeypatch.setattr(engine_env.subprocess, "Popen", fake_popen)
    return proc, captured_cmd


def test_start_worker_server_returns_proc_and_dict_on_ready_json(engines_root, monkeypatch):
    """Happy path: a ready JSON line → (proc, dict) returned; server keeps running."""
    ready = '{"ready": true, "url": "http://127.0.0.1:8080", "context": 4096}\n'
    proc, _ = _mock_server_popen(monkeypatch, ["log line\n", ready])
    result_proc, result_dict = engine_env.start_worker_server(
        "apple", ["-m", "ara_engine_mlx.serve"])
    assert result_proc is proc
    assert result_dict == {"ready": True, "url": "http://127.0.0.1:8080", "context": 4096}
    # proc must NOT be waited on — the server keeps running
    assert not proc.waited


@pytest.mark.parametrize("ready", [
    '{"ready": false, "url": "http://127.0.0.1:8080", "context": 4096}\n',
    '{"url": "http://127.0.0.1:8080", "context": 4096}\n',
    '[]\n',
])
def test_start_worker_server_rejects_nonready_payload_and_reaps(
        engines_root, monkeypatch, ready):
    proc, _ = _mock_server_popen(monkeypatch, [ready])
    with pytest.raises(engine_env.EngineEnvError, match="ready signal"):
        engine_env.start_worker_server("apple", ["-m", "ara_engine_mlx.serve"])
    assert proc.killed and proc.waited


def test_start_worker_server_raises_on_refused(engines_root, monkeypatch):
    """refused=true JSON → EngineEnvError with the reason; child is reaped (kill + wait)."""
    refused = '{"refused": true, "reason": "model exceeds safe budget"}\n'
    proc, _ = _mock_server_popen(monkeypatch, [refused])
    with pytest.raises(engine_env.EngineEnvError, match="model exceeds safe budget"):
        engine_env.start_worker_server("apple", ["-m", "ara_engine_mlx.serve"])
    assert proc.killed and proc.waited   # reap signal — the happy path asserts `not proc.waited`


def test_start_worker_server_raises_on_error(engines_root, monkeypatch):
    """error key in JSON → EngineEnvError with the error text; child is reaped (kill + wait)."""
    error = '{"error": "load failed: model not found"}\n'
    proc, _ = _mock_server_popen(monkeypatch, [error])
    with pytest.raises(engine_env.EngineEnvError, match="load failed"):
        engine_env.start_worker_server("apple", ["-m", "ara_engine_mlx.serve"])
    assert proc.killed and proc.waited   # reap signal — the happy path asserts `not proc.waited`


def test_start_worker_server_raises_when_no_json_before_exit(engines_root, monkeypatch):
    """Stdout exhausted with no JSON line → EngineEnvError 'exited without a ready signal'."""
    proc, _ = _mock_server_popen(monkeypatch, ["log: starting\n", "log: crash\n"])
    with pytest.raises(engine_env.EngineEnvError, match="exited without a ready signal"):
        engine_env.start_worker_server("apple", ["-m", "ara_engine_mlx.serve"])


def test_start_worker_server_spawns_envs_own_python(engines_root, monkeypatch):
    """start_worker_server uses the engine's isolated python, not the ambient interpreter."""
    monkeypatch.setattr(engine_env, "_is_windows", lambda: False)
    ready = '{"ready": true, "url": "http://127.0.0.1:9000", "context": 2048}\n'
    _proc, cmd = _mock_server_popen(monkeypatch, [ready])
    engine_env.start_worker_server("apple", ["-m", "ara_engine_mlx.serve", "org/m"])
    assert cmd[0] == str(engines_root / "apple" / "bin" / "python")
    assert cmd[1:] == ["-m", "ara_engine_mlx.serve", "org/m"]


def test_start_worker_server_times_out_and_kills_child(engines_root, monkeypatch):
    # A child that never emits a ready line must not hang us forever — bounded + child reaped.
    import time

    class _Blocking:
        def __iter__(self):
            return self

        def __next__(self):
            time.sleep(5)
            raise StopIteration

    proc = _FakeServerProc([])
    proc.stdout = _Blocking()
    monkeypatch.setattr(engine_env.subprocess, "Popen",
                        lambda cmd, *, stdout, text: proc)
    with pytest.raises(engine_env.EngineEnvError, match="timed out"):
        engine_env.start_worker_server("apple", ["-m", "x"], ready_timeout=0.2)
    assert proc.killed


def test_start_worker_server_invalid_json_raises_and_kills(engines_root, monkeypatch):
    # A malformed ready line → EngineEnvError (not a raw JSONDecodeError) + child reaped (no leak).
    proc, _ = _mock_server_popen(monkeypatch, ["{not valid json\n"])
    with pytest.raises(engine_env.EngineEnvError, match="invalid ready signal"):
        engine_env.start_worker_server("apple", ["-m", "x"])
    assert proc.killed


class _BoomStdout:
    """A stdout whose iteration raises — models a broken pipe / decode error in the reader."""
    def __iter__(self):
        return self

    def __next__(self):
        raise OSError("pipe broke")


def test_start_worker_server_reader_exception_is_surfaced(engines_root, monkeypatch):
    # The handshake reader runs in a thread; if reading stdout raises, it records the error and the
    # caller reports "failed reading the ready signal" (covers the reader's except + the error raise).
    proc, _ = _mock_server_popen(monkeypatch, [])
    proc.stdout = _BoomStdout()
    with pytest.raises(engine_env.EngineEnvError, match="failed reading the ready signal"):
        engine_env.start_worker_server("apple", ["-m", "x"])


def test_start_worker_server_reap_swallows_kill_error(engines_root, monkeypatch):
    # If reaping the child itself raises (e.g. kill() on an already-dead proc), it's swallowed and
    # the original EngineEnvError still propagates — covers the reap loop's except: pass.
    proc, _ = _mock_server_popen(monkeypatch, ['{"refused": true, "reason": "nope"}\n'])

    def boom_kill():
        proc.killed = True
        raise ProcessLookupError("already dead")
    proc.kill = boom_kill
    with pytest.raises(engine_env.EngineEnvError, match="refused"):
        engine_env.start_worker_server("apple", ["-m", "x"])
    assert proc.killed and proc.waited   # kill attempted (raised, swallowed), wait still ran
