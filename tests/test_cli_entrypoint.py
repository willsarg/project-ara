# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Black-box contract for ARA's three blessed CLI entry paths."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from ara import cli


ROOT = Path(__file__).resolve().parents[1]
ARA = Path(sys.executable).with_name("ara")


def _direct(argv: list[str]) -> subprocess.CompletedProcess[str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        rc = cli.main(argv)
    return subprocess.CompletedProcess(["main", *argv], rc, stdout.getvalue(), stderr.getvalue())


def _subprocess(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    return subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)


def _in_process_entrypoint(path: str, argv: list[str], monkeypatch) -> subprocess.CompletedProcess[str]:
    """Execute a script/module entry path in-process so operational boundaries can be mocked."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    with monkeypatch.context() as patch:
        patch.setattr(sys, "argv", ["ara", *argv])
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with pytest.raises(SystemExit) as exc:
                if path == "console":
                    runpy.run_path(str(ARA), run_name="__main__")
                else:
                    runpy.run_module("ara.__main__", run_name="__main__")
    return subprocess.CompletedProcess(path, exc.value.code, stdout.getvalue(), stderr.getvalue())


@pytest.mark.parametrize("argv", [
    ["--help"],
    ["--version"],
    ["search"],
    ["apps", "--json"],
    ["profile", "--engine", "not-an-engine", "--json"],
])
def test_blessed_entrypoints_are_equivalent(argv):
    results = [
        _direct(argv),
        _subprocess([str(ARA), *argv]),
        _subprocess([sys.executable, "-m", "ara", *argv]),
    ]
    expected = (results[0].returncode, results[0].stdout, results[0].stderr)
    assert [(r.returncode, r.stdout, r.stderr) for r in results] == [expected] * 3


def test_click_usage_error_is_stderr_exit_two_and_never_json():
    result = _direct(["search", "--json"])
    assert result.returncode == 2
    assert result.stdout == ""
    assert "Usage: ara search" in result.stderr
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stderr)


def test_mocked_operational_failure_is_json_at_the_execution_boundary(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("worker crashed")

    monkeypatch.setattr(cli, "render_detect", fail)
    argv = ["detect", "--json"]
    results = [
        _direct(argv),
        _in_process_entrypoint("console", argv, monkeypatch),
        _in_process_entrypoint("module", argv, monkeypatch),
    ]
    expected = (1, json.dumps({"error": "ara failed: worker crashed"}) + "\n", "")
    assert [(r.returncode, r.stdout, r.stderr) for r in results] == [expected] * 3


def test_module_entrypoint_exits_with_shared_main_result(monkeypatch):
    monkeypatch.setattr(cli, "main", lambda: 7)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("ara.__main__", run_name="__main__")
    assert exc.value.code == 7
