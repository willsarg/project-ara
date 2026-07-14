# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node wiring — providers/workers translate to CLI args, and ``_run_cli`` is the only seam."""
from __future__ import annotations

import json
import sys

import pytest

from ara.node import wiring


class _FakeProc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_cli_success_parses_json(monkeypatch):
    captured = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd
        return _FakeProc(0, stdout=json.dumps({"ok": True}))

    monkeypatch.setattr(wiring.subprocess, "run", fake_run)
    assert wiring._run_cli(["detect"]) == {"ok": True}
    # The boundary always runs the module CLI and appends --json itself.
    assert captured["cmd"] == [sys.executable, "-m", "ara.cli", "detect", "--json"]


def test_run_cli_nonzero_exit_returns_error_dict(monkeypatch):
    monkeypatch.setattr(wiring.subprocess, "run",
                        lambda *a, **k: _FakeProc(1, stdout="", stderr="boom\n"))
    out = wiring._run_cli(["status"])
    assert out == {"error": "`ara status` exited 1", "stderr": "boom"}


def test_run_cli_unparseable_output_returns_error_dict(monkeypatch):
    monkeypatch.setattr(wiring.subprocess, "run",
                        lambda *a, **k: _FakeProc(0, stdout="not json", stderr=" noise "))
    out = wiring._run_cli(["models"])
    assert out == {"error": "`ara models` produced unparseable output", "stderr": "noise"}


def test_default_providers_map_each_key_to_its_canonical_cli_args(monkeypatch):
    calls = []
    monkeypatch.setattr(wiring, "_run_cli", lambda args: calls.append(args) or {"v": args[0]})
    providers = wiring.default_providers()
    assert set(providers) == {"status", "detect", "profile", "models"}
    expected = {
        "status": ["status"],
        "detect": ["detect"],
        "profile": ["profile"],
        "models": ["detect", "--models"],
    }
    for key, args in expected.items():
        assert providers[key]() == {"v": args[0]}
    assert calls == list(expected.values())


def test_characterize_worker_builds_args(monkeypatch):
    seen = {}
    monkeypatch.setattr(wiring, "_run_cli", lambda args: seen.setdefault("a", args) or {})
    wiring._characterize({"model": "org/m"})
    assert seen["a"] == ["characterize", "org/m"]


def test_characterize_worker_with_engine(monkeypatch):
    seen = {}
    monkeypatch.setattr(wiring, "_run_cli", lambda args: seen.setdefault("a", args) or {})
    wiring._characterize({"model": "org/m", "engine": "cpu"})
    assert seen["a"] == ["characterize", "org/m", "--engine", "cpu"]


def test_run_worker_builds_args_with_engine_and_prompt(monkeypatch):
    seen = {}
    monkeypatch.setattr(wiring, "_run_cli", lambda args: seen.setdefault("a", args) or {})
    wiring._run({"model": "org/m", "engine": "cuda", "prompt": "hello world"})
    assert seen["a"] == ["run", "org/m", "--engine", "cuda", "--yes", "hello world"]


def test_run_worker_without_optional_fields(monkeypatch):
    seen = {}
    monkeypatch.setattr(wiring, "_run_cli", lambda args: seen.setdefault("a", args) or {})
    wiring._run({"model": "org/m"})
    assert seen["a"] == ["run", "org/m", "--yes"]


def test_serve_worker_full_args(monkeypatch):
    seen = {}
    monkeypatch.setattr(wiring, "_run_cli", lambda args: seen.setdefault("a", args) or {})
    wiring._serve({"model": "org/m", "engine": "cpu", "ctx": 4096, "name": "svc"})
    assert seen["a"] == ["serve", "org/m", "--engine", "cpu", "--ctx", "4096",
                         "--name", "svc", "--yes"]


def test_serve_worker_minimal(monkeypatch):
    seen = {}
    monkeypatch.setattr(wiring, "_run_cli", lambda args: seen.setdefault("a", args) or {})
    wiring._serve({"model": "org/m"})
    assert seen["a"] == ["serve", "org/m", "--yes"]


def test_benchmark_worker_full_args(monkeypatch):
    seen = {}
    monkeypatch.setattr(wiring, "_run_cli", lambda args: seen.setdefault("a", args) or {})
    wiring._benchmark({"model": "org/m", "use_case": "coding", "engine": "cuda",
                       "ctx": 8192, "max_tokens": 512, "exec_consent": True})
    assert seen["a"] == ["benchmark", "org/m", "--use-case", "coding", "--engine", "cuda",
                         "--ctx", "8192", "--max-tokens", "512", "--exec-consent", "--yes"]


def test_benchmark_worker_minimal_no_exec_consent(monkeypatch):
    seen = {}
    monkeypatch.setattr(wiring, "_run_cli", lambda args: seen.setdefault("a", args) or {})
    wiring._benchmark({"model": "org/m", "use_case": "rag"})
    # --exec-consent is never auto-added; the gate stays intact.
    assert seen["a"] == ["benchmark", "org/m", "--use-case", "rag", "--yes"]


def test_default_workers_keys():
    assert set(wiring.default_workers()) == {"characterize", "run", "serve", "benchmark"}


# --- argv flag-injection guard (_safe): no job arg may be smuggled in as a CLI flag ---
@pytest.mark.parametrize("worker, args", [
    (wiring._characterize, {"model": "--evil"}),
    (wiring._run, {"model": "-x"}),
    (wiring._serve, {"model": "--engine"}),
    (wiring._benchmark, {"model": "--exec-consent", "use_case": "coding"}),
])
def test_workers_reject_flag_like_model(worker, args):
    with pytest.raises(ValueError):
        worker(args)


def test_worker_rejects_flag_like_engine():
    with pytest.raises(ValueError):
        wiring._characterize({"model": "org/m", "engine": "--bad"})


def test_serve_rejects_flag_like_name():
    with pytest.raises(ValueError):
        wiring._serve({"model": "org/m", "name": "--n"})


def test_benchmark_rejects_flag_like_use_case():
    with pytest.raises(ValueError):
        wiring._benchmark({"model": "org/m", "use_case": "--x"})


def test_safe_rejects_non_string():
    with pytest.raises(ValueError):
        wiring._safe(123, "model")
