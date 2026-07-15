# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The node wiring — providers/workers translate to CLI args, and ``_run_cli`` is the only seam."""
from __future__ import annotations

import inspect
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
    # The boundary always runs the canonical module CLI and appends --json itself.
    assert captured["cmd"] == [sys.executable, "-m", "ara", "detect", "--json"]


def test_run_cli_nonzero_exit_returns_error_dict(monkeypatch):
    monkeypatch.setattr(wiring.subprocess, "run",
                        lambda *a, **k: _FakeProc(1, stdout="", stderr="boom\n"))
    out = wiring._run_cli(["status"])
    assert out == {"error": "`ara status` exited 1", "stderr": "boom"}


def test_run_cli_nonzero_exit_preserves_valid_operational_json(monkeypatch):
    payload = {"error": "model has not been characterized", "model": "org/m"}
    monkeypatch.setattr(
        wiring.subprocess, "run",
        lambda *a, **k: _FakeProc(1, stdout=json.dumps(payload), stderr="internal detail\n"),
    )
    assert wiring._run_cli(["run", "--", "org/m"]) == {
        **payload, "stderr": "internal detail",
    }


@pytest.mark.parametrize("payload", [
    {}, {"warning": "partial"}, {"error": ""}, {"error": 7}, {"error": None},
])
def test_run_cli_nonzero_rejects_objects_without_real_error(monkeypatch, payload):
    monkeypatch.setattr(
        wiring.subprocess, "run",
        lambda *a, **k: _FakeProc(6, stdout=json.dumps(payload), stderr="daemon detail\n"),
    )
    assert wiring._run_cli(["status"]) == {
        "error": "`ara status` exited 6", "stderr": "daemon detail",
    }


def test_run_cli_nonzero_operational_error_retains_stderr(monkeypatch):
    monkeypatch.setattr(
        wiring.subprocess, "run",
        lambda *a, **k: _FakeProc(
            2, stdout=json.dumps({"error": "operation failed"}), stderr="actionable detail\n"),
    )
    assert wiring._run_cli(["run"]) == {
        "error": "operation failed", "stderr": "actionable detail",
    }


def test_run_cli_nonzero_operational_error_does_not_overwrite_stderr(monkeypatch):
    payload = {"error": "operation failed", "stderr": "public detail"}
    monkeypatch.setattr(
        wiring.subprocess, "run",
        lambda *a, **k: _FakeProc(2, stdout=json.dumps(payload), stderr="internal detail\n"),
    )
    assert wiring._run_cli(["run"]) == payload


def test_run_models_inventory_wraps_successful_array(monkeypatch):
    payload = [{"name": "HF cache", "present": True, "count": 1, "size_gb": 0.0}]
    monkeypatch.setattr(
        wiring.subprocess, "run",
        lambda *a, **k: _FakeProc(0, stdout=json.dumps(payload)),
    )
    assert wiring._run_models_inventory() == {"models": payload}


def test_run_models_inventory_preserves_nonzero_error_object(monkeypatch):
    payload = {"error": "inventory unavailable", "store": "HF cache"}
    monkeypatch.setattr(
        wiring.subprocess, "run",
        lambda *a, **k: _FakeProc(1, stdout=json.dumps(payload), stderr="detail\n"),
    )
    assert wiring._run_models_inventory() == {**payload, "stderr": "detail"}


@pytest.mark.parametrize("payload", [{"unexpected": []}, None, "text", 7, True])
def test_run_models_inventory_rejects_wrong_success_shape(monkeypatch, payload):
    monkeypatch.setattr(
        wiring.subprocess, "run",
        lambda *a, **k: _FakeProc(0, stdout=json.dumps(payload), stderr="shape detail\n"),
    )
    assert wiring._run_models_inventory() == {
        "error": "`ara detect` produced non-array model inventory",
        "stderr": "shape detail",
    }


@pytest.mark.parametrize("payload", [[], ["wrong"]])
def test_run_models_inventory_nonzero_array_is_generic_failure(monkeypatch, payload):
    monkeypatch.setattr(
        wiring.subprocess, "run",
        lambda *a, **k: _FakeProc(5, stdout=json.dumps(payload), stderr="failed detail\n"),
    )
    assert wiring._run_models_inventory() == {
        "error": "`ara detect` exited 5",
        "stderr": "failed detail",
    }


def test_run_cli_nonzero_malformed_output_falls_back_without_raising(monkeypatch):
    monkeypatch.setattr(
        wiring.subprocess, "run",
        lambda *a, **k: _FakeProc(7, stdout="not json", stderr="worker died\n"),
    )
    assert wiring._run_cli(["benchmark"]) == {
        "error": "`ara benchmark` exited 7",
        "stderr": "worker died",
    }


def test_run_cli_unparseable_output_returns_error_dict(monkeypatch):
    monkeypatch.setattr(wiring.subprocess, "run",
                        lambda *a, **k: _FakeProc(0, stdout="not json", stderr=" noise "))
    out = wiring._run_cli(["models"])
    assert out == {"error": "`ara models` produced unparseable output", "stderr": "noise"}


@pytest.mark.parametrize("payload", [None, [], ["x"], "text", 0, 1.5, True])
def test_run_cli_zero_rejects_valid_json_non_object(monkeypatch, payload):
    monkeypatch.setattr(
        wiring.subprocess, "run",
        lambda *a, **k: _FakeProc(0, stdout=json.dumps(payload), stderr="shape detail\n"),
    )
    assert wiring._run_cli(["status"]) == {
        "error": "`ara status` produced non-object JSON output",
        "stderr": "shape detail",
    }


@pytest.mark.parametrize("payload", [None, [], ["x"], "text", 0, 1.5, False])
def test_run_cli_nonzero_rejects_valid_json_non_object(monkeypatch, payload):
    monkeypatch.setattr(
        wiring.subprocess, "run",
        lambda *a, **k: _FakeProc(9, stdout=json.dumps(payload), stderr="failed detail\n"),
    )
    assert wiring._run_cli(["run"]) == {
        "error": "`ara run` exited 9",
        "stderr": "failed detail",
    }


def test_default_providers_map_each_key_to_its_canonical_cli_args(monkeypatch):
    calls = []
    monkeypatch.setattr(wiring, "_run_cli", lambda args: calls.append(args) or {"v": args[0]})
    monkeypatch.setattr(
        wiring, "_run_models_inventory",
        lambda: calls.append(["detect", "--models"]) or {"models": []},
    )
    providers = wiring.default_providers()
    assert set(providers) == {"status", "detect", "profile", "models"}
    expected = {
        "status": ["status"],
        "detect": ["detect"],
        "profile": ["profile"],
        "models": ["detect", "--models"],
    }
    for key, args in expected.items():
        expected_payload = {"models": []} if key == "models" else {"v": args[0]}
        assert providers[key]() == expected_payload
    assert calls == list(expected.values())


def test_default_models_provider_runs_real_canonical_cli_and_wraps_inventory(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    hf_home = home / "hf"
    repo = hf_home / "hub" / "models--org--cached"
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "llama", "num_hidden_layers": 2}),
        encoding="utf-8",
    )
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40, encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / "data"))

    payload = wiring.default_providers()["models"]()

    assert set(payload) == {"models"}
    assert len(payload["models"]) == 1
    assert payload["models"][0]["model_id"] == "org/cached"
    assert payload["models"][0]["n_layers"] == 2
    assert payload["models"][0]["characterized"] is False


def test_characterize_worker_builds_args(monkeypatch):
    seen = {}
    monkeypatch.setattr(wiring, "_run_cli", lambda args: seen.setdefault("a", args) or {})
    wiring._characterize({"model": "org/m"})
    assert seen["a"] == ["characterize", "--", "org/m"]


def test_characterize_worker_with_engine(monkeypatch):
    seen = {}
    monkeypatch.setattr(wiring, "_run_cli", lambda args: seen.setdefault("a", args) or {})
    wiring._characterize({"model": "org/m", "engine": "cpu"})
    assert seen["a"] == ["characterize", "--engine", "cpu", "--", "org/m"]


def test_run_worker_builds_args_with_engine_and_prompt(monkeypatch):
    seen = {}
    monkeypatch.setattr(wiring, "_run_cli", lambda args: seen.setdefault("a", args) or {})
    wiring._run({"model": "org/m", "engine": "cuda", "prompt": "hello world"})
    assert seen["a"] == ["run", "--engine", "cuda", "--yes", "--", "org/m", "hello world"]


def test_run_worker_without_optional_fields(monkeypatch):
    seen = {}
    monkeypatch.setattr(wiring, "_run_cli", lambda args: seen.setdefault("a", args) or {})
    wiring._run({"model": "org/m"})
    assert seen["a"] == ["run", "--yes", "--", "org/m"]


def test_serve_worker_full_args(monkeypatch):
    seen = {}
    monkeypatch.setattr(wiring, "_run_cli", lambda args: seen.setdefault("a", args) or {})
    wiring._serve({"model": "org/m", "engine": "cpu", "ctx": 4096, "name": "svc"})
    assert seen["a"] == ["serve", "--engine", "cpu", "--ctx", "4096",
                         "--name", "svc", "--yes", "--", "org/m"]


def test_serve_worker_minimal(monkeypatch):
    seen = {}
    monkeypatch.setattr(wiring, "_run_cli", lambda args: seen.setdefault("a", args) or {})
    wiring._serve({"model": "org/m"})
    assert seen["a"] == ["serve", "--yes", "--", "org/m"]


def test_benchmark_worker_full_args(monkeypatch):
    seen = {}
    monkeypatch.setattr(wiring, "_run_cli", lambda args: seen.setdefault("a", args) or {})
    wiring._benchmark({"model": "org/m", "use_case": "coding", "engine": "cuda",
                       "ctx": 8192, "max_tokens": 512, "exec_consent": True})
    assert seen["a"] == ["benchmark", "--use-case", "coding", "--engine", "cuda",
                         "--ctx", "8192", "--max-tokens", "512", "--exec-consent", "--yes",
                         "--", "org/m"]


def test_benchmark_worker_minimal_no_exec_consent(monkeypatch):
    seen = {}
    monkeypatch.setattr(wiring, "_run_cli", lambda args: seen.setdefault("a", args) or {})
    wiring._benchmark({"model": "org/m", "use_case": "rag"})
    # --exec-consent is never auto-added; the gate stays intact.
    assert seen["a"] == ["benchmark", "--use-case", "rag", "--yes", "--", "org/m"]


@pytest.mark.parametrize("not_consent", [False, "false", "true", 1, [], {}])
def test_benchmark_worker_requires_literal_true_exec_consent(monkeypatch, not_consent):
    seen = {}
    monkeypatch.setattr(wiring, "_run_cli", lambda args: seen.setdefault("a", args) or {})
    wiring._benchmark({"model": "org/m", "use_case": "coding",
                       "exec_consent": not_consent})
    assert "--exec-consent" not in seen["a"]


def test_run_cli_inserts_json_before_end_of_options_separator(monkeypatch):
    captured = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd
        return _FakeProc(0, stdout="{}")

    monkeypatch.setattr(wiring.subprocess, "run", fake_run)
    wiring._run_cli(["run", "--yes", "--", "org/m", "--prompt-like-text"])
    assert captured["cmd"] == [sys.executable, "-m", "ara", "run", "--yes", "--json", "--",
                               "org/m", "--prompt-like-text"]


def test_default_workers_keys():
    assert set(wiring.default_workers()) == {"characterize", "run", "serve", "benchmark"}


def test_node_workers_inherit_lifecycle_from_canonical_cli_without_second_registry():
    source = inspect.getsource(wiring)
    assert 'cli = ["run"' in source
    assert 'cli = ["serve"' in source
    assert "ara.activity" not in source
    assert "activity.track" not in source


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
