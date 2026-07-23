# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""CLI contract matrix — every command's public surface, end-to-end through ``cli.main``.

Each command is dispatched through the real entry point with all external boundaries mocked to
canned data, and its ``--json`` output is asserted to be valid, well-typed JSON with a stable
anchor key. This locks the *interface* (exit code + output shape) so it can't silently drift as
the internals churn — complementing the per-command behaviour tests in test_cli.py.
"""
from __future__ import annotations

import json
import types

import pytest

from ara import cli
from ara.detect import Accelerator, Machine, ModelStore, Runtime


def _machine() -> Machine:
    return Machine(
        system="Darwin", os_version="macOS 15.0", chip="Apple M4 Pro", arch="arm64",
        cpu_physical=12, cpu_logical=12, cpu_features=["NEON"],
        python_version="3.12.8", ram_total_gb=48.0, ram_available_gb=20.0, swap_gb=2.0,
        accel=Accelerator("apple", "Apple M4 Pro GPU", None, "Metal", cores=16),
        disk_free_gb=500.0, physical_memory_bytes=48 * 1024 ** 3,
        runtimes=[Runtime("MLX", True, "0.18", kind="engine", accels=("apple",), usable=True)],
        framework_python="/usr/bin/python3",
        model_stores=[ModelStore("HF cache", True, 3, 12.0)],
        hf_token=True, power="AC power", backend="apple", engine="mlx",
        engine_ready=True,
    )


class _FakeBackend:
    CALIBRATION_MODEL = "org/calib"

    def characterize(self, model, *, progress=False, kv_quant="f16"):
        return {"model": model, "safe_context": 8192, "decode_context": None,
                "binding": "context_window",
                "points": [{"context": 512, "mem_gb": 1.2}]}   # real dict-point shape

    def safe_limits(self):
        return {"device": "Apple M4 Pro", "total_gb": 48.0, "wall_gb": 40.0,
                "safe_budget_gb": 36.0, "margin_gb": 4.0, "headroom_gb": 28.0,
                "swap_free_gb": 2.0, "overhead_gb": None, "calibrated": True,
                "calibrated_at": None}

    def calibration_model_cached(self, model=CALIBRATION_MODEL):
        return True

    def download_calibration_model(self, model=CALIBRATION_MODEL, *, progress=False):
        pass

    def calibrate(self, model=CALIBRATION_MODEL):
        return self.safe_limits()


@pytest.fixture
def mocked_world(monkeypatch, store):
    """Stub every external boundary cli touches, so any command can run through main()."""
    m = _machine()
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli.detect, "chip_name", lambda: "Apple M4 Pro")
    monkeypatch.setattr(cli.detect, "accelerator",
                        lambda chip: Accelerator("apple", "Apple M4 Pro GPU", None, "Metal", cores=16))
    monkeypatch.setattr(cli.apps, "scan", lambda: [])
    monkeypatch.setattr(cli.activity, "snapshot", lambda: [])
    monkeypatch.setattr(cli.pythons, "discover", lambda: [])
    monkeypatch.setattr(cli.pythons, "count", lambda: 0)
    monkeypatch.setattr(cli.mlx, "scan", lambda: [])
    monkeypatch.setattr(cli.mlx, "lmstudio_mlx_runtimes", lambda: [])
    monkeypatch.setattr(cli.mlx, "mlx_community_model_count", lambda: 0)
    monkeypatch.setattr(cli.catalog, "scan", lambda con: None)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/m", "modality": "text"}])
    monkeypatch.setattr(cli.catalog, "describe", lambda model_id: {"modality": "text"})
    monkeypatch.setattr(cli.catalog, "remember", lambda con, model: None)
    monkeypatch.setattr(cli.staleness, "artifact_identity", lambda model: f"artifact:{model}")
    monkeypatch.setattr(
        cli.staleness, "pinned_model_ref", lambda model, _artifact, **_kwargs: model)
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.calibration, "get_calibration", lambda con, key, **kwargs: None)
    monkeypatch.setattr(
        cli.measurement_authority,
        "current_measurement_authority",
        lambda _engine: types.SimpleNamespace(
            key="mlx-authority:contract",
            environment_key="mlx-environment:contract",
            evidence={"schema": "mlx-memory-authority:v1"},
        ),
    )
    monkeypatch.setattr(cli.hub, "search", lambda q: [{"id": "org/m"}])
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "MLX engine"))
    monkeypatch.setattr(cli, "get_backend", lambda b=None: _FakeBackend())
    monkeypatch.setattr(cli.engine_audit, "audit_engine", lambda *_args, **_kwargs: {
        "key": "mlx", "package_version": "0.32.0",
        "build": {"status": "matched"}, "runtime": {"status": "matched"},
        "fingerprint": "engine:v1:sha256:contract",
    })
    monkeypatch.setattr(cli.engines, "install",
                        lambda key, **kw: cli.engines.InstallResult(key, "installed"))
    monkeypatch.setattr(cli.engines, "uninstall",
                        lambda key: cli.engines.InstallResult(key, "removed"))


# (argv, expected JSON type, an anchor key that must be present | None for a list)
CONTRACT = [
    (["detect", "--json"], dict, "backend"),
    (["detect", "--models", "--json"], list, None),
    (["status", "--json"], dict, "state"),
    (["python", "--json"], list, None),
    (["apps", "--json"], list, None),
    (["mlx", "--json"], dict, "apple_silicon"),
    (["models", "show", "org/m", "--json"], dict, "model_id"),
    (["models", "search", "smol", "--json"], list, None),
    (["characterize", "org/m", "--json"], dict, "safe_context"),
    (["profile", "--json"], dict, "device"),
    (["install", "--engine", "mlx", "--json"], dict, "status"),
    (["uninstall", "--engine", "mlx", "--json"], dict, "status"),
]


def _extract_json(out: str):
    """The JSON payload, even if a command printed a progress line before it (characterize)."""
    starts = [i for i in (out.find("{"), out.find("[")) if i != -1]
    assert starts, f"no JSON object/array in output: {out!r}"
    return json.loads(out[min(starts):])


@pytest.mark.parametrize("argv, kind, anchor", CONTRACT,
                         ids=[a[0] if len(a) == 1 else "_".join(a[:2]) for a, _, _ in CONTRACT])
def test_command_emits_valid_json(mocked_world, monkeypatch, capsys, argv, kind, anchor):
    monkeypatch.setattr("sys.argv", ["ara", *argv])
    rc = cli.main()
    assert rc == 0, f"{argv} exited {rc}"
    payload = _extract_json(capsys.readouterr().out)
    assert isinstance(payload, kind), f"{argv} → {type(payload).__name__}, want {kind.__name__}"
    if anchor is not None:
        assert anchor in payload, f"{argv} JSON missing '{anchor}'"


def test_models_recommend_json_errors_without_current_mlx_budget(
        mocked_world, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["ara", "models", "recommend", "--json"])
    assert cli.main() == 1
    assert _extract_json(capsys.readouterr().out) == {
        "error": "no current MLX budget is available for a safe ranking"
    }


def test_models_recommend_explicit_mlx_json_errors_without_current_budget(
        mocked_world, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv", ["ara", "models", "recommend", "--engine", "mlx", "--json"])
    assert cli.main() == 1
    assert _extract_json(capsys.readouterr().out) == {
        "error": "no current MLX budget is available for a safe ranking"
    }


def test_unknown_command_is_a_click_usage_error(mocked_world, capsys):
    assert cli.main(["frobnicate", "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No such command 'frobnicate'" in captured.err


def test_models_group_bare_and_help_show_generated_help(mocked_world, capsys):
    for argv in (["models"], ["models", "--help"]):
        assert cli.main(argv) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        usage = captured.out.splitlines()[0]
        assert usage in {
            "Usage: ara models [OPTIONS] COMMAND [ARGS]...",
            "Usage: ara models [OPTIONS] [COMMAND] [ARGS]...",
        }
        assert "search" in captured.out
        assert "recommend" in captured.out
        assert "show" in captured.out


def test_models_list_is_a_click_usage_error_not_a_model_alias(mocked_world, capsys):
    assert cli.main(["models", "list", "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No such command 'list'" in captured.err


@pytest.mark.parametrize("argv, warning", [
    (["search", "smol", "--json"],
     "ara: search is deprecated; use models search\n"),
    (["recommend", "--json"],
     "ara: recommend is deprecated; use models recommend\n"),
    (["models", "org/m", "--json"],
     "ara: models MODEL is deprecated; use models show MODEL\n"),
    (["python", "--json"],
     "ara: python is deprecated; use detect --python\n"),
    (["apps", "--json"],
     "ara: apps is deprecated; use detect --apps\n"),
    (["mlx", "--json"],
     "ara: mlx is deprecated; use detect --runtime\n"),
])
def test_deprecated_aliases_warn_only_on_stderr_and_keep_json_exact(
        mocked_world, capsys, argv, warning):
    expected_status = 1 if argv[0] == "recommend" else 0
    assert cli.main(argv) == expected_status
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    if argv[0] == "recommend":
        assert payload == {"error": "no current MLX budget is available for a safe ranking"}
    assert captured.err == warning


def test_deprecated_aliases_are_hidden_from_root_help(mocked_world, capsys):
    assert cli.main(["--help"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    command_names = {line.split()[0] for line in captured.out.split("Commands:\n", 1)[1].splitlines()
                     if line.startswith("  ")}
    assert command_names.isdisjoint({"search", "recommend", "python", "apps", "mlx"})


@pytest.mark.parametrize("argv, message", [
    (["characterize", "--json"], "Missing argument 'MODEL'"),
    (["benchmark", "org/m", "--json"], "Missing option '--use-case'"),
    (["profile", "--json", "--model"], "Option '--model' requires an argument"),
    (["serve", "org/m", "--json", "--ctx", "many"], "Invalid value for '--ctx'"),
    (["hub", "--port", "70000"], "Invalid value for '--port'"),
    (["benchmark", "org/m", "--json", "--use-case", "reasoning", "--repeat", "lots"],
     "Invalid value for '--repeat'"),
    (["detect", "--json", "--made-up"], "No such option '--made-up'"),
])
def test_click_grammar_errors_are_stderr_exit_two_and_never_json(
        mocked_world, capsys, argv, message):
    assert cli.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err
    with pytest.raises(json.JSONDecodeError):
        json.loads(captured.err)


def test_typed_options_and_trailing_json_reach_callback(mocked_world, monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(
        cli,
        "render_benchmark",
        lambda c, model, **kwargs: seen.update(model=model, **kwargs) or 7,
    )
    assert cli.main([
        "benchmark", "org/m", "--use-case=reasoning", "--ctx", "4096",
        "--max-tokens=512", "--repeat", "3", "--yes", "--json",
    ]) == 7
    assert seen == {
        "model": "org/m", "use_case": "reasoning", "engine": None, "ctx": 4096,
        "max_tokens": 512, "repeat": 3, "assume_yes": True,
        "exec_consent": False, "as_json": True,
    }
    assert capsys.readouterr().err == ""


def test_repeatable_include_exclude_accept_space_and_equals_forms(
        mocked_world, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        cli,
        "_resolve_want",
        lambda cmd, include, exclude, c, **kw: seen.update(
            cmd=cmd, include=include, exclude=exclude, **kw) or (lambda section: True),
    )
    assert cli.main([
        "detect", "--include", "system,memory", "--include=accelerator",
        "--exclude", "apps", "--exclude=models", "--json",
    ]) == 0
    assert seen == {
        "cmd": "detect", "include": ["system", "memory", "accelerator"],
        "exclude": ["apps", "models"], "as_json": True,
    }


@pytest.mark.parametrize("option", ["--include", "--exclude"])
def test_status_rejects_recon_filters_as_click_usage_errors(mocked_world, capsys, option):
    assert cli.main(["status", option, "processes"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Usage: ara status" in captured.err
    assert f"No such option '{option}'" in captured.err


@pytest.mark.parametrize("group", ["hf", "node"])
def test_explicit_groups_require_a_subcommand(mocked_world, capsys, group):
    assert cli.main([group]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"Usage: ara {group}" in captured.err
    assert "Missing command" in captured.err


@pytest.mark.parametrize("subcommand", ["logout", "status"])
def test_hf_subcommands_dispatch_through_click(mocked_world, monkeypatch, subcommand):
    seen = {}
    monkeypatch.setattr(
        cli,
        "render_hf",
        lambda c, sub, **kwargs: seen.update(sub=sub, **kwargs) or 0,
    )
    assert cli.main(["hf", subcommand, "--json"]) == 0
    assert seen == {"sub": subcommand, "as_json": True}


@pytest.mark.parametrize("subcommand", ["install", "start", "stop", "status", "uninstall"])
def test_node_lifecycle_subcommands_dispatch_through_click(
        mocked_world, monkeypatch, subcommand):
    seen = {}
    monkeypatch.setattr(
        cli,
        "render_node",
        lambda c, rest, **kwargs: seen.update(rest=rest, **kwargs) or 0,
    )
    assert cli.main(["node", subcommand, "--json"]) == 0
    assert seen == {"rest": ["node", subcommand], "as_json": True}


def test_detect_json_includes_accelerated(mocked_world, monkeypatch, capsys):
    # asdict() skips the @property; the CPU-fallback distinction must still reach --json consumers
    monkeypatch.setattr("sys.argv", ["ara", "detect", "--json"])
    cli.main()
    payload = _extract_json(capsys.readouterr().out)
    assert isinstance(payload.get("accelerated"), bool)


# --json error paths must emit a JSON {"error": ...}, never human text (a script piping
# `ara <cmd> --json` should always get parseable output, even on failure).
@pytest.mark.parametrize("argv, setup", [
    (["characterize", "org/m", "--json"], "engine_off"),
    (["profile", "--engine", "bogus", "--json"], "bad_engine"),   # profile is engine-free; only
    (["models", "bad/x", "--json"], "undescribable"),             # an unknown engine errors
])
def test_json_error_paths_emit_json(mocked_world, monkeypatch, capsys, argv, setup):
    if setup == "engine_off":
        monkeypatch.setattr(cli, "engine_status", lambda b=None: (False, "MLX engine"))
    elif setup == "undescribable":
        monkeypatch.setattr(cli.catalog, "describe", lambda model_id: None)
    # bad_engine needs no stub: resolve_engine raises UnknownEngine for 'bogus'
    monkeypatch.setattr("sys.argv", ["ara", *argv])
    assert cli.main() == 1
    payload = _extract_json(capsys.readouterr().out)
    assert "error" in payload
