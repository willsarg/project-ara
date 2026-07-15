# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""cli.py — formatters, arg parsing/dispatch, and the render_* surfaces."""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
import sqlite3
import sys
import types

import pytest

import ara.cli as cli
from ara.detect import Accelerator, Machine, ModelStore, Runtime
from ara.hardware import (BoardInfo, CpuInfo, Drive, MemoryInfo, MemoryModule, StorageInfo)


@pytest.fixture(autouse=True)
def _stable_test_artifact(monkeypatch):
    monkeypatch.setattr(cli.staleness, "artifact_identity", lambda _model: "artifact:test")
    monkeypatch.setattr(cli.staleness, "artifact_size_gb", lambda _model: 1.0)
    monkeypatch.setattr(
        cli.staleness, "pinned_model_ref", lambda model, _artifact: model)


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


# --------------------------------------------------------------------------- #
# main(): arg parsing + dispatch
# --------------------------------------------------------------------------- #
def _capture_dispatch(monkeypatch):
    """Replace the render_* entry points with recorders; return the record dict."""
    rec = {}
    monkeypatch.setattr(cli, "render_landing", lambda c: rec.update(landing=True))
    monkeypatch.setattr(cli, "render_detect", lambda c, as_json=False, want=None: rec.update(detect=as_json, detect_want=want))
    monkeypatch.setattr(cli, "render_status", lambda c, as_json=False: rec.update(status=as_json))
    monkeypatch.setattr(cli, "render_python", lambda c, as_json=False, want=None: rec.update(python=as_json))
    monkeypatch.setattr(cli, "render_apps", lambda c, as_json=False, want=None: rec.update(apps=as_json))
    monkeypatch.setattr(cli, "render_runtime", lambda c, as_json=False, want=None: rec.update(runtime=as_json))
    monkeypatch.setattr(cli, "render_mlx", lambda c, as_json=False, want=None: rec.update(mlx=as_json))
    monkeypatch.setattr(cli, "render_models", lambda c, as_json=False, want=None: rec.update(models=as_json))
    monkeypatch.setattr(cli, "render_characterize",
                        lambda c, m, engine=None, as_json=False, flash_attn=True,
                        flash_attn_optin=False, kv_quant="f16", weight_quant="none",
                        prefill_chunk=None:
                        (rec.update(characterize=m, characterize_engine=engine,
                                    characterize_fa=flash_attn, characterize_fa_optin=flash_attn_optin,
                                    characterize_kv=kv_quant, characterize_wq=weight_quant,
                                    characterize_chunk=prefill_chunk) or 0))
    monkeypatch.setattr(cli, "render_profile",
                        lambda c, **kw: (rec.update(profile=kw) or 0))
    monkeypatch.setattr(cli, "render_recommend",
                        lambda c, as_json=False, use_case=None:
                        (rec.update(recommend=as_json, recommend_uc=use_case) or 0))
    monkeypatch.setattr(cli, "render_run",
                        lambda c, model, **kw: (rec.update(run={"model": model, **kw}) or 0))
    monkeypatch.setattr(cli, "render_serve",
                        lambda c, model, **kw: (rec.update(serve={"model": model, **kw}) or 0))
    monkeypatch.setattr(cli, "render_benchmark",
                        lambda c, model, **kw: (rec.update(benchmark={"model": model, **kw}) or 0))
    monkeypatch.setattr(cli, "render_install", lambda c, **kw: (rec.update(install=kw) or 0))
    monkeypatch.setattr(cli, "render_uninstall", lambda c, **kw: (rec.update(uninstall=kw) or 0))
    monkeypatch.setattr(cli, "render_hf",
                        lambda c, sub, token=None, as_json=False:
                        (rec.update(hf=sub, hf_token=token) or 0))
    monkeypatch.setattr(cli, "render_node",
                        lambda c, rest, token=None, as_json=False:
                        (rec.update(node=rest[1:], node_token=token) or 0))
    monkeypatch.setattr(cli, "render_doctor",
                        lambda c, rekey=False, as_json=False:
                        (rec.update(doctor=True, doctor_rekey=rekey, doctor_json=as_json) or 0))
    return rec


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["ara", *argv])
    return cli.main()


def test_main_node_routes_subcommand(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["node", "run"]) == 0
    assert rec["node"] == ["run"]              # push-only node: no --host/--port plumbing


def test_main_node_enroll_threads_token(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["node", "enroll", "https://c.example", "--token", "ENR"]) == 0
    assert rec["node"] == ["enroll", "https://c.example"] and rec["node_token"] == "ENR"


def test_main_doctor_routes(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["doctor"]) == 0
    assert rec["doctor"] and rec["doctor_rekey"] is False


def test_main_doctor_rekey_flag(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["doctor", "--rekey", "--json"]) == 0
    assert rec["doctor_rekey"] is True and rec["doctor_json"] is True


def test_doctor_help_explains_its_purpose(capsys):
    assert cli.main(["--help"]) == 0
    root_help = capsys.readouterr().out
    assert "doctor        Diagnose ARA's stored identity and records for this machine." in root_help

    assert cli.main(["doctor", "--help"]) == 0
    doctor_help = capsys.readouterr().out
    assert ("Show how ARA identifies this machine, count records stored for it, and report records "
            "under other machine identities.") in " ".join(doctor_help.split())
    assert "Rewrite legacy machine identity keys in ARA's database." in doctor_help


def test_benchmark_help_explains_its_purpose_and_safety_contract(capsys):
    assert cli.main(["--help"]) == 0
    root_help = capsys.readouterr().out
    assert ("benchmark Measure MODEL under its characterized safe ceiling."
            in " ".join(root_help.split()))

    assert cli.main(["benchmark", "--help"]) == 0
    benchmark_help = capsys.readouterr().out
    normalized = " ".join(benchmark_help.split())
    assert ("Run a judge-free capability probe against MODEL's actual quant on the selected engine, "
            "then store the measured score for model recommendations.") in normalized
    assert "Requires a prior matching characterization; --ctx may lower, but never replace or exceed" in normalized
    assert "Measured capability category." in benchmark_help
    assert "Choices: auto, mlx, cuda, cpu, vulkan, cuda-gguf." in normalized
    assert "Authorize execution of coding-probe output." in normalized


def test_hf_help_explains_group_and_subcommands(capsys):
    expected = {
        (): "Manage Hugging Face authentication for gated model access.",
        ("login",): "Store and verify a Hugging Face token for gated models.",
        ("logout",): "Remove the locally stored Hugging Face token.",
        ("status",): "Check whether a Hugging Face token is active and verified.",
    }
    rendered = {}
    for path, summary in expected.items():
        assert cli.main(["hf", *path, "--help"]) == 0
        help_text = capsys.readouterr().out
        rendered[path] = help_text
        assert summary in " ".join(help_text.split())
    assert "visible in shell history and process lists" in rendered[("login",)]


def test_render_doctor_json_reports_key_and_counts(store, monkeypatch, capsys):
    from ara import db, profile
    monkeypatch.setattr(profile, "machine_key", lambda: "ara1|C|G|32|Linux")
    db.save_characterization(store, "ara1|C|G|32|Linux", "cpu", "m1", safe_context=8, points=[])
    db.save_characterization(store, "someoneelse|X", "cpu", "m2", safe_context=8, points=[])
    c = cli.Console.from_env()
    assert cli.render_doctor(c, as_json=True) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["machine_key"] == "ara1|C|G|32|Linux"
    assert out["counts"]["characterizations"] == 1      # only this machine's rows
    assert out["other_keys_rows"] >= 1                    # foreign/legacy rows counted separately


def test_render_doctor_rekey_reports_migrated_count(store, monkeypatch, capsys):
    from ara import db, profile
    monkeypatch.setattr(profile, "machine_key", lambda: "ara1|TestCPU|TestGPU|32|Linux")
    db.save_characterization(store, "TestCPU|TestGPU|34359738368|Linux", "cpu", "m1",
                             safe_context=8, points=[])
    c = cli.Console.from_env()
    assert cli.render_doctor(c, rekey=True, as_json=True) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["rekeyed_rows"] == 1


def test_render_doctor_text_variants(store, monkeypatch, make_console):
    from ara import db, profile
    monkeypatch.setattr(profile, "machine_key", lambda: "ara1|C|G|32|Linux")
    db.save_characterization(store, "someoneelse|X", "cpu", "m2", safe_context=8, points=[])

    # rekey=False with foreign rows present → no rekey line, "other machine keys" line shown
    c, buf = make_console()
    assert cli.render_doctor(c, rekey=False) == 0
    text = buf.getvalue()
    assert "rekeyed" not in text and "ara1|C|G|32|Linux" in text
    assert "other machine keys" in text and "characterizations" in text

    # rekey=True that migrates a legacy row → "rekeyed 1" line (good style)
    db.save_characterization(store, "C|G|34359738368|Linux", "cpu", "m1", safe_context=8, points=[])
    c, buf = make_console()
    assert cli.render_doctor(c, rekey=True) == 0
    assert "rekeyed 1 legacy row" in buf.getvalue()

    # rekey=True with nothing legacy left → "rekeyed 0" line (dim style); drop the foreign row so
    # the "other machine keys" line is skipped too
    db.save_characterization(store, "C|G|34359738368|Linux", "cpu", "m1", safe_context=8, points=[])
    with db.connected() as con:
        con.execute("DELETE FROM characterizations WHERE machine_key='someoneelse|X'")
        con.commit()
    # first call migrates the just-added legacy row; second finds nothing to rekey
    cli.render_doctor(make_console()[0], rekey=True)
    c, buf = make_console()
    assert cli.render_doctor(c, rekey=True) == 0
    out = buf.getvalue()
    assert "rekeyed 0 legacy row" in out and "other machine keys" not in out


def test_render_doctor_verbose_reports_store_details(store, monkeypatch, make_console, capsys):
    from ara import db, profile
    monkeypatch.setattr(profile, "machine_key", lambda: "ara1|C|G|32|Linux")
    c, buf = make_console(verbose=True)

    assert cli.render_doctor(c) == 0

    out = buf.getvalue()
    assert f"database             {db._db_path()}" in out
    assert "schema version       3" in out

    c = cli.Console.from_env(verbose=True)
    assert cli.render_doctor(c, as_json=True) == 0
    out_json = json.loads(capsys.readouterr().out)
    assert out_json["database"] == str(db._db_path())
    assert out_json["schema_version"] == 3


@pytest.mark.parametrize("as_json", [False, True])
def test_render_doctor_reports_database_failure(monkeypatch, make_console, capsys, as_json):
    from ara import db

    @contextlib.contextmanager
    def broken_store():
        raise sqlite3.DatabaseError("file is not a database")
        yield  # pragma: no cover - contextmanager requires an iterator

    monkeypatch.setattr(db, "connected", broken_store)
    c, buf = make_console()

    assert cli.render_doctor(c, as_json=as_json) == 1

    rendered = capsys.readouterr().out if as_json else buf.getvalue()
    assert "database problem" in rendered
    assert str(db._db_path()) in rendered
    assert "file is not a database" in rendered
    if as_json:
        assert json.loads(rendered)["error"].startswith("database problem")


def test_main_no_args_shows_landing(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, []) == 0
    assert rec == {"landing": True}


def test_main_help_shows_landing(monkeypatch):
    # Click owns the root grammar and both conventional help spellings.
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["-h"]) == 0
    assert rec == {}


def test_main_detect(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["detect"])
    assert rec["detect"] is False


def test_main_detect_json(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["detect", "--json"])
    assert rec["detect"] is True


def test_main_detect_python_facet_routes_to_render_python(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["detect", "--python"])
    assert "python" in rec and "detect" not in rec


def test_main_detect_apps_facet_routes_to_render_apps(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["detect", "--apps"])
    assert "apps" in rec and "detect" not in rec


def test_main_detect_first_facet_wins_python_then_apps(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["detect", "--python", "--apps"])
    assert "python" in rec and "apps" not in rec


def test_main_detect_first_facet_wins_apps_then_python(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["detect", "--apps", "--python"])
    assert "apps" in rec and "python" not in rec


def test_main_detect_runtime_facet_routes_to_cross_platform_runtime_report(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["detect", "--runtime"])
    assert "runtime" in rec and "detect" not in rec and "mlx" not in rec


def test_main_detect_models_facet_routes_to_render_models(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["detect", "--models"])
    assert "models" in rec and "detect" not in rec


def test_main_detect_bare_routes_to_render_detect_only(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["detect"])
    assert "detect" in rec and "python" not in rec and "apps" not in rec


def test_main_detect_models_facet_threads_json(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["detect", "--models", "--json"])
    assert rec["models"] is True


def test_main_status(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["status"])
    assert rec["status"] is False


def test_main_model_detail_filters_preserve_not_applicable_warning(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(
        cli,
        "render_model_detail",
        lambda c, model_id, **kwargs: seen.update(model_id=model_id, **kwargs) or 0,
    )
    assert _run_main(monkeypatch, ["models", "org/m", "--include", "system"]) == 0
    assert seen == {"model_id": "org/m", "as_json": False}
    assert "--include/--exclude don't apply to 'models'" in capsys.readouterr().out


@pytest.mark.parametrize("argv, expected", [
    (["models", "search", "smol model", "--json", "-v"],
     {"query": "smol model", "as_json": True}),
    (["models", "recommend", "--use-case", "coding", "--json", "--verbose"],
     {"as_json": True, "use_case": "coding"}),
    (["models", "show", "org/m", "--json", "-v"],
     {"model_id": "org/m", "as_json": True}),
])
def test_models_subcommands_dispatch_to_existing_renderers(monkeypatch, argv, expected):
    seen = {}
    if argv[1] == "search":
        monkeypatch.setattr(
            cli, "render_search",
            lambda c, query, **kwargs: seen.update(query=query, **kwargs) or 0,
        )
    elif argv[1] == "recommend":
        monkeypatch.setattr(
            cli, "render_recommend",
            lambda c, **kwargs: seen.update(**kwargs) or 0,
        )
    else:
        monkeypatch.setattr(
            cli, "render_model_detail",
            lambda c, model_id, **kwargs: seen.update(model_id=model_id, **kwargs) or 0,
        )
    assert _run_main(monkeypatch, argv) == 0
    assert seen == expected


# --no-flash-attn flag threading (vulkan FA is on by default). Slug: 2026-06-25-vulkan-flash-attention
def test_main_run_flash_attn_on_by_default(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["run", "org/m", "hello"])
    assert rec["run"]["flash_attn"] is True
    assert rec["run"]["model"] == "org/m"


def test_main_run_no_flash_attn_disables_and_is_not_in_prompt(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["run", "org/m", "hello", "--no-flash-attn"])
    assert rec["run"]["flash_attn"] is False
    assert rec["run"]["prompt"] == "hello"   # the flag is filtered out of the positional prompt


def test_run_help_explains_governance_and_generation_controls(capsys):
    assert cli.main(["run", "--help"]) == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "characterized safe ceiling" in out
    assert "auto, mlx, cuda, cpu, vulkan, cuda-gguf" in out
    assert "--max-tokens N" in out and "Maximum new tokens" in out
    assert "KV-cache format (mlx/cuda/vulkan): f16, q8_0, or q4_0" in out


@contextlib.contextmanager
def _busy_lock():
    raise cli.MeasurementBusy("another ARA measurement is already running on this machine")
    yield  # pragma: no cover — unreachable; makes this a generator for @contextmanager


def test_characterize_refused_when_a_measurement_is_running_json(monkeypatch, capsys):
    """A concurrent measurement holds the lock → refuse with a clear message, don't corrupt the
    reading (Rule #1, G9). Routed through main() so the front-door guard formats it."""
    monkeypatch.setattr(cli.locking, "measurement_lock", _busy_lock)
    assert _run_main(monkeypatch, ["characterize", "org/m", "--json"]) == 1
    assert "already running" in json.loads(capsys.readouterr().out)["error"]


def test_benchmark_refused_when_a_measurement_is_running_text(monkeypatch, capsys):
    monkeypatch.setattr(cli.locking, "measurement_lock", _busy_lock)
    assert _run_main(monkeypatch, ["benchmark", "org/m", "--use-case", "coding"]) == 1
    assert "already running" in capsys.readouterr().out


def test_main_characterize_no_flash_attn(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["characterize", "org/m", "--no-flash-attn"])
    assert rec["characterize_fa"] is False


def test_main_characterize_flash_attn_on_by_default(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["characterize", "org/m"])
    assert rec["characterize_fa"] is True


def test_characterize_help_explains_measurement_and_engine_specific_choices(capsys):
    assert cli.main(["characterize", "--help"]) == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "Safely measure MODEL's real context ceiling by loading it on an engine" in out
    assert "auto, mlx, cuda, cpu, vulkan, cuda-gguf, ollama" in out
    assert "KV-cache format (mlx/cuda/vulkan): f16, q8_0, or q4_0" in out
    assert "CUDA weight format: none, int8, int4, or fp8" in out
    assert "selected engine's safety boundary" in out


# --kv-quant flag threading (default f16). Slug: 2026-06-25-vulkan-kv-cache-quant
def test_main_characterize_kv_quant_value(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["characterize", "org/m", "--kv-quant", "q8_0"])
    assert rec["characterize_kv"] == "q8_0"


def test_main_characterize_kv_quant_equals_form(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["characterize", "org/m", "--kv-quant=q4_0"])
    assert rec["characterize_kv"] == "q4_0"


def test_main_characterize_kv_quant_default_f16(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["characterize", "org/m"])
    assert rec["characterize_kv"] == "f16"


def test_main_run_kv_quant_value_not_in_prompt(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["run", "org/m", "hello", "--kv-quant", "q8_0"])
    assert rec["run"]["kv_quant"] == "q8_0"
    assert rec["run"]["prompt"] == "hello"   # the value flag + its value are filtered from prompt


def test_main_profile_model_separate_value(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["profile", "--model", "org/repo"])
    assert rec["profile"]["model"] == "org/repo"


def test_main_profile_model_flag_requires_value(monkeypatch, capsys):
    _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["profile", "--model"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Option '--model' requires an argument" in captured.err


def test_main_profile_model_equals_form(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["profile", "--model=org/repo"])
    assert rec["profile"]["model"] == "org/repo"


def test_main_profile_passes_json(monkeypatch):
    # profile is engine-free analytic now — it takes --json/--model/--engine, no calibrate flags.
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["profile", "--json"])
    assert rec["profile"]["as_json"] is True


def test_main_unknown_command_is_click_usage_error(monkeypatch, capsys):
    _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["frobnicate"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No such command 'frobnicate'" in captured.err


def test_main_version_flag_prints_installed_version(monkeypatch, capsys):
    from importlib.metadata import version
    assert _run_main(monkeypatch, ["--version"]) == 0
    assert capsys.readouterr().out.strip() == version("project-ara")


def test_ara_version_falls_back_when_not_installed(monkeypatch):
    # _ara_version now lives in engines.py (it also stamps engine envs); cli re-exports it.
    def _missing(name):
        raise cli.engines.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(cli.engines.metadata, "version", _missing)
    assert cli._ara_version() == "0+unknown"


# ---- --help / -h routing (per subcommand, no side effects) --------------------------------
def test_main_subcommand_help_routes_to_help_not_action(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["install", "--help"]) == 0
    assert "install" not in rec                 # Click exits before the action callback


def test_main_dash_h_after_command_routes_to_help(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["characterize", "-h"]) == 0
    assert "characterize" not in rec


def test_main_bare_help_routes_to_help_with_no_command(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["--help"]) == 0
    assert rec == {}


def test_generated_install_help_is_stdout(monkeypatch, capsys):
    assert _run_main(monkeypatch, ["install", "--help"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Usage: ara install [OPTIONS] [ENGINE]" in captured.out
    assert "ENGINE_ARG" not in captured.out
    assert "--engine ENGINE" in captured.out
    assert "Engines: auto, mlx, cuda, cpu, vulkan, cuda-gguf." in captured.out
    assert "ara install --engine --help" in captured.out
    assert "Backend/env:" not in captured.out


def test_generated_uninstall_help_names_public_argument_and_removal_boundary(monkeypatch, capsys):
    assert _run_main(monkeypatch, ["uninstall", "--help"]) == 0
    captured = capsys.readouterr()

    assert captured.err == ""
    assert "Usage: ara uninstall [OPTIONS] [ENGINE]" in captured.out
    assert "ENGINE_ARG" not in captured.out
    assert "Engines: auto, mlx, cuda, cpu, vulkan, cuda-gguf." in captured.out
    assert "Keeps models, the shared uv cache, ARA's database and characterizations," in captured.out
    assert "other engines." in captured.out


def test_install_engine_help_lists_every_canonical_choice(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.engines, "auto_decision",
        lambda: cli.engines.AutoDecision(
            "mlx", "Darwin arm64 identifies Apple Silicon, so ARA selects MLX.",
            "Darwin", "arm64", None),
    )

    assert _run_main(monkeypatch, ["install", "--engine", "--help"]) == 0
    captured = capsys.readouterr()

    assert captured.err == ""
    for key in ("auto", "mlx", "cuda", "cpu", "vulkan", "cuda-gguf"):
        assert captured.out.count(f"Engine: {key}\n") == 1
    assert "wmx" not in captured.out and "wcx" not in captured.out
    assert "Backend/env:" not in captured.out


def test_install_focused_engine_help_only_describes_requested_engine(monkeypatch, capsys):
    assert _run_main(monkeypatch, ["install", "--engine", "mlx", "--help"]) == 0
    captured = capsys.readouterr()

    assert captured.err == ""
    assert "Engine: mlx\n" in captured.out
    for key in ("auto", "cuda", "cpu", "vulkan", "cuda-gguf"):
        assert f"Engine: {key}\n" not in captured.out
    assert "Apple Silicon" in captured.out
    assert "MLX" in captured.out
    assert "Backend/env:" not in captured.out


def test_install_verbose_engine_help_includes_exact_plan(monkeypatch, capsys):
    monkeypatch.delenv("ARA_MLX_SOURCE", raising=False)
    monkeypatch.delenv("ARA_WMX_SOURCE", raising=False)
    monkeypatch.setattr(cli.engines.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.engines.platform, "machine", lambda: "arm64")

    assert _run_main(
        monkeypatch, ["install", "--engine", "mlx", "--help", "--verbose"]
    ) == 0
    captured = capsys.readouterr()

    assert captured.err == ""
    assert "Engine: mlx\n" in captured.out
    assert "Backend/env: apple" in captured.out
    assert "Python: 3.12" in captured.out
    assert "Platform: Darwin arm64" in captured.out
    assert "Install arguments:" in captured.out
    assert str(cli.engines._bundled_source("mlx")) in captured.out
    assert "Source override: none (ARA_MLX_SOURCE)" in captured.out


def test_install_auto_help_explains_current_selection_and_reason(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.engines, "auto_decision",
        lambda: cli.engines.AutoDecision(
            "cuda", "nvidia-smi is available on PATH, so ARA selects CUDA.",
            "Windows", "AMD64", "nvidia-smi.exe"),
    )

    assert _run_main(monkeypatch, ["install", "--engine", "auto", "--help"]) == 0
    captured = capsys.readouterr()

    assert captured.err == ""
    assert "Engine: auto\n" in captured.out
    assert "Selected: cuda" in captured.out
    assert "Why: nvidia-smi is available on PATH, so ARA selects CUDA." in captured.out
    assert "Engine: cuda\n" in captured.out
    assert "Engine: mlx\n" not in captured.out


def test_install_auto_help_reports_no_match_honestly(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.engines, "auto_decision",
        lambda: cli.engines.AutoDecision(
            None,
            "Linux x86_64 is not Apple Silicon and nvidia-smi is not available on PATH, "
            "so ARA has no automatic match.",
            "Linux", "x86_64", None),
    )

    assert _run_main(monkeypatch, ["install", "--engine", "auto", "--help"]) == 0
    captured = capsys.readouterr()

    assert captured.err == ""
    assert "Selected: no automatic match" in captured.out
    assert "ARA has no automatic match" in captured.out
    assert "Choose cpu, vulkan, cuda-gguf, or another engine explicitly." in captured.out


def test_install_contextual_help_supports_equals_short_help_and_short_verbose(monkeypatch, capsys):
    assert _run_main(monkeypatch, ["install", "--engine=cpu", "-h", "-v"]) == 0
    captured = capsys.readouterr()

    assert captured.err == ""
    assert "Engine: cpu\n" in captured.out
    assert "Backend/env: cpu" in captured.out


def test_install_contextual_help_legacy_alias_warns_and_renders_canonical(monkeypatch, capsys):
    assert _run_main(monkeypatch, ["install", "--engine", "wmx", "--help"]) == 0
    captured = capsys.readouterr()

    assert captured.err == "ara: --engine wmx is deprecated; use --engine mlx\n"
    assert "Engine: mlx\n" in captured.out
    assert "Engine: wmx\n" not in captured.out


def test_install_contextual_help_unknown_engine_is_click_usage_error(monkeypatch, capsys):
    assert _run_main(monkeypatch, ["install", "--engine", "bogus", "--help"]) == 2
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "Usage: ara install [OPTIONS] [ENGINE]" in captured.err
    assert "Invalid value for --engine" in captured.err
    assert "bogus" in captured.err


def test_install_contextual_help_never_dispatches_install(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.engines, "install", lambda *args, **kwargs: pytest.fail("help attempted an install"))

    assert _run_main(monkeypatch, ["install", "--engine", "cpu", "--help"]) == 0
    assert capsys.readouterr().err == ""


# ---- install / uninstall take a positional engine (not just --engine) ---------------------
def test_main_install_honors_legacy_positional_engine(monkeypatch, capsys):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["install", "wmx"])
    assert rec["install"]["engine"] == "mlx"
    assert "--engine wmx is deprecated; use --engine mlx" in capsys.readouterr().err


def test_main_legacy_alias_with_json_keeps_stdout_parseable(monkeypatch, capsys):
    monkeypatch.setattr(cli.engines, "install",
                        lambda key, **kw: cli.engines.InstallResult(key, "installed"))
    assert _run_main(monkeypatch, ["install", "wcx", "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["key"] == "cuda"
    assert captured.err == "ara: --engine wcx is deprecated; use --engine cuda\n"


def test_main_uninstall_legacy_positional_with_json_keeps_stdout_parseable(monkeypatch, capsys):
    monkeypatch.setattr(cli.engines, "uninstall",
                        lambda key: cli.engines.InstallResult(key, "removed"))
    assert _run_main(monkeypatch, ["uninstall", "wcx", "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["key"] == "cuda"
    assert captured.err == "ara: --engine wcx is deprecated; use --engine cuda\n"


@pytest.mark.parametrize("command,args,field", [
    ("characterize", ["model", "--engine", "wmx"], "characterize_engine"),
    ("profile", ["--engine=wcx"], "profile"),
    ("run", ["model", "prompt", "--engine", "wmx"], "run"),
    ("serve", ["model", "--engine=wcx"], "serve"),
    ("benchmark", ["model", "--use-case", "rag", "--engine", "wmx"], "benchmark"),
])
def test_main_canonicalizes_legacy_engine_flags(monkeypatch, capsys, command, args, field):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, [command, *args])
    value = rec[field]
    engine = value["engine"] if isinstance(value, dict) else value
    assert engine in ("mlx", "cuda")
    assert "is deprecated; use --engine" in capsys.readouterr().err


def test_main_install_engine_flag_still_works(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["install", "--engine", "cpu"])
    assert rec["install"]["engine"] == "cpu"


def test_main_uninstall_honors_positional_engine(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["uninstall", "vulkan"])
    assert rec["uninstall"]["engine"] == "vulkan"


# ---- HF auth polish: quiet the generic warning, nudge toward `ara hf login` ----------------
def test_hf_hint_nudges_when_unauthenticated(monkeypatch, make_console):
    monkeypatch.setattr(cli.hf_auth, "has_token", lambda: False)
    c, buf = make_console()
    cli._hf_hint(c, as_json=False)
    assert "ara hf login" in buf.getvalue()


def test_hf_hint_silent_when_authenticated(monkeypatch, make_console):
    monkeypatch.setattr(cli.hf_auth, "has_token", lambda: True)
    c, buf = make_console()
    cli._hf_hint(c, as_json=False)
    assert buf.getvalue() == ""


def test_hf_hint_silent_under_json(monkeypatch, make_console):
    monkeypatch.setattr(cli.hf_auth, "has_token", lambda: False)
    c, buf = make_console()
    cli._hf_hint(c, as_json=True)
    assert buf.getvalue() == ""


def test_render_search_nudges_to_login_when_unauthenticated(monkeypatch, make_console):
    monkeypatch.setattr(cli.hub, "search", lambda q: [{"id": "org/m", "downloads": 1, "likes": 2}])
    monkeypatch.setattr(cli.hf_auth, "has_token", lambda: False)
    c, buf = make_console()
    assert cli.render_search(c, "llama") == 0
    assert "ara hf login" in buf.getvalue()




def test_main_verbose_flag_sets_console(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "render_detect",
                        lambda c, as_json=False, want=None: captured.update(verbose=c.verbose))
    _run_main(monkeypatch, ["detect", "--verbose"])
    assert captured["verbose"] is True


# --------------------------------------------------------------------------- #
# render_landing
# --------------------------------------------------------------------------- #
def test_landing_hardware_apple_shows_unified_memory_and_gpu_cores():
    # Apple: memory is the unified pool ARA governs; GPU shown by core count.
    assert cli._landing_hardware("Apple M4 Pro", "apple", 24.0, 16, None) == [
        "Apple M4 Pro", "24 GB unified memory", "16-core GPU"]


def test_landing_hardware_cpu_shows_plain_ram_no_gpu():
    assert cli._landing_hardware("Intel i7", "cpu", 32.0, None, None) == [
        "Intel i7", "32 GB RAM"]


def test_landing_hardware_cuda_names_the_discrete_gpu():
    assert cli._landing_hardware("Ryzen 9", "cuda", 32.0, None, "NVIDIA RTX 4090") == [
        "Ryzen 9", "32 GB RAM", "NVIDIA RTX 4090"]


def test_landing_hardware_omits_memory_when_unknown():
    assert cli._landing_hardware("Apple M1", "apple", None, None, None) == ["Apple M1"]


def test_render_landing_supported(make_console, monkeypatch):
    monkeypatch.setattr(cli.detect, "chip_name", lambda: "Apple M4 Pro")
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli.detect, "_memory_gb", lambda: (24.0, 5.0))
    monkeypatch.setattr(cli.detect, "accelerator",
                        lambda chip: cli.detect.Accelerator("apple", f"{chip} GPU", None, "Metal", cores=16))
    c, buf = make_console()
    cli.render_landing(c)
    out = buf.getvalue()
    assert "ara" in out and "Apple M4 Pro" in out
    assert "24 GB unified memory" in out and "16-core GPU" in out   # hardware, in plain terms
    assert "ara-engine-mlx" not in out and "backend apple" not in out    # no internal jargon on the line
    assert "GETTING STARTED" in out
    assert "detect" in out and "status" in out and "profile" in out
    assert "--python" in out and "--runtime" in out   # detect facet hint (python/apps/mlx collapsed in)
    assert "CPU fallback" not in out


def test_render_landing_cpu_fallback_notes_no_gpu(make_console, monkeypatch):
    monkeypatch.setattr(cli.detect, "chip_name", lambda: "Intel i7")
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cpu")
    monkeypatch.setattr(cli.detect, "_memory_gb", lambda: (32.0, 8.0))
    monkeypatch.setattr(cli.detect, "accelerator",
                        lambda chip: cli.detect.Accelerator("none", "CPU", None, None))
    c, buf = make_console()
    cli.render_landing(c)
    out = buf.getvalue()
    assert "32 GB RAM" in out                  # plain 'RAM' label off Apple
    assert "no GPU backend detected" in out and "ara install --engine cpu" in out
    assert "mlx" not in out                    # no stray internal jargon on a non-Apple box


def test_cmd_long_name_keeps_gap_before_gloss(make_console):
    """A command label longer than the alignment column must not collide with its gloss."""
    c, _ = make_console()  # color off → plain text
    row = cli._cmd(c, "characterize <model>", "measure a model's safe context ceiling here")
    assert "<model>measure" not in row     # the bug: label runs straight into the gloss
    assert "<model>  measure" in row       # at least a two-space gap


def test_cmd_short_name_stays_column_aligned(make_console):
    """Short labels still align to the fixed command column (no regression)."""
    c, _ = make_console()
    row = cli._cmd(c, "detect", "inspect this machine")
    assert "detect" + " " * 10 + "inspect" in row   # 6-char name padded to the 16-col gutter


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
        hf_token=True, power="AC power", backend="apple", engine="mlx",
        engine_ready=False,
    )
    base.update(over)
    return Machine(**base)


def test_render_detect_text(make_console, monkeypatch, stub_pythons):
    stub_pythons(count=3)
    monkeypatch.setattr(cli.detect, "machine", lambda: _machine())
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
    assert "interpreters on this machine" in out
    assert "ara detect --python" in out
    assert "3 models" in out


# ollama liveness in the ENGINES section (2026-06-26-detect-ollama-liveness)
def test_det_engines_serving(make_console):
    c, buf = make_console()
    m = _machine(runtimes=[Runtime("Ollama", True, "0.30.10", kind="engine", serving=True)])
    cli._det_engines(c, m)
    out = buf.getvalue()
    assert "Ollama 0.30.10" in out
    assert "serving" in out


def test_det_engines_installed_not_serving_guides_serve(make_console):
    c, buf = make_console()
    m = _machine(runtimes=[Runtime("Ollama", True, "0.30.10", kind="engine", serving=False)])
    cli._det_engines(c, m)
    out = buf.getvalue()
    assert "not serving" in out
    assert "ollama serve" in out         # actionable guidance, not suppression


def test_det_engines_serving_none_renders_found(make_console):
    c, buf = make_console()
    m = _machine(runtimes=[Runtime("llama.cpp", True, "9780", kind="engine")])
    cli._det_engines(c, m)
    out = buf.getvalue()
    assert "found" in out
    assert "serving" not in out          # non-server engines carry no serving state


def test_render_detect_nvidia_accel(make_console, monkeypatch, stub_pythons):
    stub_pythons(count=1)
    m = _machine(accel=Accelerator("nvidia", "RTX 4090", 24.0, "CUDA", count=1,
                                   compute="8.9", cuda_version="12.4", driver_version="550.00"))
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
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
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
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
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=False)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "12 cores" in out                  # cpu row emitted, no features tail
    assert "your default python3" not in out  # python row skipped entirely
    assert "ARA's python" not in out


def test_render_detect_json(monkeypatch, capsys):
    monkeypatch.setattr(cli.detect, "machine", lambda: _machine())
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_detect(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["chip"] == "Apple M4 Pro"
    assert payload["accel"]["api"] == "Metal"
    assert payload["backend"] == "apple"


# --------------------------------------------------------------------------- #
# render_status — ARA-owned activity only
# --------------------------------------------------------------------------- #
def _activity(**over):
    from ara.activity import Activity
    base = dict(kind="running", model="org/model", pid=1234, started_at=100.0)
    base.update(over)
    return Activity(**base)


def test_render_status_empty(make_console, monkeypatch):
    monkeypatch.setattr(cli.activity, "snapshot", lambda: [])
    c, buf = make_console()
    cli.render_status(c)
    assert buf.getvalue() == "ARA is idle.\n"


@pytest.mark.parametrize("kind,model,expected", [
    ("characterizing", "org/model", "ARA is characterizing org/model.\n"),
    ("benchmarking", "org/model", "ARA is benchmarking org/model.\n"),
    ("searching", None, "ARA is searching for models.\n"),
    ("running", "org/model", "ARA is running org/model.\n"),
    ("serving", "org/model", "ARA is serving org/model.\n"),
])
def test_render_status_single_activity_text(make_console, monkeypatch, kind, model, expected):
    monkeypatch.setattr(cli.activity, "snapshot", lambda: [_activity(kind=kind, model=model)])
    c, buf = make_console()
    cli.render_status(c)
    assert buf.getvalue() == expected


def test_render_status_multiple_activities_shows_every_activity(make_console, monkeypatch):
    monkeypatch.setattr(cli.activity, "snapshot", lambda: [
        _activity(kind="benchmarking", model="org/a", pid=1, started_at=10.0),
        _activity(kind="serving", model="org/b", pid=2, started_at=20.0),
    ])
    c, buf = make_console()
    cli.render_status(c)
    assert buf.getvalue() == (
        "ARA is active:\n"
        "  benchmarking org/a\n"
        "  serving org/b\n"
    )


@pytest.mark.parametrize("activities,state", [
    ([], "idle"),
    ([_activity(kind="characterizing")], "characterizing"),
    ([_activity(kind="benchmarking")], "benchmarking"),
    ([_activity(kind="searching", model=None)], "searching"),
    ([_activity(kind="running")], "running"),
    ([_activity(kind="serving")], "serving"),
    ([_activity(), _activity(kind="serving", pid=5)], "active"),
])
def test_render_status_json_has_exact_state_and_activity_shape(
        monkeypatch, capsys, activities, state):
    monkeypatch.setattr(cli.activity, "snapshot", lambda: activities)
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_status(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "state": state,
        "activities": [
            {
                "kind": item.kind,
                **({"model": item.model} if item.model is not None else {}),
                "pid": item.pid,
                "started_at": item.started_at,
            }
            for item in activities
        ],
    }


# --------------------------------------------------------------------------- #
# render_profile — every branch, via a fake backend
# --------------------------------------------------------------------------- #
def _limits(calibrated=False, **over):
    base = dict(
        device="Apple M4 Pro", total_gb=48.0, wall_gb=40.0, safe_budget_gb=36.0,
        margin_gb=4.0, headroom_gb=28.0, overhead_gb=6.0, swap_free_gb=2.0,
        calibrated=calibrated, calibrated_at="2026-06-18" if calibrated else None,
        basis="estimated",
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

    def download_calibration_model(self, model, *, progress=False):
        self.downloaded.append(model)

    def calibrate(self, model=None):   # real backends default model=CALIBRATION_MODEL
        if self.calibrate_exc:
            raise self.calibrate_exc
        return self.calibrate_result


def _wire_profile(monkeypatch, set_platform, machine=None):
    """Wire render_profile engine-free (Spec 2026-06-23-capability-pipeline, Slice 2 Task 2):
    a stubbed Machine + machine_key on Apple. profile makes NO engine call — there is
    deliberately no backend wired here."""
    set_platform("Darwin", "arm64")  # resolve_engine(None) -> apple/mlx
    monkeypatch.setattr(cli.detect, "machine", lambda: machine if machine is not None else _machine())
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")


def test_profile_never_loads_an_engine(make_console, monkeypatch, set_platform, store):
    # The defining property: profile is analytic. If it reached for a backend, this would blow up.
    _wire_profile(monkeypatch, set_platform)
    monkeypatch.setattr(cli, "get_backend",
                        lambda *a, **k: pytest.fail("profile loaded an engine"))
    c, buf = make_console()
    assert cli.render_profile(c) == 0
    out = buf.getvalue()
    assert "SAFE LIMITS" in out and "estimated" in out
    assert "ara characterize" in out                              # points at the empirical step
    assert cli.db.get_latest_profile(store, "mkey") is None       # profile is read-only


def test_profile_does_not_create_database(tmp_path, make_console, monkeypatch, set_platform):
    path = tmp_path / "missing" / "ara.db"
    monkeypatch.setenv("ARA_DB_PATH", str(path))
    _wire_profile(monkeypatch, set_platform)

    c, _ = make_console()
    assert cli.render_profile(c) == 0
    assert not path.exists()


def test_profile_estimated_budget_mirrors_wall(make_console, monkeypatch, set_platform):
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    c, buf = make_console()
    assert cli.render_profile(c) == 0
    out = buf.getvalue()
    assert "36.0 GB" in out          # Apple working set: 0.75 × 48
    assert "34.0 GB" in out          # safe budget: wall − 2 GB margin


def test_profile_json_emits_estimate(monkeypatch, set_platform, capsys):
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_profile(c, as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["basis"] == "estimated"
    assert payload["calibrated"] is False
    assert payload["safe_budget_gb"] == 48.0 * 0.75 - 2.0


def test_profile_explicit_cpu_uses_system_ram_on_cuda_host(
        monkeypatch, set_platform, capsys):
    machine = _machine(
        backend="cuda", ram_total_gb=64.0,
        accel=Accelerator("nvidia", "RTX 2070", 8.0, "CUDA", count=1),
    )
    _wire_profile(monkeypatch, set_platform, machine)
    c = cli.Console(color=False, stream=sys.stderr)

    assert cli.render_profile(c, as_json=True, engine="cpu") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"] == "cpu"
    assert payload["total_gb"] == 64.0
    assert payload["wall_gb"] == 64.0


def test_profile_text_names_selected_engine(make_console, monkeypatch, set_platform):
    _wire_profile(monkeypatch, set_platform)
    c, buf = make_console()

    assert cli.render_profile(c, engine="cpu") == 0
    assert "engine      cpu" in buf.getvalue()


def test_profile_text_handles_unknown_memory(make_console, monkeypatch, set_platform):
    _wire_profile(monkeypatch, set_platform,
                  _machine(backend="cpu", ram_total_gb=None, chip="Unknown CPU"))
    c, buf = make_console()

    assert cli.render_profile(c, engine="cpu") == 0
    out = buf.getvalue()
    assert "Unknown CPU · unknown" in out
    assert "crash wall unknown" in " ".join(out.split())


def test_profile_reports_measured_wall_after_calibration(make_console, monkeypatch, set_platform, store):
    # Spec 2026-06-23-capability-pipeline: once a measured wall is stored for the detected engine,
    # profile reports the MEASURED numbers (labelled), not the heuristic.
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    cli.calibration.save_calibration(store, "mlx", fixed_overhead_gb=1.7,
                                     wall_gb=41.3, safe_budget_gb=39.3)
    c, buf = make_console()
    assert cli.render_profile(c) == 0
    out = buf.getvalue()
    assert "41.3 GB" in out and "39.3 GB" in out      # the measured wall + budget
    assert "not calibrated" not in out                # not an estimate anymore


def test_profile_json_reports_measured_basis(monkeypatch, set_platform, capsys, store):
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    cli.calibration.save_calibration(store, "mlx", fixed_overhead_gb=1.7,
                                     wall_gb=41.3, safe_budget_gb=39.3)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_profile(c, as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["basis"] == "measured"
    assert payload["calibrated"] is True
    assert payload["wall_gb"] == 41.3 and payload["safe_budget_gb"] == 39.3


def test_profile_reads_legacy_calibration_without_migrating(
        monkeypatch, set_platform, capsys, store):
    from ara import db

    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    store.execute(
        "INSERT INTO calibrations "
        "(machine_key, engine, fixed_overhead_gb, calibrated_at, wall_gb, safe_budget_gb) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("mkey", "wmx", 1.7, "2026-06-19T12:00:00+00:00", 41.3, 39.3),
    )
    store.execute("PRAGMA user_version = 2")
    store.commit()
    store.close()
    path = db._db_path()
    backup_path = path.with_name(path.name + ".pre-engine-identity-v3.bak")
    backup_path.unlink(missing_ok=True)

    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_profile(c, as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"] == "mlx"
    assert payload["basis"] == "measured"
    assert payload["wall_gb"] == 41.3
    with sqlite3.connect(path) as check:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 2
        assert check.execute("SELECT engine FROM calibrations").fetchone()[0] == "wmx"
    assert not backup_path.exists()


def test_profile_uncalibrated_stays_estimated(monkeypatch, set_platform, capsys, store):
    # No stored wall → profile must STILL say estimated (no fabrication).
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_profile(c, as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["basis"] == "estimated" and payload["calibrated"] is False


def test_profile_footer_drops_estimated_when_measured(make_console, monkeypatch, set_platform, store):
    # Spec 2026-06-23-capability-pipeline: once a wall is measured, the header reads (measured),
    # so the footer must NOT contradict it with the "estimated —" framing.
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    cli.calibration.save_calibration(store, "mlx", fixed_overhead_gb=1.7,
                                     wall_gb=41.3, safe_budget_gb=39.3)
    c, buf = make_console()
    assert cli.render_profile(c) == 0
    out = buf.getvalue()
    assert "estimated" not in out                       # header is (measured); footer must agree
    assert "ara characterize <model>" in out            # still nudges to measure more models
    assert "a model's real ceiling" in out


def test_profile_footer_keeps_estimated_when_uncalibrated(make_console, monkeypatch, set_platform, store):
    # No measured wall → the footer keeps the honest "estimated —" framing.
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    c, buf = make_console()
    assert cli.render_profile(c) == 0
    out = buf.getvalue()
    assert "estimated — run " in out
    assert "ara characterize <model>" in out


def test_profile_model_fits_full_window(make_console, monkeypatch, set_platform):
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.catalog, "describe",
                        lambda m: dict(n_layers=32, kv_heads=8, head_dim=128, max_context=8192))
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: 4.0)
    c, buf = make_console()
    assert cli.render_profile(c, model="org/small") == 0
    out = buf.getvalue()
    assert "MODEL FIT: org/small" in out
    assert "full 8192 ctx" in out


def test_profile_model_context_limited(make_console, monkeypatch, set_platform):
    # A tiny budget binds before the model's window → context-limited verdict.
    _wire_profile(monkeypatch, set_platform, _machine(backend="cpu", ram_total_gb=8.0))
    monkeypatch.setattr(cli.catalog, "describe",
                        lambda m: dict(n_layers=32, kv_heads=8, head_dim=128, max_context=131072))
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: 4.0)
    c, buf = make_console()
    assert cli.render_profile(c, model="org/big-ctx") == 0
    assert "context-limited" in buf.getvalue()


def test_profile_model_wont_fit(make_console, monkeypatch, set_platform):
    _wire_profile(monkeypatch, set_platform, _machine(backend="cpu", ram_total_gb=8.0))
    monkeypatch.setattr(cli.catalog, "describe",
                        lambda m: dict(n_layers=32, kv_heads=8, head_dim=128, max_context=8192))
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: 20.0)   # weights > budget
    c, buf = make_console()
    assert cli.render_profile(c, model="org/huge") == 0
    assert "won't fit" in buf.getvalue()


def test_profile_model_fits_unknown_architecture(make_console, monkeypatch, set_platform):
    # Describable + fits, but missing dims → no slope → "fits" with an honest unknown-context note.
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.catalog, "describe",
                        lambda m: dict(n_layers=None, kv_heads=None, head_dim=None, max_context=8192))
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: 4.0)
    c, buf = make_console()
    assert cli.render_profile(c, model="org/odd") == 0
    assert "context estimate unavailable" in buf.getvalue()


def test_profile_model_undescribable(make_console, monkeypatch, set_platform):
    _wire_profile(monkeypatch, set_platform)
    monkeypatch.setattr(cli.catalog, "describe", lambda m: None)
    c, buf = make_console()
    assert cli.render_profile(c, model="org/mystery") == 0
    assert "couldn't describe org/mystery" in buf.getvalue()


def test_profile_model_uses_cataloged_weight_no_network(
        make_console, monkeypatch, set_platform, store):
    # profile --model and recommend must compute identically: a cataloged model's weight comes
    # from the local catalog (catalog.get → weights_gb), never a network repo_size_gb call.
    # Spec 2026-06-23-capability-pipeline.
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.catalog, "describe",
                        lambda m: dict(n_layers=32, kv_heads=8, head_dim=128, max_context=8192))
    monkeypatch.setattr(cli.catalog, "get",
                        lambda con, m: {"model_id": m, "weights_gb": 4.0})
    monkeypatch.setattr(cli.acquire, "repo_size_gb",
                        lambda m: pytest.fail("repo_size_gb hit the network for a cataloged model"))
    c, buf = make_console()
    assert cli.render_profile(c, model="org/small") == 0
    assert "full 8192 ctx" in buf.getvalue()        # the 4.0 GB cataloged weight drove the fit


def test_profile_model_falls_back_to_network_when_uncataloged(make_console, monkeypatch, set_platform):
    # No catalog weight (model not cataloged, or weights_gb None) → fall back to repo_size_gb.
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.catalog, "describe",
                        lambda m: dict(n_layers=32, kv_heads=8, head_dim=128, max_context=8192))
    monkeypatch.setattr(cli.catalog, "get", lambda con, m: None)
    monkeypatch.setattr(cli.catalog, "remember",
                        lambda con, m: pytest.fail("profile wrote to the model catalog"))
    monkeypatch.setattr(cli.catalog, "_cache_size_gb", lambda m: None)
    called = []
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: called.append(m) or 4.0)
    c, buf = make_console()
    assert cli.render_profile(c, model="org/fresh") == 0
    assert called == ["org/fresh"]                   # network fallback fired
    assert "full 8192 ctx" in buf.getvalue()


def test_profile_model_json_includes_fit(monkeypatch, set_platform, capsys):
    _wire_profile(monkeypatch, set_platform, _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.catalog, "describe",
                        lambda m: dict(n_layers=32, kv_heads=8, head_dim=128, max_context=8192))
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: 4.0)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_profile(c, as_json=True, model="org/small") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_fit"]["fits"] is True
    assert payload["model_fit"]["max_context"] == 8192


def test_main_profile_passes_engine(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["profile", "--engine", "mlx"])
    assert rec["profile"]["engine"] == "mlx"


def test_profile_help_explains_analytic_boundary_and_options(capsys):
    assert cli.main(["profile", "--help"]) == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "Estimate this machine's safe memory budget" in out
    assert "without loading an engine or model" in out
    assert "Estimate whether MODEL fits and its usable context" in out
    assert "Estimate for ENGINE; defaults to the detected engine" in out


def test_profile_unknown_engine_errors(make_console, monkeypatch):
    c, buf = make_console()
    assert cli.render_profile(c, engine="bogus") == 1
    assert "unknown engine" in buf.getvalue().lower()


def test_emit_characterized_shows_stored_models(make_console, store, monkeypatch):
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    cli.db.save_characterization(store, "mkey", "cuda", "org/SmolLM", safe_context=16000, points=[],
                                 decode_context=None)
    cli.db.save_characterization(store, "mkey", "cuda", "org/Unbound", safe_context=None, points=[],
                                 decode_context=None)
    c, buf = make_console()
    cli._emit_characterized(c, "cuda")
    out = buf.getvalue()
    assert "CHARACTERIZED" in out and "SmolLM" in out and "16000" in out
    assert "—" in out               # the None-ceiling model


def test_emit_characterized_empty_shows_nothing(make_console, store, monkeypatch):
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    c, buf = make_console()
    cli._emit_characterized(c, "cuda")
    assert buf.getvalue() == ""


def test_emit_characterized_none_engine_key(make_console):
    c, buf = make_console()
    cli._emit_characterized(c, None)
    assert buf.getvalue() == ""


# --------------------------------------------------------------------------- #
# _emit_limits helper
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


def test_emit_limits_measured_reads_as_measured(make_console):
    # A measured profile must NOT read as estimated. Spec 2026-06-23-capability-pipeline.
    c, buf = make_console()
    cli._emit_limits(c, _limits(calibrated=True, basis="measured"))
    out = buf.getvalue()
    assert "measured" in out
    assert "estimated" not in out and "not calibrated" not in out


def test_emit_limits_estimated_says_estimated(make_console):
    c, buf = make_console()
    cli._emit_limits(c, _limits(calibrated=False, basis="estimated"))
    assert "not calibrated" in buf.getvalue()


def test_emit_limits_verbose_estimated_names_readonly_provenance(make_console):
    c, buf = make_console(verbose=True)
    cli._emit_limits(c, _limits(calibrated=False, basis="estimated"))
    out = " ".join(buf.getvalue().split())
    assert "provenance analytic estimate read-only hardware facts" in out


def test_emit_limits_verbose_measured_shows_calibration_and_analytic_baseline(make_console):
    c, buf = make_console(verbose=True)
    cli._emit_limits(c, _limits(
        calibrated=True, basis="measured", calibrated_at="2026-07-02T19:25:54+00:00",
        estimated_wall_gb=36.0, estimated_safe_budget_gb=34.0,
    ))
    out = " ".join(buf.getvalue().split())
    assert "provenance stored measurement calibrated 2026-07-02T19:25:54+00:00" in out
    assert "analytic wall 36.0 GB before measured correction" in out
    assert "analytic budget 34.0 GB before measured correction" in out


def test_emit_limits_verbose_measured_keeps_missing_provenance_unknown(make_console):
    c, buf = make_console(verbose=True)
    cli._emit_limits(c, _limits(calibrated=True, basis="measured", calibrated_at=None))
    out = " ".join(buf.getvalue().split())
    assert "provenance stored measurement calibrated unknown" in out
    assert "analytic wall" not in out and "analytic budget" not in out
    assert "None" not in out


# --------------------------------------------------------------------------- #
# render_detect verbose + CPU-fallback branches
# --------------------------------------------------------------------------- #
def test_render_detect_verbose_and_cpu_fallback(monkeypatch, make_console, stub_pythons):
    # `accelerated` is a property — drive it via backend="cpu" (no GPU-class adapter).
    stub_pythons(count=1, discover=[])
    m = _machine(
        backend="cpu", engine="llama.cpp", engine_ready=False,
        cpu_logical=24, hf_token=False,
        accel=Accelerator("none", "none detected", None, None),
        runtimes=[Runtime("Ollama", False, None, kind="engine"),
                  Runtime("PyTorch", False, None, kind="framework")],
        model_stores=[ModelStore("HF cache", False)],
    )
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "physical" in out and "logical" in out      # verbose cpu line
    assert "CPU fallback — no GPU backend detected" in out   # _det_ara backend hint
    assert "install: ara install" in out               # engine install hint (not gated)
    # verbose lists absent engine and absent store as "not found"
    assert out.count("not found") >= 2


def test_gpu_line_shows_gtt_shared_pool_for_apu(make_console):
    """An APU's tiny VRAM carveout must not be shown as its whole GPU memory — surface the GTT
    shared pool too (Rule #3: reading only mem_info_vram_total under-sold APUs ~2.4x)."""
    from ara.hardware import GpuInfo
    g = GpuInfo(vendor="amd", name="Phoenix1", vram_gb=0.5, gtt_gb=24.0, integrated=True)
    c, buf = make_console()
    cli._gpu_line(c, g)
    out = buf.getvalue()
    assert "carveout" in out and "GTT" in out and "24 GB" in out


def test_gpu_line_discrete_gpu_shows_no_gtt(make_console):
    from ara.hardware import GpuInfo
    g = GpuInfo(vendor="nvidia", name="RTX 3070", vram_gb=8.0, integrated=False)
    c, buf = make_console()
    cli._gpu_line(c, g)
    out = buf.getvalue()
    assert "8 GB" in out and "GTT" not in out and "carveout" not in out


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
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=False)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "Mystery GPU" in out and "VRAM" not in out    # nvidia, but no detail bits
    # no third-party launchers, non-verbose: the note must NOT read as a bare "none" that
    # contradicts the ARA section's own-engine readiness — it points there instead (bug fix).
    assert "own engine" in out and "ENGINES" in out
    assert "no separate user Python found" in out
    assert "Your default python" not in out
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
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console()
    cli.render_detect(c)
    out = buf.getvalue()
    assert "has no AI frameworks" in out
    assert "But you've got them in" in out
    assert "Homebrew 3.12.4" in out
    assert "torch 2.1.0" in out and "transformers 4.40.0" in out
    assert "ara detect --python" in out


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
    monkeypatch.setattr(cli, "Path", types.SimpleNamespace(home=lambda: r"C:\Users\dev"))
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


def test_resolve_want_suppresses_advisory_warnings_under_json(make_console):
    # Rule #3 (Honesty): advisory --include/--exclude warnings are styled text; under --json they
    # would corrupt the parse, so they're suppressed. The section filter still applies.
    c, buf = make_console()
    pred = cli._resolve_want("detect", ["bogus"], [], c, as_json=True)
    assert buf.getvalue() == "" and pred is not None


def test_resolve_want_command_without_sections_quiet_under_json(make_console):
    c, buf = make_console()
    assert cli._resolve_want("profile", ["x"], [], c, as_json=True) is None
    assert buf.getvalue() == ""


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
    monkeypatch.setattr(cli.versions, "cask_auto_updates", lambda tokens: {})  # no auto_updates known
    c, buf = make_console()
    cli.render_apps(c)
    out = buf.getvalue()
    assert "AI/ML APPS" in out and "model runners" in out
    assert "LM Studio 0.3.5" in out
    # the per-app clueless-drift gloss — NOT the bare "clobber" (the footer legend's "can clobber"
    # is always present, so a bare substring check would pass even if the per-app gloss broke).
    assert "self-updated past brew" in out and "will clobber it" in out
    assert "likely duplicate" in out                              # ollama dup


def test_render_apps_drift_with_auto_updates_is_benign(make_console, monkeypatch):
    inv = [_capp(label="Claude", category="assistant", cask=True, in_app=True,
                 version="2.0", brew_recorded="1.0", cask_token="claude")]
    monkeypatch.setattr(cli.apps, "scan", lambda: inv)
    monkeypatch.setattr(cli.versions, "cask_auto_updates", lambda tokens: {"claude": True})
    c, buf = make_console()
    cli.render_apps(c)
    out = buf.getvalue()
    # benign drift: "brew defers", not the clueless "will clobber" gloss (the footer
    # legend's "can clobber" is always present, so assert on the per-app wording).
    assert "brew defers" in out and "will clobber it" not in out


def test_render_apps_empty(make_console, monkeypatch):
    monkeypatch.setattr(cli.apps, "scan", lambda: [])
    monkeypatch.setattr(cli.versions, "cask_auto_updates", lambda tokens: {})
    c, buf = make_console()
    cli.render_apps(c)
    assert "none detected" in buf.getvalue()


def test_render_apps_want_filters_category(make_console, monkeypatch):
    inv = [_capp(label="Ollama", category="runner", cask=True, cask_token="ollama"),
           _capp(label="Cursor", category="coding", cask=True, cask_token="cursor")]
    monkeypatch.setattr(cli.apps, "scan", lambda: inv)
    monkeypatch.setattr(cli.versions, "cask_auto_updates", lambda tokens: {})
    c, buf = make_console()
    cli.render_apps(c, want=lambda k: k == "coding")
    out = buf.getvalue()
    assert "Cursor" in out and "Ollama" not in out


def test_render_apps_json(monkeypatch, capsys):
    inv = [_capp(label="LM Studio", cask=True, in_app=True, version="0.3.5",
                 brew_recorded="0.3.0", cask_token="lm-studio")]
    monkeypatch.setattr(cli.apps, "scan", lambda: inv)
    monkeypatch.setattr(cli.versions, "cask_auto_updates", lambda tokens: {"lm-studio": False})
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
    assert "ara detect --apps" in out


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
    monkeypatch.setattr(cli.detect, "machine", lambda: _machine())
    c, buf = make_console()
    cli.render_detect(c, want=lambda k: k == "system")
    out = buf.getvalue()
    assert "SYSTEM" in out and "MEMORY" not in out and "ACCELERATOR" not in out


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


# --------------------------------------------------------------------------- #
# ara install / ara uninstall — engine bootstrap commands
# --------------------------------------------------------------------------- #
def test_render_install_installs_resolved_engine(make_console, monkeypatch):
    monkeypatch.setattr(cli.engines, "resolve", lambda v: "mlx")
    monkeypatch.setattr(cli.engines, "install",
                        lambda k, **kw: cli.engines.InstallResult("mlx", "installed", "ok"))
    c, buf = make_console()
    rc = cli.render_install(c, engine="auto")
    assert rc == 0
    assert "ara-engine-mlx" in buf.getvalue()
    assert "installed" in buf.getvalue().lower()


def _stub_install(monkeypatch, key, status, detail=""):
    monkeypatch.setattr(cli.engines, "resolve", lambda v: key)
    monkeypatch.setattr(cli.engines, "install",
                        lambda k, **kw: cli.engines.InstallResult(k, status, detail))


def test_render_install_already_present_is_success(make_console, monkeypatch):
    _stub_install(monkeypatch, "mlx", "already")
    c, buf = make_console()
    assert cli.render_install(c, engine="mlx") == 0
    assert "already" in buf.getvalue().lower()


def test_render_install_refreshed_is_success(make_console, monkeypatch):
    _stub_install(monkeypatch, "mlx", "refreshed")
    c, buf = make_console()
    assert cli.render_install(c, engine="mlx") == 0
    assert "refreshed" in buf.getvalue().lower()


def test_render_install_threads_refresh_to_engines(make_console, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.engines, "resolve", lambda v: "mlx")
    monkeypatch.setattr(cli.engines, "install",
                        lambda k, *, refresh=False:
                        seen.update(refresh=refresh) or cli.engines.InstallResult(k, "refreshed"))
    c, _ = make_console()
    assert cli.render_install(c, engine="mlx", refresh=True) == 0
    assert seen == {"refresh": True}


def test_render_install_refreshed_json_is_success(monkeypatch, capsys):
    _stub_install(monkeypatch, "mlx", "refreshed")
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_install(c, engine="mlx", as_json=True)
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "refreshed" and rc == 0


def test_render_install_coming_soon_exits_nonzero(make_console, monkeypatch):
    _stub_install(monkeypatch, "cuda", "coming_soon", "CUDA engine isn't available yet")
    c, buf = make_console()
    assert cli.render_install(c, engine="cuda") == 1
    assert "coming soon" in buf.getvalue().lower()


def test_render_install_failed_shows_detail_and_exits_nonzero(make_console, monkeypatch):
    _stub_install(monkeypatch, "mlx", "failed", "git clone exploded")
    c, buf = make_console()
    assert cli.render_install(c, engine="mlx") == 1
    assert "git clone exploded" in buf.getvalue()


def test_render_install_no_hardware_match_exits_nonzero(make_console, monkeypatch):
    monkeypatch.setattr(cli.engines, "resolve", lambda v: None)
    c, buf = make_console()
    assert cli.render_install(c, engine="auto") == 1
    assert "no engine" in buf.getvalue().lower()


def test_render_install_json(monkeypatch, capsys):
    _stub_install(monkeypatch, "mlx", "installed", "ok")
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_install(c, engine="mlx", as_json=True)
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "installed" and out["key"] == "mlx" and rc == 0


def test_render_install_json_legacy_source_warning_keeps_stdout_clean(monkeypatch, capsys):
    monkeypatch.delenv("ARA_MLX_SOURCE", raising=False)
    monkeypatch.setenv("ARA_WMX_SOURCE", "../legacy-wmx-suite")
    monkeypatch.setattr(cli.engines, "resolve", lambda value: "mlx")

    def fake_install(key, *, refresh=False):
        assert cli.engines._install_targets(key) == ["-e", "../legacy-wmx-suite"]
        return cli.engines.InstallResult(key, "installed")

    monkeypatch.setattr(cli.engines, "install", fake_install)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_install(c, engine="mlx", as_json=True) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"key": "mlx", "status": "installed", "detail": ""}
    assert captured.err == "ara: ARA_WMX_SOURCE is deprecated; use ARA_MLX_SOURCE\n"


def _stub_uninstall(monkeypatch, key, status, detail=""):
    monkeypatch.setattr(cli.engines, "resolve", lambda v: key)
    monkeypatch.setattr(cli.activity, "snapshot", lambda: [])
    monkeypatch.setattr(cli.engines, "uninstall",
                        lambda k: cli.engines.InstallResult(k, status, detail))


def test_render_uninstall_removes_engine(make_console, monkeypatch):
    _stub_uninstall(monkeypatch, "mlx", "removed")
    c, buf = make_console()
    assert cli.render_uninstall(c, engine="mlx") == 0
    assert "removed" in buf.getvalue().lower() and "ara-engine-mlx" in buf.getvalue()


def test_render_uninstall_absent_is_success(make_console, monkeypatch):
    _stub_uninstall(monkeypatch, "mlx", "absent")
    c, buf = make_console()
    assert cli.render_uninstall(c, engine="mlx") == 0
    assert "not installed" in buf.getvalue().lower()


def test_render_uninstall_verbose_shows_exact_scope(make_console, monkeypatch):
    _stub_uninstall(monkeypatch, "mlx", "removed")
    monkeypatch.setattr(cli.engines.engine_env, "env_path", lambda name: cli.Path("/engines") / name)
    c, buf = make_console(verbose=True)

    assert cli.render_uninstall(c, engine="mlx") == 0

    out = buf.getvalue()
    assert "environment: /engines/apple" in out
    assert "kept: models, shared uv cache, ARA database/characterizations, and other engines" in out


@pytest.mark.parametrize("kind", ["running", "characterizing", "benchmarking", "serving"])
def test_render_uninstall_refuses_during_engine_backed_activity(
        make_console, monkeypatch, kind):
    called = []
    monkeypatch.setattr(cli.engines, "resolve", lambda _value: "mlx")
    monkeypatch.setattr(
        cli.activity, "snapshot",
        lambda: [cli.activity.Activity(kind, "org/model", 42, 1.0)],
    )
    monkeypatch.setattr(cli.engines, "uninstall", lambda key: called.append(key))
    c, buf = make_console()

    assert cli.render_uninstall(c, engine="mlx") == 1

    assert called == []
    assert "refusing to remove ara-engine-mlx while ARA work is active" in buf.getvalue()
    assert f"active: {kind}" in buf.getvalue()
    assert "ara status" in buf.getvalue()


def test_render_uninstall_busy_json_is_structured(monkeypatch, capsys):
    monkeypatch.setattr(cli.engines, "resolve", lambda _value: "cuda")
    monkeypatch.setattr(
        cli.activity, "snapshot",
        lambda: [cli.activity.Activity("running", "org/model", 42, 1.0)],
    )
    monkeypatch.setattr(
        cli.engines, "uninstall", lambda _key: pytest.fail("busy uninstall reached removal"))
    c = cli.Console(color=False, stream=sys.stderr)

    assert cli.render_uninstall(c, engine="cuda", as_json=True) == 1

    assert json.loads(capsys.readouterr().out) == {
        "status": "busy",
        "engine": "cuda",
        "activities": [{"kind": "running", "model": "org/model"}],
    }


def test_render_uninstall_ignores_search_and_persistent_ollama_serving(
        make_console, monkeypatch):
    monkeypatch.setattr(cli.engines, "resolve", lambda _value: "cpu")
    monkeypatch.setattr(
        cli.activity, "snapshot",
        lambda: [
            cli.activity.Activity("searching", None, 42, 1.0),
            cli.activity.Activity(
                "serving", "org/model", None, 2.0, runtime="ollama",
                served_name="model-ara", context=4096, endpoint="http://localhost:11434"),
        ],
    )
    monkeypatch.setattr(
        cli.engines, "uninstall",
        lambda key: cli.engines.InstallResult(key, "removed"),
    )
    c, buf = make_console()

    assert cli.render_uninstall(c, engine="cpu") == 0
    assert "removed llama.cpp" in buf.getvalue()


def test_render_uninstall_failed_exits_nonzero(make_console, monkeypatch):
    _stub_uninstall(monkeypatch, "mlx", "failed", "permission denied")
    c, buf = make_console()
    assert cli.render_uninstall(c, engine="mlx") == 1
    assert "permission denied" in buf.getvalue()


def test_render_uninstall_no_match_exits_nonzero(make_console, monkeypatch):
    monkeypatch.setattr(cli.engines, "resolve", lambda v: None)
    c, buf = make_console()
    assert cli.render_uninstall(c, engine="auto") == 1
    assert "no engine" in buf.getvalue().lower()


def test_render_uninstall_json(monkeypatch, capsys):
    _stub_uninstall(monkeypatch, "mlx", "removed")
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_uninstall(c, engine="mlx", as_json=True)
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "removed" and rc == 0


def test_main_install_defaults_to_auto(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["install"]) == 0
    assert rec["install"] == {"engine": "auto", "refresh": False, "as_json": False}


def test_main_install_with_engine_flag(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["install", "--engine", "mlx"])
    assert rec["install"]["engine"] == "mlx"


def test_main_install_engine_equals_form(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["install", "--engine=cuda", "--json"])
    assert rec["install"] == {"engine": "cuda", "refresh": False, "as_json": True}


def test_main_install_refresh_flag_forwarded(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["install", "mlx", "--refresh"])
    assert rec["install"]["engine"] == "mlx" and rec["install"]["refresh"] is True


def test_main_uninstall_with_engine_flag(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["uninstall", "--engine", "mlx"])
    assert rec["uninstall"]["engine"] == "mlx"


def test_render_install_no_match_json(monkeypatch, capsys):
    monkeypatch.setattr(cli.engines, "resolve", lambda v: None)
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_install(c, engine="auto", as_json=True)
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "no_match" and rc == 1


def test_render_uninstall_no_match_json(monkeypatch, capsys):
    monkeypatch.setattr(cli.engines, "resolve", lambda v: None)
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_uninstall(c, engine="auto", as_json=True)
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "no_match" and rc == 1


def test_render_landing_lists_install_command(make_console, monkeypatch):
    monkeypatch.setattr(cli.detect, "chip_name", lambda: "Apple M4 Pro")
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (False, "MLX engine"))
    c, buf = make_console()
    cli.render_landing(c)
    assert "install the engine" in buf.getvalue()


# --------------------------------------------------------------------------- #
# ara models — the catalog view (scan HF cache + list with characterization)
# --------------------------------------------------------------------------- #
def test_detect_models_does_not_create_database(tmp_path, monkeypatch, capsys):
    path = tmp_path / "missing" / "ara.db"
    monkeypatch.setenv("ARA_DB_PATH", str(path))
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models", lambda con: [])

    assert cli.main(["detect", "--models", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert not path.exists()


def test_detect_models_reads_existing_database_without_migrating(
        store, monkeypatch, capsys):
    from ara import db

    path = db._db_path()
    backup_path = path.with_name(path.name + ".pre-engine-identity-v3.bak")
    store.execute(
        "INSERT INTO characterizations "
        "(machine_key, engine, model_id, safe_context, points_json) VALUES (?, ?, ?, ?, ?)",
        ("mkey", "wmx", "org/model", 4096, "[]"),
    )
    store.execute("PRAGMA user_version = 2")
    store.commit()
    store.close()
    backup_path.unlink(missing_ok=True)
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/model", "modality": "text"}])
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")

    assert cli.main(["detect", "--models", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["characterized"] is True
    assert payload[0]["safe_context"] == 4096
    assert payload[0]["engine"] == "mlx"
    with sqlite3.connect(path) as check:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 2
        assert check.execute("SELECT engine FROM characterizations").fetchone()[0] == "wmx"
    assert not backup_path.exists()


def test_render_models_lists_catalog(make_console, store, monkeypatch):
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/A", "modality": "text"},
                                     {"model_id": "org/B", "modality": "text"}])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, e: [{"model_id": "org/A", "safe_context": 16000,
                                             "decode_context": None,
                                             "config": {"weight_quant": "int4"}}])
    c, buf = make_console()
    cli.render_models(c)
    out = buf.getvalue()
    assert "MODEL CATALOG" in out
    assert "org/A" in out and "16000" in out
    assert "weight-quant=int4" in out
    assert "org/B" in out and "not characterized" in out
    assert "2 cataloged" in out


def test_render_models_distinguishes_measured_no_ceiling(make_console, store, monkeypatch):
    """A model ARA measured but couldn't fit (row with safe_context=None) reads as
    characterized-with-no-ceiling — not lumped with never-measured models. Keeps
    `ara models` consistent with `ara profile`'s CHARACTERIZED list (which shows '—')."""
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/Fits", "modality": "text"},
                                     {"model_id": "org/NoCeiling", "modality": "text"},
                                     {"model_id": "org/Never", "modality": "text"}])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, e: [{"model_id": "org/Fits", "safe_context": 16000,
                                             "decode_context": None},
                                            {"model_id": "org/NoCeiling", "safe_context": None,
                                             "decode_context": None}])
    c, buf = make_console()
    cli.render_models(c)
    lines = {ln.split()[0]: ln for ln in buf.getvalue().splitlines() if "org/" in ln}
    assert "16000" in lines["org/Fits"]
    assert "no safe ceiling" in lines["org/NoCeiling"]
    assert "not characterized" not in lines["org/NoCeiling"]
    assert "not characterized" in lines["org/Never"]      # the only never-measured one
    assert "2 characterized on this machine" in buf.getvalue()


def test_render_models_empty_and_no_engine(make_console, store, monkeypatch):
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models", lambda con: [])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "unsupported")  # engine_key None
    c, buf = make_console()
    cli.render_models(c)
    assert "empty" in buf.getvalue()


def test_render_models_json(monkeypatch, capsys, store):
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/A", "modality": "text"}])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, e: [{"model_id": "org/A", "safe_context": 9000,
                                             "decode_context": None,
                                             "config": {"weight_quant": "int4"}}])
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_models(c, as_json=True)
    data = json.loads(capsys.readouterr().out)
    assert data[0]["safe_context"] == 9000
    assert data[0]["config"] == {"weight_quant": "int4"}


def test_render_models_json_has_characterized_flag(monkeypatch, capsys, store):
    """Models JSON carries a `characterized` flag so a null ceiling that was *measured*
    (no fit) is distinguishable from one that was *never measured*."""
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/Fits", "modality": "text"},
                                     {"model_id": "org/NoCeiling", "modality": "text"},
                                     {"model_id": "org/Never", "modality": "text"}])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, e: [{"model_id": "org/Fits", "safe_context": 16000,
                                             "decode_context": None},
                                            {"model_id": "org/NoCeiling", "safe_context": None,
                                             "decode_context": None}])
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_models(c, as_json=True)
    data = {d["model_id"]: d for d in json.loads(capsys.readouterr().out)}
    assert data["org/Fits"].get("characterized") is True
    assert data["org/NoCeiling"].get("characterized") is True
    assert data["org/NoCeiling"]["safe_context"] is None
    assert data["org/Never"].get("characterized") is False


def test_render_models_best_fit_across_engines(make_console, store, monkeypatch):
    """A model characterized under two engines shows the LARGER ceiling + which engine reached
    it — the winbox case (GPU 3500 vs CPU 8192 for one model → 8192, cpu)."""
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/L", "modality": "text"}])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")     # default engine = cuda
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    per_engine = {"cuda": [{"model_id": "org/L", "safe_context": 3500,
                             "decode_context": None, "config": {}}],
                  "cpu": [{"model_id": "org/L", "safe_context": 8192,
                            "decode_context": None, "config": {}}]}
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, e: per_engine.get(e, []))
    c, buf = make_console()
    cli.render_models(c)
    line = next(ln for ln in buf.getvalue().splitlines() if "org/L" in ln)
    assert "8192" in line and "(cpu)" in line and "3500" not in line


def test_main_models_bare_does_not_dispatch_cached_inventory(monkeypatch, capsys):
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["models"]) == 0
    assert "models" not in rec
    assert "Usage: ara models" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# ara recommend — analytic: catalog models that fit, ranked by context
# Spec 2026-06-23-capability-pipeline (Slice 3)
# --------------------------------------------------------------------------- #
def _model_row(model_id, *, weights_gb=4.0, max_context=8192, **over):
    base = dict(model_id=model_id, modality="text", params=None, quant=None,
                n_layers=32, hidden_size=4096, kv_heads=8, head_dim=128,
                weights_gb=weights_gb, max_context=max_context, updated_at="t")
    base.update(over)
    return base


def _wire_recommend(monkeypatch, set_platform, models, machine=None):
    set_platform("Darwin", "arm64")
    monkeypatch.setattr(cli.detect, "machine",
                        lambda: machine if machine is not None
                        else _machine(backend="apple", ram_total_gb=48.0))
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models", lambda con: models)
    monkeypatch.setattr(cli.db, "list_characterizations", lambda con, mk, e: [])
    monkeypatch.setattr(cli.staleness, "artifact_identity",
                        lambda model: f"artifact:{model}")


def _measured_row(model_id, use_case="coding", score=0.6, source="mlx probe", **over):
    canonical = cli.scoring.canonical_model_id(model_id)
    row = {
        "model_id": model_id, "use_case": use_case, "score": score, "source": source,
        "tier": "measured", "engine_key": "mlx", "backend": "apple",
        "base_model": cli.scoring.base_key(canonical),
        "quant": cli.scoring.quant_key(model_id), "benchmark_id": use_case,
        "methodology_id": cli.benchmark.methodology_id(use_case),
        "sample_size": len(cli.benchmark.load_probe(use_case)),
        "max_score": 1.0, "refused_n": 0, "errored_n": 0,
        "artifact_id": f"artifact:{model_id}",
        "canonical_model_id": canonical, "measured_at": "2026-07-15T12:00:00+00:00",
    }
    row.update(over)
    return row


def _accept_render_test_evidence(monkeypatch):
    """Keep presentation-only tests focused on rendering synthetic evidence shapes."""
    def validate(row):
        return ({
            "score": float(row["score"]), "source": row["source"],
            "sample_size": row.get("sample_size"),
            "refused_n": row.get("refused_n"), "errored_n": row.get("errored_n"),
            "probe_context": row.get("probe_context"),
            "generation_cap": row.get("generation_cap"),
            "repeat_count": row.get("repeat_count"),
            "total_generations": row.get("total_generations"),
            "run_scores": None,
        }, None)
    monkeypatch.setattr(cli.scoring, "validate_measured_evidence", validate)


def test_recommend_ranks_fits_by_context(make_console, monkeypatch, set_platform):
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Small", max_context=8192),
                     _model_row("org/Big", max_context=131072)])
    c, buf = make_console()
    assert cli.render_recommend(c) == 0
    out = buf.getvalue()
    assert "RECOMMENDED MODELS" in out
    assert out.index("org/Big") < out.index("org/Small")   # most context first
    assert "full window" in out                            # both window-bound on a 48 GB Mac


def test_recommend_excludes_models_that_dont_fit(make_console, monkeypatch, set_platform):
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Fits", weights_gb=4.0),
                     _model_row("org/TooBig", weights_gb=200.0)])
    c, buf = make_console()
    assert cli.render_recommend(c) == 0
    out = buf.getvalue()
    assert "org/Fits" in out and "org/TooBig" not in out


def test_recommend_memory_bound_label(make_console, monkeypatch, set_platform):
    # A tiny CPU budget binds before a huge window → ranked, but not "full window".
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/BigCtx", weights_gb=4.0, max_context=131072)],
                    machine=_machine(backend="cpu", ram_total_gb=8.0))
    c, buf = make_console()
    assert cli.render_recommend(c) == 0
    out = buf.getvalue()
    assert "org/BigCtx" in out and "tok est." in out
    assert "full window" not in out


def test_recommend_marks_characterized(make_console, monkeypatch, set_platform):
    _wire_recommend(monkeypatch, set_platform, [_model_row("org/Known")])
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, e: [{"model_id": "org/Known", "safe_context": 12000,
                                             "decode_context": None}] if e == "mlx" else [])
    c, buf = make_console()
    assert cli.render_recommend(c) == 0
    assert "characterized" in buf.getvalue()


def test_recommend_none_fit(make_console, monkeypatch, set_platform):
    _wire_recommend(monkeypatch, set_platform, [_model_row("org/TooBig", weights_gb=500.0)])
    c, buf = make_console()
    assert cli.render_recommend(c) == 0
    assert "nothing in the catalog fits" in buf.getvalue()


def test_recommend_notes_fitting_but_unrankable(make_console, monkeypatch, set_platform):
    # Spec 2026-06-23-capability-pipeline: a model whose weights fit but whose architecture ARA
    # can't read (no slope → est_context None) used to be dropped silently. It must now be counted
    # and honestly disclosed, pointing at ara profile --model.
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Rankable", weights_gb=4.0),
                     _model_row("org/Unknown", weights_gb=4.0, n_layers=None)])
    c, buf = make_console()
    assert cli.render_recommend(c) == 0
    out = buf.getvalue()
    assert "org/Rankable" in out
    assert "1 more fit but can't be ranked (architecture unknown)" in out
    assert "ara profile --model" in out


def test_recommend_no_unrankable_no_note(make_console, monkeypatch, set_platform):
    _wire_recommend(monkeypatch, set_platform, [_model_row("org/Rankable", weights_gb=4.0)])
    c, buf = make_console()
    assert cli.render_recommend(c) == 0
    assert "can't be ranked" not in buf.getvalue()


def test_recommend_none_ranked_but_unrankable_exist(make_console, monkeypatch, set_platform):
    # The empty-recs path must still disclose models that fit but can't be ranked, instead of only
    # saying nothing fits. Spec 2026-06-23-capability-pipeline.
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Unknown", weights_gb=4.0, n_layers=None)])
    c, buf = make_console()
    assert cli.render_recommend(c) == 0
    out = buf.getvalue()
    assert "1 more fit but can't be ranked (architecture unknown)" in out
    assert "ara profile --model" in out


def test_recommend_uses_measured_wall(make_console, monkeypatch, set_platform, store):
    # Spec 2026-06-23-capability-pipeline: after the detected engine is calibrated, recommend ranks
    # against the MEASURED budget, not the heuristic. A measured wall that's tighter than the
    # heuristic must actually bind the fit — proving the measurement drove the math.
    captured = {}
    real_limits = cli.estimate.limits

    def spy_limits(machine, measured=None):
        captured["measured"] = measured
        return real_limits(machine, measured=measured)

    _wire_recommend(monkeypatch, set_platform, [_model_row("org/Small", max_context=8192)])
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.estimate, "limits", spy_limits)
    cli.calibration.save_calibration(store, "mlx", fixed_overhead_gb=1.7,
                                     wall_gb=41.3, safe_budget_gb=39.3)
    c, buf = make_console()
    assert cli.render_recommend(c) == 0
    assert captured["measured"] is not None
    assert captured["measured"]["wall_gb"] == 41.3        # the stored measurement was passed in


def test_recommend_json(monkeypatch, set_platform, capsys):
    _wire_recommend(monkeypatch, set_platform, [_model_row("org/Big", max_context=131072)])
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_recommend(c, as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["model_id"] == "org/Big"
    assert payload[0]["fits"] is True and "est_context" in payload[0]


def test_recommend_verbose_discloses_budget_provenance(
        make_console, monkeypatch, set_platform):
    _wire_recommend(monkeypatch, set_platform, [_model_row("org/Small")])
    c, buf = make_console(verbose=True)
    assert cli.render_recommend(c) == 0
    out = " ".join(buf.getvalue().split())
    assert "provenance wall estimated · mlx · 34.0 GB safe budget" in out
    assert "catalog 1 cached model · ephemeral read-only scan" in out


def test_recommend_verbose_distinguishes_measured_wall_from_analytic_fit(
        store, make_console, monkeypatch, set_platform):
    _wire_recommend(monkeypatch, set_platform, [_model_row("org/Small")])
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    cli.calibration.save_calibration(store, "mlx", fixed_overhead_gb=1.7,
                                     wall_gb=20.0, safe_budget_gb=18.0)
    c, buf = make_console(verbose=True)
    assert cli.render_recommend(c) == 0
    out = " ".join(buf.getvalue().split())
    assert "provenance wall measured · mlx · 18.0 GB safe budget" in out
    assert "estimated — fits this machine" in out


def test_recommend_does_not_create_database(
        tmp_path, make_console, monkeypatch, set_platform):
    path = tmp_path / "missing" / "ara.db"
    monkeypatch.setenv("ARA_DB_PATH", str(path))
    _wire_recommend(monkeypatch, set_platform, [_model_row("org/Small")])

    c, _ = make_console()
    assert cli.render_recommend(c) == 0
    assert not path.exists()


def test_recommend_use_case_ranks_by_capability_and_labels(make_console, monkeypatch, set_platform):
    # With --use-case, models rank by capability (not context), and provenance is labelled.
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Weak", weights_gb=4.0, max_context=131072),
                     _model_row("org/Strong", weights_gb=4.0, max_context=131072)])
    monkeypatch.setattr(cli.scoring, "load_imported",
                        lambda: {"org/Strong": {"coding": {"score": 0.9, "source": "HumanEval"}},
                                 "org/Weak": {"coding": {"score": 0.3, "source": "HumanEval"}}})
    c, buf = make_console()
    assert cli.render_recommend(c, use_case="coding") == 0
    out = buf.getvalue()
    assert out.index("org/Strong") < out.index("org/Weak")   # capability-ranked, not context
    assert "imported" in out                                 # provenance shown, not a bare number


def test_recommend_use_case_unknown_score_is_honest(make_console, monkeypatch, set_platform):
    # A fitting model with no score is shown as `unknown`, never dropped or guessed (Rule #3).
    _wire_recommend(monkeypatch, set_platform, [_model_row("org/Unscored", weights_gb=4.0)])
    monkeypatch.setattr(cli.scoring, "load_imported", lambda: {})
    c, buf = make_console()
    assert cli.render_recommend(c, use_case="coding") == 0
    out = buf.getvalue()
    assert "org/Unscored" in out and "unknown" in out


def test_recommend_rejects_unknown_use_case(make_console, monkeypatch, set_platform):
    _wire_recommend(monkeypatch, set_platform, [_model_row("org/X")])
    c, buf = make_console()
    assert cli.render_recommend(c, use_case="cooking") == 1
    assert "cooking" in buf.getvalue()


def _wire_inversion_bench(monkeypatch, four_bit=0.098, eight_bit=0.061, n=None):
    # Same base model, two quants, both measured here — a lower-precision upset.
    n = len(cli.benchmark.load_probe("coding")) if n is None else n
    monkeypatch.setattr(cli.scoring, "load_imported", lambda: {})
    monkeypatch.setattr(cli.db, "list_benchmark_results", lambda con, mk: [
        _measured_row("org/Model-4bit", score=four_bit, sample_size=n,
                      refused_n=0, errored_n=0),
        _measured_row("org/Model-8bit", score=eight_bit, sample_size=n,
                      refused_n=0, errored_n=0)])


def test_recommend_text_discloses_quant_inversion(make_console, monkeypatch, set_platform):
    # A measured lower-precision upset is DISCLOSED inline on both rows (Rule #3), not reordered.
    # Spec 2026-07-02-recommend-inversion-guard.
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Model-4bit", weights_gb=4.0),
                     _model_row("org/Model-8bit", weights_gb=4.0)])
    _wire_inversion_bench(monkeypatch)
    c, buf = make_console()
    assert cli.render_recommend(c, use_case="coding") == 0
    out = buf.getvalue()
    assert "[quant-inversion: outscores 8bit within noise]" in out
    assert "[quant-inversion: outscored by 4bit within noise]" in out


def test_recommend_json_carries_inversion_field(monkeypatch, set_platform, capsys):
    # The JSON path serializes the new inversion field alongside the other Score fields.
    # Spec 2026-07-02-recommend-inversion-guard.
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Model-4bit", weights_gb=4.0),
                     _model_row("org/Model-8bit", weights_gb=4.0)])
    _wire_inversion_bench(monkeypatch)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_recommend(c, use_case="coding", as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    by_id = {r["model_id"]: r["score"] for r in payload}
    assert by_id["org/Model-4bit"]["inversion"] == "outscores 8bit within noise"
    assert by_id["org/Model-8bit"]["inversion"] == "outscored by 4bit within noise"


def test_recommend_clean_pair_has_no_inversion_disclosure(monkeypatch, set_platform, capsys):
    # A non-inverted pair (higher precision scores higher) shows no inversion — text or JSON.
    # Spec 2026-07-02-recommend-inversion-guard.
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Model-4bit", weights_gb=4.0),
                     _model_row("org/Model-8bit", weights_gb=4.0)])
    _wire_inversion_bench(monkeypatch, four_bit=0.30, eight_bit=0.90)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_recommend(c, use_case="coding", as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert all(r["score"]["inversion"] is None for r in payload)


# quant-aware recommend: surface each rec's quant + effective bit-width and base, and disclose the
# precision↔context tradeoff when one base fits at multiple quants.
# Spec 2026-07-04-recommend-quant-aware.
def test_recommend_json_carries_quant_fields(monkeypatch, set_platform, capsys):
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Llama-3.2-3B-Instruct-4bit", quant="4bit",
                                weights_gb=2.0, max_context=131072)])
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_recommend(c, as_json=True) == 0
    r = json.loads(capsys.readouterr().out)[0]
    assert r["quant"] == "4bit" and r["quant_bits"] == 4.0
    assert r["base"] == "llama-3.2-3b-instruct"


def test_recommend_quant_falls_back_to_id_when_catalog_has_none(monkeypatch, set_platform, capsys):
    # Catalog didn't record a quant, but the id carries one → parse it (don't drop the fact).
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Qwen3-8B-Q4_K_M", quant=None, weights_gb=5.0)])
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_recommend(c, as_json=True) == 0
    r = json.loads(capsys.readouterr().out)[0]
    assert r["quant"] == "q4_k_m" and r["quant_bits"] == 4.0


def test_recommend_text_shows_quant(make_console, monkeypatch, set_platform):
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Model-4bit", quant="4bit", weights_gb=4.0)])
    c, buf = make_console()
    assert cli.render_recommend(c) == 0
    assert "4bit" in buf.getvalue()


def test_recommend_surfaces_quant_tradeoff(make_console, monkeypatch, set_platform):
    # Same base at two quants both fit → a tradeoff note names both quants (fewer bits = more ctx).
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Llama-3.2-3B-Instruct-4bit", quant="4bit",
                                weights_gb=2.0, max_context=131072),
                     _model_row("org/Llama-3.2-3B-Instruct-8bit", quant="8bit",
                                weights_gb=4.0, max_context=131072)],
                    machine=_machine(backend="apple", ram_total_gb=10.0))   # memory-bound → ctx differs
    c, buf = make_console()
    assert cli.render_recommend(c) == 0
    out = buf.getvalue()
    assert "tradeoff" in out.lower()
    note = out[out.lower().index("tradeoff"):]
    assert "4bit" in note and "8bit" in note
    assert "llama-3.2-3b-instruct" in note


def test_recommend_no_tradeoff_note_for_distinct_bases(make_console, monkeypatch, set_platform):
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Llama-3B-4bit", quant="4bit", weights_gb=2.0),
                     _model_row("org/Qwen-7B-4bit", quant="4bit", weights_gb=4.0)])
    c, buf = make_console()
    assert cli.render_recommend(c) == 0
    assert "tradeoff" not in buf.getvalue().lower()


def test_recommend_quant_tradeoff_survives_unmappable_bits(make_console, monkeypatch, set_platform):
    # A quant token quant_key recognises but quant_bits can't map (e.g. "q8bit") must NOT crash the
    # tradeoff sort — it's ordered last, never a TypeError comparing None to a float (Rule #2).
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Foo-q8bit", quant="q8bit", weights_gb=2.0, max_context=131072),
                     _model_row("org/Foo-4bit", quant="4bit", weights_gb=2.0, max_context=131072)],
                    machine=_machine(backend="apple", ram_total_gb=10.0))
    c, buf = make_console()
    assert cli.render_recommend(c) == 0                    # no crash
    note = buf.getvalue().lower()
    assert "tradeoff" in note and "q8bit" in note and "4bit" in note


def test_main_recommend_dispatch(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["recommend", "--json"])
    assert rec["recommend"] is True


def test_main_recommend_use_case_dispatch(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["recommend", "--use-case", "coding"])
    assert rec["recommend_uc"] == "coding"


# --------------------------------------------------------------------------- #
# ara serve --engine mlx — governed MLX endpoint (this Mac)
# Spec 2026-06-28-recommend-use-case-and-serve-selection
# --------------------------------------------------------------------------- #
def test_serve_mlx_governs_via_measured_ceiling(make_console, monkeypatch, set_platform):
    # `serve --engine mlx` stands the model up on the governed MLX server at the MEASURED apple
    # ceiling, hands back an OpenAI-compatible /v1 endpoint, and stays foreground (proc.wait).
    set_platform("Darwin", "arm64")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    # Characterizations are keyed by ENGINE KEY ("mlx"), not backend name ("apple").
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, m: {"safe_context": 8000} if e == "mlx" else None)
    monkeypatch.setattr(cli, "_free_port", lambda: 12399)
    captured = {}

    class _Proc:
        def wait(self):
            captured["waited"] = True

    def _fake_serve(model, *, port, max_context, kv_quant="f16", measured_slope_gb_per_k=None):
        captured.update(model=model, port=port, max_context=max_context)
        return _Proc(), f"http://127.0.0.1:{port}", max_context

    monkeypatch.setattr("ara.backends.apple.serve", _fake_serve)
    c, buf = make_console()
    rc = cli.render_serve(c, "mlx-community/Llama-3.2-3B-Instruct-4bit",
                          engine="mlx", assume_yes=True)
    assert rc == 0
    assert captured["max_context"] == 8000 and captured["port"] == 12399
    assert captured["waited"] is True                      # foreground: our child IS the server
    assert "OPENAI_BASE_URL=http://127.0.0.1:12399/v1" in buf.getvalue()


def test_serve_mlx_json_carries_stale_ceiling_flag(make_console, monkeypatch, set_platform, capsys):
    """MLX serve --json must also carry the stale flag (mirrors the Ollama path) — Fix #4."""
    set_platform("Darwin", "arm64")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, m: ({"safe_context": 8000,
                                                "measured_at": "2026-01-01T00:00:00+00:00"}
                                               if e == "mlx" else None))
    monkeypatch.setattr(cli.staleness, "ceiling_is_stale", lambda mid, at: True)
    monkeypatch.setattr(cli, "_free_port", lambda: 12399)

    def _fake_serve(model, *, port, max_context, kv_quant="f16", measured_slope_gb_per_k=None):
        return types.SimpleNamespace(wait=lambda: None), f"http://127.0.0.1:{port}", max_context

    monkeypatch.setattr("ara.backends.apple.serve", _fake_serve)
    c, _ = make_console()
    assert cli.render_serve(c, "org/m", engine="mlx", assume_yes=True, as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["stale_ceiling"] is True


def test_serve_mlx_refuses_without_measured_ceiling(make_console, monkeypatch, set_platform):
    # No measured MLX ceiling + no --ctx → refuse, point at characterize (never a guessed ceiling).
    set_platform("Darwin", "arm64")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization", lambda con, mk, e, m: None)
    c, buf = make_console()
    assert cli.render_serve(c, "org/Uncharacterized", engine="mlx", assume_yes=True) == 1
    assert "characterize" in buf.getvalue()


def test_serve_mlx_refuses_ceiling_measured_with_nondefault_config(
        make_console, monkeypatch, set_platform):
    set_platform("Darwin", "arm64")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization", lambda con, mk, e, m: {
        "safe_context": 8000, "config": {"kv_quant": "q4_0"}
    })
    c, buf = make_console()
    assert cli.render_serve(c, "org/m", engine="mlx", assume_yes=True) == 1
    assert "different engine settings" in buf.getvalue()


def test_serve_mlx_explicit_ctx_still_refuses_mismatched_measurement_config(
        make_console, monkeypatch, set_platform):
    set_platform("Darwin", "arm64")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization", lambda con, mk, e, m: {
        "safe_context": 8000, "config": {"kv_quant": "q4_0"}
    })
    c, buf = make_console()
    assert cli.render_serve(c, "org/m", engine="mlx", ctx=4000, assume_yes=True) == 1
    assert "different engine settings" in buf.getvalue()


# --- measured-provenance slope (2026-07-02-mlx-serve-measured-provenance-gate) -- #
def test_measured_ramp_slope_fits_and_skips_incomplete_points():
    row = {"points": [{"context": 2000, "mem_gb": 1.5},
                      {"context": None, "mem_gb": 9.9},      # incomplete → skipped
                      {"context": 16000, "mem_gb": 4.5}]}
    slope = cli._measured_ramp_slope(row)
    assert slope is not None and abs(slope - 3.0 / 14) < 1e-6      # (4.5-1.5)/((16000-2000)/1000)


def test_measured_ramp_slope_none_when_too_few_points():
    assert cli._measured_ramp_slope({"points": [{"context": 2000, "mem_gb": 1.5}]}) is None
    assert cli._measured_ramp_slope({"points": []}) is None
    assert cli._measured_ramp_slope(None) is None


def test_measured_ramp_slope_none_on_degenerate_fit():
    # two measurements at the SAME context can't fit a slope (RampError) → None, not a crash
    row = {"points": [{"context": 4000, "mem_gb": 1.5}, {"context": 4000, "mem_gb": 2.5}]}
    assert cli._measured_ramp_slope(row) is None


def test_serve_mlx_passes_measured_slope_from_points(make_console, monkeypatch, set_platform):
    # Serving the measured ceiling fits the real ramp slope and hands it to the engine gate, so a
    # long-window measured serve isn't falsely refused by the a-priori prior.
    set_platform("Darwin", "arm64")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    row = {"safe_context": 40960,
           "points": [{"context": 2000, "mem_gb": 1.535}, {"context": 16000, "mem_gb": 4.48}]}
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, m: row if e == "mlx" else None)
    monkeypatch.setattr(cli, "_free_port", lambda: 12399)
    captured = {}

    class _Proc:
        def wait(self):
            captured["waited"] = True

    def _fake_serve(model, *, port, max_context, kv_quant="f16", measured_slope_gb_per_k=None):
        captured["slope"] = measured_slope_gb_per_k
        return _Proc(), f"http://127.0.0.1:{port}", max_context

    monkeypatch.setattr("ara.backends.apple.serve", _fake_serve)
    c, _ = make_console()
    assert cli.render_serve(c, "org/m", engine="mlx", assume_yes=True) == 0
    assert captured["slope"] is not None and abs(captured["slope"] - (4.48 - 1.535) / 14) < 1e-6


# --------------------------------------------------------------------------- #
# ara run — governed one-shot inference (Spec 2026-06-23-capability-pipeline, Slice 4)
# --------------------------------------------------------------------------- #
_CHAR = {"model_id": "org/m", "safe_context": 8192, "decode_context": None,
         "artifact_id": "artifact:test"}


def _ok_generate(*a, **k):
    return {"completion": "ok"}


def _wire_run(monkeypatch, *, engine_ok=True, generate=_ok_generate, characterization=None,
              isatty=False):
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cpu")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (engine_ok, "llama.cpp"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    if characterization is not None and "artifact_id" not in characterization:
        characterization = {**characterization, "artifact_id": "artifact:test"}
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, m: characterization)
    monkeypatch.setattr(cli.staleness, "artifact_matches", lambda *_a: True)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: isatty))
    bk = types.SimpleNamespace()
    if generate is not None:
        bk.generate = generate
    monkeypatch.setattr(cli, "get_backend", lambda b=None: bk)


def test_run_refuses_uncharacterized(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization=None)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi") == 1
    out = buf.getvalue()
    assert "isn't characterized" in out and "ara characterize org/m" in out


def test_run_refuses_when_no_safe_ceiling(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization={"model_id": "org/m", "safe_context": None})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi") == 1
    assert "didn't fit" in buf.getvalue()


def test_run_engine_not_installed(make_console, monkeypatch):
    _wire_run(monkeypatch, engine_ok=False, characterization=_CHAR)
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (False, "CUDA engine"))
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi") == 1
    assert buf.getvalue() == \
        "  the CUDA engine isn't installed — run: ara install --engine cpu\n"


def test_run_engine_not_installed_json_uses_complete_label(monkeypatch, capsys):
    _wire_run(monkeypatch, engine_ok=False, characterization=_CHAR)
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (False, "CUDA engine"))
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_run(c, "org/m", prompt="hi", as_json=True) == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": "the CUDA engine isn't installed — run: ara install --engine cpu"}


def test_run_pinned_engine_not_installed_uses_pinned_hint(make_console, monkeypatch):
    _wire_run(monkeypatch, engine_ok=False, characterization=_CHAR)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="cpu") == 1
    assert "ara install --engine cpu" in buf.getvalue()


def test_run_unsupported_engine(make_console, monkeypatch):
    # A backend with no generate method reports its complete public engine label once.
    _wire_run(monkeypatch, characterization=_CHAR, generate=None)
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "CUDA engine"))
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="cuda") == 1
    assert buf.getvalue() == "  run isn't supported on the CUDA engine yet\n"


def test_run_generates_capped_at_ceiling(make_console, monkeypatch):
    seen = {}

    def gen(model, prompt, *, max_context, max_tokens):
        seen.update(model=model, prompt=prompt, max_context=max_context)
        return {"context": max_context, "completion": "the answer is 42"}

    _wire_run(monkeypatch, characterization=_CHAR, generate=gen, isatty=False)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="meaning?") == 0
    assert "the answer is 42" in buf.getvalue()
    assert seen["max_context"] == 8192 and seen["prompt"] == "meaning?"   # governed ceiling


def test_run_loads_immutable_pinned_artifact_reference(make_console, monkeypatch):
    seen = {}

    def gen(model, *_a, **_k):
        seen["model"] = model
        return {"completion": "ok"}

    _wire_run(monkeypatch, characterization=_CHAR, generate=gen)
    monkeypatch.setattr(cli.staleness, "pinned_model_ref",
                        lambda _model, _artifact: "/cache/snapshots/rev")
    c, _ = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", assume_yes=True) == 0
    assert seen["model"] == "/cache/snapshots/rev"


def test_run_refuses_when_authorized_artifact_cannot_be_pinned(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization=_CHAR)
    monkeypatch.setattr(cli.staleness, "pinned_model_ref", lambda *_a: None)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", assume_yes=True) == 1
    assert "cannot pin" in buf.getvalue()


def test_run_refuses_missing_or_changed_artifact_authority(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization={**_CHAR, "artifact_id": None})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi") == 1
    assert "not bound to an exact artifact" in buf.getvalue()

    _wire_run(monkeypatch, characterization=_CHAR)
    monkeypatch.setattr(cli.staleness, "artifact_matches", lambda *_a: False)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi") == 1
    assert "differs from its measured ceiling" in buf.getvalue()


def test_run_pinned_refuses_missing_artifact_authority(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization={**_CHAR, "artifact_id": None})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="cpu") == 1
    assert "not bound to an exact artifact" in buf.getvalue()


def test_run_refuses_when_artifact_changes_after_auto_selection_before_load(
        make_console, monkeypatch):
    _wire_run(monkeypatch, characterization=_CHAR)
    monkeypatch.setattr(
        cli.db, "get_characterization",
        lambda _con, _mk, engine, _model: _CHAR if engine == "cpu" else None)
    matches = iter((True, False))
    monkeypatch.setattr(cli.staleness, "artifact_matches", lambda *_a: next(matches))
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi") == 1
    assert "differs from its measured ceiling" in buf.getvalue()


def test_run_refuses_result_when_artifact_changes_during_generation(
        make_console, monkeypatch):
    generated = []
    _wire_run(monkeypatch, characterization=_CHAR,
              generate=lambda *_a, **_k: generated.append(True) or {"completion": "unsafe"})
    monkeypatch.setattr(
        cli.db, "get_characterization",
        lambda _con, _mk, engine, _model: _CHAR if engine == "cpu" else None)
    matches = iter((True, True, False))
    monkeypatch.setattr(cli.staleness, "artifact_matches", lambda *_a: next(matches))
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi") == 1
    assert generated == [True]
    assert "changed during the run" in buf.getvalue()
    assert "unsafe" not in buf.getvalue()


def test_run_confirm_declined_skips(make_console, monkeypatch):
    called = []
    _wire_run(monkeypatch, characterization=_CHAR,
              generate=lambda *a, **k: called.append(1) or {"completion": "x"}, isatty=True)
    monkeypatch.setattr(cli, "_confirm", lambda q: False)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi") == 0
    assert "skipped" in buf.getvalue() and called == []


def test_run_yes_skips_confirm(make_console, monkeypatch):
    def no_ask(q):
        raise AssertionError("should not prompt with --yes")
    _wire_run(monkeypatch, characterization=_CHAR,
              generate=lambda *a, **k: {"completion": "ok"}, isatty=True)
    monkeypatch.setattr(cli, "_confirm", no_ask)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", assume_yes=True) == 0
    assert "ok" in buf.getvalue()


def test_run_confirm_accepted_runs(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization=_CHAR,
              generate=lambda *a, **k: {"completion": "done"}, isatty=True)
    monkeypatch.setattr(cli, "_confirm", lambda q: True)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi") == 0
    assert "done" in buf.getvalue()


def test_run_worker_refused(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization=_CHAR,
              generate=lambda *a, **k: {"refused": True, "reason": "memory pressure"})
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "CUDA engine"))
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="cuda", assume_yes=True) == 1
    assert buf.getvalue().splitlines()[-1] == "  the CUDA engine refused: memory pressure"


def test_run_failure(make_console, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("worker died")
    _wire_run(monkeypatch, characterization=_CHAR, generate=boom)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", assume_yes=True) == 1
    assert "run failed" in buf.getvalue()


def test_run_worker_error_payload_is_failure(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization=_CHAR,
              generate=lambda *a, **k: {"error": "generation exploded"})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", assume_yes=True) == 1
    assert "generation exploded" in buf.getvalue()


@pytest.mark.parametrize("result", [None, [], {}, {"completion": None}, {"completion": 7}])
def test_run_rejects_malformed_completion_payload(make_console, monkeypatch, result):
    _wire_run(monkeypatch, characterization=_CHAR, generate=lambda *a, **k: result)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", assume_yes=True) == 1
    assert "invalid completion" in buf.getvalue()


def test_run_json(monkeypatch, capsys):
    _wire_run(monkeypatch, characterization=_CHAR,
              generate=lambda *a, **k: {"completion": "hello"})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_run(c, "org/m", prompt="hi", as_json=True, assume_yes=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["completion"] == "hello" and payload["safe_context"] == 8192


def test_run_usage_without_prompt(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization=_CHAR)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt=None) == 1
    assert "usage" in buf.getvalue()


def test_run_rejects_whitespace_only_prompt(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization=_CHAR)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="   ") == 1
    assert "usage" in buf.getvalue()


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_run_rejects_nonpositive_max_tokens(make_console, monkeypatch, max_tokens):
    _wire_run(monkeypatch, characterization=_CHAR)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", max_tokens=max_tokens) == 1
    assert "max-tokens" in buf.getvalue()


def test_run_unknown_engine(make_console):
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="bogus") == 1
    assert "unknown engine" in buf.getvalue().lower()


# Cross-engine selection: with no --engine, run scans every engine this model is characterized
# under on this machine and picks the largest safe_context whose backend can actually generate —
# not just the detected engine. Spec 2026-06-23-capability-pipeline.
def _wire_run_cross(monkeypatch, *, detected, chars, supports, engine_ok=True, isatty=False):
    """chars: {engine_key: characterization|None}; supports: {backend: bool} (has .generate)."""
    chars = {key: ({**row, "artifact_id": row.get("artifact_id", "artifact:test")}
                   if row is not None else None) for key, row in chars.items()}
    monkeypatch.setattr(cli.detect, "backend_name", lambda: detected)
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (engine_ok, f"{b} pkg"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, m: chars.get(e))
    monkeypatch.setattr(cli.staleness, "artifact_matches", lambda *_a: True)

    def backend(b=None):
        bk = types.SimpleNamespace()
        if supports.get(b):
            bk.generate = lambda model, prompt, *, max_context, max_tokens, kv_quant="f16", flash_attn=False, weight_quant="none", prefill_chunk=None: {
                "engine_backend": b, "max_context": max_context, "completion": f"ran on {b}"}
        return bk
    monkeypatch.setattr(cli, "get_backend", backend)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: isatty))


def test_run_picks_characterized_engine_across_backends(make_console, monkeypatch):
    # Detected apple (no run support), but the model is characterized on cpu → run on cpu.
    _wire_run_cross(
        monkeypatch, detected="apple",
        chars={"cpu": {"model_id": "org/m", "safe_context": 4096}},
        supports={"cpu": True, "apple": False})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", assume_yes=True) == 0
    assert "ran on cpu" in buf.getvalue()


def test_run_picks_largest_safe_context_engine(monkeypatch, capsys):
    # Characterized on two run-capable engines → pick the largest safe_context.
    _wire_run_cross(
        monkeypatch, detected="cpu",
        chars={"cpu": {"model_id": "org/m", "safe_context": 4096},
               "cuda": {"model_id": "org/m", "safe_context": 16000}},
        supports={"cpu": True, "cuda": True})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_run(c, "org/m", prompt="hi", as_json=True, assume_yes=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"] == "cuda" and payload["safe_context"] == 16000


def test_run_auto_skips_larger_stale_artifact_for_valid_fallback(
        make_console, monkeypatch):
    _wire_run_cross(
        monkeypatch, detected="cuda",
        chars={
            "cuda": {"model_id": "org/m", "safe_context": 16000,
                     "artifact_id": "artifact:stale"},
            "cpu": {"model_id": "org/m", "safe_context": 8000,
                    "artifact_id": "artifact:test"},
        },
        supports={"cpu": True, "cuda": True})
    monkeypatch.setattr(
        cli.staleness, "artifact_matches",
        lambda _model, artifact: artifact == "artifact:test")
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", assume_yes=True) == 0
    assert "ran on cpu" in buf.getvalue()


def test_run_auto_reports_only_uninstalled_characterized_engine(
        make_console, monkeypatch):
    _wire_run_cross(
        monkeypatch, detected="cuda",
        chars={"cuda": {"model_id": "org/m", "safe_context": 16000}},
        supports={"cuda": True}, engine_ok=False)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", assume_yes=True) == 1
    assert "cuda pkg isn't installed" in buf.getvalue()
    assert "ara install --engine cuda" in buf.getvalue()


def test_run_engine_override_pins_named_engine(make_console, monkeypatch):
    # --engine pins exactly that engine even if another engine has a bigger ceiling.
    _wire_run_cross(
        monkeypatch, detected="cpu",
        chars={"cpu": {"model_id": "org/m", "safe_context": 4096},
               "cuda": {"model_id": "org/m", "safe_context": 16000}},
        supports={"cpu": True, "cuda": True})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="cpu", assume_yes=True) == 0
    assert "ran on cpu" in buf.getvalue()        # pinned to cpu (4096), not the bigger cuda


def test_run_characterized_only_on_unsupported_engine(make_console, monkeypatch):
    # Characterized on apple alone, whose backend can't generate yet → honest "not supported",
    # NOT a silent "uncharacterized" refusal.
    _wire_run_cross(
        monkeypatch, detected="apple",
        chars={"mlx": {"model_id": "org/m", "safe_context": 8192}},
        supports={"apple": False, "cpu": True})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", assume_yes=True) == 1
    out = buf.getvalue()
    assert "mlx" in out and "isn't supported" in out
    assert "ara characterize" not in out         # it IS characterized — don't point at characterize


def test_run_auto_refuses_when_only_runnable_ceiling_has_different_config(
        make_console, monkeypatch):
    _wire_run_cross(
        monkeypatch, detected="apple",
        chars={"mlx": {"model_id": "org/m", "safe_context": 8192,
                       "config": {"kv_quant": "q4_0"}}},
        supports={"apple": True})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", assume_yes=True) == 1
    assert "different engine settings" in buf.getvalue()


def test_run_auto_skips_larger_engine_that_cannot_honor_requested_lever(
        make_console, monkeypatch):
    _wire_run_cross(
        monkeypatch, detected="cpu",
        chars={
            "cpu": {"model_id": "org/m", "safe_context": 32000, "config": {}},
            "vulkan": {"model_id": "org/m", "safe_context": 16000,
                       "config": {"kv_quant": "q4_0"}},
        },
        supports={"cpu": True, "vulkan": True})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", kv_quant="q4_0", assume_yes=True) == 0
    assert "ran on vulkan" in buf.getvalue()


def test_run_uncharacterized_on_every_engine_refuses(make_console, monkeypatch):
    # Characterized nowhere → refuse, pointing at characterize.
    _wire_run_cross(monkeypatch, detected="apple", chars={}, supports={"cpu": True})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", assume_yes=True) == 1
    assert "isn't characterized" in buf.getvalue() and "ara characterize org/m" in buf.getvalue()


def test_run_pinned_refuses_uncharacterized(make_console, monkeypatch):
    # --engine pins exactly that engine: uncharacterized THERE refuses, pointing at characterize.
    _wire_run(monkeypatch, characterization=None)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="cpu") == 1
    out = buf.getvalue()
    assert "isn't characterized on cpu" in out and "ara characterize org/m --engine cpu" in out


def test_run_pinned_refuses_when_no_safe_ceiling(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization={"model_id": "org/m", "safe_context": None})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="cpu") == 1
    assert "didn't fit on cpu" in buf.getvalue()


def test_run_pinned_unsupported_engine(make_console, monkeypatch):
    # --engine pins an engine whose backend can't generate yet → honest "isn't supported".
    _wire_run(monkeypatch, characterization=_CHAR, generate=None)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="cpu") == 1
    assert "isn't supported on the" in buf.getvalue()


def test_main_run_dispatch(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["run", "org/m", "hello", "world"])
    assert rec["run"]["model"] == "org/m"
    assert rec["run"]["prompt"] == "hello world"


def test_main_run_parses_max_tokens(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["run", "org/m", "hello", "--max-tokens", "32"]) == 0
    assert rec["run"]["max_tokens"] == 32


def test_main_run_requires_prompt(monkeypatch, capsys):
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["run", "org/m"]) == 2
    assert "run" not in rec
    assert "Missing argument 'PROMPT...'" in capsys.readouterr().err


@pytest.mark.parametrize(("flag", "value"), [
    ("--max-tokens", "0"), ("--max-tokens", "-1"),
    ("--prefill-chunk", "0"), ("--prefill-chunk", "-1"),
])
def test_main_run_rejects_nonpositive_generation_limits(monkeypatch, capsys, flag, value):
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["run", "org/m", "hello", flag, value]) == 2
    assert "run" not in rec
    assert "Invalid value" in capsys.readouterr().err


def test_main_run_usage_no_model(make_console, monkeypatch):
    monkeypatch.setattr("sys.argv", ["ara", "run"])
    assert cli.main() == 2


# --------------------------------------------------------------------------- #
# ara search — Hub search (engine-agnostic)
# --------------------------------------------------------------------------- #
def test_render_search_lists_results(make_console, monkeypatch):
    monkeypatch.setattr(cli.hub, "search",
                        lambda q: [{"id": "org/Smol", "downloads": 1000, "likes": 5}])
    c, buf = make_console()
    assert cli.render_search(c, "smol") == 0
    out = buf.getvalue()
    assert "HUB SEARCH: smol" in out and "org/Smol" in out and "1000" in out


def test_render_search_verbose_discloses_query_provenance(make_console, monkeypatch):
    monkeypatch.setattr(cli.hub, "search",
                        lambda q: [{"id": "org/Smol", "downloads": 1000, "likes": 5}])
    c, buf = make_console(verbose=True)
    assert cli.render_search(c, "smol") == 0
    out = " ".join(buf.getvalue().split())
    assert "source hf models list sorted by downloads · limit 20" in out


def test_render_search_empty(make_console, monkeypatch):
    monkeypatch.setattr(cli.hub, "search", lambda q: [])
    c, buf = make_console()
    assert cli.render_search(c, "zzz") == 0
    assert "no models found" in buf.getvalue()


def test_render_search_hf_missing(make_console, monkeypatch):
    monkeypatch.setattr(cli.hub, "search", lambda q: None)
    c, buf = make_console()
    assert cli.render_search(c, "x") == 1
    assert "hf command" in buf.getvalue() and "uv sync --frozen" in buf.getvalue()


def test_render_search_json(monkeypatch, capsys):
    monkeypatch.setattr(cli.hub, "search", lambda q: [{"id": "a", "downloads": 1, "likes": 0}])
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_search(c, "a", as_json=True) == 0
    assert json.loads(capsys.readouterr().out)[0]["id"] == "a"


def test_render_search_failure_json(monkeypatch, capsys):
    # Rule #3 (Honesty): a failed search under --json must emit {"error": ...}, not styled text.
    monkeypatch.setattr(cli.hub, "search", lambda q: None)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_search(c, "x", as_json=True) == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_main_search_dispatch(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "render_search",
                        lambda c, q, as_json=False: (seen.update(q=q, json=as_json) or 0))
    _run_main(monkeypatch, ["search", "smol", "lm", "--json"])
    assert seen == {"q": "smol lm", "json": True}


def test_main_search_no_query(monkeypatch, capsys):
    assert _run_main(monkeypatch, ["search"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Usage: ara search" in captured.err


# --------------------------------------------------------------------------- #
# Context-lever kwargs + flash-attention (SDPA default, FA2 opt-in on CUDA)
# --------------------------------------------------------------------------- #
def test_kv_fa_kwargs_per_backend():
    assert cli._kv_fa_kwargs("vulkan", flash_attn=True, flash_attn_optin=False,
                             kv_quant="q4_0") == {"flash_attn": True, "kv_quant": "q4_0"}
    # cuda: flash is the OPT-IN (default off → SDPA); kv-quant + weight-quant + prefill-chunk carried
    assert cli._kv_fa_kwargs("cuda", flash_attn=True, flash_attn_optin=True, kv_quant="q8_0",
                             weight_quant="int4") == {"kv_quant": "q8_0", "flash_attn": True,
                                                      "weight_quant": "int4", "prefill_chunk": None}
    assert cli._kv_fa_kwargs("apple", flash_attn=True, flash_attn_optin=True,
                             kv_quant="f16") == {"kv_quant": "f16"}
    assert cli._kv_fa_kwargs("cpu", flash_attn=True, flash_attn_optin=True, kv_quant="f16") == {}


def test_unsupported_lever_error():
    # kv-quant on an fp16-only engine → rejected; flash on apple (SDPA fused) → rejected
    assert "kv-quant" in cli._unsupported_lever_error(
        "cpu", kv_quant="q4_0", flash_attn=True, flash_attn_optin=False)
    assert "flash" in cli._unsupported_lever_error(
        "apple", kv_quant="f16", flash_attn=True, flash_attn_optin=True)     # --flash-attn
    assert "flash" in cli._unsupported_lever_error(
        "apple", kv_quant="f16", flash_attn=False, flash_attn_optin=False)   # --no-flash-attn
    # supported combos and bare defaults → no error
    assert cli._unsupported_lever_error(
        "cuda", kv_quant="q4_0", flash_attn=True, flash_attn_optin=True) is None
    assert cli._unsupported_lever_error(
        "apple", kv_quant="q8_0", flash_attn=True, flash_attn_optin=False) is None
    assert cli._unsupported_lever_error(
        "cpu", kv_quant="f16", flash_attn=True, flash_attn_optin=False) is None


def test_unsupported_lever_error_weight_quant():
    # --weight-quant only on cuda; rejected on apple/vulkan/cpu
    assert "weight-quant" in cli._unsupported_lever_error(
        "apple", kv_quant="f16", flash_attn=True, flash_attn_optin=False, weight_quant="int4")
    assert cli._unsupported_lever_error(
        "cuda", kv_quant="f16", flash_attn=True, flash_attn_optin=False, weight_quant="int4") is None
    assert cli._unsupported_lever_error(
        "cpu", kv_quant="f16", flash_attn=True, flash_attn_optin=False, weight_quant="none") is None


def test_weight_quant_hw_error_fp8():
    incapable = types.SimpleNamespace(fp8_capable=lambda: False)
    capable = types.SimpleNamespace(fp8_capable=lambda: True)
    assert "Ada/Hopper" in cli._weight_quant_hw_error(incapable, "cuda", "fp8")
    assert cli._weight_quant_hw_error(capable, "cuda", "fp8") is None       # capable → ok
    assert cli._weight_quant_hw_error(incapable, "cuda", "int4") is None    # only fp8 is gated
    assert cli._weight_quant_hw_error(object(), "cpu", "fp8") is None       # non-cuda never reaches


def test_measurement_config_captures_all_cuda_deviations():
    assert cli._measurement_config(
        "cuda", kv_quant="q8_0", flash_attn_optin=True,
        weight_quant="int4", prefill_chunk=256,
    ) == {
        "kv_quant": "q8_0", "flash_attn": True,
        "weight_quant": "int4", "prefill_chunk": 256,
    }


def test_legacy_nonconfigurable_measurement_remains_usable():
    assert cli._measurement_config_error(
        {"config": None}, {}, "cpu", "org/m") is None


# --------------------------------------------------------------------------- #
# characterize --engine ollama: residency ramp → measured ceiling (Slice 2)
# Spec 2026-07-04-characterize-through-ollama-ramp
# --------------------------------------------------------------------------- #
def test_ollama_ramp_contexts_schedule():
    assert cli._ollama_ramp_contexts(8192) == [2048, 4096, 8192]
    assert cli._ollama_ramp_contexts(3000) == [2048, 3000]
    assert cli._ollama_ramp_contexts(1000) == [1000]        # below the floor → just the max


def test_ollama_measure_ceiling_finds_the_wall(monkeypatch):
    st = {"ctx": 0}
    monkeypatch.setattr(cli.ollama, "create", lambda p, m, ctx: (st.update(ctx=ctx), True)[1])
    monkeypatch.setattr(cli.ollama, "load", lambda p: {})
    monkeypatch.setattr(cli.ollama, "ps",  # spill (size_vram < size) once ctx reaches 8192
                        lambda: [{"name": "pr", "context_length": st["ctx"],
                                  "size": 1000, "size_vram": 500 if st["ctx"] >= 8192 else 1000}])
    best, points = cli._ollama_measure_ceiling("m", 8192, "pr")
    assert best == 4096                                     # largest no-spill rung
    assert [p["fit"] for p in points] == [True, True, False]


def test_ollama_measure_ceiling_all_rungs_fit(monkeypatch):
    st = {"ctx": 0}
    monkeypatch.setattr(cli.ollama, "create", lambda p, m, ctx: (st.update(ctx=ctx), True)[1])
    monkeypatch.setattr(cli.ollama, "load", lambda p: {})
    monkeypatch.setattr(cli.ollama, "ps",   # never spills → ramp runs to the top rung
                        lambda: [{"name": "pr", "context_length": st["ctx"],
                                  "size": 1000, "size_vram": 1000}])
    best, points = cli._ollama_measure_ceiling("m", 8192, "pr")
    assert best == 8192 and all(p["fit"] for p in points)


def test_ollama_measure_ceiling_none_when_floor_spills(monkeypatch):
    monkeypatch.setattr(cli.ollama, "create", lambda p, m, ctx: True)
    monkeypatch.setattr(cli.ollama, "load", lambda p: {})
    monkeypatch.setattr(cli.ollama, "ps",
                        lambda: [{"name": "pr", "context_length": 2048,
                                  "size": 1000, "size_vram": 400}])
    best, points = cli._ollama_measure_ceiling("m", 2048, "pr")
    assert best is None and points[0]["fit"] is False


def test_ollama_measure_ceiling_stops_on_create_fail(monkeypatch):
    monkeypatch.setattr(cli.ollama, "create", lambda p, m, ctx: False)
    best, points = cli._ollama_measure_ceiling("m", 4096, "pr")
    assert best is None and points == []


def test_ollama_measure_ceiling_stops_when_governance_not_taken(monkeypatch):
    monkeypatch.setattr(cli.ollama, "create", lambda p, m, ctx: True)
    monkeypatch.setattr(cli.ollama, "load", lambda p: {})
    monkeypatch.setattr(cli.ollama, "ps", lambda: [])       # nothing loaded → entry None
    best, points = cli._ollama_measure_ceiling("m", 2048, "pr")
    assert best is None and points[0]["fit"] is False


@pytest.mark.parametrize("size,vram", [
    (None, 1000), (1000, None), ("1000", 1000), (1000, "1000"),
    (True, 1000), (1000, True), (0, 0), (-1, 0), (1000, -1),
])
def test_ollama_measure_ceiling_rejects_unverified_residency(monkeypatch, size, vram):
    monkeypatch.setattr(cli.ollama, "create", lambda p, m, ctx: True)
    monkeypatch.setattr(cli.ollama, "load", lambda p: {})
    monkeypatch.setattr(cli.ollama, "ps", lambda: [{
        "name": "pr", "context_length": 2048, "size": size, "size_vram": vram,
    }])
    best, points = cli._ollama_measure_ceiling("m", 2048, "pr")
    assert best is None and points[0]["fit"] is False


def test_ollama_measure_ceiling_binds_probe_and_base_manifest_before_load(monkeypatch):
    artifact = "ollama-manifest-sha256:" + "a" * 64
    monkeypatch.setattr(cli.ollama, "create", lambda *_a: True)
    monkeypatch.setattr(cli.ollama, "manifest_digest",
                        lambda name: "a" * 64 if name == "base" else "b" * 64)
    loads = []
    monkeypatch.setattr(cli.ollama, "load", lambda name: loads.append(name) or {})
    monkeypatch.setattr(cli.ollama, "ps", lambda: [{
        "name": "probe:latest", "context_length": 2048,
        "size": 10, "size_vram": 10, "digest": "b" * 64,
    }])
    provenance = {}
    best, points = cli._ollama_measure_ceiling(
        "base", 2048, "probe", base_artifact_id=artifact, provenance=provenance)
    assert best == 2048 and points[0]["fit"] is True and loads == ["probe"]
    assert provenance == {"created": True,
                          "artifact_id": "ollama-manifest-sha256:" + "b" * 64}


def test_ollama_measure_ceiling_can_track_probe_without_base_gate(monkeypatch):
    monkeypatch.setattr(cli.ollama, "create", lambda *_a: True)
    monkeypatch.setattr(cli.ollama, "manifest_digest", lambda _name: "b" * 64)
    monkeypatch.setattr(cli.ollama, "load", lambda *_a: {})
    monkeypatch.setattr(cli.ollama, "ps", lambda: [{
        "name": "probe:latest", "context_length": 2048,
        "size": 10, "size_vram": 10, "digest": "b" * 64,
    }])
    provenance = {}
    best, _points = cli._ollama_measure_ceiling(
        "base", 2048, "probe", provenance=provenance)
    assert best == 2048 and provenance["artifact_id"].endswith("b" * 64)


def test_ollama_measure_ceiling_refuses_retargeted_base_before_probe_load(monkeypatch):
    base_digests = iter(["a" * 64, "c" * 64])
    monkeypatch.setattr(cli.ollama, "create", lambda *_a: True)
    monkeypatch.setattr(
        cli.ollama, "manifest_digest",
        lambda name: next(base_digests) if name == "base" else "b" * 64,
    )
    monkeypatch.setattr(cli.ollama, "load",
                        lambda *_a: pytest.fail("loaded probe from retargeted base"))
    with pytest.raises(RuntimeError, match="changed during probe creation"):
        cli._ollama_measure_ceiling(
            "base", 2048, "probe",
            base_artifact_id="ollama-manifest-sha256:" + "a" * 64,
            provenance={},
        )


def test_ollama_measure_ceiling_refuses_retargeted_base_before_create(monkeypatch):
    monkeypatch.setattr(cli.ollama, "manifest_digest", lambda _name: "c" * 64)
    monkeypatch.setattr(cli.ollama, "create",
                        lambda *_a: pytest.fail("created from retargeted base"))
    with pytest.raises(RuntimeError, match="changed before probe creation"):
        cli._ollama_measure_ceiling(
            "base", 2048, "probe",
            base_artifact_id="ollama-manifest-sha256:" + "a" * 64,
            provenance={},
        )


def test_ollama_measure_ceiling_requires_probe_identity_without_provenance_dict(monkeypatch):
    monkeypatch.setattr(cli.ollama, "create", lambda *_a: True)
    monkeypatch.setattr(cli.ollama, "manifest_digest",
                        lambda name: "a" * 64 if name == "base" else None)
    with pytest.raises(RuntimeError, match="could not be identified"):
        cli._ollama_measure_ceiling(
            "base", 2048, "probe",
            base_artifact_id="ollama-manifest-sha256:" + "a" * 64,
        )


def test_ollama_measure_ceiling_accepts_verified_probe_without_provenance_dict(monkeypatch):
    monkeypatch.setattr(cli.ollama, "create", lambda *_a: True)
    monkeypatch.setattr(cli.ollama, "manifest_digest",
                        lambda name: "a" * 64 if name == "base" else "b" * 64)
    monkeypatch.setattr(cli.ollama, "load", lambda *_a: {})
    monkeypatch.setattr(cli.ollama, "ps", lambda: [{
        "name": "probe:latest", "context_length": 2048,
        "size": 10, "size_vram": 10, "digest": "b" * 64,
    }])
    best, points = cli._ollama_measure_ceiling(
        "base", 2048, "probe",
        base_artifact_id="ollama-manifest-sha256:" + "a" * 64,
    )
    assert best == 2048 and points[0]["fit"] is True


def test_cleanup_ollama_probe_refuses_retargeted_manifest(monkeypatch):
    monkeypatch.setattr(cli.ollama, "manifest_digest", lambda _name: "c" * 64)
    monkeypatch.setattr(cli.ollama, "load",
                        lambda *_a, **_k: pytest.fail("unloaded retargeted probe"))
    monkeypatch.setattr(cli.ollama, "delete",
                        lambda *_a, **_k: pytest.fail("deleted retargeted probe"))
    error = cli._cleanup_ollama_probe(
        "probe", "ollama-manifest-sha256:" + "b" * 64)
    assert "identity changed" in error and "refused" in error


def test_cleanup_ollama_probe_refuses_retarget_before_delete(monkeypatch):
    digests = iter(["b" * 64, "c" * 64])
    monkeypatch.setattr(cli.ollama, "manifest_digest", lambda _name: next(digests))
    monkeypatch.setattr(cli.ollama, "load", lambda *_a, **_k: {})
    monkeypatch.setattr(cli.ollama, "ps", lambda: [])
    monkeypatch.setattr(cli.ollama, "delete",
                        lambda *_a, **_k: pytest.fail("deleted retargeted probe"))
    error = cli._cleanup_ollama_probe(
        "probe", "ollama-manifest-sha256:" + "b" * 64)
    assert "identity changed" in error and "refused delete" in error


def test_cleanup_ollama_probe_unloads_verifies_then_deletes(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.ollama, "load",
                        lambda name, keep_alive=-1: calls.append(("load", name, keep_alive)) or {})
    monkeypatch.setattr(cli.ollama, "ps", lambda: [])
    monkeypatch.setattr(cli.ollama, "delete",
                        lambda name: calls.append(("delete", name)) or True)
    assert cli._cleanup_ollama_probe("probe") is None
    assert calls == [("load", "probe", 0), ("delete", "probe")]


def test_cleanup_ollama_probe_reports_still_resident_and_deletes(monkeypatch):
    monkeypatch.setattr(cli.ollama, "load", lambda name, keep_alive=-1: {})
    monkeypatch.setattr(cli.ollama, "ps", lambda: [{"name": "probe"}])
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    deleted = []
    monkeypatch.setattr(cli.ollama, "delete", lambda name: deleted.append(name) or True)
    assert "still resident" in cli._cleanup_ollama_probe("probe")
    assert deleted == []


def test_cleanup_ollama_probe_polls_until_absent_before_delete(monkeypatch):
    states = iter([[{"name": "probe"}], [{"name": "probe:latest"}], []])
    monkeypatch.setattr(cli.ollama, "load", lambda name, keep_alive=-1: {})
    monkeypatch.setattr(cli.ollama, "ps", lambda: next(states))
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    deleted = []
    monkeypatch.setattr(cli.ollama, "delete", lambda name: deleted.append(name) or True)
    assert cli._cleanup_ollama_probe("probe") is None
    assert deleted == ["probe"]


def test_cleanup_ollama_probe_reports_api_failures(monkeypatch):
    monkeypatch.setattr(cli.ollama, "load", lambda name, keep_alive=-1: None)
    monkeypatch.setattr(cli.ollama, "ps", lambda: None)
    deleted = []
    monkeypatch.setattr(cli.ollama, "delete", lambda name: deleted.append(name) or False)
    error = cli._cleanup_ollama_probe("probe")
    assert "request probe unload" in error
    assert "verify probe unload" in error
    assert deleted == []


def test_cleanup_ollama_probe_reports_delete_failure_after_verified_unload(monkeypatch):
    monkeypatch.setattr(cli.ollama, "load", lambda name, keep_alive=-1: {})
    monkeypatch.setattr(cli.ollama, "ps", lambda: [])
    monkeypatch.setattr(cli.ollama, "delete", lambda name: False)
    assert "delete probe model" in cli._cleanup_ollama_probe("probe")


def _wire_char_ollama(monkeypatch, *, in_store=True, max_ctx=8192):
    monkeypatch.setattr(cli.ollama, "version", lambda t=0.5: "0.30")
    monkeypatch.setattr(cli.ollama, "tags", lambda t=2.0: ["qwen3:0.6b"] if in_store else [])
    monkeypatch.setattr(cli, "_ollama_max_context", lambda model: max_ctx)
    monkeypatch.setattr(cli.ollama, "manifest_digest", lambda model: "a" * 64)
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mk")
    monkeypatch.setattr(cli, "_cleanup_ollama_probe",
                        lambda probe, expected_artifact_id=None: None)


def _fake_ollama_measure(result=None, error=None):
    def measure(*_args, provenance=None, **_kwargs):
        if provenance is not None:
            provenance.update(created=True,
                              artifact_id="ollama-manifest-sha256:" + "a" * 64)
        if error is not None:
            raise error
        return result
    return measure


def test_characterize_ollama_measures_and_records(store, monkeypatch, capsys):
    _wire_char_ollama(monkeypatch)
    monkeypatch.setattr(cli, "_ollama_measure_ceiling", _fake_ollama_measure(
        (4096, [{"context": 4096, "fit": True}])))
    c = cli.Console.from_env()
    # a bare Ollama name (colon) must NOT be rejected as an invalid HF ref — the ollama path
    # branches before valid_model_ref.
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama", as_json=True) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["safe_context"] == 4096 and out["source"] == "measured" and out["engine"] == "ollama"
    with cli.db.connected() as con:
        row = cli.db.get_characterization(con, "mk", "ollama", "qwen3:0.6b")
        assert row["safe_context"] == 4096
        assert row["artifact_id"] == "ollama-manifest-sha256:" + "a" * 64


def test_characterize_ollama_refuses_if_manifest_changes_during_measurement(
        store, monkeypatch, make_console):
    _wire_char_ollama(monkeypatch)
    base_digests = iter(["a" * 64, "b" * 64])
    monkeypatch.setattr(cli.ollama, "manifest_digest",
                        lambda name: next(base_digests) if name == "qwen3:0.6b" else "a" * 64)
    monkeypatch.setattr(cli, "_ollama_measure_ceiling",
                        _fake_ollama_measure((4096, [])))
    c, buf = make_console()
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama") == 1
    assert "manifest changed" in buf.getvalue()
    with cli.db.connected() as con:
        assert cli.db.get_characterization(con, "mk", "ollama", "qwen3:0.6b") is None


def test_characterize_ollama_refuses_unidentified_manifest(
        store, monkeypatch, make_console):
    _wire_char_ollama(monkeypatch)
    monkeypatch.setattr(cli.ollama, "manifest_digest", lambda _model: None)
    monkeypatch.setattr(cli, "_ollama_measure_ceiling",
                        lambda *_a: pytest.fail("measured unidentified manifest"))
    c, buf = make_console()
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama") == 1
    assert "artifact provenance" in buf.getvalue()


def test_characterize_ollama_refuses_preexisting_content_addressed_probe(
        store, monkeypatch, make_console):
    _wire_char_ollama(monkeypatch)
    artifact = "ollama-manifest-sha256:" + "a" * 64
    probe = cli._governed_name(
        "qwen3:0.6b", artifact_id=artifact, context=8192) + "-probe"
    monkeypatch.setattr(cli.ollama, "tags", lambda _t=2.0: ["qwen3:0.6b", probe])
    monkeypatch.setattr(cli, "_ollama_measure_ceiling",
                        lambda *_a, **_k: pytest.fail("overwrote preexisting probe"))
    monkeypatch.setattr(cli.ollama, "delete",
                        lambda *_a, **_k: pytest.fail("deleted preexisting probe"))
    c, buf = make_console()
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama") == 1
    assert "probe" in buf.getvalue() and "already exists" in buf.getvalue()


def test_characterize_ollama_refuses_when_final_probe_inventory_is_unavailable(
        store, monkeypatch, make_console):
    _wire_char_ollama(monkeypatch)
    calls = 0

    def tags(_timeout=2.0):
        nonlocal calls
        calls += 1
        return ["qwen3:0.6b"] if calls == 1 else None

    monkeypatch.setattr(cli.ollama, "tags", tags)
    c, buf = make_console()
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama") == 1
    assert "recheck" in buf.getvalue() and "probe collision" in buf.getvalue()


def test_characterize_ollama_refuses_when_probe_setup_is_locked(
        store, monkeypatch, make_console):
    _wire_char_ollama(monkeypatch)

    @contextlib.contextmanager
    def busy(_endpoint, _probe):
        raise cli.locking.OllamaSetupBusy("probe setup busy")
        yield

    monkeypatch.setattr(cli.locking, "ollama_setup_lock", busy)
    c, buf = make_console()
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama") == 1
    assert "probe setup busy" in buf.getvalue()


def test_characterize_ollama_refuses_cleanup_without_probe_provenance(
        store, monkeypatch, make_console):
    _wire_char_ollama(monkeypatch)

    def unidentified(*_args, provenance=None, **_kwargs):
        provenance["created"] = True
        raise RuntimeError("probe identity unavailable")

    monkeypatch.setattr(cli, "_ollama_measure_ceiling", unidentified)
    monkeypatch.setattr(cli, "_cleanup_ollama_probe",
                        lambda *_a, **_k: pytest.fail("cleaned unproven probe"))
    c, buf = make_console()
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama") == 1
    out = buf.getvalue()
    assert "identity unavailable" in out and "ownership could not be proven" in out


def test_characterize_ollama_handles_create_failure_without_cleanup(
        store, monkeypatch, make_console):
    _wire_char_ollama(monkeypatch)
    monkeypatch.setattr(cli, "_ollama_measure_ceiling",
                        lambda *_a, **_k: (None, []))
    monkeypatch.setattr(cli, "_cleanup_ollama_probe",
                        lambda *_a, **_k: pytest.fail("cleaned nonexistent probe"))
    c, buf = make_console()
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama") == 0
    assert "no no-spill ceiling" in buf.getvalue()


def test_characterize_ollama_refuses_to_store_when_cleanup_is_unverified(
        store, monkeypatch, make_console):
    _wire_char_ollama(monkeypatch)
    monkeypatch.setattr(cli, "_ollama_measure_ceiling", _fake_ollama_measure((4096, [])))
    monkeypatch.setattr(cli, "_cleanup_ollama_probe",
                        lambda probe, expected_artifact_id=None: "probe is still resident")
    c, buf = make_console()
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama") == 1
    assert "still resident" in buf.getvalue()
    with cli.db.connected() as con:
        assert cli.db.get_characterization(con, "mk", "ollama", "qwen3:0.6b") is None


def test_characterize_ollama_measurement_exception_still_cleans_up(
        store, monkeypatch, make_console):
    _wire_char_ollama(monkeypatch)
    monkeypatch.setattr(cli, "_ollama_measure_ceiling",
                        _fake_ollama_measure(error=RuntimeError("probe failed")))
    cleaned = []
    monkeypatch.setattr(cli, "_cleanup_ollama_probe",
                        lambda probe, expected_artifact_id=None: cleaned.append(probe) or None)
    c, buf = make_console()
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama") == 1
    assert "probe failed" in buf.getvalue()
    assert cleaned and cleaned[0].endswith("-probe")


def test_characterize_ollama_reports_measurement_and_cleanup_failures(
        store, monkeypatch, make_console):
    _wire_char_ollama(monkeypatch)
    monkeypatch.setattr(cli, "_ollama_measure_ceiling",
                        _fake_ollama_measure(error=RuntimeError("probe failed")))
    monkeypatch.setattr(cli, "_cleanup_ollama_probe",
                        lambda probe, expected_artifact_id=None: "probe still resident")
    c, buf = make_console()
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama") == 1
    assert "probe failed" in buf.getvalue() and "still resident" in buf.getvalue()


def test_characterize_ollama_text_reports_measured_ceiling(store, monkeypatch, make_console):
    _wire_char_ollama(monkeypatch)
    monkeypatch.setattr(cli, "_ollama_measure_ceiling", _fake_ollama_measure(
        (4096, [{"context": 4096, "fit": True}])))
    monkeypatch.setattr(cli.ollama, "delete", lambda name: True)
    c, buf = make_console()
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama") == 0
    out = buf.getvalue()
    assert "measured ceiling" in out and "4096" in out


def test_characterize_ollama_verbose_discloses_runtime_and_model_limit(
        store, monkeypatch, make_console):
    _wire_char_ollama(monkeypatch)
    monkeypatch.setattr(cli, "_ollama_measure_ceiling", _fake_ollama_measure(
        (4096, [{"context": 4096, "fit": True}])))
    monkeypatch.setattr(cli.ollama, "delete", lambda name: True)
    c, buf = make_console(verbose=True)
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama") == 0
    out = " ".join(buf.getvalue().split())
    assert "engine ollama external runtime" in out
    assert "model limit 8192 tokens architecture maximum" in out


@pytest.mark.parametrize("kwargs", [
    {"kv_quant": "q4_0"},
    {"weight_quant": "int4"},
    {"prefill_chunk": 256},
    {"flash_attn": False},
    {"flash_attn_optin": True},
])
def test_characterize_ollama_rejects_unsupported_context_levers(
        store, monkeypatch, make_console, kwargs):
    _wire_char_ollama(monkeypatch)
    c, buf = make_console()
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama", **kwargs) == 1
    out = buf.getvalue()
    assert "supported" in out or "tunable" in out


def test_characterize_ollama_rejects_unsupported_context_lever_json(
        store, monkeypatch, capsys):
    _wire_char_ollama(monkeypatch)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(
        c, "qwen3:0.6b", engine="ollama", kv_quant="q4_0", as_json=True) == 1
    assert "supported" in json.loads(capsys.readouterr().out)["error"]


def test_characterize_ollama_records_none_when_it_spills(store, monkeypatch, make_console):
    _wire_char_ollama(monkeypatch)
    monkeypatch.setattr(cli, "_ollama_measure_ceiling", _fake_ollama_measure(
        (None, [{"context": 2048, "fit": False}])))
    monkeypatch.setattr(cli.ollama, "delete", lambda name: True)
    c, buf = make_console()
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama") == 0
    assert "no no-spill ceiling" in buf.getvalue()
    with cli.db.connected() as con:
        assert cli.db.get_characterization(con, "mk", "ollama", "qwen3:0.6b")["safe_context"] is None


def test_characterize_ollama_not_serving(make_console, monkeypatch):
    monkeypatch.setattr(cli.ollama, "version", lambda t=0.5: None)
    c, buf = make_console()
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama") == 1
    assert "Ollama isn't serving" in buf.getvalue()


def test_characterize_ollama_tags_unreachable(make_console, monkeypatch):
    monkeypatch.setattr(cli.ollama, "version", lambda t=0.5: "0.30")
    monkeypatch.setattr(cli.ollama, "tags", lambda t=2.0: None)
    c, buf = make_console()
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama") == 1
    assert "couldn't list Ollama models" in buf.getvalue()


def test_characterize_ollama_not_in_store(make_console, monkeypatch):
    _wire_char_ollama(monkeypatch, in_store=False)
    c, buf = make_console()
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama") == 1
    assert "isn't in Ollama" in buf.getvalue()


def test_characterize_ollama_no_max_context(make_console, monkeypatch):
    _wire_char_ollama(monkeypatch, max_ctx=None)
    c, buf = make_console()
    assert cli.render_characterize(c, "qwen3:0.6b", engine="ollama") == 1
    assert "context length" in buf.getvalue()


def test_ollama_max_context_reads_arch(monkeypatch):
    monkeypatch.setattr(cli.ollama, "show", lambda m: {"model_info": {
        "general.architecture": "qwen3", "qwen3.context_length": 32768}})
    assert cli._ollama_max_context("qwen3:0.6b") == 32768
    monkeypatch.setattr(cli.ollama, "show", lambda m: None)
    assert cli._ollama_max_context("x") is None
    monkeypatch.setattr(cli.ollama, "show", lambda m: {"model_info": {"general.architecture": 5}})
    assert cli._ollama_max_context("x") is None
    monkeypatch.setattr(cli.ollama, "show", lambda m: {"model_info": {
        "general.architecture": "llama", "llama.context_length": 0}})
    assert cli._ollama_max_context("x") is None


def test_render_characterize_rejects_invalid_weight_quant(make_console, monkeypatch):
    _wire_characterize(monkeypatch, backend="cuda")
    c, buf = make_console()
    assert cli.render_characterize(c, "org/M", weight_quant="int3") == 1
    assert "invalid --weight-quant" in buf.getvalue()


def test_render_characterize_rejects_fp8_on_incapable_gpu(make_console, monkeypatch):
    _wire_characterize(monkeypatch, backend="cuda")
    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(
        fp8_capable=lambda: False,
        characterize=lambda m, **k: {"model": m, "safe_context": 1, "points": []}))
    c, buf = make_console()
    assert cli.render_characterize(c, "org/M", weight_quant="fp8") == 1
    assert "Ada/Hopper" in buf.getvalue()


def test_characterize_cuda_persists_effective_sdpa_fallback_config(
        make_console, store, monkeypatch):
    _wire_characterize(monkeypatch, backend="cuda")
    seen = {}

    def characterize(model, *, progress=False, flash_attn=False, **kwargs):
        seen["flash_attn_requested"] = flash_attn
        return {"model": model, "safe_context": 4096, "decode_context": None, "points": []}

    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(
        flash_attn_capable=lambda: False, characterize=characterize,
        calibration_model_cached=lambda model: True,
        download_calibration_model=lambda model, *, progress=False: None,
    ))
    c, _ = make_console()
    assert cli.render_characterize(c, "org/M", engine="cuda", flash_attn_optin=True) == 0
    assert seen["flash_attn_requested"] is True  # backend performs the documented SDPA fallback
    with cli.db.connected() as con:
        assert cli.db.get_characterization(con, "mkey", "cuda", "org/M")["config"] == {}


def test_render_run_rejects_fp8_on_incapable_gpu(monkeypatch, capsys):
    _wire_run_cross(monkeypatch, detected="cuda",
                    chars={"cuda": {"model_id": "org/m", "safe_context": 4096,
                                    "config": {"weight_quant": "fp8"}}},
                    supports={"cuda": True})
    # the fake backend must expose generate (run-selectable) + fp8_capable (False → reject fp8)
    bk = types.SimpleNamespace(generate=lambda *a, **k: {"completion": "x"},
                               fp8_capable=lambda: False)
    monkeypatch.setattr(cli, "get_backend", lambda b=None: bk)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_run(c, "org/m", prompt="hi", as_json=True, assume_yes=True,
                          weight_quant="fp8") == 1
    assert "Ada/Hopper" in json.loads(capsys.readouterr().out)["error"]


def test_render_run_rejects_invalid_weight_quant(monkeypatch, capsys):
    _wire_run_cross(monkeypatch, detected="cuda",
                    chars={"cuda": {"model_id": "org/m", "safe_context": 4096}},
                    supports={"cuda": True})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_run(c, "org/m", prompt="hi", as_json=True, assume_yes=True,
                          weight_quant="int3") == 1
    assert "invalid --weight-quant" in json.loads(capsys.readouterr().out)["error"]


def test_main_characterize_threads_weight_quant(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["characterize", "org/M", "--weight-quant", "int4"])
    assert rec["characterize_wq"] == "int4"
    rec2 = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["characterize", "org/M", "--weight-quant=int8"])   # = form
    assert rec2["characterize_wq"] == "int8"


def test_kv_fa_kwargs_includes_prefill_chunk_for_cuda():
    kw = cli._kv_fa_kwargs("cuda", flash_attn=True, flash_attn_optin=False, kv_quant="f16",
                           weight_quant="none", prefill_chunk=256)
    assert kw["prefill_chunk"] == 256
    # engines with no chunked-prefill path never carry the kwarg
    assert "prefill_chunk" not in cli._kv_fa_kwargs(
        "vulkan", flash_attn=True, flash_attn_optin=False, kv_quant="f16", prefill_chunk=256)


def test_unsupported_lever_error_prefill_chunk():
    # chunked prefill is cuda-only; rejected (explicitly) on the others
    assert "chunked prefill" in cli._unsupported_lever_error(
        "cpu", kv_quant="f16", flash_attn=True, flash_attn_optin=False, prefill_chunk=256)
    assert cli._unsupported_lever_error(
        "cuda", kv_quant="f16", flash_attn=True, flash_attn_optin=False, prefill_chunk=256) is None
    assert cli._unsupported_lever_error(            # not requested → no error anywhere
        "cpu", kv_quant="f16", flash_attn=True, flash_attn_optin=False, prefill_chunk=None) is None


def test_main_characterize_threads_prefill_chunk(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["characterize", "org/M", "--prefill-chunk", "256"])
    assert rec["characterize_chunk"] == 256
    rec2 = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["characterize", "org/M", "--chunked-prefill"])   # bare → default 512
    assert rec2["characterize_chunk"] == 512
    rec3 = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["characterize", "org/M"])                        # off by default
    assert rec3["characterize_chunk"] is None


def test_main_run_threads_prefill_chunk(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["run", "org/M", "hello", "--prefill-chunk", "128"])
    assert rec["run"]["prefill_chunk"] == 128


def test_main_characterize_prefill_chunk_eq_form(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["characterize", "org/M", "--prefill-chunk=384"])   # = form
    assert rec["characterize_chunk"] == 384


@pytest.mark.parametrize("value", ["0", "-1"])
def test_main_characterize_rejects_nonpositive_prefill_chunk(monkeypatch, capsys, value):
    _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["characterize", "org/M", "--prefill-chunk", value]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Invalid value for '--prefill-chunk'" in captured.err


def test_main_prefill_chunk_non_integer_is_click_error(monkeypatch, capsys):
    _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["characterize", "org/M", "--prefill-chunk", "big"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Invalid value for '--prefill-chunk'" in captured.err


def test_int_or_none_parses_and_rejects():
    assert cli._int_or_none("512") == 512
    assert cli._int_or_none("nope") is None
    assert cli._int_or_none("") is None


def test_render_characterize_rejects_unsupported_chunked_prefill(make_console, monkeypatch):
    _wire_characterize(monkeypatch, backend="cpu")
    c, buf = make_console()
    assert cli.render_characterize(c, "org/M", prefill_chunk=256) == 1
    assert "chunked prefill" in buf.getvalue() and "cpu" in buf.getvalue()


def test_render_characterize_rejects_unsupported_lever(make_console, monkeypatch):
    _wire_characterize(monkeypatch, backend="cpu")
    c, buf = make_console()
    assert cli.render_characterize(c, "org/M", kv_quant="q4_0") == 1
    assert "kv-quant" in buf.getvalue() and "cpu" in buf.getvalue()


def test_render_characterize_rejects_unsupported_lever_json(monkeypatch, capsys):
    _wire_characterize(monkeypatch, backend="cpu")
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(c, "org/M", kv_quant="q4_0", as_json=True) == 1
    assert "kv-quant" in json.loads(capsys.readouterr().out)["error"]


@pytest.mark.parametrize("as_json", [False, True])
def test_render_characterize_rejects_conflicting_flash_flags(
        make_console, monkeypatch, capsys, as_json):
    _wire_characterize(monkeypatch, backend="cuda", characterize=lambda m: {
        "model": m, "safe_context": 1, "decode_context": None, "points": []})
    c, buf = make_console()
    assert cli.render_characterize(
        c, "org/M", as_json=as_json, flash_attn=False, flash_attn_optin=True) == 1
    out = json.loads(capsys.readouterr().out)["error"] if as_json else buf.getvalue()
    assert "--flash-attn" in out and "--no-flash-attn" in out


def test_render_run_rejects_unsupported_lever(monkeypatch, capsys):
    _wire_run_cross(monkeypatch, detected="cpu",
                    chars={"cpu": {"model_id": "org/m", "safe_context": 4096}},
                    supports={"cpu": True})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_run(c, "org/m", prompt="hi", as_json=True, assume_yes=True,
                          kv_quant="q4_0") == 1
    assert "kv-quant" in json.loads(capsys.readouterr().out)["error"]


def test_render_run_pinned_rejects_unsupported_lever(monkeypatch, capsys):
    _wire_run_cross(monkeypatch, detected="cpu",
                    chars={"cpu": {"model_id": "org/m", "safe_context": 4096}},
                    supports={"cpu": True})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_run(c, "org/m", prompt="hi", engine="cpu", as_json=True,
                          assume_yes=True, kv_quant="q4_0") == 1
    assert "kv-quant" in json.loads(capsys.readouterr().out)["error"]


def test_flash_sdpa_note_warns_when_cuda_incapable(make_console):
    c, buf = make_console()
    bk = types.SimpleNamespace(flash_attn_capable=lambda: False)
    cli._flash_sdpa_note(c, bk, "cuda", True, False)
    assert "SDPA" in buf.getvalue()


def test_flash_sdpa_note_silent_when_capable(make_console):
    c, buf = make_console()
    bk = types.SimpleNamespace(flash_attn_capable=lambda: True)
    cli._flash_sdpa_note(c, bk, "cuda", True, False)
    assert buf.getvalue() == ""


def test_flash_sdpa_note_skipped_under_json(make_console):
    c, buf = make_console()
    bk = types.SimpleNamespace(flash_attn_capable=lambda: False)
    cli._flash_sdpa_note(c, bk, "cuda", True, True)        # as_json → no styled line
    assert buf.getvalue() == ""


def test_flash_sdpa_note_silent_without_optin(make_console):
    c, buf = make_console()
    bk = types.SimpleNamespace(flash_attn_capable=lambda: False)
    cli._flash_sdpa_note(c, bk, "cuda", False, False)      # no opt-in → nothing to say
    assert buf.getvalue() == ""


def test_main_characterize_flash_attn_optin(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["characterize", "org/M", "--flash-attn"])
    assert rec["characterize_fa_optin"] is True


def test_main_characterize_flash_attn_optin_default_false(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["characterize", "org/M"])
    assert rec["characterize_fa_optin"] is False


# --------------------------------------------------------------------------- #
# --json honesty (Rule #3): Click grammar errors stay stderr/2; operational
# exceptions after a valid command emit {"error": ...} under --json.
# --------------------------------------------------------------------------- #
def test_main_usage_errors_json(monkeypatch, capsys):
    assert _run_main(monkeypatch, ["search", "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Missing argument" in captured.err
    for argv in (["characterize"], ["run"]):
        assert _run_main(monkeypatch, [*argv, "--json"]) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Missing argument" in captured.err


def test_main_unknown_command_json_flag_does_not_override_click(monkeypatch, capsys):
    assert _run_main(monkeypatch, ["bogus", "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No such command 'bogus'" in captured.err


def _raise_engine_env(*a, **k):
    raise RuntimeError("worker crashed")


def test_main_uncaught_exception_json_emits_error(monkeypatch, capsys):
    # The front-door guard: an exception bubbling out of a renderer (e.g. EngineEnvError) under
    # --json becomes a structured error, not a raw traceback.
    monkeypatch.setattr(cli, "render_detect", _raise_engine_env)
    assert _run_main(monkeypatch, ["detect", "--json"]) == 1
    assert "worker crashed" in json.loads(capsys.readouterr().out)["error"]


def test_main_uncaught_exception_without_json_propagates(monkeypatch):
    # Without --json, a NON-EngineEnvError still propagates unchanged (the friendly branch is
    # scoped to EngineEnvError; anything else surfaces its traceback).
    monkeypatch.setattr(cli, "render_detect", _raise_engine_env)
    with pytest.raises(RuntimeError, match="worker crashed"):
        _run_main(monkeypatch, ["detect"])


def _raise_engine_env_error(*a, **k):
    raise cli.EngineEnvError("cuda env is broken")


def test_main_engine_env_error_without_json_is_friendly(monkeypatch, capsys):
    # An EngineEnvError escaping a command WITHOUT --json prints a friendly one-line diagnostic
    # (no raw traceback) and exits 1 — the non-json front-door honesty branch (Rule #3).
    monkeypatch.setattr(cli, "render_detect", _raise_engine_env_error)
    assert _run_main(monkeypatch, ["detect"]) == 1
    out = capsys.readouterr().out
    assert "engine env problem: cuda env is broken" in out
    assert "ara install" in out
    assert "Traceback" not in out


def test_main_engine_env_error_with_json_is_structured(monkeypatch, capsys):
    # Under --json the EngineEnvError still becomes a structured {"error": ...} (json branch wins).
    monkeypatch.setattr(cli, "render_detect", _raise_engine_env_error)
    assert _run_main(monkeypatch, ["detect", "--json"]) == 1
    assert "cuda env is broken" in json.loads(capsys.readouterr().out)["error"]


# --------------------------------------------------------------------------- #
# ara models <id> — single-model detail (mlx's `show`)
# --------------------------------------------------------------------------- #
def _meta(**over):
    base = dict(modality="text", n_layers=30, hidden_size=576, kv_heads=3,
                head_dim=64, max_context=8192, quant="mlx-4bit")
    base.update(over)
    return base


def test_model_detail_full(make_console, monkeypatch):
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: {
                            "safe_context": 16000, "decode_context": None, "config": {}
                        })
    c, buf = make_console()
    assert cli.render_model_detail(c, "org/Smol") == 0
    out = buf.getvalue()
    assert "org/Smol" in out and "3 heads × 64 dim" in out
    assert "8192" in out and "mlx-4bit" in out and "16000" in out


def test_model_detail_looks_up_local_evidence_by_absolute_key(
        make_console, monkeypatch, tmp_path):
    model = tmp_path / "relative:Model-Q4_K_M.gguf"
    model.write_bytes(b"weights")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.catalog, "describe", lambda _mid: _meta())
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    seen = []
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda _con, _mk, _engine, model_id:
                        seen.append(model_id) or None)
    c, _ = make_console()

    assert cli.render_model_detail(c, model.name) == 0
    assert set(seen) == {str(model.resolve())}


def test_model_detail_marks_replaced_local_artifact_ceiling_stale(
        monkeypatch, capsys, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"original")
    old_artifact = "local-gguf:old"
    model.write_bytes(b"replacement weights")
    monkeypatch.setattr(cli.staleness, "artifact_matches", lambda *_a: False)
    monkeypatch.setattr(cli.catalog, "describe", lambda _mid: _meta())
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda _con, _mk, engine, _model: {
                            "safe_context": 8000, "decode_context": None, "config": {},
                            "artifact_id": old_artifact,
                        } if engine == "cpu" else None)
    c = cli.Console(color=False, stream=sys.stderr)

    assert cli.render_model_detail(c, str(model), as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["stale_ceiling"] is True


def test_model_detail_verbose_discloses_measurement_time(make_console, monkeypatch):
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: ({"safe_context": 16000,
                                                  "decode_context": None,
                                                  "measured_at": "2026-07-02T12:00:00+00:00"}
                                                 if e == "mlx" else None))
    c, buf = make_console(verbose=True)
    assert cli.render_model_detail(c, "org/Smol") == 0
    assert "measured 2026-07-02T12:00:00+00:00" in buf.getvalue()


def test_models_help_is_clear_and_lists_recommend_use_cases(capsys):
    assert cli.main(["models", "--help"]) == 0
    group_help = " ".join(capsys.readouterr().out.split())
    assert "Search the Hub, rank cached models, or inspect one cached model" in group_help

    assert cli.main(["models", "recommend", "--help"]) == 0
    recommend_help = " ".join(capsys.readouterr().out.split())
    assert "estimated usable context or capability evidence" in recommend_help
    assert "extraction, reasoning, rag, agentic, or coding" in recommend_help

    assert cli.main(["models", "show", "--help"]) == 0
    show_help = " ".join(capsys.readouterr().out.split())
    assert "cached architecture and this machine's measured ceilings" in show_help


def test_model_detail_sparse_no_engine(make_console, monkeypatch):
    monkeypatch.setattr(cli.catalog, "describe",
                        lambda mid: _meta(modality=None, n_layers=None, kv_heads=None,
                                          head_dim=None, max_context=None, quant=None))
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "unsupported")  # engine_key None
    c, buf = make_console()
    assert cli.render_model_detail(c, "x") == 0
    out = buf.getvalue()
    assert "?" in out and "none" in out and "not characterized" in out


def test_model_detail_not_found(make_console, monkeypatch):
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: None)
    c, buf = make_console()
    assert cli.render_model_detail(c, "nope") == 1
    assert "couldn't describe" in buf.getvalue()


def test_model_detail_does_not_create_database(tmp_path, make_console, monkeypatch):
    path = tmp_path / "missing" / "ara.db"
    monkeypatch.setenv("ARA_DB_PATH", str(path))
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())

    c, _ = make_console()
    assert cli.render_model_detail(c, "org/Smol") == 0
    assert not path.exists()


def test_model_detail_json(monkeypatch, capsys):
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: {
                            "safe_context": 9000, "decode_context": None,
                            "config": {"kv_quant": "q4_0"},
                        })
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_model_detail(c, "org/A", as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["model_id"] == "org/A" and data["safe_context"] == 9000
    assert data["decode_context"] is None
    assert all(config == {"kv_quant": "q4_0"}
               for config in data["engine_configs"].values())


def test_model_detail_text_discloses_nondefault_measurement_config(make_console, monkeypatch):
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization", lambda con, mk, e, mid: (
        {"safe_context": 16000, "decode_context": None,
         "config": {"kv_quant": "q4_0", "flash_attn": False}}
        if e == "vulkan" else None
    ))
    c, buf = make_console()
    assert cli.render_model_detail(c, "org/A") == 0
    assert "flash-attn=false" in buf.getvalue()
    assert "kv-quant=q4_0" in buf.getvalue()


def test_model_detail_reads_legacy_characterization_without_migrating(
        store, monkeypatch, capsys):
    from ara import db

    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    store.execute(
        "INSERT INTO characterizations "
        "(machine_key, engine, model_id, safe_context, decode_context, points_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("mkey", "wmx", "org/A", 4096, 8192, "[]"),
    )
    store.execute("PRAGMA user_version = 2")
    store.commit()
    store.close()
    path = db._db_path()
    backup_path = path.with_name(path.name + ".pre-engine-identity-v3.bak")
    backup_path.unlink(missing_ok=True)

    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_model_detail(c, "org/A", as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["safe_context"] == 4096
    assert data["decode_context"] == 8192
    assert data["engines"] == {"mlx": 4096}
    with sqlite3.connect(path) as check:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 2
        assert check.execute("SELECT engine FROM characterizations").fetchone()[0] == "wmx"
    assert not backup_path.exists()


def test_model_detail_flags_stale_ceiling_text(make_console, monkeypatch):
    """`ara models <id>` must not present a stored ceiling as authoritative when the model's cache
    changed since it was measured — flag it inline (Rule #3), same honesty serve/run already give."""
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: {"safe_context": 16000, "decode_context": None,
                                                 "measured_at": "2026-01-01T00:00:00+00:00"})
    monkeypatch.setattr(cli.staleness, "ceiling_is_stale", lambda mid, at: True)
    c, buf = make_console()
    assert cli.render_model_detail(c, "org/Smol") == 0
    out = buf.getvalue()
    assert "16000" in out and "stale" in out.lower()


def test_model_detail_json_stale_flag(monkeypatch, capsys):
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: {"safe_context": 9000, "decode_context": None,
                                                 "measured_at": "2026-01-01T00:00:00+00:00"})
    monkeypatch.setattr(cli.staleness, "ceiling_is_stale", lambda mid, at: True)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_model_detail(c, "org/A", as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["stale_ceiling"] is True


def test_model_detail_measured_no_ceiling(make_console, monkeypatch):
    """`ara models <id>` for a measured-but-unfit model reads 'no safe ceiling',
    not 'not characterized' — consistent with `ara models` and `ara profile`."""
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: {"safe_context": None, "decode_context": None})
    c, buf = make_console()
    assert cli.render_model_detail(c, "org/Unfit") == 0
    out = buf.getvalue()
    assert "no safe ceiling" in out
    assert "not characterized" not in out


def test_model_detail_json_characterized_flag(monkeypatch, capsys):
    """Detail JSON flags a measured-but-unfit model as characterized with a null ceiling."""
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: {"safe_context": None, "decode_context": None})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_model_detail(c, "org/Unfit", as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["safe_context"] is None
    assert data.get("characterized") is True
    assert data.get("decode_context") is None


def test_model_detail_json_uncharacterized_flag(monkeypatch, capsys):
    """Detail JSON flags a never-measured model as not characterized."""
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "unsupported")  # engine_key None → ch None
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_model_detail(c, "x", as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["safe_context"] is None
    assert data.get("characterized") is False
    assert data.get("decode_context") is None


def test_main_models_id_dispatch(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "render_model_detail",
                        lambda c, mid, as_json=False: (seen.update(mid=mid) or 0))
    _run_main(monkeypatch, ["models", "org/Smol"])
    assert seen["mid"] == "org/Smol"


# --------------------------------------------------------------------------- #
# ara characterize <model> — measure + store a model's ceiling (any engine)
# --------------------------------------------------------------------------- #
def _wire_characterize(monkeypatch, *, backend="apple", engine_ok=True, characterize=None):
    monkeypatch.setattr(cli.detect, "backend_name", lambda: backend)
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (engine_ok, "ara-engine-mlx"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    monkeypatch.setattr(cli.staleness, "artifact_identity", lambda _model: "artifact:test")
    monkeypatch.setattr(cli.staleness, "artifact_size_gb", lambda _model: 1.0)
    if characterize is not None:
        # Wrap plain lambdas so they accept the progress= kwarg render_characterize passes.
        _char = characterize
        def _char_wrapper(m, *, progress=False, **_kwargs):
            return _char(m)
        monkeypatch.setattr(cli, "get_backend",
                            lambda b=None: types.SimpleNamespace(
                                characterize=_char_wrapper,
                                calibration_model_cached=lambda m: True,   # skip pre-fetch
                                download_calibration_model=lambda m, *, progress=False: None,
                            ))


def test_render_characterize_persists_and_shows(make_console, store, monkeypatch):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 20000,
                                               "decode_context": None, "points": [[512, 1.4]]})
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Model") == 0
    assert "20000" in buf.getvalue() and "mlx" in buf.getvalue()
    row = cli.db.get_characterization(store, "mkey", "mlx", "org/Model")
    assert row["safe_context"] == 20000 and row["points"] == [[512, 1.4]]
    assert row["artifact_id"] == "artifact:test"


def test_render_characterize_loads_immutable_pinned_artifact(
        make_console, store, monkeypatch):
    seen = {}
    _wire_characterize(
        monkeypatch,
        characterize=lambda model: seen.update(model=model) or {
            "model": model, "safe_context": 20000,
            "decode_context": None, "points": [[512, 1.4]]})
    monkeypatch.setattr(cli.staleness, "pinned_model_ref",
                        lambda _model, _artifact: "/cache/snapshots/rev")
    c, _ = make_console()
    assert cli.render_characterize(c, "org/Model") == 0
    assert seen["model"] == "/cache/snapshots/rev"


def test_render_characterize_refuses_when_artifact_cannot_be_pinned(
        make_console, monkeypatch):
    _wire_characterize(
        monkeypatch,
        characterize=lambda _model: pytest.fail("characterize must not run"))
    monkeypatch.setattr(cli.staleness, "pinned_model_ref", lambda *_a: None)
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Model") == 1
    assert "cannot pin" in buf.getvalue()


def test_render_characterize_refuses_to_store_unidentified_artifact(
        make_console, monkeypatch):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 20000,
                                                "decode_context": None, "points": []})
    monkeypatch.setattr(cli.staleness, "artifact_identity", lambda _model: None)
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Model") == 1
    assert "result not stored" in buf.getvalue()


def test_render_characterize_refuses_artifact_changed_during_measurement(
        make_console, monkeypatch):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 20000,
                                                "decode_context": None, "points": []})
    identities = iter(("artifact:before", "artifact:after"))
    monkeypatch.setattr(cli.staleness, "artifact_identity", lambda _model: next(identities))
    monkeypatch.setattr(cli.db, "save_characterization",
                        lambda *_a, **_k: pytest.fail("changed evidence must not be stored"))
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Model") == 1
    assert "changed during characterization" in buf.getvalue()


def test_render_characterize_persists_absolute_local_evidence_key(
        make_console, monkeypatch, tmp_path):
    model = tmp_path / "local:Model-Q4_K_M.gguf"
    model.write_bytes(b"weights")
    monkeypatch.chdir(tmp_path)
    _wire_characterize(monkeypatch, backend="cpu",
                       characterize=lambda m: {"model": m, "safe_context": 20000,
                                                "decode_context": None, "points": []})
    saved = {}
    monkeypatch.setattr(cli.db, "save_characterization",
                        lambda _con, _mk, _engine, model_id, **_kw:
                        saved.update(model_id=model_id))
    c, _ = make_console()
    assert cli.render_characterize(c, model.name, engine="cpu") == 0
    assert saved["model_id"] == str(model.resolve())


def test_render_characterize_catalogs_exact_gguf_variant(make_console, monkeypatch):
    _wire_characterize(monkeypatch, backend="cpu",
                       characterize=lambda m: {"model": m, "safe_context": 20000,
                                                "decode_context": None, "points": []})
    captured = {}
    monkeypatch.setattr(cli.catalog, "remember_variant",
                        lambda con, model, canonical, **kw:
                        captured.update(model=model, canonical=canonical, **kw))
    selector = "org/repo:Model-Q4_K_M.gguf"
    c, _ = make_console()
    assert cli.render_characterize(c, selector, engine="cpu") == 0
    assert captured == {"model": selector, "canonical": "org/repo",
                        "quant": "q4_k_m", "weights_gb": 1.0}


def test_characterize_verbose_discloses_engine_and_effective_kv_cache(
        make_console, store, monkeypatch):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 20000,
                                               "decode_context": None, "points": []})
    c, buf = make_console(verbose=True)
    assert cli.render_characterize(c, "org/Model", kv_quant="q8_0") == 0
    out = " ".join(buf.getvalue().split())
    assert "engine mlx" in out
    assert "KV cache q8_0" in out


def test_characterize_self_calibrates_when_uncalibrated(make_console, store, monkeypatch):
    # Spec 2026-06-23-capability-pipeline (Slice 2): characterize owns calibration — it measures +
    # persists the engine baseline once when none is stored, before the ramp.
    calls = []
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "MLX engine"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(
        characterize=lambda m, *, progress=False, kv_quant="f16": {"model": m, "safe_context": 9000, "decode_context": None, "points": []},
        calibration_model_cached=lambda m: True,
        download_calibration_model=lambda m, *, progress=False: None,
        calibrate=lambda: (calls.append("cal") or {"overhead_gb": 1.7,
                                                    "wall_gb": 41.3, "safe_budget_gb": 39.3}),
    ))
    c, _ = make_console()
    assert cli.render_characterize(c, "org/M") == 0
    assert calls == ["cal"]                                            # calibrated once, before ramp
    row = cli.db.get_calibration(store, "mkey", "mlx")
    assert row["fixed_overhead_gb"] == 1.7                             # persisted
    # The measured wall + budget ride alongside so profile/recommend can report reality.
    assert row["wall_gb"] == 41.3 and row["safe_budget_gb"] == 39.3


def test_characterize_warns_when_calibration_unavailable(make_console, store, monkeypatch):
    # Honesty (Rule #3): a failed calibration must be surfaced, not silently replaced by the
    # conservative default. The ramp still proceeds; the user is just told it's a fallback.
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "MLX engine"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(
        characterize=lambda m, *, progress=False, kv_quant="f16": {"model": m, "safe_context": 9000, "decode_context": None, "points": []},
        calibration_model_cached=lambda m: True,
        download_calibration_model=lambda m, *, progress=False: None,
        calibrate=lambda: {"calibrated": False, "overhead_gb": None, "wall_gb": None,
                           "calibration_error": "calibration unavailable for 'x': boom"},
    ))
    c, buf = make_console()
    assert cli.render_characterize(c, "org/M") == 0          # ramp still proceeds on the default
    out = buf.getvalue()
    assert "calibration skipped" in out and "boom" in out    # the failure is surfaced, with reason
    assert "conservative default" in out
    # nothing measured (overhead and wall both None) → no calibration row persisted
    assert cli.db.get_calibration(store, "mkey", "mlx") is None


def test_characterize_surfaces_measured_wall(make_console, store, monkeypatch):
    # Spec 2026-06-23-capability-pipeline: on first run characterize must SHOW the measured wall
    # (and safe budget) at the moment it's measured, not just persist it silently.
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "MLX engine"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(
        characterize=lambda m, *, progress=False, kv_quant="f16": {"model": m, "safe_context": 9000, "decode_context": None, "points": []},
        calibration_model_cached=lambda m: True,
        download_calibration_model=lambda m, *, progress=False: None,
        calibrate=lambda: {"overhead_gb": 1.7, "wall_gb": 17.2, "safe_budget_gb": 15.2},
    ))
    c, buf = make_console()
    assert cli.render_characterize(c, "org/M") == 0
    out = buf.getvalue()
    assert "measured wall" in out and "17.2 GB" in out
    assert "safe budget" in out and "15.2 GB" in out


def test_characterize_wall_line_without_budget(make_console, store, monkeypatch):
    # A wall with no safe budget: show the wall, but don't append a budget clause.
    # Spec 2026-06-23-capability-pipeline.
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "MLX engine"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(
        characterize=lambda m, *, progress=False, kv_quant="f16": {"model": m, "safe_context": 9000, "decode_context": None, "points": []},
        calibration_model_cached=lambda m: True,
        download_calibration_model=lambda m, *, progress=False: None,
        calibrate=lambda: {"overhead_gb": 1.7, "wall_gb": 17.2, "safe_budget_gb": None},
    ))
    c, buf = make_console()
    assert cli.render_characterize(c, "org/M") == 0
    out = buf.getvalue()
    assert "measured wall" in out and "17.2 GB" in out
    assert "safe budget" not in out


def test_characterize_omits_wall_line_when_wall_none(make_console, store, monkeypatch):
    # An engine that only measures cold-start overhead (wall_gb=None) must NOT print an empty/
    # misleading wall line. Spec 2026-06-23-capability-pipeline.
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "MLX engine"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(
        characterize=lambda m, *, progress=False, kv_quant="f16": {"model": m, "safe_context": 9000, "decode_context": None, "points": []},
        calibration_model_cached=lambda m: True,
        download_calibration_model=lambda m, *, progress=False: None,
        calibrate=lambda: {"overhead_gb": 1.7, "wall_gb": None, "safe_budget_gb": None},
    ))
    c, buf = make_console()
    assert cli.render_characterize(c, "org/M") == 0
    assert "measured wall" not in buf.getvalue()


def test_characterize_persists_measured_wall_when_overhead_none(make_console, store, monkeypatch):
    # CPU/CUDA read an EXACT wall, so calibrate returns overhead_gb=None. The measured wall must
    # still be stored — otherwise profile/recommend report a perpetual estimate on the very engines
    # `run` works on. Spec 2026-06-23-capability-pipeline.
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cpu")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "llama.cpp"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(
        characterize=lambda m, *, progress=False, kv_quant="f16": {"model": m, "safe_context": 9000, "decode_context": None, "points": []},
        calibration_model_cached=lambda m: True,
        download_calibration_model=lambda m, *, progress=False: None,
        calibrate=lambda: {"overhead_gb": None, "wall_gb": 30.0, "safe_budget_gb": 28.0},
    ))
    c, _ = make_console()
    assert cli.render_characterize(c, "org/M") == 0
    row = cli.db.get_calibration(store, "mkey", "cpu")
    assert row is not None                                    # persisted despite overhead None
    assert row["wall_gb"] == 30.0 and row["safe_budget_gb"] == 28.0


def test_characterize_skips_calibration_when_already_calibrated(make_console, store, monkeypatch):
    # Already calibrated → characterize does NOT recalibrate (idempotent).
    calls = []
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "MLX engine"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    cli.calibration.save_calibration(store, "mlx", fixed_overhead_gb=2.0)   # already calibrated
    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(
        characterize=lambda m, *, progress=False, kv_quant="f16": {"model": m, "safe_context": 5000, "decode_context": None, "points": []},
        calibration_model_cached=lambda m: True,
        download_calibration_model=lambda m, *, progress=False: None,
        calibrate=lambda: (calls.append("cal") or {"overhead_gb": 9.9}),
    ))
    c, _ = make_console()
    assert cli.render_characterize(c, "org/M") == 0
    assert calls == []                                                 # not recalibrated
    assert cli.db.get_calibration(store, "mkey", "mlx")["fixed_overhead_gb"] == 2.0   # unchanged


def test_render_characterize_no_ceiling(make_console, store, monkeypatch):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": None,
                                               "decode_context": None, "points": []})
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Big") == 0
    out = buf.getvalue()
    assert "couldn't fit" in out and "pre-quantized MLX model" in out
    assert "--weight-quant" not in out
    assert cli.db.get_characterization(store, "mkey", "mlx", "org/Big")["safe_context"] is None


def test_render_characterize_no_ceiling_explains_with_budget(make_console, store, monkeypatch):
    # When the driver surfaces base_gb/budget_gb for a null ceiling (#105), the message explains
    # why (base near budget) and suggests an engine-valid recovery — not the vague
    # "too big or borderline" or CUDA-only advice on MLX.
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": None,
                                               "decode_context": None, "points": [],
                                               "stopped_reason": "insufficient points",
                                               "base_gb": 6.68, "budget_gb": 7.0})
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Big") == 0
    out = buf.getvalue()
    assert "6.68" in out and "7.0" in out and "pre-quantized MLX model" in out
    assert "--weight-quant" not in out


@pytest.mark.parametrize(("backend", "expected"), [
    ("cuda", "--weight-quant int4 or int8"),
    ("cpu", "more heavily quantized GGUF model"),
])
def test_render_characterize_no_ceiling_gives_engine_valid_recovery(
        make_console, store, monkeypatch, backend, expected):
    _wire_characterize(
        monkeypatch, backend=backend,
        characterize=lambda m: {"model": m, "safe_context": None,
                                 "decode_context": None, "points": []})
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Big") == 0
    assert expected in buf.getvalue()


def test_render_characterize_no_ceiling_json_forwards_diagnostics(make_console, store, monkeypatch, capsys):
    # --json null path forwards the present diagnostic fields and SKIPS the ones that are None
    # (here stopped_reason is absent → must not appear), so automated callers get only real values.
    # The JSON payload goes to stdout via print() (read it from capsys), not the console buffer.
    import json as _j
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": None,
                                               "decode_context": None, "points": [],
                                               "base_gb": 6.68, "budget_gb": 7.0})  # no stopped_reason
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Big", as_json=True) == 0
    data = _j.loads(capsys.readouterr().out)
    assert data["safe_context"] is None
    assert data["base_gb"] == 6.68 and data["budget_gb"] == 7.0
    assert "stopped_reason" not in data        # a None field is skipped, not forwarded as null


def test_render_characterize_engine_not_installed(make_console, monkeypatch):
    _wire_characterize(monkeypatch, engine_ok=False)
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (False, "MLX engine"))
    c, buf = make_console()
    assert cli.render_characterize(c, "x") == 1
    assert buf.getvalue() == "  the MLX engine isn't installed — run: ara install --engine mlx\n"


def test_render_characterize_engine_not_installed_json_uses_complete_label(monkeypatch, capsys):
    _wire_characterize(monkeypatch, engine_ok=False)
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (False, "MLX engine"))
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(c, "x", as_json=True) == 1
    assert json.loads(capsys.readouterr().out) == {"error": "MLX engine not installed"}


def test_render_characterize_engine_error(make_console, monkeypatch):
    def boom(m):
        raise RuntimeError("OOM guard tripped")
    _wire_characterize(monkeypatch, characterize=boom)
    c, buf = make_console()
    assert cli.render_characterize(c, "x") == 1
    assert "characterization failed" in buf.getvalue()


def test_render_characterize_engine_exception_json(monkeypatch, capsys):
    # Rule #3 (Honesty): when the engine RAISES mid-characterize (refuse/abort/OOM-guard), --json
    # must emit {"error": ...}, not styled text — a --json consumer would choke on the styled
    # line. (Distinct from the returned-{"error":...} path tested below.)
    def boom(m):
        raise RuntimeError("OOM guard tripped")
    _wire_characterize(monkeypatch, characterize=boom)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(c, "x", as_json=True) == 1
    out = json.loads(capsys.readouterr().out)
    assert "error" in out and "characterization failed" in out["error"]


def test_render_characterize_json(monkeypatch, capsys, store):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 9000,
                                               "decode_context": None, "points": []})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(c, "org/M", as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["safe_context"] == 9000
    assert data["engine"] == "mlx"
    assert "decode_context" in data


def test_render_characterize_json_stdout_is_one_document_even_on_first_calibration(
        monkeypatch, capsys, store):
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "MLX engine"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(
        characterize=lambda m, *, progress=False, kv_quant="f16": {
            "model": m, "safe_context": 2048, "decode_context": 2048, "points": []},
        calibration_model_cached=lambda m: True,
        download_calibration_model=lambda m, *, progress=False: None,
        calibrate=lambda: {"calibration_error": "calibration unavailable: boom"},
    ))
    c = cli.Console(color=False, stream=sys.stdout)

    assert cli.render_characterize(c, "org/M", as_json=True) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "model": "org/M", "engine": "mlx", "safe_context": 2048, "decode_context": 2048,
        "config": {},
        "calibration_error": "calibration unavailable: boom",
        "calibration_fallback": True,
    }


@pytest.mark.parametrize(("characterize", "expected_error"), [
    (lambda m, **kw: (_ for _ in ()).throw(RuntimeError("ramp boom")),
     "characterization failed: ramp boom"),
    (lambda m, **kw: {"error": "load boom"}, "load boom"),
])
def test_render_characterize_json_error_carries_calibration_fallback(
        monkeypatch, capsys, store, characterize, expected_error):
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "MLX engine"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.calibration, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(
        characterize=characterize,
        calibration_model_cached=lambda m: True,
        download_calibration_model=lambda m, *, progress=False: None,
        calibrate=lambda: {"calibration_error": "calibration unavailable: boom"},
    ))
    c = cli.Console(color=False, stream=sys.stdout)

    assert cli.render_characterize(c, "org/M", as_json=True) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "error": expected_error,
        "calibration_error": "calibration unavailable: boom",
        "calibration_fallback": True,
    }


def test_render_characterize_engine_flag_overrides_detected_backend(make_console, store, monkeypatch):
    # winbox's case: a GPU is detected (cuda), but `--engine cpu` must run on the CPU backend
    # and store under the cpu engine key — never silently fall through to the detected GPU.
    seen = {}
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli, "engine_status",
                        lambda b=None: (seen.update(status_backend=b) or (True, "llama.cpp")))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)

    def fake_get_backend(b=None):
        seen["backend"] = b
        return types.SimpleNamespace(
            characterize=lambda m, *, progress=False, kv_quant="f16": {"model": m, "safe_context": 8192, "points": [[2000, 0.2]]},
            calibration_model_cached=lambda m: True,   # skip pre-fetch in this test
            download_calibration_model=lambda m, *, progress=False: None,
        )

    monkeypatch.setattr(cli, "get_backend", fake_get_backend)
    c, _buf = make_console()
    assert cli.render_characterize(c, "org/G", engine="cpu") == 0
    assert seen["backend"] == "cpu"          # ran on the CPU backend, not the detected cuda
    assert seen["status_backend"] == "cpu"   # install check targeted the CPU engine
    assert cli.db.get_characterization(store, "mkey", "cpu", "org/G")["safe_context"] == 8192


def test_render_characterize_unknown_engine_errors(make_console, monkeypatch):
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    c, buf = make_console()
    assert cli.render_characterize(c, "x", engine="bogus") == 1
    assert "unknown engine" in buf.getvalue().lower()


def _error_characterize(model, *, progress=False):
    # the driver shape when an engine couldn't even LOAD the model (preflight error)
    return {"model": model, "safe_context": None, "points": [], "error": "no transformers config"}


def test_render_characterize_skips_persist_on_engine_error(make_console, store, monkeypatch):
    # An engine that can't load the model returns `error` (not a measurement): don't persist a
    # misleading null row, and suggest a compatible engine when we can tell (a .gguf → cpu).
    _wire_characterize(monkeypatch, characterize=_error_characterize)   # default backend apple→mlx
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Model.gguf") == 1
    out = buf.getvalue()
    assert "couldn't load" in out
    assert "--engine cpu" in out                       # suggested the GGUF-capable engine
    assert cli.db.get_characterization(store, "mkey", "mlx", "org/Model.gguf") is None   # not stored


def test_render_characterize_engine_error_json(monkeypatch, capsys, store):
    _wire_characterize(monkeypatch, characterize=_error_characterize)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(c, "org/Model.gguf", as_json=True) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "no transformers config"


def test_render_characterize_engine_error_no_suggestion(make_console, store, monkeypatch):
    # A non-GGUF ref on the wrong engine: we can't cheaply tell which engine fits → no hint.
    _wire_characterize(monkeypatch, characterize=_error_characterize)
    c, buf = make_console()
    assert cli.render_characterize(c, "org/PlainModel") == 1
    out = buf.getvalue()
    assert "couldn't load" in out and "--engine" not in out


def test_main_characterize_dispatch(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["characterize", "org/Model"])
    assert rec["characterize"] == "org/Model"
    assert rec["characterize_engine"] is None    # no --engine → detected backend


def test_main_characterize_passes_engine(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    _run_main(monkeypatch, ["characterize", "org/Model", "--engine", "cpu"])
    assert rec["characterize"] == "org/Model"
    assert rec["characterize_engine"] == "cpu"


def test_main_characterize_no_model(monkeypatch, capsys):
    assert _run_main(monkeypatch, ["characterize"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Usage: ara characterize" in captured.err


# --------------------------------------------------------------------------- #
# render_characterize — pre-fetch block (task #47)
# --------------------------------------------------------------------------- #
def _wire_characterize_bk(monkeypatch, bk, *, backend="apple", engine_ok=True,
                          size_gb=4.0, free_gb=50.0):
    """Wire render_characterize with a FakeBackend and stubbed acquire functions."""
    monkeypatch.setattr(cli.detect, "backend_name", lambda: backend)
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (engine_ok, "ara-engine-mlx"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    monkeypatch.setattr(cli, "get_backend", lambda b=None: bk)
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: size_gb)
    monkeypatch.setattr(cli.acquire, "gguf_size_gb", lambda m: size_gb)
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: free_gb)


def _fake_bk_characterize(model, *, progress=False, kv_quant="f16"):
    return {"model": model, "safe_context": 16000, "decode_context": None, "points": [[1024, 1.2]]}


def test_render_characterize_prefetch_uncached_transformers(make_console, store, monkeypatch):
    # Uncached transformers model on a compatible engine → download fired, then characterize runs.
    bk = FakeBackend(_limits(), cached=False)
    bk.calibrate_result = None  # self-calibrate runs but returns None → nothing persisted
    # Give the FakeBackend a characterize method
    bk.characterize = _fake_bk_characterize
    _wire_characterize_bk(monkeypatch, bk)
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Model") == 0
    assert bk.downloaded == ["org/Model"]            # download was called
    assert "downloading" in buf.getvalue()            # status line emitted
    assert "16000" in buf.getvalue()                  # characterize result shown
    row = cli.db.get_characterization(store, "mkey", "mlx", "org/Model")
    assert row["safe_context"] == 16000               # result persisted


def test_render_characterize_cold_remote_gguf_downloads_then_pins(
        make_console, store, monkeypatch):
    state = {"artifact": None, "loaded": None}
    bk = FakeBackend(_limits(), cached=False)
    bk.calibrate_result = None

    def download(model, *, progress=False):
        bk.downloaded.append(model)
        state["artifact"] = "hf-gguf:org/model@" + "a" * 40 + ":model-q4.gguf:digest"

    def characterize(model, **_kwargs):
        state["loaded"] = model
        return {"safe_context": 4096, "decode_context": None, "points": []}

    bk.download_calibration_model = download
    bk.characterize = characterize
    _wire_characterize_bk(monkeypatch, bk, backend="cpu")
    monkeypatch.setattr(cli.staleness, "artifact_identity", lambda _model: state["artifact"])
    monkeypatch.setattr(cli.staleness, "pinned_model_ref",
                        lambda _model, artifact: "/cache/model-q4.gguf" if artifact else None)

    c, _ = make_console()
    assert cli.render_characterize(c, "org/model", engine="cpu") == 0
    assert bk.downloaded == ["org/model"]
    assert state["loaded"] == "/cache/model-q4.gguf"


def test_render_characterize_prefetch_json_stdout_is_pure(store, monkeypatch, capsys):
    bk = FakeBackend(_limits(), cached=False)
    bk.calibrate_result = None
    bk.characterize = _fake_bk_characterize
    _wire_characterize_bk(monkeypatch, bk)
    c = cli.Console.from_env()
    assert cli.render_characterize(c, "org/Model", as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["safe_context"] == 16000


def test_render_characterize_prefetch_already_cached(make_console, store, monkeypatch):
    # Already-cached model → download NOT called; characterize still runs.
    bk = FakeBackend(_limits(), cached=True)
    bk.characterize = _fake_bk_characterize
    _wire_characterize_bk(monkeypatch, bk)
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Model") == 0
    assert bk.downloaded == []                        # no download
    assert "16000" in buf.getvalue()


def test_render_characterize_prefetch_incompatible_engine(make_console, store, monkeypatch):
    # A .gguf model on the mlx (apple) engine: engine_for_model returns "cpu" != "mlx"
    # → incompatible=True → download NOT called; existing flow proceeds (engine error path).
    bk = FakeBackend(_limits(), cached=False)
    bk.characterize = _fake_bk_characterize
    _wire_characterize_bk(monkeypatch, bk)
    c, buf = make_console()
    # "org/model.gguf" → engine_for_model returns "cpu"; sel.engine_key is "mlx" → incompatible
    assert cli.render_characterize(c, "org/model.gguf") == 0
    assert bk.downloaded == []                        # download skipped (incompatible)


def test_render_characterize_prefetch_insufficient_disk(make_console, monkeypatch):
    # Not enough disk → error emitted, returns 1, characterize NOT called.
    bk = FakeBackend(_limits(), cached=False)
    bk.characterize = _fake_bk_characterize
    # size_gb=10, free_gb=5, DISK_BUFFER_GB=2 → 5 < 10+2 → shortfall
    _wire_characterize_bk(monkeypatch, bk, size_gb=10.0, free_gb=5.0)
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Big") == 1
    assert "not enough disk" in buf.getvalue()
    assert bk.downloaded == []                        # characterize never reached


def test_render_characterize_prefetch_insufficient_disk_json(monkeypatch, capsys):
    # --json variant of the insufficient-disk branch.
    bk = FakeBackend(_limits(), cached=False)
    bk.characterize = _fake_bk_characterize
    _wire_characterize_bk(monkeypatch, bk, size_gb=10.0, free_gb=5.0)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(c, "org/Big", as_json=True) == 1
    out = json.loads(capsys.readouterr().out)
    assert "error" in out and "disk" in out["error"]


# --------------------------------------------------------------------------- #
# render_characterize — download failure: honest, actionable error messages
# --------------------------------------------------------------------------- #
def _bk_download_raises(exc):
    """FakeBackend whose download_calibration_model raises *exc*."""
    bk = FakeBackend(_limits(), cached=False)
    bk.characterize = _fake_bk_characterize

    def _boom(model, *, progress=False):
        raise exc
    bk.download_calibration_model = _boom
    return bk


def _fake_response_stub(code):
    class FakeResp:
        headers = {}
        request = None
        status_code = code
    return FakeResp()


def test_render_characterize_gated_gives_actionable_message(make_console, monkeypatch):
    # A GatedRepoError must tell the user WHY and HOW to fix it — not swallow the reason.
    from huggingface_hub.errors import GatedRepoError
    exc = GatedRepoError("403 gated", response=_fake_response_stub(403))
    bk = _bk_download_raises(exc)
    _wire_characterize_bk(monkeypatch, bk)
    c, buf = make_console()
    assert cli.render_characterize(c, "org/gated-model") == 1
    out = buf.getvalue()
    assert "gated" in out
    assert "HF_TOKEN" in out or "terms" in out   # actionable: how to fix it


def test_render_characterize_gated_json(monkeypatch, capsys):
    from huggingface_hub.errors import GatedRepoError
    exc = GatedRepoError("403 gated", response=_fake_response_stub(403))
    bk = _bk_download_raises(exc)
    _wire_characterize_bk(monkeypatch, bk)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(c, "org/gated-model", as_json=True) == 1
    out = json.loads(capsys.readouterr().out)
    assert "error" in out and "gated" in out["error"]


def test_render_characterize_not_found_gives_actionable_message(make_console, monkeypatch):
    from huggingface_hub.errors import RepositoryNotFoundError
    exc = RepositoryNotFoundError("404", response=_fake_response_stub(404))
    bk = _bk_download_raises(exc)
    _wire_characterize_bk(monkeypatch, bk)
    c, buf = make_console()
    assert cli.render_characterize(c, "org/missing") == 1
    out = buf.getvalue()
    # "not found" or "don't have access" — either phrasing is honest
    assert "not found" in out or "access" in out


def test_render_characterize_offline_gives_actionable_message(make_console, monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError
    exc = LocalEntryNotFoundError("not cached")
    bk = _bk_download_raises(exc)
    _wire_characterize_bk(monkeypatch, bk)
    c, buf = make_console()
    assert cli.render_characterize(c, "org/model") == 1
    out = buf.getvalue()
    assert "offline" in out or "connection" in out or "cached" in out


def test_render_characterize_download_error_json(monkeypatch, capsys):
    # Any download failure emits {"error": ...} in --json mode (not a traceback).
    exc = ConnectionError("no route")
    bk = _bk_download_raises(exc)
    _wire_characterize_bk(monkeypatch, bk)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(c, "org/model", as_json=True) == 1
    out = json.loads(capsys.readouterr().out)
    assert "error" in out


def test_render_characterize_auth_error_mentions_token(make_console, monkeypatch):
    # 401 HfHubHTTPError → auth reason → message mentions HF_TOKEN.
    from huggingface_hub.errors import HfHubHTTPError
    exc = HfHubHTTPError("unauthorized", response=_fake_response_stub(401))
    bk = _bk_download_raises(exc)
    _wire_characterize_bk(monkeypatch, bk)
    c, buf = make_console()
    assert cli.render_characterize(c, "org/private") == 1
    assert "HF_TOKEN" in buf.getvalue()


def test_render_characterize_unknown_download_error(make_console, monkeypatch):
    # A totally unknown exception (not a HF type) → "unknown error" fallback message.
    exc = RuntimeError("some strange error")
    bk = _bk_download_raises(exc)
    _wire_characterize_bk(monkeypatch, bk)
    c, buf = make_console()
    assert cli.render_characterize(c, "org/model") == 1
    assert "couldn't fetch" in buf.getvalue()


# --------------------------------------------------------------------------- #
# render_characterize — progress flag threading (2026-06-24-download-progress)
# --------------------------------------------------------------------------- #
def _wire_characterize_progress(monkeypatch, *, backend="apple", cached=True):
    """Wire render_characterize with a backend that captures progress kwarg on both calls.
    Returns the capture dict: {download_progress: bool|None, characterize_progress: bool|None}.
    """
    captured = {}

    def _download(model, *, progress=False):
        captured["download_progress"] = progress

    def _characterize(model, *, progress=False, kv_quant="f16"):
        captured["characterize_progress"] = progress
        return {"model": model, "safe_context": 8000, "decode_context": None, "points": []}

    monkeypatch.setattr(cli.detect, "backend_name", lambda: backend)
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "MLX engine"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)
    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(
        characterize=_characterize,
        calibration_model_cached=lambda m: cached,
        download_calibration_model=_download,
    ))
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: 1.0)
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: 50.0)
    return captured


def test_render_characterize_progress_true_when_tty_and_not_json(make_console, store, monkeypatch):
    """progress=True when stderr is a TTY and --json is not set.

    Slug: 2026-06-24-download-progress
    """
    captured = _wire_characterize_progress(monkeypatch, cached=False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    c, _ = make_console()
    assert cli.render_characterize(c, "org/M") == 0
    assert captured["download_progress"] is True
    assert captured["characterize_progress"] is True


def test_render_characterize_progress_false_when_not_tty(make_console, store, monkeypatch):
    """progress=False when stderr is not a TTY (piped/CI).

    Slug: 2026-06-24-download-progress
    """
    captured = _wire_characterize_progress(monkeypatch, cached=False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    c, _ = make_console()
    assert cli.render_characterize(c, "org/M") == 0
    assert captured["download_progress"] is False
    assert captured["characterize_progress"] is False


def test_render_characterize_progress_false_when_as_json(make_console, store, monkeypatch,
                                                          capsys):
    """progress=False when --json is set, even if stderr is a TTY.

    Slug: 2026-06-24-download-progress
    """
    captured = _wire_characterize_progress(monkeypatch, cached=False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(c, "org/M", as_json=True) == 0
    capsys.readouterr()  # consume stdout
    assert captured["download_progress"] is False
    assert captured["characterize_progress"] is False


def test_render_characterize_progress_not_passed_to_download_when_already_cached(
        make_console, store, monkeypatch):
    """When model is already cached, download_calibration_model is not called at all.
    characterize still receives the correct progress value.

    Slug: 2026-06-24-download-progress
    """
    captured = _wire_characterize_progress(monkeypatch, cached=True)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    c, _ = make_console()
    assert cli.render_characterize(c, "org/M") == 0
    assert "download_progress" not in captured   # download never called
    assert captured["characterize_progress"] is True


# --------------------------------------------------------------------------- #
# decode-safe ceiling display (task #48b)
# --------------------------------------------------------------------------- #
def test_render_characterize_decode_shown_when_greater(make_console, store, monkeypatch):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 20000,
                                               "decode_context": 25000, "points": []})
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Model") == 0
    out = buf.getvalue()
    assert "25000" in out and "decode" in out and "est." in out


def test_render_characterize_decode_hidden_when_none(make_console, store, monkeypatch):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 20000,
                                               "decode_context": None, "points": []})
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Model") == 0
    assert "decode" not in buf.getvalue()


def test_render_characterize_decode_hidden_when_not_greater(make_console, store, monkeypatch):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 20000,
                                               "decode_context": 15000, "points": []})
    c, buf = make_console()
    assert cli.render_characterize(c, "org/Model") == 0
    assert "decode" not in buf.getvalue()


def test_render_characterize_decode_persisted(store, monkeypatch):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 20000,
                                               "decode_context": 25000, "points": []})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(c, "org/Model") == 0
    row = cli.db.get_characterization(store, "mkey", "mlx", "org/Model")
    assert row["decode_context"] == 25000


def test_render_characterize_json_includes_decode_context(monkeypatch, capsys, store):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 9000,
                                               "decode_context": 12000, "points": []})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(c, "org/M", as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert "decode_context" in data and data["decode_context"] == 12000


def test_render_characterize_json_decode_context_none(monkeypatch, capsys, store):
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 9000,
                                               "decode_context": None, "points": []})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_characterize(c, "org/M", as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert "decode_context" in data and data["decode_context"] is None


def test_render_models_decode_gloss_when_greater(make_console, store, monkeypatch):
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/A", "modality": "text"}])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, e: [{"model_id": "org/A", "safe_context": 16000,
                                             "decode_context": 20000}])
    c, buf = make_console()
    cli.render_models(c)
    out = buf.getvalue()
    assert "20000" in out and "stream-only" in out


def test_render_models_decode_hidden_when_not_greater(make_console, store, monkeypatch):
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/A", "modality": "text"}])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, e: [{"model_id": "org/A", "safe_context": 16000,
                                             "decode_context": 8000}])
    c, buf = make_console()
    cli.render_models(c)
    assert "decode" not in buf.getvalue()


def test_render_models_json_includes_decode_context(monkeypatch, capsys, store):
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/A", "modality": "text"}])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, e: [{"model_id": "org/A", "safe_context": 16000,
                                             "decode_context": 20000}])
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_models(c, as_json=True)
    data = json.loads(capsys.readouterr().out)
    assert data[0].get("decode_context") == 20000


def test_render_models_marks_missing_or_mismatched_artifact_authority_stale(
        make_console, monkeypatch, capsys, store):
    monkeypatch.setattr(cli.catalog, "scan", lambda con: 0)
    monkeypatch.setattr(cli.catalog, "all_models",
                        lambda con: [{"model_id": "org/A", "modality": "text"}])
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cpu")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "list_characterizations",
                        lambda con, mk, engine: [{
                            "model_id": "org/A", "safe_context": 16000,
                            "decode_context": None, "config": {},
                            "artifact_id": None,
                        }] if engine == "cpu" else [])

    c, buf = make_console()
    cli.render_models(c)
    assert "stale — re-characterize" in buf.getvalue()

    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_models(c, as_json=True)
    assert json.loads(capsys.readouterr().out)[0]["stale_ceiling"] is True


def test_render_models_merges_present_durable_gguf_selector(
        monkeypatch, capsys, store):
    selector = "org/repo:Model-Q4_K_M.gguf"
    cli.db.upsert_model(store, selector, modality="text", quant="q4_k_m",
                        n_layers=12, hidden_size=768, kv_heads=4, head_dim=64,
                        max_context=4096, weights_gb=1.0)
    cli.db.save_characterization(
        store, "mkey", "cpu", selector, safe_context=3000, points=[],
        config={}, artifact_id="artifact:test")

    def scan(con):
        cli.db.upsert_model(con, "org/repo", modality="text", n_layers=12,
                            hidden_size=768, kv_heads=4, head_dim=64,
                            max_context=4096, weights_gb=2.0)
        return 1

    monkeypatch.setattr(cli.catalog, "scan", scan)
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cpu")
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_models(c, as_json=True)
    by_id = {row["model_id"]: row for row in json.loads(capsys.readouterr().out)}
    assert by_id[selector]["safe_context"] == 3000
    assert by_id[selector]["characterized"] is True
    assert by_id[selector]["stale_ceiling"] is False


def test_model_detail_per_engine_decode_gloss(make_console, monkeypatch):
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: {"safe_context": 16000, "decode_context": 20000})
    c, buf = make_console()
    assert cli.render_model_detail(c, "org/Smol") == 0
    out = buf.getvalue()
    assert "20000" in out and "stream-only" in out


def test_model_detail_per_engine_decode_hidden_when_not_greater(make_console, monkeypatch):
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: {"safe_context": 16000, "decode_context": 8000})
    c, buf = make_console()
    assert cli.render_model_detail(c, "org/Smol") == 0
    assert "decode" not in buf.getvalue()


def test_model_detail_json_has_decode_context(monkeypatch, capsys):
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: {"safe_context": 9000, "decode_context": 12000})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_model_detail(c, "org/A", as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert data.get("decode_context") == 12000
    assert all(isinstance(v, int) for v in data["engines"].values())


def test_model_detail_json_decode_context_paired_with_best_safe_engine(monkeypatch, capsys):
    # M1: top-level decode_context must come from the same engine that has the highest
    # safe_context — NOT a global max across engines. Here cuda has safe_context=16000/decode=18000
    # and cpu has safe_context=8000/decode=25000. Top-level decode_context must be 18000, not 25000.
    monkeypatch.setattr(cli.catalog, "describe", lambda mid: _meta())
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    _per_engine = {"cuda": {"safe_context": 16000, "decode_context": 18000},
                   "cpu": {"safe_context": 8000, "decode_context": 25000}}
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, mid: _per_engine.get(e))
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_model_detail(c, "org/A", as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["safe_context"] == 16000
    assert data["decode_context"] == 18000   # paired with cuda (best safe), not cpu's 25000


def test_emit_characterized_decode_gloss_when_greater(make_console, store, monkeypatch):
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    cli.db.save_characterization(store, "mkey", "cuda", "org/Model",
                                 safe_context=16000, points=[], decode_context=20000)
    c, buf = make_console()
    cli._emit_characterized(c, "cuda")
    out = buf.getvalue()
    assert "20000" in out and "stream-only" in out


def test_emit_characterized_decode_hidden_when_not_greater(make_console, store, monkeypatch):
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    cli.db.save_characterization(store, "mkey", "cuda", "org/Model",
                                 safe_context=16000, points=[], decode_context=8000)
    c, buf = make_console()
    cli._emit_characterized(c, "cuda")
    assert "decode" not in buf.getvalue()


# =========================================================================== #
# Task 7: hardware verbose detail blocks + JSON nesting
# =========================================================================== #

def _apple_cpu() -> CpuInfo:
    """Apple M4 Pro–shaped CpuInfo: no clock/L3; L1+L2 only; no vendor features."""
    return CpuInfo(
        brand="Apple M4 Pro", vendor="Apple", arch_id="arm64",
        physical=12, logical=12,
        base_mhz=None, max_mhz=None,
        l1_kb=192, l2_kb=4096, l3_kb=None,
        features=[],
    )


def _windows_cpu() -> CpuInfo:
    """Ryzen 9 5900X–shaped CpuInfo: full clocks + L2/L3; no features (WMI gap)."""
    return CpuInfo(
        brand="AMD Ryzen 9 5900X 12-Core Processor", vendor="AuthenticAMD", arch_id="AMD64",
        physical=12, logical=24,
        base_mhz=None, max_mhz=3701,
        l1_kb=None, l2_kb=6144, l3_kb=65536,
        features=[],
    )


def _windows_memory() -> MemoryInfo:
    """winbox-shaped MemoryInfo: 4×DDR4 8 GB DIMMs, 4/4 slots used."""
    modules = [
        MemoryModule(slot=f"DIMM_A{i}", capacity_gb=8.0, speed_mts=3400,
                     manufacturer="G-Skill", part_number="F4-3200C14-8GFX")
        for i in range(1, 5)
    ]
    return MemoryInfo(
        total_gb=32.0, available_gb=20.0, swap_gb=0.0,
        kind="DDR4", speed_mts=3400,
        slots_used=4, slots_total=4,
        modules=modules,
    )


def _apple_memory() -> MemoryInfo:
    """Apple-shaped MemoryInfo: kind/speed present but no slots/modules (soldered)."""
    return MemoryInfo(
        total_gb=24.0, available_gb=18.0, swap_gb=0.0,
        kind="LPDDR5", speed_mts=None,
        slots_used=None, slots_total=None,
        modules=[],
    )


def _windows_storage() -> StorageInfo:
    """winbox-shaped StorageInfo: NVMe SSD + HDD."""
    return StorageInfo(
        free_gb=500.0,
        drives=[
            Drive(model="Generic NVMe SSD 1TB", media="nvme-ssd", size_gb=1000.2),
            Drive(model="Generic SATA HDD 2TB", media="hdd", size_gb=2000.4),
        ],
    )


def _apple_storage() -> StorageInfo:
    """Apple-shaped StorageInfo: one NVMe drive."""
    return StorageInfo(
        free_gb=200.0,
        drives=[Drive(model="APPLE SSD AP0512Z", media="nvme-ssd", size_gb=500.3)],
    )


def _windows_board() -> BoardInfo:
    """winbox-shaped BoardInfo: ASUS ROG STRIX; system_* → None (custom PC)."""
    return BoardInfo(
        board_vendor="ASUSTeK COMPUTER INC.", board_model="ROG STRIX X470-F GAMING",
        bios_version="6042", bios_date="2022-04-28",
        system_vendor=None, system_model=None,
    )


def _apple_board() -> BoardInfo:
    """Mac-shaped BoardInfo: system_vendor/model from SPHardwareDataType; board_* → None."""
    return BoardInfo(
        board_vendor=None, board_model=None,
        bios_version="13822.81.10", bios_date=None,
        system_vendor="Apple", system_model="MacBook Pro",
    )


# --------------------------------------------------------------------------- #
# verbose CPU block
# --------------------------------------------------------------------------- #

def test_verbose_cpu_detail_apple_silicon(make_console, monkeypatch, stub_pythons):
    """Apple Silicon: vendor shown, no clock/L3; L1+L2 shown; features empty → skipped."""
    stub_pythons(count=1)
    m = _machine(cpu=_apple_cpu(), memory=_apple_memory(), storage=_apple_storage(),
                 board=_apple_board())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "vendor" in out and "Apple" in out
    assert "threads" in out and "12" in out
    assert "L1" in out and "L2" in out
    assert "L3" not in out             # Apple Silicon has no L3
    assert "clocks" not in out         # no clock data on Apple Silicon
    assert "features" not in out       # empty features list → line skipped


def test_verbose_cpu_detail_windows_ryzen(make_console, monkeypatch, stub_pythons):
    """Ryzen: max clock shown; L2+L3 shown; no features (WMI gap) → skipped."""
    stub_pythons(count=1)
    m = _machine(cpu=_windows_cpu(), memory=_windows_memory(), storage=_windows_storage(),
                 board=_windows_board(), chip="AMD Ryzen 9 5900X 12-Core Processor",
                 cpu_physical=12, cpu_logical=24,
                 accel=Accelerator("none", "none detected", None, None),
                 backend="cpu", engine="llama.cpp", engine_ready=False)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "AuthenticAMD" in out        # vendor
    assert "24" in out                  # logical threads
    assert "max 3701 MHz" in out        # max clock
    assert "L2 6144 KB" in out
    assert "L3 65536 KB" in out
    assert "features" not in out        # empty features → skipped


def test_verbose_cpu_detail_with_features(make_console, monkeypatch, stub_pythons):
    """CPU with features list → features line rendered."""
    stub_pythons(count=1)
    cpu = CpuInfo(brand="Intel Core i9", vendor="GenuineIntel", logical=16,
                  features=["AVX-512", "AVX2"])
    m = _machine(cpu=cpu)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "features" in out and "AVX-512" in out and "AVX2" in out


def test_verbose_cpu_detail_base_clock_only(make_console, monkeypatch, stub_pythons):
    """Base clock with no max → shows 'base N MHz' only."""
    stub_pythons(count=1)
    cpu = CpuInfo(brand="FakeCPU", base_mhz=2400)
    m = _machine(cpu=cpu)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "base 2400 MHz" in out
    assert "max" not in out.split("clocks")[-1].split("\n")[0]


def test_verbose_cpu_all_none_no_detail_lines(make_console, monkeypatch, stub_pythons):
    """All CpuInfo fields None → no vendor/threads/clocks/L1/L2/L3/features lines."""
    stub_pythons(count=1)
    m = _machine(cpu=CpuInfo())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    # None of the optional sub-fields should appear (use specific indented labels)
    assert "  vendor" not in out
    assert "  threads" not in out
    assert "  clocks" not in out
    assert "  cache" not in out
    assert "  features" not in out


def test_non_verbose_no_cpu_detail(make_console, monkeypatch, stub_pythons):
    """Non-verbose: CPU detail block not shown."""
    stub_pythons(count=1)
    m = _machine(cpu=_windows_cpu())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=False)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "AuthenticAMD" not in out
    assert "max 3701 MHz" not in out


# --------------------------------------------------------------------------- #
# verbose MEMORY detail block
# --------------------------------------------------------------------------- #

def test_verbose_memory_detail_windows_4modules(make_console, monkeypatch, stub_pythons):
    """Windows 4-DIMM system: kind, speed, slots 4/4, 4 module rows."""
    stub_pythons(count=1)
    m = _machine(memory=_windows_memory(), cpu=_windows_cpu(), storage=_windows_storage(),
                 board=_windows_board())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "DDR4" in out
    assert "3400 MT/s" in out
    assert "4 / 4 used" in out
    assert out.count("module") == 4      # one row per DIMM
    assert "G-Skill" in out
    assert "F4-3200C14-8GFX" in out


def test_verbose_memory_detail_apple_no_modules(make_console, monkeypatch, stub_pythons):
    """Apple Silicon: kind shown, no slots, no modules → '(not reported)' line."""
    stub_pythons(count=1)
    m = _machine(memory=_apple_memory(), cpu=_apple_cpu(), storage=_apple_storage(),
                 board=_apple_board())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "LPDDR5" in out
    assert "not reported on this system" in out
    assert "slots" not in out            # no slot count (soldered; slots_used/total both None)


def test_verbose_memory_kind_none_skipped(make_console, monkeypatch, stub_pythons):
    """kind=None → kind line not rendered."""
    stub_pythons(count=1)
    mem = MemoryInfo(total_gb=16.0, kind=None, modules=[])
    m = _machine(memory=mem)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    assert "kind" not in buf.getvalue()


def test_verbose_memory_partial_slots(make_console, monkeypatch, stub_pythons):
    """slots_used=2 but slots_total=None → renders '2 / ? used'."""
    stub_pythons(count=1)
    mem = MemoryInfo(total_gb=16.0, slots_used=2, slots_total=None, modules=[
        MemoryModule(slot="A1", capacity_gb=8.0, speed_mts=3200),
        MemoryModule(slot="A2", capacity_gb=8.0, speed_mts=3200),
    ])
    m = _machine(memory=mem)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "2 / ? used" in out


def test_non_verbose_no_memory_detail(make_console, monkeypatch, stub_pythons):
    """Non-verbose: memory detail block not shown."""
    stub_pythons(count=1)
    m = _machine(memory=_windows_memory())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=False)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "DDR4" not in out
    assert "DIMM" not in out


# --------------------------------------------------------------------------- #
# verbose STORAGE detail block
# --------------------------------------------------------------------------- #

def test_verbose_storage_detail_drives(make_console, monkeypatch, stub_pythons):
    """Storage drives listed in verbose mode."""
    stub_pythons(count=1)
    m = _machine(storage=_windows_storage(), cpu=_windows_cpu(), memory=_windows_memory(),
                 board=_windows_board())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "Generic NVMe SSD 1TB" in out
    assert "nvme-ssd" in out
    assert "1000 GB" in out
    assert "Generic SATA HDD 2TB" in out
    assert "hdd" in out


def test_verbose_storage_drive_none_fields_skipped(make_console, monkeypatch, stub_pythons):
    """Drive with model=None, media=None: only size shown."""
    stub_pythons(count=1)
    storage = StorageInfo(free_gb=100.0, drives=[Drive(model=None, media=None, size_gb=500.0)])
    m = _machine(storage=storage)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "drive" in out
    assert "500 GB" in out


def test_verbose_storage_no_drives(make_console, monkeypatch, stub_pythons):
    """No drives in StorageInfo → storage verbose block is silent (no drive rows)."""
    stub_pythons(count=1)
    storage = StorageInfo(free_gb=200.0, drives=[])
    m = _machine(storage=storage)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    assert "  drive" not in buf.getvalue()


def test_non_verbose_no_storage_detail(make_console, monkeypatch, stub_pythons):
    """Non-verbose: drives not listed."""
    stub_pythons(count=1)
    m = _machine(storage=_windows_storage())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=False)
    cli.render_detect(c)
    assert "Generic NVMe SSD" not in buf.getvalue()   # drive models hidden unless --verbose


# --------------------------------------------------------------------------- #
# verbose BOARD section
# --------------------------------------------------------------------------- #

def test_verbose_board_windows(make_console, monkeypatch, stub_pythons):
    """Windows board: vendor, model, BIOS; system_* None → those lines skipped."""
    stub_pythons(count=1)
    m = _machine(board=_windows_board(), cpu=_windows_cpu(), memory=_windows_memory(),
                 storage=_windows_storage())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "BOARD" in out
    assert "ASUSTeK COMPUTER INC." in out
    assert "ROG STRIX X470-F GAMING" in out
    assert "board vendorASUSTeK" not in out  # label must not butt against value (pad width)
    assert "6042" in out               # bios version
    assert "2022-04-28" in out         # bios date
    assert "system vendor" not in out  # None → skipped
    assert "system model" not in out   # None → skipped


def test_verbose_board_apple(make_console, monkeypatch, stub_pythons):
    """Mac board: system_vendor/model shown; board_* None → those lines skipped."""
    stub_pythons(count=1)
    m = _machine(board=_apple_board(), cpu=_apple_cpu(), memory=_apple_memory(),
                 storage=_apple_storage())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "BOARD" in out
    assert "13822.81.10" in out        # bios version (firmware)
    assert "system vendor Apple" in out  # padded label + value (no run-together)
    assert "MacBook Pro" in out        # system model
    assert "board vendor" not in out   # None → skipped
    assert "board model" not in out    # None → skipped


def test_board_all_none_no_section(make_console, monkeypatch, stub_pythons):
    """BoardInfo with all fields None → BOARD section not rendered at all."""
    stub_pythons(count=1)
    m = _machine(board=BoardInfo())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    assert "BOARD" not in buf.getvalue()


def test_non_verbose_no_board_section(make_console, monkeypatch, stub_pythons):
    """Non-verbose: BOARD section never shown even with board data."""
    stub_pythons(count=1)
    m = _machine(board=_windows_board())
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=False)
    cli.render_detect(c)
    assert "BOARD" not in buf.getvalue()


# --------------------------------------------------------------------------- #
# JSON: nested hardware structures present + all existing keys intact
# --------------------------------------------------------------------------- #

def test_render_detect_json_has_cpu_nested(monkeypatch, capsys):
    """--json includes cpu as a nested dict with all CpuInfo fields."""
    monkeypatch.setattr(cli.detect, "machine", lambda: _machine(cpu=_apple_cpu()))
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_detect(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert "cpu" in payload
    assert payload["cpu"]["brand"] == "Apple M4 Pro"
    assert payload["cpu"]["vendor"] == "Apple"
    assert payload["cpu"]["l3_kb"] is None
    assert payload["cpu"]["features"] == []


def test_render_detect_json_has_memory_nested(monkeypatch, capsys):
    """--json includes memory as a nested dict with modules list."""
    monkeypatch.setattr(cli.detect, "machine", lambda: _machine(memory=_windows_memory()))
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_detect(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert "memory" in payload
    assert payload["memory"]["kind"] == "DDR4"
    assert payload["memory"]["slots_used"] == 4
    mods = payload["memory"]["modules"]
    assert len(mods) == 4
    assert mods[0]["manufacturer"] == "G-Skill"


def test_render_detect_json_has_storage_nested(monkeypatch, capsys):
    """--json includes storage as a nested dict with drives list."""
    monkeypatch.setattr(cli.detect, "machine", lambda: _machine(storage=_windows_storage()))
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_detect(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert "storage" in payload
    drives = payload["storage"]["drives"]
    assert len(drives) == 2
    assert drives[0]["model"] == "Generic NVMe SSD 1TB"
    assert drives[0]["media"] == "nvme-ssd"


def test_render_detect_json_has_board_nested(monkeypatch, capsys):
    """--json includes board as a nested dict."""
    monkeypatch.setattr(cli.detect, "machine", lambda: _machine(board=_windows_board()))
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_detect(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert "board" in payload
    assert payload["board"]["board_vendor"] == "ASUSTeK COMPUTER INC."
    assert payload["board"]["bios_version"] == "6042"
    assert payload["board"]["system_vendor"] is None


def test_render_detect_json_existing_keys_intact(monkeypatch, capsys):
    """--json still has the existing flat keys (chip, backend, accel, etc.)."""
    monkeypatch.setattr(cli.detect, "machine", lambda: _machine(cpu=_apple_cpu()))
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_detect(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    # All pre-existing keys must survive
    for key in ("chip", "backend", "accelerated", "accel", "ram_total_gb",
                "disk_free_gb", "os_version", "arch"):
        assert key in payload, f"missing key: {key}"
    assert payload["chip"] == "Apple M4 Pro"
    assert payload["accelerated"] is True


def test_render_detect_json_apple_memory_no_modules(monkeypatch, capsys):
    """Apple-shaped memory: modules=[] in JSON, no slots info."""
    monkeypatch.setattr(cli.detect, "machine", lambda: _machine(memory=_apple_memory()))
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_detect(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["memory"]["modules"] == []
    assert payload["memory"]["slots_used"] is None
    assert payload["memory"]["kind"] == "LPDDR5"


# --------------------------------------------------------------------------- #
# branch corners: None-field skipping inside loops
# --------------------------------------------------------------------------- #

def test_verbose_memory_module_with_some_none_fields(make_console, monkeypatch, stub_pythons):
    """MemoryModule with mixed None/non-None fields covers all field-skip branches."""
    stub_pythons(count=1)
    # mod1: slot+manufacturer+part_number=None, capacity+speed present
    mod1 = MemoryModule(slot=None, capacity_gb=16.0, speed_mts=4800,
                        manufacturer=None, part_number=None)
    # mod2: capacity_gb=None and speed_mts=None → those append branches skipped
    mod2 = MemoryModule(slot="B1", capacity_gb=None, speed_mts=None,
                        manufacturer="Kingston", part_number="KVR32N22D8/16")
    mem = MemoryInfo(total_gb=32.0, slots_used=2, slots_total=4, modules=[mod1, mod2])
    m = _machine(memory=mem)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert out.count("module") == 2    # both modules rendered
    assert "16 GB" in out and "4800 MT/s" in out
    assert "Kingston" in out and "B1" in out
    # None fields must be absent
    assert "DIMM" not in out
    assert "G-Skill" not in out


def test_verbose_storage_drive_all_none_fields_no_row(make_console, monkeypatch, stub_pythons):
    """Drive with all fields None → parts is empty → no row emitted."""
    stub_pythons(count=1)
    storage = StorageInfo(free_gb=100.0, drives=[Drive(model=None, media=None, size_gb=None)])
    m = _machine(storage=storage)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    # The drive exists but all fields None → no "drive" row
    assert "  drive" not in buf.getvalue()


def test_verbose_board_bios_version_none_bios_date_present(make_console, monkeypatch, stub_pythons):
    """Board with bios_version=None but bios_date present → date shown, version line skipped."""
    stub_pythons(count=1)
    board = BoardInfo(board_vendor="ACME", bios_version=None, bios_date="2023-01-01")
    m = _machine(board=board)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "BOARD" in out
    assert "bios date" in out and "2023-01-01" in out
    assert "bios  " not in out   # bios version line skipped (only "bios date" present)


def test_verbose_memory_module_all_none_then_real(make_console, monkeypatch, stub_pythons):
    """First module all-None (parts empty → skipped), second module has data → row shown.
    Covers the 'if parts' False-branch with more iterations remaining in the loop."""
    stub_pythons(count=1)
    empty_mod = MemoryModule(slot=None, capacity_gb=None, speed_mts=None,
                             manufacturer=None, part_number=None)
    real_mod = MemoryModule(slot="A2", capacity_gb=8.0, speed_mts=3200, manufacturer=None,
                            part_number=None)
    mem = MemoryInfo(total_gb=8.0, slots_used=1, slots_total=2, modules=[empty_mod, real_mod])
    m = _machine(memory=mem)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    # Only one module row (the real one)
    assert out.count("  module") == 1
    assert "A2" in out and "8 GB" in out


def test_verbose_memory_no_modules_but_slots_known_no_not_reported(
        make_console, monkeypatch, stub_pythons):
    """modules=[] but slots_used is not None → elif is False, 'not reported' NOT printed."""
    stub_pythons(count=1)
    # Rare case: slot count known but no per-module data (e.g. Linux non-root dmidecode)
    mem = MemoryInfo(total_gb=32.0, slots_used=2, slots_total=4, modules=[])
    m = _machine(memory=mem)
    monkeypatch.setattr(cli.detect, "machine", lambda: m)
    c, buf = make_console(verbose=True)
    cli.render_detect(c)
    out = buf.getvalue()
    assert "2 / 4 used" in out
    assert "not reported on this system" not in out
    assert "module" not in out


# --------------------------------------------------------------------------- #
# Task 7: _det_accelerator — present-vs-usable GPU rendering
# --------------------------------------------------------------------------- #
from ara.hardware import GpuInfo  # noqa: E402 — grouped with hardware imports above


def _fake_machine(**over):
    """Thin wrapper over _machine() for accelerator tests that need gpus=."""
    return _machine(**over)


def test_accelerator_amd_vulkan_present_usable(make_console):
    # accel.kind none, but a Vulkan-usable AMD iGPU present → reported as usable (the `vulkan`
    # engine ships now: `ara ... --engine vulkan`), no engine-coming hint.
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(accel=detect.Accelerator("none", "none detected", None, None),
                      gpus=[GpuInfo(vendor="amd", name="AMD Radeon 780M",
                            vram_gb=4.0, integrated=True,
                            compute_runtime="Vulkan 1.4.318 · radv · coopmat",
                            usable_backend="vulkan")])
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "AMD Radeon 780M" in s
    assert "Vulkan 1.4.318 · radv · coopmat" in s
    assert "usable" in s
    assert "engine coming" not in s
    assert "no GPU detected" not in s


def test_accelerator_detected_backend_without_shipped_engine_shows_coming(make_console):
    # The forward-looking seam: a usable_backend that detection knows but ARA doesn't yet run
    # (not in _ARA_ENGINE_BACKENDS) renders the honest "engine coming (not yet runnable)" hint.
    # `vulkan` graduated out of this branch; the next backend (e.g. a discrete-Radeon ROCm lane)
    # would land here until its engine ships.
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(accel=detect.Accelerator("none", "none detected", None, None),
                      gpus=[GpuInfo(vendor="amd", name="Radeon RX 7900 XTX",
                            vram_gb=24.0, integrated=False,
                            compute_runtime="ROCm 6.1",
                            usable_backend="rocm")])
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "Radeon RX 7900 XTX" in s
    assert "engine coming (not yet runnable)" in s


def test_accelerator_empty_still_says_none(make_console):
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(accel=detect.Accelerator("none", "none detected", None, None), gpus=[])
    cli._det_accelerator(c, m)
    assert "no GPU detected" in out.getvalue()


def test_accelerator_nvidia_rich_block_preserved(make_console):
    # existing NVIDIA rich rendering must be unchanged
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(accel=detect.Accelerator("nvidia", "RTX 2070", 8.0, "CUDA",
                      compute="7.5", cuda_version="13.1", driver_version="591"), gpus=[])
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "RTX 2070" in s and "CUDA 13.1" in s


def test_accelerator_rocm_noted_not_usable(make_console):
    # usable_backend=None, but compute_runtime set (ROCm noted) → "not ARA's path" hint
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(accel=detect.Accelerator("none", "none detected", None, None),
                      gpus=[GpuInfo(vendor="amd", name="RX 6700 XT",
                            vram_gb=12.0, integrated=False,
                            compute_runtime="ROCm 6.1",
                            usable_backend=None)])
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "RX 6700 XT" in s
    assert "not ARA's path" in s
    assert "engine coming" not in s


def test_accelerator_no_runtime_detected(make_console):
    # usable_backend=None, compute_runtime=None → "no usable GPU runtime detected"
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(accel=detect.Accelerator("none", "none detected", None, None),
                      gpus=[GpuInfo(vendor="intel", name="Intel UHD 770",
                            vram_gb=None, integrated=True,
                            compute_runtime=None,
                            usable_backend=None)])
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "Intel UHD 770" in s
    assert "no usable GPU runtime detected" in s


def test_accelerator_cuda_usable_backend_has_engine(make_console):
    # usable_backend="cuda" is in _ARA_ENGINE_BACKENDS → hint says "usable" without "coming"
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(accel=detect.Accelerator("none", "none detected", None, None),
                      gpus=[GpuInfo(vendor="nvidia", name="RTX 3080",
                            vram_gb=10.0, integrated=False,
                            compute_runtime="CUDA 12.4",
                            usable_backend="cuda")])
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "RTX 3080" in s
    assert "usable" in s
    assert "engine coming" not in s


def test_accelerator_nvidia_with_other_gpu_not_double_printed(make_console):
    # accel.kind=nvidia + gpus has the same nvidia GPU + an amd GPU
    # nvidia GPU should NOT be double-printed; the amd GPU SHOULD be shown
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(
        accel=detect.Accelerator("nvidia", "RTX 4090", 24.0, "CUDA", count=1,
                                  compute="8.9", cuda_version="12.4", driver_version="550"),
        gpus=[
            GpuInfo(vendor="nvidia", name="RTX 4090", vram_gb=24.0),   # same vendor → skip
            GpuInfo(vendor="amd", name="Radeon Pro 580", vram_gb=8.0,
                    compute_runtime=None, usable_backend=None),          # other → show
        ],
    )
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "RTX 4090" in s            # shown once via the rich block
    assert "Radeon Pro 580" in s      # other GPU shown
    # The RTX 4090 must not appear twice (rich block + gpu_line)
    assert s.count("RTX 4090") == 1


def test_accelerator_apple_rich_block_preserved(make_console):
    # Apple GPU: Metal line unchanged
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(accel=detect.Accelerator("apple", "Apple M4 Pro GPU", None, "Metal",
                                               cores=16), gpus=[])
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "Apple M4 Pro GPU" in s and "Metal" in s


def test_accelerator_gpu_vram_none_omitted(make_console):
    # GpuInfo with vram_gb=None → VRAM token absent from the line
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(accel=detect.Accelerator("none", "none detected", None, None),
                      gpus=[GpuInfo(vendor="intel", name="Intel Arc A770",
                            vram_gb=None, integrated=False,
                            compute_runtime=None, usable_backend=None)])
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "Intel Arc A770" in s
    assert "GB" not in s


def test_accelerator_apple_with_other_gpu_shows_it(make_console):
    # accel.kind=apple + gpus has the apple GPU + a non-apple GPU → non-apple shown
    from ara import detect
    c, out = make_console(verbose=True)
    m = _fake_machine(
        accel=detect.Accelerator("apple", "Apple M4 Pro GPU", None, "Metal", cores=16),
        gpus=[
            GpuInfo(vendor="apple", name="Apple M4 Pro GPU"),   # same vendor → skip
            GpuInfo(vendor="intel", name="Thunderbolt eGPU", vram_gb=8.0,
                    compute_runtime=None, usable_backend=None),  # other → show
        ],
    )
    cli._det_accelerator(c, m)
    s = out.getvalue()
    assert "Apple M4 Pro GPU" in s and "Metal" in s
    assert "Thunderbolt eGPU" in s


def test_render_detect_json_has_gpus_key(monkeypatch, capsys):
    # --json must emit a 'gpus' key (Task 6 plumbing, confirmed via asdict)
    monkeypatch.setattr(cli.detect, "machine",
                        lambda: _machine(gpus=[GpuInfo(vendor="amd", name="Radeon 780M",
                                                        vram_gb=4.0, integrated=True)]))
    c = cli.Console(color=False, stream=sys.stderr)
    cli.render_detect(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert "gpus" in payload
    assert payload["gpus"][0]["vendor"] == "amd"


def test_run_rejects_malformed_model_id(make_console, monkeypatch):
    # Security: the model becomes a worker argv positional, so a flag-like id is rejected before
    # dispatch rather than handed to the engine worker. Spec 2026-06-23-capability-pipeline.
    _wire_run(monkeypatch, characterization=_CHAR)
    c, buf = make_console()
    assert cli.render_run(c, "--oops", prompt="hi") == 1
    assert "invalid model" in buf.getvalue()


def test_run_accepts_local_gguf_path(make_console, monkeypatch, tmp_path):
    # A local .gguf path passes the guard (the worker resolves it); uncharacterized here, so it
    # reaches the 'not characterized' message — proving the guard accepted the path.
    # Slug: 2026-06-25-local-gguf-cli-support
    f = tmp_path / "Local-Q4_K_M.gguf"
    f.write_bytes(b"\x00")
    _wire_run(monkeypatch, characterization=None)
    c, buf = make_console()
    assert cli.render_run(c, str(f), prompt="hi") == 1
    out = buf.getvalue()
    assert "invalid model" not in out          # guard accepted the local path
    assert "isn't characterized" in out


def test_run_looks_up_local_evidence_by_absolute_key(make_console, monkeypatch, tmp_path):
    model = tmp_path / "relative:Model-Q4_K_M.gguf"
    model.write_bytes(b"weights")
    monkeypatch.chdir(tmp_path)
    _wire_run(monkeypatch, characterization=None)
    seen = []
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda _con, _mk, _engine, model_id:
                        seen.append(model_id) or None)
    c, _ = make_console()

    assert cli.render_run(c, model.name, prompt="hi", engine="cpu") == 1
    assert seen == [str(model.resolve())]


def test_characterize_rejects_malformed_model_id(make_console, monkeypatch):
    # Same argv-injection guard on the other worker-exec'ing command.
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cpu")
    c, buf = make_console()
    assert cli.render_characterize(c, "bad=id") == 1
    assert "invalid model" in buf.getvalue()


def test_characterize_accepts_local_gguf_path(make_console, store, monkeypatch, tmp_path):
    # A local .gguf path passes the guard and flows through to a measured ceiling.
    # Slug: 2026-06-25-local-gguf-cli-support
    f = tmp_path / "Local-Q4_K_M.gguf"
    f.write_bytes(b"\x00")
    _wire_characterize(monkeypatch,
                       characterize=lambda m: {"model": m, "safe_context": 12345,
                                               "decode_context": None, "points": [[512, 1.0]]})
    c, buf = make_console()
    assert cli.render_characterize(c, str(f)) == 0
    assert "12345" in buf.getvalue()
    assert "invalid model" not in buf.getvalue()


def test_characterize_threads_flash_attn_to_vulkan_backend(make_console, store, monkeypatch):
    # flash-attention is a vulkan-only kwarg; render_characterize passes it only on that engine.
    # Slug: 2026-06-25-vulkan-flash-attention
    seen = {}
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cpu")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "llama.cpp (Vulkan)"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)

    def char(m, *, progress=False, flash_attn=True, kv_quant="f16"):
        seen["flash_attn"], seen["kv_quant"] = flash_attn, kv_quant
        return {"model": m, "safe_context": 9000, "decode_context": None, "points": [[512, 1.0]]}

    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(
        characterize=char, calibration_model_cached=lambda m: True,
        download_calibration_model=lambda m, *, progress=False: None))
    c, _ = make_console()
    assert cli.render_characterize(c, "org/m", engine="vulkan", flash_attn=False,
                                   kv_quant="q8_0") == 0
    assert seen["flash_attn"] is False
    assert seen["kv_quant"] == "q8_0"


def test_characterize_rejects_invalid_kv_quant(make_console, monkeypatch):
    # Slug: 2026-06-25-vulkan-kv-cache-quant
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "vulkan")
    c, buf = make_console()
    assert cli.render_characterize(c, "org/m", engine="vulkan", kv_quant="q3") == 1
    assert "invalid --kv-quant" in buf.getvalue()


def test_run_threads_flash_attn_to_vulkan_backend(make_console, monkeypatch):
    # Slug: 2026-06-25-vulkan-flash-attention
    seen = {}
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cpu")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "llama.cpp (Vulkan)"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, m: {
                                "safe_context": 8000,
                                "config": {"kv_quant": "q4_0", "flash_attn": False},
                                "artifact_id": "artifact:test",
                            })
    monkeypatch.setattr(cli.staleness, "artifact_matches", lambda *_a: True)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: False))

    def gen(model, prompt, *, max_context, max_tokens, flash_attn=True, kv_quant="f16"):
        seen["flash_attn"], seen["kv_quant"] = flash_attn, kv_quant
        return {"context": max_context, "completion": "hi"}

    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(generate=gen))
    c, _ = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="vulkan", flash_attn=False,
                          kv_quant="q4_0") == 0
    assert seen["flash_attn"] is False
    assert seen["kv_quant"] == "q4_0"


def test_run_rejects_invalid_kv_quant(make_console, monkeypatch):
    # Slug: 2026-06-25-vulkan-kv-cache-quant
    _wire_run(monkeypatch, characterization=_CHAR)
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="vulkan", kv_quant="nope") == 1
    assert "invalid --kv-quant" in buf.getvalue()


def test_characterize_threads_kv_quant_to_apple_backend(make_console, store, monkeypatch):
    # KV-quant is also a mlx(apple) lever; render_characterize passes kv_quant (NOT flash_attn,
    # which MLX has no knob for). Slug: 2026-06-25-mlx-kv-quant-lever
    seen = {}
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "MLX engine"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.catalog, "remember", lambda con, m: None)

    def char(m, *, progress=False, kv_quant="f16"):  # note: no flash_attn kwarg
        seen["kv_quant"] = kv_quant
        return {"model": m, "safe_context": 9000, "decode_context": None, "points": [[512, 1.0]]}

    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(
        characterize=char, calibration_model_cached=lambda m: True,
        download_calibration_model=lambda m, *, progress=False: None))
    c, _ = make_console()
    assert cli.render_characterize(c, "org/m", engine="mlx", kv_quant="q8_0") == 0
    assert seen["kv_quant"] == "q8_0"
    with cli.db.connected() as con:
        assert cli.db.get_characterization(con, "mkey", "mlx", "org/m")["config"] == {
            "kv_quant": "q8_0"
        }


def test_run_threads_kv_quant_to_apple_backend(make_console, monkeypatch):
    # Slug: 2026-06-25-mlx-kv-quant-lever
    seen = {}
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "apple")
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "MLX engine"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, m: {
                                "safe_context": 8000, "config": {"kv_quant": "q4_0"},
                                "artifact_id": "artifact:test",
                            })
    monkeypatch.setattr(cli.staleness, "artifact_matches", lambda *_a: True)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: False))

    def gen(model, prompt, *, max_context, max_tokens, kv_quant="f16"):  # no flash_attn
        seen["kv_quant"] = kv_quant
        return {"context": max_context, "completion": "hi"}

    monkeypatch.setattr(cli, "get_backend", lambda b=None: types.SimpleNamespace(generate=gen))
    c, _ = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="mlx", kv_quant="q4_0") == 0
    assert seen["kv_quant"] == "q4_0"


def test_run_refuses_ceiling_measured_with_different_config(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization={**_CHAR, "config": {"kv_quant": "q4_0"}})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="mlx", assume_yes=True) == 1
    assert "different engine settings" in buf.getvalue()
    assert "characterize" in buf.getvalue()


def test_run_accepts_ceiling_measured_with_same_config(make_console, monkeypatch):
    seen = {}

    def generate(*args, **kwargs):
        seen.update(kwargs)
        return {"completion": "ok"}

    _wire_run(monkeypatch, generate=generate,
              characterization={**_CHAR, "config": {"kv_quant": "q4_0"}})
    c, _ = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="mlx", kv_quant="q4_0",
                          assume_yes=True) == 0
    assert seen["kv_quant"] == "q4_0"


def test_run_refuses_legacy_configurable_ceiling(make_console, monkeypatch):
    _wire_run(monkeypatch, characterization={**_CHAR, "config": None})
    c, buf = make_console()
    assert cli.render_run(c, "org/m", prompt="hi", engine="mlx", assume_yes=True) == 1
    assert "predates engine-setting tracking" in buf.getvalue()


# =========================================================================== #
# render_hf — hf login / logout / status
# Slug: 2026-06-24-hf-token-auth
# =========================================================================== #

def _stub_hf_auth(monkeypatch):
    """Replace hf_auth.set_token / clear_token / status with controllable fakes.
    Returns a dict of 'last' call args/results so tests can assert what was called."""
    import ara.hf_auth as hf_auth
    state = {}

    def fake_set_token(token, *, verify=True):
        state["set_token_called"] = token
        return state.get("set_token_result",
                         {"saved": True, "user": "alice", "verified": True, "error": None})

    def fake_clear_token():
        state["clear_token_called"] = True
        return state.get("clear_token_result", {"removed": True, "shadowed_by_env": False})

    def fake_status():
        state["status_called"] = True
        return state.get("status_result",
                         {"present": True, "source": "file", "user": "alice",
                          "verified": True, "error": None})

    monkeypatch.setattr(cli.hf_auth, "set_token", fake_set_token)
    monkeypatch.setattr(cli.hf_auth, "clear_token", fake_clear_token)
    monkeypatch.setattr(cli.hf_auth, "status", fake_status)
    return state


# --------------------------------------------------------------------------- #
# _read_token — both branches
# --------------------------------------------------------------------------- #

def test_read_token_tty_calls_getpass(monkeypatch):
    import io
    import ara.cli as _cli
    monkeypatch.setattr("sys.stdin", type("FakeTTY", (), {"isatty": lambda self: True})())
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "hf_from_getpass")
    c, _ = make_console_bare = __import__("io").StringIO(), None
    result = _cli._read_token(None)
    assert result == "hf_from_getpass"


def test_read_token_non_tty_reads_stdin(monkeypatch):
    import io
    import ara.cli as _cli
    fake_stdin = io.StringIO("hf_from_pipe\n")
    monkeypatch.setattr("sys.stdin", fake_stdin)
    result = _cli._read_token(None)
    assert result == "hf_from_pipe\n"


# --------------------------------------------------------------------------- #
# login
# --------------------------------------------------------------------------- #

def test_render_hf_login_with_token_emits_warning(make_console, monkeypatch):
    """--token flag → warn about shell history leakage, then proceed."""
    state = _stub_hf_auth(monkeypatch)
    c, buf = make_console()
    rc = cli.render_hf(c, "login", token="hf_mytok")
    assert rc == 0
    out = buf.getvalue()
    assert "history" in out.lower() or "shell" in out.lower()
    assert state.get("set_token_called") == "hf_mytok"


def test_render_hf_login_token_logs_in_as_user(make_console, monkeypatch):
    state = _stub_hf_auth(monkeypatch)
    state["set_token_result"] = {"saved": True, "user": "alice", "verified": True, "error": None}
    c, buf = make_console()
    rc = cli.render_hf(c, "login", token="hf_tok")
    assert rc == 0
    assert "logged in as alice" in buf.getvalue()


def test_render_hf_login_stdin(make_console, monkeypatch):
    """No --token → read from _read_token (monkeypatched)."""
    state = _stub_hf_auth(monkeypatch)
    monkeypatch.setattr(cli, "_read_token", lambda c: "hf_stdin_tok")
    c, buf = make_console()
    rc = cli.render_hf(c, "login")
    assert rc == 0
    assert state.get("set_token_called") == "hf_stdin_tok"


def test_render_hf_login_rejected_returns_1(make_console, monkeypatch):
    """set_token says saved=False/invalid → exit 1, 'rejected' message."""
    state = _stub_hf_auth(monkeypatch)
    state["set_token_result"] = {"saved": False, "user": None, "verified": False, "error": "invalid"}
    c, buf = make_console()
    rc = cli.render_hf(c, "login", token="hf_bad")
    assert rc == 1
    assert "rejected" in buf.getvalue().lower()


def test_render_hf_login_empty_token_returns_1(make_console, monkeypatch):
    """Empty token from stdin → exit 1."""
    state = _stub_hf_auth(monkeypatch)
    monkeypatch.setattr(cli, "_read_token", lambda c: "   ")
    c, buf = make_console()
    rc = cli.render_hf(c, "login")
    assert rc == 1
    out = buf.getvalue()
    assert "no token" in out.lower()


def test_render_hf_login_empty_flag_token_returns_1(make_console, monkeypatch):
    """--token '' (empty string) → exit 1 before calling set_token."""
    state = _stub_hf_auth(monkeypatch)
    c, buf = make_console()
    rc = cli.render_hf(c, "login", token="")
    assert rc == 1
    assert "set_token_called" not in state


def test_render_hf_login_json(monkeypatch, capsys):
    """--json → prints JSON result, no styled output."""
    state = _stub_hf_auth(monkeypatch)
    state["set_token_result"] = {"saved": True, "user": "alice", "verified": True, "error": None}
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_hf(c, "login", token="hf_tok", as_json=True)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["saved"] is True and payload["user"] == "alice"
    assert payload["error"] is None
    assert payload["shadowed_by_env"] is False


def test_render_hf_login_reports_environment_override(make_console, monkeypatch, capsys):
    state = _stub_hf_auth(monkeypatch)
    state["set_token_result"] = {"saved": True, "user": "alice", "verified": True,
                                 "error": None}
    monkeypatch.setattr(cli.hf_auth, "_env_token_present", lambda: True)

    c, buf = make_console()
    assert cli.render_hf(c, "login", token="hf_tok") == 0
    out = buf.getvalue().lower()
    assert "stored token verified as alice" in out
    assert "environment" in out and "active" in out
    assert "logged in as" not in out

    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_hf(c, "login", token="hf_tok", as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["shadowed_by_env"] is True


def test_render_hf_login_offline_save_warns(make_console, monkeypatch):
    """Offline error during verify → saved=True but emits a warn about not verified."""
    state = _stub_hf_auth(monkeypatch)
    state["set_token_result"] = {"saved": True, "user": None, "verified": False, "error": "offline"}
    c, buf = make_console()
    rc = cli.render_hf(c, "login", token="hf_tok")
    assert rc == 0
    out = buf.getvalue()
    assert "couldn't verify" in out.lower() or "offline" in out.lower()


def test_render_hf_login_non_invalid_save_error(make_console, monkeypatch):
    """saved=False with error not 'invalid' → 'no token provided' fallback message."""
    state = _stub_hf_auth(monkeypatch)
    state["set_token_result"] = {"saved": False, "user": None, "verified": False, "error": "empty"}
    c, buf = make_console()
    rc = cli.render_hf(c, "login", token="   x")  # non-empty so we get past the pre-check
    assert rc == 1
    out = buf.getvalue()
    # "empty" error from set_token → "no token provided" message per spec
    assert "no token" in out.lower()


def test_render_hf_login_json_rejected(monkeypatch, capsys):
    """--json + rejected token → JSON error, exit 1."""
    state = _stub_hf_auth(monkeypatch)
    state["set_token_result"] = {"saved": False, "user": None, "verified": False, "error": "invalid"}
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_hf(c, "login", token="hf_bad", as_json=True)
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert "error" in payload


def test_render_hf_login_json_empty_token(monkeypatch, capsys):
    """--json + empty token → JSON error, exit 1."""
    state = _stub_hf_auth(monkeypatch)
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_hf(c, "login", token="", as_json=True)
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert "error" in payload


# --------------------------------------------------------------------------- #
# logout
# --------------------------------------------------------------------------- #

def test_render_hf_logout_present(make_console, monkeypatch):
    state = _stub_hf_auth(monkeypatch)
    state["clear_token_result"] = {"removed": True, "shadowed_by_env": False}
    c, buf = make_console()
    rc = cli.render_hf(c, "logout")
    assert rc == 0
    assert "removed" in buf.getvalue().lower()


def test_render_hf_logout_absent(make_console, monkeypatch):
    state = _stub_hf_auth(monkeypatch)
    state["clear_token_result"] = {"removed": False, "shadowed_by_env": False}
    c, buf = make_console()
    rc = cli.render_hf(c, "logout")
    assert rc == 0
    out = buf.getvalue()
    assert "no stored hugging face token to remove" in out.lower()


def test_render_hf_logout_env_shadowed(make_console, monkeypatch):
    state = _stub_hf_auth(monkeypatch)
    state["clear_token_result"] = {"removed": True, "shadowed_by_env": True}
    c, buf = make_console()
    rc = cli.render_hf(c, "logout")
    assert rc == 0
    out = buf.getvalue()
    assert "HF_TOKEN" in out or "env" in out.lower()


def test_render_hf_logout_json(monkeypatch, capsys):
    state = _stub_hf_auth(monkeypatch)
    state["clear_token_result"] = {"removed": True, "shadowed_by_env": False}
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_hf(c, "logout", as_json=True)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["removed"] is True


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #

def test_render_hf_status_none(make_console, monkeypatch):
    state = _stub_hf_auth(monkeypatch)
    state["status_result"] = {"present": False, "source": None, "user": None,
                              "verified": None, "error": None}
    c, buf = make_console()
    rc = cli.render_hf(c, "status")
    assert rc == 0
    out = buf.getvalue()
    assert "not logged in" in out.lower() or "ara hf login" in out


def test_render_hf_status_verified(make_console, monkeypatch):
    state = _stub_hf_auth(monkeypatch)
    state["status_result"] = {"present": True, "source": "file", "user": "alice",
                              "verified": True, "error": None}
    c, buf = make_console()
    rc = cli.render_hf(c, "status")
    assert rc == 0
    out = buf.getvalue()
    assert "alice" in out


def test_render_hf_status_unverified(make_console, monkeypatch):
    state = _stub_hf_auth(monkeypatch)
    state["status_result"] = {"present": True, "source": "env", "user": None,
                              "verified": False, "error": "offline"}
    c, buf = make_console()
    rc = cli.render_hf(c, "status")
    assert rc == 0
    out = buf.getvalue()
    assert "couldn't verify" in out.lower() or "offline" in out.lower()


def test_render_hf_status_json(monkeypatch, capsys):
    state = _stub_hf_auth(monkeypatch)
    state["status_result"] = {"present": True, "source": "file", "user": "alice",
                              "verified": True, "error": None}
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_hf(c, "status", as_json=True)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["user"] == "alice" and payload["present"] is True


@pytest.mark.parametrize("sub", ["login", "logout", "status"])
def test_render_hf_verbose_reports_token_store(sub, make_console, monkeypatch):
    _stub_hf_auth(monkeypatch)
    token_path = Path("/safe/huggingface/token")
    monkeypatch.setattr(cli.hf_auth, "_token_path", lambda: token_path)
    c, buf = make_console(verbose=True)

    token = "hf_example" if sub == "login" else None
    assert cli.render_hf(c, sub, token=token) == 0
    assert str(token_path) in buf.getvalue()


def test_render_hf_verbose_json_reports_token_store(monkeypatch, capsys):
    _stub_hf_auth(monkeypatch)
    token_path = Path("/safe/huggingface/token")
    monkeypatch.setattr(cli.hf_auth, "_token_path", lambda: token_path)
    c = cli.Console(color=False, verbose=True, stream=sys.stderr)

    assert cli.render_hf(c, "status", as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["token_path"] == str(token_path)


@pytest.mark.parametrize("sub", ["login", "logout", "status"])
@pytest.mark.parametrize("as_json", [False, True])
def test_render_hf_store_failures_are_safe(sub, as_json, make_console, monkeypatch, capsys):
    secret = "hf_never_print_this"

    def fail(*args, **kwargs):
        detail = f"permission denied while handling {secret}" if sub == "login" \
            else "permission denied"
        raise OSError(detail)

    monkeypatch.setattr(cli.hf_auth,
                        {"login": "set_token", "logout": "clear_token", "status": "status"}[sub],
                        fail)
    c, buf = make_console()

    assert cli.render_hf(c, sub, token=secret if sub == "login" else None,
                         as_json=as_json) == 1

    rendered = capsys.readouterr().out if as_json else buf.getvalue()
    assert f"hugging face {sub} failed" in rendered.lower()
    assert secret not in rendered
    if as_json:
        assert "error" in json.loads(rendered)


# --------------------------------------------------------------------------- #
# unknown / missing subcommand
# --------------------------------------------------------------------------- #

def test_render_hf_unknown_sub_returns_1(make_console, monkeypatch):
    _stub_hf_auth(monkeypatch)
    c, buf = make_console()
    rc = cli.render_hf(c, "bogus")
    assert rc == 1
    out = buf.getvalue()
    assert "bogus" in out and ("login" in out or "logout" in out or "status" in out)


def test_render_hf_none_sub_returns_1(make_console, monkeypatch):
    _stub_hf_auth(monkeypatch)
    c, buf = make_console()
    rc = cli.render_hf(c, None)
    assert rc == 1
    assert "specify an hf subcommand" in buf.getvalue()


def test_render_hf_unknown_sub_json(monkeypatch, capsys):
    _stub_hf_auth(monkeypatch)
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_hf(c, "bogus", as_json=True)
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert "error" in payload


# --------------------------------------------------------------------------- #
# main() dispatch for hf
# --------------------------------------------------------------------------- #

def test_main_hf_dispatch(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    monkeypatch.setattr(cli, "render_hf",
                        lambda c, sub, token=None, as_json=False: rec.update(hf=sub) or 0)
    _run_main(monkeypatch, ["hf", "status"])
    assert rec.get("hf") == "status"


def test_main_hf_with_token_flag(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    monkeypatch.setattr(cli, "render_hf",
                        lambda c, sub, token=None, as_json=False:
                        rec.update(hf=sub, hf_token=token) or 0)
    _run_main(monkeypatch, ["hf", "login", "--token", "hf_mytok"])
    assert rec.get("hf_token") == "hf_mytok"


def test_main_hf_with_token_equals_form(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    monkeypatch.setattr(cli, "render_hf",
                        lambda c, sub, token=None, as_json=False:
                        rec.update(hf=sub, hf_token=token) or 0)
    _run_main(monkeypatch, ["hf", "login", "--token=hf_eqtok"])
    assert rec.get("hf_token") == "hf_eqtok"


def test_main_hf_no_sub_dispatches_none(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    monkeypatch.setattr(cli, "render_hf",
                        lambda c, sub, token=None, as_json=False: rec.update(hf=sub) or 1)
    _run_main(monkeypatch, ["hf"])
    assert rec.get("hf") is None


# --------------------------------------------------------------------------- #
# serve — governed Ollama endpoint (spec 2026-06-26-ara-serve-governed-endpoint)
# --------------------------------------------------------------------------- #
# A loaded /api/ps entry for the default derived name at the requested 8192 ctx (no spill).
_SERVE_LOADED = [{"name": "qwen3-0.6b-ara:latest", "context_length": 8192,
                  "size": 100, "size_vram": 100, "digest": "b" * 64}]
_SERVE_BASE_ARTIFACT = "ollama-manifest-sha256:" + "a" * 64
_SERVE_DERIVED_ARTIFACT = "ollama-manifest-sha256:" + "b" * 64


def _serve_name(context=8192):
    return cli._governed_name(
        "qwen3:0.6b", artifact_id=_SERVE_BASE_ARTIFACT, context=context,
    )


def _wire_serve(monkeypatch, *, version="0.30.10", names=("qwen3:0.6b",), create_ok=True,
                ps_rows=None, characterization=None, isatty=False, pull_ok=True,
                show=None, size=None, estimated=8192):
    """Wire render_serve's Ollama + db seams. ``names=None`` ⇒ tags() unreachable;
    ``ps_rows`` is what /api/ps returns after load (set per-test for the verify branches).
    ``pull_ok`` is ollama.pull's result; ``show``/``size`` feed the estimated-ceiling fallback."""
    monkeypatch.setattr(cli.ollama, "pull", lambda n, timeout=600.0: pull_ok)
    monkeypatch.setattr(cli.ollama, "show", lambda n, timeout=30.0: show)
    monkeypatch.setattr(cli.ollama, "size_bytes", lambda n, timeout=2.0: size)
    monkeypatch.setattr(cli.ollama, "version", lambda timeout=0.5: version)
    monkeypatch.setattr(cli.ollama, "tags",
                        lambda timeout=2.0: (None if names is None else list(names)))
    state = {"served": None}

    def create(n, f, ctx, timeout=300.0):
        state["served"] = n
        return create_ok

    monkeypatch.setattr(cli.ollama, "create", create)
    monkeypatch.setattr(cli.ollama, "load", lambda n, keep_alive=-1, timeout=300.0: {"done": True})
    normalized_rows = [({**row, "digest": row.get("digest", "b" * 64)})
                       for row in (ps_rows or [])]
    def ps(timeout=2.0):
        rows = []
        for row in normalized_rows:
            copy = dict(row)
            if (state["served"] and isinstance(copy.get("name"), str)
                    and copy["name"] in ("qwen3-0.6b-ara", "qwen3-0.6b-ara:latest")):
                copy["name"] = state["served"] + (":latest" if copy["name"].endswith(":latest")
                                                   else "")
            rows.append(copy)
        return rows

    monkeypatch.setattr(cli.ollama, "ps", ps)
    monkeypatch.setattr(cli.ollama, "manifest_digest",
                        lambda name, timeout=2.0: "a" * 64 if name == "qwen3:0.6b" else "b" * 64)
    monkeypatch.setattr(cli.ollama, "delete", lambda n, timeout=30.0: True)
    monkeypatch.setattr(cli.ollama, "base_url", lambda: "http://127.0.0.1:11434")
    monkeypatch.setattr(cli.db, "connect", lambda: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    if characterization is not None and "artifact_id" not in characterization:
        characterization = {**characterization,
                            "artifact_id": "ollama-manifest-sha256:" + "a" * 64}
    monkeypatch.setattr(cli.db, "get_characterization", lambda con, mk, e, m: characterization)
    monkeypatch.setattr(cli.db, "save_characterization", lambda *a, **k: None)
    if show is None and size is None:
        monkeypatch.setattr(
            cli, "_ollama_estimated_ceiling",
            lambda _m: ((estimated, "estimated", None) if estimated is not None else None),
        )
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: isatty))


def test_serve_refuses_when_not_serving(make_console, monkeypatch):
    _wire_serve(monkeypatch, version=None)
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "isn't serving" in buf.getvalue()


def test_serve_refuses_when_tags_unreachable(make_console, monkeypatch):
    _wire_serve(monkeypatch, names=None)
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "couldn't list Ollama models" in buf.getvalue()


def test_serve_pulls_missing_model_then_serves(make_console, monkeypatch):
    pulled = []
    _wire_serve(monkeypatch, names=("other:1",), characterization={"safe_context": 8192},
                ps_rows=_SERVE_LOADED)
    monkeypatch.setattr(cli.ollama, "pull", lambda n, timeout=600.0: pulled.append(n) or True)
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b") == 0     # no --ctx: serves at the measured ceiling
    assert pulled == ["qwen3:0.6b"]
    assert "pulling qwen3:0.6b" in buf.getvalue()


def test_serve_pulls_missing_model_json_is_quiet(make_console, monkeypatch, capsys):
    # JSON mode: pull still happens, but the "pulling…"/"pulled." lines stay off stdout so the
    # payload is clean (covers the as_json branches of the pull block).
    _wire_serve(monkeypatch, names=("other:1",), characterization={"safe_context": 8192},
                ps_rows=_SERVE_LOADED)
    c, _ = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["base_model"] == "qwen3:0.6b" and payload["ceiling_source"] == "measured"


def test_serve_ollama_json_carries_stale_ceiling_flag(make_console, monkeypatch, capsys):
    """Rule #3: a `serve --json` consumer must learn the ceiling it's handed is stale, exactly as
    the text path warns. Regression for the dropped `_stale_ceiling_note` return (Fix #4)."""
    _wire_serve(monkeypatch, ps_rows=_SERVE_LOADED,
                characterization={"safe_context": 8192, "measured_at": "2026-01-01T00:00:00+00:00"})
    monkeypatch.setattr(cli.staleness, "ceiling_is_stale", lambda mid, at: True)
    c, _ = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["stale_ceiling"] is True


def test_serve_refuses_when_pull_fails(make_console, monkeypatch):
    _wire_serve(monkeypatch, names=("other:1",), pull_ok=False)
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b") == 1
    assert "couldn't pull" in buf.getvalue()


def test_serve_refuses_new_custom_name_before_pulling_missing_model(
        make_console, monkeypatch):
    _wire_serve(monkeypatch, names=("other:1",))
    monkeypatch.setattr(cli.ollama, "pull",
                        lambda *_a, **_k: pytest.fail("pulled for unsupported custom name"))
    c, buf = make_console()
    assert cli.render_serve(c, "missing:model", ctx=8192, name="mysrv") == 1
    assert "custom --name" in buf.getvalue() and "atomic" in buf.getvalue()


def test_serve_refuses_unidentified_base_manifest(make_console, monkeypatch):
    _wire_serve(monkeypatch)
    monkeypatch.setattr(cli.ollama, "manifest_digest", lambda _name: None)
    monkeypatch.setattr(cli.ollama, "create",
                        lambda *_a, **_k: pytest.fail("created unidentified base"))
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "artifact provenance" in buf.getvalue()


def test_serve_rejects_nonpositive_ctx(make_console, monkeypatch):
    _wire_serve(monkeypatch)
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=0) == 1
    assert "positive integer" in buf.getvalue()


def test_serve_rejects_nonpositive_ctx_before_ollama_side_effects(make_console, monkeypatch):
    calls = []
    monkeypatch.setattr(cli.ollama, "version", lambda *_a: calls.append("version") or "1.0")
    monkeypatch.setattr(cli.ollama, "pull", lambda *_a, **_k: calls.append("pull") or True)
    c, buf = make_console()
    assert cli.render_serve(c, "missing:model", ctx=0) == 1
    assert "positive integer" in buf.getvalue()
    assert calls == []


def test_serve_explicit_ctx_cannot_exceed_estimated_bound(make_console, monkeypatch):
    _wire_serve(monkeypatch, characterization=None)
    monkeypatch.setattr(cli, "_ollama_estimated_ceiling", lambda _m: (8192, "estimated", None))
    monkeypatch.setattr(cli.ollama, "create", lambda *_a, **_k: pytest.fail("unsafe create"))
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=16384) == 1
    assert "estimated safe bound 8192" in buf.getvalue()


def test_serve_explicit_ctx_refuses_without_measured_or_estimated_bound(
        make_console, monkeypatch):
    _wire_serve(monkeypatch, characterization=None)
    monkeypatch.setattr(cli, "_ollama_estimated_ceiling", lambda _m: None)
    monkeypatch.setattr(cli.ollama, "create", lambda *_a, **_k: pytest.fail("unbounded create"))
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    out = buf.getvalue()
    assert "no measured or estimated safe bound" in out
    assert "ara characterize" in out


def test_serve_rejects_invalid_custom_name_before_ollama_side_effects(make_console, monkeypatch):
    calls = []
    monkeypatch.setattr(cli.ollama, "version", lambda *_a: calls.append("version") or "1.0")
    monkeypatch.setattr(cli.ollama, "pull", lambda *_a, **_k: calls.append("pull") or True)
    monkeypatch.setattr(cli.ollama, "base_url", lambda: "http://127.0.0.1:11434")
    c, buf = make_console()
    assert cli.render_serve(c, "missing:model", ctx=8, name="bad\nname") == 1
    assert "invalid serving identity" in buf.getvalue()
    assert calls == []


def test_serve_rejects_invalid_model_before_ollama_side_effects(make_console, monkeypatch):
    calls = []
    monkeypatch.setattr(cli.ollama, "version", lambda *_a: calls.append("version") or "1.0")
    monkeypatch.setattr(cli.ollama, "base_url", lambda: "http://127.0.0.1:11434")
    c, buf = make_console()
    assert cli.render_serve(c, "bad\nmodel", ctx=8) == 1
    assert "invalid serving identity" in buf.getvalue()
    assert calls == []


@pytest.mark.parametrize("engine", ["cuda", "cpu", "vulkan", "cuda-gguf", "bogus"])
def test_serve_rejects_unsupported_explicit_engine(make_console, monkeypatch, engine):
    monkeypatch.setattr(cli.ollama, "version",
                        lambda *_a: pytest.fail("unsupported engine reached Ollama"))
    c, buf = make_console()
    assert cli.render_serve(c, "org/model", engine=engine) == 1
    assert "serve" in buf.getvalue().lower() and "engine" in buf.getvalue().lower()


@pytest.mark.parametrize("engine", ["ollama", "auto"])
def test_serve_accepts_ollama_and_nonapple_auto(
        make_console, monkeypatch, set_platform, engine):
    set_platform("Linux", "x86_64")
    _wire_serve(monkeypatch, ps_rows=_SERVE_LOADED)
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192, engine=engine) == 0
    assert _serve_name() in buf.getvalue()


def test_serve_uses_estimated_ceiling_when_unmeasured(make_console, monkeypatch, capsys):
    rows = [{"name": "qwen3-0.6b-ara:latest", "context_length": 8192,
             "size": 100, "size_vram": 100}]
    _wire_serve(monkeypatch, characterization=None, ps_rows=rows)   # nothing measured
    monkeypatch.setattr(cli, "_ollama_estimated_ceiling",
                        lambda m: (8192, "estimated", None))
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["served_context"] == 8192 and payload["ceiling_source"] == "estimated"


def test_serve_estimated_ceiling_nudges_to_characterize(make_console, monkeypatch):
    rows = [{"name": "qwen3-0.6b-ara:latest", "context_length": 8192,
             "size": 100, "size_vram": 100}]
    _wire_serve(monkeypatch, characterization=None, ps_rows=rows)
    monkeypatch.setattr(cli, "_ollama_estimated_ceiling",
                        lambda m: (8192, "estimated", None))
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b") == 0
    out = buf.getvalue()
    assert "estimated" in out and "ara characterize qwen3:0.6b" in out


def test_serve_refuses_when_neither_measured_nor_estimable(make_console, monkeypatch):
    _wire_serve(monkeypatch, characterization=None, show=None, estimated=None)
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b") == 1               # no --ctx, no guess
    assert "couldn't determine a safe ceiling" in buf.getvalue()


def test_serve_uses_measured_ceiling(make_console, monkeypatch, capsys):
    rows = [{"name": "qwen3-0.6b-ara:latest", "context_length": 4096,
             "size": 100, "size_vram": 100}]
    _wire_serve(monkeypatch, characterization={"safe_context": 4096}, ps_rows=rows)
    c, _ = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["served_context"] == 4096 and payload["ceiling_source"] == "measured"


def test_serve_uses_measured_ceiling_only_for_current_ollama_manifest(
        make_console, monkeypatch, capsys):
    artifact = "ollama-manifest-sha256:" + "a" * 64
    rows = [{"name": "qwen3-0.6b-ara:latest", "context_length": 4096,
             "size": 100, "size_vram": 100}]
    _wire_serve(monkeypatch, characterization={"safe_context": 4096,
                                               "artifact_id": artifact}, ps_rows=rows)
    c, _ = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["ceiling_source"] == "measured"


def test_serve_ignores_measurement_for_retargeted_ollama_manifest(
        make_console, monkeypatch, capsys):
    old_artifact = "ollama-manifest-sha256:" + "c" * 64
    rows = [{"name": "qwen3-0.6b-ara:latest", "context_length": 2048,
             "size": 100, "size_vram": 100}]
    _wire_serve(monkeypatch, characterization={"safe_context": 4096,
                                               "artifact_id": old_artifact}, ps_rows=rows)
    monkeypatch.setattr(cli, "_ollama_estimated_ceiling", lambda _m: (2048, "estimated", None))
    c, _ = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ceiling_source"] == "estimated"
    assert payload["served_context"] == 2048


def test_serve_refuses_when_loaded_manifest_digest_differs_from_created_manifest(
        make_console, monkeypatch):
    rows = [{"name": "qwen3-0.6b-ara:latest", "context_length": 8192,
             "size": 100, "size_vram": 100, "digest": "c" * 64}]
    _wire_serve(monkeypatch, ps_rows=rows, characterization={
        "safe_context": 8192, "artifact_id": "ollama-manifest-sha256:" + "a" * 64})
    cleaned = []
    monkeypatch.setattr(cli, "_cleanup_ollama_model",
                        lambda name, **_k: cleaned.append(name) or None)
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "manifest" in buf.getvalue() and "governance" in buf.getvalue()
    assert len(cleaned) == 1


def test_serve_passes_verified_digest_into_failure_cleanup(make_console, monkeypatch):
    _wire_serve(monkeypatch)
    monkeypatch.setattr(cli.ollama, "load", lambda *_a, **_k: None)
    cleanup = {}

    def capture(name, *, label, delete, expected_artifact_id=None):
        cleanup.update(name=name, expected=expected_artifact_id)
        return None

    monkeypatch.setattr(cli, "_cleanup_ollama_model", capture)
    c, _ = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert cleanup == {"name": _serve_name(), "expected": _SERVE_DERIVED_ARTIFACT}


def test_serve_records_base_and_served_manifest_identity(make_console, monkeypatch):
    _wire_serve(monkeypatch, ps_rows=_SERVE_LOADED, characterization={
        "safe_context": 8192, "artifact_id": "ollama-manifest-sha256:" + "a" * 64})
    recorded = {}
    monkeypatch.setattr(cli.activity, "record_ollama_serving",
                        lambda **fields: recorded.update(fields))
    c, _ = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 0
    assert recorded["base_artifact_id"] == "ollama-manifest-sha256:" + "a" * 64
    assert recorded["served_artifact_id"] == "ollama-manifest-sha256:" + "b" * 64


def test_serve_refuses_when_create_fails(make_console, monkeypatch):
    _wire_serve(monkeypatch, create_ok=False)
    cleaned = []
    monkeypatch.setattr(cli.ollama, "load",
                        lambda name, keep_alive=-1, timeout=300.0:
                        cleaned.append(name) or {} if keep_alive == 0 else {"done": True})
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "couldn't confirm creation" in buf.getvalue()
    assert "ownership is unknown" in buf.getvalue()
    assert cleaned == []


def test_serve_refuses_created_manifest_without_identity(make_console, monkeypatch):
    _wire_serve(monkeypatch)
    monkeypatch.setattr(cli.ollama, "manifest_digest",
                        lambda name: "a" * 64 if name == "qwen3:0.6b" else None)
    monkeypatch.setattr(cli.ollama, "load",
                        lambda *_a, **_k: pytest.fail("loaded unidentified manifest"))
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "couldn't identify its Ollama manifest" in buf.getvalue()


def test_serve_does_not_cleanup_when_derived_manifest_retargets_before_failure(
        make_console, monkeypatch):
    _wire_serve(monkeypatch)
    served_reads = iter(["b" * 64, "c" * 64])
    monkeypatch.setattr(
        cli.ollama, "manifest_digest",
        lambda name: "a" * 64 if name == "qwen3:0.6b" else next(served_reads),
    )
    monkeypatch.setattr(cli.ollama, "load", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_cleanup_ollama_model",
                        lambda *_a, **_k: pytest.fail("retargeted manifest deleted"))
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "changed before load" in buf.getvalue() and "did not unload or delete" in buf.getvalue()


def test_serve_does_not_cleanup_when_manifest_becomes_unverifiable_after_load_failure(
        make_console, monkeypatch):
    _wire_serve(monkeypatch)
    served_reads = iter(["b" * 64, "b" * 64, None])
    monkeypatch.setattr(
        cli.ollama, "manifest_digest",
        lambda name: "a" * 64 if name == "qwen3:0.6b" else next(served_reads),
    )
    monkeypatch.setattr(cli.ollama, "load", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_cleanup_ollama_model",
                        lambda *_a, **_k: pytest.fail("cleaned unverifiable manifest"))
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "ownership of the derived manifest is unverified" in buf.getvalue()


def test_serve_refuses_if_base_manifest_retargets_during_setup(make_console, monkeypatch):
    _wire_serve(monkeypatch, ps_rows=_SERVE_LOADED)
    base_reads = iter(["a" * 64, "c" * 64])
    monkeypatch.setattr(
        cli.ollama, "manifest_digest",
        lambda name: next(base_reads) if name == "qwen3:0.6b" else "b" * 64,
    )
    monkeypatch.setattr(
        cli.ollama, "load",
        lambda *_a, keep_alive=-1, **_k: (
            {} if keep_alive == 0 else pytest.fail("loaded retargeted base")),
    )
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "manifest changed during setup" in buf.getvalue()


def test_serve_refuses_if_base_manifest_retargets_during_load(make_console, monkeypatch):
    _wire_serve(monkeypatch, ps_rows=_SERVE_LOADED)
    base_reads = iter(["a" * 64, "a" * 64, "c" * 64])
    monkeypatch.setattr(
        cli.ollama, "manifest_digest",
        lambda name: next(base_reads) if name == "qwen3:0.6b" else "b" * 64,
    )
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "manifest changed during setup" in buf.getvalue()


def test_serve_interrupt_after_create_cleans_up_then_propagates(make_console, monkeypatch):
    _wire_serve(monkeypatch)
    cleaned = []

    def load(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.ollama, "load", load)
    monkeypatch.setattr(cli, "_cleanup_ollama_model",
                        lambda name, *, label, delete, expected_artifact_id=None:
                        cleaned.append((name, expected_artifact_id)) or None)
    c, _ = make_console()
    with pytest.raises(KeyboardInterrupt):
        cli.render_serve(c, "qwen3:0.6b", ctx=8192)
    assert cleaned == [(_serve_name(), _SERVE_DERIVED_ARTIFACT)]


def test_serve_interrupt_reports_cleanup_failure(make_console, monkeypatch):
    _wire_serve(monkeypatch)
    monkeypatch.setattr(cli.ollama, "load", lambda *_a, **_k: (_ for _ in ()).throw(SystemExit(2)))
    monkeypatch.setattr(cli, "_cleanup_ollama_model",
                        lambda *_a, **_k: "couldn't verify unload")
    c, buf = make_console()
    with pytest.raises(SystemExit):
        cli.render_serve(c, "qwen3:0.6b", ctx=8192)
    assert "cleanup failed" in buf.getvalue() and "verify unload" in buf.getvalue()


def test_serve_refuses_name_appearing_at_final_collision_check(make_console, monkeypatch):
    _wire_serve(monkeypatch)
    calls = 0
    served = _serve_name()

    def tags(*_a, **_k):
        nonlocal calls
        calls += 1
        return ["qwen3:0.6b"] if calls == 1 else ["qwen3:0.6b", f"{served}:latest"]

    monkeypatch.setattr(cli.ollama, "tags", tags)
    monkeypatch.setattr(cli.ollama, "create",
                        lambda *_a, **_k: pytest.fail("late collision overwritten"))
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "appeared" in buf.getvalue() and "refusing" in buf.getvalue()


def test_serve_refuses_when_final_collision_check_is_unavailable(make_console, monkeypatch):
    _wire_serve(monkeypatch)
    calls = 0

    def tags(*_a, **_k):
        nonlocal calls
        calls += 1
        return ["qwen3:0.6b"] if calls == 1 else None

    monkeypatch.setattr(cli.ollama, "tags", tags)
    monkeypatch.setattr(cli.ollama, "create",
                        lambda *_a, **_k: pytest.fail("created without collision check"))
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "recheck" in buf.getvalue() and "collision" in buf.getvalue()


def test_serve_refuses_when_same_identity_setup_is_locked(make_console, monkeypatch):
    _wire_serve(monkeypatch)

    @contextlib.contextmanager
    def busy(_endpoint, served):
        raise cli.locking.OllamaSetupBusy(f"busy setting up {served}")
        yield

    monkeypatch.setattr(cli.locking, "ollama_setup_lock", busy)
    monkeypatch.setattr(cli.ollama, "create",
                        lambda *_a, **_k: pytest.fail("created despite setup lock"))
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "busy setting up" in buf.getvalue()


def test_serve_refuses_when_model_does_not_load(make_console, monkeypatch):
    _wire_serve(monkeypatch, ps_rows=[])                   # nothing in /api/ps
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "didn't load" in buf.getvalue()


def test_serve_load_failure_cannot_reuse_stale_ps_row(make_console, monkeypatch):
    _wire_serve(monkeypatch, ps_rows=_SERVE_LOADED)
    cleanup_started = []
    deleted = []

    def load(name, keep_alive=-1, timeout=300.0):
        if keep_alive == 0:
            cleanup_started.append(name)
            return {}
        return None

    monkeypatch.setattr(cli.ollama, "load", load)
    monkeypatch.setattr(cli.ollama, "ps",
                        lambda *_a: [] if cleanup_started else _SERVE_LOADED)
    monkeypatch.setattr(cli.ollama, "delete", lambda name: deleted.append(name) or True)
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "couldn't load" in buf.getvalue()
    assert cleanup_started == [_serve_name()]
    assert deleted == [_serve_name()]


def test_serve_refuses_unowned_preexisting_governed_manifest(make_console, monkeypatch):
    _wire_serve(monkeypatch, names=("qwen3:0.6b", f"{_serve_name()}:latest"))
    monkeypatch.setattr(cli.ollama, "create",
                        lambda *_a, **_k: pytest.fail("pre-existing manifest overwritten"))
    monkeypatch.setattr(cli.ollama, "load",
                        lambda *_a, **_k: pytest.fail("pre-existing service disrupted"))
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "already exists" in buf.getvalue() and "refusing" in buf.getvalue()


def test_serve_refuses_custom_name_before_pulling_a_colliding_base(make_console, monkeypatch):
    _wire_serve(monkeypatch, names=("other:1",))
    monkeypatch.setattr(cli.ollama, "create",
                        lambda *_a, **_k: pytest.fail("pulled base manifest overwritten"))
    c, buf = make_console()
    assert cli.render_serve(c, "new:model", ctx=8192, name="new:model") == 1
    assert "custom --name" in buf.getvalue() and "atomic" in buf.getvalue()


def test_serve_reuses_exact_owned_manifest_without_recreating(make_console, monkeypatch):
    served = _serve_name()
    _wire_serve(monkeypatch, names=("qwen3:0.6b", f"{served}:latest"),
                ps_rows=[{"name": f"{served}:latest", "context_length": 8192,
                          "size": 100, "size_vram": 100,
                          "digest": "b" * 64}])
    monkeypatch.setattr(
        cli.activity, "snapshot",
        lambda: [types.SimpleNamespace(runtime="ollama", served_name=served,
                                       context=8192, endpoint="http://127.0.0.1:11434",
                                       model="qwen3:0.6b",
                                       base_artifact_id=_SERVE_BASE_ARTIFACT,
                                       served_artifact_id=_SERVE_DERIVED_ARTIFACT)],
    )
    monkeypatch.setattr(cli.ollama, "create",
                        lambda *_a, **_k: pytest.fail("owned manifest recreated"))
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 0
    assert served in buf.getvalue()


def test_serve_refuses_duplicate_while_legacy_ara_service_is_live(make_console, monkeypatch):
    _wire_serve(monkeypatch)
    monkeypatch.setattr(cli.activity, "snapshot", lambda: [types.SimpleNamespace(
        runtime="ollama", served_name="qwen3-0.6b-ara", context=8192,
        endpoint="http://127.0.0.1:11434", model="qwen3:0.6b",
        base_artifact_id=None, served_artifact_id=None,
    )])
    monkeypatch.setattr(cli.ollama, "create",
                        lambda *_a, **_k: pytest.fail("duplicated legacy service"))
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "legacy ARA service" in buf.getvalue()


def test_serve_owned_manifest_failure_does_not_cleanup_user_state(make_console, monkeypatch):
    served = _serve_name()
    _wire_serve(monkeypatch, names=("qwen3:0.6b", f"{served}:latest"))
    monkeypatch.setattr(
        cli.activity, "snapshot",
        lambda: [types.SimpleNamespace(runtime="ollama", served_name=served,
                                       context=8192, endpoint="http://127.0.0.1:11434",
                                       model="qwen3:0.6b",
                                       base_artifact_id=_SERVE_BASE_ARTIFACT,
                                       served_artifact_id=_SERVE_DERIVED_ARTIFACT)],
    )
    monkeypatch.setattr(cli.ollama, "load", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_cleanup_ollama_model",
                        lambda *_a, **_k: pytest.fail("owned model cleaned up"))
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "couldn't load" in buf.getvalue()


def test_serve_reports_cleanup_failure(make_console, monkeypatch):
    _wire_serve(monkeypatch)
    monkeypatch.setattr(cli.ollama, "load", lambda *_a, **_k: None)
    monkeypatch.setattr(cli.ollama, "ps", lambda *_a: None)
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    out = buf.getvalue()
    assert "cleanup also failed" in out and "verify" in out


def test_serve_unverifiable_ps_cleans_up_derived_model(make_console, monkeypatch):
    _wire_serve(monkeypatch)
    cleanup_started = []
    deleted = []
    monkeypatch.setattr(cli.ollama, "load",
                        lambda name, keep_alive=-1, timeout=300.0:
                        cleanup_started.append(name) or {} if keep_alive == 0 else {"done": True})
    monkeypatch.setattr(cli.ollama, "ps", lambda *_a: [] if cleanup_started else None)
    monkeypatch.setattr(cli.ollama, "delete", lambda name: deleted.append(name) or True)
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "verify" in buf.getvalue().lower()
    assert deleted == [_serve_name()]


def test_serve_refuses_on_governance_mismatch(make_console, monkeypatch):
    rows = [{"name": "qwen3-0.6b-ara:latest", "context_length": 40960,
             "size": 100, "size_vram": 100}]
    _wire_serve(monkeypatch, ps_rows=rows)
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert "governance failed" in buf.getvalue()


def test_serve_governance_mismatch_cleans_up_derived_model(make_console, monkeypatch):
    rows = [{"name": "qwen3-0.6b-ara:latest", "context_length": 40960,
             "size": 100, "size_vram": 100}]
    _wire_serve(monkeypatch, ps_rows=rows)
    cleanup_started = []
    deleted = []
    monkeypatch.setattr(cli.ollama, "load",
                        lambda name, keep_alive=-1, timeout=300.0:
                        cleanup_started.append(name) or {} if keep_alive == 0 else {"done": True})
    monkeypatch.setattr(cli.ollama, "ps", lambda *_a: [] if cleanup_started else rows)
    monkeypatch.setattr(cli.ollama, "delete", lambda name: deleted.append(name) or True)
    c, _ = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 1
    assert deleted == [_serve_name()]


def test_serve_confirm_declined_skips(make_console, monkeypatch):
    _wire_serve(monkeypatch, ps_rows=_SERVE_LOADED, isatty=True)
    monkeypatch.setattr(cli, "_confirm", lambda q: False)
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 0
    assert "skipped" in buf.getvalue()


def test_serve_confirm_accepted_proceeds(make_console, monkeypatch):
    _wire_serve(monkeypatch, ps_rows=_SERVE_LOADED, isatty=True)
    monkeypatch.setattr(cli, "_confirm", lambda q: True)
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 0
    assert _serve_name() in buf.getvalue()


def test_serve_happy_text(make_console, monkeypatch):
    _wire_serve(monkeypatch, ps_rows=_SERVE_LOADED, isatty=False)
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 0
    out = buf.getvalue()
    assert _serve_name() in out
    assert "OPENAI_BASE_URL=http://127.0.0.1:11434/v1" in out


def test_serve_happy_json(make_console, monkeypatch, capsys):
    _wire_serve(monkeypatch, ps_rows=_SERVE_LOADED)
    c, _ = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192, as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == _serve_name()
    assert payload["base_model"] == "qwen3:0.6b"
    assert payload["served_context"] == 8192
    assert payload["endpoint"] == "http://127.0.0.1:11434/v1"
    assert payload["ceiling_source"] == "requested"
    assert payload["spilled"] is False


def test_serve_custom_name_and_assume_yes_still_refuses_unsafe_creation(
        make_console, monkeypatch):
    rows = [{"name": "mysrv:latest", "context_length": 8192, "size": 100, "size_vram": 100}]
    _wire_serve(monkeypatch, ps_rows=rows, isatty=True)    # tty, but --yes bypasses consent
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192, name="mysrv", assume_yes=True) == 1
    assert "custom --name" in buf.getvalue()


def test_serve_refuses_new_custom_name_without_atomic_create(make_console, monkeypatch):
    _wire_serve(monkeypatch, isatty=False)
    monkeypatch.setattr(cli.ollama, "create",
                        lambda *_a, **_k: pytest.fail("created racy custom name"))
    c, buf = make_console()
    assert cli.render_serve(
        c, "qwen3:0.6b", ctx=8192, name="mysrv", assume_yes=True) == 1
    assert "custom --name" in buf.getvalue() and "atomic" in buf.getvalue()


def test_serve_warns_on_spill(make_console, monkeypatch):
    rows = [{"name": "qwen3-0.6b-ara:latest", "context_length": 8192,
             "size": 200, "size_vram": 100}]            # size_vram < size ⇒ partial offload
    _wire_serve(monkeypatch, ps_rows=rows)
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=8192) == 0
    assert "partially offloaded" in buf.getvalue()


# serve helpers
def test_governed_name_legacy_compatibility_sanitizes():
    assert cli._governed_name("qwen3:0.6b") == "qwen3-0.6b-ara"
    assert cli._governed_name("Org/Repo-Name") == "org-repo-name-ara"


def test_governed_name_is_content_addressed_by_manifest_and_context():
    artifact = "ollama-manifest-sha256:" + "a" * 64
    name = cli._governed_name("Org/Model:latest", artifact_id=artifact, context=8192)
    assert name.startswith("ara-org-model-latest-ctx8192-")
    assert len(name.rsplit("-", 1)[-1]) == 24
    assert name.isascii() and len(name) <= 96
    assert cli._governed_name("Org/Model:latest", artifact_id=artifact, context=4096) != name
    assert cli._governed_name(
        "Org/Model:latest", artifact_id="ollama-manifest-sha256:" + "b" * 64,
        context=8192) != name


def test_governed_name_hash_prevents_slug_collisions():
    artifact = "ollama-manifest-sha256:" + "a" * 64
    assert cli._governed_name("a/b:c", artifact_id=artifact, context=8192) != cli._governed_name(
        "a-b-c", artifact_id=artifact, context=8192)


def test_find_loaded_matches_exact_tagged_and_none():
    assert cli._find_loaded([{"name": "srv"}], "srv") == {"name": "srv"}
    assert cli._find_loaded([{"name": "srv:latest"}], "srv") == {"name": "srv:latest"}
    assert cli._find_loaded([{"name": "other"}], "srv") is None


def test_ollama_safe_ceiling_requires_matching_manifest_artifact(monkeypatch):
    artifact = "ollama-manifest-sha256:" + "a" * 64
    chars = {"ollama": {"safe_context": 8000,
                         "measured_at": "2026-06-01T00:00:00+00:00",
                         "artifact_id": artifact}}
    monkeypatch.setattr(cli.db, "get_characterization", lambda con, mk, e, m: chars.get(e))
    assert cli._ollama_safe_ceiling(object(), "mk", "m", artifact) == (
        8000, "measured", "2026-06-01T00:00:00+00:00")
    assert cli._ollama_safe_ceiling(
        object(), "mk", "m", "ollama-manifest-sha256:" + "b" * 64) is None


def test_ollama_safe_ceiling_rejects_nondefault_measurement_config(monkeypatch):
    artifact = "ollama-manifest-sha256:" + "a" * 64
    monkeypatch.setattr(cli.db, "get_characterization", lambda *_a: {
        "safe_context": 8000, "artifact_id": artifact, "config": {"kv_quant": "q4_0"},
    })
    assert cli._ollama_safe_ceiling(object(), "mk", "m", artifact) is None


def test_ollama_safe_ceiling_does_not_transfer_other_runtime_or_legacy_rows(monkeypatch):
    chars = {
        "ollama": {"safe_context": 7000, "artifact_id": None},
        "cpu": {"safe_context": 8000},
        "cuda-gguf": {"safe_context": 9000},
    }
    monkeypatch.setattr(cli.db, "get_characterization", lambda con, mk, e, m: chars.get(e))
    artifact = "ollama-manifest-sha256:" + "a" * 64
    assert cli._ollama_safe_ceiling(object(), "mk", "m", artifact) is None


def test_ollama_safe_ceiling_vulkan_only_is_none(monkeypatch):
    monkeypatch.setattr(cli.db, "get_characterization", lambda con, mk, e, m: (
        {"safe_context": 8000, "config": {}} if e == "vulkan" else None
    ))
    assert cli._ollama_safe_ceiling(
        object(), "mk", "m", "ollama-manifest-sha256:" + "a" * 64) is None


def test_ollama_safe_ceiling_none_when_unfitted(monkeypatch):
    chars = {"cpu": None, "vulkan": {"safe_context": None}}
    monkeypatch.setattr(cli.db, "get_characterization", lambda con, mk, e, m: chars.get(e))
    assert cli._ollama_safe_ceiling(
        object(), "mk", "m", "ollama-manifest-sha256:" + "a" * 64) is None


def test_ollama_safe_ceiling_does_not_transfer_cuda_gguf_without_artifact_proof(monkeypatch):
    chars = {"cpu": {"safe_context": 4000}, "vulkan": {"safe_context": 5000},
             "cuda-gguf": {"safe_context": 12000}}
    monkeypatch.setattr(cli.db, "get_characterization", lambda con, mk, e, m: chars.get(e))
    assert cli._ollama_safe_ceiling(
        object(), "mk", "m", "ollama-manifest-sha256:" + "a" * 64) is None


def test_ollama_safe_ceiling_skips_other_runtime_configs(monkeypatch):
    rows = {
        "vulkan": {"safe_context": 16000, "config": {"kv_quant": "q4_0"}},
        "cpu": {"safe_context": 4000, "config": {}},
    }
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, key, model: rows.get(key))
    assert cli._ollama_safe_ceiling(
        object(), "mk", "m", "ollama-manifest-sha256:" + "a" * 64) is None


# _ollama_estimated_ceiling — engine-free fallback via Ollama's own /api/show
# Spec 2026-07-04-ara-serve-one-command-estimated-ceiling.
_SHOW_QWEN3 = {"model_info": {"general.architecture": "qwen3", "qwen3.block_count": 28,
               "qwen3.attention.head_count_kv": 8, "qwen3.attention.key_length": 128,
               "qwen3.context_length": 40960}}


def test_ollama_estimated_ceiling_maps_meta_and_delegates(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli.ollama, "show", lambda n, timeout=30.0: _SHOW_QWEN3)
    monkeypatch.setattr(cli.ollama, "size_bytes", lambda n, timeout=2.0: 500_000_000)
    monkeypatch.setattr(cli.detect, "machine", lambda: object())
    monkeypatch.setattr(cli.estimate, "limits", lambda m: {"safe_budget_gb": 10.0})
    monkeypatch.setattr(cli.estimate, "model_fit",
                        lambda lim, meta, w: captured.update(meta=meta, w=w, lim=lim)
                        or {"est_context": 12345})
    assert cli._ollama_estimated_ceiling("qwen3:0.6b") == (12345, "estimated", None)
    # /api/show model_info → estimator meta, weights in DECIMAL GB (bytes / 1e9)
    assert captured["meta"] == {"n_layers": 28, "kv_heads": 8, "head_dim": 128,
                                "max_context": 40960}
    assert captured["w"] == 0.5
    assert captured["lim"] == {"safe_budget_gb": 10.0}


def test_ollama_estimated_ceiling_none_when_show_unavailable(monkeypatch):
    monkeypatch.setattr(cli.ollama, "show", lambda n, timeout=30.0: None)
    assert cli._ollama_estimated_ceiling("m") is None


def test_ollama_estimated_ceiling_none_when_arch_missing(monkeypatch):
    monkeypatch.setattr(cli.ollama, "show", lambda n, timeout=30.0: {"model_info": {}})
    assert cli._ollama_estimated_ceiling("m") is None


def test_ollama_estimated_ceiling_none_when_estimator_cannot(monkeypatch):
    monkeypatch.setattr(cli.ollama, "show", lambda n, timeout=30.0: _SHOW_QWEN3)
    monkeypatch.setattr(cli.ollama, "size_bytes", lambda n, timeout=2.0: None)
    monkeypatch.setattr(cli.detect, "machine", lambda: object())
    monkeypatch.setattr(cli.estimate, "limits", lambda m: {"safe_budget_gb": 10.0})
    monkeypatch.setattr(cli.estimate, "model_fit", lambda lim, meta, w: {"est_context": None})
    assert cli._ollama_estimated_ceiling("m") is None


# _ollama_pick_best — zero-arg `ara serve` selection: best-fitting model in the Ollama store
# Spec 2026-07-04-ara-serve-zero-arg-recommend-then-serve.
def _wire_pick(monkeypatch):
    monkeypatch.setattr(cli.db, "connect", lambda: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mk")


def test_ollama_pick_best_ranks_by_ceiling_measured_or_estimated(monkeypatch):
    _wire_pick(monkeypatch)
    ceils = {"a:1": (4000, "estimated", None), "b:1": (9000, "estimated", None),
             "c:1": (6000, "estimated", None)}
    monkeypatch.setattr(cli, "_ollama_safe_ceiling", lambda con, mk, m: None)
    monkeypatch.setattr(cli, "_ollama_estimated_ceiling", lambda m: ceils.get(m))
    assert cli._ollama_pick_best(["a:1", "b:1", "c:1"]) == "b:1"


def test_ollama_pick_best_uses_measured_when_present(monkeypatch):
    _wire_pick(monkeypatch)
    # a has a measured ceiling; b only an estimate — each model uses its own best source, then
    # we rank by value (b's 9000 estimate legitimately beats a's 5000 measured).
    monkeypatch.setattr(cli, "_ollama_safe_ceiling",
                        lambda con, mk, m: (5000, "measured", None) if m == "a:1" else None)
    monkeypatch.setattr(cli, "_ollama_estimated_ceiling",
                        lambda m: (9000, "estimated", None) if m == "b:1" else None)
    assert cli._ollama_pick_best(["a:1", "b:1"]) == "b:1"


def test_ollama_pick_best_none_when_nothing_fits(monkeypatch):
    _wire_pick(monkeypatch)
    monkeypatch.setattr(cli, "_ollama_safe_ceiling", lambda con, mk, m: None)
    monkeypatch.setattr(cli, "_ollama_estimated_ceiling", lambda m: None)
    assert cli._ollama_pick_best(["a:1", "b:1"]) is None


def test_serve_zero_arg_selects_and_serves_json(make_console, monkeypatch, capsys):
    _wire_serve(monkeypatch, names=("qwen3:0.6b",), characterization={"safe_context": 8192},
                ps_rows=_SERVE_LOADED)
    monkeypatch.setattr(cli, "_ollama_pick_best", lambda names: "qwen3:0.6b")
    c, _ = make_console()
    assert cli.render_serve(c, None, as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["base_model"] == "qwen3:0.6b" and payload["auto_selected"] is True


def test_serve_zero_arg_text_announces_pick(make_console, monkeypatch):
    _wire_serve(monkeypatch, names=("qwen3:0.6b",), characterization={"safe_context": 8192},
                ps_rows=_SERVE_LOADED)
    monkeypatch.setattr(cli, "_ollama_pick_best", lambda names: "qwen3:0.6b")
    c, buf = make_console()
    assert cli.render_serve(c, None) == 0
    assert "auto-selected qwen3:0.6b" in buf.getvalue()


def test_serve_zero_arg_refuses_when_nothing_fits(make_console, monkeypatch):
    _wire_serve(monkeypatch, names=("qwen3:0.6b",))
    monkeypatch.setattr(cli, "_ollama_pick_best", lambda names: None)
    c, buf = make_console()
    assert cli.render_serve(c, None) == 1
    assert "no model in Ollama fits" in buf.getvalue()


def test_serve_zero_arg_refuses_when_store_empty(make_console, monkeypatch):
    _wire_serve(monkeypatch, names=())
    c, buf = make_console()
    assert cli.render_serve(c, None) == 1
    assert "no models in Ollama" in buf.getvalue()


def test_serve_zero_arg_rejects_unsafe_selected_model(make_console, monkeypatch):
    _wire_serve(monkeypatch, names=("present:model",))
    monkeypatch.setattr(cli, "_ollama_pick_best", lambda _names: "bad\nmodel")
    c, buf = make_console()
    assert cli.render_serve(c, None) == 1
    assert "invalid serving identity" in buf.getvalue()


def test_serve_zero_arg_with_engine_refuses(make_console, monkeypatch):
    _wire_serve(monkeypatch)
    c, buf = make_console()
    assert cli.render_serve(c, None, engine="mlx") == 1   # can't pick + honor --engine at once
    assert "pass a model to use --engine" in buf.getvalue()


# self-healing ceiling: an estimated serve that loads cleanly (no spill) records a `measured`
# ceiling so the next serve skips the estimate. Spec 2026-07-04-ara-serve-self-healing-ceiling.
def _wire_save_recorder(monkeypatch):
    saved = []
    monkeypatch.setattr(cli.db, "save_characterization",
                        lambda con, mk, e, m, **kw: saved.append({"engine": e, "model": m, **kw}))
    return saved


def test_serve_estimated_heals_to_measured(make_console, monkeypatch):
    rows = [{"name": "qwen3-0.6b-ara:latest", "context_length": 8192,
             "size": 100, "size_vram": 100}]                # no spill
    _wire_serve(monkeypatch, characterization=None, ps_rows=rows)
    monkeypatch.setattr(cli, "_ollama_estimated_ceiling", lambda m: (8192, "estimated", None))
    saved = _wire_save_recorder(monkeypatch)
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b") == 0
    assert saved == [{"engine": "ollama", "model": "qwen3:0.6b", "safe_context": 8192,
                      "points": [{"context": 8192, "fit": True}], "measured_at": None,
                      "artifact_id": "ollama-manifest-sha256:" + "a" * 64}]
    assert "recorded a measured ceiling" in buf.getvalue()


def test_serve_estimated_does_not_heal_on_spill(make_console, monkeypatch):
    rows = [{"name": "qwen3-0.6b-ara:latest", "context_length": 8192,
             "size": 200, "size_vram": 100}]                # spilled — no clean evidence
    _wire_serve(monkeypatch, characterization=None, ps_rows=rows)
    monkeypatch.setattr(cli, "_ollama_estimated_ceiling", lambda m: (8192, "estimated", None))
    saved = _wire_save_recorder(monkeypatch)
    c, _ = make_console()
    assert cli.render_serve(c, "qwen3:0.6b") == 0
    assert saved == []                                      # never record an uncertain ceiling


def test_serve_measured_does_not_reheal(make_console, monkeypatch):
    _wire_serve(monkeypatch, characterization={"safe_context": 8192}, ps_rows=_SERVE_LOADED)
    saved = _wire_save_recorder(monkeypatch)
    c, _ = make_console()
    assert cli.render_serve(c, "qwen3:0.6b") == 0
    assert saved == []                                      # already measured — nothing to heal


def test_serve_estimated_heal_json_flag(make_console, monkeypatch, capsys):
    rows = [{"name": "qwen3-0.6b-ara:latest", "context_length": 8192,
             "size": 100, "size_vram": 100}]
    _wire_serve(monkeypatch, characterization=None, ps_rows=rows)
    monkeypatch.setattr(cli, "_ollama_estimated_ceiling", lambda m: (8192, "estimated", None))
    _wire_save_recorder(monkeypatch)
    c, _ = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["recorded_measured"] is True


def test_serve_unknown_residency_is_not_reported_or_saved_as_measured(
        make_console, monkeypatch, capsys):
    rows = [{"name": "qwen3-0.6b-ara:latest", "context_length": 8192}]
    _wire_serve(monkeypatch, characterization=None, ps_rows=rows)
    monkeypatch.setattr(cli, "_ollama_estimated_ceiling", lambda _m: (8192, "estimated", None))
    saved = _wire_save_recorder(monkeypatch)
    c, _ = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["spilled"] is None
    assert payload["residency_verified"] is False
    assert payload["recorded_measured"] is False
    assert saved == []


def test_serve_unknown_residency_is_clear_in_text(make_console, monkeypatch):
    rows = [{"name": "qwen3-0.6b-ara:latest", "context_length": 8192,
             "size": True, "size_vram": 100}]
    _wire_serve(monkeypatch, characterization=None, ps_rows=rows)
    monkeypatch.setattr(cli, "_ollama_estimated_ceiling", lambda _m: (8192, "estimated", None))
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b") == 0
    out = buf.getvalue()
    assert "residency" in out and "unknown" in out and "no measured ceiling" in out


def test_ollama_safe_ceiling_includes_ollama_engine(monkeypatch):
    # a ceiling measured through Ollama (engine "ollama") is picked up as measured on later serves.
    artifact = "ollama-manifest-sha256:" + "a" * 64
    chars = {"ollama": {"safe_context": 7777, "artifact_id": artifact}}
    monkeypatch.setattr(cli.db, "get_characterization", lambda con, mk, e, m: chars.get(e))
    assert cli._ollama_safe_ceiling(object(), "mk", "m", artifact) == (
        7777, "measured", None)


# --- stale-ceiling advisory (2026-07-02-ara-ceiling-staleness) --------------- #
def test_stale_ceiling_note_warns_in_text_mode(make_console, monkeypatch):
    monkeypatch.setattr(cli.staleness, "ceiling_is_stale", lambda m, ts: True)
    c, buf = make_console()
    assert cli._stale_ceiling_note(c, "org/m", "2020-01-01T00:00:00+00:00",
                                   as_json=False) is True
    out = buf.getvalue()
    assert "stale" in out and "org/m" in out and "ara characterize org/m" in out


def test_stale_ceiling_note_silent_but_flags_in_json_mode(make_console, monkeypatch):
    monkeypatch.setattr(cli.staleness, "ceiling_is_stale", lambda m, ts: True)
    c, buf = make_console()
    assert cli._stale_ceiling_note(c, "org/m", "x", as_json=True) is True   # flag, no print
    assert buf.getvalue() == ""


def test_stale_ceiling_note_false_and_silent_when_fresh(make_console, monkeypatch):
    monkeypatch.setattr(cli.staleness, "ceiling_is_stale", lambda m, ts: False)
    c, buf = make_console()
    assert cli._stale_ceiling_note(c, "org/m", "x", as_json=False) is False
    assert buf.getvalue() == ""


# serve dispatch (main argv parsing)
def test_main_serve_no_model_dispatches_zero_arg(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    monkeypatch.setattr(cli, "render_serve",
                        lambda c, model, **kw: rec.update(serve={"model": model, **kw}) or 0)
    assert _run_main(monkeypatch, ["serve"]) == 0     # no model → zero-arg select-then-serve
    assert rec["serve"]["model"] is None


def test_main_serve_dispatches_with_ctx_and_name(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    monkeypatch.setattr(cli, "render_serve",
                        lambda c, model, **kw: rec.update(serve={"model": model, **kw}) or 0)
    _run_main(monkeypatch, ["serve", "qwen3:0.6b", "--ctx", "8192", "--name", "srv"])
    assert rec["serve"] == {"model": "qwen3:0.6b", "ctx": 8192, "name": "srv", "engine": None,
                            "assume_yes": False, "as_json": False}


def test_main_serve_equals_forms(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    monkeypatch.setattr(cli, "render_serve",
                        lambda c, model, **kw: rec.update(serve=kw) or 0)
    _run_main(monkeypatch, ["serve", "m", "--ctx=4096", "--name=x"])
    assert rec["serve"]["ctx"] == 4096 and rec["serve"]["name"] == "x"


def test_main_serve_empty_name_is_none(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    monkeypatch.setattr(cli, "render_serve",
                        lambda c, model, **kw: rec.update(serve=kw) or 0)
    _run_main(monkeypatch, ["serve", "m", "--name=", "--ctx", "8"])
    assert rec["serve"]["name"] is None and rec["serve"]["ctx"] == 8


def test_main_serve_ctx_flag_requires_value(monkeypatch, capsys):
    _capture_dispatch(monkeypatch)
    monkeypatch.setattr(cli, "render_serve",
                        lambda c, model, **kw: pytest.fail("renderer must not run"))
    assert _run_main(monkeypatch, ["serve", "m", "--ctx"]) == 2
    assert "Option '--ctx' requires an argument" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1"])
def test_main_serve_rejects_nonpositive_ctx(monkeypatch, capsys, value):
    rec = _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["serve", "m", "--ctx", value]) == 2
    assert "serve" not in rec
    assert "Invalid value" in capsys.readouterr().err


def test_main_serve_name_flag_requires_value(monkeypatch, capsys):
    _capture_dispatch(monkeypatch)
    monkeypatch.setattr(cli, "render_serve",
                        lambda c, model, **kw: pytest.fail("renderer must not run"))
    assert _run_main(monkeypatch, ["serve", "m", "--name"]) == 2
    assert "Option '--name' requires an argument" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# ara benchmark — capability probe + measured tier (Spec 2026-06-28)
# --------------------------------------------------------------------------- #
def _wire_benchmark(monkeypatch, *, ceiling=8000, score=0.75, items=None, engine_key="mlx"):
    """Wire up all dependencies for render_benchmark; returns the captured-save dict."""
    if items is None:
        items = [{"id": 0}, {"id": 1}]
    monkeypatch.setattr(cli.engines, "for_hardware", lambda: engine_key)
    monkeypatch.setattr(cli.engines, "resolve", lambda e: engine_key)
    monkeypatch.setattr(cli.staleness, "artifact_identity", lambda _model: "artifact:test")
    monkeypatch.setattr(cli.staleness, "artifact_size_gb", lambda _model: 1.0)
    # Patch ENGINES so key → backend mapping works without touching real hardware detection.
    orig = dict(cli.engines.ENGINES)
    orig[engine_key] = {**orig.get(engine_key, {}), "backend": "apple"}
    monkeypatch.setattr(cli.engines, "ENGINES", orig)
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli, "engine_status", lambda _backend=None: (True, "test engine"))
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, m: {
                            "safe_context": ceiling, "artifact_id": "artifact:test"
                        } if e == engine_key else None)

    saved = {}

    def fake_save(con, mk, model, uc, *, score, source, engine_key=None, backend=None,
                  base_model=None, benchmark_id=None, sample_size=None, quant=None,
                  refused_n=None, errored_n=None, **kw):
        saved.update(model=model, use_case=uc, score=score, source=source,
                     engine_key=engine_key, sample_size=sample_size, quant=quant,
                     refused_n=refused_n, errored_n=errored_n, **kw)

    monkeypatch.setattr(cli.db, "save_benchmark_result", fake_save)
    monkeypatch.setattr(cli.benchmark, "load_probe", lambda uc: list(items))
    monkeypatch.setattr(cli.benchmark, "prompt_for", lambda uc, it: f"prompt-{it['id']}")
    monkeypatch.setattr(cli.benchmark, "score_probe_set", lambda uc, its, comps: score)

    n = len(items)
    def fake_bench(model, prompts, *, max_context, **kw):
        saved["backend_model"] = model
        saved["bench_kw"] = kw          # capture max_tokens threading
        return {
            "context": max_context,
            "results": [{"prompt_index": i, "completion": f"ans{i}"} for i in range(n)],
        }

    bk = types.SimpleNamespace(benchmark=fake_bench,
                               calibration_model_cached=lambda m: True)  # cached -> skip pre-fetch (#109)
    monkeypatch.setattr(cli, "get_backend", lambda b: bk)
    return saved


def test_render_benchmark_happy_path(make_console, monkeypatch):
    saved = _wire_benchmark(monkeypatch, ceiling=8000, score=0.75)
    c, buf = make_console()
    rc = cli.render_benchmark(c, "org/m", use_case="coding", assume_yes=True, exec_consent=True)
    assert rc == 0
    assert saved["model"] == "org/m"
    assert saved["use_case"] == "coding"
    assert saved["score"] == 0.75
    assert saved["engine_key"] == "mlx"
    assert saved["methodology_id"].startswith("sha256:")
    assert saved["sample_size"] == 2
    out = buf.getvalue()
    assert "coding" in out and "75%" in out and "stored" in out
    assert "ara models recommend --use-case coding" in out


def test_render_benchmark_loads_pinned_gguf_and_records_exact_quant(
        make_console, monkeypatch):
    saved = _wire_benchmark(monkeypatch, engine_key="cpu")
    pinned = "/cache/snapshots/rev/Model-Q4_K_M.gguf"
    monkeypatch.setattr(cli.staleness, "pinned_model_ref",
                        lambda _model, _artifact: pinned)
    c, _ = make_console()
    assert cli.render_benchmark(
        c, "org/model-GGUF", use_case="reasoning", engine="cpu",
        assume_yes=True) == 0
    assert saved["backend_model"] == pinned
    assert saved["quant"] == "q4_k_m"


def test_render_benchmark_refuses_when_authorized_artifact_cannot_be_pinned(
        make_console, monkeypatch):
    _wire_benchmark(monkeypatch, engine_key="cpu")
    monkeypatch.setattr(cli.staleness, "pinned_model_ref", lambda *_a: None)
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", engine="cpu",
                                assume_yes=True) == 1
    assert "cannot pin" in buf.getvalue()


def test_render_benchmark_prefetches_uncached_and_errors_cleanly(make_console, monkeypatch):
    """render_benchmark pre-fetches uncached weights for cuda/mlx (like the GGUF engines, #109);
    a disk shortfall during that fetch is a clean error, not a crash."""
    _wire_benchmark(monkeypatch, ceiling=8000, score=0.5)
    bk = cli.get_backend("apple")
    bk.calibration_model_cached = lambda m: False        # uncached -> pre-fetch fires
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda m: 999.0)
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: 1.0)
    c, buf = make_console()
    rc = cli.render_benchmark(c, "org/m", use_case="extraction", assume_yes=True)
    assert rc == 1
    assert "not enough disk" in buf.getvalue()


def test_render_benchmark_coding_requires_exec_consent(make_console, monkeypatch):
    # Coding runs model-generated code — refuse without explicit consent, NOT bypassable by --yes.
    saved = _wire_benchmark(monkeypatch)
    c, buf = make_console()
    rc = cli.render_benchmark(c, "org/m", use_case="coding", assume_yes=True)
    assert rc == 1
    assert "exec-consent" in buf.getvalue()
    assert "model" not in saved          # nothing executed or stored


def test_render_benchmark_coding_json_mode_does_not_bypass_consent(monkeypatch):
    # --json must NOT silently bypass the code-execution gate.
    _wire_benchmark(monkeypatch)
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_benchmark(c, "org/m", use_case="coding", as_json=True)
    assert rc == 1


def test_render_benchmark_json_flags_stale_ceiling(monkeypatch, capsys):
    # A stored ceiling measured against since-changed cache files is flagged, not silently trusted.
    _wire_benchmark(monkeypatch, ceiling=8000, score=0.5)
    monkeypatch.setattr(cli.staleness, "ceiling_is_stale", lambda m, ts: True)
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_benchmark(c, "org/m", use_case="extraction", as_json=True, assume_yes=True)
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["stale_ceiling"] is True


def test_render_benchmark_lower_ctx_preserves_ceiling_staleness(monkeypatch, capsys):
    _wire_benchmark(monkeypatch, ceiling=8000, score=0.5)
    monkeypatch.setattr(cli.db, "get_characterization", lambda *_a: {
        "safe_context": 8000, "measured_at": "2026-01-02T03:04:05+00:00",
        "artifact_id": "artifact:test"})
    seen = {}
    monkeypatch.setattr(
        cli.staleness, "ceiling_is_stale",
        lambda model, measured_at: seen.update(model=model, measured_at=measured_at) or True,
    )
    c = cli.Console(color=False, stream=sys.stderr)

    assert cli.render_benchmark(c, "org/m", use_case="extraction", ctx=4000,
                                as_json=True, assume_yes=True) == 0

    assert seen == {"model": "org/m", "measured_at": "2026-01-02T03:04:05+00:00"}
    assert json.loads(capsys.readouterr().out)["stale_ceiling"] is True


def test_render_benchmark_rejects_unknown_use_case(make_console, monkeypatch):
    _wire_benchmark(monkeypatch)
    c, buf = make_console()
    rc = cli.render_benchmark(c, "org/m", use_case="chatting", assume_yes=True)
    assert rc == 1
    assert "chatting" in buf.getvalue() and "choose one of" in buf.getvalue()


def test_render_benchmark_refuses_no_ceiling(make_console, monkeypatch):
    _wire_benchmark(monkeypatch, ceiling=8000)
    # Override get_characterization to return None (no ceiling stored).
    monkeypatch.setattr(cli.db, "get_characterization", lambda con, mk, e, m: None)
    c, buf = make_console()
    rc = cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True)
    assert rc == 1
    assert "no measured ceiling" in buf.getvalue() and "ara characterize" in buf.getvalue()


def test_render_benchmark_refuses_ceiling_measured_with_nondefault_config(
        make_console, monkeypatch):
    _wire_benchmark(monkeypatch, ceiling=8000)
    monkeypatch.setattr(cli.db, "get_characterization", lambda con, mk, e, m: {
        "safe_context": 8000, "config": {"kv_quant": "q4_0"}
    })
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 1
    assert "different engine settings" in buf.getvalue()


def test_render_benchmark_explicit_ctx_still_refuses_mismatched_measurement_config(
        make_console, monkeypatch):
    _wire_benchmark(monkeypatch, ceiling=8000)
    monkeypatch.setattr(cli.db, "get_characterization", lambda con, mk, e, m: {
        "safe_context": 8000, "config": {"kv_quant": "q4_0"}
    })
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", engine="mlx", ctx=4000,
                                assume_yes=True) == 1
    assert "different engine settings" in buf.getvalue()


def test_render_benchmark_explicit_ctx_cannot_replace_characterization(
        make_console, monkeypatch):
    _wire_benchmark(monkeypatch, ceiling=8000)
    monkeypatch.setattr(cli.db, "get_characterization", lambda *_a: None)
    c, buf = make_console()

    assert cli.render_benchmark(c, "org/m", use_case="reasoning", ctx=4000,
                                assume_yes=True) == 1

    assert "no measured ceiling" in buf.getvalue()
    assert "ara characterize org/m" in buf.getvalue()


def test_render_benchmark_unsupported_backend(make_console, monkeypatch):
    # A backend without a `benchmark` attr → clear engine-named error.
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "unsupported")
    monkeypatch.setattr(cli.engines, "for_hardware", lambda: "fauxengine")
    orig = dict(cli.engines.ENGINES)
    orig["fauxengine"] = {**orig.get("fauxengine", {}), "backend": "fauxengine"}
    monkeypatch.setattr(cli.engines, "ENGINES", orig)
    bk = types.SimpleNamespace()   # intentionally no .benchmark attr
    monkeypatch.setattr(cli, "get_backend", lambda b: bk)
    c, buf = make_console()
    rc = cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True)
    assert rc == 1
    assert "benchmark isn't supported on the fauxengine engine" in buf.getvalue()


@pytest.mark.parametrize("engine", ["bogus", "ollama"])
def test_render_benchmark_names_invalid_or_unsupported_engine(
        make_console, monkeypatch, engine):
    monkeypatch.setattr(cli.engines, "resolve", lambda _engine: None)
    c, buf = make_console()

    assert cli.render_benchmark(c, "org/m", use_case="reasoning", engine=engine,
                                assume_yes=True) == 1

    out = buf.getvalue()
    assert f"benchmark doesn't support --engine {engine!r}" in out
    assert "none engine" not in out


def test_render_benchmark_reports_when_no_engine_matches(make_console, monkeypatch):
    monkeypatch.setattr(cli.engines, "for_hardware", lambda: None)
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "unsupported")
    c, buf = make_console()

    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 1

    assert "no benchmark-capable engine matches this machine" in buf.getvalue()


@pytest.mark.parametrize("engine", [None, "auto"])
def test_render_benchmark_auto_uses_detected_cpu_backend(
        make_console, monkeypatch, engine):
    saved = _wire_benchmark(monkeypatch, engine_key="cpu")
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cpu")
    monkeypatch.setattr(cli.engines, "for_hardware", lambda: None)
    monkeypatch.setattr(cli.engines, "resolve",
                        lambda value: cli.engines.for_hardware() if value == "auto" else value)
    catalog = dict(cli.engines.ENGINES)
    catalog["cpu"] = {**catalog["cpu"], "backend": "cpu"}
    monkeypatch.setattr(cli.engines, "ENGINES", catalog)
    c, _ = make_console()

    assert cli.render_benchmark(c, "org/m", use_case="reasoning", engine=engine,
                                assume_yes=True) == 0
    assert saved["engine_key"] == "cpu"


def test_render_benchmark_auto_reuses_cpu_ceiling_on_cuda_host(
        make_console, monkeypatch):
    saved = _wire_benchmark(monkeypatch, engine_key="cpu")
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    catalog = dict(cli.engines.ENGINES)
    catalog["cuda"] = {**catalog["cuda"], "backend": "cuda"}
    catalog["cpu"] = {**catalog["cpu"], "backend": "cpu"}
    monkeypatch.setattr(cli.engines, "ENGINES", catalog)
    monkeypatch.setattr(cli.engines, "for_backend",
                        lambda backend: "cuda" if backend == "cuda" else None)
    c, _ = make_console()

    assert cli.render_benchmark(c, "org/m", use_case="reasoning",
                                assume_yes=True) == 0
    assert saved["engine_key"] == "cpu"


def test_render_benchmark_auto_skips_uninstalled_engine_for_installed_fallback(
        make_console, monkeypatch):
    saved = _wire_benchmark(monkeypatch, engine_key="cpu")
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    catalog = dict(cli.engines.ENGINES)
    catalog["cuda"] = {**catalog["cuda"], "backend": "cuda"}
    catalog["cpu"] = {**catalog["cpu"], "backend": "cpu"}
    monkeypatch.setattr(cli.engines, "ENGINES", catalog)
    monkeypatch.setattr(cli.engines, "for_backend",
                        lambda backend: "cuda" if backend == "cuda" else None)
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda _con, _mk, engine, _model: {
                            "safe_context": 16000 if engine == "cuda" else 8000,
                            "artifact_id": "artifact:test",
                        } if engine in {"cuda", "cpu"} else None)
    monkeypatch.setattr(cli, "engine_status",
                        lambda backend=None: (backend == "cpu", f"{backend} engine"))
    c, _ = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning",
                                assume_yes=True) == 0
    assert saved["engine_key"] == "cpu"


def test_render_benchmark_auto_reports_only_uninstalled_characterized_engine(
        make_console, monkeypatch):
    _wire_benchmark(monkeypatch, engine_key="cpu")
    monkeypatch.setattr(cli, "engine_status", lambda _backend=None: (False, "CPU engine"))
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning",
                                assume_yes=True) == 1
    assert "CPU engine isn't installed" in buf.getvalue()
    assert "ara install --engine cpu" in buf.getvalue()


def test_render_benchmark_explicit_uninstalled_engine_is_actionable(
        make_console, monkeypatch):
    _wire_benchmark(monkeypatch, engine_key="cpu")
    monkeypatch.setattr(cli, "engine_status", lambda _backend=None: (False, "CPU engine"))
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", engine="cpu",
                                assume_yes=True) == 1
    assert "CPU engine isn't installed" in buf.getvalue()
    assert "ara install --engine cpu" in buf.getvalue()


def test_render_benchmark_backend_exception_is_clean_json(monkeypatch, capsys):
    _wire_benchmark(monkeypatch, engine_key="cpu")
    bk = cli.get_backend("cpu")
    bk.benchmark = lambda *_a, **_k: (_ for _ in ()).throw(
        cli.EngineEnvError("worker environment missing"))
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", engine="cpu",
                                assume_yes=True, as_json=True) == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": "benchmark failed: worker environment missing"}


def test_render_benchmark_auto_skips_higher_stale_engine_ceiling(
        make_console, monkeypatch):
    saved = _wire_benchmark(monkeypatch, engine_key="cpu")
    monkeypatch.setattr(cli.detect, "backend_name", lambda: "cuda")
    catalog = dict(cli.engines.ENGINES)
    catalog["cuda"] = {**catalog["cuda"], "backend": "cuda"}
    catalog["cpu"] = {**catalog["cpu"], "backend": "cpu"}
    monkeypatch.setattr(cli.engines, "ENGINES", catalog)
    monkeypatch.setattr(cli.engines, "for_backend",
                        lambda backend: "cuda" if backend == "cuda" else None)
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda _con, _mk, engine, _model: {
                            "safe_context": 16000 if engine == "cuda" else 8000,
                            "artifact_id": ("artifact:stale" if engine == "cuda"
                                            else "artifact:test"),
                        } if engine in {"cuda", "cpu"} else None)
    c, _ = make_console()

    assert cli.render_benchmark(c, "org/m", use_case="reasoning",
                                assume_yes=True) == 0
    assert saved["engine_key"] == "cpu"


def test_render_benchmark_auto_ignores_nonbenchmark_candidate_backend(
        make_console, monkeypatch):
    saved = _wire_benchmark(monkeypatch, engine_key="cpu")
    working_backend = cli.get_backend("cpu")
    catalog = dict(cli.engines.ENGINES)
    catalog["faux"] = {"backend": "faux"}
    monkeypatch.setattr(cli.engines, "ENGINES", catalog)
    monkeypatch.setattr(cli, "get_backend",
                        lambda backend: (types.SimpleNamespace()
                                         if backend == "faux" else working_backend))
    c, _ = make_console()

    assert cli.render_benchmark(c, "org/m", use_case="reasoning",
                                assume_yes=True) == 0
    assert saved["engine_key"] == "cpu"


def test_render_benchmark_explicit_engine_requires_its_own_ceiling(
        make_console, monkeypatch):
    _wire_benchmark(monkeypatch, engine_key="cpu")
    monkeypatch.setattr(cli.db, "get_characterization", lambda *_a: None)
    c, buf = make_console()

    assert cli.render_benchmark(c, "org/m", use_case="reasoning", engine="cpu",
                                assume_yes=True) == 1
    assert "no measured ceiling" in buf.getvalue()


def test_render_benchmark_persists_absolute_local_evidence_key(
        make_console, monkeypatch, tmp_path):
    model = tmp_path / "relative:Model-Q4_K_M.gguf"
    model.write_bytes(b"weights")
    monkeypatch.chdir(tmp_path)
    saved = _wire_benchmark(monkeypatch, engine_key="cpu")
    absolute = str(model.resolve())
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda _con, _mk, engine, model_id: {
                            "safe_context": 8000, "artifact_id": "artifact:test"
                        } if engine == "cpu" and model_id == absolute else None)
    c, _ = make_console()

    assert cli.render_benchmark(c, model.name, use_case="reasoning", engine="cpu",
                                assume_yes=True) == 0
    assert saved["model"] == absolute


def test_render_benchmark_verbose_discloses_execution_and_evidence(
        make_console, monkeypatch):
    _wire_benchmark(monkeypatch, ceiling=8000, score=0.75)
    monkeypatch.setattr(cli.db, "get_model", lambda *_a: {"quant": "q4_0"})
    c, buf = make_console(verbose=True)

    assert cli.render_benchmark(c, "org/m", use_case="reasoning", ctx=4000,
                                max_tokens=512, repeat=2, assume_yes=True) == 0

    out = buf.getvalue()
    assert "engine" in out and "mlx (apple)" in out
    assert "probe context" in out and "4000 tokens" in out
    assert "generation cap" in out and "512 tokens" in out
    assert "evidence" in out and "2 prompts × 2 runs" in out
    assert "quant" in out and "q4_0" in out


def test_render_benchmark_verbose_labels_default_generation_cap(
        make_console, monkeypatch):
    _wire_benchmark(monkeypatch, ceiling=8000, score=0.75)
    c, buf = make_console(verbose=True)

    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 0

    assert "backend default" in buf.getvalue()


def test_render_benchmark_verbose_json_includes_execution_evidence(
        monkeypatch, capsys):
    _wire_benchmark(monkeypatch, ceiling=8000, score=0.75)
    c = cli.Console(color=False, stream=sys.stderr, verbose=True)

    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True,
                                as_json=True) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "apple"
    assert payload["probe_context"] == 8000
    assert payload["generation_cap"] == 256
    assert payload["generation_cap_source"] == "backend_default"
    assert payload["total_generations"] == 2


def test_render_benchmark_consent_decline(make_console, monkeypatch):
    _wire_benchmark(monkeypatch)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    c, buf = make_console()
    rc = cli.render_benchmark(c, "org/m", use_case="extraction", assume_yes=False)
    assert rc == 0
    assert "skipped" in buf.getvalue()


def test_render_benchmark_coding_shows_sandbox_warning(make_console, monkeypatch):
    _wire_benchmark(monkeypatch)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    warned = []
    monkeypatch.setattr("builtins.input", lambda prompt="": (warned.append(True), "n")[1])
    c, buf = make_console()
    cli.render_benchmark(c, "org/m", use_case="coding", assume_yes=False, exec_consent=True)
    assert "EXECUTES model-generated Python" in buf.getvalue()
    assert "NOT a security sandbox" in buf.getvalue()   # honest wording, not "sandboxed"


def test_render_benchmark_engine_refused(make_console, monkeypatch):
    _wire_benchmark(monkeypatch)
    # Override the backend to return a pre-load refusal.
    bk = types.SimpleNamespace(
        benchmark=lambda model, prompts, *, max_context, **kw: {
            "context": max_context, "refused": True, "reason": "model too large"
        }
    )
    monkeypatch.setattr(cli, "get_backend", lambda b: bk)
    c, buf = make_console()
    rc = cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True)
    assert rc == 1
    assert "model too large" in buf.getvalue()


def test_render_benchmark_json_output(monkeypatch, capsys):
    _wire_benchmark(monkeypatch, score=0.6)
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_benchmark(c, "org/m", use_case="agentic", assume_yes=True, as_json=True)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "org/m"
    assert payload["use_case"] == "agentic"
    assert payload["score"] == 0.6
    assert payload["stored"] is True


# benchmark dispatch (main argv parsing)
def test_main_benchmark_dispatch(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    monkeypatch.setattr(cli, "render_benchmark",
                        lambda c, model, **kw: rec.update(benchmark={"model": model, **kw}) or 0)
    _run_main(monkeypatch, ["benchmark", "org/m", "--use-case", "coding"])
    assert rec["benchmark"]["model"] == "org/m"
    assert rec["benchmark"]["use_case"] == "coding"
    assert rec["benchmark"]["max_tokens"] is None       # default: backend's own default applies


def test_main_benchmark_parses_max_tokens(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    monkeypatch.setattr(cli, "render_benchmark",
                        lambda c, model, **kw: rec.update(benchmark={"model": model, **kw}) or 0)
    _run_main(monkeypatch, ["benchmark", "org/m", "--use-case", "reasoning", "--max-tokens", "512"])
    assert rec["benchmark"]["max_tokens"] == 512


def test_main_benchmark_parses_max_tokens_equals_form(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    monkeypatch.setattr(cli, "render_benchmark",
                        lambda c, model, **kw: rec.update(benchmark={"model": model, **kw}) or 0)
    _run_main(monkeypatch, ["benchmark", "org/m", "--use-case", "reasoning", "--max-tokens=768"])
    assert rec["benchmark"]["max_tokens"] == 768


def test_render_benchmark_threads_max_tokens_to_backend(make_console, monkeypatch):
    saved = _wire_benchmark(monkeypatch)
    c, _ = make_console()
    cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True, max_tokens=512)
    assert saved["bench_kw"].get("max_tokens") == 512


def test_render_benchmark_omits_max_tokens_when_unset(make_console, monkeypatch):
    saved = _wire_benchmark(monkeypatch)
    c, _ = make_console()
    cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True)
    assert "max_tokens" not in saved["bench_kw"]        # backend default (256) applies


def test_render_benchmark_rejects_nonpositive_max_tokens(make_console, monkeypatch):
    _wire_benchmark(monkeypatch)
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True,
                                max_tokens=0) == 1
    assert "max-tokens" in buf.getvalue()


def test_main_benchmark_missing_use_case_returns_error(monkeypatch):
    _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["benchmark", "org/m"]) == 2


def test_main_benchmark_missing_model_returns_error(monkeypatch):
    _capture_dispatch(monkeypatch)
    assert _run_main(monkeypatch, ["benchmark"]) == 2


# --------------------------------------------------------------------------- #
# Rule #1: --ctx above the measured ceiling is REFUSED, not warned past
# (was an advisory; hardened per the 2026-06-28 audit follow-up).
# Slug: 2026-07-02-rule1-ctx-gate
# --------------------------------------------------------------------------- #

def test_render_benchmark_refuses_ctx_above_measured_ceiling(make_console, monkeypatch):
    """--ctx above the stored ceiling is a hard refusal (rc=1) pointing at re-characterize —
    never exceed the measured memory wall (Rule #1); re-measuring is the sanctioned path up."""
    _wire_benchmark(monkeypatch, ceiling=8000)
    c, buf = make_console()
    rc = cli.render_benchmark(c, "org/m", use_case="reasoning", ctx=16000, assume_yes=True)
    assert rc == 1
    out = buf.getvalue()
    assert "--ctx 16000 exceeds the measured safe ceiling 8000" in out
    assert "Rule #1" in out and "ara characterize org/m" in out


def test_render_benchmark_refuses_ctx_above_ceiling_in_json(make_console, monkeypatch, capsys):
    """The refusal honors --json: {"error": ...}, nothing stored."""
    _wire_benchmark(monkeypatch, ceiling=8000)
    c, _ = make_console()
    rc = cli.render_benchmark(c, "org/m", use_case="reasoning", ctx=16000, assume_yes=True,
                              as_json=True)
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert "exceeds the measured safe ceiling" in payload["error"]


def test_render_benchmark_allows_ctx_at_or_under_measured_ceiling(make_console, monkeypatch):
    """--ctx at or below the stored ceiling proceeds (rc=0) with no gate noise. ctx == ceiling
    is ALLOWED — the measured ceiling itself is safe by definition (a `>=` mutant would refuse)."""
    _wire_benchmark(monkeypatch, ceiling=8000)
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", ctx=8000,
                                assume_yes=True) == 0
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", ctx=4000,
                                assume_yes=True) == 0
    assert "exceeds the measured safe ceiling" not in buf.getvalue()


# --------------------------------------------------------------------------- #
# Fix 1: load-failure refused renders as clean one-line error (no traceback)
# --------------------------------------------------------------------------- #

def test_render_benchmark_load_failure_renders_clean_error(make_console, monkeypatch):
    """Backend refused due to load failure → clean one-line error; no raw traceback."""
    _wire_benchmark(monkeypatch)
    bk = types.SimpleNamespace(
        benchmark=lambda model, prompts, *, max_context, **kw: {
            "context": max_context,
            "refused": True,
            "reason": "failed to load org/m: RuntimeError: mlx version too old",
        }
    )
    monkeypatch.setattr(cli, "get_backend", lambda b: bk)
    c, buf = make_console()
    rc = cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True)
    assert rc == 1
    out = buf.getvalue()
    assert "failed to load" in out
    assert "RuntimeError" in out
    assert "Traceback" not in out
    assert "File " not in out


# --------------------------------------------------------------------------- #
# Fix 2: low_confidence label when sample_size < 100
# --------------------------------------------------------------------------- #

def test_render_benchmark_low_confidence_note_when_small_probe_set(make_console, monkeypatch):
    """n < 100 → low-confidence note in text output; source string carries the flag."""
    items = [{"id": i} for i in range(30)]   # 30 < 100
    _wire_benchmark(monkeypatch, score=0.7, items=items)
    saved_source: list[str] = []

    def capture_source(con, mk, model, uc, *, score, source, **kw):
        saved_source.append(source)

    monkeypatch.setattr(cli.db, "save_benchmark_result", capture_source)
    c, buf = make_console()
    rc = cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True)
    assert rc == 0
    out = buf.getvalue()
    assert "low-confidence" in out
    assert "n=30" in out
    assert saved_source and "low_confidence n=30" in saved_source[0]


def test_render_benchmark_no_low_confidence_note_when_100_items(make_console, monkeypatch):
    """n == 100 (threshold) → no low-confidence annotation."""
    items = [{"id": i} for i in range(100)]
    _wire_benchmark(monkeypatch, score=0.7, items=items)
    c, buf = make_console()
    rc = cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True)
    assert rc == 0
    assert "low-confidence" not in buf.getvalue()


def test_render_benchmark_low_confidence_in_json_output(monkeypatch, capsys):
    """--json with n < 100 → low_confidence: true in payload."""
    items = [{"id": i} for i in range(30)]
    _wire_benchmark(monkeypatch, score=0.7, items=items)
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True, as_json=True)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["low_confidence"] is True


def test_render_benchmark_no_low_confidence_in_json_for_large_set(monkeypatch, capsys):
    """--json with n >= 100 → low_confidence key absent from payload."""
    items = [{"id": i} for i in range(100)]
    _wire_benchmark(monkeypatch, score=0.7, items=items)
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True, as_json=True)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "low_confidence" not in payload


# --------------------------------------------------------------------------- #
# Fix 3: impossible-result advisory for flat 0% / 100% scores
# --------------------------------------------------------------------------- #

def test_render_benchmark_advisory_on_zero_score(make_console, monkeypatch):
    """score == 0.0 → impossible-result advisory emitted."""
    _wire_benchmark(monkeypatch, score=0.0)
    c, buf = make_console()
    cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True)
    assert "flat 0%/100%" in buf.getvalue()


def test_render_benchmark_advisory_on_perfect_score(make_console, monkeypatch):
    """score == 1.0 → impossible-result advisory emitted."""
    _wire_benchmark(monkeypatch, score=1.0)
    c, buf = make_console()
    cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True)
    assert "flat 0%/100%" in buf.getvalue()


def test_render_benchmark_no_advisory_on_mid_score(make_console, monkeypatch):
    """score == 0.5 → no impossible-result advisory."""
    _wire_benchmark(monkeypatch, score=0.5)
    c, buf = make_console()
    cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True)
    assert "flat 0%/100%" not in buf.getvalue()


def test_serve_mlx_refuses_ctx_above_measured_ceiling(make_console, monkeypatch, set_platform):
    """serve --engine mlx --ctx above the stored ceiling is a hard refusal (rc=1) — Rule #1.
    Slug: 2026-07-02-rule1-ctx-gate"""
    set_platform("Darwin", "arm64")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, m: {"safe_context": 8000} if e == "mlx" else None)
    monkeypatch.setattr(cli, "_free_port", lambda: 12399)
    c, buf = make_console()
    rc = cli.render_serve(c, "org/m", engine="mlx", ctx=16000, assume_yes=True)
    assert rc == 1
    out = buf.getvalue()
    assert "--ctx 16000 exceeds the measured safe ceiling 8000" in out
    assert "Rule #1" in out


def test_serve_mlx_allows_ctx_under_measured_ceiling(make_console, monkeypatch, set_platform):
    """serve --engine mlx --ctx at/below the stored ceiling proceeds with no gate noise.
    Slug: 2026-07-02-rule1-ctx-gate"""
    set_platform("Darwin", "arm64")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, m: {"safe_context": 8000} if e == "mlx" else None)
    monkeypatch.setattr(cli, "_free_port", lambda: 12399)

    class _Proc:
        def wait(self): pass

    def _fake_serve(model, *, port, max_context, kv_quant="f16", measured_slope_gb_per_k=None):
        return _Proc(), f"http://127.0.0.1:{port}", max_context

    monkeypatch.setattr("ara.backends.apple.serve", _fake_serve)
    c, buf = make_console()
    rc = cli.render_serve(c, "org/m", engine="mlx", ctx=4000, assume_yes=True)
    assert rc == 0
    assert "exceeds the measured safe ceiling" not in buf.getvalue()


def test_serve_ollama_refuses_ctx_above_measured_ceiling(make_console, monkeypatch):
    """Ollama serve with an explicit --ctx above the measured llama.cpp-class ceiling is a hard
    refusal (rc=1). This path previously never consulted the measurement at all — an explicit
    override sailed straight past the wall. Slug: 2026-07-02-rule1-ctx-gate"""
    _wire_serve(monkeypatch, characterization={"safe_context": 4096})
    c, buf = make_console()
    rc = cli.render_serve(c, "qwen3:0.6b", ctx=8192)
    assert rc == 1
    out = buf.getvalue()
    assert "--ctx 8192 exceeds the measured safe ceiling 4096" in out
    assert "Rule #1" in out


def test_serve_ollama_allows_ctx_under_measured_ceiling(make_console, monkeypatch):
    """Ollama serve with --ctx at/below the measured ceiling proceeds.
    Slug: 2026-07-02-rule1-ctx-gate"""
    rows = [{"name": "qwen3-0.6b-ara:latest", "context_length": 4096,
             "size": 100, "size_vram": 100}]
    _wire_serve(monkeypatch, characterization={"safe_context": 4096}, ps_rows=rows)
    c, buf = make_console()
    assert cli.render_serve(c, "qwen3:0.6b", ctx=4096) == 0
    assert "exceeds the measured safe ceiling" not in buf.getvalue()


# --------------------------------------------------------------------------- #
# recommend: measured tier beats imported (Spec 2026-06-28)
# --------------------------------------------------------------------------- #
def test_recommend_measured_score_beats_imported(make_console, monkeypatch, set_platform):
    # A measured score in db overrides an imported one: Strong has measured 0.9, Weak has imported 0.7.
    # Strong must rank first (measured wins) and the output must show "measured".
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Strong", weights_gb=4.0, max_context=131072),
                     _model_row("org/Weak", weights_gb=4.0, max_context=131072)])
    monkeypatch.setattr(cli.db, "list_benchmark_results",
                        lambda con, mk: [_measured_row(
                            "org/Strong", score=0.9, source="mlx probe=5 (org/Strong)")])
    monkeypatch.setattr(cli.scoring, "load_imported",
                        lambda: {"org/Strong": {"coding": {"score": 0.5, "source": "HumanEval"}},
                                 "org/Weak": {"coding": {"score": 0.7, "source": "HumanEval"}}})
    c, buf = make_console()
    assert cli.render_recommend(c, use_case="coding") == 0
    out = buf.getvalue()
    # Strong's measured 0.9 beats Weak's imported 0.7, so Strong ranks first.
    assert out.index("org/Strong") < out.index("org/Weak")
    assert "measured" in out


# --------------------------------------------------------------------------- #
# Coverage completion — branches not hit by the behaviour tests above (serve/benchmark
# edge paths + the --use-case= argv form). Kept host-independent (no real engine/hardware).
# --------------------------------------------------------------------------- #
def test_recommend_use_case_json_serializes_scores(monkeypatch, set_platform, capsys):
    # --json + --use-case: each score object is flattened to {tier,value,source}; an unscored model
    # is an honest null (covers both arms of the per-rec score ternary in the json branch).
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Strong", weights_gb=4.0, max_context=131072),
                     _model_row("org/Unscored", weights_gb=4.0, max_context=131072)])
    monkeypatch.setattr(cli.scoring, "load_imported",
                        lambda: {"org/Strong": {"coding": {"score": 0.9, "source": "HumanEval"}}})
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_recommend(c, as_json=True, use_case="coding") == 0
    by = {p["model_id"]: p for p in json.loads(capsys.readouterr().out)}
    assert by["org/Strong"]["score"]["value"] == 0.9 and by["org/Strong"]["score"]["source"]
    assert by["org/Unscored"]["score"] is None


# --------------------------------------------------------------------------- #
# recommend: annotate measured scores from partial / low-confidence runs (Rule #3).
# Spec 2026-07-02-benchmark-honesty-persistence.
# --------------------------------------------------------------------------- #
def test_recommend_measured_partial_refusal_and_low_confidence_annotated(make_console, monkeypatch, set_platform):
    # A measured score from a partial run (refused prompts) at a small sample is flagged partial +
    # low-confidence so the ranking discloses its shaky provenance.
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Partial", weights_gb=4.0, max_context=131072)])
    monkeypatch.setattr(cli.db, "list_benchmark_results",
                        lambda con, mk: [_measured_row(
                            "org/Partial", score=0.4, sample_size=30,
                            refused_n=2, errored_n=0)])
    monkeypatch.setattr(cli.scoring, "load_imported", lambda: {})
    _accept_render_test_evidence(monkeypatch)
    c, buf = make_console()
    assert cli.render_recommend(c, use_case="coding") == 0
    out = buf.getvalue()
    assert "[partial: 2 refused]" in out
    assert "[low-confidence n=30]" in out


def test_recommend_measured_errored_partial_no_low_confidence_at_threshold(make_console, monkeypatch, set_platform):
    # errored-only partial annotation; sample_size == 100 (threshold) → NO low-confidence tag.
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Err", weights_gb=4.0, max_context=131072)])
    monkeypatch.setattr(cli.db, "list_benchmark_results",
                        lambda con, mk: [_measured_row(
                            "org/Err", score=0.6, source="cuda probe", sample_size=100,
                            refused_n=0, errored_n=3)])
    monkeypatch.setattr(cli.scoring, "load_imported", lambda: {})
    _accept_render_test_evidence(monkeypatch)
    c, buf = make_console()
    assert cli.render_recommend(c, use_case="coding") == 0
    out = buf.getvalue()
    assert "[partial: 3 errored]" in out
    assert "low-confidence" not in out


def test_recommend_use_case_json_carries_partial_fields(monkeypatch, set_platform, capsys):
    # --json surfaces sample_size + refusal/error counts for measured scores.
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Partial", weights_gb=4.0, max_context=131072)])
    monkeypatch.setattr(cli.db, "list_benchmark_results",
                        lambda con, mk: [_measured_row(
                            "org/Partial", score=0.4, sample_size=30,
                            refused_n=2, errored_n=1)])
    monkeypatch.setattr(cli.scoring, "load_imported", lambda: {})
    _accept_render_test_evidence(monkeypatch)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_recommend(c, as_json=True, use_case="coding") == 0
    payload = json.loads(capsys.readouterr().out)
    sc = payload[0]["score"]
    assert sc["sample_size"] == 30 and sc["refused_n"] == 2 and sc["errored_n"] == 1


def test_recommend_carries_repeat_wide_benchmark_provenance(
        make_console, monkeypatch, set_platform, capsys):
    probe_n = len(cli.benchmark.load_probe("coding"))
    total_n = probe_n * 3
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Repeated", weights_gb=4.0, max_context=131072)])
    monkeypatch.setattr(cli.scoring, "load_imported", lambda: {})
    monkeypatch.setattr(cli.db, "list_benchmark_results", lambda _con, _mk: [_measured_row(
        "org/Repeated", score=0.6, source=f"mlx probe={probe_n}", sample_size=probe_n,
        refused_n=5, errored_n=2,
        probe_context=4096, generation_cap=512, repeat_count=3,
        total_generations=total_n, run_scores_json="[0.55, 0.6, 0.65]",
    )])

    c, buf = make_console(verbose=True)
    assert cli.render_recommend(c, use_case="coding") == 0
    out = buf.getvalue()
    assert f"[partial: 5/{total_n} refused, 2/{total_n} errored]" in out
    assert f"[evidence: {probe_n} prompts × 3 runs; ctx 4096; max 512]" in out

    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_recommend(c, use_case="coding", as_json=True) == 0
    score = json.loads(capsys.readouterr().out)[0]["score"]
    assert score["probe_context"] == 4096
    assert score["generation_cap"] == 512
    assert score["repeat_count"] == 3
    assert score["total_generations"] == total_n
    assert score["run_scores"] == [0.55, 0.6, 0.65]


def test_recommend_joins_exact_gguf_variant_evidence(
        make_console, monkeypatch, set_platform):
    selector = "org/repo:Model-Q4_K_M.gguf"
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row(selector, weights_gb=4.0, quant="q4_k_m")])
    monkeypatch.setattr(cli.scoring, "load_imported", lambda: {})
    monkeypatch.setattr(cli.db, "list_benchmark_results", lambda _con, _mk: [
        _measured_row(selector, score=0.7,
                      sample_size=len(cli.benchmark.load_probe("coding")),
                      refused_n=0, errored_n=0)
    ])
    c, buf = make_console()

    assert cli.render_recommend(c, use_case="coding") == 0
    assert selector in buf.getvalue() and "coding 70% (measured)" in buf.getvalue()


def test_recommend_verbose_handles_legacy_benchmark_provenance(
        make_console, monkeypatch, set_platform):
    probe_n = len(cli.benchmark.load_probe("coding"))
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Legacy", weights_gb=4.0, max_context=131072)])
    monkeypatch.setattr(cli.scoring, "load_imported", lambda: {})
    monkeypatch.setattr(cli.db, "list_benchmark_results", lambda _con, _mk: [_measured_row(
        "org/Legacy", score=0.6, source=f"mlx probe={probe_n}", sample_size=probe_n,
        refused_n=0, errored_n=0,
        probe_context=None, generation_cap=None, repeat_count=None,
        total_generations=None, run_scores_json=None,
    )])

    c, buf = make_console(verbose=True)
    assert cli.render_recommend(c, use_case="coding") == 0
    assert f"[evidence: {probe_n} prompts]" in buf.getvalue()


@pytest.mark.parametrize("stored", ["not-json", "{}", "[0.5, true]", "[1.5, 0.5]",
                                     "[0.5]", "[NaN, 0.5]", "[0.1, 0.1]"])
def test_recommend_discloses_invalid_stored_run_scores(
        make_console, monkeypatch, set_platform, capsys, stored):
    probe_n = len(cli.benchmark.load_probe("coding"))
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Corrupt", weights_gb=4.0, max_context=131072)])
    monkeypatch.setattr(cli.scoring, "load_imported", lambda: {})
    monkeypatch.setattr(cli.db, "list_benchmark_results", lambda _con, _mk: [_measured_row(
        "org/Corrupt", score=0.6, source=f"mlx probe={probe_n}", sample_size=probe_n,
        refused_n=0, errored_n=0, probe_context=4096, generation_cap=512,
        repeat_count=2, total_generations=probe_n * 2, run_scores_json=stored)])

    c, buf = make_console()
    assert cli.render_recommend(c, use_case="coding") == 0
    assert "unknown" in buf.getvalue()
    assert "invalid stored benchmark evidence" in buf.getvalue()

    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_recommend(c, use_case="coding", as_json=True) == 0
    rec = json.loads(capsys.readouterr().out)[0]
    assert rec["score"] is None
    assert rec["evidence_warning"] == "invalid stored benchmark evidence"


@pytest.mark.parametrize("field,value", [
    ("score", "bad"), ("score", float("nan")), ("score", 1.1),
    ("sample_size", 0), ("sample_size", None), ("refused_n", None), ("refused_n", -1),
    ("refused_n", 329), ("errored_n", True),
    ("errored_n", None),
    ("probe_context", 0), ("probe_context", None),
    ("generation_cap", -1), ("repeat_count", 0),
    ("total_generations", 327), ("source", ""), ("measured_at", "not-a-date"),
    ("measured_at", 123), ("tier", "imported"), ("benchmark_id", "rag"),
    ("engine_key", "bogus"), ("engine_key", None),
    ("backend", "bogus"), ("backend", None), ("base_model", "bogus"),
    ("canonical_model_id", "other/model"), ("artifact_id", ""),
    ("max_score", None), ("max_score", 2.0),
    ("quant", "q8_0"),
])
def test_recommend_downgrades_other_invalid_stored_evidence(
        make_console, monkeypatch, set_platform, field, value):
    probe_n = len(cli.benchmark.load_probe("coding"))
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Corrupt", weights_gb=4.0, max_context=131072)])
    monkeypatch.setattr(cli.scoring, "load_imported", lambda: {})
    row = _measured_row(
        "org/Corrupt", score=0.6, source=f"mlx probe={probe_n}", sample_size=probe_n,
        refused_n=0, errored_n=0, probe_context=4096, generation_cap=512,
        repeat_count=2, total_generations=probe_n * 2, run_scores_json="[0.6, 0.6]")
    row[field] = value
    monkeypatch.setattr(cli.db, "list_benchmark_results", lambda _con, _mk: [row])
    monkeypatch.setattr(cli.staleness, "fit_is_stale", lambda *_a: False)
    c, buf = make_console()

    assert cli.render_recommend(c, use_case="coding") == 0
    assert "unknown" in buf.getvalue()
    assert "invalid stored benchmark evidence" in buf.getvalue()


def test_recommend_downgrades_stale_benchmark_for_changed_cached_artifact(
        make_console, monkeypatch, set_platform, capsys):
    probe_n = len(cli.benchmark.load_probe("coding"))
    _wire_recommend(monkeypatch, set_platform,
                    [_model_row("org/Changed", weights_gb=4.0, max_context=131072)])
    monkeypatch.setattr(cli.scoring, "load_imported", lambda: {})
    monkeypatch.setattr(cli.db, "list_benchmark_results", lambda _con, _mk: [_measured_row(
        "org/Changed", score=0.9, source=f"mlx probe={probe_n}", sample_size=probe_n,
        refused_n=0, errored_n=0, probe_context=4096, generation_cap=512,
        repeat_count=1, total_generations=probe_n, run_scores_json="[0.9]")])
    monkeypatch.setattr(cli.staleness, "artifact_identity", lambda _model: "artifact:changed")

    c, buf = make_console()
    assert cli.render_recommend(c, use_case="coding") == 0
    assert "unknown" in buf.getvalue()
    assert "cached model changed since benchmark" in buf.getvalue()

    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_recommend(c, use_case="coding", as_json=True) == 0
    rec = json.loads(capsys.readouterr().out)[0]
    assert rec["score"] is None
    assert rec["evidence_warning"] == "cached model changed since benchmark"


def test_render_benchmark_rejects_nonpositive_ctx(make_console, monkeypatch):
    _wire_benchmark(monkeypatch)
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", ctx=0, assume_yes=True) == 1
    assert "positive" in buf.getvalue()


def test_render_benchmark_ctx_below_ceiling_no_advisory(make_console, monkeypatch):
    # explicit --ctx at/under the measured ceiling → no advisory warning (the not-taken branch)
    _wire_benchmark(monkeypatch, ceiling=8000)
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", ctx=4000, assume_yes=True) == 0
    assert "exceeds the measured safe ceiling" not in buf.getvalue()


def test_render_benchmark_all_prompts_refused_stores_nothing(make_console, monkeypatch):
    # Every prompt refused by per-prompt governance is NOT a 0% measurement — refuse to store it.
    _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}])
    bk = types.SimpleNamespace(benchmark=lambda model, prompts, *, max_context, **kw: {
        "context": max_context,
        "results": [{"prompt_index": i, "refused": True, "reason": "ctx too small"}
                    for i in range(2)]})
    monkeypatch.setattr(cli, "get_backend", lambda b: bk)
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 1
    assert "every prompt was refused" in buf.getvalue()


def test_render_benchmark_some_prompts_refused_warns(make_console, monkeypatch):
    # A partial refusal depresses the score and emits a note — but still stores a measurement.
    _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}], score=0.5)
    bk = types.SimpleNamespace(benchmark=lambda model, prompts, *, max_context, **kw: {
        "context": max_context,
        "results": [{"prompt_index": 0, "refused": True, "reason": "x"},
                    {"prompt_index": 1, "completion": "ans"}]})
    monkeypatch.setattr(cli, "get_backend", lambda b: bk)
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 0
    assert "were refused by" in buf.getvalue()


def test_render_benchmark_refuses_out_of_range_prompt_index(make_console, monkeypatch):
    # A malformed worker response is not a capability measurement and must never be stored.
    saved = _wire_benchmark(monkeypatch, items=[{"id": 0}], score=1.0)
    bk = types.SimpleNamespace(benchmark=lambda model, prompts, *, max_context, **kw: {
        "context": max_context,
        "results": [{"prompt_index": 99, "completion": "out-of-range"}]})
    monkeypatch.setattr(cli, "get_backend", lambda b: bk)
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 1
    assert "invalid benchmark result" in buf.getvalue()
    assert "model" not in saved


def test_render_benchmark_refuses_missing_prompt_result(make_console, monkeypatch):
    saved = _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}], score=0.5)
    _bench_backend(monkeypatch, [{"prompt_index": 0, "completion": "ans"}])
    c, buf = make_console()

    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 1

    assert "invalid benchmark result" in buf.getvalue()
    assert "one result per prompt" in buf.getvalue()
    assert "model" not in saved


@pytest.mark.parametrize("reported_context", [8001, None, True])
def test_render_benchmark_refuses_wrong_reported_context(
        make_console, monkeypatch, reported_context):
    saved = _wire_benchmark(monkeypatch, items=[{"id": 0}], score=1.0)
    bk = types.SimpleNamespace(benchmark=lambda model, prompts, *, max_context, **kw: {
        "context": reported_context,
        "results": [{"prompt_index": 0, "completion": "ans"}]})
    monkeypatch.setattr(cli, "get_backend", lambda b: bk)
    c, buf = make_console()

    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 1

    assert "reported context" in buf.getvalue()
    assert "model" not in saved


@pytest.mark.parametrize("entry", [
    None,
    {"prompt_index": 0},
    {"prompt_index": 0, "completion": None},
    {"prompt_index": 0, "completion": "answer", "error": "also failed"},
    {"prompt_index": True, "completion": "answer"},
    {"prompt_index": 0, "refused": False},
    {"prompt_index": 0, "error": None},
])
def test_render_benchmark_refuses_malformed_prompt_result(
        make_console, monkeypatch, entry):
    saved = _wire_benchmark(monkeypatch, items=[{"id": 0}], score=1.0)
    _bench_backend(monkeypatch, [entry])
    c, buf = make_console()

    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 1

    assert "invalid benchmark result" in buf.getvalue()
    assert "model" not in saved


@pytest.mark.parametrize("response", [
    None,
    {"context": 8000, "results": {}},
])
def test_render_benchmark_refuses_malformed_worker_response(
        make_console, monkeypatch, response):
    saved = _wire_benchmark(monkeypatch, items=[{"id": 0}], score=1.0)
    bk = types.SimpleNamespace(benchmark=lambda *_a, **_k: response)
    monkeypatch.setattr(cli, "get_backend", lambda _b: bk)
    c, buf = make_console()

    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 1

    assert "invalid benchmark result" in buf.getvalue()
    assert "model" not in saved


def test_render_benchmark_refuses_duplicate_prompt_index(make_console, monkeypatch):
    saved = _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}], score=1.0)
    _bench_backend(monkeypatch, [
        {"prompt_index": 0, "completion": "a"},
        {"prompt_index": 0, "completion": "b"},
    ])
    c, buf = make_console()

    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 1

    assert "prompt indexes must be unique" in buf.getvalue()
    assert "model" not in saved


def test_render_benchmark_refuses_empty_probe_set(make_console, monkeypatch):
    saved = _wire_benchmark(monkeypatch, items=[], score=0.0)
    c, buf = make_console()

    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 1

    assert "probe set is empty" in buf.getvalue()
    assert "model" not in saved


# --------------------------------------------------------------------------- #
# Benchmark honesty persistence — refused/errored counts + quant (Rule #3).
# Spec 2026-07-02-benchmark-honesty-persistence.
# --------------------------------------------------------------------------- #

def _bench_backend(monkeypatch, results):
    """Override the wired backend with one that returns the given per-prompt result list."""
    bk = types.SimpleNamespace(
        benchmark=lambda model, prompts, *, max_context, **kw: {
            "context": max_context, "results": results},
        calibration_model_cached=lambda m: True)
    monkeypatch.setattr(cli, "get_backend", lambda b: bk)


def test_render_benchmark_clean_run_stores_zero_counts(make_console, monkeypatch):
    # A clean full run persists refused_n=0 / errored_n=0 (measured clean, NOT legacy NULL).
    saved = _wire_benchmark(monkeypatch, score=0.75)
    c, _ = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 0
    assert saved["refused_n"] == 0 and saved["errored_n"] == 0


def test_render_benchmark_persists_structured_execution_provenance(
        make_console, monkeypatch):
    saved = _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}])
    seq = iter([0.4, 0.8])
    monkeypatch.setattr(cli.benchmark, "score_probe_set", lambda *_a: next(seq))
    c, _ = make_console()

    assert cli.render_benchmark(c, "org/m", use_case="reasoning", ctx=4000,
                                max_tokens=512, repeat=2, assume_yes=True) == 0

    assert saved["probe_context"] == 4000
    assert saved["generation_cap"] == 512
    assert saved["repeat_count"] == 2
    assert saved["total_generations"] == 4
    assert saved["run_scores"] == [0.4, 0.8]
    assert saved["max_score"] == 1.0
    assert saved["artifact_id"] == "artifact:test"
    assert saved["canonical_model_id"] == "org/m"


def test_render_benchmark_refuses_unknown_or_changing_artifact(
        make_console, monkeypatch):
    saved = _wire_benchmark(monkeypatch)
    monkeypatch.setattr(cli.staleness, "artifact_identity", lambda _model: None)
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 1
    assert "differs from its measured ceiling" in buf.getvalue()
    assert "model" not in saved

    identities = iter(["artifact:test", "artifact:test", "artifact:b"])
    monkeypatch.setattr(cli.staleness, "artifact_identity", lambda _model: next(identities))
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 1
    assert "changed during the benchmark" in buf.getvalue()
    assert "model" not in saved


@pytest.mark.parametrize("engine", [None, "mlx"])
def test_render_benchmark_refuses_ceiling_without_artifact_authority(
        make_console, monkeypatch, engine):
    _wire_benchmark(monkeypatch)
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda *_a: {"safe_context": 8000, "artifact_id": None})
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", engine=engine,
                                assume_yes=True) == 1
    assert "not bound to an exact artifact" in buf.getvalue()


def test_render_benchmark_refuses_artifact_changed_after_auto_selection(
        make_console, monkeypatch):
    saved = _wire_benchmark(monkeypatch)
    identities = iter(["artifact:test", "artifact:changed"])
    monkeypatch.setattr(cli.staleness, "artifact_identity", lambda _model: next(identities))
    c, buf = make_console()

    assert cli.render_benchmark(c, "org/m", use_case="reasoning",
                                assume_yes=True) == 1
    assert "differs from its measured ceiling" in buf.getvalue()
    assert "model" not in saved


def test_render_benchmark_catalogs_exact_gguf_variant(make_console, monkeypatch):
    saved = _wire_benchmark(monkeypatch, engine_key="cpu")
    captured = {}
    monkeypatch.setattr(cli.catalog, "remember_variant",
                        lambda con, model, canonical, **kw:
                        captured.update(model=model, canonical=canonical, **kw))
    c, _ = make_console()
    selector = "org/repo:Model-Q4_K_M.gguf"

    assert cli.render_benchmark(c, selector, use_case="reasoning", engine="cpu",
                                assume_yes=True) == 0
    assert captured == {"model": selector, "canonical": "org/repo",
                        "quant": "q4_k_m", "weights_gb": 1.0}
    assert saved["canonical_model_id"] == "org/repo"
    assert saved["quant"] == "q4_k_m"


def test_render_benchmark_persists_effective_default_generation_cap(
        make_console, monkeypatch):
    saved = _wire_benchmark(monkeypatch)
    c, _ = make_console()

    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 0

    assert saved["generation_cap"] == 256


def test_render_benchmark_partial_refusal_stores_counts_and_annotates(make_console, monkeypatch):
    # A partial refusal stores refused_n and appends the partial suffix to the score line.
    saved = _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}], score=0.5)
    _bench_backend(monkeypatch, [{"prompt_index": 0, "refused": True, "reason": "x"},
                                 {"prompt_index": 1, "completion": "ans"}])
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 0
    assert saved["refused_n"] == 1 and saved["errored_n"] == 0
    assert "(partial: 1 refused)" in buf.getvalue()


def test_render_benchmark_partial_json_is_one_clean_document(monkeypatch, capsys):
    _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}], score=0.5)
    _bench_backend(monkeypatch, [{"prompt_index": 0, "refused": True, "reason": "x"},
                                 {"prompt_index": 1, "completion": "ans"}])
    c = cli.Console(color=False, stream=sys.stdout)

    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True,
                                as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["refused"] == 1
    assert payload["errored"] == 0


def test_render_benchmark_some_prompts_errored_warns_and_depresses(make_console, monkeypatch):
    # A mid-generation engine exception is captured per-prompt, scored 0, warned about, and the
    # errored count is persisted — the depressed score is disclosed, never hidden (Rule #3).
    saved = _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}], score=0.5)
    _bench_backend(monkeypatch, [{"prompt_index": 0, "error": "CUDA OOM mid-generation"},
                                 {"prompt_index": 1, "completion": "ans"}])
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 0
    out = buf.getvalue()
    assert "1/2 prompts errored" in out and "engine exception" in out
    assert saved["errored_n"] == 1 and saved["refused_n"] == 0
    assert "(partial: 1 errored)" in out


def test_render_benchmark_all_prompts_errored_stores_nothing(make_console, monkeypatch):
    # Every prompt erroring is NOT a 0% measurement — refuse to store (Rule #3).
    _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}])
    _bench_backend(monkeypatch, [{"prompt_index": i, "error": "boom"} for i in range(2)])
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 1
    assert "every prompt was refused or errored" in buf.getvalue()


def test_render_benchmark_all_refused_or_errored_mixed_stores_nothing(make_console, monkeypatch):
    # A mix that leaves no successful completion (some refused, some errored) also stores nothing.
    _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}])
    _bench_backend(monkeypatch, [{"prompt_index": 0, "refused": True, "reason": "x"},
                                 {"prompt_index": 1, "error": "boom"}])
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 1
    assert "no measurement taken" in buf.getvalue()


def test_render_benchmark_quant_from_catalog(make_console, monkeypatch):
    # Quant is taken from the models catalog when known (the actual measured quant).
    saved = _wire_benchmark(monkeypatch, score=0.7)
    monkeypatch.setattr(cli.db, "get_model", lambda con, m: {"quant": "q4_0"})
    c, _ = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 0
    assert saved["quant"] == "q4_0"


def test_render_benchmark_quant_falls_back_to_model_id_token(make_console, monkeypatch):
    # No catalog quant → derive it from the model id's quant token.
    saved = _wire_benchmark(monkeypatch, score=0.7)
    monkeypatch.setattr(cli.db, "get_model", lambda con, m: None)
    c, _ = make_console()
    assert cli.render_benchmark(c, "org/Model-4bit", use_case="reasoning", assume_yes=True) == 0
    assert saved["quant"] == "4bit"


def test_render_benchmark_quant_none_when_unknown(make_console, monkeypatch):
    # Neither catalog nor model id reveals a quant → None (honest unknown, not a guess).
    saved = _wire_benchmark(monkeypatch, score=0.7)
    monkeypatch.setattr(cli.db, "get_model", lambda con, m: None)
    c, _ = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True) == 0
    assert saved["quant"] is None


def test_render_benchmark_json_includes_partial_counts_and_quant(monkeypatch, capsys):
    # --json surfaces the refusal/error counts (when nonzero) and the known quant.
    _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}], score=0.5)
    monkeypatch.setattr(cli.db, "get_model", lambda con, m: {"quant": "q4_0"})
    _bench_backend(monkeypatch, [{"prompt_index": 0, "error": "boom"},
                                 {"prompt_index": 1, "completion": "ans"}])
    c = cli.Console(color=False, stream=sys.stderr)
    rc = cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True, as_json=True)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["refused"] == 0 and payload["errored"] == 1
    assert payload["quant"] == "q4_0"


# --------------------------------------------------------------------------- #
# ara benchmark --repeat N: variance bands — store the MEAN + LO–HI band across
# N runs instead of a single lucky roll (pass^k spirit; never report one roll as
# THE number). Slug: 2026-07-02-benchmark-repeat-passk
# --------------------------------------------------------------------------- #

def test_render_benchmark_repeat_one_text_identical_to_default(make_console, monkeypatch):
    # repeat=1 must reproduce today's single-run text output byte-for-byte (no band, no "mean of").
    _wire_benchmark(monkeypatch, score=0.75)
    c1, b1 = make_console()
    assert cli.render_benchmark(c1, "org/m", use_case="reasoning", assume_yes=True) == 0
    c2, b2 = make_console()
    assert cli.render_benchmark(c2, "org/m", use_case="reasoning", assume_yes=True, repeat=1) == 0
    assert b1.getvalue() == b2.getvalue()
    assert "mean of" not in b1.getvalue() and "band" not in b1.getvalue()


def test_render_benchmark_repeat_one_json_identical_to_default(monkeypatch, capsys):
    # repeat=1 --json must be byte-identical to today (no runs/band/repeat keys).
    _wire_benchmark(monkeypatch, score=0.6)
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_benchmark(c, "org/m", use_case="agentic", assume_yes=True, as_json=True) == 0
    default = capsys.readouterr().out
    assert cli.render_benchmark(c, "org/m", use_case="agentic", assume_yes=True,
                                as_json=True, repeat=1) == 0
    with_flag = capsys.readouterr().out
    assert default == with_flag
    payload = json.loads(with_flag)
    assert "runs" not in payload and "band" not in payload and "repeat" not in payload


def test_render_benchmark_repeat_runs_backend_n_times(make_console, monkeypatch):
    # N=3 loads + benchmarks the model three times (N separate model loads — acceptable v1).
    _wire_benchmark(monkeypatch, score=0.5)
    calls = {"n": 0}

    def counting_bench(model, prompts, *, max_context, **kw):
        calls["n"] += 1
        return {"context": max_context,
                "results": [{"prompt_index": i, "completion": f"a{i}"}
                            for i in range(len(prompts))]}

    bk = types.SimpleNamespace(benchmark=counting_bench, calibration_model_cached=lambda m: True)
    monkeypatch.setattr(cli, "get_backend", lambda b: bk)
    c, _ = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True, repeat=3) == 0
    assert calls["n"] == 3


def test_render_benchmark_repeat_mean_and_band_text(make_console, monkeypatch):
    # Three runs scoring 40/60/80% → stored MEAN 60%, band 40–80% shown; NOT the determinism note.
    saved = _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}])
    seq = iter([0.4, 0.6, 0.8])
    monkeypatch.setattr(cli.benchmark, "score_probe_set", lambda uc, its, comps: next(seq))
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True, repeat=3) == 0
    out = buf.getvalue()
    assert "60% measured here" in out
    assert "mean of 3 runs" in out
    assert "band 40–80%" in out
    assert "scored identically" not in out           # a real spread is NOT the determinism note
    assert saved["score"] == pytest.approx(0.6)       # the mean, not any single roll
    assert "repeat=3 band=40-80" in saved["source"]   # band stamped into the source string


def test_render_benchmark_repeat_mean_and_band_json(monkeypatch, capsys):
    # --json adds runs/band/repeat; the score field is the mean.
    _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}])
    seq = iter([0.4, 0.6, 0.8])
    monkeypatch.setattr(cli.benchmark, "score_probe_set", lambda uc, its, comps: next(seq))
    c = cli.Console(color=False, stream=sys.stderr)
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True,
                                as_json=True, repeat=3) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runs"] == [0.4, 0.6, 0.8]
    assert payload["band"] == [0.4, 0.8]
    assert payload["repeat"] == 3
    assert payload["score"] == pytest.approx(0.6)


def test_render_benchmark_repeat_identical_scores_emits_determinism_note(make_console, monkeypatch):
    # Zero variance under greedy decoding is determinism, not measured robustness — say so honestly.
    _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}], score=0.75)
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True, repeat=3) == 0
    out = buf.getvalue()
    assert "all 3 runs scored identically" in out
    assert "deterministic" in out and "not evidence of stability" in out
    assert "band 75–75%" in out                        # the (equal) LO–HI still renders


def test_render_benchmark_repeat_sums_refused_across_runs(make_console, monkeypatch):
    # Per-run refusals are summed into the stored total; denominator = total generations attempted.
    saved = _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}], score=0.5)
    bk = types.SimpleNamespace(
        benchmark=lambda model, prompts, *, max_context, **kw: {
            "context": max_context,
            "results": [{"prompt_index": 0, "refused": True, "reason": "x"},
                        {"prompt_index": 1, "completion": "ans"}]},
        calibration_model_cached=lambda m: True)
    monkeypatch.setattr(cli, "get_backend", lambda b: bk)
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True, repeat=3) == 0
    assert saved["refused_n"] == 3 and saved["errored_n"] == 0   # 1 refusal × 3 runs
    assert "3/6 prompts were refused" in buf.getvalue()          # 2 prompts × 3 runs = 6 attempts


def test_render_benchmark_repeat_sums_errored_across_runs(make_console, monkeypatch):
    # Per-run engine errors are summed across runs too.
    saved = _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}], score=0.5)
    bk = types.SimpleNamespace(
        benchmark=lambda model, prompts, *, max_context, **kw: {
            "context": max_context,
            "results": [{"prompt_index": 0, "error": "boom"},
                        {"prompt_index": 1, "completion": "ans"}]},
        calibration_model_cached=lambda m: True)
    monkeypatch.setattr(cli, "get_backend", lambda b: bk)
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True, repeat=2) == 0
    assert saved["errored_n"] == 2 and saved["refused_n"] == 0
    assert "2/4 prompts errored" in buf.getvalue()


def test_render_benchmark_repeat_all_failed_across_all_runs_stores_nothing(make_console, monkeypatch):
    # Every generation across every run refused/errored → no measurement; refuse to store (Rule #3).
    _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}])
    bk = types.SimpleNamespace(
        benchmark=lambda model, prompts, *, max_context, **kw: {
            "context": max_context,
            "results": [{"prompt_index": i, "refused": True, "reason": "x"} for i in range(2)]},
        calibration_model_cached=lambda m: True)
    monkeypatch.setattr(cli, "get_backend", lambda b: bk)
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True, repeat=2) == 1
    assert "no measurement taken" in buf.getvalue()


def test_render_benchmark_repeat_whole_run_refusal_on_later_run_aborts(make_console, monkeypatch):
    # A whole-run refusal on ANY run (here the 2nd) aborts — no partial band from a failed load.
    _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}], score=0.5)
    calls = {"n": 0}

    def flaky(model, prompts, *, max_context, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            return {"context": max_context, "refused": True, "reason": "OOM on reload"}
        return {"context": max_context,
                "results": [{"prompt_index": i, "completion": f"a{i}"}
                            for i in range(len(prompts))]}

    bk = types.SimpleNamespace(benchmark=flaky, calibration_model_cached=lambda m: True)
    monkeypatch.setattr(cli, "get_backend", lambda b: bk)
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True, repeat=3) == 1
    assert "OOM on reload" in buf.getvalue()
    assert calls["n"] == 2                    # aborted at the failing run; never ran the 3rd


def test_render_benchmark_rejects_repeat_zero(make_console, monkeypatch):
    _wire_benchmark(monkeypatch)
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True, repeat=0) == 1
    assert "--repeat must be a positive integer" in buf.getvalue()


def test_render_benchmark_rejects_repeat_negative(make_console, monkeypatch):
    _wire_benchmark(monkeypatch)
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=True, repeat=-2) == 1
    assert "positive integer" in buf.getvalue()


def test_render_benchmark_confirm_prompt_mentions_runs_when_repeated(make_console, monkeypatch):
    # The interactive confirm should disclose the run count when N > 1.
    _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}])
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    seen = {}
    monkeypatch.setattr("builtins.input", lambda prompt="": seen.update(prompt=prompt) or "n")
    c, _ = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=False, repeat=3) == 0
    assert "2 prompts × 3 runs" in seen["prompt"]


def test_render_benchmark_confirm_prompt_omits_runs_when_single(make_console, monkeypatch):
    # N == 1 keeps today's "(N prompts)" wording — no "× runs" suffix.
    _wire_benchmark(monkeypatch, items=[{"id": 0}, {"id": 1}])
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    seen = {}
    monkeypatch.setattr("builtins.input", lambda prompt="": seen.update(prompt=prompt) or "n")
    c, _ = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning", assume_yes=False, repeat=1) == 0
    assert "2 prompts)" in seen["prompt"] and "runs" not in seen["prompt"]


def test_main_benchmark_default_repeat_is_one(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    monkeypatch.setattr(cli, "render_benchmark",
                        lambda c, model, **kw: rec.update(benchmark={"model": model, **kw}) or 0)
    _run_main(monkeypatch, ["benchmark", "org/m", "--use-case", "coding"])
    assert rec["benchmark"]["repeat"] == 1


def test_main_benchmark_parses_repeat(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    monkeypatch.setattr(cli, "render_benchmark",
                        lambda c, model, **kw: rec.update(benchmark={"model": model, **kw}) or 0)
    _run_main(monkeypatch, ["benchmark", "org/m", "--use-case", "reasoning", "--repeat", "3"])
    assert rec["benchmark"]["repeat"] == 3


def test_main_benchmark_parses_repeat_equals_form(monkeypatch):
    rec = _capture_dispatch(monkeypatch)
    monkeypatch.setattr(cli, "render_benchmark",
                        lambda c, model, **kw: rec.update(benchmark={"model": model, **kw}) or 0)
    _run_main(monkeypatch, ["benchmark", "org/m", "--use-case", "reasoning", "--repeat=5"])
    assert rec["benchmark"]["repeat"] == 5


def test_main_benchmark_noninteger_repeat_is_click_error(monkeypatch, capsys):
    _capture_dispatch(monkeypatch)
    monkeypatch.setattr(cli, "render_benchmark",
                        lambda c, model, **kw: pytest.fail("renderer must not run"))
    assert _run_main(monkeypatch, ["benchmark", "org/m", "--use-case", "reasoning",
                                   "--repeat", "lots"]) == 2
    assert "Invalid value for '--repeat'" in capsys.readouterr().err


# --- serve: MLX (mlx) governed-server path ---
def _wire_serve_mlx(monkeypatch, set_platform, *, ceiling=8000, serve=None):
    """Route `serve --engine mlx` to the MLX path with the db + port + apple.serve seams stubbed."""
    set_platform("Darwin", "arm64")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.db, "get_characterization",
                        lambda con, mk, e, m: {"safe_context": ceiling} if e == "mlx" else None)
    monkeypatch.setattr(cli, "_free_port", lambda: 12399)
    if serve is not None:
        monkeypatch.setattr("ara.backends.apple.serve", serve)


def test_serve_help_explains_runtimes_governance_and_lifecycle(capsys):
    assert cli.main(["serve", "--help"]) == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "Ollama" in out and "MLX" in out
    assert "ollama, mlx, or auto" in out
    assert "measured or estimated safe bound" in out
    assert "foreground" in out
    assert "--name NAME" in out and "Ollama" in out


def test_serve_mlx_rejects_nonpositive_ctx(make_console, monkeypatch, set_platform):
    _wire_serve_mlx(monkeypatch, set_platform)
    c, buf = make_console()
    assert cli.render_serve(c, "org/m", engine="mlx", ctx=0, assume_yes=True) == 1
    assert "positive" in buf.getvalue()


def test_serve_mlx_confirm_declined_skips(make_console, monkeypatch, set_platform):
    _wire_serve_mlx(monkeypatch, set_platform)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(cli, "_confirm", lambda q: False)
    c, buf = make_console()
    assert cli.render_serve(c, "org/m", engine="mlx") == 0     # not --yes → prompts → declined
    assert "skipped" in buf.getvalue()


def test_serve_mlx_handles_serve_failure(make_console, monkeypatch, set_platform):
    def boom(model, *, port, max_context, kv_quant="f16", measured_slope_gb_per_k=None):
        raise RuntimeError("gate refused")
    _wire_serve_mlx(monkeypatch, set_platform, serve=boom)
    c, buf = make_console()
    assert cli.render_serve(c, "org/m", engine="mlx", assume_yes=True) == 1
    out = buf.getvalue()
    assert "couldn't start the MLX server" in out and "gate refused" in out


def test_serve_mlx_rejects_custom_ollama_name(make_console, monkeypatch, set_platform):
    _wire_serve_mlx(monkeypatch, set_platform,
                    serve=lambda *_a, **_k: pytest.fail("MLX server started"))
    c, buf = make_console()
    assert cli.render_serve(c, "org/m", engine="mlx", name="custom") == 1
    assert "--name" in buf.getvalue() and "Ollama" in buf.getvalue()


@pytest.mark.parametrize(("url", "served_ctx"), [
    ("http://127.0.0.1:12399", 4096),
    (7, 8000),
    ("http://evil.invalid:12399", 8000),
])
def test_serve_mlx_rejects_malformed_ready_contract_and_terminates_child(
        make_console, monkeypatch, set_platform, url, served_ctx):
    state = {"terminated": False, "waited": False}

    class _Proc:
        def terminate(self):
            state["terminated"] = True

        def wait(self):
            state["waited"] = True
            return 0

    _wire_serve_mlx(
        monkeypatch, set_platform,
        serve=lambda *_a, **_k: (_Proc(), url, served_ctx))
    c, buf = make_console()
    assert cli.render_serve(c, "org/m", engine="mlx", assume_yes=True) == 1
    assert "governance" in buf.getvalue().lower() or "invalid" in buf.getvalue().lower()
    assert state == {"terminated": True, "waited": True}


def test_serve_mlx_nonzero_child_exit_is_failure(make_console, monkeypatch, set_platform):
    class _Proc:
        def wait(self):
            return 7

    _wire_serve_mlx(
        monkeypatch, set_platform,
        serve=lambda *_a, **_k: (_Proc(), "http://127.0.0.1:12399", 8000))
    c, buf = make_console()
    assert cli.render_serve(c, "org/m", engine="mlx", assume_yes=True) == 1
    assert "exited" in buf.getvalue() and "7" in buf.getvalue()


def test_serve_mlx_wait_exception_terminates_child(make_console, monkeypatch, set_platform):
    state = {"terminated": False}

    class _Proc:
        def terminate(self):
            state["terminated"] = True

        def wait(self):
            raise RuntimeError("server wait failed")

    _wire_serve_mlx(
        monkeypatch, set_platform,
        serve=lambda *_a, **_k: (_Proc(), "http://127.0.0.1:12399", 8000))
    c, _ = make_console()
    with pytest.raises(RuntimeError, match="server wait failed"):
        cli.render_serve(c, "org/m", engine="mlx", assume_yes=True)
    assert state["terminated"] is True


def test_serve_mlx_keyboard_interrupt_is_clean_stop(make_console, monkeypatch, set_platform):
    state = {"terminated": False, "waited": 0}

    class _Proc:
        def terminate(self):
            state["terminated"] = True

        def wait(self):
            state["waited"] += 1
            if state["waited"] == 1:
                raise KeyboardInterrupt
            return 0

    _wire_serve_mlx(
        monkeypatch, set_platform,
        serve=lambda *_a, **_k: (_Proc(), "http://127.0.0.1:12399", 8000))
    c, _ = make_console()
    assert cli.render_serve(c, "org/m", engine="mlx", assume_yes=True) == 0
    assert state == {"terminated": True, "waited": 2}


def test_serve_mlx_json_output(make_console, monkeypatch, set_platform, capsys):
    class _Proc:
        def wait(self):
            pass
    _wire_serve_mlx(monkeypatch, set_platform, ceiling=8000,
                    serve=lambda model, *, port, max_context, kv_quant="f16", measured_slope_gb_per_k=None:
                    (_Proc(), f"http://127.0.0.1:{port}", max_context))
    c, _ = make_console()
    assert cli.render_serve(c, "org/m", engine="mlx", assume_yes=True, as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runtime"] == "mlx" and payload["served_context"] == 8000
    assert payload["endpoint"].endswith("/v1") and payload["ceiling_source"] == "measured"


def test_serve_mlx_sigterm_handler_terminates_child(make_console, monkeypatch, set_platform):
    import signal as _signal
    terminated = {}

    class _Proc:
        def terminate(self):
            terminated["yes"] = True

        def wait(self):
            pass

    _wire_serve_mlx(monkeypatch, set_platform,
                    serve=lambda model, *, port, max_context, kv_quant="f16", measured_slope_gb_per_k=None:
                    (_Proc(), f"http://127.0.0.1:{port}", max_context))
    captured = {}

    def fake_signal(sig, handler):
        if sig == _signal.SIGTERM and callable(handler):
            captured["handler"] = handler
        return _signal.SIG_DFL
    monkeypatch.setattr(_signal, "signal", fake_signal)
    c, _ = make_console()
    assert cli.render_serve(c, "org/m", engine="mlx", assume_yes=True) == 0
    # the installed SIGTERM handler must terminate the child, then exit (no orphaned server)
    with pytest.raises(SystemExit):
        captured["handler"](_signal.SIGTERM, None)
    assert terminated.get("yes") is True


def test_main_benchmark_use_case_equals_form(monkeypatch):
    # the `--use-case=coding` (joined) argv form parses identically to the spaced form.
    captured = {}
    monkeypatch.setattr(cli, "render_benchmark",
                        lambda *a, **k: captured.update(k) or 0)
    monkeypatch.setattr(cli, "_resolve_want", lambda *a, **k: None)
    _run_main(monkeypatch, ["benchmark", "org/m", "--use-case=coding", "--yes"])
    assert captured.get("use_case") == "coding"


def test_render_benchmark_confirm_accepted_proceeds(make_console, monkeypatch):
    # interactive (tty), no --yes: accepting the prompt proceeds to run the benchmark.
    _wire_benchmark(monkeypatch, score=0.6)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(cli, "_confirm", lambda q: True)
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/m", use_case="reasoning") == 0
    assert "measured here" in buf.getvalue()


def test_serve_mlx_confirm_accepted_proceeds(make_console, monkeypatch, set_platform):
    class _Proc:
        def wait(self):
            pass
    _wire_serve_mlx(monkeypatch, set_platform,
                    serve=lambda model, *, port, max_context, kv_quant="f16", measured_slope_gb_per_k=None:
                    (_Proc(), f"http://127.0.0.1:{port}", max_context))
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(cli, "_confirm", lambda q: True)
    c, buf = make_console()
    assert cli.render_serve(c, "org/m", engine="mlx") == 0      # tty + accepted → serves
    assert "serving" in buf.getvalue()


def test_free_port_returns_an_available_port():
    # exercised directly: every serve test stubs it, so the real bind/close needs its own test.
    p = cli._free_port()
    assert isinstance(p, int) and 1024 <= p <= 65535
