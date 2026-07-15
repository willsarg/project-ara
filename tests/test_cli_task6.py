# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Task 6 cross-surface contracts: runtime recon, generated help, and public docs."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from ara import cli, mlx, pythons
from ara.detect import Accelerator, Machine, Runtime
from ara.mlx import MlxInterpreter


ROOT = Path(__file__).resolve().parents[1]


def _commands(help_text: str) -> set[str]:
    block = help_text.split("Commands:\n", 1)[1]
    return {line.split()[0] for line in block.splitlines() if line.startswith("  ")}


def _machine() -> Machine:
    return Machine(
        system="Darwin", os_version="macOS 15.0", chip="Apple M4 Pro", arch="arm64",
        cpu_physical=12, cpu_logical=12, cpu_features=["NEON"], python_version="3.12.8",
        ram_total_gb=48.0, ram_available_gb=20.0, swap_gb=2.0,
        accel=Accelerator("apple", "Apple M4 Pro GPU", None, "Metal", cores=16),
        disk_free_gb=500.0,
        runtimes=[Runtime("MLX", True, "0.18", kind="engine", accels=("apple",), usable=True)],
        framework_python="/usr/bin/python3", backend="apple", engine="mlx", engine_ready=True,
    )


def test_detect_runtime_json_reports_common_inventory_on_linux_without_mlx_detail(
        monkeypatch, capsys):
    machine = replace(
        _machine(), system="Linux", backend="cpu", engine="cpu", engine_ready=True,
        runtimes=[
            Runtime("llama.cpp", True, "b5200", kind="engine"),
            Runtime("PyTorch", True, "2.7.1", kind="framework"),
        ],
    )
    monkeypatch.setattr(cli.detect, "machine", lambda: machine)

    def apple_only_probe():
        raise AssertionError("non-Apple runtime recon must not run MLX-specific probes")

    monkeypatch.setattr(cli.mlx, "scan", apple_only_probe)
    monkeypatch.setattr(cli.mlx, "lmstudio_mlx_runtimes", apple_only_probe)
    monkeypatch.setattr(cli.mlx, "mlx_community_model_count", apple_only_probe)
    monkeypatch.setattr(cli, "get_backend", apple_only_probe)
    monkeypatch.setattr(cli.engines, "install", apple_only_probe)

    assert cli.main(["detect", "--runtime", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "system": "Linux",
        "backend_selection": {"name": "cpu", "source": "observed hardware selection"},
        "ara_engine": {
            "name": "cpu", "ready": True, "source": "ARA isolated engine environment",
        },
        "user_environment": {
            "source": "user environment",
            "runtimes": [
                {"name": "llama.cpp", "present": True, "version": "b5200", "kind": "engine",
                 "accels": [], "usable": None, "serving": None},
                {"name": "PyTorch", "present": True, "version": "2.7.1", "kind": "framework",
                 "accels": [], "usable": None, "serving": None},
            ],
        },
    }


def test_detect_runtime_text_reports_observed_common_inventory_on_windows(
        monkeypatch, capsys):
    machine = replace(
        _machine(), system="Windows", backend="cuda", engine="cuda", engine_ready=False,
        runtimes=[
            Runtime("vLLM", False, kind="engine", accels=("nvidia",), usable=True),
            Runtime("Ollama", True, "0.9.0", kind="engine", serving=False),
            Runtime("transformers", True, "4.53.0", kind="framework"),
        ],
    )
    monkeypatch.setattr(cli.detect, "machine", lambda: machine)
    monkeypatch.setattr(cli.pythons, "discover", lambda: [])

    assert cli.main(["detect", "--runtime"]) == 0
    out = capsys.readouterr().out
    assert "RUNTIME" in out
    assert "backend selection  cuda" in out and "observed hardware selection" in out
    assert "ARA ISOLATED ENGINE ENVIRONMENT" in out
    assert "USER ENVIRONMENT" in out
    assert "Ollama 0.9.0" in out and "not serving" in out
    assert "transformers 4.53.0" in out
    assert "vLLM" in out and "not found" in out
    assert "MLX ECOSYSTEM" not in out


def test_detect_runtime_adds_observed_mlx_detail_only_on_apple(
        monkeypatch, capsys):
    machine = _machine()
    monkeypatch.setattr(cli.detect, "machine", lambda: machine)
    monkeypatch.setattr(cli.mlx, "scan", lambda: [
        MlxInterpreter("/opt/venv/bin/python", "venv", "3.12.8", packages={"mlx": "0.26"}),
    ])
    monkeypatch.setattr(cli.mlx, "lmstudio_mlx_runtimes", lambda: ["1.2.3"])
    monkeypatch.setattr(cli.mlx, "mlx_community_model_count", lambda: 4)

    assert cli.main(["detect", "--runtime", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend_selection"] == {
        "name": "apple", "source": "observed hardware selection",
    }
    assert payload["ara_engine"] == {
        "name": "mlx", "ready": True, "source": "ARA isolated engine environment",
    }
    assert payload["mlx_ecosystem"] == {
        "source": "read-only user ecosystem probes",
        "gpu": {"name": "Apple M4 Pro GPU", "cores": 16},
        "mlx_community_models": 4,
        "lmstudio_mlx_runtimes": ["1.2.3"],
        "interpreters": [{
            "path": "/opt/venv/bin/python", "origin": "venv", "version": "3.12.8",
            "packages": {"mlx": "0.26"},
        }],
    }


def test_detect_runtime_explains_ready_isolated_mlx_and_absent_user_runtime(
        monkeypatch, capsys):
    machine = replace(
        _machine(),
        engine="mlx", engine_ready=True,
        runtimes=[Runtime("MLX", False, kind="engine", accels=("apple",), usable=True)],
        framework_python=None,
    )
    monkeypatch.setattr(cli.detect, "machine", lambda: machine)
    monkeypatch.setattr(cli.pythons, "discover", lambda: [])
    monkeypatch.setattr(cli.mlx, "scan", lambda: [])
    monkeypatch.setattr(cli.mlx, "lmstudio_mlx_runtimes", lambda: [])
    monkeypatch.setattr(cli.mlx, "mlx_community_model_count", lambda: 0)

    assert cli.main(["detect", "--runtime"]) == 0
    out = capsys.readouterr().out
    assert "ARA ISOLATED ENGINE ENVIRONMENT" in out
    assert "mlx" in out and "ready" in out
    assert "USER ENVIRONMENT" in out
    assert "MLX" in out and "not found" in out
    assert "no separate user Python found" in out
    assert "read-only user ecosystem probes" in out


def test_runtime_user_fallback_excludes_active_ara_venv_torch(monkeypatch, capsys):
    venv = "/opt/ara-env"
    monkeypatch.setenv("VIRTUAL_ENV", venv)
    monkeypatch.setenv("PATH", os.path.join(venv, "bin"))
    monkeypatch.setattr(pythons, "_known_patterns", lambda: [])
    monkeypatch.setattr(
        pythons, "_candidates",
        lambda: {"/real/base-python": {os.path.join(venv, "bin", "python3")}},
    )

    def forbidden_probe(real):
        raise AssertionError(f"excluded ARA interpreter was probed: {real}")

    monkeypatch.setattr(pythons, "_probe", forbidden_probe)
    machine = replace(
        _machine(), system="Linux", backend="cpu", engine="cpu",
        framework_python=None,
        runtimes=[Runtime("PyTorch", False, kind="framework")],
    )
    monkeypatch.setattr(cli.detect, "machine", lambda: machine)

    assert cli.main(["detect", "--runtime"]) == 0
    out = capsys.readouterr().out
    assert "no separate user Python found" in out
    assert "torch" not in out.lower()


def test_runtime_mlx_json_excludes_active_ara_venv_interpreter(monkeypatch, capsys):
    venv = "/opt/ara-env"
    monkeypatch.setenv("VIRTUAL_ENV", venv)
    monkeypatch.setenv("PATH", os.path.join(venv, "bin"))
    monkeypatch.setattr(
        pythons, "_candidates",
        lambda: {"/real/base-python": {os.path.join(venv, "bin", "python3")}},
    )

    def forbidden_probe(real):
        raise AssertionError(f"excluded ARA interpreter was probed: {real}")

    monkeypatch.setattr(mlx, "_probe", forbidden_probe)
    monkeypatch.setattr(mlx, "mlx_community_model_count", lambda: 0)
    monkeypatch.setattr(mlx, "lmstudio_mlx_runtimes", lambda: [])
    monkeypatch.setattr(cli.detect, "machine", lambda: replace(_machine(), framework_python=None))

    assert cli.main(["detect", "--runtime", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mlx_ecosystem"]["interpreters"] == []


@pytest.mark.parametrize(("cores", "lmstudio", "packages", "anchors"), [
    (16, ["1.2.3"], {"mlx": "0.26"}, ("16-core Metal", "MLX runtime 1.2.3", "mlx")),
    (None, [], {}, ("Metal", "not found", "none found")),
])
def test_detect_runtime_text_renders_truthful_apple_mlx_detail(
        monkeypatch, capsys, cores, lmstudio, packages, anchors):
    machine = replace(
        _machine(),
        accel=Accelerator("apple", "Apple M4 Pro GPU", None, "Metal", cores=cores),
    )
    monkeypatch.setattr(cli.detect, "machine", lambda: machine)
    monkeypatch.setattr(cli.pythons, "discover", lambda: [])
    monkeypatch.setattr(cli.mlx, "scan", lambda: [
        MlxInterpreter("/opt/venv/bin/python", "venv", "3.12.8", packages=packages),
    ])
    monkeypatch.setattr(cli.mlx, "lmstudio_mlx_runtimes", lambda: lmstudio)
    monkeypatch.setattr(cli.mlx, "mlx_community_model_count", lambda: 4)

    assert cli.main(["detect", "--runtime"]) == 0
    out = capsys.readouterr().out
    assert "MLX ECOSYSTEM" in out
    assert all(anchor in out for anchor in anchors)


def test_generated_help_matches_frozen_visible_tree_and_examples(capsys):
    expected = {
        (): {"benchmark", "characterize", "detect", "doctor", "hf", "install", "models",
             "node", "profile", "run", "serve", "status", "uninstall"},
        ("models",): {"recommend", "search", "show"},
        ("node",): {"enroll", "install", "run", "start", "status", "stop", "uninstall"},
        ("hf",): {"login", "logout", "status"},
    }
    for path, commands in expected.items():
        assert cli.main([*path, "--help"]) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        assert _commands(captured.out) == commands

    examples = {
        ("detect",): ("Usage: ara detect [OPTIONS]", "ara detect --runtime --json"),
        ("models", "search"): (
            "Usage: ara models search [OPTIONS] QUERY...", 'ara models search "small vision model" --json'),
        ("node", "enroll"): (
            "Usage: ara node enroll [OPTIONS] [SERVER_URL]", "ara node enroll https://ara.example --token TOKEN"),
        ("run",): ("Usage: ara run [OPTIONS] MODEL PROMPT...", 'ara run org/model "Explain this" --json'),
    }
    for path, anchors in examples.items():
        assert cli.main([*path, "--help"]) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        assert all(anchor in captured.out for anchor in anchors)


def test_click_honors_node_end_of_options_safety_for_prompt_text(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(
        cli, "render_run",
        lambda c, model, **kwargs: seen.update(model=model, **kwargs) or 0,
    )
    assert cli.main(["run", "--yes", "--json", "--", "org/model", "--prompt-like-text"]) == 0
    assert seen["model"] == "org/model"
    assert seen["prompt"] == "--prompt-like-text"
    assert seen["as_json"] is True and seen["assume_yes"] is True
    assert capsys.readouterr().err == ""


def test_public_docs_and_search_guidance_use_canonical_surface_and_uv_only():
    docs = "\n".join((ROOT / name).read_text(encoding="utf-8")
                     for name in ("README.md", "AGENTS.md", "CONTRIBUTING.md"))
    assert "python -m ara.cli" not in docs
    assert "pip install huggingface_hub" not in docs
    assert "ara status` | Live view of AI/ML processes" not in docs
    assert "`ara models search" in docs
    assert "`ara detect --runtime" in docs
    assert "macOS" in docs and "CPU + MLX" in docs
    assert "Windows" in docs and "RTX 2070" in docs
    assert "Linux" in docs and "CPU" in docs

    source = (ROOT / "ara" / "cli.py").read_text(encoding="utf-8")
    assert "pip install " not in source
    assert "uv run ara" in source


def test_live_cli_guidance_never_points_to_hidden_aliases():
    source = "\n".join(path.read_text(encoding="utf-8")
                       for path in (ROOT / "ara").rglob("*.py"))
    for stale in (
        "ara python", "ara apps", "ara recommend", "see ara models)",
    ):
        assert stale not in source


def test_mlx_view_guidance_never_emits_models_show_without_required_model():
    views = ROOT / "ara" / "_engine_packages" / "mlx" / "ara_engine_mlx" / "views"
    calibrate = (views / "calibrate.py").read_text(encoding="utf-8")
    characterize = (views / "characterize.py").read_text(encoding="utf-8")
    assert '("ara models show",' not in calibrate
    assert '("ara models show MODEL",' in calibrate
    assert '("ara models show",' not in characterize
    assert '(f"ara models show {model}",' in characterize
