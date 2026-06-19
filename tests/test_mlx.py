"""mlx.py — MLX ecosystem discovery (libraries by modality + Apple readiness)."""
from __future__ import annotations

import types

from ara import mlx, pythons
from ara.mlx import MlxInterpreter


# --------------------------------------------------------------------------- #
# MlxInterpreter.caution (delegates to pythons.caution_for)
# --------------------------------------------------------------------------- #
def test_caution_delegates_to_pythons():
    assert MlxInterpreter("p", "Homebrew", "3.12", externally_managed=True).caution \
        == pythons._CAUTION["Homebrew"]
    assert MlxInterpreter("p", "macOS system", "3.9").caution == pythons._CAUTION["macOS system"]
    assert MlxInterpreter("p", "pyenv", "3.12", externally_managed=False).caution is None


# --------------------------------------------------------------------------- #
# _run / _probe
# --------------------------------------------------------------------------- #
def test_run_none_on_failure():
    assert mlx._run(["definitely-not-a-real-binary-xyz"]) is None


def test_probe_parses(monkeypatch):
    payload = '{"v": "3.12.4", "pkgs": {"mlx": "0.18", "mlx-lm": "0.20"}, "em": true}'
    monkeypatch.setattr(mlx, "_run", lambda cmd, timeout=8: "noise\n" + payload)
    ver, pkgs, em = mlx._probe("/usr/bin/python3")
    assert ver == "3.12.4" and pkgs == {"mlx": "0.18", "mlx-lm": "0.20"} and em is True


def test_probe_blank_on_failure(monkeypatch):
    monkeypatch.setattr(mlx, "_run", lambda cmd, timeout=8: None)
    assert mlx._probe("/x") == (None, {}, False)


def test_probe_blank_on_bad_json(monkeypatch):
    monkeypatch.setattr(mlx, "_run", lambda cmd, timeout=8: "not json")
    assert mlx._probe("/x") == (None, {}, False)


# --------------------------------------------------------------------------- #
# scan() — keep only interpreters with MLX packages, richest first
# --------------------------------------------------------------------------- #
def test_scan_filters_and_sorts(monkeypatch):
    interps = [
        types.SimpleNamespace(real="/a", path="/a", origin="venv", version="3.12", externally_managed=False),
        types.SimpleNamespace(real="/b", path="/b", origin="Homebrew", version="3.11", externally_managed=True),
        types.SimpleNamespace(real="/c", path="/c", origin="venv", version="3.12", externally_managed=False),
    ]
    monkeypatch.setattr(mlx.pythons, "discover", lambda probe=False: interps)
    probes = {
        "/a": ("3.12.0", {"mlx": "0.18"}, False),
        "/b": ("3.11.0", {"mlx": "0.18", "mlx-lm": "0.20", "mlx-vlm": "0.1"}, True),
        "/c": ("3.12.0", {}, False),   # no MLX packages → dropped
    }
    monkeypatch.setattr(mlx, "_probe", lambda real: probes[real])

    out = mlx.scan()
    assert [m.path for m in out] == ["/b", "/a"]   # richest (3 pkgs) first; /c dropped
    assert out[0].externally_managed is True
    assert out[0].version == "3.11.0"              # probed version wins over discover's


# --------------------------------------------------------------------------- #
# mlx_community_model_count
# --------------------------------------------------------------------------- #
def test_mlx_community_model_count(fake_home):
    hub = fake_home / ".cache" / "huggingface" / "hub"
    (hub / "models--mlx-community--SmolLM").mkdir(parents=True)
    (hub / "models--mlx-community--Qwen").mkdir(parents=True)
    (hub / "models--meta--Llama").mkdir(parents=True)   # not mlx-community → ignored
    assert mlx.mlx_community_model_count() == 2


def test_mlx_community_model_count_no_cache(fake_home):
    assert mlx.mlx_community_model_count() == 0


# --------------------------------------------------------------------------- #
# lmstudio_mlx_runtimes
# --------------------------------------------------------------------------- #
def test_lmstudio_mlx_runtimes_sorted_newest_first(fake_home):
    base = fake_home / ".lmstudio" / "extensions" / "backends"
    (base / "mlx-llm-0.2.0").mkdir(parents=True)
    (base / "mlx-llm-0.3.1").mkdir(parents=True)
    (base / "llama-cpp-1.0").mkdir(parents=True)   # not mlx-llm → ignored
    assert mlx.lmstudio_mlx_runtimes() == ["0.3.1", "0.2.0"]


def test_lmstudio_mlx_runtimes_none(fake_home):
    assert mlx.lmstudio_mlx_runtimes() == []
