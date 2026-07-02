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

import pytest

from ara import cli
from ara.detect import Accelerator, Machine, ModelStore, Runtime


def _machine() -> Machine:
    return Machine(
        system="Darwin", os_version="macOS 15.0", chip="Apple M4 Pro", arch="arm64",
        cpu_physical=12, cpu_logical=12, cpu_features=["NEON"],
        python_version="3.12.8", ram_total_gb=48.0, ram_available_gb=20.0, swap_gb=2.0,
        accel=Accelerator("apple", "Apple M4 Pro GPU", None, "Metal", cores=16),
        disk_free_gb=500.0,
        runtimes=[Runtime("MLX", True, "0.18", kind="engine", accels=("apple",), usable=True)],
        framework_python="/usr/bin/python3",
        model_stores=[ModelStore("HF cache", True, 3, 12.0)],
        hf_token=True, power="AC power", backend="apple", engine="wmx-suite",
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
    monkeypatch.setattr(cli.status, "scan", lambda: [])
    monkeypatch.setattr(cli.status, "scan_apps", lambda: [])
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
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "mkey")
    monkeypatch.setattr(cli.calibration, "get_calibration", lambda con, key: None)
    monkeypatch.setattr(cli.hub, "search", lambda q: [{"id": "org/m"}])
    monkeypatch.setattr(cli, "engine_status", lambda b=None: (True, "wmx-suite"))
    monkeypatch.setattr(cli, "get_backend", lambda b=None: _FakeBackend())
    monkeypatch.setattr(cli.engines, "install",
                        lambda key, **kw: cli.engines.InstallResult(key, "installed"))
    monkeypatch.setattr(cli.engines, "uninstall",
                        lambda key: cli.engines.InstallResult(key, "removed"))


# (argv, expected JSON type, an anchor key that must be present | None for a list)
CONTRACT = [
    (["detect", "--json"], dict, "backend"),
    (["status", "--json"], dict, "workloads"),
    (["python", "--json"], list, None),
    (["apps", "--json"], list, None),
    (["mlx", "--json"], dict, "apple_silicon"),
    (["models", "--json"], list, None),
    (["models", "org/m", "--json"], dict, "model_id"),
    (["search", "smol", "--json"], list, None),
    (["characterize", "org/m", "--json"], dict, "safe_context"),
    (["profile", "--json"], dict, "device"),
    (["install", "--engine", "wmx", "--json"], dict, "status"),
    (["uninstall", "--engine", "wmx", "--json"], dict, "status"),
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


def test_unknown_command_is_nonzero_and_not_json(mocked_world, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["ara", "frobnicate"])
    assert cli.main() == 1
    with pytest.raises(json.JSONDecodeError):
        json.loads(capsys.readouterr().out.strip())


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
        monkeypatch.setattr(cli, "engine_status", lambda b=None: (False, "wmx-suite"))
    elif setup == "undescribable":
        monkeypatch.setattr(cli.catalog, "describe", lambda model_id: None)
    # bad_engine needs no stub: resolve_engine raises UnknownEngine for 'bogus'
    monkeypatch.setattr("sys.argv", ["ara", *argv])
    assert cli.main() == 1
    payload = _extract_json(capsys.readouterr().out)
    assert "error" in payload
