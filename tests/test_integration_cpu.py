# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Real-engine integration smoke — the CPU/llama.cpp path, end to end, for real.

Everything else in the suite mocks the engine seam; this is the one tier that actually creates
an isolated engine env, installs llama-cpp-python, pulls a tiny GGUF, and runs ARA's real
methodology against it. It exists because the bug that 100% coverage missed lived exactly at the
mock↔reality boundary (a real preflight ``max_context`` flowing through the driver).

Opt-in and kept out of the fast unit loop:

    pytest -m integration --no-cov

CPU/llama.cpp is the target because it needs no Apple/NVIDIA hardware, so this runs in generic
CI. It skips cleanly when uv or the network/model aren't available.
"""
from __future__ import annotations

import json
import shutil

import pytest

pytestmark = pytest.mark.integration

# A tiny instruct GGUF (worker auto-picks the smallest quant, ~100 MB; 8192-token window). This
# tier checks the real end-to-end path — worker preflight reads the GGUF's real max_context, real
# RSS measurements flow through the real driver. The tiny-window edge case itself (window below
# the 2nd schedule rung) is covered exhaustively at unit level in test_methodology_matrix.py.
SMOKE_MODEL = "bartowski/SmolLM2-135M-Instruct-GGUF"


@pytest.fixture
def cpu_engine(tmp_path, monkeypatch, capsys):
    """Install the cpu engine into a throwaway env; yield the cpu backend. Skips if it can't."""
    monkeypatch.setenv("ARA_ENGINES_DIR", str(tmp_path / "engines"))
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    from ara import cli
    from ara.backends import cpu

    rc = cli.main(["install", "--engine", "cpu", "--json"])
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    if rc != 0:
        pytest.skip(f"cpu engine install unavailable: {result}")
    assert result["key"] == "cpu" and result["status"] in ("installed", "already", "refreshed")
    return cpu


def test_safe_limits_reads_a_real_ram_wall(cpu_engine):
    m = cpu_engine.safe_limits()
    assert m["total_gb"] > 0 and m["wall_gb"] > 0
    # safe_budget = wall - the EFFECTIVE margin, which is min(cap, max(0.5, 0.1*RAM)) — not the raw
    # cap. On a small-RAM box (~11 GB) 10% < the 2 GB cap, so asserting against
    # DEFAULT_MARGIN_GB is wrong; assert against the margin the limits dict actually reports.
    assert 0.5 <= m["margin_gb"] <= cpu_engine.DEFAULT_MARGIN_GB
    assert m["safe_budget_gb"] == pytest.approx(m["wall_gb"] - m["margin_gb"])
    assert m["calibrated"] is True and m["overhead_gb"] is None   # exact wall, nothing to calibrate


def test_characterize_finds_a_sane_ceiling(cpu_engine):
    # Pre-fetch the GGUF so we can tell "offline" apart from "bug". The worker folds fetch
    # failures into a None ceiling (no exception), so skipping on None would HIDE the exact
    # bug this tier targets (a fitting model reported as None). Fetch first: a failure here is
    # a legitimate skip; once the model is cached, a None ceiling is a real failure.
    from ara.workers import cpu_llama
    try:
        cpu_llama._resolve_gguf(SMOKE_MODEL)     # downloads the smallest quant; raises offline
    except Exception as e:
        pytest.skip(f"{SMOKE_MODEL} unavailable (offline?): {e}")

    r = cpu_engine.characterize(SMOKE_MODEL)
    # A 135M model fits its whole trained window on any real machine → a positive, window-bound
    # ceiling. With the model cached, None would mean the methodology regressed — fail, don't skip.
    assert r["safe_context"] is not None, "cached model must yield a ceiling, got None"
    assert isinstance(r["safe_context"], int) and r["safe_context"] >= 2048
    assert r["binding"] == "context_window"
    assert r["points"] and all(p["mem_gb"] > 0 for p in r["points"])


def test_generate_produces_a_real_completion(cpu_engine):
    # The governed run, end to end: load the real model under a safe ceiling and generate. Same
    # fetch-first discipline as characterize — an offline fetch is a skip, not a failure.
    from ara.workers import cpu_llama
    try:
        cpu_llama._resolve_gguf(SMOKE_MODEL)
    except Exception as e:
        pytest.skip(f"{SMOKE_MODEL} unavailable (offline?): {e}")

    out = cpu_engine.generate(SMOKE_MODEL, "The capital of France is",
                              max_context=2048, max_tokens=8)
    assert "refused" not in out, f"governed generate was refused: {out.get('reason')}"
    assert out["context"] == 2048                       # KV capped at the governed ceiling
    assert isinstance(out["completion"], str) and out["completion"].strip()


def test_public_cli_reaches_a_governed_first_cpu_completion(
        tmp_path, monkeypatch, capsys):
    """The README's CPU path works through public commands and persists authoritative evidence."""
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")

    # Fetch independently before invoking the workflow. A network/model failure is environmental
    # and may skip; after this succeeds, every CLI or evidence failure is an ARA regression.
    from ara.workers import cpu_llama
    try:
        cpu_llama._resolve_gguf(SMOKE_MODEL)
    except Exception as exc:
        pytest.skip(f"{SMOKE_MODEL} unavailable (offline?): {exc}")

    monkeypatch.setenv("ARA_ENGINES_DIR", str(tmp_path / "engines"))
    monkeypatch.setenv("ARA_DB_PATH", str(tmp_path / "ara.db"))

    from ara import cli, db, profile, scoring

    assert cli.main(["install", "--engine", "cpu", "--json"]) == 0
    install_payload = json.loads(capsys.readouterr().out)
    assert install_payload["key"] == "cpu"
    assert install_payload["status"] in ("installed", "already", "refreshed")

    assert cli.main([
        "characterize", SMOKE_MODEL, "--engine", "cpu", "--json",
    ]) == 0
    characterize_payload = json.loads(capsys.readouterr().out)
    safe_context = characterize_payload["safe_context"]
    assert characterize_payload["engine"] == "cpu"
    assert isinstance(safe_context, int) and not isinstance(safe_context, bool)
    assert safe_context > 0

    with db.connected_readonly() as con:
        row = db.get_characterization(
            con, profile.machine_key(), "cpu", scoring.durable_model_id(SMOKE_MODEL),
        )
    assert row is not None
    assert row["engine"] == "cpu"
    assert row["safe_context"] == safe_context
    assert isinstance(row["artifact_id"], str) and row["artifact_id"]

    assert cli.main([
        "run", SMOKE_MODEL, "Reply with one word", "--engine", "cpu", "--yes",
        "--max-tokens", "8", "--json",
    ]) == 0
    run_payload = json.loads(capsys.readouterr().out)
    assert run_payload["engine"] == "cpu"
    assert run_payload["safe_context"] == safe_context
    assert isinstance(run_payload["completion"], str) and run_payload["completion"].strip()
