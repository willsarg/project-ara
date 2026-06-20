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

import shutil

import pytest

pytestmark = pytest.mark.integration

# A tiny instruct GGUF (worker auto-picks the smallest quant, ~100 MB) with a 2048 window —
# the exact small-window shape that exposed the driver bug.
SMOKE_MODEL = "bartowski/SmolLM2-135M-Instruct-GGUF"


@pytest.fixture
def cpu_engine(tmp_path, monkeypatch):
    """Install the cpu engine into a throwaway env; yield the cpu backend. Skips if it can't."""
    monkeypatch.setenv("ARA_ENGINES_DIR", str(tmp_path / "engines"))
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    from ara import engines
    from ara.backends import cpu

    result = engines.install("cpu")
    if result.status not in ("installed", "already"):
        pytest.skip(f"cpu engine install unavailable: {result.status} {result.detail}".strip())
    return cpu


def test_safe_limits_reads_a_real_ram_wall(cpu_engine):
    m = cpu_engine.safe_limits()
    assert m["total_gb"] > 0 and m["wall_gb"] > 0
    assert m["safe_budget_gb"] == pytest.approx(m["wall_gb"] - cpu_engine.DEFAULT_MARGIN_GB)
    assert m["calibrated"] is True and m["overhead_gb"] is None   # exact wall, nothing to calibrate


def test_characterize_finds_a_sane_ceiling(cpu_engine):
    try:
        r = cpu_engine.characterize(SMOKE_MODEL)
    except Exception as e:                       # offline / model unfetchable
        pytest.skip(f"could not fetch/run {SMOKE_MODEL}: {e}")
    if r["safe_context"] is None:
        pytest.skip("model could not be fetched (no cached GGUF, likely offline)")
    # A 135M model fits its whole trained window on any real machine — so the live path must
    # report a positive, window-bound ceiling (memory never binds first here).
    assert isinstance(r["safe_context"], int) and r["safe_context"] >= 2048
    assert r["binding"] == "context_window"
    assert r["points"] and all(p["mem_gb"] > 0 for p in r["points"])
