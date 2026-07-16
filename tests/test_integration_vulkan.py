# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Real-engine integration smoke — the Vulkan (AMD iGPU) path, end to end, for real.

Like test_integration_cpu.py but offloaded to the GPU: it installs the isolated ``vulkan`` env
(the prebuilt Vulkan llama-cpp-python wheel), pulls a tiny GGUF, and runs ARA's real methodology
with every layer on the Radeon iGPU — proving the offload, the GTT-sysfs memory measurement, and
the honest offload guard against actual hardware (the mock↔reality boundary the unit tests can't
reach).

Opt-in and kept out of the fast unit loop:

    pytest -m integration --no-cov

**Hardware-gated.** It skips unless an amdgpu Vulkan GPU is present (the amdgpu GTT sysfs exists),
so it's a clean skip on the macOS/CI dev machines — and never triggers a CPU-only source build off
the target hardware. Validated on an x86_64 Linux host (Ryzen Z1 Extreme / Radeon 780M, RADV).

Slug: 2026-06-25-vulkan-amd-engine-lane
"""
from __future__ import annotations

import glob
import shutil

import pytest

pytestmark = pytest.mark.integration

# Same tiny instruct GGUF the CPU smoke uses (worker auto-picks the smallest quant, ~100 MB).
SMOKE_MODEL = "bartowski/SmolLM2-135M-Instruct-GGUF"


def _has_amdgpu() -> bool:
    """An amdgpu GPU is present iff its GTT-accounting sysfs exists (Linux, no root)."""
    return bool(glob.glob("/sys/class/drm/card*/device/mem_info_gtt_total"))


@pytest.fixture
def vulkan_engine(tmp_path, monkeypatch):
    """Install the vulkan engine into a throwaway env; yield the backend. Skips if unsupported."""
    if not _has_amdgpu():
        pytest.skip("no amdgpu Vulkan GPU on this host")
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    monkeypatch.setenv("ARA_ENGINES_DIR", str(tmp_path / "engines"))
    from ara import engines
    from ara.backends import vulkan

    result = engines.install("vulkan")
    if result.status not in ("installed", "already"):
        pytest.skip(f"vulkan engine install unavailable: {result.status} {result.detail}".strip())
    return vulkan


def test_safe_limits_reads_the_shared_ram_wall(vulkan_engine):
    m = vulkan_engine.safe_limits()
    assert m["total_gb"] > 0 and m["wall_gb"] > 0
    assert 0.5 <= m["margin_gb"] <= vulkan_engine.DEFAULT_MARGIN_GB
    assert m["safe_budget_gb"] == pytest.approx(m["wall_gb"] - m["margin_gb"])
    # the GPU pool is carved from system RAM → exact wall, nothing to calibrate (like CPU)
    assert m["calibrated"] is True and m["overhead_gb"] is None


def test_characterize_finds_a_sane_ceiling_offloaded(vulkan_engine):
    # Same fetch-first discipline as the CPU smoke: an offline fetch is a legitimate skip; once
    # cached, a None ceiling would be a real regression (or a silent offload failure) — fail then.
    from ara.workers import vulkan_llama
    try:
        vulkan_llama._resolve_gguf(SMOKE_MODEL)
    except Exception as e:
        pytest.skip(f"{SMOKE_MODEL} unavailable (offline?): {e}")

    r = vulkan_engine.characterize(SMOKE_MODEL)
    # A non-None ceiling means the worker offloaded to the GPU (the offload guard would have
    # returned an error → None otherwise), measured real GTT footprints, and the ramp fit.
    assert r["safe_context"] is not None, "cached model offloaded must yield a ceiling, got None"
    assert isinstance(r["safe_context"], int) and r["safe_context"] >= 2048
    assert r["points"] and all(p["mem_gb"] > 0 for p in r["points"])


def test_generate_produces_a_real_completion_on_gpu(vulkan_engine):
    from ara.workers import vulkan_llama
    try:
        vulkan_llama._resolve_gguf(SMOKE_MODEL)
    except Exception as e:
        pytest.skip(f"{SMOKE_MODEL} unavailable (offline?): {e}")

    out = vulkan_engine.generate(SMOKE_MODEL, "The capital of France is",
                                 max_context=2048, max_tokens=8)
    assert "refused" not in out, f"governed generate was refused: {out.get('reason')}"
    assert out["context"] == 2048
    assert isinstance(out["completion"], str) and out["completion"].strip()
