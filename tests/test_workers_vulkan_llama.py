# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""workers/vulkan_llama.py — pure logic (no llama.cpp, no GPU).

The worker is a self-contained script that never imports ``ara`` and imports ``llama_cpp`` only
inside functions, so its top-level pure logic is unit-testable in ARA's own venv. These tests
cover the Vulkan-specific bits that aren't shared with the CPU worker: the offload/device log
parsers, the honest offload guard (Rule #3), and the amdgpu GTT sysfs reader — plus the shared
budget arithmetic, to prove it's reused unchanged.

Slug: 2026-06-25-vulkan-amd-engine-lane
"""
from __future__ import annotations

from ara.workers import vulkan_llama as w


# --------------------------------------------------------------------------- #
# parse_offloaded — "offloaded N/M layers to GPU"
# --------------------------------------------------------------------------- #
def test_parse_offloaded_full():
    stderr = "load_tensors: offloaded 17/17 layers to GPU\nother line"
    assert w.parse_offloaded(stderr) == (17, 17)


def test_parse_offloaded_partial_and_spacing():
    assert w.parse_offloaded("offloaded 12 / 28 layers to GPU") == (12, 28)


def test_parse_offloaded_absent_is_none():
    assert w.parse_offloaded("llama_model_loader: loaded meta data\n") is None


# --------------------------------------------------------------------------- #
# parse_vulkan_device — the ggml_vulkan device line
# --------------------------------------------------------------------------- #
_DEVLINE = ("ggml_vulkan: 0 = AMD Ryzen Z1 Extreme (RADV PHOENIX) (radv) | uma: 1 | fp16: 1 "
            "| bf16: 0 | warp size: 64 | matrix cores: KHR_coopmat")


def test_parse_vulkan_device_name_and_coopmat():
    d = w.parse_vulkan_device(_DEVLINE)
    assert d["name"] == "AMD Ryzen Z1 Extreme (RADV PHOENIX) (radv)"
    assert d["coopmat"] == "KHR_coopmat"


def test_parse_vulkan_device_no_coopmat_field():
    d = w.parse_vulkan_device("ggml_vulkan: 0 = Some GPU | uma: 0")
    assert d == {"name": "Some GPU", "coopmat": None}


def test_parse_vulkan_device_absent_is_none():
    assert w.parse_vulkan_device("no vulkan here") is None


# --------------------------------------------------------------------------- #
# offload_ok — the honest guard (Rule #3): refuse a non-GPU run
# --------------------------------------------------------------------------- #
def test_offload_ok_when_fully_offloaded_to_real_gpu():
    device = {"name": "AMD Ryzen Z1 Extreme (RADV PHOENIX)", "coopmat": "KHR_coopmat"}
    assert w.offload_ok(device, (17, 17)) is None


def test_offload_ok_refuses_when_no_offload_line():
    reason = w.offload_ok(None, None)
    assert reason and "not active" in reason


def test_offload_ok_refuses_when_zero_layers_offloaded():
    reason = w.offload_ok({"name": "AMD …", "coopmat": None}, (0, 28))
    assert reason and "ran on CPU" in reason


def test_offload_ok_refuses_software_rasterizer():
    device = {"name": "llvmpipe (LLVM 20.1.2, 256 bits)", "coopmat": None}
    reason = w.offload_ok(device, (28, 28))
    assert reason and "software rasterizer" in reason


# --------------------------------------------------------------------------- #
# _gpu_used_gb — amdgpu GTT+VRAM sysfs reader (the memory-governance signal)
# --------------------------------------------------------------------------- #
def _make_drm(tmp_path, *, gtt_used, vram_used, card="card1"):
    dev = tmp_path / card / "device"
    dev.mkdir(parents=True)
    (dev / "mem_info_gtt_used").write_text(str(gtt_used))
    (dev / "mem_info_vram_used").write_text(str(vram_used))
    return dev


def test_gpu_used_gb_sums_gtt_and_vram(tmp_path, monkeypatch):
    _make_drm(tmp_path, gtt_used=1 * w.GIB, vram_used=2 * w.GIB)
    monkeypatch.setattr(w, "DRM_DEVICE_GLOB", str(tmp_path / "card*" / "device"))
    assert w._gpu_used_gb() == 3.0


def test_gpu_used_gb_sums_across_multiple_cards(tmp_path, monkeypatch):
    _make_drm(tmp_path, gtt_used=1 * w.GIB, vram_used=0, card="card0")
    _make_drm(tmp_path, gtt_used=2 * w.GIB, vram_used=1 * w.GIB, card="card1")
    monkeypatch.setattr(w, "DRM_DEVICE_GLOB", str(tmp_path / "card*" / "device"))
    assert w._gpu_used_gb() == 4.0


def test_gpu_used_gb_zero_when_no_amdgpu(tmp_path, monkeypatch):
    # non-amdgpu host: the glob matches nothing → 0.0 (RSS delta + offload guard still cover us)
    monkeypatch.setattr(w, "DRM_DEVICE_GLOB", str(tmp_path / "card*" / "device"))
    assert w._gpu_used_gb() == 0.0


def test_gpu_used_gb_skips_unreadable_files(tmp_path, monkeypatch):
    dev = tmp_path / "card1" / "device"
    dev.mkdir(parents=True)
    (dev / "mem_info_gtt_used").write_text("4294967296")   # 4 GiB
    (dev / "mem_info_vram_used").write_text("not-a-number")  # ValueError → skipped
    monkeypatch.setattr(w, "DRM_DEVICE_GLOB", str(tmp_path / "card*" / "device"))
    assert w._gpu_used_gb() == 4.0


# --------------------------------------------------------------------------- #
# Shared budget arithmetic — reused unchanged from the CPU worker's methodology
# --------------------------------------------------------------------------- #
def test_effective_margin_scales_to_small_apu():
    # ~11 GB APU → 10% = 1.1 GB (below the 2 GB cap, above the 0.5 GB floor)
    assert w.effective_margin_gb(11.0, 2.0) == 1.1


def test_safe_threshold_clamps_at_zero():
    assert w.safe_threshold_gb(1.0, 2.0) == 0.0


def test_safety_gate_refuses_when_base_exceeds_budget():
    assert "won't load" in w.safety_gate(base_gb=10.0, slope_gb_per_k=1.0, ctx=4000,
                                         budget_gb=9.9)


def test_safety_gate_refuses_when_prediction_exceeds_budget():
    r = w.safety_gate(base_gb=5.0, slope_gb_per_k=2.0, ctx=4000, budget_gb=9.9)
    assert r and "9.90" in r


def test_safety_gate_passes_when_safe():
    assert w.safety_gate(base_gb=2.0, slope_gb_per_k=0.5, ctx=2000, budget_gb=9.9) is None


def test_kv_slope_uses_gqa_kv_heads():
    meta = {"general.architecture": "llama", "llama.block_count": "2",
            "llama.embedding_length": "16", "llama.attention.head_count": "4",
            "llama.attention.head_count_kv": "2"}
    # 2 (K+V) × 2 layers × (head_dim 4 × 2 kv heads = 8) × 2 bytes × 1000 / GIB
    assert w.kv_slope_gb_per_k(meta) == (2 * 2 * 8 * 2) * 1000 / w.GIB


def test_limits_from_reports_shared_ram_wall():
    d = w.limits_from(total_gb=11.0, used_gb=1.0, swap_free_gb=2.0, device="GPU (Vulkan)",
                      margin_gb=1.1)
    assert d["wall_gb"] == 11.0 and d["safe_budget_gb"] == 9.9
    assert d["device"] == "GPU (Vulkan)"
