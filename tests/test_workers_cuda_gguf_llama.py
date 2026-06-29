# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Pure logic of the CUDA-GGUF hybrid worker (ara/workers/cuda_gguf_llama.py).

The worker runs in the isolated ``cuda-gguf`` env (CUDA llama.cpp); ARA's venv can't load it.
But the novel, safety-critical math — the per-layer split, the K auto-fit, the TWO-wall gate
(VRAM *and* RAM), the load-log buffer parse, and the partial-offload honest check — is plain
arithmetic/regex with no engine import, so it's unit-tested here. The single-wall helpers reused
unchanged from the cpu/vulkan workers (kv slope, margins) are covered there.

Slug: 2026-06-29-cuda-gguf-hybrid
"""
from __future__ import annotations

from ara.workers import cuda_gguf_llama as w


# --------------------------------------------------------------------------- #
# per-layer split + a-priori VRAM/RAM estimates (coarse + conservative; the
# load-log measurement is the real certifier — see the design spec §4)
# --------------------------------------------------------------------------- #
def test_per_layer_gb_divides_weights_across_layers():
    assert w.per_layer_gb(8.0, 32) == 0.25


def test_vram_estimate_is_floor_plus_offloaded_weights_plus_their_kv():
    # N=32, weights 8GB → per_w 0.25; kv_slope 0.32 → kv/layer/1k = 0.01; ctx 4000 → 0.04/layer.
    # vram(18) = 1.5 floor + 18*0.25 + 18*0.01*4 = 1.5 + 4.5 + 0.72 = 6.72
    assert w.vram_estimate(18, 32, 8.0, 0.32, 4000, cuda_floor_gb=1.5) == 6.72


def test_ram_estimate_is_base_plus_cpu_layers_plus_their_kv():
    # remainder = 32-18 = 14 layers on CPU; live base 2.0
    # ram(18) = 2.0 + 14*0.25 + 14*0.01*4 = 2.0 + 3.5 + 0.56 = 6.06
    assert w.ram_estimate(18, 32, 8.0, 0.32, 4000, live_base_gb=2.0) == 6.06


# --------------------------------------------------------------------------- #
# fit_layers — pick the LARGEST K whose VRAM estimate fits the budget
# --------------------------------------------------------------------------- #
def test_fit_layers_picks_largest_k_under_vram_budget():
    # vram(k) = 1.5 + 0.29k ; <= 7.0 → k <= 18.96 → 18
    assert w.fit_layers(32, 8.0, 0.32, 4000, 7.0, cuda_floor_gb=1.5) == 18


def test_fit_layers_caps_at_n_layers_when_everything_fits():
    # tiny model, huge budget → offload all N (never more than N)
    assert w.fit_layers(32, 0.5, 0.01, 2000, 80.0, cuda_floor_gb=1.5) == 32


def test_fit_layers_returns_zero_when_floor_alone_busts_budget():
    # even K=0 costs the CUDA floor; a 1GB budget can't hold a 1.5GB floor → 0 layers
    assert w.fit_layers(32, 8.0, 0.32, 4000, 1.0, cuda_floor_gb=1.5) == 0


# --------------------------------------------------------------------------- #
# two_wall_gate — refuse if EITHER wall would be breached (Rule #1)
# --------------------------------------------------------------------------- #
def test_two_wall_gate_passes_when_both_walls_have_headroom():
    assert w.two_wall_gate(18, 32, 8.0, 0.32, 4000, vram_budget_gb=7.0,
                           ram_budget_gb=30.0, live_base_gb=2.0, cuda_floor_gb=1.5) is None


def test_two_wall_gate_refuses_on_vram_wall():
    # vram(18)=6.72 >= 6.0 budget → refuse, naming VRAM
    reason = w.two_wall_gate(18, 32, 8.0, 0.32, 4000, vram_budget_gb=6.0,
                             ram_budget_gb=30.0, live_base_gb=2.0, cuda_floor_gb=1.5)
    assert reason is not None and "VRAM" in reason


def test_two_wall_gate_refuses_on_ram_wall():
    # ram(18)=6.06 >= 5.0 budget → refuse, naming RAM (VRAM is fine here)
    reason = w.two_wall_gate(18, 32, 8.0, 0.32, 4000, vram_budget_gb=7.0,
                             ram_budget_gb=5.0, live_base_gb=2.0, cuda_floor_gb=1.5)
    assert reason is not None and "RAM" in reason


# --------------------------------------------------------------------------- #
# parse_cuda_buffers — the post-load TRUTH (sum CUDA0 → VRAM, CPU* → RAM)
# RSS is unreliable (mmap page cache); the load-log buffer lines are trusted.
# --------------------------------------------------------------------------- #
_LOG = (
    "load_tensors: offloaded 18/32 layers to GPU\n"
    "load_tensors:        CUDA0 model buffer size =  4608.00 MiB\n"
    "load_tensors:   CPU_Mapped model buffer size =  3584.00 MiB\n"
    "llama_kv_cache:      CUDA0 KV buffer size =   144.00 MiB\n"
    "llama_kv_cache:        CPU KV buffer size =   112.00 MiB\n"
    "llama_context:        CUDA0 compute buffer size =   304.00 MiB\n"
)


def test_parse_cuda_buffers_sums_vram_side():
    # VRAM = 4608 + 144 + 304 = 5056 MiB
    b = w.parse_cuda_buffers(_LOG)
    assert b["vram_gb"] == round(5056 / 1024, 4)


def test_parse_cuda_buffers_sums_ram_side():
    # RAM = CPU_Mapped 3584 + CPU KV 112 = 3696 MiB
    b = w.parse_cuda_buffers(_LOG)
    assert b["ram_gb"] == round(3696 / 1024, 4)


def test_parse_cuda_buffers_zero_when_absent():
    b = w.parse_cuda_buffers("nothing to see here\n")
    assert b == {"vram_gb": 0.0, "ram_gb": 0.0}


# --------------------------------------------------------------------------- #
# Honest offload (Rule #3) — partial (0<K<N) is VALID here, unlike Vulkan which
# demanded full offload. Refuse silent-CPU-fallback (#2079): absent, K==0, sw.
# --------------------------------------------------------------------------- #
def test_offload_ok_partial_accepts_partial_offload():
    # 18/32 on a real CUDA device is the expected hybrid state → no refusal
    assert w.offload_ok_partial({"name": "NVIDIA GeForce RTX 2070"}, (18, 32)) is None


def test_offload_ok_partial_refuses_when_no_offload_line():
    reason = w.offload_ok_partial(None, None)
    assert reason and "not active" in reason


def test_offload_ok_partial_refuses_when_zero_layers_offloaded():
    # K==0 ran entirely on CPU (the silent-CPU-fallback, #2079) → not a hybrid run
    reason = w.offload_ok_partial({"name": "NVIDIA …"}, (0, 32))
    assert reason and "ran on CPU" in reason


def test_governed_max_tokens_reused_unchanged():
    # parity with cpu/vulkan/MLX per-prompt governance
    assert w.governed_max_tokens(100, 256, 2048) == 256
    assert w.governed_max_tokens(2048, 1, 2048) is None
    assert w.governed_max_tokens(1900, 256, 2048) is None
