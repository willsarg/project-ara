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

import math

import pytest

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
    "ggml_cuda_init: found 1 CUDA devices:\n"
    "  Device 0: NVIDIA GeForce RTX 2070, compute capability 7.5\n"
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


def test_parse_cuda_buffers_records_line_provenance():
    b = w.parse_cuda_buffers(_LOG)
    assert b["vram_buffer_lines"] == 3
    assert b["ram_buffer_lines"] == 2


@pytest.mark.parametrize(
    "log",
    [
        "nothing to see here\n",
        "load_tensors: CUDA0 model buffer size = 100 MiB\n",
        "load_tensors: CPU_Mapped model buffer size = 100 MiB\n",
        "load_tensors: CUDA1 model buffer size = 100 MiB\n"
        "load_tensors: CPU_Mapped model buffer size = 100 MiB\n",
    ],
)
def test_parse_cuda_buffers_refuses_missing_or_ambiguous_walls(log):
    with pytest.raises(w.BufferTelemetryError, match="buffer telemetry"):
        w.parse_cuda_buffers(log)


@pytest.mark.parametrize("value", ["NaN", "inf", "-1"])
def test_parse_cuda_buffers_refuses_nonfinite_or_negative_values(value):
    log = (
        f"load_tensors: CUDA0 model buffer size = {value} MiB\n"
        "load_tensors: CPU_Mapped model buffer size = 100 MiB\n"
    )
    with pytest.raises(w.BufferTelemetryError, match="finite non-negative"):
        w.parse_cuda_buffers(log)


# --------------------------------------------------------------------------- #
# Honest offload (Rule #3) — partial (0<K<N) is VALID here, unlike Vulkan which
# demanded full offload. Refuse silent-CPU-fallback (#2079): absent, K==0, sw.
# --------------------------------------------------------------------------- #
def test_offload_ok_partial_accepts_partial_offload():
    # 18/32 on a real CUDA device is the expected hybrid state → no refusal
    assert w.offload_ok_partial({"name": "NVIDIA GeForce RTX 2070"}, (18, 32)) is None


def test_offload_ok_partial_refuses_missing_device_identity():
    reason = w.offload_ok_partial(None, (18, 32))
    assert reason and "device identity" in reason


def test_offload_ok_partial_refuses_when_no_offload_line():
    reason = w.offload_ok_partial(None, None)
    assert reason and "not active" in reason


def test_offload_ok_partial_refuses_when_zero_layers_offloaded():
    # K==0 ran entirely on CPU (the silent-CPU-fallback, #2079) → not a hybrid run
    reason = w.offload_ok_partial({"name": "NVIDIA …"}, (0, 32))
    assert reason and "ran on CPU" in reason


def test_verify_offload_refuses_ram_bound_vram_bound_and_contradictory_logs():
    ram_bound = {
        "vram_budget_gb": 8.0,
        "ram_used_gb": 2.0,
        "ram_budget_gb": 5.0,
    }
    vram_bound = {**ram_bound, "vram_budget_gb": 4.0, "ram_budget_gb": 20.0}

    assert "measured RAM" in w._verify_offload(_LOG, ram_bound, expected_gpu_layers=18)
    assert "measured VRAM" in w._verify_offload(_LOG, vram_bound, expected_gpu_layers=18)
    assert "requested 17" in w._verify_offload(
        _LOG, {**ram_bound, "ram_budget_gb": 20.0}, expected_gpu_layers=17)


def test_verify_offload_refuses_incomplete_buffer_telemetry():
    log = (
        "Device 0: NVIDIA GeForce RTX 2070, compute capability 7.5\n"
        "offloaded 18/32 layers to GPU\n"
    )
    assert "buffer telemetry" in w._verify_offload(
        log,
        {"vram_budget_gb": 8.0, "ram_used_gb": 2.0, "ram_budget_gb": 20.0},
        expected_gpu_layers=18,
    )


def test_run_reports_absolute_ram_fit_and_both_walls_with_provenance(monkeypatch):
    budgets = {
        "vram_budget_gb": 8.0,
        "ram_used_gb": 2.0,
        "ram_budget_gb": 20.0,
    }
    monkeypatch.setattr(w, "_resolve_gguf", lambda _model: "/x.gguf")
    monkeypatch.setattr(w, "_read_meta", lambda _path: {})
    monkeypatch.setattr(w, "_gate", lambda *_args: (18, budgets))
    monkeypatch.setattr(w, "_load", lambda *_args: (object(), _LOG))

    point = w.run("org/m", 4096, vram_margin_gb=1.0, ram_margin_gb=2.0, repeats=1)

    ram_absolute = round(2.0 + round(3696 / 1024, 4), 4)
    assert point["mem_gb"] == ram_absolute
    assert point["telemetry"] == {
        "schema": "cuda-gguf-two-wall-telemetry:v1",
        "fit_dimension": "ram_absolute",
        "unit": "GiB",
        "gpu_layers": 18,
        "vram": {
            "observed_gb": round(5056 / 1024, 4),
            "budget_gb": 8.0,
        },
        "ram": {
            "observed_buffers_gb": round(3696 / 1024, 4),
            "baseline_gb": 2.0,
            "observed_absolute_gb": ram_absolute,
            "budget_gb": 20.0,
        },
        "provenance": {
            "source": "llama.cpp-load-log",
            "aggregation": "median",
            "repeat_count": 1,
            "vram_buffer_lines": 3,
            "ram_buffer_lines": 2,
        },
    }


def test_preflight_binds_shared_fit_to_absolute_ram_budget(monkeypatch):
    meta = {
        "general.architecture": "qwen3",
        "qwen3.block_count": 32,
        "qwen3.embedding_length": 4096,
        "qwen3.attention.head_count": 32,
        "qwen3.attention.head_count_kv": 8,
        "qwen3.context_length": 8192,
    }
    budgets = {
        "vram_budget_gb": 8.0,
        "ram_used_gb": 2.0,
        "ram_budget_gb": 20.0,
    }
    monkeypatch.setattr(w, "_resolve_gguf", lambda _model: "/x.gguf")
    monkeypatch.setattr(w, "_read_meta", lambda _path: meta)
    monkeypatch.setattr(w, "_budgets", lambda *_args: budgets)
    monkeypatch.setattr(w, "_model_weights_gb", lambda _path: 8.0)
    monkeypatch.setattr(
        w, "_used_ram_gb", lambda: pytest.fail("read a second RAM baseline"))

    estimate = w.preflight("org/m", vram_margin_gb=1.0, ram_margin_gb=2.0)

    assert estimate["fit_dimension"] == "ram_absolute"
    assert estimate["memory_unit"] == "GiB"
    assert estimate["ref_baseline_gb"] == 0.0
    assert estimate["budget_gb"] == estimate["ram_budget_gb"] == 20.0
    assert estimate["base_gb"] >= budgets["ram_used_gb"]


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_two_wall_payload_never_emits_nonfinite_observations(value):
    with pytest.raises(w.BufferTelemetryError, match="finite"):
        w.two_wall_measurement(
            18,
            {
                "vram_gb": value,
                "ram_gb": 1.0,
                "vram_buffer_lines": 1,
                "ram_buffer_lines": 1,
            },
            {"vram_budget_gb": 8.0, "ram_used_gb": 2.0, "ram_budget_gb": 20.0},
        )


def test_governed_max_tokens_reused_unchanged():
    # parity with cpu/vulkan/MLX per-prompt governance
    assert w.governed_max_tokens(100, 256, 2048) == 256
    assert w.governed_max_tokens(2048, 1, 2048) is None
    assert w.governed_max_tokens(1900, 256, 2048) is None


def test_governed_chat_completion_refuses_expanded_template_before_inference():
    class _TemplatedLlama:
        def create_completion(self, *, prompt, max_tokens, **_kw):
            pytest.fail("inference must not start")

        def create_chat_completion(self, messages, max_tokens):
            return self.create_completion(prompt=[1] * 9, max_tokens=max_tokens)

    out, reason = w._governed_chat_completion(_TemplatedLlama(), "short", 2, 10)
    assert out is None and "ceiling 10" in reason


def test_benchmark_governs_rendered_template_tokens(monkeypatch):
    monkeypatch.setattr(w, "_resolve_gguf", lambda _m: "/x.gguf")
    monkeypatch.setattr(w, "_read_meta", lambda _p: {})
    monkeypatch.setattr(w, "_gate", lambda *_a: (4, {"vram_gb": 1.0, "ram_gb": 1.0}))
    monkeypatch.setattr(w, "_verify_offload", lambda *_a, **_k: None)

    class _Llama:
        def create_completion(self, *, prompt, max_tokens, **_kw):
            pytest.fail("inference must not start")

        def create_chat_completion(self, messages, max_tokens):
            return self.create_completion(prompt=[1] * 9, max_tokens=max_tokens)

    monkeypatch.setattr(w, "_load", lambda *_a: (_Llama(), "offloaded 4/8 layers to GPU"))
    out = w.benchmark("org/m", 10, ["short"], vram_margin_gb=1.0,
                      ram_margin_gb=2.0, max_tokens=2)
    assert out["results"] == [{"prompt_index": 0, "refused": True,
                               "reason": "prompt fills context ceiling 10"}]
