# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Pure logic of the built-in CPU/llama.cpp worker (ara/workers/cpu_llama.py).

The worker runs in the isolated ``cpu`` env (with llama-cpp-python); ARA's venv can't load
that. But its math — the KV-cache slope from GGUF metadata, the RAM budget, and the
refuse-before-load gate — is plain arithmetic with no engine import, so it's unit-tested here
for confidence. (The file is out of the 100% core coverage gate; see pyproject ``omit``.)
"""
from __future__ import annotations

import pytest

from ara.workers import cpu_llama as w

# Real SmolLM2-135M architecture (llama arch): 30 layers, 576 embd, 9 heads, 3 KV heads (GQA).
_META = {
    "general.architecture": "llama",
    "llama.block_count": "30",
    "llama.embedding_length": "576",
    "llama.attention.head_count": "9",
    "llama.attention.head_count_kv": "3",
    "llama.context_length": "8192",
}


def test_kv_slope_from_gguf_metadata():
    # head_dim = 576/9 = 64; n_embd_kv = 64*3 = 192; bytes/tok = 2(K,V)*30*192*2(f16) = 23040
    expected = 23040 * 1000 / (1024 ** 3)
    assert w.kv_slope_gb_per_k(_META) == pytest.approx(expected, rel=1e-9)


def test_kv_slope_defaults_kv_heads_to_head_count_when_absent():
    meta = {k: v for k, v in _META.items() if "head_count_kv" not in k}
    # no GQA → n_head_kv == n_head=9 → n_embd_kv = 64*9 = 576 (full); 3× the GQA slope
    full = 2 * 30 * 576 * 2 * 1000 / (1024 ** 3)
    assert w.kv_slope_gb_per_k(meta) == pytest.approx(full, rel=1e-9)


def test_max_context_from_metadata():
    assert w.max_context_from(_META) == 8192


def test_safe_threshold_is_total_minus_margin():
    assert w.safe_threshold_gb(48.0, 2.0) == 46.0


def test_safe_threshold_clamps_at_zero_on_small_machines():
    # never a negative budget — a 1GB box with a 2GB margin reports 0, not -1
    assert w.safe_threshold_gb(1.0, 2.0) == 0.0


def test_effective_margin_scales_with_ram_capped_and_floored():
    assert w.effective_margin_gb(48.0, 2.0) == 2.0     # large box → full cap
    assert w.effective_margin_gb(24.0, 2.0) == 2.0     # 10% = 2.4 → capped at 2
    assert w.effective_margin_gb(16.0, 2.0) == 1.6     # 10% = 1.6 < cap
    assert w.effective_margin_gb(2.0, 2.0) == 0.5      # 10% = 0.2 → floored at 0.5
    assert w.effective_margin_gb(1.0, 2.0) == 0.5      # tiny box → floor, not the flat 2GB


def test_limits_from_computes_wall_budget_and_headroom():
    out = w.limits_from(total_gb=24.0, used_gb=10.0, swap_free_gb=1.0,
                        device="x86_64", margin_gb=2.0)
    assert out == {
        "device": "x86_64", "total_gb": 24.0, "wall_gb": 24.0, "safe_budget_gb": 22.0,
        "margin_gb": 2.0, "headroom_gb": 12.0, "swap_free_gb": 1.0,
    }


def test_limits_from_clamps_budget_and_headroom_on_tiny_machine():
    # 1GB box, 0.5GB margin, 0.8GB already used → budget clamps to 0.5, headroom clamps to 0
    out = w.limits_from(total_gb=1.0, used_gb=0.8, swap_free_gb=0.0,
                        device="pi", margin_gb=0.5)
    assert out["safe_budget_gb"] == 0.5 and out["headroom_gb"] == 0.0   # never negative


def test_safety_gate_passes_with_headroom():
    assert w.safety_gate(base_gb=5.0, slope_gb_per_k=0.02, ctx=4000, budget_gb=46.0) is None


def test_safety_gate_refuses_when_base_alone_exceeds_budget():
    reason = w.safety_gate(base_gb=50.0, slope_gb_per_k=0.02, ctx=2000, budget_gb=46.0)
    assert reason is not None and "won't load" in reason


def test_safety_gate_refuses_when_predicted_reaches_budget():
    # base 45 + 1.0/1k * 4000 = 49 >= 46 → refuse before probing
    reason = w.safety_gate(base_gb=45.0, slope_gb_per_k=1.0, ctx=4000, budget_gb=46.0)
    assert reason is not None and "predicted" in reason


def test_probe_refuses_without_an_l5_abort_limit():
    # L5 must never fail open: probing with no abort wall is refused (no model load attempted).
    out = w._probe("/nonexistent.gguf", 2000, None)
    assert out["status"] == "error" and "abort" in out["note"].lower()


# --- generate: governed one-shot inference (Spec 2026-06-23-capability-pipeline, Slice 4) ---
_GEN_META = {"general.architecture": "llama", "llama.block_count": "2",
             "llama.embedding_length": "16", "llama.attention.head_count": "4"}


def test_generate_refuses_when_gate_blocks(monkeypatch):
    # The a-priori L4 gate refuses before any model load when the base alone busts the budget.
    monkeypatch.setattr(w, "_resolve_gguf", lambda m: "/x.gguf")
    monkeypatch.setattr(w, "_read_meta", lambda p: _GEN_META)
    monkeypatch.setattr(w, "_total_gb", lambda: 8.0)
    monkeypatch.setattr(w, "_used_gb", lambda: 1.0)
    monkeypatch.setattr(w, "_model_base_gb", lambda p, o: 50.0)   # base alone > budget
    out = w.generate("org/m", 4000, "hi", margin_gb=2.0, overhead_gb=1.0, max_tokens=16)
    assert out["refused"] is True and out["reason"]


def test_generate_refuses_on_resolve_error(monkeypatch):
    def boom(m):
        raise RuntimeError("no gguf for model")
    monkeypatch.setattr(w, "_resolve_gguf", boom)
    out = w.generate("bad/m", 4000, "hi", margin_gb=2.0, overhead_gb=1.0, max_tokens=16)
    assert out["refused"] is True and "no gguf" in out["reason"]


def test_generate_returns_completion_when_safe(monkeypatch):
    import sys as _sys
    import types as _t

    monkeypatch.setattr(w, "_resolve_gguf", lambda m: "/x.gguf")
    monkeypatch.setattr(w, "_read_meta", lambda p: _GEN_META)
    monkeypatch.setattr(w, "_total_gb", lambda: 64.0)
    monkeypatch.setattr(w, "_used_gb", lambda: 1.0)
    monkeypatch.setattr(w, "_model_base_gb", lambda p, o: 2.0)
    monkeypatch.setattr(w, "safety_gate", lambda **k: None)        # safe

    seen = {}

    class _Llama:
        def __init__(self, **kw):
            seen["n_ctx"] = kw.get("n_ctx")

        def create_chat_completion(self, messages, max_tokens):
            seen["messages"], seen["max_tokens"] = messages, max_tokens
            return {"choices": [{"message": {"content": " 42"}}]}

    monkeypatch.setitem(_sys.modules, "llama_cpp", _t.SimpleNamespace(Llama=_Llama))
    out = w.generate("org/m", 4000, "meaning?", margin_gb=2.0, overhead_gb=1.0, max_tokens=8)
    assert out == {"context": 4000, "completion": " 42"}
    # the prompt is wrapped as a chat message so llama.cpp applies the GGUF's embedded template
    assert seen["messages"] == [{"role": "user", "content": "meaning?"}]
    assert seen["max_tokens"] == 8 and seen["n_ctx"] == 4000   # KV capped at ceiling


def test_used_gb_takes_conservative_max_of_samples(monkeypatch):
    """Rule #1 (Safety): the ambient baseline must take the MAX of repeated reads, never the
    min — an under-reported baseline over-states headroom, a crash-wall trap. base_gb =
    used_gb + model_base is checked >= budget to refuse, so under-counting `used` is unsafe."""
    import types

    import psutil

    reads = iter([1.0 * w.GIB, 3.0 * w.GIB, 2.0 * w.GIB])
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: types.SimpleNamespace(used=next(reads)))
    assert w._used_gb() == pytest.approx(3.0)   # max(1,3,2) GiB, not min (1.0)


# ── per-prompt governance (parity with the MLX/Vulkan governed_max_tokens) ──
def test_governed_max_tokens_allows_when_fits():
    # prompt 100 + request 256 = 356 <= 2048 → allow the full request.
    assert w.governed_max_tokens(100, 256, 2048) == 256


def test_governed_max_tokens_refuses_when_prompt_alone_fills_ceiling():
    # prompt >= ceiling → None (can't even ingest the prompt under the wall).
    assert w.governed_max_tokens(2048, 1, 2048) is None
    assert w.governed_max_tokens(3000, 256, 2048) is None


def test_governed_max_tokens_refuses_when_prompt_plus_request_exceeds_ceiling():
    # 1900 + 256 = 2156 > 2048 → refuse (no silent truncation, matches MLX/Vulkan).
    assert w.governed_max_tokens(1900, 256, 2048) is None


def test_governed_max_tokens_clamps_to_remaining_room():
    assert w.governed_max_tokens(2000, 40, 2048) == 40       # 2040 <= 2048 → allow 40
    assert w.governed_max_tokens(2000, 49, 2048) is None     # 2049 > 2048 → refuse


# --- benchmark: governed multi-prompt completion, model loaded ONCE ---
def _patch_safe_bench(monkeypatch, *, meta=None):
    """Patch the gate-passing path so benchmark reaches the load+iterate body."""
    monkeypatch.setattr(w, "_resolve_gguf", lambda m: "/x.gguf")
    monkeypatch.setattr(w, "_read_meta", lambda p: meta or _GEN_META)
    monkeypatch.setattr(w, "_total_gb", lambda: 64.0)
    monkeypatch.setattr(w, "_used_gb", lambda: 1.0)
    monkeypatch.setattr(w, "_model_base_gb", lambda p, o: 2.0)
    monkeypatch.setattr(w, "safety_gate", lambda **k: None)        # safe


class _FakeLlama:
    """Counts instantiations (load-once proof) and tokenizes by whitespace word count."""
    instances = 0

    def __init__(self, **kw):
        type(self).instances += 1
        self.n_ctx = kw.get("n_ctx")

    def tokenize(self, b):
        return b.split()                       # 1 "token" per whitespace word

    def create_chat_completion(self, messages, max_tokens):
        return {"choices": [{"message": {"content": f"<{max_tokens}>{messages[0]['content']}"}}]}


def test_benchmark_refuses_whole_load_when_gate_blocks(monkeypatch):
    # The a-priori L4 gate refuses the whole load before any model load when base busts budget.
    monkeypatch.setattr(w, "_resolve_gguf", lambda m: "/x.gguf")
    monkeypatch.setattr(w, "_read_meta", lambda p: _GEN_META)
    monkeypatch.setattr(w, "_total_gb", lambda: 8.0)
    monkeypatch.setattr(w, "_used_gb", lambda: 1.0)
    monkeypatch.setattr(w, "_model_base_gb", lambda p, o: 50.0)    # base alone > budget
    out = w.benchmark("org/m", 4000, ["a", "b"], margin_gb=2.0, overhead_gb=1.0, max_tokens=16)
    assert out["refused"] is True and out["reason"] and out["context"] == 4000


def test_benchmark_refuses_on_resolve_error(monkeypatch):
    def boom(m):
        raise RuntimeError("no gguf for model")
    monkeypatch.setattr(w, "_resolve_gguf", boom)
    out = w.benchmark("bad/m", 4000, ["a"], margin_gb=2.0, overhead_gb=1.0, max_tokens=16)
    assert out["refused"] is True and "no gguf" in out["reason"]


def test_benchmark_loads_once_and_completes_each_prompt(monkeypatch):
    import sys as _sys
    import types as _t

    _patch_safe_bench(monkeypatch)
    _FakeLlama.instances = 0
    monkeypatch.setitem(_sys.modules, "llama_cpp", _t.SimpleNamespace(Llama=_FakeLlama))
    out = w.benchmark("org/m", 2048, ["hi there", "yo"],
                      margin_gb=2.0, overhead_gb=1.0, max_tokens=8)
    assert _FakeLlama.instances == 1                       # model loaded ONCE, not per-prompt
    assert out["context"] == 2048
    assert [r["prompt_index"] for r in out["results"]] == [0, 1]
    assert out["results"][0]["completion"] == "<8>hi there"   # governed max_tokens reached gen
    assert out["results"][1]["completion"] == "<8>yo"


def test_benchmark_refuses_individual_prompt_that_fills_ceiling(monkeypatch):
    import sys as _sys
    import types as _t

    _patch_safe_bench(monkeypatch)
    _FakeLlama.instances = 0
    monkeypatch.setitem(_sys.modules, "llama_cpp", _t.SimpleNamespace(Llama=_FakeLlama))
    # ceiling 3, max_tokens 2: "a b c d" = 4 words >= 3 → refused; "a" (1 tok) + 2 = 3 ≤ 3 → runs.
    out = w.benchmark("org/m", 3, ["a b c d", "a"],
                      margin_gb=2.0, overhead_gb=1.0, max_tokens=2)
    assert out["results"][0]["refused"] is True and "ceiling 3" in out["results"][0]["reason"]
    assert out["results"][1]["completion"] == "<2>a"          # small prompt runs, clamped to 2
    assert _FakeLlama.instances == 1                          # still a single load
