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


def test_limits_from_computes_wall_budget_and_headroom():
    out = w.limits_from(total_gb=24.0, used_gb=10.0, swap_free_gb=1.0,
                        device="x86_64", margin_gb=2.0)
    assert out == {
        "device": "x86_64", "total_gb": 24.0, "wall_gb": 24.0, "safe_budget_gb": 22.0,
        "margin_gb": 2.0, "headroom_gb": 12.0, "swap_free_gb": 1.0,
    }


def test_safety_gate_passes_with_headroom():
    assert w.safety_gate(base_gb=5.0, slope_gb_per_k=0.02, ctx=4000, budget_gb=46.0) is None


def test_safety_gate_refuses_when_base_alone_exceeds_budget():
    reason = w.safety_gate(base_gb=50.0, slope_gb_per_k=0.02, ctx=2000, budget_gb=46.0)
    assert reason is not None and "won't load" in reason


def test_safety_gate_refuses_when_predicted_reaches_budget():
    # base 45 + 1.0/1k * 4000 = 49 >= 46 → refuse before probing
    reason = w.safety_gate(base_gb=45.0, slope_gb_per_k=1.0, ctx=4000, budget_gb=46.0)
    assert reason is not None and "predicted" in reason
