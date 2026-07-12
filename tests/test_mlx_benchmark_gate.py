# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""MLX benchmark pre-load gate — gate the EFFECTIVE context, not the raw ceiling.

MLX grows its KV cache dynamically, so a benchmark batch of short prompts only ever reaches
``longest_prompt + max_tokens`` of context — not the measured ceiling it is governed under.
``generate.run`` and CUDA's benchmark already gate this way (their docstrings: gating the raw
ceiling "would over-predict memory and refuse runs that characterize already certified safe");
``ara_engine_mlx.benchmark`` was the one verb still gating the raw ceiling, which refused e.g.
Qwen3-0.6B at its measured window-bound 40960 ("predicted 28.05GB ... >= safe budget 15.18GB")
for a run whose prompts would never exceed ~1k tokens. (The llama.cpp-family workers correctly
gate the raw ceiling — llama.cpp allocates the full KV at n_ctx up front; MLX does not.)

The separately packaged MLX code is outside the coverage gate (omit list) but tested here directly — the gate
is Rule #1 logic.

Slug: 2026-07-02-wmx-benchmark-effective-ctx-gate
"""
from __future__ import annotations

import sys
import types

import pytest

from ara._engine_packages.mlx.ara_engine_mlx import benchmark as wmx_benchmark
from ara._engine_packages.mlx.ara_engine_mlx import generate as wmx_generate


class _FakeTok:
    """Tokenizer double: 1 token per word; optional chat template adds 3 wrapper tokens."""

    def __init__(self, chat_template: str | None = None):
        self.chat_template = chat_template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        return "<t> " + messages[0]["content"] + " </t> <go>"

    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split())))


def _fake_transformers(monkeypatch, tok: _FakeTok):
    mod = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda hf_id: tok))
    monkeypatch.setitem(sys.modules, "transformers", mod)


# --- _render_and_count: the shared render-to-ids + count core ---------------- #

def test_render_and_count_template_branch_returns_ids_and_count():
    tok = _FakeTok(chat_template="{{...}}")
    ids, n = wmx_generate._render_and_count(tok, "hello world")
    # "<t> hello world </t> <go>" -> 5 whitespace tokens
    assert ids == [0, 1, 2, 3, 4] and n == 5


def test_render_and_count_raw_branch_returns_prompt_and_count():
    tok = _FakeTok(chat_template=None)
    out, n = wmx_generate._render_and_count(tok, "one two three")
    assert out == "one two three" and n == 3


def test_render_and_count_empty_prompt_is_zero():
    out, n = wmx_generate._render_and_count(_FakeTok("{{...}}"), "")
    assert out == "" and n == 0


def test_prepare_prompt_still_delegates(monkeypatch):
    # The one-shot path keeps its behavior through the refactor (Rule of least surprise).
    _fake_transformers(monkeypatch, _FakeTok(chat_template="{{...}}"))
    ids, n = wmx_generate._prepare_prompt("org/m", "hello world")
    assert ids == [0, 1, 2, 3, 4] and n == 5


# --- _max_effective_ctx: worst-case context of the batch, tokenizer-only ----- #

def test_max_effective_ctx_is_longest_prompt_plus_max_tokens(monkeypatch):
    _fake_transformers(monkeypatch, _FakeTok(chat_template=None))
    got = wmx_benchmark._max_effective_ctx(
        "org/m", ["a b", "a b c d", "a"], max_tokens=256, ceiling=40960)
    assert got == 4 + 256


def test_max_effective_ctx_is_capped_at_the_ceiling(monkeypatch):
    _fake_transformers(monkeypatch, _FakeTok(chat_template=None))
    got = wmx_benchmark._max_effective_ctx(
        "org/m", ["w " * 5000], max_tokens=256, ceiling=2048)
    assert got == 2048


def test_max_effective_ctx_counts_templated_tokens(monkeypatch):
    # Instruct models are gated on the RENDERED length (template wrapper included).
    _fake_transformers(monkeypatch, _FakeTok(chat_template="{{...}}"))
    got = wmx_benchmark._max_effective_ctx("org/m", ["hello world"], max_tokens=100,
                                           ceiling=40960)
    assert got == 5 + 100


def test_max_effective_ctx_empty_batch_is_max_tokens(monkeypatch):
    _fake_transformers(monkeypatch, _FakeTok(chat_template=None))
    assert wmx_benchmark._max_effective_ctx("org/m", [], max_tokens=64,
                                            ceiling=40960) == 64


# --- benchmark(): the gate receives the effective ctx, ceiling stays reported - #

def _capture_gate(monkeypatch, seen: list):
    def fake_gate(hf_id, ctx, *, margin_gb, overhead_gb, kv_bits=None):
        seen.append(ctx)
        return ({"refused": True, "reason": "stub veto"}, None)   # stop before any load
    monkeypatch.setattr(wmx_benchmark, "_pre_load_gate", fake_gate)


def test_benchmark_gates_effective_ctx_not_raw_ceiling(monkeypatch):
    """The Qwen3-0.6B scenario in miniature: a huge measured window-bound ceiling with short
    prompts must be gated at what the run actually reaches, not the raw ceiling."""
    _fake_transformers(monkeypatch, _FakeTok(chat_template=None))
    seen: list = []
    _capture_gate(monkeypatch, seen)
    out = wmx_benchmark.benchmark("org/qwen", 40960, prompts=["a b c"],
                                  margin_gb=2.0, overhead_gb=1.0, max_tokens=256)
    assert seen == [3 + 256]
    assert out == {"context": 40960, "refused": True, "reason": "stub veto"}


def test_benchmark_falls_back_to_raw_ceiling_when_counting_fails(monkeypatch):
    """Tokenizer-side failure (model not cached, template error) must degrade CONSERVATIVELY —
    gate the raw ceiling as before, so _pre_load_gate still reports not-cached etc. cleanly."""
    def boom(hf_id, prompts, *, max_tokens, ceiling):
        raise RuntimeError("no tokenizer artefacts")
    monkeypatch.setattr(wmx_benchmark, "_max_effective_ctx", boom)
    seen: list = []
    _capture_gate(monkeypatch, seen)
    wmx_benchmark.benchmark("org/uncached", 40960, prompts=["a"],
                            margin_gb=2.0, overhead_gb=1.0, max_tokens=256)
    assert seen == [40960]
