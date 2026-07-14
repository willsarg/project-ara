# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Native dynamic-KV engines must never generate past a characterized ceiling."""
from __future__ import annotations

import sys
import types

from ara._engine_packages.cuda.ara_engine_cuda import benchmark as cuda_benchmark
from ara._engine_packages.cuda.ara_engine_cuda import generate as cuda_generate
from ara._engine_packages.mlx.ara_engine_mlx import generate as mlx_generate


def _causal_model():
    return types.SimpleNamespace(is_causal=True)


class _ChatTokenizer:
    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt is True and tokenize is False
        return f"<bos> {messages[0]['content']} <end> <assistant>"

    def encode(self, text):
        return text.split()


def test_cuda_prompt_count_includes_chat_template(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda _model: _ChatTokenizer())))

    assert cuda_generate._count_prompt_tokens("org/model", "hello world") == 5


def test_mlx_generate_refuses_prompt_plus_output_past_ceiling(monkeypatch):
    monkeypatch.setattr(mlx_generate.models, "describe", lambda _model: _causal_model())
    monkeypatch.setattr(mlx_generate, "_prepare_prompt", lambda *_args: ([1] * 7, 7))
    monkeypatch.setattr(mlx_generate.measure_one, "_effective_kv_bits", lambda *_args: None)
    monkeypatch.setattr(mlx_generate.measure_one, "safety_gate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mlx_generate.system, "read_limits", lambda: object())
    monkeypatch.setattr(mlx_generate.system, "sample_settled_baseline", lambda: 0.0)
    monkeypatch.setattr(mlx_generate, "register_turn_end_tokens", lambda *_args: None)
    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(
        load=lambda _model: (object(), object()),
        generate=lambda *_args, **_kwargs: "generated past ceiling",
    ))

    out = mlx_generate.generate(
        "org/model", 10, prompt="hello", margin_gb=2.0, overhead_gb=1.0, max_tokens=4)

    assert out["refused"] is True
    assert "needed 11" in out["reason"] and "ceiling 10" in out["reason"]


def test_mlx_generate_accepts_prompt_plus_output_at_ceiling(monkeypatch):
    monkeypatch.setattr(mlx_generate.models, "describe", lambda _model: _causal_model())
    monkeypatch.setattr(mlx_generate, "_prepare_prompt", lambda *_args: ([1] * 7, 7))
    monkeypatch.setattr(mlx_generate.measure_one, "_effective_kv_bits", lambda *_args: None)
    monkeypatch.setattr(mlx_generate.measure_one, "safety_gate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mlx_generate.system, "read_limits", lambda: object())
    monkeypatch.setattr(mlx_generate.system, "sample_settled_baseline", lambda: 0.0)
    monkeypatch.setattr(mlx_generate, "register_turn_end_tokens", lambda *_args: None)
    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(
        load=lambda _model: (object(), object()),
        generate=lambda *_args, **_kwargs: "exact fit",
    ))

    out = mlx_generate.generate(
        "org/model", 10, prompt="hello", margin_gb=2.0, overhead_gb=1.0, max_tokens=3)

    assert out == {"context": 10, "completion": "exact fit"}


def test_cuda_generate_refuses_prompt_plus_output_past_ceiling(monkeypatch):
    monkeypatch.setattr(cuda_generate.models, "describe", lambda _model: _causal_model())
    monkeypatch.setattr(cuda_generate.system, "read_limits",
                        lambda: types.SimpleNamespace(used_gb=0.0))
    monkeypatch.setattr(cuda_generate.measure_one, "_effective_kv_bits", lambda *_args: None)
    monkeypatch.setattr(cuda_generate.measure_one, "safety_gate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cuda_generate, "_count_prompt_tokens", lambda *_args: 7)
    monkeypatch.setattr(cuda_generate, "_generate",
                        lambda *_args, **_kwargs: "generated past ceiling")

    out = cuda_generate.run(
        "org/model", 10, prompt="hello", margin_gb=2.0, overhead_gb=1.0, max_tokens=4)

    assert out["refused"] is True
    assert "needed 11" in out["reason"] and "ceiling 10" in out["reason"]


def test_cuda_generate_accepts_prompt_plus_output_at_ceiling(monkeypatch):
    monkeypatch.setattr(cuda_generate.models, "describe", lambda _model: _causal_model())
    monkeypatch.setattr(cuda_generate.system, "read_limits",
                        lambda: types.SimpleNamespace(used_gb=0.0))
    monkeypatch.setattr(cuda_generate.measure_one, "_effective_kv_bits", lambda *_args: None)
    monkeypatch.setattr(cuda_generate.measure_one, "safety_gate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cuda_generate, "_count_prompt_tokens", lambda *_args: 7)
    monkeypatch.setattr(cuda_generate, "_generate", lambda *_args, **_kwargs: "exact fit")

    out = cuda_generate.run(
        "org/model", 10, prompt="hello", margin_gb=2.0, overhead_gb=1.0, max_tokens=3)

    assert out == {"context": 10, "completion": "exact fit"}


def test_cuda_benchmark_refuses_item_past_ceiling_before_model_load(monkeypatch):
    tokenizer = types.SimpleNamespace(encode=lambda _prompt: [1] * 7)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda _model: tokenizer)))
    monkeypatch.setattr(cuda_benchmark.models, "describe", lambda _model: _causal_model())
    monkeypatch.setattr(cuda_benchmark.system, "read_limits",
                        lambda: types.SimpleNamespace(used_gb=0.0))
    monkeypatch.setattr(cuda_benchmark.measure_one, "_effective_kv_bits", lambda *_args: None)
    monkeypatch.setattr(cuda_benchmark.measure_one, "safety_gate", lambda *_args, **_kwargs: None)
    loads = []
    monkeypatch.setattr(cuda_benchmark.probe_worker, "_load_model",
                        lambda *_args, **_kwargs: loads.append(True) or object())
    monkeypatch.setattr(cuda_benchmark, "_generate_one",
                        lambda *_args, **_kwargs: "generated past ceiling")

    out = cuda_benchmark.run(
        "org/model", 10, prompts=["hello"], margin_gb=2.0, overhead_gb=1.0, max_tokens=4)

    assert out["results"][0]["refused"] is True
    assert "needed 11" in out["results"][0]["reason"]
    assert loads == []


def test_cuda_benchmark_accepts_item_at_ceiling(monkeypatch):
    tokenizer = types.SimpleNamespace(encode=lambda _prompt: [1] * 7)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda _model: tokenizer)))
    monkeypatch.setattr(cuda_benchmark.models, "describe", lambda _model: _causal_model())
    monkeypatch.setattr(cuda_benchmark.system, "read_limits",
                        lambda: types.SimpleNamespace(used_gb=0.0))
    monkeypatch.setattr(cuda_benchmark.measure_one, "_effective_kv_bits", lambda *_args: None)
    monkeypatch.setattr(cuda_benchmark.measure_one, "safety_gate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cuda_benchmark.probe_worker, "_load_model",
                        lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cuda_benchmark, "_generate_one",
                        lambda *_args, **_kwargs: "exact fit")

    out = cuda_benchmark.run(
        "org/model", 10, prompts=["hello"], margin_gb=2.0, overhead_gb=1.0, max_tokens=3)

    assert out == {"context": 10, "results": [
        {"prompt_index": 0, "completion": "exact fit"},
    ]}


def test_cuda_benchmark_governs_rendered_prompt_length(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda _model: _ChatTokenizer())))
    monkeypatch.setattr(cuda_benchmark.models, "describe", lambda _model: _causal_model())
    monkeypatch.setattr(cuda_benchmark.system, "read_limits",
                        lambda: types.SimpleNamespace(used_gb=0.0))
    monkeypatch.setattr(cuda_benchmark.measure_one, "_effective_kv_bits", lambda *_args: None)
    monkeypatch.setattr(cuda_benchmark.measure_one, "safety_gate", lambda *_args, **_kwargs: None)
    loads = []
    monkeypatch.setattr(cuda_benchmark.probe_worker, "_load_model",
                        lambda *_args, **_kwargs: loads.append(True) or object())

    out = cuda_benchmark.run(
        "org/model", 8, prompts=["hello world"], margin_gb=2.0, overhead_gb=1.0,
        max_tokens=4)

    assert out["results"][0]["refused"] is True
    assert "needed 9" in out["results"][0]["reason"]
    assert loads == []
