# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""catalog.py — model metadata (describe) + the db-backed model catalog."""
from __future__ import annotations

from ara import catalog


def test_describe_parses_config(monkeypatch):
    monkeypatch.setattr(catalog, "_read_config", lambda m: {
        "num_hidden_layers": 30, "hidden_size": 576, "num_attention_heads": 9,
        "num_key_value_heads": 3, "max_position_embeddings": 8192})
    d = catalog.describe("smol")
    assert d["n_layers"] == 30 and d["hidden_size"] == 576
    assert d["kv_heads"] == 3 and d["head_dim"] == 64       # 576 / 9
    assert d["max_context"] == 8192 and d["modality"] == "text"
    assert d["quant"] is None


def test_describe_modality_text_by_default(monkeypatch):
    monkeypatch.setattr(catalog, "_read_config", lambda m: {
        "model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    assert catalog.describe("m")["modality"] == "text"


def test_describe_modality_vision(monkeypatch):
    monkeypatch.setattr(catalog, "_read_config", lambda m: {
        "model_type": "qwen2_5_vl", "architectures": ["Qwen2_5_VLForConditionalGeneration"]})
    assert catalog.describe("m")["modality"] == "vision"


def test_describe_modality_speech(monkeypatch):
    monkeypatch.setattr(catalog, "_read_config", lambda m: {
        "model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]})
    assert catalog.describe("m")["modality"] == "speech"


def test_describe_modality_embedding(monkeypatch):
    monkeypatch.setattr(catalog, "_read_config", lambda m: {
        "model_type": "modernbert", "architectures": ["ModernBertModel"]})
    assert catalog.describe("m")["modality"] == "embedding"


def test_describe_modality_from_model_id(monkeypatch):
    # config lacks a modality signal, but the repo name carries one (e.g. Kokoro TTS).
    monkeypatch.setattr(catalog, "_read_config", lambda m: {"model_type": "generic"})
    assert catalog.describe("org/Kokoro-82M")["modality"] == "speech"


def test_describe_kv_heads_defaults_to_attention_heads(monkeypatch):
    monkeypatch.setattr(catalog, "_read_config", lambda m: {
        "num_hidden_layers": 12, "hidden_size": 768, "num_attention_heads": 12,
        "max_position_embeddings": 2048})   # no num_key_value_heads (plain MHA)
    assert catalog.describe("m")["kv_heads"] == 12


def test_describe_reads_quantization(monkeypatch):
    monkeypatch.setattr(catalog, "_read_config", lambda m: {
        "num_attention_heads": 2, "hidden_size": 8,
        "quantization_config": {"quant_method": "bitsandbytes"}})
    assert catalog.describe("m")["quant"] == "bitsandbytes"


def test_describe_handles_missing_dims(monkeypatch):
    monkeypatch.setattr(catalog, "_read_config", lambda m: {})   # empty config
    d = catalog.describe("m")
    assert d["head_dim"] is None and d["n_layers"] is None


def test_describe_none_when_config_unavailable(monkeypatch):
    monkeypatch.setattr(catalog, "_read_config", lambda m: None)
    monkeypatch.setattr(catalog, "_describe_gguf", lambda m: None)
    assert catalog.describe("m") is None


# ---------------------------------------------------------------------------
# GGUF helpers
# ---------------------------------------------------------------------------

class _F:
    """Minimal ReaderField fake."""
    def __init__(self, v):
        self.v = v

    def contents(self):
        return self.v


def _gguf_fields_fake(include_kv=True):
    d = {
        "general.architecture": _F("llama"),
        "llama.block_count": _F(30),
        "llama.attention.head_count": _F(9),
        "llama.embedding_length": _F(576),
        "llama.context_length": _F(8192),
    }
    if include_kv:
        d["llama.attention.head_count_kv"] = _F(3)
    return d


# describe() dispatch tests

def test_describe_dispatches_to_gguf_when_no_config(monkeypatch):
    gguf_meta = {"modality": "text", "n_layers": 30, "hidden_size": 576,
                 "kv_heads": 3, "head_dim": 64, "max_context": 8192, "quant": "Q4_K_M"}
    monkeypatch.setattr(catalog, "_read_config", lambda m: None)
    monkeypatch.setattr(catalog, "_describe_gguf", lambda m: gguf_meta)
    assert catalog.describe("org/repo-Q4_K_M.gguf") == gguf_meta


def test_describe_transformers_path_unchanged(monkeypatch):
    monkeypatch.setattr(catalog, "_read_config", lambda m: {
        "num_hidden_layers": 30, "hidden_size": 576, "num_attention_heads": 9,
        "num_key_value_heads": 3, "max_position_embeddings": 8192})
    d = catalog.describe("smol")
    assert d["n_layers"] == 30 and d["modality"] == "text"


# _describe_gguf mapping

def test_describe_gguf_full_mapping(monkeypatch):
    monkeypatch.setattr(catalog, "_cached_gguf_path", lambda m: "org/repo-Q4_K_M.gguf")
    monkeypatch.setattr(catalog, "_gguf_fields", lambda p: _gguf_fields_fake(include_kv=True))
    d = catalog._describe_gguf("org/repo")
    assert d["modality"] == "text"
    assert d["n_layers"] == 30
    assert d["hidden_size"] == 576
    assert d["kv_heads"] == 3
    assert d["head_dim"] == 64        # 576 // 9
    assert d["max_context"] == 8192
    assert d["quant"] == "Q4_K_M"


def test_describe_gguf_kv_heads_falls_back_to_head_count(monkeypatch):
    monkeypatch.setattr(catalog, "_cached_gguf_path", lambda m: "org/repo-Q4_K_M.gguf")
    monkeypatch.setattr(catalog, "_gguf_fields", lambda p: _gguf_fields_fake(include_kv=False))
    d = catalog._describe_gguf("org/repo")
    assert d["kv_heads"] == 9


def test_describe_gguf_none_when_not_cached(monkeypatch):
    monkeypatch.setattr(catalog, "_cached_gguf_path", lambda m: None)
    assert catalog._describe_gguf("org/repo") is None


def test_describe_gguf_none_when_arch_missing(monkeypatch):
    monkeypatch.setattr(catalog, "_cached_gguf_path", lambda m: "some.gguf")
    monkeypatch.setattr(catalog, "_gguf_fields", lambda p: {})
    assert catalog._describe_gguf("org/repo") is None


def test_describe_gguf_none_on_reader_exception(monkeypatch):
    monkeypatch.setattr(catalog, "_cached_gguf_path", lambda m: "some.gguf")
    def boom(p):
        raise RuntimeError("bad file")
    monkeypatch.setattr(catalog, "_gguf_fields", boom)
    assert catalog._describe_gguf("org/repo") is None


# _gguf_fields wrapper

def test_gguf_fields_wrapper(monkeypatch):
    fake_fields = {"general.architecture": _F("llama")}
    class FakeReader:
        def __init__(self, path):
            self.fields = fake_fields
    monkeypatch.setattr(catalog.gguf, "GGUFReader", FakeReader)
    assert catalog._gguf_fields("some/path.gguf") is fake_fields


# _cached_gguf_path

import types as _types


def _make_cache(repos):
    return _types.SimpleNamespace(repos=repos)


def _make_repo(repo_id, repo_type, files):
    rev = _types.SimpleNamespace(files=files)
    return _types.SimpleNamespace(repo_id=repo_id, repo_type=repo_type, revisions=[rev])


def _make_file(name, path, size):
    return _types.SimpleNamespace(file_name=name, file_path=path, size_on_disk=size)


def test_cached_gguf_path_returns_smallest(monkeypatch):
    big = _make_file("big.gguf", "/cache/big.gguf", 2000)
    small = _make_file("small.gguf", "/cache/small.gguf", 500)
    repo = _make_repo("org/myrepo", "model", [big, small])
    cache = _make_cache([repo])
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: cache)
    result = catalog._cached_gguf_path("org/myrepo")
    assert result == "/cache/small.gguf"


def test_cached_gguf_path_none_when_repo_not_present(monkeypatch):
    repo = _make_repo("org/other", "model", [_make_file("m.gguf", "/c/m.gguf", 100)])
    cache = _make_cache([repo])
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: cache)
    assert catalog._cached_gguf_path("org/myrepo") is None


def test_cached_gguf_path_none_when_no_gguf_files(monkeypatch):
    # Repo found but has no .gguf files
    non_gguf = _make_file("tokenizer.json", "/c/tokenizer.json", 100)
    repo = _make_repo("org/myrepo", "model", [non_gguf])
    cache = _make_cache([repo])
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: cache)
    assert catalog._cached_gguf_path("org/myrepo") is None


def test_cached_gguf_path_none_on_scan_error(monkeypatch):
    def boom():
        raise RuntimeError("no cache")
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", boom)
    assert catalog._cached_gguf_path("org/myrepo") is None


# _cache_size_gb — on-disk weight, no network (Spec 2026-06-23-capability-pipeline, Slice 3)

def test_cache_size_gb_reads_local_cache(monkeypatch):
    repo = _make_repo("org/myrepo", "model", [])
    repo.size_on_disk = 4_200_000_000
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: _make_cache([repo]))
    assert catalog._cache_size_gb("org/myrepo") == 4.2


def test_cache_size_gb_none_when_not_cached(monkeypatch):
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: _make_cache([]))
    assert catalog._cache_size_gb("org/myrepo") is None


def test_cache_size_gb_none_when_size_zero(monkeypatch):
    repo = _make_repo("org/myrepo", "model", [])
    repo.size_on_disk = 0
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: _make_cache([repo]))
    assert catalog._cache_size_gb("org/myrepo") is None


def test_cache_size_gb_none_on_error(monkeypatch):
    def boom():
        raise RuntimeError("no cache")
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", boom)
    assert catalog._cache_size_gb("org/x") is None


# _quant_from_filename

def test_quant_from_filename_iq():
    assert catalog._quant_from_filename("a-IQ3_XS.gguf") == "IQ3_XS"


def test_quant_from_filename_q():
    assert catalog._quant_from_filename("a.Q4_K_M.gguf") == "Q4_K_M"


def test_quant_from_filename_f16():
    assert catalog._quant_from_filename("a-F16.gguf") == "F16"


def test_quant_from_filename_none():
    assert catalog._quant_from_filename("a-plain.gguf") is None


# ---------------------------------------------------------------------------
# Integration test (real local file, skipped if not present)
# ---------------------------------------------------------------------------

import os as _os
import pytest


LOCAL_GGUF_DIR = _os.path.expanduser(
    "~/.cache/huggingface/hub/models--bartowski--SmolLM2-135M-Instruct-GGUF"
)


@pytest.mark.integration
def test_describe_real_gguf():
    if not _os.path.isdir(LOCAL_GGUF_DIR):
        pytest.skip("SmolLM2-135M-Instruct-GGUF not in local HF cache")
    d = catalog.describe("bartowski/SmolLM2-135M-Instruct-GGUF")
    assert d is not None, "describe returned None for a cached GGUF repo"
    assert d["modality"] == "text"
    assert d["n_layers"] == 30
    assert d["max_context"] == 8192
    assert d["kv_heads"] == 3
    assert d["head_dim"] == 64


def test_remember_persists_metadata(store, monkeypatch):
    monkeypatch.setattr(catalog, "describe", lambda m: {"modality": "text", "n_layers": 30})
    monkeypatch.setattr(catalog, "_cache_size_gb", lambda m: None)
    row = catalog.remember(store, "smol")
    assert row["n_layers"] == 30
    assert catalog.get(store, "smol")["n_layers"] == 30


def test_remember_records_cache_weight(store, monkeypatch):
    # Spec 2026-06-23-capability-pipeline (Slice 3): the catalog stores each model's on-disk
    # weight (no network) so recommend can rank fits without re-reading the cache.
    monkeypatch.setattr(catalog, "describe", lambda m: {"modality": "text", "n_layers": 30})
    monkeypatch.setattr(catalog, "_cache_size_gb", lambda m: 4.2)
    assert catalog.remember(store, "smol")["weights_gb"] == 4.2


def test_remember_none_when_undescribable(store, monkeypatch):
    monkeypatch.setattr(catalog, "describe", lambda m: None)
    assert catalog.remember(store, "x") is None
    assert catalog.get(store, "x") is None


def test_all_models(store, monkeypatch):
    monkeypatch.setattr(catalog, "describe", lambda m: {"modality": "text"})
    monkeypatch.setattr(catalog, "_cache_size_gb", lambda m: None)
    catalog.remember(store, "a")
    catalog.remember(store, "b")
    assert {m["model_id"] for m in catalog.all_models(store)} == {"a", "b"}


def test_read_config_loads_json(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text('{"num_hidden_layers": 2}')
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda m, f: str(cfg))
    assert catalog._read_config("m") == {"num_hidden_layers": 2}


def test_read_config_none_on_error(monkeypatch):
    def boom(m, f):
        raise RuntimeError("no network")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    assert catalog._read_config("m") is None


def test_scan_catalogs_cached_models(store, monkeypatch):
    monkeypatch.setattr(catalog, "_hf_cache_models", lambda: ["a/m1", "b/m2"])
    monkeypatch.setattr(catalog, "describe", lambda mid: {"modality": "text"})
    monkeypatch.setattr(catalog, "_cache_size_gb", lambda m: None)
    assert catalog.scan(store) == 2
    assert {m["model_id"] for m in catalog.all_models(store)} == {"a/m1", "b/m2"}


def test_scan_skips_undescribable(store, monkeypatch):
    monkeypatch.setattr(catalog, "_hf_cache_models", lambda: ["a/m1", "bad/m"])
    monkeypatch.setattr(catalog, "describe",
                        lambda mid: {"modality": "text"} if mid == "a/m1" else None)
    monkeypatch.setattr(catalog, "_cache_size_gb", lambda m: None)
    assert catalog.scan(store) == 1


def test_hf_cache_models_filters_to_models(monkeypatch):
    import types
    cache = types.SimpleNamespace(repos=[
        types.SimpleNamespace(repo_id="org/model", repo_type="model"),
        types.SimpleNamespace(repo_id="org/data", repo_type="dataset"),
    ])
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: cache)
    assert catalog._hf_cache_models() == ["org/model"]


def test_hf_cache_models_empty_on_error(monkeypatch):
    def boom():
        raise RuntimeError("no cache")
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", boom)
    assert catalog._hf_cache_models() == []
