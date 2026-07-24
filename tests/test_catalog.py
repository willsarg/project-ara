# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""catalog.py — model metadata (describe) + the db-backed model catalog."""
from __future__ import annotations

import json
from pathlib import Path

from ara import catalog, db


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


def test_describe_gguf_reads_a_local_path_directly(tmp_path, monkeypatch):
    # A loose local .gguf file is read directly, NOT via the HF cache — the local-model-library
    # path. Slug: 2026-06-25-local-gguf-cli-support
    f = tmp_path / "MyModel-Q4_K_M.gguf"
    f.write_bytes(b"\x00")

    def _no_cache(m):
        raise AssertionError("a local .gguf path must not consult the HF cache")

    monkeypatch.setattr(catalog, "_cached_gguf_path", _no_cache)
    monkeypatch.setattr(catalog, "_gguf_fields", lambda p: _gguf_fields_fake(include_kv=True))
    d = catalog._describe_gguf(str(f))
    assert d["n_layers"] == 30 and d["quant"] == "Q4_K_M"


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


def _make_repo(repo_id, repo_type, files, *, revisions=None):
    rev = _types.SimpleNamespace(files=files, refs={"main"})
    return _types.SimpleNamespace(
        repo_id=repo_id, repo_type=repo_type, revisions=revisions or [rev])


def _make_file(name, path, size):
    return _types.SimpleNamespace(file_name=name, file_path=path, size_on_disk=size)


def test_artifact_evidence_resolves_local_gguf_and_exact_selector(tmp_path):
    local = tmp_path / "Model-Q4_K_M.gguf"
    local.write_bytes(b"gguf")

    assert catalog.artifact_evidence(str(local)) == {
        "status": "resolved",
        "kind": "gguf",
        "source": "local_path",
        "reason": None,
    }
    assert catalog.artifact_evidence(
        "org/repo:Model-Q4_K_M.gguf") == {
            "status": "resolved",
            "kind": "gguf",
            "source": "exact_selector",
            "reason": None,
        }


def test_artifact_evidence_resolves_cached_transformers_weights(monkeypatch):
    files = [
        _make_file("config.json", "/cache/config.json", 100),
        _make_file("model-00001-of-00002.safetensors", "/cache/one", 1000),
        _make_file("model-00002-of-00002.safetensors", "/cache/two", 1000),
    ]
    repo = _make_repo("org/transformer", "model", files)
    monkeypatch.setattr(
        "huggingface_hub.scan_cache_dir", lambda: _make_cache([repo]))

    assert catalog.artifact_evidence("org/transformer") == {
        "status": "resolved",
        "kind": "transformers",
        "source": "local_cache",
        "reason": None,
    }


def test_artifact_evidence_keeps_bare_gguf_repo_unresolved(monkeypatch):
    files = [
        _make_file("Model-Q4_K_M.gguf", "/cache/q4.gguf", 1000),
        _make_file("Model-Q8_0.gguf", "/cache/q8.gguf", 2000),
    ]
    repo = _make_repo("org/gguf", "model", files)
    monkeypatch.setattr(
        "huggingface_hub.scan_cache_dir", lambda: _make_cache([repo]))

    assert catalog.artifact_evidence("org/gguf") == {
        "status": "unresolved",
        "kind": "gguf",
        "source": "local_cache",
        "reason": "exact_gguf_selector_required",
    }


def test_artifact_evidence_keeps_mixed_or_incomplete_repo_unresolved(monkeypatch):
    mixed = _make_repo("org/mixed", "model", [
        _make_file("model.safetensors", "/cache/model.safetensors", 1000),
        _make_file("Model-Q4_K_M.gguf", "/cache/model.gguf", 500),
    ])
    incomplete = _make_repo("org/incomplete", "model", [
        _make_file("config.json", "/cache/config.json", 100),
    ])
    monkeypatch.setattr(
        "huggingface_hub.scan_cache_dir",
        lambda: _make_cache([mixed, incomplete]))

    assert catalog.artifact_evidence("org/mixed") == {
        "status": "unresolved",
        "kind": "mixed",
        "source": "local_cache",
        "reason": "ambiguous_formats",
    }
    assert catalog.artifact_evidence("org/incomplete") == {
        "status": "unresolved",
        "kind": None,
        "source": "local_cache",
        "reason": "weight_artifact_unavailable",
    }


def test_artifact_evidence_handles_uncached_invalid_and_probe_failure(monkeypatch):
    monkeypatch.setattr(
        "huggingface_hub.scan_cache_dir", lambda: _make_cache([]))
    assert catalog.artifact_evidence("org/missing") == {
        "status": "unresolved",
        "kind": None,
        "source": "local_cache",
        "reason": "artifact_not_cached",
    }
    assert catalog.artifact_evidence("../bad") == {
        "status": "unresolved",
        "kind": None,
        "source": "input",
        "reason": "invalid_model_reference",
    }

    monkeypatch.setattr(
        "huggingface_hub.scan_cache_dir",
        lambda: (_ for _ in ()).throw(RuntimeError("broken cache")))
    assert catalog.artifact_evidence("org/broken")["reason"] == (
        "artifact_evidence_unavailable")


def test_cached_gguf_path_returns_smallest(monkeypatch):
    big = _make_file("big.gguf", "/cache/big.gguf", 2000)
    small = _make_file("small.gguf", "/cache/small.gguf", 500)
    repo = _make_repo("org/myrepo", "model", [big, small])
    cache = _make_cache([repo])
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: cache)
    result = catalog._cached_gguf_path("org/myrepo")
    assert result == "/cache/small.gguf"


def test_cached_gguf_path_resolves_exact_repo_selector(monkeypatch):
    first = _make_file("Model-Q4_K_M.gguf", "/cache/q4.gguf", 500)
    selected = _make_file("Model-Q8_0.gguf", "/cache/q8.gguf", 1000)
    repo = _make_repo("org/myrepo", "model", [first, selected])
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: _make_cache([repo]))
    assert catalog._cached_gguf_path("org/myrepo:Model-Q8_0.gguf") == "/cache/q8.gguf"
    assert catalog._cached_gguf_path("org/myrepo:missing.gguf") is None


def test_cached_gguf_path_uses_current_main_revision_only(monkeypatch):
    old_small = _make_file("Model-Q2_K.gguf", "/cache/old-q2.gguf", 100)
    current = _make_file("Model-Q4_K_M.gguf", "/cache/current-q4.gguf", 500)
    old_rev = _types.SimpleNamespace(files=[old_small], refs=set())
    main_rev = _types.SimpleNamespace(files=[current], refs={"main"})
    repo = _make_repo("org/myrepo", "model", [], revisions=[old_rev, main_rev])
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: _make_cache([repo]))
    assert catalog._cached_gguf_path("org/myrepo") == "/cache/current-q4.gguf"
    assert catalog._cached_gguf_path("org/myrepo:Model-Q2_K.gguf") is None


def test_describe_repo_selector_reads_selected_gguf_not_repo_config(monkeypatch):
    monkeypatch.setattr(catalog, "_read_config",
                        lambda _model: pytest.fail("selector must not be passed as a repo id"))
    monkeypatch.setattr(catalog, "_cached_gguf_path", lambda _model: "/cache/selected.gguf")
    monkeypatch.setattr(catalog, "_gguf_fields", lambda _path: {
        "general.architecture": _F("llama"),
        "llama.block_count": _F(12),
        "llama.embedding_length": _F(768),
        "llama.attention.head_count": _F(12),
        "llama.attention.head_count_kv": _F(4),
        "llama.context_length": _F(4096),
    })
    meta = catalog.describe("org/myrepo:Model-Q8_0.gguf")
    assert meta is not None and meta["n_layers"] == 12


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


def test_remember_variant_preserves_repo_metadata_and_exact_quant(store, monkeypatch):
    monkeypatch.setattr(catalog, "remember", lambda con, model: (
        db.upsert_model(con, model, modality="text", n_layers=32, hidden_size=4096,
                        kv_heads=8, head_dim=128, max_context=8192, weights_gb=10.0)
        or db.get_model(con, model)))
    row = catalog.remember_variant(
        store, "org/repo:Model-Q4_K_M.gguf", "org/repo",
        quant="q4_k_m", weights_gb=4.2)
    assert row["model_id"] == "org/repo:Model-Q4_K_M.gguf"
    assert row["quant"] == "q4_k_m" and row["weights_gb"] == 4.2
    assert row["n_layers"] == 32


def test_remember_variant_returns_none_when_repo_is_undescribable(store, monkeypatch):
    monkeypatch.setattr(catalog, "remember", lambda *_a: None)
    assert catalog.remember_variant(store, "org/x:m.gguf", "org/x",
                                    quant="q4", weights_gb=1.0) is None
    assert catalog.get(store, "x") is None


def test_all_models(store, monkeypatch):
    monkeypatch.setattr(catalog, "describe", lambda m: {"modality": "text"})
    monkeypatch.setattr(catalog, "_cache_size_gb", lambda m: None)
    catalog.remember(store, "a")
    catalog.remember(store, "b")
    assert {m["model_id"] for m in catalog.all_models(store)} == {"a", "b"}


def test_read_config_loads_json(tmp_path, monkeypatch):
    # Recon (catalog.scan) is read-only per the hard rule — hf_hub_download must be called
    # with local_files_only=True so a cached model never triggers a network etag round-trip.
    cfg = tmp_path / "config.json"
    cfg.write_text('{"num_hidden_layers": 2}')

    def fake_hf_hub_download(m, f, **kwargs):
        assert kwargs.get("local_files_only") is True
        return str(cfg)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    assert catalog._read_config("m") == {"num_hidden_layers": 2}


def test_read_config_none_on_error(monkeypatch):
    def boom(m, f, **kwargs):
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


# ---------------------------------------------------------------------------
# safetensors header helpers  (2026-06-24-safetensors-fallback)
# ---------------------------------------------------------------------------

import struct as _struct


def _make_safetensors(tensor_map: dict, tmp_path) -> str:
    """Write a minimal synthetic .safetensors file (header only, zero data bytes).

    Format: 8-byte LE uint64 = header length, then N UTF-8 JSON bytes.
    Data offsets are bogus (0,0) — we never read past the header.
    """
    header_bytes = json.dumps(tensor_map).encode()
    path = tmp_path / "model.safetensors"
    with open(path, "wb") as fh:
        fh.write(_struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
    return str(path)


def _make_index_json(tensor_weight_map: dict, tmp_path) -> str:
    """Write a minimal model.safetensors.index.json."""
    index = {"metadata": {}, "weight_map": tensor_weight_map}
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(json.dumps(index))
    return str(path)


# _read_safetensors_header

def test_read_safetensors_header_parses_json(tmp_path):
    tensor_map = {
        "__metadata__": {},
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [32000, 4096], "data_offsets": [0, 0]},
    }
    path = _make_safetensors(tensor_map, tmp_path)
    result = catalog._read_safetensors_header(path)
    assert result["model.embed_tokens.weight"]["shape"] == [32000, 4096]


def test_read_safetensors_header_none_on_bad_file(tmp_path):
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"\x00\x01")   # truncated — can't read 8-byte length
    assert catalog._read_safetensors_header(str(bad)) is None


def test_read_safetensors_header_none_on_truncated_body(tmp_path):
    """8-byte header says length=1000 but only 5 bytes of body follow."""
    truncated = tmp_path / "truncated.safetensors"
    with open(truncated, "wb") as fh:
        fh.write(_struct.pack("<Q", 1000))   # claims 1000 bytes
        fh.write(b"hello")                   # only 5 bytes follow
    assert catalog._read_safetensors_header(str(truncated)) is None


def test_read_safetensors_header_none_on_missing_file():
    assert catalog._read_safetensors_header("/nonexistent/model.safetensors") is None


# _cached_safetensors_paths

def test_cached_safetensors_paths_returns_st_files(monkeypatch):
    st_file = _make_file("model.safetensors", "/cache/model.safetensors", 4_000_000_000)
    other = _make_file("tokenizer.json", "/cache/tokenizer.json", 100)
    repo = _make_repo("org/myrepo", "model", [st_file, other])
    cache = _make_cache([repo])
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: cache)
    result = catalog._cached_safetensors_paths("org/myrepo")
    assert result == ["/cache/model.safetensors"]


def test_cached_safetensors_paths_returns_empty_when_none(monkeypatch):
    repo = _make_repo("org/myrepo", "model", [_make_file("config.json", "/c/config.json", 100)])
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: _make_cache([repo]))
    assert catalog._cached_safetensors_paths("org/myrepo") == []


def test_cached_safetensors_paths_empty_when_repo_missing(monkeypatch):
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: _make_cache([]))
    assert catalog._cached_safetensors_paths("org/myrepo") == []


def test_cached_safetensors_paths_none_on_scan_error(monkeypatch):
    def boom():
        raise RuntimeError("no cache")
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", boom)
    assert catalog._cached_safetensors_paths("org/myrepo") == []


def test_cached_safetensors_paths_includes_index_json(monkeypatch):
    idx = _make_file("model.safetensors.index.json", "/c/model.safetensors.index.json", 50_000)
    shard = _make_file("model-00001-of-00002.safetensors", "/c/model-00001-of-00002.safetensors", 4_000_000_000)
    repo = _make_repo("org/big", "model", [idx, shard])
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: _make_cache([repo]))
    paths = catalog._cached_safetensors_paths("org/big")
    assert "/c/model.safetensors.index.json" in paths
    assert "/c/model-00001-of-00002.safetensors" in paths


def test_cached_safetensors_paths_uses_current_main_revision_only(monkeypatch):
    stale = _make_file("model.safetensors", "/old/model.safetensors", 10)
    current = _make_file("model.safetensors", "/main/model.safetensors", 20)
    old_rev = _types.SimpleNamespace(files=[stale], refs=set())
    main_rev = _types.SimpleNamespace(files=[current], refs={"main"})
    repo = _make_repo("org/myrepo", "model", [], revisions=[old_rev, main_rev])
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: _make_cache([repo]))
    assert catalog._cached_safetensors_paths("org/myrepo") == ["/main/model.safetensors"]


# _describe_safetensors — full recovery

def test_describe_safetensors_recovers_n_layers_and_hidden(tmp_path, monkeypatch):
    """Standard single-file Llama-like model: embed + 4 layers."""
    tensor_map = {
        "__metadata__": {},
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [32000, 512], "data_offsets": [0, 0]},
        "model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [512, 512], "data_offsets": [0, 0]},
        "model.layers.1.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [512, 512], "data_offsets": [0, 0]},
        "model.layers.2.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [512, 512], "data_offsets": [0, 0]},
        "model.layers.3.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [512, 512], "data_offsets": [0, 0]},
        "model.norm.weight": {"dtype": "BF16", "shape": [512], "data_offsets": [0, 0]},
        "lm_head.weight": {"dtype": "BF16", "shape": [32000, 512], "data_offsets": [0, 0]},
    }
    st_path = _make_safetensors(tensor_map, tmp_path)
    monkeypatch.setattr(catalog, "_cached_safetensors_paths", lambda m: [st_path])
    d = catalog._describe_safetensors("org/mymodel")
    assert d is not None
    assert d["n_layers"] == 4
    assert d["hidden_size"] == 512
    # fields we deliberately do NOT infer from safetensors alone:
    assert d["kv_heads"] is None
    assert d["head_dim"] is None
    assert d["max_context"] is None
    assert d["quant"] is None
    assert d["modality"] == "text"


def test_describe_safetensors_uses_index_json_for_layer_count(tmp_path, monkeypatch):
    """Sharded model: index.json lists all tensors; one shard has only some layers."""
    # Only layer 0 is in the single cached shard's header, but index lists 0-7
    tensor_map = {
        "__metadata__": {},
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [32000, 1024], "data_offsets": [0, 0]},
        "model.layers.0.mlp.gate_proj.weight": {"dtype": "BF16", "shape": [1024, 1024], "data_offsets": [0, 0]},
    }
    shard_path = _make_safetensors(tensor_map, tmp_path)
    named_shard = tmp_path / "model-00001-of-00002.safetensors"
    Path(shard_path).rename(named_shard)
    shard_path = str(named_shard)

    # index.json weight_map references layers 0-7
    weight_map = {f"model.layers.{i}.mlp.gate_proj.weight": "model-00001-of-00002.safetensors"
                  for i in range(8)}
    weight_map["model.embed_tokens.weight"] = "model-00001-of-00002.safetensors"
    index_path = _make_index_json(weight_map, tmp_path)

    monkeypatch.setattr(catalog, "_cached_safetensors_paths", lambda m: [index_path, shard_path])
    d = catalog._describe_safetensors("org/bigmodel")
    assert d is not None
    assert d["n_layers"] == 8     # from index.json
    assert d["hidden_size"] == 1024   # from shard header


def test_describe_safetensors_none_when_no_safetensors_cached(monkeypatch):
    monkeypatch.setattr(catalog, "_cached_safetensors_paths", lambda m: [])
    assert catalog._describe_safetensors("org/mymodel") is None


def test_describe_safetensors_falls_back_to_shard_when_index_has_no_layers(tmp_path, monkeypatch):
    """index.json exists but weight_map has no layer tensors; shard header supplies n_layers."""
    # Shard has layer tensors + embed
    tensor_map = {
        "__metadata__": {},
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [32000, 256], "data_offsets": [0, 0]},
        "model.layers.0.mlp.weight": {"dtype": "BF16", "shape": [256, 256], "data_offsets": [0, 0]},
        "model.layers.1.mlp.weight": {"dtype": "BF16", "shape": [256, 256], "data_offsets": [0, 0]},
    }
    shard_path = _make_safetensors(tensor_map, tmp_path)

    # index.json weight_map has only non-layer tensors (no layer-pattern match → empty indices)
    weight_map = {"model.embed_tokens.weight": "model.safetensors"}
    index_path = _make_index_json(weight_map, tmp_path)

    monkeypatch.setattr(catalog, "_cached_safetensors_paths", lambda m: [index_path, shard_path])
    d = catalog._describe_safetensors("org/mymodel")
    assert d is not None
    assert d["n_layers"] == 2     # from shard header fallback
    assert d["hidden_size"] == 256


def test_describe_safetensors_refuses_corrupt_index(tmp_path, monkeypatch):
    shard_tensor_map = {
        "__metadata__": {},
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [32000, 128], "data_offsets": [0, 0]},
        "model.layers.0.mlp.weight": {"dtype": "BF16", "shape": [128, 128], "data_offsets": [0, 0]},
    }
    shard_path = _make_safetensors(shard_tensor_map, tmp_path)

    bad_index = tmp_path / "model.safetensors.index.json"
    bad_index.write_bytes(b"not valid json {{{")

    monkeypatch.setattr(catalog, "_cached_safetensors_paths",
                        lambda m: [str(bad_index), shard_path])
    assert catalog._describe_safetensors("org/mymodel") is None


def test_describe_safetensors_refuses_incomplete_sharded_snapshot(tmp_path, monkeypatch):
    shard_path = _make_safetensors({
        "model.layers.0.mlp.weight": {
            "dtype": "BF16", "shape": [128, 128], "data_offsets": [0, 0]}}, tmp_path)
    index_path = _make_index_json({
        "model.layers.0.mlp.weight": "model-00001-of-00002.safetensors",
        "model.layers.1.mlp.weight": "model-00002-of-00002.safetensors",
    }, tmp_path)
    monkeypatch.setattr(catalog, "_cached_safetensors_paths",
                        lambda _m: [index_path, shard_path])
    assert catalog._describe_safetensors("org/mymodel") is None


def test_describe_safetensors_refuses_multiple_indexes(tmp_path, monkeypatch):
    first = tmp_path / "model.safetensors.index.json"
    second = tmp_path / "other.safetensors.index.json"
    first.write_text("{}")
    second.write_text("{}")
    monkeypatch.setattr(catalog, "_cached_safetensors_paths",
                        lambda _m: [str(first), str(second)])
    assert catalog._describe_safetensors("org/mymodel") is None


def test_describe_safetensors_partial_when_no_embed_tensor(tmp_path, monkeypatch):
    """If there's no embed_tokens.weight (or similar), hidden_size stays None; n_layers still inferred."""
    tensor_map = {
        "__metadata__": {},
        "model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [256, 256], "data_offsets": [0, 0]},
        "model.layers.1.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [256, 256], "data_offsets": [0, 0]},
    }
    st_path = _make_safetensors(tensor_map, tmp_path)
    monkeypatch.setattr(catalog, "_cached_safetensors_paths", lambda m: [st_path])
    d = catalog._describe_safetensors("org/mymodel")
    assert d is not None
    assert d["n_layers"] == 2
    assert d["hidden_size"] is None   # no embedding tensor to read from


def test_describe_safetensors_none_when_no_layer_tensors(tmp_path, monkeypatch):
    """If we can't find any layer tensors, return None (not useful partial info)."""
    tensor_map = {
        "__metadata__": {},
        "some_random.weight": {"dtype": "BF16", "shape": [100, 100], "data_offsets": [0, 0]},
    }
    st_path = _make_safetensors(tensor_map, tmp_path)
    monkeypatch.setattr(catalog, "_cached_safetensors_paths", lambda m: [st_path])
    assert catalog._describe_safetensors("org/mymodel") is None


def test_describe_safetensors_none_on_unreadable_file(tmp_path, monkeypatch):
    """Corrupt safetensors header returns None — no crash."""
    bad = tmp_path / "model.safetensors"
    bad.write_bytes(b"\xff" * 8 + b"not json at all")
    monkeypatch.setattr(catalog, "_cached_safetensors_paths", lambda m: [str(bad)])
    assert catalog._describe_safetensors("org/mymodel") is None


# describe() dispatch — safetensors fallback path

def test_describe_dispatches_to_safetensors_when_no_config_and_no_gguf(monkeypatch):
    """Third fallback: no config.json, no GGUF, but safetensors are cached."""
    st_meta = {"modality": "text", "n_layers": 8, "hidden_size": 1024,
               "kv_heads": None, "head_dim": None, "max_context": None, "quant": None}
    monkeypatch.setattr(catalog, "_read_config", lambda m: None)
    monkeypatch.setattr(catalog, "_describe_gguf", lambda m: None)
    monkeypatch.setattr(catalog, "_describe_safetensors", lambda m: st_meta)
    assert catalog.describe("org/mymodel") == st_meta


def test_describe_safetensors_not_called_when_gguf_succeeds(monkeypatch):
    """GGUF result short-circuits safetensors lookup."""
    gguf_meta = {"modality": "text", "n_layers": 30, "hidden_size": 576,
                 "kv_heads": 3, "head_dim": 64, "max_context": 8192, "quant": "Q4_K_M"}
    called = []
    monkeypatch.setattr(catalog, "_read_config", lambda m: None)
    monkeypatch.setattr(catalog, "_describe_gguf", lambda m: gguf_meta)
    monkeypatch.setattr(catalog, "_describe_safetensors", lambda m: called.append(m) or {})
    result = catalog.describe("org/repo")
    assert result == gguf_meta
    assert called == []  # safetensors was never consulted


def test_describe_safetensors_not_called_when_config_succeeds(monkeypatch):
    """config.json result short-circuits both GGUF and safetensors lookups."""
    called = []
    monkeypatch.setattr(catalog, "_read_config", lambda m: {
        "num_hidden_layers": 12, "hidden_size": 768, "num_attention_heads": 12})
    monkeypatch.setattr(catalog, "_describe_safetensors", lambda m: called.append(m) or {})
    catalog.describe("org/repo")
    assert called == []
