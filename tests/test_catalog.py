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
    assert catalog.describe("m") is None


def test_remember_persists_metadata(store, monkeypatch):
    monkeypatch.setattr(catalog, "describe", lambda m: {"modality": "text", "n_layers": 30})
    row = catalog.remember(store, "smol")
    assert row["n_layers"] == 30
    assert catalog.get(store, "smol")["n_layers"] == 30


def test_remember_none_when_undescribable(store, monkeypatch):
    monkeypatch.setattr(catalog, "describe", lambda m: None)
    assert catalog.remember(store, "x") is None
    assert catalog.get(store, "x") is None


def test_all_models(store, monkeypatch):
    monkeypatch.setattr(catalog, "describe", lambda m: {"modality": "text"})
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
