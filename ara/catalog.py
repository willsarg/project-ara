"""The model catalog — metadata ARA knows about each model, and where it's remembered.

``describe`` reads a model's architecture from its Hugging Face ``config.json`` (the
transformers analog of wmx-suite's ``models.describe``); the rest stores/reads that metadata
in ARA's db. Discovery seeds this (a model ARA finds or characterizes gets remembered); a
curated by-modality list is a later concern (recommend's input).
"""
from __future__ import annotations

import json

from ara import db


def _read_config(model_id: str) -> dict | None:
    """Fetch and parse a model's ``config.json`` from the HF cache/hub, or None on failure."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(model_id, "config.json")
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def describe(model_id: str) -> dict | None:
    """Architecture metadata for *model_id* from its HF config, or None if unavailable."""
    cfg = _read_config(model_id)
    if cfg is None:
        return None
    n_heads = cfg.get("num_attention_heads")
    hidden = cfg.get("hidden_size")
    quant = cfg.get("quantization_config") or {}
    return {
        "modality": "text",   # MVP: causal LM; infer other modalities later
        "n_layers": cfg.get("num_hidden_layers"),
        "hidden_size": hidden,
        "kv_heads": cfg.get("num_key_value_heads", n_heads),
        "head_dim": hidden // n_heads if (hidden and n_heads) else None,
        "max_context": cfg.get("max_position_embeddings"),
        "quant": quant.get("quant_method") if quant else None,
    }


def remember(con, model_id: str) -> dict | None:
    """Describe *model_id* and store it in the catalog; return the stored row (or None)."""
    meta = describe(model_id)
    if meta is None:
        return None
    db.upsert_model(con, model_id, **meta)
    return db.get_model(con, model_id)


def get(con, model_id: str) -> dict | None:
    return db.get_model(con, model_id)


def all_models(con) -> list[dict]:
    return db.list_models(con)
