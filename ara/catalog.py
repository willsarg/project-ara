"""The model catalog — metadata ARA knows about each model, and where it's remembered.

``describe`` reads a model's architecture from its Hugging Face ``config.json`` (the
transformers analog of wmx-suite's ``models.describe``); the rest stores/reads that metadata
in ARA's db. Discovery seeds this (a model ARA finds or characterizes gets remembered); a
curated by-modality list is a later concern (recommend's input).
"""
from __future__ import annotations

import json
import os
import re

import gguf

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


def _infer_modality(cfg: dict, model_id: str = "") -> str:
    """Best-effort modality from a model's config + id — defaults to text (causal LM)."""
    blob = ((cfg.get("model_type") or "") + " "
            + " ".join(cfg.get("architectures") or []) + " " + model_id).lower()
    if any(k in blob for k in ("vl", "vision", "image", "llava", "clip")):
        return "vision"
    if any(k in blob for k in ("whisper", "wav2vec", "audio", "speech", "kokoro", "tts")):
        return "speech"
    if any(k in blob for k in ("bert", "embed", "roberta")):
        return "embedding"
    return "text"


def _cached_gguf_path(model_id: str) -> str | None:
    """Return the path to the smallest cached .gguf for *model_id*, or None (no download)."""
    from huggingface_hub import scan_cache_dir

    try:
        cache = scan_cache_dir()
        repo = next(
            (r for r in cache.repos
             if r.repo_id == model_id and r.repo_type == "model"),
            None,
        )
        if repo is None:
            return None
        gguf_files = [
            f
            for rev in repo.revisions
            for f in rev.files
            if f.file_name.endswith(".gguf")
        ]
        if not gguf_files:
            return None
        return str(min(gguf_files, key=lambda f: f.size_on_disk).file_path)
    except Exception:
        return None


def _gguf_fields(path: str):
    """Thin wrapper around GGUFReader so tests can monkeypatch it."""
    return gguf.GGUFReader(path).fields


def _fval(fields, name, default=None):
    """Return the decoded value of a GGUF field, or *default* if absent."""
    f = fields.get(name)
    return f.contents() if f is not None else default


def _quant_from_filename(path: str) -> str | None:
    """Extract a quantisation token from a .gguf filename, or None."""
    m = re.search(
        r'[.-]((?:IQ|Q)\d[A-Za-z0-9_]*|F16|F32|BF16)\.gguf$',
        os.path.basename(path),
        re.I,
    )
    return m.group(1) if m else None


def _describe_gguf(model_id: str) -> dict | None:
    """Architecture metadata from a cached GGUF file header, or None if unavailable."""
    path = _cached_gguf_path(model_id)
    if path is None:
        return None
    try:
        fields = _gguf_fields(path)
        arch = _fval(fields, "general.architecture")
        if not arch:
            return None
        n_heads = _fval(fields, f"{arch}.attention.head_count")
        hidden = _fval(fields, f"{arch}.embedding_length")
        return {
            "modality": _infer_modality({"model_type": arch}, model_id),
            "n_layers": _fval(fields, f"{arch}.block_count"),
            "hidden_size": hidden,
            "kv_heads": _fval(fields, f"{arch}.attention.head_count_kv", n_heads),
            "head_dim": hidden // n_heads if (hidden and n_heads) else None,
            "max_context": _fval(fields, f"{arch}.context_length"),
            "quant": _quant_from_filename(path),
        }
    except Exception:
        return None


def describe(model_id: str) -> dict | None:
    """Architecture metadata for *model_id* from its HF config, or None if unavailable."""
    cfg = _read_config(model_id)
    if cfg is None:
        return _describe_gguf(model_id)
    n_heads = cfg.get("num_attention_heads")
    hidden = cfg.get("hidden_size")
    quant = cfg.get("quantization_config") or {}
    return {
        "modality": _infer_modality(cfg, model_id),
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


def _hf_cache_models() -> list[str]:
    """Model repo ids currently in the local Hugging Face cache (no network)."""
    from huggingface_hub import scan_cache_dir

    try:
        return [r.repo_id for r in scan_cache_dir().repos if r.repo_type == "model"]
    except Exception:
        return []


def scan(con) -> int:
    """Catalog every describable model in the HF cache; return how many were added/updated."""
    n = 0
    for model_id in _hf_cache_models():
        if remember(con, model_id) is not None:
            n += 1
    return n
