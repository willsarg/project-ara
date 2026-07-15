# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The model catalog — metadata ARA knows about each model, and where it's remembered.

``describe`` reads a model's architecture from its Hugging Face ``config.json`` (the
transformers analog of the native MLX engine's model description); the rest stores/reads that metadata
in ARA's db. Discovery seeds this (a model ARA finds or characterizes gets remembered); a
curated by-modality list is a later concern (recommend's input).
"""
from __future__ import annotations

import json
import os
import re
import struct

import gguf

from ara import db


def _read_config(model_id: str) -> dict | None:
    """Parse a model's ``config.json`` from the local HF cache, or None on failure.

    ``local_files_only=True`` — this runs inside ``catalog.scan``, a bulk *local* recon sweep
    over already-cached models, and recon must never touch the network (hard rule). A model
    that isn't cached simply falls through to the except below, same as any other failure.
    """
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(model_id, "config.json", local_files_only=True)
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
    """Return the selected cached GGUF, or the smallest GGUF for a bare repo (no download)."""
    from huggingface_hub import scan_cache_dir

    try:
        repo_id, separator, selected_name = model_id.partition(":")
        exact_selector = separator and selected_name.lower().endswith(".gguf")
        if not exact_selector:
            repo_id, selected_name = model_id, ""
        cache = scan_cache_dir()
        repo = next(
            (r for r in cache.repos
             if r.repo_id == repo_id and r.repo_type == "model"),
            None,
        )
        if repo is None:
            return None
        main_revisions = [rev for rev in repo.revisions if "main" in getattr(rev, "refs", ())]
        gguf_files = [
            f
            for rev in main_revisions
            for f in rev.files
            if f.file_name.endswith(".gguf")
            and (exact_selector or not f.file_name.lower().startswith("mmproj-"))
            and (not exact_selector or f.file_name == selected_name)
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
    """Architecture metadata from a GGUF file header, or None if unavailable.

    *model_id* may be a local ``.gguf`` path (read directly — a loose file on disk, e.g. a local
    model library) or an HF repo id (located in the cache). (Slug: 2026-06-25-local-gguf-cli-support)
    """
    path = (model_id if (model_id.endswith(".gguf") and os.path.isfile(model_id))
            else _cached_gguf_path(model_id))
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


def _read_safetensors_header(path: str) -> dict | None:
    """Parse the JSON header of a .safetensors file without loading weight data.

    Format: 8-byte little-endian uint64 giving header length N, then N bytes of
    UTF-8 JSON mapping tensor-name → {dtype, shape, data_offsets} plus optional
    ``__metadata__``.  Returns the parsed dict, or None on any error.
    """
    try:
        with open(path, "rb") as fh:
            size_bytes = fh.read(8)
            if len(size_bytes) < 8:
                return None
            (header_len,) = struct.unpack("<Q", size_bytes)
            header_bytes = fh.read(header_len)
            if len(header_bytes) < header_len:
                return None
            return json.loads(header_bytes.decode())
    except Exception:
        return None


def _cached_safetensors_paths(model_id: str) -> list[str]:
    """Return paths to all cached .safetensors and .safetensors.index.json files for
    *model_id* in the local HF cache (no download).  Returns [] on failure or if absent.
    """
    from huggingface_hub import scan_cache_dir

    try:
        cache = scan_cache_dir()
        repo = next(
            (r for r in cache.repos
             if r.repo_id == model_id and r.repo_type == "model"),
            None,
        )
        if repo is None:
            return []
        return [
            str(f.file_path)
            for rev in repo.revisions
            if "main" in getattr(rev, "refs", ())
            for f in rev.files
            if f.file_name.endswith(".safetensors")
            or f.file_name.endswith(".safetensors.index.json")
        ]
    except Exception:
        return []


def _describe_safetensors(model_id: str) -> dict | None:
    """Architecture metadata inferred from cached .safetensors header(s), or None.

    Recovery strategy:
    - ``n_layers``: prefer model.safetensors.index.json (lists ALL tensor names across
      shards); fall back to counting distinct layer indices in a single shard's header.
    - ``hidden_size``: read from model.embed_tokens.weight shape (shape[1] = hidden dim).
    - ``kv_heads``, ``head_dim``, ``max_context``, ``quant``: deliberately left None.
      kv_heads/head_dim require cross-referencing q/k projection shapes with the head
      count — unreliable without config.json.  max_context and quant are not stored in
      weight tensors at all.  Returning None is correct per the honesty constraint
      (Rule #3: don't fabricate).
    - Returns None when no .safetensors are cached or layer tensors can't be found
      (not useful to return a completely empty description).
    """
    paths = _cached_safetensors_paths(model_id)
    if not paths:
        return None

    # Separate index.json from weight shards
    index_paths = [p for p in paths if p.endswith(".safetensors.index.json")]
    shard_paths = [p for p in paths if p.endswith(".safetensors")]
    if len(index_paths) > 1 or (not index_paths and len(shard_paths) > 1):
        return None

    # --- n_layers: prefer index.json (covers all shards) ---
    _LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
    n_layers: int | None = None

    if index_paths:
        try:
            with open(index_paths[0]) as fh:
                index = json.load(fh)
            weight_map = index.get("weight_map", {})
            cached_shards = {os.path.basename(path) for path in shard_paths}
            if not isinstance(weight_map, dict) or not set(weight_map.values()) <= cached_shards:
                return None
            indices = {int(m.group(1)) for k in weight_map for m in [_LAYER_RE.search(k)] if m}
            if indices:
                n_layers = max(indices) + 1
        except Exception:
            return None

    # --- hidden_size + fallback n_layers: read one shard header ---
    hidden_size: int | None = None
    header: dict | None = None

    for shard_path in shard_paths:
        header = _read_safetensors_header(shard_path)
        if header is not None:
            break

    if header is not None:
        # hidden_size from embedding table: shape is [vocab_size, hidden_size]
        embed = header.get("model.embed_tokens.weight")
        if embed and len(embed.get("shape", [])) == 2:
            hidden_size = embed["shape"][1]

        # fallback n_layers from this shard if index.json wasn't usable
        if n_layers is None:
            indices = {int(m.group(1)) for k in header for m in [_LAYER_RE.search(k)] if m}
            if indices:
                n_layers = max(indices) + 1

    # Require at least n_layers — if we can't find it the description isn't useful
    if n_layers is None:
        return None

    return {
        "modality": _infer_modality({}, model_id),
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "kv_heads": None,    # not reliably inferable from tensor shapes alone
        "head_dim": None,    # requires kv_heads + q_proj shape cross-check; omitted
        "max_context": None, # not stored in weight tensors
        "quant": None,       # safetensors dtype (BF16/F32) ≠ post-training quantisation
    }


def describe(model_id: str) -> dict | None:
    """Architecture metadata for *model_id* from its HF config, or None if unavailable.

    Fallback chain:
    1. config.json via HF Hub (Transformers models) — most complete.
    2. Cached .gguf header — GGUF models (GGUF-specific quant, kv_heads, context).
    3. Cached .safetensors header — safetensors models with no config.json cached;
       recovers n_layers + hidden_size; leaves kv_heads/head_dim/max_context/quant None.
    """
    _, separator, selected_name = model_id.partition(":")
    if separator and selected_name.lower().endswith(".gguf"):
        return _describe_gguf(model_id)
    cfg = _read_config(model_id)
    if cfg is None:
        gguf_result = _describe_gguf(model_id)
        if gguf_result is not None:
            return gguf_result
        return _describe_safetensors(model_id)
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


def _cache_size_gb(model_id: str) -> float | None:
    """On-disk size of *model_id* in the local HF cache, in GB (decimal). None if it isn't cached
    or the cache can't be read. No network — a weights-footprint proxy for the analytic estimate."""
    from huggingface_hub import scan_cache_dir

    try:
        repo = next((r for r in scan_cache_dir().repos
                     if r.repo_id == model_id and r.repo_type == "model"), None)
        if repo is None or not repo.size_on_disk:
            return None
        return round(repo.size_on_disk / 1e9, 3)
    except Exception:
        return None


def remember(con, model_id: str) -> dict | None:
    """Describe *model_id* and store it in the catalog; return the stored row (or None).

    Also records the model's on-disk weight (``weights_gb``) from the local cache, so the
    analytic estimate (profile/recommend) has a footprint without a network call."""
    meta = describe(model_id)
    if meta is None:
        return None
    db.upsert_model(con, model_id, weights_gb=_cache_size_gb(model_id), **meta)
    return db.get_model(con, model_id)


def get(con, model_id: str) -> dict | None:
    return db.get_model(con, model_id)


def remember_variant(con, model_id: str, canonical_model_id: str, *,
                     quant: str | None, weights_gb: float | None) -> dict | None:
    """Catalog an exact artifact selector using its repo's architecture metadata."""
    base = db.get_model(con, canonical_model_id) or remember(con, canonical_model_id)
    if base is None:
        return None
    fields = {name: base.get(name) for name in db._MODEL_COLS}
    fields["quant"] = quant
    fields["weights_gb"] = weights_gb
    db.upsert_model(con, model_id, **fields)
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
