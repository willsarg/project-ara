# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Model registry — reads HF-cache config.json and classifies each model's memory behavior.

Key distinction (learned the hard way):
  * Sliding-window models (Gemma, GPT-OSS) get a RotatingKVCache, which MLX CANNOT
    quantize (`to_quantized` raises NotImplementedError). Forcing --kv-bits 4 on them
    crashes any prompt >= quantized_kv_start (5000) tokens. They must run fp16 KV.
    Their sliding window caps most layers, so KV growth is already gentle.
  * Linear+full models (Qwen3.5 9B/27B) get a standard KVCache and quantize fine;
    4-bit meaningfully lowers high-context memory.

Only `full_attention` layers grow unboundedly with context. `sliding_attention` layers
are window-capped and `linear_attention` layers use a fixed recurrent state, so neither
contributes to context-scaling KV growth.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

HUB = os.path.expanduser("~/.cache/huggingface/hub")
UNBOUNDED_CUSTOM_CACHE_TYPES = {"qwen3_5", "qwen3_5_text"}

# Measured: the OS-wired slope is dominated by the prefill transient, ~5x the analytic
# fp16 KV-cache slope (Gemma: analytic 0.0143 vs measured 0.070 GB/1k). For UNCHARACTERIZED
# models we apply this multiplier so estimates stay conservative.
PREFILL_SPIKE_MULT = 5.0

# MLX quantized-KV group size (matches production: kv_group_size=64). Each group also carries
# an fp16 scale + bias, the per-element overhead in kv_bytes_per_token().
KV_GROUP_SIZE = 64
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


@dataclass
class ModelInfo:
    hf_id: str
    weights_gb: float
    n_layers: int
    growing_layers: int          # full_attention layers — the ones whose KV grows with context
    kv_heads: int | None
    head_dim: int | None
    hidden_size: int | None
    max_context: int | None
    cache_type: str              # "RotatingKVCache" | "standard"
    can_quantize_kv: bool        # False => must run fp16 KV (do NOT pass --kv-bits 4)
    layer_types: dict
    max_kv_size_enforced: bool = True
    is_causal: bool = True

    def kv_bytes_per_token(self, kv_bits: int | None = None) -> float:
        """Analytic KV-cache growth per token (K and V), counting only growing layers.

        ``kv_bits=None`` is fp16 (2 bytes/elem). A quantized cache stores ``kv_bits/8`` bytes
        per element plus an fp16 scale + bias per ``KV_GROUP_SIZE`` group, so it grows
        proportionally slower — the a-priori slope must reflect this or opting into quant
        buys no extra context (the same lesson the Vulkan lane learned)."""
        if not (self.kv_heads and self.head_dim):
            return 0.0
        elems = self.growing_layers * self.kv_heads * self.head_dim * 2  # 2 = (K, V)
        if kv_bits is None:
            bytes_per_elem = 2.0  # fp16
        else:
            bytes_per_elem = kv_bits / 8 + 2 * 2 / KV_GROUP_SIZE  # payload + fp16 scale+bias/group
        return elems * bytes_per_elem

    def fp16_kv_bytes_per_token(self) -> float:
        """Back-compat alias: fp16 KV growth per token. Matches measured mlx_true slope
        closely on Gemma/Qwen."""
        return self.kv_bytes_per_token(None)

    def estimated_slope_gb_per_k(self, kv_bits: int | None = None) -> float:
        """Conservative os-wired slope estimate (GB per 1k tokens) for an UNcharacterized
        model: steady-state KV growth (at the given ``kv_bits``) scaled by the prefill-spike
        multiplier. Defaults to fp16."""
        kv_slope = self.kv_bytes_per_token(kv_bits) * 1000 / 1e9
        return kv_slope * PREFILL_SPIKE_MULT

    def as_dict(self) -> dict:
        d = asdict(self)
        d["layer_types"] = json.dumps(self.layer_types)
        return d


def _cache_dir(hf_id: str) -> str:
    return os.path.join(HUB, "models--" + hf_id.replace("/", "--"))


def _local_snapshot(hf_id: str) -> str | None:
    path = os.path.abspath(os.path.expanduser(hf_id))
    return path if os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")) else None


def _snapshot_weight_bytes(snapshot: str) -> int:
    """Weight bytes in a local snapshot, deduplicating resolved HF blob links."""
    indexes = []
    for dirpath, _, filenames in os.walk(snapshot):
        indexes.extend(os.path.join(dirpath, filename) for filename in filenames
                       if (filename.endswith(".safetensors.index.json")
                           or filename.endswith(".bin.index.json")))
    selected: list[str] | None = None
    if indexes:
        if len(indexes) != 1:
            return 0
        try:
            with open(indexes[0], encoding="utf-8") as stream:
                weight_map = json.load(stream).get("weight_map")
            names = set(weight_map.values()) if isinstance(weight_map, dict) else set()
            root = os.path.abspath(snapshot)
            selected = []
            for name in names:
                if (not isinstance(name, str) or not name or "\\" in name
                        or os.path.isabs(name)
                        or any(part in ("", ".", "..") for part in name.split("/"))):
                    return 0
                candidate = os.path.abspath(os.path.join(root, *name.split("/")))
                if os.path.commonpath((root, candidate)) != root or not os.path.isfile(candidate):
                    return 0
                selected.append(candidate)
        except (OSError, UnicodeError, ValueError, AttributeError):
            return 0
    seen: set[str] = set()
    total = 0
    paths = selected
    if paths is None:
        paths = [os.path.join(dirpath, filename)
                 for dirpath, _, filenames in os.walk(snapshot)
                 for filename in filenames if filename.lower().endswith(_WEIGHT_SUFFIXES)]
    for path in paths:
        resolved = os.path.realpath(path)
        if resolved in seen or not os.path.isfile(resolved):
            continue
        seen.add(resolved)
        total += os.path.getsize(resolved)
    return total


def cache_updated_at(hf_id: str) -> float | None:
    """Newest artifact mtime in any locally cached model snapshot."""
    root = _cache_dir(hf_id)
    if not os.path.isdir(root):
        return None
    latest: float | None = None
    for dirpath, _, filenames in os.walk(os.path.join(root, "snapshots")):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            try:
                mtime = max(os.lstat(path).st_mtime, os.path.getmtime(path))
            except OSError:
                continue
            latest = mtime if latest is None else max(latest, mtime)
    return latest


def fit_is_stale(hf_id: str, characterized_at: str | None) -> bool:
    """Whether cache artifacts are newer than a characterization run."""
    if not characterized_at:
        return False
    try:
        characterized = datetime.fromisoformat(characterized_at)
    except ValueError:
        return False
    if characterized.tzinfo is None:
        characterized = characterized.replace(tzinfo=timezone.utc)
    cache_mtime = cache_updated_at(hf_id)
    if cache_mtime is None:
        return False
    # DB timestamps use second precision; avoid false positives within that second.
    return cache_mtime > characterized.timestamp() + 1.0


def weights_gb(hf_id: str) -> float:
    """Real on-disk weight size from the cache `blobs/` dir (not the symlinked snapshot)."""
    local = _local_snapshot(hf_id)
    if local is not None:
        return _snapshot_weight_bytes(local) / 1e9
    blobs = os.path.join(_cache_dir(hf_id), "blobs")
    total = 0
    for f in glob.glob(os.path.join(blobs, "*")):
        if os.path.islink(f) or not os.path.isfile(f):
            continue
        # weights only — skip tokenizer/json blobs by size heuristic is unreliable; sum all
        total += os.path.getsize(f)
    return total / 1e9


def _read_raw_config(hf_id: str) -> dict | None:
    local = _local_snapshot(hf_id)
    cfgs = ([os.path.join(local, "config.json")] if local is not None else
            glob.glob(os.path.join(_cache_dir(hf_id), "snapshots", "*", "config.json")))
    if not cfgs:
        return None
    with open(cfgs[0]) as fh:
        return json.load(fh)


def _read_config(hf_id: str) -> dict | None:
    c = _read_raw_config(hf_id)
    if c is None:
        return None
    # Gemma/VLM nest the text model config
    return c.get("text_config", c) if isinstance(c.get("text_config"), dict) else c


def describe(hf_id: str) -> ModelInfo | None:
    raw = _read_raw_config(hf_id)
    if raw is None:
        return None
    t = raw.get("text_config", raw) if isinstance(raw.get("text_config"), dict) else raw
    layer_types = t.get("layer_types", []) or []
    lt = Counter(layer_types)
    n_layers = t.get("num_hidden_layers", len(layer_types))
    growing = lt.get("full_attention", 0) if layer_types else n_layers
    sliding_enabled = t.get("use_sliding_window", True)
    has_sliding = (
        (sliding_enabled is not False and bool(t.get("sliding_window")))
        or lt.get("sliding_attention", 0) > 0
    )
    # Separate two distinct failure modes:
    #   1. _get_classes failing to import (mlx_lm API drift) → hard error, not is_causal=False.
    #      Importing here (outside the narrow except) means an ImportError propagates up visibly.
    #   2. _get_classes(t) raising ValueError/KeyError for an unsupported model_type → is_causal=False.
    from mlx_lm.utils import _get_classes  # ImportError here is intentional: surfaces API drift loudly
    is_causal = True
    try:
        _get_classes(t)
    except (ValueError, KeyError):
        is_causal = False

    return ModelInfo(
        hf_id=hf_id,
        weights_gb=round(weights_gb(hf_id), 2),
        n_layers=n_layers,
        growing_layers=growing,
        kv_heads=t.get("num_key_value_heads"),
        head_dim=t.get("head_dim") or (t.get("hidden_size") // t.get("num_attention_heads") if t.get("hidden_size") and t.get("num_attention_heads") else None),
        hidden_size=t.get("hidden_size"),
        max_context=t.get("max_position_embeddings"),
        cache_type="RotatingKVCache" if has_sliding else "standard",
        can_quantize_kv=not has_sliding,
        layer_types=dict(lt),
        max_kv_size_enforced=(
            raw.get("model_type") not in UNBOUNDED_CUSTOM_CACHE_TYPES
            and t.get("model_type") not in UNBOUNDED_CUSTOM_CACHE_TYPES
        ),
        is_causal=is_causal,
    )


def scan_cache() -> list[str]:
    """Return hf_ids of all mlx-community models present in the local HF cache."""
    out = []
    for d in glob.glob(os.path.join(HUB, "models--mlx-community--*")):
        name = os.path.basename(d).replace("models--", "").replace("--", "/", 1).replace("--", "-")
        # reconstruct: models--org--rest -> org/rest
        base = os.path.basename(d)[len("models--"):]
        org, _, rest = base.partition("--")
        out.append(f"{org}/{rest}")
    return sorted(set(out))


def resolve_hf_id(name: str) -> str:
    """Resolve a possibly-short model name to a full ``org/name`` HF id.

    A name containing ``/`` is returned unchanged. A bare name (e.g.
    ``gemma-4-e4b-it-4bit``) is matched against the local cache by its final
    path segment; a unique cache hit wins. With no cache match the org defaults
    to ``mlx-community`` (the suite's convention), so a freshly-downloaded
    ``gemma-4-e4b-it-4bit`` still resolves to ``mlx-community/gemma-4-e4b-it-4bit``.
    """
    if "/" in name:
        return name
    matches = [c for c in scan_cache() if c.split("/", 1)[1] == name]
    if len(matches) == 1:
        return matches[0]
    return f"mlx-community/{name}"
