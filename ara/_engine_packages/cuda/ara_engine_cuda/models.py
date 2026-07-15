# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Model registry — read the HF-cache config.json and classify each model's VRAM behaviour.

The CUDA counterpart to the MLX engine's model inspector: same job (estimate weights + KV-cache growth from a
model's config, with no torch import and no GPU), against transformers-format checkpoints rather
than the mlx-community 4-bit builds. ARA owns the ramp methodology; this only supplies the facts
ARA's driver fits and solves on top — the context→0 base, the a-priori slope, the window.

Only layers with unbounded (``full_attention``) KV grow with context. ``sliding_attention`` /
``linear_attention`` layers are window-capped or recurrent, so they don't scale, mirroring MLX.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter
from dataclasses import dataclass

HUB = os.path.expanduser("~/.cache/huggingface/hub")

# nvidia-smi `used` (which captures torch's retained reserved pool) tracks the analytic fp16
# KV slope far more closely than Apple's wired-memory sampler did — but the prefill activation
# transient still reserves extra VRAM, so we keep a conservative factor for the a-priori gate.
# The final ceiling comes from the ACTUAL measured points (driver L2), not this estimate; an
# over-estimate only makes the a-priori (L1/L4) gates stop escalating a little sooner — safe.
PREFILL_SPIKE_MULT = 2.0

# VRAM is binary GiB everywhere here (system.py reads the wall/used as bytes ÷ 1024**3). The KV
# slope must use the SAME unit or the a-priori gate compares a GiB budget against a decimal-GB
# slope — a ~7.37% dimensional error (Rule #1/#3). The mirror of project-ara's ramp.py GIB fix.
GIB = 1024 ** 3

# KV-cache quantization (opt-in; fp16 is the default everywhere). transformers quantizes the cache
# in groups of KV_GROUP_SIZE elements, each group carrying an fp16 scale + zero (2 bytes each), so
# the effective bytes/elem is kv_bits/8 + 2·2/group — the same scheme MLX uses. The `hqq` backend
# does both 8- and 4-bit (unlike quanto's 2/4-only KV path), so q8_0 and q4_0 both work.
KV_GROUP_SIZE = 64
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


# Runtime WEIGHT quantization (NVIDIA-native; no GGUF/MLX parallel — they ship pre-quantized files).
# The factor is the fraction of the on-disk fp16 weight size the quantized weights occupy — kept
# slightly conservative (the measured ramp corrects it; an over-estimate only makes the a-priori
# gate stop escalating sooner — safe, Rule #1). int8 = LLM.int8, int4 = NF4 + double-quant, fp8 =
# FineGrainedFP8 (Ada/Hopper only). None/"none" = fp16 weights as shipped.
_WEIGHT_BYTES_FACTOR = {"none": 1.0, "int8": 0.55, "int4": 0.35, "fp8": 0.55}


def weight_bytes_factor(weight_quant: str | None) -> float:
    """Fraction of the on-disk fp16 weight size the weights occupy at *weight_quant* (1.0 = none)."""
    return _WEIGHT_BYTES_FACTOR.get(weight_quant or "none", 1.0)


def is_prequantized(hf_id: str) -> bool:
    """True if the repo ships already-quantized weights (GPTQ/AWQ/etc. — a ``quantization_config``
    in its config). transformers loads those as-is, so we must NOT stack bitsandbytes on top, and
    the on-disk weight size already reflects the quantization (factor 1.0)."""
    raw = _read_config(hf_id) or {}
    t = raw.get("text_config", raw) if isinstance(raw.get("text_config"), dict) else raw
    return bool(raw.get("quantization_config") or t.get("quantization_config"))


def weight_quant_kwargs(weight_quant: str | None, *, prequantized: bool = False) -> dict:
    """``from_pretrained`` kwargs to load weights at *weight_quant*, or ``{}`` for none / a
    pre-quantized repo (transformers handles those from the repo's own config). Lazily imports
    transformers. fp8 is gated on hardware by the caller (see :func:`system.fp8_capable`)."""
    if weight_quant in (None, "none") or prequantized:
        return {}
    import torch
    from transformers import BitsAndBytesConfig
    if weight_quant == "int8":
        return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    if weight_quant == "int4":
        return {"quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)}
    if weight_quant == "fp8":
        from transformers import FineGrainedFP8Config
        return {"quantization_config": FineGrainedFP8Config()}
    return {}


def attn_implementation(prefer_flash: bool) -> str:
    """The transformers ``attn_implementation`` to load with. SDPA (always available, fused) is the
    default; FlashAttention-2 is used ONLY when the user opted in AND the GPU+package support it
    (see :func:`system.flash_attn_capable`). An opt-in on unsupported hardware falls back to SDPA,
    never crashes — the honest, availability-gated behaviour."""
    from . import system
    return "flash_attention_2" if (prefer_flash and system.flash_attn_capable()) else "sdpa"


def kv_cache_kwargs(kv_bits: int | None) -> dict:
    """``model.generate`` kwargs that quantize the KV cache at *kv_bits*, or ``{}`` for fp16.

    Uses transformers' public generate-time cache API (``cache_implementation="quantized"`` + a
    ``cache_config`` dict) rather than constructing a cache class. The backend string is the
    lowercase ``"hqq"`` transformers expects (verified live against transformers 5.x on willw11)."""
    if kv_bits is None:
        return {}
    return {"cache_implementation": "quantized",
            "cache_config": {"backend": "hqq", "nbits": kv_bits, "q_group_size": KV_GROUP_SIZE}}


@dataclass
class ModelInfo:
    hf_id: str
    weights_gb: float
    n_layers: int
    growing_layers: int          # full_attention layers — the only ones whose KV grows with ctx
    kv_heads: int | None
    head_dim: int | None
    hidden_size: int | None
    max_context: int | None
    can_quantize_kv: bool        # False => sliding-window model, keep fp16 KV
    is_causal: bool = True

    def kv_bytes_per_token(self, kv_bits: int | None = None) -> float:
        """Analytic KV-cache growth per token (K and V), counting only growing layers.

        fp16 (``kv_bits=None``) is 2 bytes/elem; quantized is ``kv_bits/8`` plus the group
        scale+zero overhead (see KV_GROUP_SIZE) — so opting into quant lowers the slope, raising
        the gated ceiling (the slope-bug lesson: the a-priori gate must know the cache shrank)."""
        if not (self.kv_heads and self.head_dim):
            return 0.0
        elems = self.growing_layers * self.kv_heads * self.head_dim * 2  # 2 = K and V
        bytes_per_elem = 2.0 if kv_bits is None else kv_bits / 8 + 2 * 2 / KV_GROUP_SIZE
        return elems * bytes_per_elem

    def fp16_kv_bytes_per_token(self) -> float:
        """fp16 KV-cache growth per token — the default precision."""
        return self.kv_bytes_per_token(None)

    def estimated_slope_gb_per_k(self, kv_bits: int | None = None) -> float:
        """Conservative VRAM slope (binary GiB per 1k tokens) for an UNcharacterized model: KV
        growth at *kv_bits* scaled by the prefill-spike factor. GiB to match the wall (GIB note)."""
        return self.kv_bytes_per_token(kv_bits) * 1000 / GIB * PREFILL_SPIKE_MULT


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
        root = os.path.abspath(snapshot)
        if os.path.islink(path):
            blob_dir = os.path.join(os.path.dirname(os.path.dirname(root)), "blobs")
            blob_name = os.path.basename(resolved)
            authorized = (os.path.dirname(resolved) == blob_dir and len(blob_name) in (40, 64)
                          and all(char in "0123456789abcdefABCDEF" for char in blob_name))
        else:
            try:
                authorized = os.path.commonpath((root, resolved)) == root
            except ValueError:
                authorized = False
        if not authorized:
            return 0
        if resolved in seen or not os.path.isfile(resolved):
            continue
        seen.add(resolved)
        total += os.path.getsize(resolved)
    return total


def weights_gb(hf_id: str) -> float:
    """Real on-disk weight size from the cache ``blobs/`` dir (not the symlinked snapshot)."""
    local = _local_snapshot(hf_id)
    if local is not None:
        return _snapshot_weight_bytes(local) / GIB
    total = 0
    for f in glob.glob(os.path.join(_cache_dir(hf_id), "blobs", "*")):
        if os.path.islink(f) or not os.path.isfile(f):
            continue
        total += os.path.getsize(f)
    return total / GIB    # binary GiB — consistent with the VRAM wall/used/budget (all /1024**3);
                          # /1e9 (decimal GB) over-stated the footprint ~7% and over-refused (#105)


def _read_config(hf_id: str) -> dict | None:
    local = _local_snapshot(hf_id)
    cfgs = ([os.path.join(local, "config.json")] if local is not None else
            glob.glob(os.path.join(_cache_dir(hf_id), "snapshots", "*", "config.json")))
    if not cfgs:
        return None
    with open(cfgs[0]) as fh:
        return json.load(fh)


def describe(hf_id: str) -> ModelInfo | None:
    """Classify *hf_id* from its cached config, or None if it isn't in the HF cache."""
    raw = _read_config(hf_id)
    if raw is None:
        return None
    # VLMs nest the language model under text_config; KV growth lives there.
    t = raw.get("text_config", raw) if isinstance(raw.get("text_config"), dict) else raw

    layer_types = t.get("layer_types") or []
    lt = Counter(layer_types)
    n_layers = t.get("num_hidden_layers", len(layer_types))
    growing = lt.get("full_attention", 0) if layer_types else n_layers

    sliding_enabled = t.get("use_sliding_window", True)
    has_sliding = (
        (sliding_enabled is not False and bool(t.get("sliding_window")))
        or lt.get("sliding_attention", 0) > 0
    )

    attn_heads = t.get("num_attention_heads")
    hidden = t.get("hidden_size")
    head_dim = t.get("head_dim") or (hidden // attn_heads if hidden and attn_heads else None)

    # transformers convention: a causal LM declares a *ForCausalLM architecture. Absent the
    # field we assume causal (best effort) rather than refuse a measurable model.
    architectures = raw.get("architectures") or t.get("architectures") or []
    is_causal = (not architectures) or any(a.endswith("ForCausalLM") for a in architectures)

    return ModelInfo(
        hf_id=hf_id,
        weights_gb=round(weights_gb(hf_id), 2),
        n_layers=n_layers,
        growing_layers=growing,
        kv_heads=t.get("num_key_value_heads") or attn_heads,
        head_dim=head_dim,
        hidden_size=hidden,
        max_context=t.get("max_position_embeddings"),
        can_quantize_kv=not has_sliding,
        is_causal=is_causal,
    )
