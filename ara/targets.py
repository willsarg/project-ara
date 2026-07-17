# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Canonical runtime × backend identities for durable capability evidence."""
from __future__ import annotations

import json

from ara.engine_identity import canonical_engine


ENGINE_TARGETS = {
    "mlx": ("mlx", "apple"),
    "cuda": ("torch", "cuda"),
    "cpu": ("llamacpp", "cpu"),
    "vulkan": ("llamacpp", "vulkan"),
    "cuda-gguf": ("llamacpp", "cuda"),
    "ollama": ("ollama", "unknown"),
}

_NATIVE_DEFAULTS = {
    ("mlx", "apple"): {"kv_quant": "f16"},
    ("torch", "cuda"): {
        "flash_attn": False,
        "kv_quant": "f16",
        "prefill_chunk": None,
        "weight_quant": "none",
    },
    ("llamacpp", "vulkan"): {"flash_attn": True, "kv_quant": "f16"},
    ("llamacpp", "cuda"): {},
    ("llamacpp", "cpu"): {},
}

_OLLAMA_CONFIG_FIELDS = (
    "configured_flash_attention",
    "configured_inputs",
    "configured_kv_cache_type",
    "configured_num_parallel",
    "configured_num_parallel_authority",
    "configured_scheduler_spread",
    "effective_flash_attention",
    "effective_kv_cache_type",
    "effective_num_parallel",
    "effective_num_parallel_authority",
    "effective_scheduler_spread",
    "endpoint_authority",
    "runtime_version",
    "server_instance_id",
)


def for_engine(engine: str | None) -> tuple[str, str, str | None]:
    """Return ``(runtime, backend, legacy_engine)`` without rejecting unknown identities."""
    canonical = canonical_engine(engine)
    runtime, backend = ENGINE_TARGETS.get(
        canonical, (canonical or "unknown", "unknown"))
    return runtime, backend, canonical


def config_key(runtime: str, backend: str, *, placement_policy: str | None = None,
               options: dict | None = None) -> str:
    """Return the versioned canonical identity for effective target configuration."""
    payload = {
        "runtime": runtime,
        "backend": backend,
        "placement_policy": placement_policy,
        "options": {} if options is None else options,
    }
    return "cfg:v1:" + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def characterization_config_key(engine: str | None, config: dict | None) -> str:
    """Build one target key from effective ceiling-changing settings only."""
    runtime, backend, _legacy_engine = for_engine(engine)
    supplied = config if isinstance(config, dict) else {}
    placement = supplied.get("placement")
    if runtime == "ollama":
        backend = {
            "cpu": "cpu",
            "unified": "apple",
            "accelerator": "cuda",
            "partial_offload": "cuda",
        }.get(placement, backend)
        options = {field: supplied.get(field) for field in _OLLAMA_CONFIG_FIELDS}
    else:
        defaults = _NATIVE_DEFAULTS.get((runtime, backend), {})
        options = {**defaults, **{
            key: value for key, value in supplied.items() if key in defaults}}
    return config_key(
        runtime, backend, placement_policy=placement, options=options)


def calibration_config_key() -> str:
    """The canonical default target-overhead calibration contract."""
    return "cal:v1:default"
