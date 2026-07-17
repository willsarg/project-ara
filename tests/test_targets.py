# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Canonical durable target identities."""

from ara import targets


def test_native_config_keys_expand_effective_defaults():
    implicit = targets.characterization_config_key("cuda", {})
    explicit = targets.characterization_config_key(
        "cuda", {
            "kv_quant": "f16",
            "flash_attn": False,
            "weight_quant": "none",
            "prefill_chunk": None,
        })

    assert implicit == explicit
    assert implicit == (
        'cfg:v1:{"backend":"cuda","options":{"flash_attn":false,'
        '"kv_quant":"f16","prefill_chunk":null,"weight_quant":"none"},'
        '"placement_policy":null,"runtime":"torch"}')
    assert targets.characterization_config_key("vulkan", {}) != implicit


def test_ollama_config_key_uses_authority_not_transient_measurements():
    config = {
        "placement": "unified",
        "runtime_version": "0.30.10",
        "server_instance_id": "42:1:/usr/bin/ollama",
        "endpoint_authority": "http://127.0.0.1:11434",
        "configured_inputs": {"OLLAMA_KV_CACHE_TYPE": "q8_0"},
        "configured_num_parallel": 1,
        "configured_num_parallel_authority": "configured",
        "effective_num_parallel": 1,
        "configured_kv_cache_type": "q8_0",
        "effective_kv_cache_type": "unknown",
        "configured_flash_attention": "unknown",
        "effective_flash_attention": "unknown",
        "configured_scheduler_spread": "unknown",
        "effective_scheduler_spread": "unknown",
        "resident_total_bytes": 100,
        "system_memory_delta_bytes": 50,
        "requested_context": 4096,
    }
    changed_measurement = {
        **config,
        "resident_total_bytes": 999,
        "system_memory_delta_bytes": 888,
        "requested_context": 8192,
    }

    first = targets.characterization_config_key("ollama", config)
    second = targets.characterization_config_key("ollama", changed_measurement)

    assert first == second
    assert '"placement_policy":"unified"' in first
    assert '"runtime_version":"0.30.10"' in first


def test_unknown_engine_target_and_config_remain_non_null():
    assert targets.for_engine("future") == ("future", "unknown", "future")
    assert targets.characterization_config_key("future", {}) == (
        'cfg:v1:{"backend":"unknown","options":{},"placement_policy":null,'
        '"runtime":"future"}')
