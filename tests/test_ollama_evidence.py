# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Topology-aware and strictly reusable Ollama characterization evidence."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from ara import db, ollama, ollama_evidence as evidence


GIB = 1024 ** 3


def _snapshot(*, total=16, available=8, kind=None, count=0,
              accelerator_total=None, accelerator_available=None, unified=False):
    return evidence.MemorySnapshot(
        system_total_bytes=total * GIB,
        system_available_bytes=available * GIB,
        accelerator_kind=kind,
        accelerator_count=count,
        accelerator_total_bytes=(accelerator_total * GIB
                                 if accelerator_total is not None else None),
        accelerator_available_bytes=(accelerator_available * GIB
                                     if accelerator_available is not None else None),
        unified=unified,
    )


def _process(*, size=4, accelerator=0, context=4096):
    return ollama.OllamaProcess(
        name="probe:latest",
        size_bytes=size * GIB,
        size_vram_bytes=accelerator * GIB,
        effective_context_per_request=context,
    )


def _model_info(*, layers=28, kv_heads=8, head_dim=128):
    return {
        "general.architecture": "qwen3",
        "qwen3.block_count": layers,
        "qwen3.attention.head_count_kv": kv_heads,
        "qwen3.attention.key_length": head_dim,
    }


def _preflight(snapshot, model_size, *, context=2048, model_info=None):
    return evidence.preflight_refusal_reason(
        snapshot,
        model_size,
        requested_context=context,
        model_info=_model_info() if model_info is None else model_info,
    )


def _authority(**changes):
    authority = ollama.OllamaRuntimeAuthority(
        endpoint=ollama.OllamaEndpoint("http://127.0.0.1:11434", "loopback"),
        server_version="0.30.10",
        server_instance_id="42:1234.500000:/usr/bin/ollama",
        listener_pid=42,
        listener_bind_host="127.0.0.1",
        configured_inputs=(("OLLAMA_KEEP_ALIVE", "2m"),),
        configured_num_parallel=1,
        configured_num_parallel_authority="exact_version_default",
    )
    return replace(authority, **changes)


def test_methodology_and_runtime_fingerprint_are_versioned_stable_and_scoped():
    authority = _authority(configured_inputs=(
        ("OLLAMA_KEEP_ALIVE", "2m"),
        ("OLLAMA_KV_CACHE_TYPE", "q8_0"),
    ))

    assert evidence.CHARACTERIZATION_METHODOLOGY_KEY.startswith(
        "methodology:v1:sha256:")
    fingerprint = evidence.runtime_fingerprint(authority)
    assert fingerprint.startswith("engine:v1:sha256:")
    assert evidence.runtime_fingerprint(authority) == fingerprint
    assert evidence.runtime_fingerprint(replace(
        authority,
        configured_inputs=tuple(reversed(authority.configured_inputs)),
    )) == fingerprint
    assert evidence.runtime_fingerprint(replace(
        authority, server_version="0.30.11")) != fingerprint
    assert evidence.runtime_fingerprint(replace(
        authority, server_instance_id="99:999.0:/usr/bin/ollama")) != fingerprint
    assert evidence.runtime_fingerprint(replace(
        authority, issue="listener_unattributed")) is None


def _model(**changes):
    model = ollama.OllamaModel(
        name="qwen3:0.6b",
        digest="a" * 64,
        size_bytes=522_000_000,
        format="gguf",
        capabilities=("completion",),
        scope="local",
    )
    return replace(model, **changes)


def _complete_characterization(store, *, authority=None, model=None):
    authority = authority or _authority()
    model = model or _model()
    snapshot = _snapshot(total=24, available=8, kind="apple", count=1, unified=True)
    point = evidence.characterization_point(
        snapshot,
        snapshot,
        ollama.OllamaProcess(
            name="probe", size_bytes=785_000_000, size_vram_bytes=785_000_000,
            effective_context_per_request=4096),
        4096,
    )
    admission = evidence.preflight_admission(
        snapshot,
        model.size_bytes,
        requested_context=4096,
        model_info=_model_info(),
    )
    assert admission.reason is None
    point["preload_admission"] = admission.as_dict()
    config = {
        "methodology": "ollama-physical-walls-v1",
        "runtime": "ollama",
        "runtime_version": authority.server_version,
        "endpoint_authority": authority.endpoint.url,
        "server_instance_id": authority.server_instance_id,
        "format": "gguf",
        "capability": "completion",
        "configured_inputs": dict(authority.configured_inputs),
        "configured_num_parallel": 1,
        "configured_num_parallel_authority": "exact_version_default",
        "effective_num_parallel": 1,
        "effective_num_parallel_authority": "configured_maximum_is_one",
        "requested_context": 4096,
        "effective_per_request_context": 4096,
        "placement": "unified",
        "resident_total_bytes": point["resident_total_bytes"],
        "resident_accelerator_bytes": point["resident_accelerator_bytes"],
        "applicable_walls": point["applicable_walls"],
        "system_memory_delta_bytes": point["system_memory_delta_bytes"],
        "accelerator_memory_delta_bytes": point["accelerator_memory_delta_bytes"],
        "system_margin_bytes": point["system_margin_bytes"],
        "accelerator_margin_bytes": point["accelerator_margin_bytes"],
        "configured_kv_cache_type": "unknown",
        "effective_kv_cache_type": "unknown",
        "configured_flash_attention": "unknown",
        "effective_flash_attention": "unknown",
        "configured_scheduler_spread": "unknown",
        "effective_scheduler_spread": "unknown",
        "preload_admission": admission.as_dict(),
        "watchdog": evidence.WATCHDOG_STATUS,
    }
    db.save_characterization(
        store,
        "machine",
        "ollama",
        model.name,
        safe_context=4096,
        points=[point],
        artifact_id="ollama-manifest-sha256:" + model.digest,
        config=config,
        methodology_key=evidence.CHARACTERIZATION_METHODOLOGY_KEY,
        engine_fingerprint=evidence.runtime_fingerprint(authority),
    )
    return model, authority


def _rewrite_characterization_config(store, **changes):
    row = db.get_characterization(store, "machine", "ollama", "qwen3:0.6b")
    config = {**row["config"], **changes}
    store.execute(
        "UPDATE characterizations SET config_json=? WHERE machine_key='machine'",
        (json.dumps(config, sort_keys=True),),
    )
    store.commit()


def _rewrite_characterization_points(store, points):
    store.execute(
        "UPDATE characterizations SET points_json=? WHERE machine_key='machine'",
        (json.dumps(points),),
    )
    store.commit()


def _rewrite_matching_wall_evidence(store, **changes):
    row = db.get_characterization(store, "machine", "ollama", "qwen3:0.6b")
    _rewrite_characterization_config(store, **changes)
    _rewrite_characterization_points(store, [{**row["points"][0], **changes}])


def _replace_characterization_evidence(store, before, after, process):
    point = evidence.characterization_point(before, after, process, 4096)
    admission = evidence.preflight_admission(
        before,
        _model().size_bytes,
        requested_context=4096,
        model_info=_model_info(),
    )
    assert admission.reason is None
    point["preload_admission"] = admission.as_dict()
    _rewrite_characterization_config(
        store,
        preload_admission=admission.as_dict(),
        watchdog=evidence.WATCHDOG_STATUS,
        **{key: point[key] for key in (
            "placement",
            "resident_total_bytes",
            "resident_accelerator_bytes",
            "applicable_walls",
            "system_memory_delta_bytes",
            "accelerator_memory_delta_bytes",
            "system_margin_bytes",
            "accelerator_margin_bytes",
        )},
    )
    _rewrite_characterization_points(store, [point])


def test_assessment_separates_display_row_from_strict_reusable_row(store):
    model, authority = _complete_characterization(store)

    assessment = evidence.assess_characterization(
        store, "machine", model, authority)

    assert assessment.display["safe_context"] == 4096
    assert assessment.reusable is assessment.display
    assert assessment.reason is None


@pytest.mark.parametrize(
    ("column", "value", "reason"),
    [
        ("methodology_key", "methodology:v1:sha256:old", "methodology_mismatch"),
        ("engine_fingerprint", "engine:v1:sha256:old", "runtime_fingerprint_mismatch"),
    ],
)
def test_assessment_rejects_methodology_or_runtime_fingerprint_drift(
        store, column, value, reason):
    model, authority = _complete_characterization(store)
    store.execute(
        f"UPDATE characterizations SET {column}=? WHERE machine_key='machine'",
        (value,),
    )
    store.commit()

    assessment = evidence.assess_characterization(
        store, "machine", model, authority)

    assert assessment.display is not None
    assert assessment.reusable is None
    assert assessment.reason == reason


def test_assessment_keeps_size_only_preflight_history_display_only(store):
    model, authority = _complete_characterization(store)
    _rewrite_characterization_config(store, preload_admission=None)

    assessment = evidence.assess_characterization(
        store, "machine", model, authority)

    assert assessment.display is not None
    assert assessment.reusable is None
    assert assessment.reason == "preload_admission_evidence_incomplete"


@pytest.mark.parametrize(
    "change",
    [
        {"watchdog": "claimed-active"},
        {"model_residency_bound_bytes": 0},
        {"applicable_walls": ["unknown"]},
        {"system_margin_bytes": 0},
    ],
)
def test_preload_admission_evidence_rejects_invalid_proof_terms(change):
    admission = evidence.preflight_admission(
        _snapshot(available=8),
        1 * GIB,
        requested_context=2048,
        model_info=_model_info(),
    ).as_dict()
    changed = {**admission, **change}
    config = {
        "requested_context": 2048,
        "preload_admission": changed,
        "watchdog": evidence.WATCHDOG_STATUS,
    }

    assert evidence._admission_evidence_complete(
        config, {"preload_admission": changed}) is False


def test_assessment_rejects_storage_row_marked_nonreusable(store):
    model, authority = _complete_characterization(store)
    store.execute(
        "UPDATE characterizations SET reusable=0 WHERE machine_key='machine'")
    store.commit()

    assessment = evidence.assess_characterization(
        store, "machine", model, authority)

    assert assessment.display is not None
    assert assessment.reusable is None
    assert assessment.reason == "storage_evidence_not_reusable"


def test_assessment_selects_exact_artifact_when_newer_history_coexists(store):
    model, authority = _complete_characterization(store)
    current = db.get_characterization(store, "machine", "ollama", model.name)
    db.save_characterization(
        store, "machine", "ollama", model.name, safe_context=2048,
        points=current["points"],
        artifact_id="ollama-manifest-sha256:" + "b" * 64,
        config=current["config"], measured_at="9999-01-01T00:00:00+00:00",
    )

    assessment = evidence.assess_characterization(
        store, "machine", model, authority)

    assert assessment.reusable is not None
    assert assessment.reusable["artifact_id"] == "ollama-manifest-sha256:" + "a" * 64
    assert assessment.reusable["safe_context"] == 4096


def test_assessment_prefers_current_identity_over_newer_mismatched_history(store):
    model, authority = _complete_characterization(store)
    current = db.get_characterization(store, "machine", "ollama", model.name)
    db.save_characterization(
        store,
        "machine",
        "ollama",
        model.name,
        safe_context=2048,
        points=current["points"],
        artifact_id=current["artifact_id"],
        config=current["config"],
        measured_at="9999-01-01T00:00:00+00:00",
        methodology_key="methodology:v1:sha256:old",
        engine_fingerprint=current["engine_fingerprint"],
    )

    assessment = evidence.assess_characterization(
        store, "machine", model, authority)

    assert assessment.reason is None
    assert assessment.reusable["safe_context"] == 4096
    assert assessment.reusable["methodology_key"] == (
        evidence.CHARACTERIZATION_METHODOLOGY_KEY)


def test_legacy_characterization_stays_displayable_but_is_not_reusable(store):
    model = _model()
    db.save_characterization(
        store,
        "machine",
        "ollama",
        model.name,
        safe_context=4096,
        points=[],
        artifact_id="ollama-manifest-sha256:" + model.digest,
        config={},
    )

    assessment = evidence.assess_characterization(
        store, "machine", model, _authority())

    assert assessment.display["safe_context"] == 4096
    assert assessment.reusable is None
    assert assessment.reason == "methodology_missing_or_unsupported"


@pytest.mark.parametrize(("model_changes", "reason"), [
    ({"digest": "b" * 64}, "artifact_mismatch"),
    ({"scope": "cloud", "remote_model": "qwen3"}, "unsupported_model_cell"),
    ({"format": "safetensors"}, "unsupported_model_cell"),
    ({"capabilities": ("embedding",)}, "unsupported_model_cell"),
])
def test_reuse_requires_the_exact_supported_model_cell(store, model_changes, reason):
    model, authority = _complete_characterization(store)

    assessment = evidence.assess_characterization(
        store, "machine", replace(model, **model_changes), authority)

    assert assessment.display is not None
    assert assessment.reusable is None
    assert assessment.reason == reason


def test_reuse_requires_a_positive_measured_ceiling(store):
    model, authority = _complete_characterization(store)
    store.execute(
        "UPDATE characterizations SET safe_context=NULL WHERE machine_key='machine'")
    store.commit()

    assessment = evidence.assess_characterization(
        store, "machine", model, authority)

    assert assessment.display is not None
    assert assessment.reusable is None
    assert assessment.reason == "safe_context_missing"


def test_reuse_requires_complete_current_runtime_authority(store):
    model, _authority_at_measurement = _complete_characterization(store)

    assessment = evidence.assess_characterization(
        store, "machine", model, _authority(issue="listener_unattributed"))

    assert assessment.display is not None
    assert assessment.reusable is None
    assert assessment.reason == "runtime_authority_incomplete"


def test_reuse_refuses_runtime_without_fingerprintable_endpoint(store):
    model, authority = _complete_characterization(store)
    endpoint = ollama.OllamaEndpoint(None, "loopback")
    changed = replace(authority, endpoint=endpoint)
    _rewrite_characterization_config(store, endpoint_authority=None)

    assessment = evidence.assess_characterization(
        store, "machine", model, changed)

    assert assessment.display is not None
    assert assessment.reusable is None
    assert assessment.reason == "runtime_authority_incomplete"


@pytest.mark.parametrize(("authority_changes", "reason"), [
    ({"endpoint": ollama.OllamaEndpoint("http://127.0.0.1:22434", "loopback")},
     "endpoint_mismatch"),
    ({"server_version": "0.30.11"}, "runtime_version_mismatch"),
    ({"server_instance_id": "99:999.000000:/usr/bin/ollama"},
     "server_instance_mismatch"),
    ({"configured_inputs": (("OLLAMA_KEEP_ALIVE", "5m"),)},
     "configured_inputs_mismatch"),
    ({"configured_num_parallel": 2}, "parallelism_mismatch"),
    ({"configured_num_parallel_authority": "explicit_process_environment"},
     "parallelism_mismatch"),
])
def test_reuse_invalidates_on_exact_runtime_or_config_drift(
        store, authority_changes, reason):
    model, authority = _complete_characterization(store)

    assessment = evidence.assess_characterization(
        store, "machine", model, replace(authority, **authority_changes))

    assert assessment.display is not None
    assert assessment.reusable is None
    assert assessment.reason == reason
    assert db.get_characterization(
        store, "machine", "ollama", model.name)["safe_context"] == 4096


@pytest.mark.parametrize("changes", [
    {"configured_kv_cache_type": "f16"},
    {"effective_kv_cache_type": "f16"},
    {"configured_flash_attention": "true"},
    {"effective_flash_attention": True},
    {"configured_scheduler_spread": "true"},
    {"effective_scheduler_spread": False},
])
def test_reuse_requires_complete_runtime_config_evidence(store, changes):
    model, authority = _complete_characterization(store)
    _rewrite_characterization_config(store, **changes)

    assessment = evidence.assess_characterization(
        store, "machine", model, authority)

    assert assessment.reusable is None
    assert assessment.reason == "runtime_config_evidence_incomplete"


@pytest.mark.parametrize("changes", [
    {"format": "safetensors"},
    {"capability": "embedding"},
])
def test_reuse_requires_stored_supported_cell_classification(store, changes):
    model, authority = _complete_characterization(store)
    _rewrite_characterization_config(store, **changes)

    assessment = evidence.assess_characterization(
        store, "machine", model, authority)

    assert assessment.reusable is None
    assert assessment.reason == "model_cell_mismatch"


@pytest.mark.parametrize("changes", [
    {"requested_context": 8192},
    {"effective_per_request_context": 2048},
])
def test_reuse_requires_exact_requested_and_effective_context(store, changes):
    model, authority = _complete_characterization(store)
    _rewrite_characterization_config(store, **changes)

    assessment = evidence.assess_characterization(
        store, "machine", model, authority)

    assert assessment.reusable is None
    assert assessment.reason == "context_evidence_incomplete"


def test_reuse_requires_a_matching_successful_characterization_point(store):
    model, authority = _complete_characterization(store)
    _rewrite_characterization_points(store, [])

    assessment = evidence.assess_characterization(
        store, "machine", model, authority)

    assert assessment.reusable is None
    assert assessment.reason == "context_evidence_incomplete"


@pytest.mark.parametrize("changes", [
    {"context": 8192},
    {"refusal_reasons": ["system_margin_breached"]},
])
def test_reuse_rejects_a_self_contradicting_success_point(store, changes):
    model, authority = _complete_characterization(store)
    row = db.get_characterization(store, "machine", "ollama", model.name)
    _rewrite_characterization_points(store, [{**row["points"][0], **changes}])

    assessment = evidence.assess_characterization(
        store, "machine", model, authority)

    assert assessment.reusable is None
    assert assessment.reason == "context_evidence_incomplete"


def test_reuse_requires_a_supported_observed_placement(store):
    model, authority = _complete_characterization(store)
    _rewrite_characterization_config(store, placement="unknown", applicable_walls=[])

    assessment = evidence.assess_characterization(
        store, "machine", model, authority)

    assert assessment.reusable is None
    assert assessment.reason == "placement_unsupported"


@pytest.mark.parametrize("changes", [
    {"resident_total_bytes": None},
    {"resident_accelerator_bytes": 900_000_000},
    {"applicable_walls": ["system", "accelerator"]},
    {"system_memory_delta_bytes": None},
    {"system_memory_delta_bytes": -1},
    {"system_margin_bytes": 0},
    {"accelerator_memory_delta_bytes": 1},
    {"accelerator_margin_bytes": 1},
])
def test_reuse_requires_complete_consistent_wall_evidence(store, changes):
    model, authority = _complete_characterization(store)
    _rewrite_characterization_config(store, **changes)

    assessment = evidence.assess_characterization(
        store, "machine", model, authority)

    assert assessment.reusable is None
    assert assessment.reason == "wall_evidence_incomplete"


@pytest.mark.parametrize("changes", [
    {"resident_total_bytes": -1},
    {"resident_total_bytes": 0},
    {"resident_accelerator_bytes": -1},
    {"resident_accelerator_bytes": 900_000_000},
    {"applicable_walls": ["system"]},
    {"system_memory_delta_bytes": -1},
    {"system_margin_bytes": 0},
    {"accelerator_memory_delta_bytes": 1},
    {"accelerator_margin_bytes": 1},
])
def test_reuse_rejects_matching_but_invalid_wall_claims(store, changes):
    model, authority = _complete_characterization(store)
    _rewrite_matching_wall_evidence(store, **changes)

    assessment = evidence.assess_characterization(
        store, "machine", model, authority)

    assert assessment.reusable is None
    assert assessment.reason == "wall_evidence_incomplete"


@pytest.mark.parametrize(("before", "after", "process", "placement"), [
    (_snapshot(available=10), _snapshot(available=6), _process(), "cpu"),
    (
        _snapshot(
            total=32, available=24, kind="nvidia", count=1,
            accelerator_total=8, accelerator_available=7),
        _snapshot(
            total=32, available=20, kind="nvidia", count=1,
            accelerator_total=8, accelerator_available=2),
        _process(size=6, accelerator=6),
        "accelerator",
    ),
    (
        _snapshot(
            total=32, available=24, kind="nvidia", count=1,
            accelerator_total=8, accelerator_available=7),
        _snapshot(
            total=32, available=20, kind="nvidia", count=1,
            accelerator_total=8, accelerator_available=2),
        _process(size=10, accelerator=6),
        "partial_offload",
    ),
])
def test_reuse_accepts_each_certifiable_placement(
        store, before, after, process, placement):
    model, authority = _complete_characterization(store)
    _replace_characterization_evidence(store, before, after, process)

    assessment = evidence.assess_characterization(
        store, "machine", model, authority)

    assert assessment.reason is None
    assert assessment.reusable["config"]["placement"] == placement


def test_cpu_point_uses_system_wall_and_records_observed_delta():
    point = evidence.characterization_point(
        _snapshot(available=10), _snapshot(available=6), _process(), 4096)

    assert point["fit"] is True
    assert point["placement"] == "cpu"
    assert point["requested_context"] == 4096
    assert point["effective_per_request_context"] == 4096
    assert point["resident_total_bytes"] == 4 * GIB
    assert point["resident_accelerator_bytes"] == 0
    assert point["system_memory_delta_bytes"] == 4 * GIB
    assert point["accelerator_memory_delta_bytes"] is None
    assert point["applicable_walls"] == ["system"]
    assert point["refusal_reasons"] == []


def test_apple_unified_point_uses_one_physical_wall_without_double_counting():
    before = _snapshot(total=24, available=15, kind="apple", count=1, unified=True)
    after = _snapshot(total=24, available=8, kind="apple", count=1, unified=True)

    point = evidence.characterization_point(
        before, after, _process(size=10, accelerator=10, context=8192), 8192)

    assert point["fit"] is True
    assert point["placement"] == "unified"
    assert point["applicable_walls"] == ["system_unified"]
    assert point["accelerator_memory_delta_bytes"] is None
    assert point["accelerator_margin_bytes"] is None


def test_single_discrete_accelerator_and_partial_offload_check_both_walls():
    before = _snapshot(
        total=32, available=24, kind="nvidia", count=1,
        accelerator_total=8, accelerator_available=7)
    after = _snapshot(
        total=32, available=20, kind="nvidia", count=1,
        accelerator_total=8, accelerator_available=2)

    full = evidence.characterization_point(
        before, after, _process(size=6, accelerator=6), 4096)
    partial = evidence.characterization_point(
        before, after, _process(size=10, accelerator=6), 4096)

    assert full["fit"] is True and full["placement"] == "accelerator"
    assert partial["fit"] is True and partial["placement"] == "partial_offload"
    assert partial["applicable_walls"] == ["system", "accelerator"]
    assert partial["accelerator_memory_delta_bytes"] == 5 * GIB


def test_unknown_or_multi_accelerator_placement_is_display_only():
    before = _snapshot(kind="nvidia", count=2)
    point = evidence.characterization_point(
        before, before, _process(size=6, accelerator=6), 4096)

    assert point["fit"] is False
    assert point["placement"] == "unknown"
    assert point["refusal_reasons"] == ["placement_unknown"]


def test_wall_margin_failures_are_explicit_and_never_fit():
    before = _snapshot(
        total=8, available=4, kind="nvidia", count=1,
        accelerator_total=8, accelerator_available=4)
    after = _snapshot(
        total=8, available=1, kind="nvidia", count=1,
        accelerator_total=8, accelerator_available=0.5)

    point = evidence.characterization_point(
        before, after, _process(size=8, accelerator=7), 4096)

    assert point["fit"] is False
    assert point["refusal_reasons"] == [
        "system_margin_breached", "accelerator_margin_breached"]


def test_missing_wall_or_invalid_residency_evidence_fails_closed():
    missing = evidence.MemorySnapshot(None, None, None, None, None, None, False)
    point = evidence.characterization_point(missing, missing, _process(), 4096)
    assert point["fit"] is False
    assert point["refusal_reasons"] == ["system_wall_unknown"]

    invalid = evidence.characterization_point(
        _snapshot(), _snapshot(), _process(size=4, accelerator=5), 4096)
    assert invalid["fit"] is False
    assert invalid["placement"] == "unknown"
    assert invalid["refusal_reasons"] == ["placement_unknown"]

    missing_total = evidence.characterization_point(
        evidence.MemorySnapshot(None, 8 * GIB, None, 0, None, None, False),
        evidence.MemorySnapshot(None, 8 * GIB, None, 0, None, None, False),
        _process(),
        4096,
    )
    assert missing_total["refusal_reasons"] == ["system_wall_unknown"]

    malformed = evidence.characterization_point(
        _snapshot(),
        _snapshot(),
        ollama.OllamaProcess(
            name="probe", size_bytes=None, size_vram_bytes=None,
            effective_context_per_request=4096),
        4096,
    )
    assert malformed["placement"] == "unknown"
    assert malformed["refusal_reasons"] == ["placement_unknown"]


@pytest.mark.parametrize(("size", "accelerator"), [
    (True, 0), (0, 0), (4, None), (4, True), (4, -1),
])
def test_malformed_resident_sizes_are_unknown(size, accelerator):
    point = evidence.characterization_point(
        _snapshot(),
        _snapshot(),
        ollama.OllamaProcess(
            name="probe", size_bytes=size, size_vram_bytes=accelerator,
            effective_context_per_request=4096),
        4096,
    )
    assert point["refusal_reasons"] == ["placement_unknown"]


def test_discrete_unknown_wall_and_effective_context_mismatch_fail_closed():
    before = _snapshot(
        total=32, available=20, kind="nvidia", count=1,
        accelerator_total=8, accelerator_available=4)
    missing_accelerator_total = evidence.MemorySnapshot(
        32 * GIB, 20 * GIB, "nvidia", 1, None, 4 * GIB, False)
    wall = evidence.characterization_point(
        before, missing_accelerator_total, _process(size=4, accelerator=4), 4096)
    assert wall["refusal_reasons"] == ["accelerator_wall_unknown"]

    mismatch = evidence.characterization_point(
        _snapshot(), _snapshot(), _process(context=2048), 4096)
    assert mismatch["refusal_reasons"] == ["effective_context_mismatch"]

    after_unknown = evidence.MemorySnapshot(16 * GIB, None, None, 0, None, None, False)
    assert evidence.characterization_point(
        _snapshot(), after_unknown, _process(), 4096)["system_memory_delta_bytes"] is None


def test_failed_point_keeps_the_requested_context_and_explicit_reason():
    point = evidence.failed_characterization_point(8192, "generation_failed")

    assert point["context"] == 8192
    assert point["requested_context"] == 8192
    assert point["effective_per_request_context"] is None
    assert point["fit"] is False
    assert point["placement"] == "unknown"
    assert point["refusal_reasons"] == ["generation_failed"]


def test_preflight_bounds_residency_kv_and_runtime_before_model_load():
    assert _preflight(_snapshot(available=8), 1 * GIB) is None
    assert _preflight(
        _snapshot(available=4), 1 * GIB) == "allocation_exceeds_system_wall"

    discrete = _snapshot(
        total=32, available=8, kind="nvidia", count=1,
        accelerator_total=8, accelerator_available=7)
    assert _preflight(discrete, 1 * GIB) is None
    assert _preflight(discrete, 4 * GIB) == "allocation_exceeds_system_wall"
    accelerator_tight = _snapshot(
        total=32, available=16, kind="nvidia", count=1,
        accelerator_total=8, accelerator_available=3)
    assert _preflight(
        accelerator_tight, 1 * GIB) == "allocation_exceeds_accelerator_wall"


def test_preflight_refuses_small_model_when_requested_context_exceeds_wall():
    small_model = 500 * 1024 ** 2

    assert _preflight(
        _snapshot(available=8),
        small_model,
        context=131_072,
        model_info=_model_info(layers=32, kv_heads=32, head_dim=128),
    ) == "allocation_exceeds_system_wall"


def test_preflight_assessment_records_every_conservative_allocation_term():
    admission = evidence.preflight_admission(
        _snapshot(available=8),
        1 * GIB,
        requested_context=2048,
        model_info=_model_info(),
    )

    assert admission.reason is None
    assert admission.model_residency_bound_bytes == 5 * GIB // 4
    assert admission.kv_cache_bound_bytes == 448 * 1024 ** 2
    assert admission.runtime_overhead_bound_bytes == evidence.RUNTIME_OVERHEAD_BYTES
    assert admission.total_allocation_bound_bytes == (
        admission.model_residency_bound_bytes
        + admission.kv_cache_bound_bytes
        + admission.runtime_overhead_bound_bytes
    )
    assert admission.applicable_walls == ("system",)


def test_preflight_fails_closed_for_unknown_size_wall_or_multi_accelerator():
    missing = evidence.MemorySnapshot(None, None, None, None, None, None, False)
    assert _preflight(missing, 1) == "system_wall_unknown"
    assert _preflight(_snapshot(), None) == "model_size_unknown"
    assert _preflight(
        _snapshot(kind="nvidia", count=2), 1) == "placement_unknown"
    assert _preflight(
        evidence.MemorySnapshot(16 * GIB, None, None, 0, None, None, False), 1,
    ) == "system_wall_unknown"
    assert _preflight(
        evidence.MemorySnapshot(16 * GIB, 8 * GIB, "nvidia", 1, 8 * GIB, None, False), 1,
    ) == "accelerator_wall_unknown"
    assert _preflight(_snapshot(), 1, context=0) == "requested_context_unknown"
    assert _preflight(
        _snapshot(), 1, model_info={}) == "context_allocation_unknown"
    assert _preflight(
        _snapshot(),
        1,
        model_info={"general.architecture": "qwen3"},
    ) == "context_allocation_unknown"
    assert _preflight(
        _snapshot(kind="amd", count=1),
        1,
    ) == "placement_unknown"


@pytest.mark.parametrize("model_size", [True, 0, "large"])
def test_preflight_rejects_invalid_model_sizes(model_size):
    assert _preflight(_snapshot(), model_size) == "model_size_unknown"


def test_preflight_capacity_never_treats_reserved_margin_as_available():
    assert _preflight(
        _snapshot(available=1), 1) == "allocation_exceeds_system_wall"
    assert _preflight(
        _snapshot(
            total=16, available=1, kind="nvidia", count=1,
            accelerator_total=1, accelerator_available=0.5),
        1,
    ) == "allocation_exceeds_system_wall"


@pytest.mark.parametrize(("resident", "available"), [
    (False, 4),
    (True, 1),
])
def test_live_headroom_reserves_the_recorded_peak_and_margin(resident, available):
    config = {
        "placement": "unified",
        "resident_total_bytes": 3 * GIB,
        "resident_accelerator_bytes": 3 * GIB,
    }

    assert evidence.live_headroom_refusal_reason(
        _snapshot(total=24, available=available, kind="apple", count=1, unified=True),
        config,
        resident=resident,
    ) == "system_headroom_insufficient"


def test_live_headroom_accepts_cold_and_resident_targets_with_capacity():
    config = {
        "placement": "unified",
        "resident_total_bytes": 3 * GIB,
        "resident_accelerator_bytes": 3 * GIB,
    }

    assert evidence.live_headroom_refusal_reason(
        _snapshot(total=24, available=6, kind="apple", count=1, unified=True),
        config,
        resident=False,
    ) is None
    assert evidence.live_headroom_refusal_reason(
        _snapshot(total=24, available=3, kind="apple", count=1, unified=True),
        config,
        resident=True,
    ) is None


@pytest.mark.parametrize(("snapshot", "config", "reason"), [
    (
        evidence.MemorySnapshot(None, None, None, None, None, None, False),
        {"placement": "cpu", "resident_total_bytes": 1,
         "resident_accelerator_bytes": 0},
        "system_wall_unknown",
    ),
    (
        _snapshot(kind="nvidia", count=1),
        {"placement": "unified", "resident_total_bytes": 1,
         "resident_accelerator_bytes": 1},
        "topology_drift",
    ),
    (
        _snapshot(kind="nvidia", count=2),
        {"placement": "accelerator", "resident_total_bytes": 1,
         "resident_accelerator_bytes": 1},
        "topology_drift",
    ),
    (
        _snapshot(),
        {"placement": "cpu", "resident_total_bytes": None,
         "resident_accelerator_bytes": 0},
        "wall_evidence_incomplete",
    ),
    (
        evidence.MemorySnapshot(32 * GIB, 8 * GIB, "nvidia", 1, None, None, False),
        {"placement": "accelerator", "resident_total_bytes": 2 * GIB,
         "resident_accelerator_bytes": 2 * GIB},
        "accelerator_wall_unknown",
    ),
    (
        _snapshot(total=32, available=8, kind="nvidia", count=1,
                  accelerator_total=8, accelerator_available=7),
        {"placement": "accelerator", "resident_total_bytes": 2 * GIB,
         "resident_accelerator_bytes": None},
        "wall_evidence_incomplete",
    ),
    (
        _snapshot(total=32, available=8, kind="nvidia", count=1,
                  accelerator_total=8, accelerator_available=1),
        {"placement": "accelerator", "resident_total_bytes": 2 * GIB,
         "resident_accelerator_bytes": 2 * GIB},
        "accelerator_headroom_insufficient",
    ),
])
def test_live_headroom_fails_closed_for_unknown_or_drifted_walls(
        snapshot, config, reason):
    assert evidence.live_headroom_refusal_reason(
        snapshot, config, resident=False) == reason


def test_live_headroom_accepts_one_discrete_accelerator_with_both_walls():
    snapshot = _snapshot(
        total=32, available=8, kind="nvidia", count=1,
        accelerator_total=8, accelerator_available=7)
    config = {
        "placement": "partial_offload",
        "resident_total_bytes": 4 * GIB,
        "resident_accelerator_bytes": 2 * GIB,
    }

    assert evidence.live_headroom_refusal_reason(
        snapshot, config, resident=False) is None


def test_live_headroom_rejects_impossible_discrete_residency_split():
    snapshot = _snapshot(
        total=32, available=8, kind="nvidia", count=1,
        accelerator_total=8, accelerator_available=7)
    config = {
        "placement": "accelerator",
        "resident_total_bytes": 1 * GIB,
        "resident_accelerator_bytes": 2 * GIB,
    }

    assert evidence.live_headroom_refusal_reason(
        snapshot, config, resident=False) == "wall_evidence_incomplete"


def test_live_residency_must_match_characterized_context_placement_and_walls():
    config = {
        "placement": "unified",
        "resident_total_bytes": 4 * GIB,
        "resident_accelerator_bytes": 4 * GIB,
        "applicable_walls": ["system_unified"],
    }
    snapshot = _snapshot(total=24, available=8, kind="apple", count=1, unified=True)

    assert evidence.live_residency_refusal_reason(
        snapshot, _process(size=4, accelerator=4, context=4096), config, 4096) is None
    assert evidence.live_residency_refusal_reason(
        snapshot, _process(size=4, accelerator=0, context=4096), config, 4096,
    ) == "placement_or_allocation_drift"
    assert evidence.live_residency_refusal_reason(
        _snapshot(total=24, available=1, kind="apple", count=1, unified=True),
        _process(size=4, accelerator=4, context=4096), config, 4096,
    ) == "system_margin_breached"


def test_capture_memory_snapshot_parses_one_nvidia_device(monkeypatch):
    monkeypatch.setattr(
        evidence.psutil, "virtual_memory",
        lambda: SimpleNamespace(total=32 * GIB, available=20 * GIB))
    monkeypatch.setattr(evidence.platform, "system", lambda: "Linux")
    monkeypatch.setattr(evidence.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(evidence.hardware, "clamp_ram_to_cgroup", lambda total: total)
    monkeypatch.setattr(evidence.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(evidence, "_run", lambda command: "8192, 2048\n")

    snapshot = evidence.capture_memory_snapshot()

    assert snapshot.system_total_bytes == 32 * GIB
    assert snapshot.accelerator_kind == "nvidia"
    assert snapshot.accelerator_count == 1
    assert snapshot.accelerator_total_bytes == 8192 * 1024 ** 2
    assert snapshot.accelerator_available_bytes == 2048 * 1024 ** 2


def test_capture_memory_snapshot_marks_apple_unified_and_fails_soft(monkeypatch):
    monkeypatch.setattr(
        evidence.psutil, "virtual_memory",
        lambda: SimpleNamespace(total=24 * GIB, available=12 * GIB))
    monkeypatch.setattr(evidence.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(evidence.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(evidence.hardware, "clamp_ram_to_cgroup", lambda total: total)
    apple = evidence.capture_memory_snapshot()
    assert apple.unified is True and apple.accelerator_kind == "apple"

    monkeypatch.setattr(
        evidence.psutil, "virtual_memory", lambda: (_ for _ in ()).throw(OSError("no memory")))
    missing = evidence.capture_memory_snapshot()
    assert missing.system_total_bytes is None
    assert missing.system_available_bytes is None


def test_snapshot_helpers_fail_soft_for_absent_malformed_and_failed_nvidia(monkeypatch):
    monkeypatch.setattr(evidence.shutil, "which", lambda name: None)
    assert evidence._nvidia_memory() == (0, None, None)

    monkeypatch.setattr(evidence.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    for output in ("", "1, 2, 3\n"):
        monkeypatch.setattr(evidence, "_run", lambda command, value=output: value)
        assert evidence._nvidia_memory() == (0, None, None)
    monkeypatch.setattr(evidence, "_run", lambda command: (_ for _ in ()).throw(OSError("bad")))
    assert evidence._nvidia_memory() == (0, None, None)


def test_run_uses_a_bounded_read_only_subprocess(monkeypatch):
    seen = {}

    def run(command, **kwargs):
        seen.update(command=command, kwargs=kwargs)
        return SimpleNamespace(stdout="ok\n")

    monkeypatch.setattr(evidence.subprocess, "run", run)
    assert evidence._run(["nvidia-smi"]) == "ok\n"
    assert seen == {
        "command": ["nvidia-smi"],
        "kwargs": {"check": True, "capture_output": True, "text": True, "timeout": 5},
    }


def test_capture_non_arm_darwin_is_not_mislabeled_as_unified(monkeypatch):
    monkeypatch.setattr(
        evidence.psutil, "virtual_memory",
        lambda: SimpleNamespace(total=16 * GIB, available=8 * GIB))
    monkeypatch.setattr(evidence.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(evidence.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(evidence.hardware, "clamp_ram_to_cgroup", lambda total: total)
    monkeypatch.setattr(evidence.shutil, "which", lambda name: None)

    snapshot = evidence.capture_memory_snapshot()

    assert snapshot.unified is False and snapshot.accelerator_kind is None


def test_system_snapshot_clamps_available_to_the_effective_memory_wall(monkeypatch):
    monkeypatch.setattr(
        evidence.psutil, "virtual_memory",
        lambda: SimpleNamespace(total=32 * GIB, available=20 * GIB))
    monkeypatch.setattr(
        evidence.hardware, "clamp_ram_to_cgroup", lambda total: 8 * GIB)

    assert evidence._system_memory() == (8 * GIB, 8 * GIB)
