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
    _rewrite_characterization_config(
        store,
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


def test_preflight_checks_available_physical_capacity_before_model_load():
    assert evidence.preflight_refusal_reason(_snapshot(available=8), 4 * GIB) is None
    assert evidence.preflight_refusal_reason(
        _snapshot(available=3), 4 * GIB) == "model_exceeds_available_memory_walls"

    discrete = _snapshot(
        total=32, available=6, kind="nvidia", count=1,
        accelerator_total=8, accelerator_available=7)
    assert evidence.preflight_refusal_reason(discrete, 8 * GIB) is None
    assert evidence.preflight_refusal_reason(discrete, 11 * GIB) == (
        "model_exceeds_available_memory_walls")


def test_preflight_fails_closed_for_unknown_size_wall_or_multi_accelerator():
    missing = evidence.MemorySnapshot(None, None, None, None, None, None, False)
    assert evidence.preflight_refusal_reason(missing, 1) == "system_wall_unknown"
    assert evidence.preflight_refusal_reason(_snapshot(), None) == "model_size_unknown"
    assert evidence.preflight_refusal_reason(
        _snapshot(kind="nvidia", count=2), 1) == "placement_unknown"
    assert evidence.preflight_refusal_reason(
        evidence.MemorySnapshot(16 * GIB, None, None, 0, None, None, False), 1,
    ) == "system_wall_unknown"
    assert evidence.preflight_refusal_reason(
        evidence.MemorySnapshot(16 * GIB, 8 * GIB, "nvidia", 1, 8 * GIB, None, False), 1,
    ) == "accelerator_wall_unknown"


@pytest.mark.parametrize("model_size", [True, 0, "large"])
def test_preflight_rejects_invalid_model_sizes(model_size):
    assert evidence.preflight_refusal_reason(
        _snapshot(), model_size) == "model_size_unknown"


def test_preflight_capacity_never_treats_reserved_margin_as_available():
    assert evidence.preflight_refusal_reason(
        _snapshot(available=1), 1) == "model_exceeds_available_memory_walls"
    assert evidence.preflight_refusal_reason(
        _snapshot(
            total=16, available=1, kind="nvidia", count=1,
            accelerator_total=1, accelerator_available=0.5),
        1,
    ) == "model_exceeds_available_memory_walls"


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
