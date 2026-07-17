# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Topology and physical-wall evidence for Ollama characterization."""

from __future__ import annotations

from dataclasses import dataclass
import platform
import shutil
import subprocess
from typing import Any

import psutil

from ara import db, hardware, ollama


_MIB = 1024 ** 2
SYSTEM_MARGIN_BYTES = 2 * 1024 ** 3
ACCELERATOR_MARGIN_BYTES = 1 * 1024 ** 3
_OLLAMA_ARTIFACT_PREFIX = "ollama-manifest-sha256:"


@dataclass(frozen=True)
class MemorySnapshot:
    """One observation of the physical memory walls around an Ollama request."""

    system_total_bytes: int | None
    system_available_bytes: int | None
    accelerator_kind: str | None
    accelerator_count: int | None
    accelerator_total_bytes: int | None
    accelerator_available_bytes: int | None
    unified: bool


@dataclass(frozen=True)
class CharacterizationAssessment:
    """One display row and the separately gated row safe for automatic reuse."""

    display: dict[str, Any] | None
    reusable: dict[str, Any] | None
    reason: str | None


def _display_only(row: dict[str, Any], reason: str) -> CharacterizationAssessment:
    return CharacterizationAssessment(row, None, reason)


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _wall_evidence_complete(config: dict[str, Any], point: dict[str, Any]) -> bool:
    placement = config.get("placement")
    expected_walls = {
        "cpu": ["system"],
        "unified": ["system_unified"],
        "accelerator": ["system", "accelerator"],
        "partial_offload": ["system", "accelerator"],
    }[placement]
    evidence_fields = (
        "placement",
        "resident_total_bytes",
        "resident_accelerator_bytes",
        "applicable_walls",
        "system_memory_delta_bytes",
        "accelerator_memory_delta_bytes",
        "system_margin_bytes",
        "accelerator_margin_bytes",
    )
    if any(config.get(field) != point.get(field) for field in evidence_fields):
        return False
    total = config.get("resident_total_bytes")
    accelerator = config.get("resident_accelerator_bytes")
    if not _nonnegative_int(total) or total == 0 or not _nonnegative_int(accelerator):
        return False
    placement_residency = {
        "cpu": accelerator == 0,
        "unified": accelerator <= total,
        "accelerator": accelerator == total,
        "partial_offload": 0 < accelerator < total,
    }
    if not placement_residency[placement]:
        return False
    if config.get("applicable_walls") != expected_walls:
        return False
    if (
        not _nonnegative_int(config.get("system_memory_delta_bytes"))
        or config.get("system_margin_bytes") != SYSTEM_MARGIN_BYTES
    ):
        return False
    if "accelerator" in expected_walls:
        return (
            _nonnegative_int(config.get("accelerator_memory_delta_bytes"))
            and config.get("accelerator_margin_bytes") == ACCELERATOR_MARGIN_BYTES
        )
    return (
        config.get("accelerator_memory_delta_bytes") is None
        and config.get("accelerator_margin_bytes") is None
    )


def assess_characterization(
    con: Any,
    machine_key: str,
    model: ollama.OllamaModel,
    authority: ollama.OllamaRuntimeAuthority,
) -> CharacterizationAssessment:
    """Read Ollama history without allowing display-only evidence into governed decisions."""

    rows = db.list_characterizations_for_display(
        con, machine_key, runtime="ollama", logical_model_id=model.name)
    if not rows:
        return CharacterizationAssessment(None, None, "missing")
    expected_artifact = (
        _OLLAMA_ARTIFACT_PREFIX + model.digest if model.digest is not None else None)
    artifact_rows = [row for row in rows if row.get("artifact_id") == expected_artifact]
    candidates = artifact_rows or rows

    def current_authority(row: dict[str, Any]) -> bool:
        config = row.get("config")
        return (isinstance(config, dict)
                and config.get("endpoint_authority") == authority.endpoint.url
                and config.get("runtime_version") == authority.server_version
                and config.get("server_instance_id") == authority.server_instance_id
                and config.get("configured_inputs") == dict(authority.configured_inputs))

    row = max(candidates, key=lambda item: (
        current_authority(item), item.get("measured_at") or "",
        item.get("config_key") or ""))
    config = row.get("config")
    if not isinstance(config, dict) or config.get("methodology") != "ollama-physical-walls-v1":
        return _display_only(row, "methodology_missing_or_unsupported")
    if ollama.initial_governed_model_error(model) is not None:
        return _display_only(row, "unsupported_model_cell")
    if expected_artifact is None or row.get("artifact_id") != expected_artifact:
        return _display_only(row, "artifact_mismatch")
    safe_context = row.get("safe_context")
    if (
        not isinstance(safe_context, int)
        or isinstance(safe_context, bool)
        or safe_context <= 0
    ):
        return _display_only(row, "safe_context_missing")
    if authority.issue is not None:
        return _display_only(row, "runtime_authority_incomplete")
    if (
        authority.endpoint.scope != "loopback"
        or config.get("endpoint_authority") != authority.endpoint.url
    ):
        return _display_only(row, "endpoint_mismatch")
    if (
        config.get("runtime") != "ollama"
        or config.get("runtime_version") != authority.server_version
    ):
        return _display_only(row, "runtime_version_mismatch")
    if config.get("server_instance_id") != authority.server_instance_id:
        return _display_only(row, "server_instance_mismatch")
    if config.get("configured_inputs") != dict(authority.configured_inputs):
        return _display_only(row, "configured_inputs_mismatch")
    configured_inputs = dict(authority.configured_inputs)
    runtime_config = {
        "configured_kv_cache_type": configured_inputs.get(
            "OLLAMA_KV_CACHE_TYPE", "unknown"),
        "effective_kv_cache_type": "unknown",
        "configured_flash_attention": configured_inputs.get(
            "OLLAMA_FLASH_ATTENTION", "unknown"),
        "effective_flash_attention": "unknown",
        "configured_scheduler_spread": configured_inputs.get(
            "OLLAMA_SCHED_SPREAD", "unknown"),
        "effective_scheduler_spread": "unknown",
    }
    if any(config.get(key) != value for key, value in runtime_config.items()):
        return _display_only(row, "runtime_config_evidence_incomplete")
    if (
        authority.configured_num_parallel != 1
        or config.get("configured_num_parallel") != authority.configured_num_parallel
        or config.get("configured_num_parallel_authority")
        != authority.configured_num_parallel_authority
        or config.get("effective_num_parallel") != 1
        or config.get("effective_num_parallel_authority") != "configured_maximum_is_one"
    ):
        return _display_only(row, "parallelism_mismatch")
    if config.get("format") != "gguf" or config.get("capability") != "completion":
        return _display_only(row, "model_cell_mismatch")
    if (
        config.get("requested_context") != safe_context
        or config.get("effective_per_request_context") != safe_context
    ):
        return _display_only(row, "context_evidence_incomplete")
    points = row.get("points")
    point = next((
        item for item in points
        if isinstance(item, dict)
        and item.get("fit") is True
        and item.get("context") == safe_context
        and item.get("requested_context") == safe_context
        and item.get("effective_per_request_context") == safe_context
        and item.get("refusal_reasons") == []
    ), None) if isinstance(points, list) else None
    if point is None:
        return _display_only(row, "context_evidence_incomplete")
    if config.get("placement") not in {
        "cpu", "unified", "accelerator", "partial_offload",
    }:
        return _display_only(row, "placement_unsupported")
    if not _wall_evidence_complete(config, point):
        return _display_only(row, "wall_evidence_incomplete")
    if not row.get("reusable"):
        return _display_only(row, "storage_evidence_not_reusable")
    return CharacterizationAssessment(row, row, None)


def _run(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout


def _system_memory() -> tuple[int | None, int | None]:
    try:
        memory = psutil.virtual_memory()
    except (OSError, RuntimeError):
        return None, None
    total = hardware.clamp_ram_to_cgroup(int(memory.total))
    return total, min(total, int(memory.available))


def _nvidia_memory() -> tuple[int, int | None, int | None]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return 0, None, None
    try:
        output = _run([
            executable,
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        rows = [row.strip() for row in output.splitlines() if row.strip()]
        values = [tuple(int(item.strip()) for item in row.split(",")) for row in rows]
        if not values or any(len(value) != 2 for value in values):
            return 0, None, None
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0, None, None
    return (
        len(values),
        sum(value[0] for value in values) * _MIB,
        sum(value[1] for value in values) * _MIB,
    )


def capture_memory_snapshot() -> MemorySnapshot:
    """Observe system and accelerator walls without loading an ML runtime."""

    system_total, system_available = _system_memory()
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return MemorySnapshot(
            system_total, system_available, "apple", 1, None, None, True)

    count, accelerator_total, accelerator_available = _nvidia_memory()
    return MemorySnapshot(
        system_total,
        system_available,
        "nvidia" if count else None,
        count,
        accelerator_total,
        accelerator_available,
        False,
    )


def _placement(snapshot: MemorySnapshot, process: ollama.OllamaProcess) -> str:
    size = process.size_bytes
    accelerator = process.size_vram_bytes
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        return "unknown"
    if (
        not isinstance(accelerator, int)
        or isinstance(accelerator, bool)
        or accelerator < 0
        or accelerator > size
    ):
        return "unknown"
    if accelerator == 0:
        return "cpu"
    if snapshot.unified and snapshot.accelerator_kind == "apple":
        return "unified"
    if snapshot.accelerator_kind == "nvidia" and snapshot.accelerator_count == 1:
        if accelerator == size:
            return "accelerator"
        return "partial_offload"
    return "unknown"


def _delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return max(0, before - after)


def characterization_point(
    before: MemorySnapshot,
    after: MemorySnapshot,
    process: ollama.OllamaProcess,
    requested_context: int,
) -> dict[str, Any]:
    """Classify placement and prove every applicable physical wall has margin."""

    placement = _placement(after, process)
    if placement == "unified":
        walls = ["system_unified"]
    elif placement in {"accelerator", "partial_offload"}:
        walls = ["system", "accelerator"]
    elif placement == "cpu":
        walls = ["system"]
    else:
        walls = []

    reasons: list[str] = []
    if placement == "unknown":
        reasons.append("placement_unknown")
    else:
        if after.system_available_bytes is None or after.system_total_bytes is None:
            reasons.append("system_wall_unknown")
        elif after.system_available_bytes < SYSTEM_MARGIN_BYTES:
            reasons.append("system_margin_breached")
        if "accelerator" in walls:
            if (
                after.accelerator_available_bytes is None
                or after.accelerator_total_bytes is None
            ):
                reasons.append("accelerator_wall_unknown")
            elif after.accelerator_available_bytes < ACCELERATOR_MARGIN_BYTES:
                reasons.append("accelerator_margin_breached")
        if process.effective_context_per_request != requested_context:
            reasons.append("effective_context_mismatch")

    accelerator_delta = None
    accelerator_margin = None
    if "accelerator" in walls:
        accelerator_delta = _delta(
            before.accelerator_available_bytes, after.accelerator_available_bytes)
        accelerator_margin = ACCELERATOR_MARGIN_BYTES

    return {
        "context": requested_context,
        "requested_context": requested_context,
        "effective_per_request_context": process.effective_context_per_request,
        "fit": not reasons,
        "placement": placement,
        "resident_total_bytes": process.size_bytes,
        "resident_accelerator_bytes": process.size_vram_bytes,
        "system_memory_delta_bytes": _delta(
            before.system_available_bytes, after.system_available_bytes),
        "accelerator_memory_delta_bytes": accelerator_delta,
        "applicable_walls": walls,
        "system_margin_bytes": SYSTEM_MARGIN_BYTES,
        "accelerator_margin_bytes": accelerator_margin,
        "refusal_reasons": reasons,
    }


def failed_characterization_point(requested_context: int, reason: str) -> dict[str, Any]:
    """Return a schema-complete non-fit point when no residency can be attested."""

    return {
        "context": requested_context,
        "requested_context": requested_context,
        "effective_per_request_context": None,
        "fit": False,
        "placement": "unknown",
        "resident_total_bytes": None,
        "resident_accelerator_bytes": None,
        "system_memory_delta_bytes": None,
        "accelerator_memory_delta_bytes": None,
        "applicable_walls": [],
        "system_margin_bytes": SYSTEM_MARGIN_BYTES,
        "accelerator_margin_bytes": None,
        "refusal_reasons": [reason],
    }


def preflight_refusal_reason(
    snapshot: MemorySnapshot,
    model_size_bytes: int | None,
) -> str | None:
    """Refuse an obviously unsafe or unprovable model load before Ollama allocates it."""

    if snapshot.system_total_bytes is None or snapshot.system_available_bytes is None:
        return "system_wall_unknown"
    if (
        not isinstance(model_size_bytes, int)
        or isinstance(model_size_bytes, bool)
        or model_size_bytes <= 0
    ):
        return "model_size_unknown"
    if snapshot.accelerator_kind == "nvidia" and snapshot.accelerator_count != 1:
        return "placement_unknown"

    capacity = max(0, snapshot.system_available_bytes - SYSTEM_MARGIN_BYTES)
    if snapshot.accelerator_kind == "nvidia":
        if snapshot.accelerator_available_bytes is None:
            return "accelerator_wall_unknown"
        capacity += max(
            0, snapshot.accelerator_available_bytes - ACCELERATOR_MARGIN_BYTES)
    if model_size_bytes > capacity:
        return "model_exceeds_available_memory_walls"
    return None


def live_headroom_refusal_reason(
    snapshot: MemorySnapshot,
    config: dict[str, Any],
    *,
    resident: bool,
) -> str | None:
    """Require current capacity for the recorded peak before a governed request."""

    placement = config.get("placement")
    if snapshot.system_total_bytes is None or snapshot.system_available_bytes is None:
        return "system_wall_unknown"
    if placement == "unified" and not (
        snapshot.unified and snapshot.accelerator_kind == "apple"
    ):
        return "topology_drift"
    if placement in {"accelerator", "partial_offload"} and not (
        snapshot.accelerator_kind == "nvidia" and snapshot.accelerator_count == 1
    ):
        return "topology_drift"
    total = config.get("resident_total_bytes")
    accelerator = config.get("resident_accelerator_bytes")
    if not _nonnegative_int(total) or not _nonnegative_int(accelerator):
        return "wall_evidence_incomplete"
    system_peak = 0 if resident else (
        total - accelerator
        if placement in {"accelerator", "partial_offload"}
        else total
    )
    if not _nonnegative_int(system_peak):
        return "wall_evidence_incomplete"
    if snapshot.system_available_bytes < system_peak + SYSTEM_MARGIN_BYTES:
        return "system_headroom_insufficient"
    if placement in {"accelerator", "partial_offload"}:
        if (
            snapshot.accelerator_total_bytes is None
            or snapshot.accelerator_available_bytes is None
        ):
            return "accelerator_wall_unknown"
        accelerator_peak = 0 if resident else accelerator
        if snapshot.accelerator_available_bytes < (
            accelerator_peak + ACCELERATOR_MARGIN_BYTES
        ):
            return "accelerator_headroom_insufficient"
    return None


def live_residency_refusal_reason(
    snapshot: MemorySnapshot,
    process: ollama.OllamaProcess,
    config: dict[str, Any],
    safe_context: int,
) -> str | None:
    """Verify that current residency still matches the reusable measurement cell."""

    point = characterization_point(snapshot, snapshot, process, safe_context)
    if point["refusal_reasons"]:
        return point["refusal_reasons"][0]
    fields = (
        "placement",
        "applicable_walls",
    )
    if any(point[field] != config.get(field) for field in fields):
        return "placement_or_allocation_drift"
    return None
