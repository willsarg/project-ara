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

from ara import hardware, ollama


_MIB = 1024 ** 2
SYSTEM_MARGIN_BYTES = 2 * 1024 ** 3
ACCELERATOR_MARGIN_BYTES = 1 * 1024 ** 3


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
