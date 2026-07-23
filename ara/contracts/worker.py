# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The worker response contract: one leaf ``(context, memory)`` measurement.

This is the seam between ARA and an engine's isolated worker. The worker — running in the
engine env, with no ``ara`` available — emits a single JSON object; ARA validates it here.

Response shapes (one JSON line):
  success:  ``{"context": <int>, "mem_gb": <number>}``
  refusal:  ``{"context": <int>, "refused": true, "reason": "<why>"}``   (RULE #1 pre-flight)

Each backend's worker is engine-native (Apple uses ``ara_engine_mlx.measure_one``; CPU will use a
llama.cpp script); the adapter maps the engine's raw output into this canonical shape, and
:func:`parse` guarantees ARA never feeds a malformed reading into the ramp fit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


class WorkerProtocolError(ValueError):
    """A worker emitted JSON that doesn't satisfy the measurement contract."""


@dataclass(frozen=True)
class Measurement:
    """One safe measurement. ``mem_gb`` is None exactly when ``refused`` is True."""
    context: int
    mem_gb: float | None
    refused: bool = False
    reason: str | None = None
    telemetry: dict | None = None


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


CUDA_GGUF_TWO_WALL_SCHEMA = "cuda-gguf-two-wall-telemetry:v1"


def _finite_number(payload: dict, field: str, *, positive: bool = False) -> float:
    value = payload.get(field)
    if not _is_number(value):
        raise WorkerProtocolError(
            f"two-wall telemetry '{field}' must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        qualifier = "positive " if positive else ""
        raise WorkerProtocolError(
            f"two-wall telemetry '{field}' must be a finite {qualifier}number")
    return value


def _exact_fields(payload: dict, expected: set[str], label: str) -> None:
    missing = expected - set(payload)
    if missing:
        raise WorkerProtocolError(
            f"two-wall telemetry {label} missing '{sorted(missing)[0]}'")
    unknown = set(payload) - expected
    if unknown:
        raise WorkerProtocolError(
            f"two-wall telemetry {label} has unknown field '{sorted(unknown)[0]}'")


def validate_two_wall_telemetry(
    telemetry: dict,
    mem_gb: float,
    *,
    expected_ram_budget_gb: float | None = None,
) -> None:
    """Validate CUDA-GGUF's dimension-bound RAM fit and both physical-wall observations."""

    _exact_fields(
        telemetry,
        {
            "schema", "fit_dimension", "unit", "gpu_layers",
            "vram", "ram", "provenance",
        },
        "payload",
    )
    if telemetry.get("schema") != CUDA_GGUF_TWO_WALL_SCHEMA:
        raise WorkerProtocolError("two-wall telemetry schema is missing or unsupported")
    if telemetry.get("fit_dimension") != "ram_absolute":
        raise WorkerProtocolError(
            "two-wall telemetry fit_dimension must be 'ram_absolute'")
    if telemetry.get("unit") != "GiB":
        raise WorkerProtocolError("two-wall telemetry unit must be 'GiB'")
    gpu_layers = telemetry.get("gpu_layers")
    if not isinstance(gpu_layers, int) or isinstance(gpu_layers, bool) or gpu_layers <= 0:
        raise WorkerProtocolError(
            "two-wall telemetry gpu_layers must be a positive integer")

    vram = telemetry.get("vram")
    ram = telemetry.get("ram")
    provenance = telemetry.get("provenance")
    if not isinstance(vram, dict) or not isinstance(ram, dict) or not isinstance(
        provenance, dict
    ):
        raise WorkerProtocolError(
            "two-wall telemetry walls and provenance must be objects")
    _exact_fields(vram, {"observed_gb", "budget_gb"}, "VRAM")
    _exact_fields(
        ram,
        {
            "observed_buffers_gb", "baseline_gb",
            "observed_absolute_gb", "budget_gb",
        },
        "RAM",
    )
    _exact_fields(
        provenance,
        {
            "source", "aggregation", "repeat_count",
            "vram_buffer_lines", "ram_buffer_lines",
        },
        "provenance",
    )
    vram_observed = _finite_number(vram, "observed_gb", positive=True)
    vram_budget = _finite_number(vram, "budget_gb", positive=True)
    ram_buffers = _finite_number(ram, "observed_buffers_gb", positive=True)
    ram_baseline = _finite_number(ram, "baseline_gb")
    ram_absolute = _finite_number(ram, "observed_absolute_gb", positive=True)
    ram_budget = _finite_number(ram, "budget_gb", positive=True)
    if not math.isclose(
        ram_absolute, ram_baseline + ram_buffers, rel_tol=0.0, abs_tol=1e-4
    ):
        raise WorkerProtocolError(
            "two-wall telemetry RAM absolute observation contradicts its components")
    if not math.isclose(mem_gb, ram_absolute, rel_tol=0.0, abs_tol=1e-9):
        raise WorkerProtocolError(
            "two-wall telemetry RAM fit value contradicts measurement 'mem_gb'")
    if vram_observed >= vram_budget or ram_absolute >= ram_budget:
        raise WorkerProtocolError(
            "two-wall telemetry contains an observation at/over its safe budget")
    if (
        expected_ram_budget_gb is not None
        and not math.isclose(
            ram_budget, expected_ram_budget_gb, rel_tol=0.0, abs_tol=1e-9
        )
    ):
        raise WorkerProtocolError(
            "two-wall telemetry RAM budget contradicts preflight")
    if provenance.get("source") != "llama.cpp-load-log":
        raise WorkerProtocolError(
            "two-wall telemetry provenance source is unsupported")
    if provenance.get("aggregation") not in {"single", "median"}:
        raise WorkerProtocolError(
            "two-wall telemetry provenance aggregation is unsupported")
    repeat_count = provenance.get("repeat_count")
    if (
        not isinstance(repeat_count, int)
        or isinstance(repeat_count, bool)
        or repeat_count <= 0
    ):
        raise WorkerProtocolError(
            "two-wall telemetry provenance 'repeat_count' must be a positive integer")
    for field in ("vram_buffer_lines", "ram_buffer_lines"):
        count = provenance.get(field)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise WorkerProtocolError(
                f"two-wall telemetry provenance '{field}' must be a positive integer")


def parse(payload: dict) -> Measurement:
    """Validate a worker's JSON object into a :class:`Measurement`, or raise."""
    ctx = payload.get("context")
    if not isinstance(ctx, int) or isinstance(ctx, bool):
        raise WorkerProtocolError("worker payload missing integer 'context'")
    if ctx <= 0:
        raise WorkerProtocolError("worker 'context' must be a positive integer")
    telemetry = payload.get("telemetry")
    if telemetry is not None and not isinstance(telemetry, dict):
        raise WorkerProtocolError("worker 'telemetry' must be an object")
    if payload.get("refused"):
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason:
            raise WorkerProtocolError("refused measurement needs a non-empty 'reason'")
        return Measurement(context=ctx, mem_gb=None, refused=True, reason=reason,
                           telemetry=telemetry)
    mem = payload.get("mem_gb")
    if not _is_number(mem):
        raise WorkerProtocolError("measurement needs numeric 'mem_gb'")
    mem = float(mem)
    if not math.isfinite(mem) or mem < 0:
        raise WorkerProtocolError("measurement 'mem_gb' must be finite non-negative")
    if (
        isinstance(telemetry, dict)
        and telemetry.get("schema") == CUDA_GGUF_TWO_WALL_SCHEMA
    ):
        validate_two_wall_telemetry(telemetry, mem)
    return Measurement(context=ctx, mem_gb=mem, refused=False, reason=None,
                       telemetry=telemetry)
