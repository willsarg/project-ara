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


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def parse(payload: dict) -> Measurement:
    """Validate a worker's JSON object into a :class:`Measurement`, or raise."""
    ctx = payload.get("context")
    if not isinstance(ctx, int) or isinstance(ctx, bool):
        raise WorkerProtocolError("worker payload missing integer 'context'")
    if payload.get("refused"):
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason:
            raise WorkerProtocolError("refused measurement needs a non-empty 'reason'")
        return Measurement(context=ctx, mem_gb=None, refused=True, reason=reason)
    mem = payload.get("mem_gb")
    if not _is_number(mem):
        raise WorkerProtocolError("measurement needs numeric 'mem_gb'")
    return Measurement(context=ctx, mem_gb=float(mem), refused=False, reason=None)
