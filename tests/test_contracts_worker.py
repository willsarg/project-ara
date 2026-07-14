# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The worker response contract — ARA-side strict parsing of a leaf measurement.

The worker runs inside the engine env (mlx/torch), which has no ``ara`` installed, so it
can't import this. It emits one raw JSON object; ARA validates it here into a Measurement.
A success carries ``mem_gb``; a RULE #1 pre-flight refusal carries ``refused`` + ``reason``.
Malformed output raises loudly rather than yielding a bogus ceiling.
"""
from __future__ import annotations

import math

import pytest

from ara.contracts import worker


def test_parse_successful_measurement():
    m = worker.parse({"context": 4096, "mem_gb": 8.2})
    assert m == worker.Measurement(context=4096, mem_gb=8.2, refused=False, reason=None)


def test_parse_integer_mem_is_coerced_to_float():
    m = worker.parse({"context": 1024, "mem_gb": 7})
    assert m.mem_gb == 7.0 and isinstance(m.mem_gb, float)


def test_parse_refusal():
    m = worker.parse({"context": 131072, "refused": True, "reason": "base exceeds budget"})
    assert m == worker.Measurement(
        context=131072, mem_gb=None, refused=True, reason="base exceeds budget"
    )


def test_parse_rejects_missing_context():
    with pytest.raises(worker.WorkerProtocolError, match="context"):
        worker.parse({"mem_gb": 8.2})


def test_parse_rejects_bool_context():
    # bool is a subclass of int — guard against it sneaking through
    with pytest.raises(worker.WorkerProtocolError, match="context"):
        worker.parse({"context": True, "mem_gb": 8.2})


@pytest.mark.parametrize("context", [0, -1])
def test_parse_rejects_nonpositive_context(context):
    with pytest.raises(worker.WorkerProtocolError, match="positive"):
        worker.parse({"context": context, "mem_gb": 8.2})


def test_parse_rejects_neither_mem_nor_refused():
    with pytest.raises(worker.WorkerProtocolError, match="mem_gb"):
        worker.parse({"context": 4096})


def test_parse_rejects_non_numeric_mem():
    with pytest.raises(worker.WorkerProtocolError, match="mem_gb"):
        worker.parse({"context": 4096, "mem_gb": "lots"})


def test_parse_rejects_bool_mem():
    with pytest.raises(worker.WorkerProtocolError, match="mem_gb"):
        worker.parse({"context": 4096, "mem_gb": True})


@pytest.mark.parametrize("mem", [-1.0, math.nan, math.inf, -math.inf])
def test_parse_rejects_nonfinite_or_negative_mem(mem):
    with pytest.raises(worker.WorkerProtocolError, match="finite non-negative"):
        worker.parse({"context": 4096, "mem_gb": mem})


def test_parse_rejects_refusal_without_reason():
    with pytest.raises(worker.WorkerProtocolError, match="reason"):
        worker.parse({"context": 4096, "refused": True})


def test_parse_rejects_refusal_with_empty_reason():
    with pytest.raises(worker.WorkerProtocolError, match="reason"):
        worker.parse({"context": 4096, "refused": True, "reason": ""})
