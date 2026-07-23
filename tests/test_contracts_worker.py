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


def test_parse_preserves_optional_telemetry_for_success_and_refusal():
    telemetry = {"schema": "macos-native-vm-telemetry:v1", "sample_count": 3}

    measured = worker.parse({"context": 4096, "mem_gb": 8.2, "telemetry": telemetry})
    refused = worker.parse({
        "context": 8192, "refused": True, "reason": "boundary", "telemetry": telemetry})

    assert measured.telemetry == telemetry
    assert refused.telemetry == telemetry


def _two_wall_payload():
    return {
        "context": 4096,
        "mem_gb": 6.0,
        "telemetry": {
            "schema": "cuda-gguf-two-wall-telemetry:v1",
            "fit_dimension": "ram_absolute",
            "unit": "GiB",
            "gpu_layers": 16,
            "vram": {"observed_gb": 4.0, "budget_gb": 8.0},
            "ram": {
                "observed_buffers_gb": 5.0,
                "baseline_gb": 1.0,
                "observed_absolute_gb": 6.0,
                "budget_gb": 20.0,
            },
            "provenance": {
                "source": "llama.cpp-load-log",
                "aggregation": "median",
                "repeat_count": 3,
                "vram_buffer_lines": 3,
                "ram_buffer_lines": 2,
            },
        },
    }


def test_parse_accepts_dimension_bound_two_wall_measurement():
    measured = worker.parse(_two_wall_payload())

    assert measured.mem_gb == 6.0
    assert measured.telemetry["vram"]["observed_gb"] == 4.0
    assert measured.telemetry["ram"]["observed_absolute_gb"] == 6.0


def test_two_wall_measurement_rejects_contradictory_fit_and_ram_components():
    payload = _two_wall_payload()
    payload["mem_gb"] = 7.0
    with pytest.raises(worker.WorkerProtocolError, match="RAM fit value"):
        worker.parse(payload)

    payload = _two_wall_payload()
    payload["telemetry"]["ram"]["observed_absolute_gb"] = 7.0
    with pytest.raises(worker.WorkerProtocolError, match="components"):
        worker.parse(payload)


def test_two_wall_measurement_rejects_missing_wall_and_unsafe_observation():
    payload = _two_wall_payload()
    del payload["telemetry"]["vram"]["observed_gb"]
    with pytest.raises(worker.WorkerProtocolError, match="VRAM missing"):
        worker.parse(payload)

    payload = _two_wall_payload()
    payload["telemetry"]["vram"]["observed_gb"] = 8.0
    with pytest.raises(worker.WorkerProtocolError, match="at/over"):
        worker.parse(payload)


def test_two_wall_measurement_rejects_invalid_provenance():
    payload = _two_wall_payload()
    payload["telemetry"]["provenance"]["source"] = "rss"
    with pytest.raises(worker.WorkerProtocolError, match="source"):
        worker.parse(payload)

    payload = _two_wall_payload()
    payload["telemetry"]["provenance"]["repeat_count"] = 0
    with pytest.raises(worker.WorkerProtocolError, match="repeat_count"):
        worker.parse(payload)


def _set_nested(payload, path, value):
    target = payload["telemetry"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return payload


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("vram", "observed_gb"), "four", "numeric"),
        (("vram", "observed_gb"), math.nan, "finite positive"),
        (("ram", "baseline_gb"), -1.0, "finite number"),
        (("fit_dimension",), "vram", "fit_dimension"),
        (("unit",), "GB", "unit"),
        (("gpu_layers",), "16", "gpu_layers"),
        (("gpu_layers",), True, "gpu_layers"),
        (("gpu_layers",), 0, "gpu_layers"),
        (("vram",), [], "objects"),
        (("ram",), [], "objects"),
        (("provenance",), [], "objects"),
        (("provenance", "aggregation"), "mean", "aggregation"),
        (("provenance", "repeat_count"), "3", "repeat_count"),
        (("provenance", "repeat_count"), True, "repeat_count"),
        (("provenance", "vram_buffer_lines"), 0, "vram_buffer_lines"),
        (("provenance", "ram_buffer_lines"), False, "ram_buffer_lines"),
    ],
)
def test_two_wall_measurement_rejects_malformed_fields(path, value, message):
    payload = _set_nested(_two_wall_payload(), path, value)

    with pytest.raises(worker.WorkerProtocolError, match=message):
        worker.parse(payload)


def test_two_wall_measurement_rejects_unknown_fields_and_schema():
    payload = _two_wall_payload()
    payload["telemetry"]["future"] = True
    with pytest.raises(worker.WorkerProtocolError, match="unknown field"):
        worker.parse(payload)

    payload = _two_wall_payload()
    payload["telemetry"]["schema"] = "cuda-gguf-two-wall-telemetry:v0"
    with pytest.raises(worker.WorkerProtocolError, match="schema"):
        worker.validate_two_wall_telemetry(
            payload["telemetry"], payload["mem_gb"])


def test_two_wall_measurement_rejects_preflight_budget_contradiction():
    payload = _two_wall_payload()

    with pytest.raises(worker.WorkerProtocolError, match="budget contradicts preflight"):
        worker.validate_two_wall_telemetry(
            payload["telemetry"],
            payload["mem_gb"],
            expected_ram_budget_gb=21.0,
        )


def test_parse_rejects_non_object_telemetry():
    with pytest.raises(worker.WorkerProtocolError, match="telemetry"):
        worker.parse({"context": 4096, "mem_gb": 8.2, "telemetry": []})


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
