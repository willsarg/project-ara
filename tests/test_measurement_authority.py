# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Backend-specific authority for persisted memory measurements."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from ara import measurement_authority as authority


_LIMITS = {
    "device": "Apple M4 Pro",
    "memory_unit": "GiB",
    "memory_size_bytes": 25_769_803_776,
    "recommended_working_set_bytes": 19_069_665_280,
    "max_buffer_length_bytes": 9_000_000_000,
}


def test_read_text_returns_stripped_stdout(monkeypatch) -> None:
    monkeypatch.setattr(
        authority.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=" value\n"),
    )

    assert authority._read_text(["tool", "arg"]) == "value"


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(returncode=1, stdout="ignored"),
        SimpleNamespace(returncode=0, stdout="  \n"),
    ],
)
def test_read_text_returns_none_for_unusable_output(monkeypatch, result) -> None:
    monkeypatch.setattr(authority.subprocess, "run", lambda *args, **kwargs: result)
    assert authority._read_text(["tool"]) is None


@pytest.mark.parametrize("error", [OSError("missing"), subprocess.TimeoutExpired("tool", 2)])
def test_read_text_returns_none_when_probe_fails(monkeypatch, error) -> None:
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(authority.subprocess, "run", fail)
    assert authority._read_text(["tool"]) is None


def _darwin(monkeypatch, *, build="25F84", wired="0", dynamic="1") -> None:
    monkeypatch.setattr(authority.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(authority.platform, "mac_ver", lambda: ("26.5.2", ("", "", ""), "arm64"))
    monkeypatch.setattr(authority.platform, "release", lambda: "25.5.0")
    values = {
        ("/usr/bin/sw_vers", "-buildVersion"): build,
        ("/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"): wired,
        ("/usr/sbin/sysctl", "-n", "iogpu.dynamic_lwm"): dynamic,
    }
    monkeypatch.setattr(authority, "_read_text", lambda argv: values.get(tuple(argv)))


def test_current_mlx_environment_is_deterministic_and_read_only(monkeypatch) -> None:
    _darwin(monkeypatch)

    first = authority.current_environment("mlx")
    second = authority.current_environment("mlx")

    assert first == second
    assert first is not None
    assert first.key.startswith("mlx-environment:v1:sha256:")
    assert first.evidence == {
        "schema": "mlx-environment:v1",
        "system": "Darwin",
        "macos_version": "26.5.2",
        "macos_build": "25F84",
        "kernel_release": "25.5.0",
        "iogpu_wired_limit_mb": 0,
        "iogpu_dynamic_lwm": 1,
    }


def test_mlx_environment_changes_with_build_or_memory_policy(monkeypatch) -> None:
    _darwin(monkeypatch, build="25F84", wired="0")
    first = authority.current_environment("mlx")
    _darwin(monkeypatch, build="25F85", wired="20480")
    second = authority.current_environment("mlx")
    assert first is not None and second is not None and first.key != second.key


def test_mlx_environment_fails_closed_when_required_fact_is_unknown(monkeypatch) -> None:
    _darwin(monkeypatch, build=None)
    assert authority.current_environment("mlx") is None


def test_mlx_environment_fails_closed_when_policy_is_not_an_integer(monkeypatch) -> None:
    _darwin(monkeypatch, wired="not-an-integer")
    assert authority.current_environment("mlx") is None


def test_mlx_environment_fails_closed_off_darwin(monkeypatch) -> None:
    monkeypatch.setattr(authority.platform, "system", lambda: "Linux")
    assert authority.current_environment("mlx") is None


def test_non_mlx_environment_is_explicitly_unscoped(monkeypatch) -> None:
    monkeypatch.setattr(authority.platform, "system", lambda: "Linux")
    result = authority.current_environment("cpu")
    assert result is not None
    assert result.key == authority.UNSCOPED_ENVIRONMENT_KEY
    assert result.evidence == {"schema": "unscoped-environment:v1", "scope": "unscoped"}


def test_mlx_measurement_authority_uses_exact_bytes(monkeypatch) -> None:
    _darwin(monkeypatch)
    environment = authority.current_environment("mlx")
    result = authority.measurement_authority("mlx", _LIMITS, environment=environment)

    assert result is not None and environment is not None
    assert result.environment_key == environment.key
    assert result.key.startswith("mlx-memory-authority:v1:sha256:")
    assert result.evidence["memory"] == {
        "unit": "GiB",
        "device": "Apple M4 Pro",
        "memory_size_bytes": 25_769_803_776,
        "recommended_working_set_bytes": 19_069_665_280,
        "max_buffer_length_bytes": 9_000_000_000,
    }


def test_mlx_measurement_authority_changes_with_exact_wall(monkeypatch) -> None:
    _darwin(monkeypatch)
    environment = authority.current_environment("mlx")
    first = authority.measurement_authority("mlx", _LIMITS, environment=environment)
    changed = {**_LIMITS, "recommended_working_set_bytes": 17_179_885_568}
    second = authority.measurement_authority("mlx", changed, environment=environment)
    assert first is not None and second is not None and first.key != second.key


def test_mlx_measurement_authority_rejects_rounded_or_ambiguous_facts(monkeypatch) -> None:
    _darwin(monkeypatch)
    environment = authority.current_environment("mlx")
    for limits in (
        {**_LIMITS, "memory_unit": "GB"},
        {**_LIMITS, "recommended_working_set_bytes": 17.76},
        {**_LIMITS, "device": ""},
    ):
        assert authority.measurement_authority(
            "mlx", limits, environment=environment) is None


def test_mlx_measurement_authority_requires_environment(monkeypatch) -> None:
    monkeypatch.setattr(authority, "current_environment", lambda _engine: None)
    assert authority.measurement_authority("mlx", _LIMITS) is None


def test_non_mlx_measurement_authority_preserves_existing_scope() -> None:
    result = authority.measurement_authority("cpu", {})
    assert result is not None
    assert result.key == authority.UNSCOPED_AUTHORITY_KEY
    assert result.environment_key == authority.UNSCOPED_ENVIRONMENT_KEY


def test_current_measurement_authority_reads_live_mlx_limits(monkeypatch) -> None:
    _darwin(monkeypatch)
    monkeypatch.setattr(authority, "_mlx_limits", lambda: dict(_LIMITS))

    result = authority.current_measurement_authority("mlx")

    assert result is not None
    assert result.evidence["memory"]["recommended_working_set_bytes"] == 19_069_665_280


def test_mlx_limits_delegates_to_apple_backend(monkeypatch) -> None:
    monkeypatch.setattr("ara.backends.apple.safe_limits", lambda: {"live": True})
    assert authority._mlx_limits() == {"live": True}


def test_current_measurement_authority_requires_environment(monkeypatch) -> None:
    monkeypatch.setattr(authority, "current_environment", lambda _engine: None)
    assert authority.current_measurement_authority("mlx") is None


def test_current_measurement_authority_fails_closed_when_worker_fails(monkeypatch) -> None:
    _darwin(monkeypatch)

    def fail():
        raise RuntimeError("worker unavailable")

    monkeypatch.setattr(authority, "_mlx_limits", fail)
    assert authority.current_measurement_authority("mlx") is None


def test_measurement_status_distinguishes_current_stale_and_legacy(monkeypatch) -> None:
    _darwin(monkeypatch)
    current = authority.measurement_authority(
        "mlx", _LIMITS, environment=authority.current_environment("mlx"))
    assert current is not None

    assert authority.measurement_status({"authority_key": current.key}, current) == "current"
    assert authority.measurement_status({"authority_key": "older"}, current) == "stale"
    assert authority.measurement_status({
        "authority_key": authority.LEGACY_UNIT_UNKNOWN_AUTHORITY_KEY,
    }, current) == "legacy-unit-unknown"
    assert authority.measurement_status({}, None) == "unknown"
