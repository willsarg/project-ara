# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Backend-specific authority for durable memory measurements."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass

from ara.engine_identity import canonical_engine

UNSCOPED_ENVIRONMENT_KEY = "environment:v1:unscoped"
UNSCOPED_AUTHORITY_KEY = "authority:v1:unscoped"
LEGACY_UNIT_UNKNOWN_ENVIRONMENT_KEY = "environment:v1:legacy-unit-unknown"
LEGACY_UNIT_UNKNOWN_AUTHORITY_KEY = "authority:v1:legacy-unit-unknown"


@dataclass(frozen=True)
class EnvironmentAuthority:
    key: str
    evidence: dict


@dataclass(frozen=True)
class MeasurementAuthority:
    key: str
    environment_key: str
    evidence: dict


def _read_text(argv: list[str]) -> str | None:
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _key(prefix: str, evidence: dict) -> str:
    encoded = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"{prefix}:sha256:{digest}"


def _unscoped_environment() -> EnvironmentAuthority:
    return EnvironmentAuthority(
        key=UNSCOPED_ENVIRONMENT_KEY,
        evidence={"schema": "unscoped-environment:v1", "scope": "unscoped"},
    )


def current_environment(engine: str) -> EnvironmentAuthority | None:
    if canonical_engine(engine) != "mlx":
        return _unscoped_environment()
    if platform.system() != "Darwin":
        return None

    macos_version = platform.mac_ver()[0]
    macos_build = _read_text(["/usr/bin/sw_vers", "-buildVersion"])
    kernel_release = platform.release()
    wired_limit = _read_text(["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"])
    dynamic_lwm = _read_text(["/usr/sbin/sysctl", "-n", "iogpu.dynamic_lwm"])
    if not all((macos_version, macos_build, kernel_release, wired_limit, dynamic_lwm)):
        return None
    try:
        wired_limit_mb = int(wired_limit)
        dynamic_lwm_value = int(dynamic_lwm)
    except ValueError:
        return None

    evidence = {
        "schema": "mlx-environment:v1",
        "system": "Darwin",
        "macos_version": macos_version,
        "macos_build": macos_build,
        "kernel_release": kernel_release,
        "iogpu_wired_limit_mb": wired_limit_mb,
        "iogpu_dynamic_lwm": dynamic_lwm_value,
    }
    return EnvironmentAuthority(
        key=_key("mlx-environment:v1", evidence),
        evidence=evidence,
    )


def measurement_authority(
    engine: str,
    limits: dict,
    *,
    environment: EnvironmentAuthority | None = None,
) -> MeasurementAuthority | None:
    if canonical_engine(engine) != "mlx":
        return MeasurementAuthority(
            key=UNSCOPED_AUTHORITY_KEY,
            environment_key=UNSCOPED_ENVIRONMENT_KEY,
            evidence={"schema": "unscoped-authority:v1", "scope": "unscoped"},
        )

    environment = environment or current_environment("mlx")
    if environment is None:
        return None
    device = limits.get("device")
    if not isinstance(device, str) or not device.strip():
        return None
    if limits.get("memory_unit") != "GiB":
        return None

    byte_fields = (
        "memory_size_bytes",
        "recommended_working_set_bytes",
        "max_buffer_length_bytes",
    )
    for field in byte_fields:
        value = limits.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None

    memory = {
        "unit": "GiB",
        "device": device.strip(),
        **{field: limits[field] for field in byte_fields},
    }
    evidence = {
        "schema": "mlx-memory-authority:v1",
        "environment": environment.evidence,
        "memory": memory,
    }
    return MeasurementAuthority(
        key=_key("mlx-memory-authority:v1", evidence),
        environment_key=environment.key,
        evidence=evidence,
    )


def _mlx_limits() -> dict:
    from ara.backends.apple import safe_limits

    return safe_limits()


def current_measurement_authority(engine: str) -> MeasurementAuthority | None:
    """Read the exact authority governing a measurement taken now.

    Non-MLX engines retain their explicit unscoped authority. MLX reads its device-limit worker;
    failures are unknown and therefore cannot authorize reuse.
    """
    canonical = canonical_engine(engine)
    environment = current_environment(canonical or engine)
    if environment is None:
        return None
    if canonical != "mlx":
        return measurement_authority(canonical or engine, {}, environment=environment)
    try:
        limits = _mlx_limits()
    except (Exception, SystemExit):
        return None
    return measurement_authority("mlx", limits, environment=environment)


def measurement_status(
    row: dict,
    current: MeasurementAuthority | None,
) -> str:
    """Classify stored evidence relative to a live authority read."""
    stored = row.get("authority_key")
    if stored == LEGACY_UNIT_UNKNOWN_AUTHORITY_KEY:
        return "legacy-unit-unknown"
    if current is None or not isinstance(stored, str):
        return "unknown"
    return "current" if stored == current.key else "stale"
