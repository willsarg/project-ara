# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The domain→JSON-ready interchange seam: where ARA's *domain objects* become plain dicts.

This is the one place a :class:`~ara.detect.Machine` (and the durable capability projection
of it) turns into JSON-ready data — "JSON to travel, DB to query". Fleet mode ships node
identity over the wire as these dicts, so the serialization lives here, not scattered across
commands. Keep it for DOMAIN objects (the Machine, the profile record); per-command result
payloads (estimate/recommend/run) are not domain objects and stay in the CLI.

Spec 2026-06-23-capability-pipeline (Slice 1)."""
from __future__ import annotations

import dataclasses

from ara import detect


def machine(m: detect.Machine) -> dict:
    """The full JSON-ready Machine — exactly the ``detect --json`` shape.

    ``dataclasses.asdict`` expands the nested hardware/runtime structures but drops the
    ``accelerated`` @property, so add it back explicitly. This is the single source of truth
    for ``detect --json``."""
    return {**dataclasses.asdict(m), "accelerated": m.accelerated}


# The durable capability fields the profile projection keeps. EXCLUDED (live/transient, would
# cause false drift): ram_available_gb, disk_free_gb, power, model_stores, apps, hf_token,
# hf_cli, hf_cli_version, python_version, framework_python, and the live-bearing nested
# `memory`/`storage` structures (available_gb/free_gb fluctuate).
_PROFILE_SCALARS = (
    "system", "os_version", "arch", "chip",
    "cpu_physical", "cpu_logical", "cpu_features",
    "ram_total_gb", "swap_gb",
    "backend", "engine", "engine_ready",
)


def profile_record(m: detect.Machine) -> dict:
    """The CURATED DURABLE capability projection of a Machine, for drift history + fleet queries.

    Built as an EXPLICIT allow-list (a curated pick, not asdict-minus-fields) so it stays STABLE
    across re-runs on an unchanged machine: re-capturing seconds apart yields a byte-identical
    projection (no false drift). It therefore includes only durable capability and excludes every
    live/transient field (available memory, free disk, power, current apps/models, tokens, …).
    Nested domain objects (accel, gpus, board, runtimes) come through asdict-expanded as
    JSON-ready dicts/lists."""
    rec = {name: getattr(m, name) for name in _PROFILE_SCALARS}
    rec["accel"] = dataclasses.asdict(m.accel)
    rec["gpus"] = [dataclasses.asdict(g) for g in m.gpus]
    rec["board"] = dataclasses.asdict(m.board)
    rec["runtimes"] = [dataclasses.asdict(r) for r in m.runtimes]
    return rec
