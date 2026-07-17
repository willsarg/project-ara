# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""What this node tells the coordinator about itself — the enroll-time self-description.

Uses a durable random node identity for coordinator ownership, while reusing ARA's own recon
(``profile.machine_key`` for local measurements and ``detect`` for the accelerator) rather than
reinventing host probing. Shapes all of it to the pinned wire contract (``enroll.request`` + the
shared ``environment`` label).

The environment label is **container-honest** (Rule #1): it reads the *real* memory ceiling from
the cgroup, not just ``psutil``. A container capped below the host's RAM is labelled ``wall_source =
cgroup`` so a coordinator never mistakes a squeezed container for a bare-metal wall; a WSL2 or Docker
layer is surfaced in ``virtualization_layer``. ``capabilities`` advertises the models this node has
actually characterized (Rule #1 evidence), read from ARA's own store. All of it validates clean
against the schema.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import psutil

from ara import db, detect, hardware, profile
from ara.node import config

# platform.system() → the environment schema's platform enum (linux | darwin | windows).
_PLATFORMS = {"Linux": "linux", "Darwin": "darwin", "Windows": "windows"}
# detect.Accelerator.kind → the environment schema's accel enum.
_ACCELS = {"apple": "metal", "nvidia": "nvidia", "none": "cpu"}


def _read_text(path: str) -> str | None:
    """Read a proc/sys file, or None if it isn't there (non-Linux, no cgroup, no permission).

    The single filesystem boundary for the cgroup/container probes — tests mock this."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _path_exists(path: str) -> bool:
    """Whether a marker file exists (e.g. ``/.dockerenv``). Mocked in tests."""
    return Path(path).exists()


def _effective_wall() -> tuple[int, bool]:
    """``(effective_wall_bytes, cgroup_binds)`` — the memory ceiling this node should plan against.

    The wall is the smaller of physical RAM and any real cgroup limit (via the shared
    :func:`hardware.clamp_ram_to_cgroup`); ``cgroup_binds`` is True iff that clamp actually bit — a
    cgroup limit below physical (a container squeezed under the host). Off Linux there is no cgroup,
    so the wall is always physical."""
    physical = psutil.virtual_memory().total
    wall = hardware.clamp_ram_to_cgroup(physical)
    return wall, wall < physical


def effective_wall() -> int:
    """The memory ceiling (bytes) this node should plan against — the binding cgroup limit when a
    container caps below the host, else physical RAM. Exposed for future gate use."""
    return _effective_wall()[0]


def is_cgroup_bound() -> bool:
    """True when a cgroup memory limit below physical RAM is the binding ceiling."""
    return _effective_wall()[1]


def _containerized(cgroup_binds: bool) -> bool:
    """Whether this node runs inside a container. True on a Docker marker file, a container manager
    named in the cgroup lineage, or a binding cgroup memory limit."""
    if _path_exists("/.dockerenv"):
        return True
    for proc in ("/proc/1/cgroup", "/proc/self/cgroup"):
        text = _read_text(proc)
        if text and any(marker in text for marker in ("docker", "containerd", "kubepods")):
            return True
    return cgroup_binds


def _virtualization_layer() -> str | None:
    """The virtualization layer, if any: ``"wsl2"`` (``/proc/version`` mentions microsoft),
    ``"docker"`` (the docker marker file), else None (bare-metal or non-Linux)."""
    version = _read_text("/proc/version")
    if version and "microsoft" in version.lower():
        return "wsl2"
    if _path_exists("/.dockerenv"):
        return "docker"
    return None


def environment() -> dict:
    """The shared ``environment`` label for this node (schema: ``environment.json``).

    Container-honest: ``wall_source`` is ``"cgroup"`` when a cgroup limit binds below physical RAM,
    else ``"physical"``; ``containerized`` and ``virtualization_layer`` surface a container/WSL2
    wall so it's never mistaken for bare metal."""
    _wall, cgroup_binds = _effective_wall()
    accel = detect.accelerator(detect.chip_name())
    return {
        "platform": _PLATFORMS.get(platform.system(), "unknown"),
        "accel": _ACCELS.get(accel.kind, "unknown"),
        "containerized": _containerized(cgroup_binds),
        "virtualization_layer": _virtualization_layer(),
        "wall_source": "cgroup" if cgroup_binds else "physical",
    }


def advertised_capabilities() -> list[dict]:
    """Advertise exact reusable targets, each bound to this durable node identity.

    Display-only/legacy characterization history is deliberately excluded: a coordinator may route
    governed work only to an artifact/runtime/config cell the node can independently re-derive.
    """
    with db.connected() as con:
        rows = db.list_reusable_characterizations(con, profile.machine_key())
    node_id = config.node_identity()
    result = []
    for row in rows:
        safe_context = row.get("safe_context")
        if (not isinstance(safe_context, int) or isinstance(safe_context, bool)
                or safe_context <= 0):
            continue
        cap = {
            "kind": "serve_model",
            "id": row["logical_model_id"],
            "engine": row["legacy_engine"],
            "evidence": "characterized",
            "runtime": row["runtime"],
            "backend": row["backend"],
            "artifact_id": row["artifact_id"],
            "config_key": row["config_key"],
            "safe_context": safe_context,
        }
        authority_payload = {"node_id": node_id, **cap}
        digest = hashlib.sha256(json.dumps(
            authority_payload, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        cap["authority"] = f"node-target:v1:{digest}"
        result.append(cap)
    return result


def require_execution_authority(kind: str, args: dict) -> None:
    """Refuse governed remote Ollama work unless this node still advertises the exact target.

    The bearer session binds the offer to this enrolled node. The authority fingerprint additionally
    binds model, runtime/backend, immutable artifact, and effective config to the node's current
    reusable evidence. Other engines retain their existing node behavior in this Ollama slice.
    """
    if kind not in {"run", "benchmark"} or args.get("engine") != "ollama":
        return
    authority = args.get("target_authority")
    model = args.get("model")
    if not isinstance(authority, str) or not isinstance(model, str) or not any(
        cap["engine"] == "ollama"
        and cap["id"] == model
        and cap["authority"] == authority
        for cap in advertised_capabilities()
    ):
        raise ValueError("Ollama work lacks current node-scoped runtime authority")


def self_description() -> dict:
    """This node's enrollment payload (schema: ``enroll.request.json``).

    ``machine_key`` is this node installation's durable unique coordinator identity; local
    characterized capability lookup remains keyed by ARA's hardware profile. ``environment`` is
    the container-honest label. Validates against the wire contract."""
    return {
        "machine_key": config.node_identity(),
        "identity": {
            "hostname": platform.node() or "unknown",
            "os": platform.system(),
            "arch": platform.machine() or "unknown",
        },
        "capabilities": advertised_capabilities(),
        "environment": environment(),
    }
