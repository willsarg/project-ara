# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Docker-backed lifecycle for the ARA fleet coordinator."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import platformdirs


_REQUIRED_FILES = (
    "Dockerfile", "compose.yaml", "next.config.ts", "package.json", "package-lock.json",
    "tsconfig.json",
)
_REQUIRED_DIRECTORIES = ("src", "public")
_PACKAGED_SOURCE = Path(__file__).with_name("_hub_source")
_DEVELOPMENT_SOURCE = Path(__file__).resolve().parent.parent / "coordinator"


class HubError(RuntimeError):
    """A failure that prevents the local coordinator from starting."""


def _complete_source(path: Path) -> bool:
    return (all((path / name).is_file() for name in _REQUIRED_FILES)
            and all((path / name).is_dir() for name in _REQUIRED_DIRECTORIES))


def coordinator_source() -> Path:
    """Return the exact coordinator build context shipped with this ARA version."""
    for source in (_PACKAGED_SOURCE, _DEVELOPMENT_SOURCE):
        if _complete_source(source):
            return source
    raise HubError("ARA's coordinator build context is missing; reinstall project-ara")


def default_data_dir() -> Path:
    """Return the durable host directory owned by ``ara hub``."""
    return platformdirs.user_data_path("ara") / "hub"


def _image_tag(version: str) -> str:
    return "ara-hub:" + re.sub(r"[^A-Za-z0-9_.-]+", "-", version)


def run(*, bind: str, port: int, data_dir: Path, version: str, rebuild: bool) -> int:
    """Build and run the coordinator through Docker Compose in the foreground."""
    source = coordinator_source()
    data_dir = Path(data_dir)
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HubError(f"cannot create hub data directory {data_dir}: {exc}") from exc

    env = {
        **os.environ,
        "ARA_HUB_DATA_DIR": str(data_dir.resolve()),
        "ARA_COORDINATOR_BIND": bind,
        "ARA_COORDINATOR_PORT": str(port),
        "ARA_HUB_IMAGE": _image_tag(version),
    }
    try:
        probe = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=15,
        )
    except FileNotFoundError as exc:
        raise HubError("Docker is required to run `ara hub`; install Docker and try again") from exc
    except subprocess.TimeoutExpired as exc:
        raise HubError("Docker daemon check timed out; verify Docker is running") from exc

    if probe.returncode != 0:
        detail = probe.stderr.strip()
        raise HubError(f"Docker daemon is unavailable: {detail}")

    compose = [
        "docker", "compose", "--project-name", "ara-hub",
        "-f", str(source / "compose.yaml"),
    ]
    try:
        if rebuild:
            built = subprocess.run(
                [*compose, "build", "--no-cache", "coordinator"], cwd=source, env=env,
            )
            if built.returncode != 0:
                return int(built.returncode)

        running = subprocess.run(
            [*compose, "up", "--build", "--remove-orphans"], cwd=source, env=env,
        )
    except KeyboardInterrupt:
        # Compose receives the same SIGINT, stops its containers, and exits. The operator asked
        # for a normal foreground shutdown, so do not turn that into a Python traceback.
        return 0
    return int(running.returncode)
