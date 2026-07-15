# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Contracts for the Docker-backed ``ara hub`` coordinator server."""
from __future__ import annotations

import subprocess
import types
from pathlib import Path

import pytest

from ara import hub_server


def _proc(returncode: int = 0, *, stderr: str = ""):
    return types.SimpleNamespace(returncode=returncode, stderr=stderr)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "coordinator"
    source.mkdir(parents=True)
    for name in (
        "Dockerfile", "compose.yaml", "next.config.ts", "package.json",
        "package-lock.json", "tsconfig.json",
    ):
        (source / name).write_text(name, encoding="utf-8")
    (source / "src").mkdir()
    (source / "public").mkdir()
    return source


def test_coordinator_source_prefers_packaged_exact_build_context(monkeypatch, tmp_path):
    packaged = _source(tmp_path / "packaged")
    development = _source(tmp_path / "development")
    monkeypatch.setattr(hub_server, "_PACKAGED_SOURCE", packaged)
    monkeypatch.setattr(hub_server, "_DEVELOPMENT_SOURCE", development)

    assert hub_server.coordinator_source() == packaged


def test_coordinator_source_falls_back_to_checkout_during_development(monkeypatch, tmp_path):
    packaged = tmp_path / "missing"
    development = _source(tmp_path / "development")
    monkeypatch.setattr(hub_server, "_PACKAGED_SOURCE", packaged)
    monkeypatch.setattr(hub_server, "_DEVELOPMENT_SOURCE", development)

    assert hub_server.coordinator_source() == development


def test_coordinator_source_refuses_an_incomplete_distribution(monkeypatch, tmp_path):
    monkeypatch.setattr(hub_server, "_PACKAGED_SOURCE", tmp_path / "missing-packaged")
    monkeypatch.setattr(hub_server, "_DEVELOPMENT_SOURCE", tmp_path / "missing-development")

    with pytest.raises(hub_server.HubError, match="coordinator build context is missing"):
        hub_server.coordinator_source()


def test_default_data_dir_is_ara_owned(monkeypatch, tmp_path):
    monkeypatch.setattr(hub_server.platformdirs, "user_data_path", lambda app: tmp_path / app)

    assert hub_server.default_data_dir() == tmp_path / "ara" / "hub"


def test_run_builds_and_attaches_compose_with_host_persistence(monkeypatch, tmp_path):
    source = _source(tmp_path)
    data_dir = tmp_path / "persistent" / "hub-data"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return _proc()

    monkeypatch.setattr(hub_server, "coordinator_source", lambda: source)
    monkeypatch.setattr(hub_server.subprocess, "run", fake_run)

    assert hub_server.run(
        bind="127.0.0.1", port=4312, data_dir=data_dir,
        version="0.0.9.dev1+gabc", rebuild=False,
    ) == 0

    assert data_dir.is_dir()
    assert calls[0][0] == ["docker", "info"]
    command, kwargs = calls[1]
    assert command == [
        "docker", "compose", "--project-name", "ara-hub",
        "-f", str(source / "compose.yaml"),
        "up", "--build", "--remove-orphans",
    ]
    assert kwargs["cwd"] == source
    assert kwargs["env"]["ARA_HUB_DATA_DIR"] == str(data_dir.resolve())
    assert kwargs["env"]["ARA_COORDINATOR_BIND"] == "127.0.0.1"
    assert kwargs["env"]["ARA_COORDINATOR_PORT"] == "4312"
    assert kwargs["env"]["ARA_HUB_IMAGE"] == "ara-hub:0.0.9.dev1-gabc"


def test_run_rebuilds_without_cache_before_attaching(monkeypatch, tmp_path):
    source = _source(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return _proc()

    monkeypatch.setattr(hub_server, "coordinator_source", lambda: source)
    monkeypatch.setattr(hub_server.subprocess, "run", fake_run)

    assert hub_server.run(
        bind="0.0.0.0", port=3000, data_dir=tmp_path / "data",
        version="1.2.3", rebuild=True,
    ) == 0

    assert calls[1][-3:] == ["build", "--no-cache", "coordinator"]
    assert calls[2][-3:] == ["up", "--build", "--remove-orphans"]


def test_run_stops_when_no_cache_build_fails(monkeypatch, tmp_path):
    source = _source(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return _proc(7 if "build" in command else 0)

    monkeypatch.setattr(hub_server, "coordinator_source", lambda: source)
    monkeypatch.setattr(hub_server.subprocess, "run", fake_run)

    assert hub_server.run(
        bind="127.0.0.1", port=3000, data_dir=tmp_path / "data",
        version="1.2.3", rebuild=True,
    ) == 7
    assert len(calls) == 2


def test_run_reports_missing_docker(monkeypatch, tmp_path):
    monkeypatch.setattr(hub_server, "coordinator_source", lambda: _source(tmp_path))
    monkeypatch.setattr(
        hub_server.subprocess, "run",
        lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError("docker")),
    )

    with pytest.raises(hub_server.HubError, match="Docker is required"):
        hub_server.run(
            bind="127.0.0.1", port=3000, data_dir=tmp_path / "data",
            version="1.2.3", rebuild=False,
        )


def test_run_reports_unavailable_docker_daemon(monkeypatch, tmp_path):
    monkeypatch.setattr(hub_server, "coordinator_source", lambda: _source(tmp_path))
    monkeypatch.setattr(hub_server.subprocess, "run", lambda *_a, **_k: _proc(1, stderr="no daemon"))

    with pytest.raises(hub_server.HubError, match="Docker daemon is unavailable: no daemon"):
        hub_server.run(
            bind="127.0.0.1", port=3000, data_dir=tmp_path / "data",
            version="1.2.3", rebuild=False,
        )


def test_run_reports_data_directory_creation_failure(monkeypatch, tmp_path):
    source = _source(tmp_path)
    data_path = tmp_path / "not-a-directory"
    data_path.write_text("occupied", encoding="utf-8")
    monkeypatch.setattr(hub_server, "coordinator_source", lambda: source)

    with pytest.raises(hub_server.HubError, match="cannot create hub data directory"):
        hub_server.run(
            bind="127.0.0.1", port=3000, data_dir=data_path,
            version="1.2.3", rebuild=False,
        )


def test_run_surfaces_compose_exit_status(monkeypatch, tmp_path):
    source = _source(tmp_path)
    results = iter([_proc(), _proc(23)])
    monkeypatch.setattr(hub_server, "coordinator_source", lambda: source)
    monkeypatch.setattr(hub_server.subprocess, "run", lambda *_a, **_k: next(results))

    assert hub_server.run(
        bind="127.0.0.1", port=3000, data_dir=tmp_path / "data",
        version="1.2.3", rebuild=False,
    ) == 23


def test_run_treats_operator_interrupt_as_a_clean_stop(monkeypatch, tmp_path):
    source = _source(tmp_path)
    results = iter([_proc(), KeyboardInterrupt()])
    monkeypatch.setattr(hub_server, "coordinator_source", lambda: source)

    def fake_run(*_args, **_kwargs):
        result = next(results)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(hub_server.subprocess, "run", fake_run)

    assert hub_server.run(
        bind="127.0.0.1", port=3000, data_dir=tmp_path / "data",
        version="1.2.3", rebuild=False,
    ) == 0


def test_run_reports_timeout_while_checking_docker(monkeypatch, tmp_path):
    monkeypatch.setattr(hub_server, "coordinator_source", lambda: _source(tmp_path))
    monkeypatch.setattr(
        hub_server.subprocess, "run",
        lambda *_a, **_k: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(["docker", "info"], 15)
        ),
    )

    with pytest.raises(hub_server.HubError, match="Docker daemon check timed out"):
        hub_server.run(
            bind="127.0.0.1", port=3000, data_dir=tmp_path / "data",
            version="1.2.3", rebuild=False,
        )
