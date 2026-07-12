# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Behavior of ARA's engine-independent characterization-staleness check."""
from __future__ import annotations

import importlib
import os
from pathlib import Path

from ara import staleness


def _point_hub_at(home: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(
        staleness, "_HUB", home / ".cache" / "huggingface" / "hub", raising=False)


def _cached_artifact(home: Path, model_id: str = "org/model") -> Path:
    artifact = (home / ".cache" / "huggingface" / "hub"
                / f"models--{model_id.replace('/', '--')}" / "snapshots" / "revision" / "weights")
    artifact.parent.mkdir(parents=True)
    artifact.touch()
    return artifact


def test_ceiling_is_not_stale_without_measurement_timestamp(tmp_path, monkeypatch):
    _point_hub_at(tmp_path, monkeypatch)
    artifact = _cached_artifact(tmp_path)
    os.utime(artifact, (2_000_000_000, 2_000_000_000))

    assert staleness.ceiling_is_stale("org/model", None) is False


def test_ceiling_is_not_stale_when_model_is_not_cached(tmp_path, monkeypatch):
    _point_hub_at(tmp_path, monkeypatch)

    assert staleness.ceiling_is_stale("org/model", "2026-01-01T00:00:00+00:00") is False


def test_ceiling_is_not_stale_when_cache_is_within_timestamp_tolerance(tmp_path, monkeypatch):
    _point_hub_at(tmp_path, monkeypatch)
    artifact = _cached_artifact(tmp_path)
    os.utime(artifact, (1_700_000_001, 1_700_000_001))

    assert staleness.ceiling_is_stale("org/model", "2023-11-14T22:13:20+00:00") is False


def test_ceiling_is_not_stale_when_cache_is_older(tmp_path, monkeypatch):
    _point_hub_at(tmp_path, monkeypatch)
    artifact = _cached_artifact(tmp_path)
    os.utime(artifact, (1_699_999_999, 1_699_999_999))

    assert staleness.ceiling_is_stale("org/model", "2023-11-14T22:13:20+00:00") is False


def test_ceiling_is_stale_when_cache_is_more_than_one_second_newer(tmp_path, monkeypatch):
    _point_hub_at(tmp_path, monkeypatch)
    artifact = _cached_artifact(tmp_path)
    os.utime(artifact, (1_700_000_002, 1_700_000_002))

    assert staleness.ceiling_is_stale("org/model", "2023-11-14T22:13:20") is True


def test_ceiling_is_not_stale_for_malformed_timestamp(tmp_path, monkeypatch):
    _point_hub_at(tmp_path, monkeypatch)
    _cached_artifact(tmp_path)

    assert staleness.ceiling_is_stale("org/model", "not-a-timestamp") is False


def test_ceiling_ignores_artifact_filesystem_errors(tmp_path, monkeypatch):
    _point_hub_at(tmp_path, monkeypatch)
    artifact = _cached_artifact(tmp_path)
    original_lstat = Path.lstat

    def failing_lstat(path):
        if path == artifact:
            raise OSError("artifact disappeared")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", failing_lstat)

    assert staleness.ceiling_is_stale("org/model", "2020-01-01T00:00:00+00:00") is False


def test_hub_root_is_captured_when_module_is_imported(tmp_path, monkeypatch):
    imported_home = tmp_path / "imported-home"
    changed_home = tmp_path / "changed-home"
    artifact = _cached_artifact(imported_home)
    os.utime(artifact, (1_700_000_002, 1_700_000_002))

    try:
        with monkeypatch.context() as context:
            context.setenv("HOME", str(imported_home))
            importlib.reload(staleness)
            context.setenv("HOME", str(changed_home))

            assert staleness.ceiling_is_stale(
                "org/model", "2023-11-14T22:13:20+00:00") is True
    finally:
        importlib.reload(staleness)
