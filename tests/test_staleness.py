# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Behavior of ARA's engine-independent characterization-staleness check."""
from __future__ import annotations

import importlib
import os
from pathlib import Path

from ara import staleness


def _point_hub_at(home: Path, monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
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
    assert staleness.ceiling_is_stale("org/model", b"2026-01-01") is False


def _revision_cache(home: Path, revision: str, *, filename: str | None = None) -> Path:
    root = home / ".cache" / "huggingface" / "hub" / "models--org--model"
    (root / "refs").mkdir(parents=True, exist_ok=True)
    (root / "refs" / "main").write_text(revision)
    snapshot = root / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    if filename is not None:
        blob = root / "blobs" / ("b" * 40)
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"weights")
        (snapshot / filename).symlink_to(blob)
    return snapshot


def test_artifact_identity_tracks_hf_revision_and_exact_gguf(tmp_path, monkeypatch):
    _point_hub_at(tmp_path, monkeypatch)
    rev_a, rev_b = "a" * 40, "c" * 40
    _revision_cache(tmp_path, rev_a, filename="Model-Q4_K_M.gguf")

    assert staleness.artifact_identity("org/model") == f"hf:org/model@{rev_a}"
    selected = staleness.artifact_identity("org/model:Model-Q4_K_M.gguf")
    assert selected.startswith(f"hf-gguf:org/model@{rev_a}:Model-Q4_K_M.gguf:")
    assert staleness.artifact_size_gb("org/model:Model-Q4_K_M.gguf") == 0.0

    _revision_cache(tmp_path, rev_b)
    assert staleness.artifact_identity("org/model") == f"hf:org/model@{rev_b}"
    assert staleness.artifact_identity("org/model:Model-Q4_K_M.gguf") is None


def test_artifact_identity_honors_hugging_face_cache_environment(tmp_path, monkeypatch):
    revision = "a" * 40
    custom = tmp_path / "custom-hub"
    root = custom / "models--org--model"
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text(revision)
    (root / "snapshots" / revision).mkdir(parents=True)

    monkeypatch.setenv("HF_HUB_CACHE", str(custom))
    assert staleness.artifact_identity("org/model") == f"hf:org/model@{revision}"

    monkeypatch.delenv("HF_HUB_CACHE")
    hf_home = tmp_path / "hf-home"
    (hf_home / "hub").mkdir(parents=True)
    (root).rename(hf_home / "hub" / root.name)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    assert staleness.artifact_identity("org/model") == f"hf:org/model@{revision}"

    monkeypatch.delenv("HF_HOME")
    xdg = tmp_path / "xdg"
    (xdg / "huggingface" / "hub").mkdir(parents=True)
    (hf_home / "hub" / root.name).rename(xdg / "huggingface" / "hub" / root.name)
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    assert staleness.artifact_identity("org/model") == f"hf:org/model@{revision}"


def test_artifact_identity_tracks_local_gguf_stat(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"one")
    first = staleness.artifact_identity(str(model))
    assert first and first.startswith("local-gguf:")
    assert staleness.artifact_size_gb(str(model)) == 0.0
    model.write_bytes(b"changed")
    assert staleness.artifact_identity(str(model)) != first


def test_artifact_identity_rejects_unknown_or_malformed_cache(tmp_path, monkeypatch):
    _point_hub_at(tmp_path, monkeypatch)
    assert staleness.artifact_identity(123) is None
    assert staleness.artifact_size_gb(123) is None
    assert staleness.artifact_identity("org/model") is None
    root = tmp_path / ".cache" / "huggingface" / "hub" / "models--org--model"
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text("not-a-revision")
    assert staleness.artifact_identity("org/model") is None
    assert staleness.artifact_size_gb("org/model:missing.gguf") is None
    (root / "refs" / "main").write_text("a" * 40)
    assert staleness.artifact_identity("org/model") is None
    assert staleness.artifact_size_gb("org/model") is None


def test_artifact_helpers_tolerate_filesystem_races(tmp_path, monkeypatch):
    model = tmp_path / "race.gguf"
    model.write_bytes(b"weights")
    original_stat = Path.stat
    calls = 0

    def fail_second_stat(path, *args, **kwargs):
        nonlocal calls
        if path == model:
            calls += 1
            if calls == 2:
                raise OSError("disappeared")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_second_stat)
    assert staleness.artifact_identity(str(model)) is None

    calls = 0
    assert staleness.artifact_size_gb(str(model)) is None


def test_hf_gguf_identity_tolerates_resolve_race(tmp_path, monkeypatch):
    _point_hub_at(tmp_path, monkeypatch)
    revision = "a" * 40
    snapshot = _revision_cache(tmp_path, revision, filename="m.gguf")
    selected = snapshot / "m.gguf"
    original_resolve = Path.resolve

    def fail_selected(path, *args, **kwargs):
        if path == selected:
            raise OSError("disappeared")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_selected)
    assert staleness.artifact_identity("org/model:m.gguf") is None


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
    home_var = "USERPROFILE" if os.name == "nt" else "HOME"
    artifact = _cached_artifact(imported_home)
    os.utime(artifact, (1_700_000_002, 1_700_000_002))

    try:
        with monkeypatch.context() as context:
            context.setenv(home_var, str(imported_home))
            importlib.reload(staleness)
            context.setenv(home_var, str(changed_home))

            assert staleness.ceiling_is_stale(
                "org/model", "2023-11-14T22:13:20+00:00") is True
    finally:
        importlib.reload(staleness)
