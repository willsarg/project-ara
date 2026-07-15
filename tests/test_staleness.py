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

    bare = staleness.artifact_identity("org/model")
    assert bare is not None and bare.startswith(f"hf:org/model@{rev_a}:")
    selected = staleness.artifact_identity("org/model:Model-Q4_K_M.gguf")
    assert selected.startswith(f"hf-gguf:org/model@{rev_a}:Model-Q4_K_M.gguf:")
    assert staleness.artifact_size_gb("org/model:Model-Q4_K_M.gguf") == 0.0

    _revision_cache(tmp_path, rev_b)
    assert staleness.artifact_identity("org/model") is None
    assert staleness.artifact_identity("org/model:Model-Q4_K_M.gguf") is None


def test_artifact_identity_honors_hugging_face_cache_environment(tmp_path, monkeypatch):
    revision = "a" * 40
    custom = tmp_path / "custom-hub"
    root = custom / "models--org--model"
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text(revision)
    snapshot = root / "snapshots" / revision
    snapshot.mkdir(parents=True)
    blob = root / "blobs" / ("b" * 40)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"weights")
    (snapshot / "model.safetensors").symlink_to(Path("../../blobs") / blob.name)

    monkeypatch.setenv("HF_HUB_CACHE", str(custom))
    assert staleness.artifact_identity("org/model").startswith(f"hf:org/model@{revision}:")

    monkeypatch.delenv("HF_HUB_CACHE")
    hf_home = tmp_path / "hf-home"
    (hf_home / "hub").mkdir(parents=True)
    (root).rename(hf_home / "hub" / root.name)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    assert staleness.artifact_identity("org/model").startswith(f"hf:org/model@{revision}:")

    monkeypatch.delenv("HF_HOME")
    xdg = tmp_path / "xdg"
    (xdg / "huggingface" / "hub").mkdir(parents=True)
    (hf_home / "hub" / root.name).rename(xdg / "huggingface" / "hub" / root.name)
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    assert staleness.artifact_identity("org/model").startswith(f"hf:org/model@{revision}:")


def test_bare_hf_identity_rejects_empty_snapshot_and_binds_weight_blob(tmp_path, monkeypatch):
    _point_hub_at(tmp_path, monkeypatch)
    revision = "a" * 40
    snapshot = _revision_cache(tmp_path, revision)
    assert staleness.artifact_identity("org/model") is None

    root = snapshot.parents[1]
    blob = root / "blobs" / ("d" * 40)
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"actual weights")
    (snapshot / "model.safetensors").symlink_to(blob)
    identity = staleness.artifact_identity("org/model")
    assert identity is not None
    assert "model.safetensors" in identity and "d" * 40 in identity
    assert staleness.pinned_model_ref("org/model", identity) == str(snapshot)


def test_pinned_model_ref_resolves_exact_and_bare_gguf_to_snapshot_file(
        tmp_path, monkeypatch):
    _point_hub_at(tmp_path, monkeypatch)
    revision = "a" * 40
    snapshot = _revision_cache(tmp_path, revision, filename="Model-Q4_K_M.gguf")
    exact = "org/model:Model-Q4_K_M.gguf"
    exact_identity = staleness.artifact_identity(exact)
    assert staleness.pinned_model_ref(exact, exact_identity) == \
        str(snapshot / "Model-Q4_K_M.gguf")
    bare_identity = staleness.artifact_identity("org/model")
    assert staleness.pinned_model_ref("org/model", bare_identity) == \
        str(snapshot / "Model-Q4_K_M.gguf")
    assert staleness.pinned_model_ref("org/model", "artifact:stale") is None


def test_pinned_model_ref_uses_authorized_revision_if_main_changes_during_pin(
        tmp_path, monkeypatch):
    _point_hub_at(tmp_path, monkeypatch)
    old_revision = "a" * 40
    old_snapshot = _revision_cache(
        tmp_path, old_revision, filename="Model-Q4_K_M.gguf")
    identity = staleness.artifact_identity("org/model")
    original_matches = staleness.artifact_matches

    def advance_after_match(model, expected):
        assert original_matches(model, expected) is True
        _revision_cache(tmp_path, "c" * 40, filename="Model-Q8_0.gguf")
        return True

    monkeypatch.setattr(staleness, "artifact_matches", advance_after_match)
    assert staleness.pinned_model_ref("org/model", identity) == \
        str(old_snapshot / "Model-Q4_K_M.gguf")


def test_pinned_model_ref_handles_multishard_transformer_and_filesystem_races(
        tmp_path, monkeypatch):
    _point_hub_at(tmp_path, monkeypatch)
    revision = "a" * 40
    snapshot = _revision_cache(tmp_path, revision, filename="model-00001.safetensors")
    root = snapshot.parents[1]
    second_blob = root / "blobs" / ("d" * 40)
    second_blob.write_bytes(b"second")
    (snapshot / "model-00002.safetensors").symlink_to(second_blob)
    identity = staleness.artifact_identity("org/model")
    assert staleness.pinned_model_ref("org/model", identity) == str(snapshot)

    local = tmp_path / "local.gguf"
    local.write_bytes(b"weights")
    monkeypatch.setattr(staleness, "artifact_matches", lambda *_a: True)
    original_resolve = Path.resolve

    def fail_local(path, *args, **kwargs):
        if path == local:
            raise OSError("disappeared")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_local)
    assert staleness.pinned_model_ref(str(local), "artifact") is None
    assert staleness.pinned_model_ref("org/not-cached", "artifact") is None
    assert staleness.pinned_model_ref("org/model:missing.gguf", "artifact") is None
    assert staleness._authorized_snapshot("org/model", "hf:other/model@bad:x") is None


def test_bare_hf_identity_refuses_unidentifiable_selected_weight(tmp_path, monkeypatch):
    _point_hub_at(tmp_path, monkeypatch)
    revision = "a" * 40
    _revision_cache(tmp_path, revision, filename="model.safetensors")
    monkeypatch.setattr(staleness, "_file_descriptor", lambda *_a: None)
    assert staleness.artifact_identity("org/model") is None


def test_bare_hf_identity_supports_direct_snapshot_files_on_windows_style_cache(
        tmp_path, monkeypatch):
    _point_hub_at(tmp_path, monkeypatch)
    revision = "a" * 40
    snapshot = _revision_cache(tmp_path, revision)
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"direct weights")
    identity = staleness.artifact_identity("org/model")
    assert identity is not None and "direct:" in identity
    assert staleness.pinned_model_ref("org/model", identity) == str(snapshot)


def test_bare_hf_identity_rejects_mixed_formats_and_noncanonical_blobs(
        tmp_path, monkeypatch):
    _point_hub_at(tmp_path, monkeypatch)
    revision = "a" * 40
    snapshot = _revision_cache(tmp_path, revision, filename="model.gguf")
    root = snapshot.parents[1]
    tensor_blob = root / "blobs" / ("d" * 40)
    tensor_blob.write_bytes(b"tensor")
    (snapshot / "model.safetensors").symlink_to(tensor_blob)
    assert staleness.artifact_identity("org/model") is None

    (snapshot / "model.gguf").unlink()
    (snapshot / "model.safetensors").unlink()
    direct = snapshot / "model.safetensors"
    direct.write_bytes(b"not a standard HF blob link")
    assert ":direct:" in staleness.artifact_identity("org/model")


def test_bare_hf_identity_tolerates_snapshot_walk_race(tmp_path, monkeypatch):
    _point_hub_at(tmp_path, monkeypatch)
    revision = "a" * 40
    snapshot = _revision_cache(tmp_path, revision, filename="model.safetensors")
    original_rglob = Path.rglob

    def fail_snapshot(path, pattern):
        if path == snapshot:
            raise OSError("snapshot disappeared")
        return original_rglob(path, pattern)

    monkeypatch.setattr(Path, "rglob", fail_snapshot)
    assert staleness.artifact_identity("org/model") is None


def test_artifact_identity_tracks_local_gguf_stat(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"one")
    first = staleness.artifact_identity(str(model))
    assert first and first.startswith("local-gguf:")
    assert staleness.artifact_size_gb(str(model)) == 0.0
    model.write_bytes(b"changed")
    assert staleness.artifact_identity(str(model)) != first


def test_artifact_matches_requires_exact_nonempty_authority(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"weights")
    artifact_id = staleness.artifact_identity(str(model))
    assert staleness.artifact_matches(str(model), artifact_id) is True
    assert staleness.artifact_matches(str(model), None) is False
    assert staleness.artifact_matches(str(model), "") is False
    model.write_bytes(b"different weights")
    assert staleness.artifact_matches(str(model), artifact_id) is False


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
