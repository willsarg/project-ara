# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Persistent, corroborated ownership records for governed Ollama serving."""
from __future__ import annotations

import json
import os
import threading

import pytest

from ara import activity, cli


_BASE_ARTIFACT = "ollama-manifest-sha256:" + "a" * 64
_SERVED_ARTIFACT = "ollama-manifest-sha256:" + "b" * 64
_POLICY = "ollama-derived-v2"
_AUTHORITY = {
    "runtime_version": "0.30.10",
    "server_instance_id": "42:1000.000000:ollama",
    "configured_inputs": {"OLLAMA_KEEP_ALIVE": "2m"},
    "configured_num_parallel": 1,
    "configured_num_parallel_authority": "exact_version_default",
}


@pytest.fixture
def registry(tmp_path, monkeypatch):
    path = tmp_path / "activity"
    monkeypatch.setenv("ARA_ACTIVITY_DIR", str(path))
    return path


def _record(**over):
    fields = {
        "served_name": "org-model-ara",
        "model": "org/model",
        "context": 4096,
        "endpoint": "http://127.0.0.1:11434",
        "started_at": 100.0,
        "base_artifact_id": _BASE_ARTIFACT,
        "served_artifact_id": _SERVED_ARTIFACT,
    }
    fields.update(over)
    return activity.record_ollama_serving(**fields)


def _record_v2(**over):
    return _record(policy_version=_POLICY, runtime_authority=_AUTHORITY, **over)


def _live(name="org-model-ara:latest", context=4096):
    return [{"name": name, "context_length": context, "size": 10, "size_vram": 10,
             "digest": "b" * 64}]


def _wire_live(monkeypatch, entries=None, endpoint="http://127.0.0.1:11434"):
    monkeypatch.setattr("ara.ollama.base_url", lambda: endpoint)
    monkeypatch.setattr("ara.ollama.manifest_digest", lambda _name: "a" * 64)
    monkeypatch.setattr("ara.ollama.ps", lambda: _live() if entries is None else entries)


def test_manifest_is_atomic_minimal_and_same_identity_updates_in_place(
        registry, monkeypatch):
    replacements = []
    real_replace = activity.os.replace

    def replace(source, target, **kwargs):
        replacements.append((source, target, kwargs))
        real_replace(source, target, **kwargs)

    monkeypatch.setattr(activity.os, "replace", replace)
    first = _record()
    second = _record(context=8192, started_at=200.0)

    assert first == second
    assert first.parent == registry / "serving"
    assert list(first.parent.glob("*.json")) == [first]
    assert not list(first.parent.glob("*.tmp"))
    assert json.loads(first.read_text()) == {
        "base_artifact_id": _BASE_ARTIFACT,
        "context": 8192,
        "endpoint": "http://127.0.0.1:11434",
        "model": "org/model",
        "runtime": "ollama",
        "served_name": "org-model-ara",
        "served_artifact_id": _SERVED_ARTIFACT,
        "started_at": 200.0,
    }
    assert len(replacements) == 2
    assert all(str(source).endswith(".tmp") for source, _target, _options in replacements)
    if os.name == "nt":
        assert all(target == first and options == {}
                   for _source, target, options in replacements)
    else:
        assert all(target == first.name
                   and options["src_dir_fd"] == options["dst_dir_fd"]
                   for _source, target, options in replacements)


def test_multiple_served_identities_have_deterministic_distinct_manifests(registry):
    paths = {
        _record(served_name="a-ara", model="org/a"),
        _record(served_name="b-ara", model="org/b"),
    }
    assert len(paths) == 2
    assert {path.parent for path in paths} == {registry / "serving"}
    assert len(list((registry / "serving").glob("*.json"))) == 2


def test_ownership_view_reads_cold_v2_claim_without_contacting_ollama(
        registry, monkeypatch):
    manifest = _record_v2()
    monkeypatch.setattr("ara.ollama.ps", lambda: pytest.fail("ownership view contacted Ollama"))
    found = activity.ollama_ownership()
    assert found == [activity.Activity(
        kind="serving", model="org/model", pid=None, started_at=100.0,
        runtime="ollama", served_name="org-model-ara", context=4096,
        endpoint="http://127.0.0.1:11434", base_artifact_id=_BASE_ARTIFACT,
        served_artifact_id=_SERVED_ARTIFACT, policy_version=_POLICY,
        runtime_authority=_AUTHORITY)]
    assert json.loads(manifest.read_text())["runtime_authority"] == _AUTHORITY


def test_v2_live_status_requires_exact_runtime_authority(registry, monkeypatch):
    _record_v2()
    _wire_live(monkeypatch)
    monkeypatch.setattr(
        "ara.ollama.endpoint_authority",
        lambda url: cli.ollama.OllamaEndpoint(url, "loopback"),
    )
    monkeypatch.setattr(
        "ara.ollama.runtime_authority",
        lambda endpoint: cli.ollama.OllamaRuntimeAuthority(
            endpoint=endpoint,
            server_version=_AUTHORITY["runtime_version"],
            server_instance_id=_AUTHORITY["server_instance_id"],
            configured_inputs=tuple(_AUTHORITY["configured_inputs"].items()),
            configured_num_parallel=1,
            configured_num_parallel_authority="exact_version_default",
        ),
    )
    assert len(activity.snapshot()) == 1

    monkeypatch.setattr(
        "ara.ollama.runtime_authority",
        lambda endpoint: cli.ollama.OllamaRuntimeAuthority(
            endpoint=endpoint, server_version="0.30.10",
            server_instance_id="different", issue=None,
            configured_num_parallel=1,
            configured_num_parallel_authority="exact_version_default",
        ),
    )
    assert activity.snapshot() == []


def test_ownership_view_closes_root_when_serving_scan_raises(registry, monkeypatch):
    registry.mkdir()
    failure = KeyboardInterrupt()
    closed = []
    monkeypatch.setattr(
        activity, "_read_serving_records",
        lambda _root: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(
        activity, "_close_guards",
        lambda guards, original=None: closed.append((guards, original)),
    )
    with pytest.raises(KeyboardInterrupt):
        activity.ollama_ownership()
    assert len(closed) == 1 and closed[0][1] is failure


@pytest.mark.parametrize("field,value", [
    ("served_name", "bad\nname"),
    ("model", "bad\x00model"),
    ("context", 0),
    ("context", True),
    ("endpoint", ""),
    ("started_at", float("nan")),
    ("base_artifact_id", None),
    ("served_artifact_id", "sha256:" + "b" * 64),
])
def test_manifest_writer_rejects_invalid_schema_without_writing(registry, field, value):
    with pytest.raises(ValueError):
        _record(**{field: value})
    assert not registry.exists()


def test_snapshot_skips_ollama_when_no_valid_manifests(registry, monkeypatch):
    serving = registry / "serving"
    serving.mkdir(parents=True)
    (serving / "broken.json").write_text("{")
    (serving / "extra.json").write_text(json.dumps({
        "runtime": "ollama", "served_name": "x", "model": "org/x", "context": 1,
        "endpoint": "http://127.0.0.1:11434", "started_at": 1, "prompt": "secret"}))
    monkeypatch.setattr("ara.ollama.ps", lambda: pytest.fail("status called Ollama"))
    assert activity.snapshot() == []


def test_snapshot_preserves_live_legacy_ara_service_visibility(registry, monkeypatch):
    serving = registry / "serving"
    serving.mkdir(parents=True)
    (serving / "legacy.json").write_text(json.dumps({
        "runtime": "ollama", "served_name": "org-model-ara", "model": "org/model",
        "context": 4096, "endpoint": "http://127.0.0.1:11434", "started_at": 100.0,
    }))
    _wire_live(monkeypatch)
    found = activity.snapshot()
    assert len(found) == 1
    assert found[0].model == "org/model" and found[0].runtime == "ollama"
    assert found[0].base_artifact_id is None and found[0].served_artifact_id is None


def test_snapshot_correlates_exact_latest_name_context_and_endpoint(registry, monkeypatch):
    _record()
    _wire_live(monkeypatch)
    assert activity.snapshot() == [activity.Activity(
        kind="serving", model="org/model", pid=None, started_at=100.0,
        runtime="ollama", served_name="org-model-ara", context=4096,
        endpoint="http://127.0.0.1:11434", base_artifact_id=_BASE_ARTIFACT,
        served_artifact_id=_SERVED_ARTIFACT)]


def test_snapshot_scans_past_malformed_and_wrong_matching_rows_to_later_valid_row(
        registry, monkeypatch):
    _record()
    _wire_live(monkeypatch, [
        {"name": "org-model-ara", "context_length": True},
        {"name": "org-model-ara:latest", "context_length": "4096"},
        {"name": "org-model-ara", "context_length": 2048},
        {"name": "org-model-ara:latest", "context_length": 4096, "digest": "b" * 64},
    ])
    assert [(item.kind, item.model) for item in activity.snapshot()] == [
        ("serving", "org/model")]


@pytest.mark.parametrize("entries,endpoint", [
    (None, "http://127.0.0.1:11434"),
    ([], "http://127.0.0.1:11434"),
    (_live(name="unrelated:latest"), "http://127.0.0.1:11434"),
    (_live(name="org-model-ara:other"), "http://127.0.0.1:11434"),
    (_live(context=2048), "http://127.0.0.1:11434"),
    (_live(), "http://other-host:11434"),
])
def test_snapshot_suppresses_every_uncorroborated_manifest(
        registry, monkeypatch, entries, endpoint):
    _record()
    monkeypatch.setattr("ara.ollama.base_url", lambda: endpoint)
    monkeypatch.setattr("ara.ollama.manifest_digest", lambda _name: "a" * 64)
    monkeypatch.setattr("ara.ollama.ps", lambda: entries)
    assert activity.snapshot() == []


@pytest.mark.parametrize("name", [None, [], {}, 7])
@pytest.mark.parametrize("context", [4096, True, "4096", 0, -1, None])
def test_snapshot_suppresses_malformed_loaded_name_and_context_without_crashing(
        registry, monkeypatch, name, context):
    _record()
    _wire_live(monkeypatch, [{"name": name, "context_length": context}])
    assert activity.snapshot() == []


def test_unrelated_live_models_never_appear(registry, monkeypatch):
    _record()
    _wire_live(monkeypatch, [
        {"name": "someone-else:latest", "context_length": 99999},
        *_live(),
    ])
    assert [(item.model, item.served_name) for item in activity.snapshot()] == [
        ("org/model", "org-model-ara")]


def test_non_ollama_ephemeral_activity_survives_unreachable_ollama(registry, monkeypatch):
    _record()
    _wire_live(monkeypatch, entries=None)
    monkeypatch.setattr("ara.ollama.ps", lambda: None)
    with activity.track("running", "org/local"):
        assert [(item.kind, item.model) for item in activity.snapshot()] == [
            ("running", "org/local")]


def test_snapshot_observation_never_mutates_persistent_manifests(registry, monkeypatch):
    manifest = _record()
    before = manifest.read_bytes()
    _wire_live(monkeypatch, _live(context=1))
    monkeypatch.setattr(activity.os, "replace", lambda *_a: pytest.fail("replace called"))
    monkeypatch.setattr(activity.os, "unlink", lambda *_a: pytest.fail("unlink called"))
    assert activity.snapshot() == []
    assert manifest.read_bytes() == before


def test_snapshot_merges_and_sorts_ephemeral_with_multiple_persistent_states(
        registry, monkeypatch):
    _record(served_name="b-ara", model="org/b", started_at=20.0)
    _record(served_name="a-ara", model="org/a", started_at=10.0)
    _wire_live(monkeypatch, [
        {"name": "a-ara:latest", "context_length": 4096, "digest": "b" * 64},
        {"name": "b-ara", "context_length": 4096, "digest": "b" * 64},
    ])
    monkeypatch.setattr(activity.time, "time", lambda: 15.0)
    with activity.track("running", "org/run"):
        assert [(item.kind, item.model) for item in activity.snapshot()] == [
            ("serving", "org/a"), ("running", "org/run"), ("serving", "org/b")]


def test_status_text_and_json_expose_persistent_serving_without_pid(
        registry, monkeypatch, make_console, capsys):
    _record()
    _wire_live(monkeypatch)
    c, buf = make_console()
    cli.render_status(c)
    assert buf.getvalue() == "ARA is serving org/model.\n"

    cli.render_status(c, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"state": "serving", "activities": [{
        "kind": "serving", "model": "org/model", "started_at": 100.0,
        "runtime": "ollama", "served_name": "org-model-ara", "context": 4096,
        "endpoint": "http://127.0.0.1:11434",
    }]}


def test_serving_child_symlink_is_never_read_or_written(registry, monkeypatch, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    serving = registry / "serving"
    registry.mkdir()
    try:
        serving.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    outside_record = outside / "owned.json"
    outside_record.write_text(json.dumps({
        "runtime": "ollama", "served_name": "org-model-ara", "model": "org/model",
        "context": 4096, "endpoint": "http://127.0.0.1:11434", "started_at": 1.0,
    }))
    monkeypatch.setattr("ara.ollama.ps", lambda: pytest.fail("followed serving symlink"))
    assert activity.snapshot() == []

    outside_record.unlink()
    with pytest.raises(OSError):
        _record()
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor assertion")
def test_serving_child_open_failure_closes_held_root_descriptor(
        registry, monkeypatch, tmp_path):
    outside = tmp_path / "descriptor-outside"
    outside.mkdir()
    registry.mkdir()
    (registry / "serving").symlink_to(outside, target_is_directory=True)
    root_descriptors = []
    closed = []
    real_open = activity.os.open
    real_close = activity.os.close

    def open_file(path, *args, **kwargs):
        descriptor = real_open(path, *args, **kwargs)
        if path == registry:
            root_descriptors.append(descriptor)
        return descriptor

    def close_file(descriptor):
        closed.append(descriptor)
        return real_close(descriptor)

    monkeypatch.setattr(activity.os, "open", open_file)
    monkeypatch.setattr(activity.os, "close", close_file)
    with pytest.raises(OSError):
        _record()
    assert root_descriptors and root_descriptors == closed


def test_root_symlink_refuses_persistent_manifest_write(registry, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        registry.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    with pytest.raises(OSError):
        _record()
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="directory-fd race coverage is POSIX-specific")
def test_manifest_serving_swap_writes_original_child(registry, monkeypatch, tmp_path):
    serving = registry / "serving"
    serving.mkdir(parents=True)
    replacement = registry / "replacement"
    replacement.mkdir()
    displaced = registry / "displaced"
    real_replace = activity.os.replace
    swapped = False

    def swapping_replace(source, target, *args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            serving.rename(displaced)
            replacement.rename(serving)
        real_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(activity.os, "replace", swapping_replace)
    path = _record()
    assert path.name in {item.name for item in displaced.glob("*.json")}
    assert list(serving.glob("*.json")) == []


def test_stale_atomic_temp_never_wedges_future_manifest_update(registry, monkeypatch):
    path = _record()
    path.unlink()
    stale = path.with_name(f".{path.stem}.tmp")
    stale.write_text("SIGKILL residue")

    assert _record(context=8192) == path
    assert json.loads(path.read_text())["context"] == 8192
    assert stale.read_text() == "SIGKILL residue"
    _wire_live(monkeypatch, [])
    assert all(item.name != stale.name for item in activity.snapshot())


def test_concurrent_same_identity_manifest_writers_both_complete_atomically(
        registry, monkeypatch):
    barrier = threading.Barrier(2)
    real_replace = activity.os.replace

    def replace(source, target, **kwargs):
        try:
            barrier.wait(timeout=0.5)
        except threading.BrokenBarrierError:
            pass
        real_replace(source, target, **kwargs)

    monkeypatch.setattr(activity.os, "replace", replace)
    errors = []

    def write(context):
        try:
            _record(context=context, started_at=float(context))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(context,)) for context in (4096, 8192)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    files = list((registry / "serving").glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["context"] in {4096, 8192}
