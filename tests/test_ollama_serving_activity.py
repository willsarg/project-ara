# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Persistent, corroborated ownership records for governed Ollama serving."""
from __future__ import annotations

import json
import threading

import pytest

from ara import activity, cli


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
    }
    fields.update(over)
    return activity.record_ollama_serving(**fields)


def _live(name="org-model-ara:latest", context=4096):
    return [{"name": name, "context_length": context, "size": 10, "size_vram": 10}]


def _wire_live(monkeypatch, entries=None, endpoint="http://127.0.0.1:11434"):
    monkeypatch.setattr("ara.ollama.base_url", lambda: endpoint)
    monkeypatch.setattr("ara.ollama.ps", lambda: _live() if entries is None else entries)


def test_manifest_is_atomic_minimal_and_same_identity_updates_in_place(
        registry, monkeypatch):
    replacements = []
    real_replace = activity.os.replace

    def replace(source, target):
        replacements.append((source, target))
        real_replace(source, target)

    monkeypatch.setattr(activity.os, "replace", replace)
    first = _record()
    second = _record(context=8192, started_at=200.0)

    assert first == second
    assert first.parent == registry / "serving"
    assert list(first.parent.glob("*.json")) == [first]
    assert not list(first.parent.glob("*.tmp"))
    assert json.loads(first.read_text()) == {
        "context": 8192,
        "endpoint": "http://127.0.0.1:11434",
        "model": "org/model",
        "runtime": "ollama",
        "served_name": "org-model-ara",
        "started_at": 200.0,
    }
    assert len(replacements) == 2
    assert all(source.parent == first.parent and target == first for source, target in replacements)


def test_multiple_served_identities_have_deterministic_distinct_manifests(registry):
    paths = {
        _record(served_name="a-ara", model="org/a"),
        _record(served_name="b-ara", model="org/b"),
    }
    assert len(paths) == 2
    assert {path.parent for path in paths} == {registry / "serving"}
    assert len(list((registry / "serving").glob("*.json"))) == 2


@pytest.mark.parametrize("field,value", [
    ("served_name", "bad\nname"),
    ("model", "bad\x00model"),
    ("context", 0),
    ("context", True),
    ("endpoint", ""),
    ("started_at", float("nan")),
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


def test_snapshot_correlates_exact_latest_name_context_and_endpoint(registry, monkeypatch):
    _record()
    _wire_live(monkeypatch)
    assert activity.snapshot() == [activity.Activity(
        kind="serving", model="org/model", pid=None, started_at=100.0,
        runtime="ollama", served_name="org-model-ara", context=4096,
        endpoint="http://127.0.0.1:11434")]


def test_snapshot_scans_past_malformed_and_wrong_matching_rows_to_later_valid_row(
        registry, monkeypatch):
    _record()
    _wire_live(monkeypatch, [
        {"name": "org-model-ara", "context_length": True},
        {"name": "org-model-ara:latest", "context_length": "4096"},
        {"name": "org-model-ara", "context_length": 2048},
        {"name": "org-model-ara:latest", "context_length": 4096},
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
        {"name": "a-ara:latest", "context_length": 4096},
        {"name": "b-ara", "context_length": 4096},
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

    def replace(source, target):
        try:
            barrier.wait(timeout=0.5)
        except threading.BrokenBarrierError:
            pass
        real_replace(source, target)

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
