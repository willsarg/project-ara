# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Lifecycle wiring tests for ARA-owned live activity."""
from __future__ import annotations

import contextlib
import json
import sys
import types

import pytest
from click.testing import CliRunner

from ara import activity, cli


@pytest.fixture
def activity_registry(tmp_path, monkeypatch):
    path = tmp_path / "activity"
    monkeypatch.setenv("ARA_ACTIVITY_DIR", str(path))
    return path


def _assert_activity(kind: str, model: str | None = None) -> None:
    found = activity.snapshot()
    assert [(item.kind, item.model) for item in found] == [(kind, model)]


def test_search_tracks_only_the_hub_request_and_never_the_query(
        make_console, monkeypatch, activity_registry):
    def search(query):
        assert query == "private search text"
        _assert_activity("searching")
        return []

    monkeypatch.setattr(cli.hub, "search", search)
    c, _ = make_console()
    assert cli.render_search(c, "private search text") == 0
    assert activity.snapshot() == []
    assert "private search text" not in "".join(
        path.read_text() for path in activity_registry.glob("**/*") if path.is_file())


@pytest.mark.parametrize("raised", [RuntimeError("boom"), KeyboardInterrupt(), SystemExit(9)])
def test_search_activity_cleans_up_on_every_exit(
        make_console, monkeypatch, activity_registry, raised):
    def search(_query):
        _assert_activity("searching")
        raise raised

    monkeypatch.setattr(cli.hub, "search", search)
    c, _ = make_console()
    with pytest.raises(type(raised)):
        cli.render_search(c, "secret")
    assert activity.snapshot() == []


def _wire_run(monkeypatch, generate):
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "machine")
    row = {"safe_context": 4096, "measured_at": None, "artifact_id": "artifact:test"}
    monkeypatch.setattr(cli.db, "get_characterization", lambda *_a: row)
    monkeypatch.setattr(
        cli.db, "get_reusable_characterization_for_engine", lambda *_a, **_k: row)
    monkeypatch.setattr(cli, "engine_status", lambda _backend: (True, "llama.cpp"))
    monkeypatch.setattr(cli, "get_backend", lambda _backend: types.SimpleNamespace(
        generate=generate))
    monkeypatch.setattr(
        cli, "_current_reuse_identity", lambda *_a: ("method:test", "engine:test", None))
    monkeypatch.setattr(cli.staleness, "artifact_matches", lambda *_a: True)
    monkeypatch.setattr(
        cli.staleness, "pinned_model_ref", lambda model, _artifact, **_kwargs: model)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: False))


def test_run_tracks_only_generate_and_cleans_after_refusal(
        make_console, monkeypatch, activity_registry):
    def generate(*_a, **_k):
        _assert_activity("running", "org/model")
        return {"context": 4096, "refused": True, "reason": "safe refusal"}

    _wire_run(monkeypatch, generate)
    c, _ = make_console()
    assert cli.render_run(c, "org/model", prompt="do not store me", engine="cpu") == 1
    assert activity.snapshot() == []


@pytest.mark.parametrize("raised", [RuntimeError("boom"), KeyboardInterrupt(), SystemExit(8)])
def test_run_activity_cleans_up_on_operational_and_base_exceptions(
        make_console, monkeypatch, activity_registry, raised):
    def generate(*_a, **_k):
        _assert_activity("running", "org/model")
        raise raised

    _wire_run(monkeypatch, generate)
    c, _ = make_console()
    if isinstance(raised, (RuntimeError, SystemExit)):
        assert cli.render_run(c, "org/model", prompt="secret", engine="cpu") == 1
    else:
        with pytest.raises(KeyboardInterrupt):
            cli.render_run(c, "org/model", prompt="secret", engine="cpu")
    assert activity.snapshot() == []


def test_run_gate_failure_never_creates_activity(make_console, monkeypatch, activity_registry):
    _wire_run(monkeypatch, lambda *_a, **_k: pytest.fail("generate called"))
    monkeypatch.setattr(cli.activity, "track", lambda *_a, **_k: pytest.fail("track called"))
    c, _ = make_console()
    assert cli.render_run(c, "org/model", prompt=None, engine="cpu") == 1
    assert activity.snapshot() == []


def _wire_characterize(monkeypatch, backend):
    descriptor = cli.methodology.characterization_descriptor(
        schedule=[512], repeats=1, reserve_policy="test", reserve_bytes=1024,
        worker_protocol="test:v1", sampling_interval_ms=50,
        telemetry_failure_policy="fail-closed", watchdog_stop_rule="test-stop")
    original_characterize = backend.characterize

    def evidenced_characterize(*args, **kwargs):
        result = dict(original_characterize(*args, **kwargs))
        if "error" not in result:
            result.update(
                direct_context=result.get("safe_context"), fitted_context=None,
                stopped_reason=None, methodology=descriptor,
                methodology_key=cli.methodology.key(descriptor))
        return result

    backend.characterize = evidenced_characterize
    monkeypatch.setattr(cli, "engine_status", lambda _backend: (True, "llama.cpp"))
    monkeypatch.setattr(cli, "get_backend", lambda _backend: backend)
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "machine")
    monkeypatch.setattr(cli.catalog, "remember", lambda *_a: None)
    monkeypatch.setattr(cli.staleness, "artifact_identity",
                        lambda _model, **_kwargs: "artifact:test")
    monkeypatch.setattr(
        cli.staleness, "pinned_model_ref", lambda model, _artifact, **_kwargs: model)
    monkeypatch.setattr(cli.engine_audit, "audit_engine", lambda *_args, **_kwargs: {
        "key": "cpu", "package_version": "0.3.34",
        "installation": {"status": "matched"},
        "build": {"status": "matched"}, "runtime": {"status": "matched"},
        "fingerprint": "engine:v1:sha256:test",
    })


@pytest.mark.parametrize("cached,expected_downloads", [(True, []), (False, ["org/model"])])
def test_prefetch_wrapper_preserves_cached_and_download_contract(
        make_console, monkeypatch, cached, expected_downloads):
    downloads = []
    backend = types.SimpleNamespace(
        calibration_model_cached=lambda _model: cached,
        download_calibration_model=lambda model, **_kwargs: downloads.append(model),
    )
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda _m: None)
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: None)
    c, _ = make_console()
    assert cli._prefetch_weights(
        c, "org/model", backend, "cpu", as_json=False, progress=False) is None
    assert downloads == expected_downloads


def test_prefetch_carries_one_immutable_plan_through_download(make_console, monkeypatch):
    plan = cli.acquire.AcquisitionPlan(
        "org/model", "org/model", "a" * 40, "model-q4.gguf", 4.0)
    seen = []
    backend = types.SimpleNamespace(
        calibration_model_cached=lambda _model: False,
        prepare_download=lambda model: seen.append(("prepare", model)) or plan,
        download_prepared_model=lambda received, **_kwargs: seen.append(("download", received)),
    )
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: 50.0)
    c, _ = make_console()

    assert cli._prefetch_weights(
        c, "org/model", backend, "cpu", as_json=False, progress=False) is None
    assert seen == [("prepare", "org/model"), ("download", plan)]


def test_prepared_plan_revision_is_carried_into_identity_and_pinning(monkeypatch):
    plan = cli.acquire.AcquisitionPlan(
        "org/model", "org/model", "a" * 40, None, 4.0)
    seen = []
    monkeypatch.setattr(cli.staleness, "artifact_identity",
                        lambda model, *, revision: seen.append(
                            ("identity", model, revision)) or "artifact")

    assert cli._artifact_identity_for_plan("org/model", plan) == "artifact"
    assert seen == [
        ("identity", "org/model", "a" * 40),
    ]

    monkeypatch.setattr(cli.staleness, "pinned_model_ref",
                        lambda model, artifact, *, revision: seen.append(
                            ("pin", model, artifact, revision)) or
                        "/cache/snapshots/exact")
    with cli._pinned_model_for_plan("org/model", "artifact", plan) as pinned:
        assert pinned == "/cache/snapshots/exact"
    assert seen[-1] == ("pin", "org/model", "artifact", "a" * 40)

    monkeypatch.setattr(cli.staleness, "artifact_identity", lambda _model: None)
    monkeypatch.setattr(cli.staleness, "artifact_matches", lambda _model, _expected: True)
    assert cli._artifact_identity_for_plan(
        "org/model", None, expected="artifact") == "artifact"


def test_prefetch_refuses_prepared_download_when_free_space_is_unknown(
        make_console, monkeypatch):
    plan = cli.acquire.AcquisitionPlan(
        "org/model", "org/model", "a" * 40, "model-q4.gguf", 4.0)
    backend = types.SimpleNamespace(
        calibration_model_cached=lambda _model: False,
        prepare_download=lambda _model: plan,
        download_prepared_model=lambda *_args, **_kwargs: pytest.fail("download called"),
    )
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: None)
    c, buf = make_console()

    assert cli._prefetch_weights(
        c, "org/model", backend, "cpu", as_json=False, progress=False) == 1
    assert "free disk space" in buf.getvalue()


def test_prefetch_enforces_buffer_for_zero_rounded_payload(make_console, monkeypatch):
    plan = cli.acquire.AcquisitionPlan(
        "org/model", "org/model", "a" * 40, "tiny.gguf", 0.0)
    backend = types.SimpleNamespace(
        calibration_model_cached=lambda _model: False,
        prepare_download=lambda _model: plan,
    )
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: 0.1)
    c, buf = make_console()
    assert cli._prefetch_weights(
        c, "org/model", backend, "cpu", as_json=False, progress=False) == 1
    assert "not enough disk" in buf.getvalue()


def test_prefetch_reserves_only_download_and_normal_headroom(
        make_console, monkeypatch):
    plan = cli.acquire.AcquisitionPlan(
        "org/model", "org/model", "a" * 40, None, 4.0)
    downloaded = []
    backend = types.SimpleNamespace(
        calibration_model_cached=lambda _model: False,
        prepare_download=lambda _model: plan,
        download_prepared_model=lambda received, **_k: downloaded.append(received),
    )
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: 7.0)
    c, _ = make_console()

    assert cli._prefetch_weights(
        c, "org/model", backend, "cpu", as_json=False, progress=False) is None
    assert downloaded == [plan]


def test_prefetch_skips_network_for_existing_authorized_snapshot(make_console, monkeypatch):
    backend = types.SimpleNamespace(
        calibration_model_cached=lambda _model: False,
        prepare_download=lambda _model: pytest.fail("prepare called"),
    )
    monkeypatch.setattr(cli.staleness, "artifact_matches", lambda _model, _artifact: True)
    c, _ = make_console()
    assert cli._prefetch_weights(
        c, "org/model", backend, "cpu", as_json=False, progress=False,
        authorized_artifact_id="hf:authority") is None


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("recoverable", [False, True])
def test_prefetch_refuses_unrecoverable_authority_or_legacy_downloader(
        make_console, monkeypatch, capsys, as_json, recoverable):
    revision = "a" * 40
    backend = types.SimpleNamespace(calibration_model_cached=lambda _model: False)
    monkeypatch.setattr(cli.staleness, "artifact_matches", lambda *_a: False)
    monkeypatch.setattr(
        cli.staleness, "authorized_download_ref",
        lambda *_a: ("org/model", revision) if recoverable else None)
    c, buf = make_console()

    needed, payload, rc = cli._prefetch_plan(
        c, "org/model", backend, "cpu", as_json=as_json,
        authorized_artifact_id="hf:authority")
    assert (needed, payload, rc) == (False, None, 1)
    output = capsys.readouterr().out if as_json else buf.getvalue()
    assert ("cannot recover" if not recoverable else "cannot recover an exact") in output


@pytest.mark.parametrize("as_json", [False, True])
def test_prefetch_preparation_failure_is_a_clean_error(
        make_console, monkeypatch, capsys, as_json):
    backend = types.SimpleNamespace(
        calibration_model_cached=lambda _model: False,
        prepare_download=lambda _model: (_ for _ in ()).throw(RuntimeError("revision race")),
    )
    c, buf = make_console()

    assert cli._prefetch_weights(
        c, "org/model", backend, "cpu", as_json=as_json, progress=False) == 1
    output = capsys.readouterr().out if as_json else buf.getvalue()
    assert "couldn't fetch" in output


def test_characterize_tracks_calibration_and_backend_measurement(
        make_console, monkeypatch, activity_registry):
    calls = []

    def calibrate():
        _assert_activity("characterizing", "org/model")
        calls.append("calibrate")
        return {"overhead_gb": None, "wall_gb": None}

    def characterize(*_a, **_k):
        _assert_activity("characterizing", "org/model")
        calls.append("characterize")
        return {"safe_context": 4096, "decode_context": None, "points": []}

    backend = types.SimpleNamespace(
        calibration_model_cached=lambda _m: True,
        download_calibration_model=lambda *_a, **_k: None,
        calibrate=calibrate,
        characterize=characterize,
    )
    _wire_characterize(monkeypatch, backend)
    c, _ = make_console()
    assert cli.render_characterize(c, "org/model", engine="cpu") == 0
    assert calls == ["calibrate", "characterize"]
    assert activity.snapshot() == []


def test_characterize_download_calibration_and_measurement_share_one_activity(
        make_console, monkeypatch, activity_registry):
    record_names = []

    def observe(stage):
        _assert_activity("characterizing", "org/model")
        record_names.append((stage, next(activity_registry.glob("*.json")).name))

    backend = types.SimpleNamespace(
        calibration_model_cached=lambda _m: False,
        download_calibration_model=lambda *_a, **_k: observe("download"),
        calibrate=lambda: (observe("calibrate") or {
            "overhead_gb": None, "wall_gb": None}),
        characterize=lambda *_a, **_k: (observe("characterize") or {
            "safe_context": 4096, "decode_context": None, "points": []}),
    )
    _wire_characterize(monkeypatch, backend)
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda _m: None)
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: None)
    c, _ = make_console()
    assert cli.render_characterize(c, "org/model", engine="cpu") == 0
    assert [stage for stage, _name in record_names] == [
        "download", "calibrate", "characterize"]
    assert len({name for _stage, name in record_names}) == 1


def test_characterize_disk_refusal_never_starts_tracking(
        make_console, monkeypatch, activity_registry):
    backend = types.SimpleNamespace(
        calibration_model_cached=lambda _m: False,
        download_calibration_model=lambda *_a, **_k: pytest.fail("download called"),
        characterize=lambda *_a, **_k: pytest.fail("characterize called"),
    )
    _wire_characterize(monkeypatch, backend)
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda _m: 10.0)
    monkeypatch.setattr(cli.acquire, "gguf_size_gb", lambda _m: 10.0)
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: 1.0)
    monkeypatch.setattr(cli.activity, "track", lambda *_a, **_k: pytest.fail("track called"))
    c, _ = make_console()
    assert cli.render_characterize(c, "org/model", engine="cpu") == 1
    assert not activity_registry.exists()


def test_characterize_prefetch_refusal_never_creates_activity(
        make_console, monkeypatch, activity_registry):
    def download(*_a, **_k):
        assert activity.snapshot() == []
        raise RuntimeError("offline")

    backend = types.SimpleNamespace(
        calibration_model_cached=lambda _m: False,
        download_calibration_model=download,
        characterize=lambda *_a, **_k: pytest.fail("characterize called"),
    )
    _wire_characterize(monkeypatch, backend)
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda _m: None)
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: None)
    c, _ = make_console()
    assert cli.render_characterize(c, "org/model", engine="cpu") == 1
    assert activity.snapshot() == []


def test_characterize_validation_failure_never_starts_tracking(make_console, monkeypatch):
    monkeypatch.setattr(cli.activity, "track", lambda *_a, **_k: pytest.fail("track called"))
    c, _ = make_console()
    assert cli.render_characterize(c, "../bad", engine="cpu") == 1


@pytest.mark.parametrize("raised", [KeyboardInterrupt(), SystemExit(4)])
def test_characterize_activity_cleans_up_on_base_exceptions(
        make_console, monkeypatch, activity_registry, raised):
    def characterize(*_a, **_k):
        _assert_activity("characterizing", "org/model")
        raise raised

    backend = types.SimpleNamespace(
        calibration_model_cached=lambda _m: True,
        download_calibration_model=lambda *_a, **_k: None,
        characterize=characterize,
    )
    _wire_characterize(monkeypatch, backend)
    c, _ = make_console()
    if isinstance(raised, SystemExit):
        assert cli.render_characterize(c, "org/model", engine="cpu") == 1
    else:
        with pytest.raises(KeyboardInterrupt):
            cli.render_characterize(c, "org/model", engine="cpu")
    assert activity.snapshot() == []


def test_characterize_click_acquires_measurement_lock_before_tracking(monkeypatch):
    order = []

    @contextlib.contextmanager
    def locked():
        order.append("lock-enter")
        yield
        order.append("lock-exit")

    @contextlib.contextmanager
    def tracked(*_a, **_k):
        order.append("track-enter")
        yield
        order.append("track-exit")

    def render(*_a, **_k):
        with cli.activity.track("characterizing", "org/model"):
            order.append("render")
        return 0

    monkeypatch.setattr(cli.locking, "measurement_lock", locked)
    monkeypatch.setattr(cli.activity, "track", tracked)
    monkeypatch.setattr(cli, "render_characterize", render)
    result = CliRunner().invoke(cli._click_cli, ["characterize", "org/model"])
    assert result.exit_code == 0
    assert order == ["lock-enter", "track-enter", "render", "track-exit", "lock-exit"]


@pytest.mark.parametrize(
    ("command", "renderer"),
    [(["install", "cpu"], "render_install"),
     (["uninstall", "cpu"], "render_uninstall")],
)
def test_engine_mutation_clicks_acquire_measurement_lock(monkeypatch, command, renderer):
    order = []

    @contextlib.contextmanager
    def locked():
        order.append("lock-enter")
        yield
        order.append("lock-exit")

    def render(*_args, **_kwargs):
        order.append("render")
        return 0

    monkeypatch.setattr(cli.locking, "measurement_lock", locked)
    monkeypatch.setattr(cli, renderer, render)

    result = CliRunner().invoke(cli._click_cli, command)

    assert result.exit_code == 0
    assert order == ["lock-enter", "render", "lock-exit"]


def _wire_benchmark(monkeypatch, backend):
    monkeypatch.setattr(cli.engines, "resolve", lambda _engine: "cpu")
    monkeypatch.setitem(cli.engines.ENGINES, "cpu", {
        **cli.engines.ENGINES["cpu"], "backend": "cpu"})
    monkeypatch.setattr(cli, "get_backend", lambda _backend: backend)
    monkeypatch.setattr(
        cli, "_current_reuse_identity", lambda *_a: ("method:test", "engine:test", None))
    monkeypatch.setattr(cli, "engine_status", lambda _backend=None: (True, "CPU engine"))
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "machine")
    row = {"safe_context": 4096, "measured_at": None, "artifact_id": "artifact:test"}
    monkeypatch.setattr(cli.db, "get_characterization", lambda *_a: row)
    monkeypatch.setattr(
        cli.db, "get_reusable_characterization_for_engine", lambda *_a, **_k: row)
    monkeypatch.setattr(cli.benchmark, "load_probe", lambda _use_case: [{"answer": "a"}])
    monkeypatch.setattr(cli.benchmark, "prompt_for", lambda *_a: "prompt")
    monkeypatch.setattr(cli.benchmark, "score_probe_set", lambda *_a: 1.0)
    monkeypatch.setattr(cli.db, "get_model", lambda *_a: None)
    monkeypatch.setattr(cli.db, "save_benchmark_result", lambda *_a, **_k: None)
    monkeypatch.setattr(cli.staleness, "artifact_identity",
                        lambda _model, **_kwargs: "artifact:test")
    monkeypatch.setattr(
        cli.staleness, "pinned_model_ref", lambda model, _artifact, **_kwargs: model)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: False))


def test_benchmark_one_activity_covers_prefetch_and_every_repeat(
        make_console, monkeypatch, activity_registry):
    record_names = []

    def download(*_a, **_k):
        _assert_activity("benchmarking", "org/model")
        record_names.append(next(activity_registry.glob("*.json")).name)

    def run(*_a, **_k):
        _assert_activity("benchmarking", "org/model")
        record_names.append(next(activity_registry.glob("*.json")).name)
        return {"context": 4096, "results": [{"prompt_index": 0, "completion": "a"}]}

    backend = types.SimpleNamespace(
        calibration_model_cached=lambda _m: False,
        download_prepared_model=download,
        benchmark=run,
    )
    _wire_benchmark(monkeypatch, backend)
    monkeypatch.setattr(cli.staleness, "artifact_matches", lambda *_a: False)
    revision = "a" * 40
    plan = cli.acquire.AcquisitionPlan("org/model", "org/model", revision, None, 1.0)
    monkeypatch.setattr(cli.staleness, "authorized_download_ref",
                        lambda *_a: ("org/model", revision))
    monkeypatch.setattr(cli.acquire, "prepare_download", lambda *_a, **_k: plan)
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: 50.0)
    c, _ = make_console()
    assert cli.render_benchmark(c, "org/model", use_case="reasoning",
                                engine="cpu", repeat=2) == 0
    assert len(set(record_names)) == 1
    assert len(record_names) == 3
    assert activity.snapshot() == []


def test_benchmark_cache_check_and_disk_refusal_happen_before_tracking(
        make_console, monkeypatch, activity_registry):
    def cached(_model):
        assert activity.snapshot() == []
        return False

    backend = types.SimpleNamespace(
        calibration_model_cached=cached,
        download_prepared_model=lambda *_a, **_k: pytest.fail("download called"),
        benchmark=lambda *_a, **_k: pytest.fail("benchmark called"),
    )
    _wire_benchmark(monkeypatch, backend)
    monkeypatch.setattr(cli.staleness, "artifact_matches", lambda *_a: False)
    revision = "a" * 40
    plan = cli.acquire.AcquisitionPlan("org/model", "org/model", revision, None, 10.0)
    monkeypatch.setattr(cli.staleness, "authorized_download_ref",
                        lambda *_a: ("org/model", revision))
    monkeypatch.setattr(cli.acquire, "prepare_download", lambda *_a, **_k: plan)
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: 1.0)
    monkeypatch.setattr(cli.activity, "track", lambda *_a, **_k: pytest.fail("track called"))
    c, _ = make_console()
    assert cli.render_benchmark(c, "org/model", use_case="reasoning", engine="cpu") == 1
    assert not activity_registry.exists()


def test_benchmark_download_failure_is_tracked_then_cleaned_without_backend_call(
        make_console, monkeypatch, activity_registry):
    def download(*_a, **_k):
        _assert_activity("benchmarking", "org/model")
        raise RuntimeError("offline")

    backend = types.SimpleNamespace(
        calibration_model_cached=lambda _m: False,
        download_prepared_model=download,
        benchmark=lambda *_a, **_k: pytest.fail("benchmark called"),
    )
    _wire_benchmark(monkeypatch, backend)
    monkeypatch.setattr(cli.staleness, "artifact_matches", lambda *_a: False)
    revision = "a" * 40
    plan = cli.acquire.AcquisitionPlan("org/model", "org/model", revision, None, 1.0)
    monkeypatch.setattr(cli.staleness, "authorized_download_ref",
                        lambda *_a: ("org/model", revision))
    monkeypatch.setattr(cli.acquire, "prepare_download", lambda *_a, **_k: plan)
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: 50.0)
    c, _ = make_console()
    assert cli.render_benchmark(c, "org/model", use_case="reasoning", engine="cpu") == 1
    assert activity.snapshot() == []


def test_benchmark_invalid_gate_never_creates_activity(
        make_console, monkeypatch, activity_registry):
    monkeypatch.setattr(cli.activity, "track", lambda *_a, **_k: pytest.fail("track called"))
    c, _ = make_console()
    assert cli.render_benchmark(c, "org/model", use_case="coding",
                                exec_consent=False) == 1
    assert activity.snapshot() == []


@pytest.mark.parametrize("raised", [RuntimeError("boom"), SystemExit(5)])
def test_benchmark_activity_cleans_up_and_reports_operational_exceptions(
        make_console, monkeypatch, activity_registry, raised):
    def run(*_a, **_k):
        _assert_activity("benchmarking", "org/model")
        raise raised

    backend = types.SimpleNamespace(
        calibration_model_cached=lambda _m: True,
        download_calibration_model=lambda *_a, **_k: None,
        benchmark=run,
    )
    _wire_benchmark(monkeypatch, backend)
    c, buf = make_console()
    assert cli.render_benchmark(c, "org/model", use_case="reasoning", engine="cpu") == 1
    assert "benchmark failed" in buf.getvalue()
    assert activity.snapshot() == []


def test_benchmark_activity_cleans_up_and_propagates_keyboard_interrupt(
        make_console, monkeypatch, activity_registry):
    def run(*_a, **_k):
        _assert_activity("benchmarking", "org/model")
        raise KeyboardInterrupt()

    backend = types.SimpleNamespace(
        calibration_model_cached=lambda _m: True,
        download_calibration_model=lambda *_a, **_k: None,
        benchmark=run,
    )
    _wire_benchmark(monkeypatch, backend)
    c, _ = make_console()
    with pytest.raises(KeyboardInterrupt):
        cli.render_benchmark(c, "org/model", use_case="reasoning", engine="cpu")
    assert activity.snapshot() == []


def test_benchmark_click_acquires_measurement_lock_before_tracking(monkeypatch):
    order = []

    @contextlib.contextmanager
    def locked():
        order.append("lock-enter")
        yield
        order.append("lock-exit")

    @contextlib.contextmanager
    def tracked(*_a, **_k):
        order.append("track-enter")
        yield
        order.append("track-exit")

    def render(*_a, **_k):
        with cli.activity.track("benchmarking", "org/model"):
            order.append("render")
        return 0

    monkeypatch.setattr(cli.locking, "measurement_lock", locked)
    monkeypatch.setattr(cli.activity, "track", tracked)
    monkeypatch.setattr(cli, "render_benchmark", render)
    result = CliRunner().invoke(
        cli._click_cli, ["benchmark", "org/model", "--use-case", "reasoning"])
    assert result.exit_code == 0
    assert order == ["lock-enter", "track-enter", "render", "track-exit", "lock-exit"]


def _wire_mlx_serve_characterization(monkeypatch):
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "machine")
    authority = types.SimpleNamespace(key="mlx-authority:test")
    monkeypatch.setattr(
        cli.measurement_authority, "current_measurement_authority",
        lambda _engine: authority)
    row = {"safe_context": 4096, "measured_at": None, "points": [],
           "authority_key": authority.key}
    monkeypatch.setattr(cli.db, "get_characterization", lambda *_a: row)
    monkeypatch.setattr(
        cli.db, "get_reusable_characterization_for_engine", lambda *_a, **_k: row)
    monkeypatch.setattr(
        cli, "_current_reuse_identity", lambda *_a: ("method:test", "engine:test", None))


def test_mlx_serve_tracks_after_ready_handshake_through_wait(
        make_console, monkeypatch, activity_registry):
    _wire_mlx_serve_characterization(monkeypatch)
    monkeypatch.setattr(cli, "_free_port", lambda: 1234)

    class Proc:
        def wait(self):
            _assert_activity("serving", "org/model")

    def serve(*_a, **_k):
        assert activity.snapshot() == []
        return Proc(), "http://127.0.0.1:1234", 4096

    monkeypatch.setattr(cli, "get_backend", lambda _backend: types.SimpleNamespace(serve=serve))
    c, _ = make_console()
    assert cli._render_serve_mlx(c, "org/model", engine_key="mlx", assume_yes=True) == 0
    assert activity.snapshot() == []


def test_mlx_backend_startup_failure_never_starts_tracking(make_console, monkeypatch):
    _wire_mlx_serve_characterization(monkeypatch)
    monkeypatch.setattr(cli, "_free_port", lambda: 1234)
    monkeypatch.setattr(cli, "get_backend", lambda _backend: types.SimpleNamespace(
        serve=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("not ready"))))
    monkeypatch.setattr(cli.activity, "track", lambda *_a, **_k: pytest.fail("track called"))
    c, _ = make_console()
    assert cli._render_serve_mlx(c, "org/model", engine_key="mlx", assume_yes=True) == 1


@pytest.mark.parametrize("raised", [RuntimeError("boom"), KeyboardInterrupt(), SystemExit(3)])
def test_mlx_serve_wait_cleanup_covers_all_exit_paths(
        make_console, monkeypatch, activity_registry, raised):
    _wire_mlx_serve_characterization(monkeypatch)
    monkeypatch.setattr(cli, "_free_port", lambda: 1234)

    class Proc:
        def wait(self):
            _assert_activity("serving", "org/model")
            raise raised

    monkeypatch.setattr(cli, "get_backend", lambda _backend: types.SimpleNamespace(
        serve=lambda *_a, **_k: (Proc(), "http://127.0.0.1:1234", 4096)))
    c, _ = make_console()
    if isinstance(raised, KeyboardInterrupt):
        assert cli._render_serve_mlx(
            c, "org/model", engine_key="mlx", assume_yes=True) == 0
    else:
        with pytest.raises(type(raised)):
            cli._render_serve_mlx(c, "org/model", engine_key="mlx", assume_yes=True)
    assert activity.snapshot() == []


def test_mlx_serve_sigterm_unwind_cleans_activity(
        make_console, monkeypatch, activity_registry):
    import signal

    _wire_mlx_serve_characterization(monkeypatch)
    monkeypatch.setattr(cli, "_free_port", lambda: 1234)
    installed = {}

    def set_signal(sig, handler):
        if sig == signal.SIGTERM and callable(handler):
            installed["handler"] = handler
        return signal.SIG_DFL

    monkeypatch.setattr(signal, "signal", set_signal)

    class Proc:
        def terminate(self):
            _assert_activity("serving", "org/model")

        def wait(self):
            _assert_activity("serving", "org/model")
            installed["handler"](signal.SIGTERM, None)

    monkeypatch.setattr(cli, "get_backend", lambda _backend: types.SimpleNamespace(
        serve=lambda *_a, **_k: (Proc(), "http://127.0.0.1:1234", 4096)))
    c, _ = make_console()
    with pytest.raises(SystemExit):
        cli._render_serve_mlx(c, "org/model", engine_key="mlx", assume_yes=True)
    assert activity.snapshot() == []


def _wire_ollama_serve(monkeypatch, *, isatty=False):
    monkeypatch.setattr(
        cli.ollama, "runtime_authority",
        lambda endpoint: cli.ollama.OllamaRuntimeAuthority(
            endpoint=endpoint,
            server_version="0.30.10",
            server_instance_id="42:1234.500000:/usr/bin/ollama",
            listener_pid=42,
            listener_bind_host="127.0.0.1",
            configured_num_parallel=1,
            configured_num_parallel_authority="exact_version_default",
        ),
    )
    monkeypatch.setattr(cli.ollama, "version", lambda: "1.0")
    monkeypatch.setattr(cli.ollama, "openai_completion_probe", lambda _name: True)
    monkeypatch.setattr(cli.ollama, "ps", lambda _timeout=2.0: [])
    monkeypatch.setattr(cli.ollama, "tags", lambda: ["base:model"])
    monkeypatch.setattr(
        cli.ollama, "inventory",
        lambda: [cli.ollama.OllamaModel(
            name="base:model", digest="a" * 64, format="gguf",
            capabilities=("completion",))],
    )
    monkeypatch.setattr(cli.ollama, "base_url", lambda: "http://127.0.0.1:11434")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "machine")
    wall_evidence = {
        "placement": "unified",
        "resident_total_bytes": 500_000_000,
        "resident_accelerator_bytes": 500_000_000,
        "applicable_walls": ["system_unified"],
        "system_memory_delta_bytes": 500_000_000,
        "accelerator_memory_delta_bytes": None,
        "system_margin_bytes": cli.ollama_evidence.SYSTEM_MARGIN_BYTES,
        "accelerator_margin_bytes": None,
    }
    admission = cli.ollama_evidence.preflight_admission(
        cli.ollama_evidence.MemorySnapshot(
            24 * 1024 ** 3, 8 * 1024 ** 3, "apple", 1, None, None, True),
        500_000_000,
        requested_context=4096,
        model_info={
            "general.architecture": "qwen3",
            "qwen3.block_count": 28,
            "qwen3.attention.head_count_kv": 8,
            "qwen3.attention.key_length": 128,
        },
    )
    assert admission.reason is None
    point = {
        "fit": True,
        "context": 4096,
        "requested_context": 4096,
        "effective_per_request_context": 4096,
        "refusal_reasons": [],
        "preload_admission": admission.as_dict(),
        **wall_evidence,
    }
    characterization = {
        "safe_context": 4096,
        "measured_at": None,
        "artifact_id": "ollama-manifest-sha256:" + "a" * 64,
        "reusable": True,
        "methodology_key": (
            cli.ollama_evidence.CHARACTERIZATION_METHODOLOGY_KEY),
        "engine_fingerprint": cli.ollama_evidence.runtime_fingerprint(
            cli.ollama.OllamaRuntimeAuthority(
                endpoint=cli.ollama.OllamaEndpoint(
                    "http://127.0.0.1:11434", "loopback"),
                server_version="0.30.10",
                server_instance_id="42:1234.500000:/usr/bin/ollama",
                listener_pid=42,
                listener_bind_host="127.0.0.1",
                configured_num_parallel=1,
                configured_num_parallel_authority="exact_version_default",
            ),
        ),
        "points": [point],
        "config": {
            "methodology": "ollama-physical-walls-v1",
            "runtime": "ollama",
            "runtime_version": "0.30.10",
            "endpoint_authority": "http://127.0.0.1:11434",
            "server_instance_id": "42:1234.500000:/usr/bin/ollama",
            "format": "gguf",
            "capability": "completion",
            "configured_inputs": {},
            "configured_num_parallel": 1,
            "configured_num_parallel_authority": "exact_version_default",
            "effective_num_parallel": 1,
            "effective_num_parallel_authority": "configured_maximum_is_one",
            "requested_context": 4096,
            "effective_per_request_context": 4096,
            "configured_kv_cache_type": "unknown",
            "effective_kv_cache_type": "unknown",
            "configured_flash_attention": "unknown",
            "effective_flash_attention": "unknown",
            "configured_scheduler_spread": "unknown",
            "effective_scheduler_spread": "unknown",
            "preload_admission": admission.as_dict(),
            "watchdog": cli.ollama_evidence.WATCHDOG_STATUS,
            **wall_evidence,
        },
    }
    monkeypatch.setattr(
        cli.db, "list_characterizations_for_display", lambda *_a, **_k: [characterization])
    monkeypatch.setattr(cli.ollama, "manifest_digest",
                        lambda name: "a" * 64 if name == "base:model" else "b" * 64)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: isatty))


_OLLAMA_BASE_ARTIFACT = "ollama-manifest-sha256:" + "a" * 64
_OLLAMA_SERVED_ARTIFACT = "ollama-manifest-sha256:" + "b" * 64


def _ollama_served_name():
    return cli._governed_name(
        "base:model", artifact_id=_OLLAMA_BASE_ARTIFACT, context=4096,
    )


def _ollama_artifact_fields():
    return {
        "base_artifact_id": _OLLAMA_BASE_ARTIFACT,
        "served_artifact_id": _OLLAMA_SERVED_ARTIFACT,
    }


def _ollama_runtime_authority_fields():
    return {
        "policy_version": cli._OLLAMA_DERIVED_POLICY_VERSION,
        "runtime_authority": {
            "runtime_version": "0.30.10",
            "server_instance_id": "42:1234.500000:/usr/bin/ollama",
            "configured_inputs": {},
            "configured_num_parallel": 1,
            "configured_num_parallel_authority": "exact_version_default",
        },
    }


def _wire_ollama_status_authority(monkeypatch):
    endpoint = cli.ollama.OllamaEndpoint(
        url="http://127.0.0.1:11434", scope="loopback")
    monkeypatch.setattr(cli.ollama, "endpoint_authority", lambda _url: endpoint)
    monkeypatch.setattr(
        cli.ollama, "runtime_authority",
        lambda received: cli.ollama.OllamaRuntimeAuthority(
            endpoint=received,
            server_version="0.30.10",
            server_instance_id="42:1234.500000:/usr/bin/ollama",
            listener_pid=42,
            listener_bind_host="127.0.0.1",
            configured_num_parallel=1,
            configured_num_parallel_authority="exact_version_default",
        ),
    )


def test_ollama_serve_temporary_activity_hands_off_to_persistent_without_overlap(
        make_console, monkeypatch, activity_registry):
    _wire_ollama_serve(monkeypatch)
    calls = []
    loaded = False
    setup_phase = True

    def create(*_a, **_k):
        _assert_activity("serving", "base:model")
        calls.append("create")
        return True

    def load(_name, keep_alive=-1, **_k):
        nonlocal loaded
        assert keep_alive is None
        _assert_activity("serving", "base:model")
        calls.append("load-temporary")
        loaded = True
        return {"done": True}

    def verify(*_a, **_k):
        if setup_phase:
            _assert_activity("serving", "base:model")
        calls.append("verify")
        if not loaded:
            return []
        return [{"name": f"{_ollama_served_name()}:latest", "context_length": 4096,
                 "size": 10, "size_vram": 10, "digest": "b" * 64}]

    real_record = activity.record_ollama_serving

    def record(**fields):
        nonlocal setup_phase
        assert activity.snapshot() == []
        calls.append("record")
        path = real_record(**fields)
        setup_phase = False
        return path

    monkeypatch.setattr(cli.ollama, "create", create)
    monkeypatch.setattr(cli.ollama, "load", load)
    monkeypatch.setattr(cli.ollama, "ps", verify)
    monkeypatch.setattr(cli.activity, "record_ollama_serving", record)
    c, _ = make_console()
    assert cli.render_serve(c, "base:model", ctx=4096) == 0
    assert calls == ["verify", "create", "load-temporary", "verify", "verify", "record"]

    monkeypatch.setattr(cli.ollama, "ps", lambda: [
        {"name": f"{_ollama_served_name()}:latest", "context_length": 4096,
         "digest": "b" * 64}])
    found = activity.snapshot()
    assert len(found) == 1 and found[0].runtime == "ollama"
    assert not list(activity_registry.glob("*.json"))


def test_ollama_reserve_same_live_identity_never_duplicates_status(
        make_console, monkeypatch, activity_registry):
    _wire_ollama_serve(monkeypatch)
    activity.record_ollama_serving(
        served_name=_ollama_served_name(), model="base:model", context=4096,
        endpoint="http://127.0.0.1:11434", started_at=1.0,
        policy_version=cli._OLLAMA_DERIVED_POLICY_VERSION,
        runtime_authority={
            "runtime_version": "0.30.10",
            "server_instance_id": "42:1234.500000:/usr/bin/ollama",
            "configured_inputs": {},
            "configured_num_parallel": 1,
            "configured_num_parallel_authority": "exact_version_default",
        },
        **_ollama_artifact_fields())
    loaded = [{"name": f"{_ollama_served_name()}:latest", "context_length": 4096,
               "size": 10, "size_vram": 10, "digest": "b" * 64}]
    monkeypatch.setattr(cli.ollama, "ps", lambda *_a: loaded)

    def create(*_a, **_k):
        found = activity.snapshot()
        assert [(item.kind, item.model, item.runtime) for item in found] == [
            ("serving", "base:model", "ollama")]
        assert not list(activity_registry.glob("*.json"))
        return True

    monkeypatch.setattr(cli.ollama, "create", create)
    monkeypatch.setattr(cli.ollama, "load", lambda *_a, **_k: {"done": True})
    c, _ = make_console()
    assert cli.render_serve(c, "base:model", ctx=4096) == 0


def test_ollama_status_requires_exact_base_and_served_manifest_digests(
        monkeypatch, activity_registry):
    base_artifact = "ollama-manifest-sha256:" + "a" * 64
    served_artifact = "ollama-manifest-sha256:" + "b" * 64
    activity.record_ollama_serving(
        served_name="base-model-ara", model="base:model", context=4096,
        endpoint="http://127.0.0.1:11434", base_artifact_id=base_artifact,
        served_artifact_id=served_artifact, started_at=1.0,
        **_ollama_runtime_authority_fields(),
    )
    _wire_ollama_status_authority(monkeypatch)
    monkeypatch.setattr(activity.ollama if hasattr(activity, "ollama") else cli.ollama,
                        "base_url", lambda: "http://127.0.0.1:11434")
    monkeypatch.setattr(cli.ollama, "manifest_digest",
                        lambda name: "a" * 64 if name == "base:model" else None)
    monkeypatch.setattr(cli.ollama, "ps", lambda: [{
        "name": "base-model-ara:latest", "context_length": 4096,
        "digest": "b" * 64,
    }])
    found = activity.snapshot()
    assert len(found) == 1
    assert found[0].base_artifact_id == base_artifact
    assert found[0].served_artifact_id == served_artifact

    monkeypatch.setattr(cli.ollama, "ps", lambda: [{
        "name": "base-model-ara:latest", "context_length": 4096,
        "digest": "c" * 64,
    }])
    assert activity.snapshot() == []


def test_ollama_status_never_promotes_pre_v2_ownership_to_live_authority(
        monkeypatch, activity_registry):
    activity.record_ollama_serving(
        served_name="base-model-ara", model="base:model", context=4096,
        endpoint="http://127.0.0.1:11434",
        base_artifact_id="ollama-manifest-sha256:" + "a" * 64,
        served_artifact_id="ollama-manifest-sha256:" + "b" * 64,
        started_at=1.0,
    )
    monkeypatch.setattr(cli.ollama, "base_url", lambda: "http://127.0.0.1:11434")
    monkeypatch.setattr(cli.ollama, "manifest_digest", lambda _name: "a" * 64)
    monkeypatch.setattr(cli.ollama, "ps", lambda: [{
        "name": "base-model-ara:latest", "context_length": 4096,
        "digest": "b" * 64,
    }])

    ownership = activity.ollama_ownership()
    assert len(ownership) == 1
    assert ownership[0].policy_version is None
    assert activity.snapshot() == []


def test_ollama_status_suppresses_retargeted_base_manifest(monkeypatch):
    activity.record_ollama_serving(
        served_name="base-model-ara", model="base:model", context=4096,
        endpoint="http://127.0.0.1:11434",
        base_artifact_id="ollama-manifest-sha256:" + "a" * 64,
        served_artifact_id="ollama-manifest-sha256:" + "b" * 64,
        started_at=1.0, **_ollama_runtime_authority_fields(),
    )
    _wire_ollama_status_authority(monkeypatch)
    monkeypatch.setattr(cli.ollama, "manifest_digest", lambda _name: "d" * 64)
    monkeypatch.setattr(cli.ollama, "ps", lambda: [{
        "name": "base-model-ara:latest", "context_length": 4096,
        "digest": "b" * 64,
    }])
    assert activity.snapshot() == []


def test_ollama_refuses_takeover_of_served_identity_owned_by_another_base(
        make_console, monkeypatch, activity_registry):
    _wire_ollama_serve(monkeypatch)
    monkeypatch.setattr(cli.ollama, "manifest_digest",
                        lambda name: "a" * 64 if name in ("base:model", "org/old") else "b" * 64)
    activity.record_ollama_serving(
        served_name="shared", model="org/old", context=4096,
        endpoint="http://127.0.0.1:11434", started_at=1.0,
        **_ollama_artifact_fields(), **_ollama_runtime_authority_fields())
    loaded = [{"name": "shared:latest", "context_length": 4096,
               "size": 10, "size_vram": 10, "digest": "b" * 64}]
    monkeypatch.setattr(cli.ollama, "ps", lambda: loaded)

    monkeypatch.setattr(cli.ollama, "tags", lambda: ["base:model", "shared:latest"])
    monkeypatch.setattr(
        cli.ollama, "inventory",
        lambda: [
            cli.ollama.OllamaModel(
                name="base:model", digest="a" * 64, format="gguf",
                capabilities=("completion",)),
            cli.ollama.OllamaModel(name="shared:latest", digest="b" * 64),
        ],
    )
    monkeypatch.setattr(cli.ollama, "create",
                        lambda *_a, **_k: pytest.fail("existing identity overwritten"))
    c, buf = make_console()
    assert cli.render_serve(c, "base:model", ctx=4096, name="shared") == 1
    assert "already exists" in buf.getvalue() and "refusing" in buf.getvalue()
    assert [(item.kind, item.model, item.runtime) for item in activity.snapshot()] == [
        ("serving", "org/old", "ollama")]


def test_ollama_serve_declined_consent_never_claims_activity(
        make_console, monkeypatch, activity_registry):
    _wire_ollama_serve(monkeypatch, isatty=True)
    monkeypatch.setattr(cli, "_confirm", lambda _question: False)
    monkeypatch.setattr(cli.ollama, "create", lambda *_a: pytest.fail("create called"))
    monkeypatch.setattr(cli.activity, "track", lambda *_a, **_k: pytest.fail("track called"))
    c, _ = make_console()
    assert cli.render_serve(c, "base:model", ctx=4096) == 0
    assert activity.snapshot() == []


def test_ollama_manifest_failure_deletes_manifest_without_expiring_runner(
        make_console, monkeypatch, activity_registry):
    _wire_ollama_serve(monkeypatch)
    monkeypatch.setattr(cli.ollama, "create", lambda *_a: True)
    unloading = []
    loaded = []
    deleted = []

    def load(name, keep_alive=-1):
        (unloading if keep_alive == 0 else loaded).append(name)
        return {} if keep_alive == 0 else {"done": True}

    monkeypatch.setattr(cli.ollama, "load", load)
    monkeypatch.setattr(cli.ollama, "ps", lambda *_a: ([] if unloading or not loaded else [
        {"name": f"{_ollama_served_name()}:latest", "context_length": 4096,
         "size": 10, "size_vram": 10, "digest": "b" * 64}]))
    monkeypatch.setattr(cli.ollama, "delete", lambda name: deleted.append(name) or True)
    monkeypatch.setattr(cli.activity, "record_ollama_serving",
                        lambda **_fields: (_ for _ in ()).throw(OSError("disk full")))
    c, buf = make_console()
    assert cli.render_serve(c, "base:model", ctx=4096) == 1
    assert "ownership could not be recorded" in buf.getvalue()
    assert "without expiring its runner" in buf.getvalue()
    assert unloading == [] and loaded == [_ollama_served_name()]
    assert deleted == [_ollama_served_name()]
    assert activity.snapshot() == []


def test_ollama_manifest_validation_failure_is_honest_json_not_raw_exception(
        make_console, monkeypatch, activity_registry, capsys):
    _wire_ollama_serve(monkeypatch)
    monkeypatch.setattr(cli.ollama, "create", lambda *_a: True)
    unloading = []
    loaded = []

    def load(name, keep_alive=-1):
        (unloading if keep_alive == 0 else loaded).append(name)
        return {} if keep_alive == 0 else {"done": True}

    monkeypatch.setattr(cli.ollama, "load", load)
    monkeypatch.setattr(cli.ollama, "ps", lambda *_a: ([] if unloading or not loaded else [
        {"name": f"{_ollama_served_name()}:latest", "context_length": 4096,
         "size": 10, "size_vram": 10, "digest": "b" * 64}]))
    monkeypatch.setattr(cli.ollama, "delete", lambda _name: True)
    monkeypatch.setattr(cli.activity, "record_ollama_serving",
                        lambda **_fields: (_ for _ in ()).throw(ValueError("invalid identity")))
    c, _ = make_console()
    assert cli.render_serve(c, "base:model", ctx=4096, as_json=True) == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": f"{_ollama_served_name()} loaded at 4096 ctx, but ARA ownership could not "
                 "be recorded: invalid identity; deleted the untracked manifest without "
                 "expiring its runner"}
    assert unloading == [] and loaded == [_ollama_served_name()]
    assert activity.snapshot() == []


@pytest.mark.parametrize("name", ["x" * 513, "bad\nname"])
def test_ollama_invalid_custom_served_name_refuses_before_create_without_raw_exception(
        make_console, monkeypatch, activity_registry, name):
    _wire_ollama_serve(monkeypatch)
    monkeypatch.setattr(cli.ollama, "create", lambda *_a: pytest.fail("create called"))
    c, buf = make_console()
    assert cli.render_serve(c, "base:model", ctx=4096, name=name) == 1
    assert "invalid" in buf.getvalue().lower()
    assert activity.snapshot() == []


@pytest.mark.parametrize("raised", [KeyboardInterrupt(), SystemExit(6)])
def test_ollama_setup_activity_cleans_up_on_base_exceptions(
        make_console, monkeypatch, activity_registry, raised):
    _wire_ollama_serve(monkeypatch)

    def create(*_a, **_k):
        _assert_activity("serving", "base:model")
        raise raised

    monkeypatch.setattr(cli.ollama, "create", create)
    c, _ = make_console()
    with pytest.raises(type(raised)):
        cli.render_serve(c, "base:model", ctx=4096)
    assert activity.snapshot() == []


def test_find_loaded_accepts_only_exact_name_or_latest_normalization():
    entries = [
        {"name": "svc:other"},
        {"name": "svc-extra:latest"},
        {"name": "svc:latest"},
    ]
    assert cli._find_loaded(entries, "svc") == {"name": "svc:latest"}
    assert cli._find_loaded(entries[:2], "svc") is None


def test_find_loaded_scans_matching_rows_for_later_valid_exact_context():
    entries = [
        {"name": "svc", "context_length": True},
        {"name": "svc:latest", "context_length": "4096"},
        {"name": "svc", "context_length": 2048},
        {"name": "svc:latest", "context_length": 4096},
    ]
    assert cli._find_loaded(entries, "svc", expected_context=4096) == entries[-1]


@pytest.mark.parametrize("name", [None, [], {}, 7])
def test_find_loaded_ignores_malformed_names_without_crashing(name):
    assert cli._find_loaded([{"name": name}], "svc") is None


@pytest.mark.parametrize("entry", [None, [], "bad", 7])
def test_find_loaded_ignores_non_object_entries_without_crashing(entry):
    assert cli._find_loaded([entry], "svc") is None
