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
    monkeypatch.setattr(cli.db, "get_characterization", lambda *_a: {
        "safe_context": 4096, "measured_at": None})
    monkeypatch.setattr(cli, "engine_status", lambda _backend: (True, "llama.cpp"))
    monkeypatch.setattr(cli, "get_backend", lambda _backend: types.SimpleNamespace(
        generate=generate))
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: False))


def test_run_tracks_only_generate_and_cleans_after_refusal(
        make_console, monkeypatch, activity_registry):
    def generate(*_a, **_k):
        _assert_activity("running", "org/model")
        return {"refused": True, "reason": "safe refusal"}

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
    monkeypatch.setattr(cli, "engine_status", lambda _backend: (True, "llama.cpp"))
    monkeypatch.setattr(cli, "get_backend", lambda _backend: backend)
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "machine")
    monkeypatch.setattr(cli.catalog, "remember", lambda *_a: None)


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


def _wire_benchmark(monkeypatch, backend):
    monkeypatch.setattr(cli.engines, "resolve", lambda _engine: "cpu")
    monkeypatch.setitem(cli.engines.ENGINES, "cpu", {
        **cli.engines.ENGINES["cpu"], "backend": "cpu"})
    monkeypatch.setattr(cli, "get_backend", lambda _backend: backend)
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "machine")
    monkeypatch.setattr(cli.db, "get_characterization", lambda *_a: {
        "safe_context": 4096, "measured_at": None})
    monkeypatch.setattr(cli.benchmark, "load_probe", lambda _use_case: [{"answer": "a"}])
    monkeypatch.setattr(cli.benchmark, "prompt_for", lambda *_a: "prompt")
    monkeypatch.setattr(cli.benchmark, "score_probe_set", lambda *_a: 1.0)
    monkeypatch.setattr(cli.db, "get_model", lambda *_a: None)
    monkeypatch.setattr(cli.db, "save_benchmark_result", lambda *_a, **_k: None)
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
        return {"results": [{"prompt_index": 0, "completion": "a"}]}

    backend = types.SimpleNamespace(
        calibration_model_cached=lambda _m: False,
        download_calibration_model=download,
        benchmark=run,
    )
    _wire_benchmark(monkeypatch, backend)
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda _m: None)
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: None)
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
        download_calibration_model=lambda *_a, **_k: pytest.fail("download called"),
        benchmark=lambda *_a, **_k: pytest.fail("benchmark called"),
    )
    _wire_benchmark(monkeypatch, backend)
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda _m: 10.0)
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
        download_calibration_model=download,
        benchmark=lambda *_a, **_k: pytest.fail("benchmark called"),
    )
    _wire_benchmark(monkeypatch, backend)
    monkeypatch.setattr(cli.acquire, "repo_size_gb", lambda _m: None)
    monkeypatch.setattr(cli.acquire, "free_disk_gb", lambda: None)
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


@pytest.mark.parametrize("raised", [RuntimeError("boom"), KeyboardInterrupt(), SystemExit(5)])
def test_benchmark_activity_cleans_up_on_operational_and_base_exceptions(
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
    c, _ = make_console()
    with pytest.raises(type(raised)):
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


def test_mlx_serve_tracks_after_ready_handshake_through_wait(
        make_console, monkeypatch, activity_registry):
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "machine")
    monkeypatch.setattr(cli.db, "get_characterization", lambda *_a: {
        "safe_context": 4096, "measured_at": None, "points": []})
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
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "machine")
    monkeypatch.setattr(cli.db, "get_characterization", lambda *_a: {
        "safe_context": 4096, "measured_at": None, "points": []})
    monkeypatch.setattr(cli, "_free_port", lambda: 1234)
    monkeypatch.setattr(cli, "get_backend", lambda _backend: types.SimpleNamespace(
        serve=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("not ready"))))
    monkeypatch.setattr(cli.activity, "track", lambda *_a, **_k: pytest.fail("track called"))
    c, _ = make_console()
    assert cli._render_serve_mlx(c, "org/model", engine_key="mlx", assume_yes=True) == 1


@pytest.mark.parametrize("raised", [RuntimeError("boom"), KeyboardInterrupt(), SystemExit(3)])
def test_mlx_serve_wait_cleanup_covers_all_exit_paths(
        make_console, monkeypatch, activity_registry, raised):
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "machine")
    monkeypatch.setattr(cli.db, "get_characterization", lambda *_a: {
        "safe_context": 4096, "measured_at": None, "points": []})
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

    monkeypatch.setattr(cli.profile, "machine_key", lambda: "machine")
    monkeypatch.setattr(cli.db, "get_characterization", lambda *_a: {
        "safe_context": 4096, "measured_at": None, "points": []})
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
    monkeypatch.setattr(cli.ollama, "version", lambda: "1.0")
    monkeypatch.setattr(cli.ollama, "tags", lambda: ["base:model"])
    monkeypatch.setattr(cli.ollama, "base_url", lambda: "http://127.0.0.1:11434")
    monkeypatch.setattr(cli.profile, "machine_key", lambda: "machine")
    monkeypatch.setattr(cli.db, "get_characterization", lambda *_a: {
        "safe_context": 4096, "measured_at": None})
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: isatty))


def test_ollama_serve_temporary_activity_hands_off_to_persistent_without_overlap(
        make_console, monkeypatch, activity_registry):
    _wire_ollama_serve(monkeypatch)
    calls = []

    def create(*_a, **_k):
        _assert_activity("serving", "base:model")
        calls.append("create")
        return True

    def load(*_a, **_k):
        _assert_activity("serving", "base:model")
        calls.append("load")
        return {"done": True}

    def verify(*_a, **_k):
        _assert_activity("serving", "base:model")
        calls.append("verify")
        return [{"name": "base-model-ara:latest", "context_length": 4096,
                 "size": 10, "size_vram": 10}]

    real_record = activity.record_ollama_serving

    def record(**fields):
        assert activity.snapshot() == []
        calls.append("record")
        return real_record(**fields)

    monkeypatch.setattr(cli.ollama, "create", create)
    monkeypatch.setattr(cli.ollama, "load", load)
    monkeypatch.setattr(cli.ollama, "ps", verify)
    monkeypatch.setattr(cli.activity, "record_ollama_serving", record)
    c, _ = make_console()
    assert cli.render_serve(c, "base:model", ctx=4096) == 0
    assert calls == ["create", "load", "verify", "record"]

    monkeypatch.setattr(cli.ollama, "ps", lambda: [
        {"name": "base-model-ara:latest", "context_length": 4096}])
    found = activity.snapshot()
    assert len(found) == 1 and found[0].runtime == "ollama"
    assert not list(activity_registry.glob("*.json"))


def test_ollama_reserve_same_live_identity_never_duplicates_status(
        make_console, monkeypatch, activity_registry):
    _wire_ollama_serve(monkeypatch)
    activity.record_ollama_serving(
        served_name="base-model-ara", model="base:model", context=4096,
        endpoint="http://127.0.0.1:11434", started_at=1.0)
    loaded = [{"name": "base-model-ara:latest", "context_length": 4096,
               "size": 10, "size_vram": 10}]
    monkeypatch.setattr(cli.ollama, "ps", lambda: loaded)

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


def test_ollama_takeover_same_served_identity_for_new_base_keeps_transient_activity(
        make_console, monkeypatch, activity_registry):
    _wire_ollama_serve(monkeypatch)
    activity.record_ollama_serving(
        served_name="shared", model="org/old", context=4096,
        endpoint="http://127.0.0.1:11434", started_at=1.0)
    loaded = [{"name": "shared:latest", "context_length": 4096,
               "size": 10, "size_vram": 10}]
    monkeypatch.setattr(cli.ollama, "ps", lambda: loaded)

    def create(*_a, **_k):
        assert [(item.kind, item.model, item.runtime) for item in activity.snapshot()] == [
            ("serving", "org/old", "ollama"),
            ("serving", "base:model", None),
        ]
        return True

    monkeypatch.setattr(cli.ollama, "create", create)
    monkeypatch.setattr(cli.ollama, "load", lambda *_a, **_k: {"done": True})
    c, _ = make_console()
    assert cli.render_serve(c, "base:model", ctx=4096, name="shared") == 0


def test_ollama_serve_declined_consent_never_claims_activity(
        make_console, monkeypatch, activity_registry):
    _wire_ollama_serve(monkeypatch, isatty=True)
    monkeypatch.setattr(cli, "_confirm", lambda _question: False)
    monkeypatch.setattr(cli.ollama, "create", lambda *_a: pytest.fail("create called"))
    monkeypatch.setattr(cli.activity, "track", lambda *_a, **_k: pytest.fail("track called"))
    c, _ = make_console()
    assert cli.render_serve(c, "base:model", ctx=4096) == 0
    assert activity.snapshot() == []


def test_ollama_manifest_failure_unloads_untrackable_service(
        make_console, monkeypatch, activity_registry):
    _wire_ollama_serve(monkeypatch)
    monkeypatch.setattr(cli.ollama, "create", lambda *_a: True)
    unloading = []
    deleted = []
    monkeypatch.setattr(cli.ollama, "load",
                        lambda name, keep_alive=-1:
                        unloading.append(name) or {} if keep_alive == 0 else {"done": True})
    monkeypatch.setattr(cli.ollama, "ps", lambda: [] if unloading else [
        {"name": "base-model-ara:latest", "context_length": 4096,
         "size": 10, "size_vram": 10}])
    monkeypatch.setattr(cli.ollama, "delete", lambda name: deleted.append(name) or True)
    monkeypatch.setattr(cli.activity, "record_ollama_serving",
                        lambda **_fields: (_ for _ in ()).throw(OSError("disk full")))
    c, buf = make_console()
    assert cli.render_serve(c, "base:model", ctx=4096) == 1
    assert "ownership could not be recorded" in buf.getvalue()
    assert "unloaded" in buf.getvalue()
    assert unloading == ["base-model-ara"] and deleted == ["base-model-ara"]
    assert activity.snapshot() == []


def test_ollama_manifest_validation_failure_is_honest_json_not_raw_exception(
        make_console, monkeypatch, activity_registry, capsys):
    _wire_ollama_serve(monkeypatch)
    monkeypatch.setattr(cli.ollama, "create", lambda *_a: True)
    unloading = []
    monkeypatch.setattr(cli.ollama, "load",
                        lambda name, keep_alive=-1:
                        unloading.append(name) or {} if keep_alive == 0 else {"done": True})
    monkeypatch.setattr(cli.ollama, "ps", lambda: [] if unloading else [
        {"name": "base-model-ara:latest", "context_length": 4096,
         "size": 10, "size_vram": 10}])
    monkeypatch.setattr(cli.ollama, "delete", lambda _name: True)
    monkeypatch.setattr(cli.activity, "record_ollama_serving",
                        lambda **_fields: (_ for _ in ()).throw(ValueError("invalid identity")))
    c, _ = make_console()
    assert cli.render_serve(c, "base:model", ctx=4096, as_json=True) == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": "base-model-ara loaded at 4096 ctx, but ARA ownership could not be recorded: "
                 "invalid identity; unloaded the untracked service"}
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
