# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Backend selection and the cheap engine-presence check."""
from __future__ import annotations

import sys

import pytest

from ara import registry


def test_get_backend_returns_apple_module(set_platform):
    set_platform("Darwin", "arm64")
    mod = registry.get_backend()
    assert mod.__name__ == "ara.backends.apple"


def test_get_backend_returns_cuda_module(set_platform, monkeypatch):
    set_platform("Linux", "x86_64")
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/nvidia-smi" if n == "nvidia-smi" else None)
    mod = registry.get_backend()
    assert mod.__name__ == "ara.backends.cuda"


def test_get_backend_falls_back_to_cpu(set_platform, monkeypatch):
    # Non-Apple, non-CUDA → backend_name() == "cpu", the universal fallback adapter.
    set_platform("Linux", "x86_64")
    monkeypatch.setattr("shutil.which", lambda n: None)   # no nvidia-smi
    mod = registry.get_backend()
    assert mod.__name__ == "ara.backends.cpu"


def test_engine_status_apple_reports_mlx(set_platform, monkeypatch):
    set_platform("Darwin", "arm64")
    monkeypatch.setattr(registry.engines, "is_installed", lambda k: True)
    installed, name = registry.engine_status()
    assert installed is True and name == "MLX engine"


def test_engine_status_apple_missing_engine(set_platform, monkeypatch):
    set_platform("Darwin", "arm64")
    monkeypatch.setattr(registry.engines, "is_installed", lambda k: False)
    installed, name = registry.engine_status()
    assert installed is False and name == "MLX engine"


def test_engine_status_cpu_fallback(set_platform, monkeypatch):
    set_platform("Linux", "x86_64")
    monkeypatch.setattr(registry.engines.engine_env, "exists", lambda name: False)
    # Suppress nvidia-smi so backend_name() resolves to cpu on any host (including a real
    # NVIDIA box), matching the sibling test test_get_backend_falls_back_to_cpu.
    monkeypatch.setattr("shutil.which", lambda n: None)
    installed, name = registry.engine_status()
    assert installed is False and name == "llama.cpp"   # the cpu engine's display name


def test_engine_status_does_not_import_native_mlx_package(set_platform, monkeypatch):
    set_platform("Darwin", "arm64")
    # presence is an env-existence check (no find_spec, no import of the engine)
    monkeypatch.setattr(registry.engines.engine_env, "exists", lambda name: True)
    monkeypatch.setattr(
        registry.engines.engine_env,
        "stamped_version",
        lambda name: registry.engines._ara_version(),
    )
    registry.engine_status()
    assert "ara_engine_mlx" not in sys.modules


def test_engine_status_reports_present_env_unready_when_schema_is_missing(
        set_platform, monkeypatch):
    set_platform("Darwin", "arm64")
    monkeypatch.setitem(
        registry.engines.ENGINES,
        "mlx",
        {**registry.engines.ENGINES["mlx"], "env_schema": "mlx-worker-v2"},
    )
    monkeypatch.setattr(registry.engines.engine_env, "exists", lambda name: True)
    monkeypatch.setattr(
        registry.engines.engine_env,
        "stamped_version",
        lambda name: registry.engines._ara_version(),
    )
    monkeypatch.setattr(registry.engines.engine_env, "stamped_schema", lambda name: None)

    installed, name = registry.engine_status()

    assert installed is False and name == "MLX engine"


# ---------------------------------------------------------------------------
# resolve_engine
# ---------------------------------------------------------------------------

def test_resolve_engine_none_on_cuda(monkeypatch):
    """resolve_engine(None) on a cuda machine returns the cuda selection."""
    monkeypatch.setattr(registry.detect, "backend_name", lambda: "cuda")
    sel = registry.resolve_engine(None)
    assert sel == registry.EngineSelection("cuda", "cuda", "ara-engine-cuda")


def test_resolve_engine_auto_identical_to_none(monkeypatch):
    """resolve_engine('auto') is identical to resolve_engine(None)."""
    monkeypatch.setattr(registry.detect, "backend_name", lambda: "cuda")
    sel = registry.resolve_engine("auto")
    assert sel == registry.EngineSelection("cuda", "cuda", "ara-engine-cuda")


def test_resolve_engine_cpu_explicit(monkeypatch):
    """resolve_engine('cpu') → cpu/cpu/llama.cpp regardless of detected hardware."""
    monkeypatch.setattr(registry.detect, "backend_name", lambda: "cuda")
    sel = registry.resolve_engine("cpu")
    assert sel == registry.EngineSelection("cpu", "cpu", "llama.cpp")


def test_resolve_engine_cuda_explicit(monkeypatch):
    """resolve_engine('cuda') returns the cuda/cuda selection."""
    monkeypatch.setattr(registry.detect, "backend_name", lambda: "cpu")
    sel = registry.resolve_engine("cuda")
    assert sel == registry.EngineSelection("cuda", "cuda", "ara-engine-cuda")


def test_resolve_engine_mlx_explicit(monkeypatch):
    """resolve_engine('mlx') returns the mlx/apple selection."""
    monkeypatch.setattr(registry.detect, "backend_name", lambda: "cpu")
    sel = registry.resolve_engine("mlx")
    assert sel == registry.EngineSelection("apple", "mlx", "ara-engine-mlx")


def test_resolve_engine_bogus_raises(monkeypatch):
    """resolve_engine with an unknown engine name raises UnknownEngine."""
    monkeypatch.setattr(registry.detect, "backend_name", lambda: "cpu")
    with pytest.raises(registry.UnknownEngine):
        registry.resolve_engine("bogus")


@pytest.mark.parametrize(
    ("backend", "engine_key", "reason"),
    [
        ("apple", "mlx", "the machine matched the Apple Silicon backend"),
        ("cuda", "cuda", "the machine matched the NVIDIA CUDA backend"),
        ("cpu", "cpu",
         "no supported accelerator was detected, so ARA used the portable CPU fallback"),
    ],
)
def test_automatic_engine_selection_record_explains_captured_backend(
        monkeypatch, backend, engine_key, reason):
    monkeypatch.setattr(registry.detect, "backend_name", lambda: backend)

    selection = registry.resolve_engine("auto")
    record = registry.engine_selection_record("auto", selection)

    assert record.as_dict() == {
        "requested": "auto",
        "resolved_engine": engine_key,
        "backend": backend,
        "mode": "automatic",
        "reason": reason,
    }


def test_explicit_engine_selection_record_identifies_user_choice(monkeypatch):
    monkeypatch.setattr(registry.detect, "backend_name", lambda: "cuda")

    selection = registry.resolve_engine("cpu")
    record = registry.engine_selection_record("cpu", selection)

    assert record.as_dict() == {
        "requested": "cpu",
        "resolved_engine": "cpu",
        "backend": "cpu",
        "mode": "explicit",
        "reason": "the user selected --engine cpu",
    }


def test_engine_selection_record_does_not_reinspect_changed_machine_state(monkeypatch):
    backends = iter(("apple", "cuda"))
    monkeypatch.setattr(registry.detect, "backend_name", lambda: next(backends))

    selection = registry.resolve_engine(None)
    record = registry.engine_selection_record(None, selection)

    assert record.resolved_engine == "mlx"
    assert record.backend == "apple"
    assert "Apple Silicon backend" in record.reason
    assert registry.detect.backend_name() == "cuda"
