"""Backend selection and the cheap engine-presence check."""
from __future__ import annotations

import sys

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


def test_get_backend_unsupported_raises_import(set_platform, monkeypatch):
    # Non-Apple, non-CUDA → backend_name() == "unsupported", which has no module.
    set_platform("Linux", "x86_64")
    monkeypatch.setattr("shutil.which", lambda n: None)   # no nvidia-smi
    try:
        registry.get_backend()
    except ModuleNotFoundError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("expected ModuleNotFoundError for unsupported backend")


def test_engine_status_apple_reports_wmx(set_platform, monkeypatch):
    set_platform("Darwin", "arm64")
    monkeypatch.setattr(registry.engines, "is_installed", lambda k: True)
    installed, name = registry.engine_status()
    assert installed is True and name == "wmx-suite"


def test_engine_status_apple_missing_engine(set_platform, monkeypatch):
    set_platform("Darwin", "arm64")
    monkeypatch.setattr(registry.engines, "is_installed", lambda k: False)
    installed, name = registry.engine_status()
    assert installed is False and name == "wmx-suite"


def test_engine_status_non_apple(set_platform):
    set_platform("Linux", "x86_64")
    installed, name = registry.engine_status()
    assert installed is False and name == "unsupported"


def test_engine_status_does_not_import_wmx(set_platform, monkeypatch):
    set_platform("Darwin", "arm64")
    # presence is an env-existence check (no find_spec, no import of the engine)
    monkeypatch.setattr(registry.engines.engine_env, "exists", lambda name: True)
    registry.engine_status()
    assert "wmx_suite" not in sys.modules
