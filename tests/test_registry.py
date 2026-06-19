"""Backend selection and the cheap engine-presence check."""
from __future__ import annotations

import sys

from ara import registry


def test_get_backend_returns_apple_module(set_platform):
    set_platform("Darwin", "arm64")
    mod = registry.get_backend()
    assert mod.__name__ == "ara.backends.apple"


def test_get_backend_unsupported_raises_import(set_platform):
    # Non-Apple, non-CUDA → backend_name() == "unsupported", which has no module.
    set_platform("Linux", "x86_64")
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
    monkeypatch.setattr(registry.engines, "find_spec", lambda name: object())
    registry.engine_status()
    assert "wmx_suite" not in sys.modules
