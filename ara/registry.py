"""Map detected hardware to a backend, importing only the chosen one.

The whole independence/leanness trick lives here: ``get_backend`` lazy-imports
exactly one adapter, so a non-Apple machine never imports the Apple adapter and
never touches MLX — even if it somehow got installed.
"""
from __future__ import annotations

import importlib

from ara import engines
from ara.detect import backend_name


def get_backend():
    """Lazy-import and return the backend module for this machine."""
    name = backend_name()
    return importlib.import_module(f"ara.backends.{name}")


def engine_status() -> tuple[bool, str]:
    """Is the active backend's engine installed? Cheap — no import of it.

    Returns ``(installed, engine_name)``, both sourced from the engine table so there's one
    place that maps hardware → engine. Presence is an isolated-env existence check, so it never
    imports the engine. Every backend (apple/cuda/cpu) has an engine entry, so the lookup always
    resolves.
    """
    key = engines.for_backend(backend_name())
    return engines.is_installed(key), engines.ENGINES[key]["package"]
