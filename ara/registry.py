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

    Returns ``(installed, engine_name)``, both sourced from the engine table so
    there's one place that maps hardware → engine. Uses ``find_spec`` under the
    hood, so it never imports the engine (and never pulls MLX).
    """
    name = backend_name()
    key = next((k for k, e in engines.ENGINES.items() if e["backend"] == name), None)
    if key is None:
        return False, name
    return engines.is_installed(key), engines.ENGINES[key]["package"]
