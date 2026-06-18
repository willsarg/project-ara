"""Map detected hardware to a backend, importing only the chosen one.

The whole independence/leanness trick lives here: ``get_backend`` lazy-imports
exactly one adapter, so a non-Apple machine never imports the Apple adapter and
never touches MLX — even if it somehow got installed.
"""
from __future__ import annotations

import importlib
from importlib.util import find_spec

from ara.detect import backend_name


def get_backend():
    """Lazy-import and return the backend module for this machine."""
    name = backend_name()
    return importlib.import_module(f"ara.backends.{name}")


def engine_status() -> tuple[bool, str]:
    """Is the active backend's engine installed? Cheap — no import of it.

    Returns ``(installed, engine_name)``. Uses ``find_spec`` so we can report
    the engine without paying the cost of importing it (and pulling MLX).
    """
    name = backend_name()
    if name == "apple":
        return find_spec("wmx_suite") is not None, "wmx-suite"
    return False, name
