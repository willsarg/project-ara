# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Architectural invariant: the core stays wmx-free until a calibration call runs.

Importing any ``ara`` module — including the Apple backend adapter — must NOT pull
``wmx_suite`` into ``sys.modules``. The heavy engine only loads inside the functions
that actually measure. Verified in a fresh interpreter so no other test's fixtures
can mask the result.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest


def _run_snippet(code: str) -> str:
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_importing_ara_does_not_load_wmx_suite():
    out = _run_snippet(
        "import sys\n"
        "import ara, ara.detect, ara.status, ara.cli, ara.registry, ara.acquire, ara.ui\n"
        "import ara.backends.apple\n"
        "print('wmx' if 'wmx_suite' in sys.modules else 'clean')\n"
    )
    assert out == "clean"


def test_registry_get_backend_does_not_load_wmx_suite():
    # Resolving + importing the Apple adapter still must not import the engine.
    out = _run_snippet(
        "import sys\n"
        "from ara.registry import get_backend\n"
        "mod = get_backend()\n"  # apple on this host
        "print(mod.__name__, 'wmx' if 'wmx_suite' in sys.modules else 'clean')\n"
    )
    assert out.endswith("clean")


def test_wmx_suite_is_genuinely_absent():
    # The suite is designed for a wmx-free env (the worktree, where ../wmx-suite
    # doesn't resolve). On a full dev checkout the engine IS installed and the
    # lazy-import invariant above still holds — so skip rather than fail here.
    if importlib.util.find_spec("wmx_suite") is not None:
        pytest.skip("wmx_suite is installed in this env (full dev checkout); "
                    "the lazy-import invariant is still verified by the tests above")
    out = _run_snippet(
        "import importlib.util\n"
        "print(importlib.util.find_spec('wmx_suite') is None)\n"
    )
    assert out == "True"
