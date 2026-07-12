# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Architectural invariant: the core stays engine-free until a worker call runs.

Importing any ``ara`` module — including the Apple backend adapter — must NOT pull
``ara_engine_mlx`` or ``ara_engine_cuda`` into ``sys.modules``. The heavy engines only
load in their isolated environments. Verified in a fresh interpreter so no other
test's fixtures can mask the result.
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


def test_importing_ara_does_not_load_native_engine_packages():
    out = _run_snippet(
        "import sys\n"
        "import ara, ara.detect, ara.status, ara.cli, ara.registry, ara.acquire, ara.ui\n"
        "import ara.backends.apple, ara.backends.cuda\n"
        "loaded = {'ara_engine_mlx', 'ara_engine_cuda'} & sys.modules.keys()\n"
        "print(','.join(sorted(loaded)) if loaded else 'clean')\n"
    )
    assert out == "clean"


def test_registry_get_backend_does_not_load_native_engine_packages():
    # Resolving + importing the Apple adapter still must not import the engine.
    out = _run_snippet(
        "import sys\n"
        "from ara.registry import get_backend\n"
        "mod = get_backend()\n"  # apple on this host
        "loaded = {'ara_engine_mlx', 'ara_engine_cuda'} & sys.modules.keys()\n"
        "print(mod.__name__, ','.join(sorted(loaded)) if loaded else 'clean')\n"
    )
    assert out.endswith("clean")


@pytest.mark.parametrize("package", ["ara_engine_mlx", "ara_engine_cuda"])
def test_native_engine_package_is_genuinely_absent(package):
    # Native engine packages are nested distribution sources, not packages in ARA's core env.
    if importlib.util.find_spec(package) is not None:
        pytest.fail(f"{package} unexpectedly resolves in ARA's core test environment")
    out = _run_snippet(
        "import importlib, importlib.util\n"
        f"package = {package!r}\n"
        "assert importlib.util.find_spec(package) is None\n"
        "try:\n"
        "    importlib.import_module(package)\n"
        "except ModuleNotFoundError:\n"
        "    print('absent')\n"
        "else:\n"
        "    raise AssertionError(f'{package} unexpectedly imported')\n"
    )
    assert out == "absent"
