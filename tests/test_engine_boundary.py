# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Boundary guard: ARA core reaches nested engines ONLY through their headless
measurement/governance subprocess entrypoints and never imports them in-process. Pins the surface trimmed by
Spec 2026-07-05-refold-engines-to-adapter-surface so the suites can't silently re-bloat (a
re-vendor that drags the standalone-app scaffolding back, or a reference to an unwired worker).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

_ARA = Path(__file__).resolve().parent.parent / "ara"
_NESTED_ENGINE_DIRS = {"_vendor", "_engine_packages"}
_CORE_PY = [p for p in _ARA.rglob("*.py") if not _NESTED_ENGINE_DIRS.intersection(p.parts)]

# The legitimate headless-engine surface: the measurement/serve/govern `-m` entrypoints plus the
# subprocess entrypoints. NOT the removed standalone-app scaffolding
# (cli/cli_benchmarks/db/ui/launcher/views), and NOT the unwired non-LLM workers
# (kokoro*/embeddings*) — a core reference to any of those must fail here until a deliberate
# follow-on (e.g. wiring non-LLM characterize) extends this list. That failure is the signal.
_ALLOWED = {"device", "measure_one", "serve", "generate", "benchmark", "probe_worker"}


def _suite_refs(text: str) -> set[str]:
    return set(re.findall(r"(?:wmx|wcx)_suite\.([a-z_]+)", text))


def test_core_only_references_the_headless_engine_surface():
    offenders = {}
    for p in _CORE_PY:
        extra = _suite_refs(p.read_text(encoding="utf-8")) - _ALLOWED
        if extra:
            offenders[str(p.relative_to(_ARA.parent))] = sorted(extra)
    assert not offenders, (
        f"ARA core references vendored engine modules outside the allowed headless surface "
        f"{sorted(_ALLOWED)}. Either standalone-app scaffolding / an unwired non-LLM worker crept "
        f"back, or a deliberate follow-on must extend the allow-list: {offenders}")


def _nested_engine_imports(path: Path) -> list[str]:
    imports = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
            if node.module in {None, "ara"}:
                names.extend(alias.name for alias in node.names)
        for name in names:
            if _NESTED_ENGINE_DIRS.intersection(name.split(".")):
                imports.append(f"{path.relative_to(_ARA.parent)}:{node.lineno}: {name}")
    return imports


def test_core_never_imports_nested_engine_packages_in_process():
    offenders = []
    for p in _CORE_PY:
        offenders.extend(_nested_engine_imports(p))
    assert not offenders, (
        f"ara/ imports nested engine code in-process — this breaks the isolation contract "
        f"(ARA's process must never import an engine package): {offenders}")
