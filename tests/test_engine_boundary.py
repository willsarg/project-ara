# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Boundary guard: ARA core reaches the vendored engines ONLY through their headless
measurement/governance entrypoints, and imports vendored code in-process only via the one
stdlib-only staleness helper. Pins the surface trimmed by
Spec 2026-07-05-refold-engines-to-adapter-surface so the suites can't silently re-bloat (a
re-vendor that drags the standalone-app scaffolding back, or a reference to an unwired worker).
"""
from __future__ import annotations

import re
from pathlib import Path

_ARA = Path(__file__).resolve().parent.parent / "ara"
_CORE_PY = [p for p in _ARA.rglob("*.py") if "_vendor" not in p.parts]

# The legitimate headless-engine surface: the measurement/serve/govern `-m` entrypoints plus the
# stdlib-only staleness helper module. NOT the removed standalone-app scaffolding
# (cli/cli_benchmarks/db/ui/launcher/views), and NOT the unwired non-LLM workers
# (kokoro*/embeddings*) — a core reference to any of those must fail here until a deliberate
# follow-on (e.g. wiring non-LLM characterize) extends this list. That failure is the signal.
_ALLOWED = {"device", "measure_one", "serve", "generate", "benchmark", "probe_worker", "models"}


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


def test_vendor_imported_in_process_only_by_staleness():
    offenders = []
    for p in _CORE_PY:
        if p.name == "staleness.py":
            continue
        for m in re.finditer(r"(?m)^\s*(?:from|import)\s+\S*_vendor\S*",
                             p.read_text(encoding="utf-8")):
            offenders.append(f"{p.relative_to(_ARA.parent)}: {m.group().strip()}")
    assert not offenders, (
        f"ara/ imports vendored engine code in-process outside ara/staleness.py — this breaks the "
        f"isolation contract (ARA's process must never import mlx/torch): {offenders}")
