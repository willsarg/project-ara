# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Map detected hardware to a backend, importing only the chosen one.

The whole independence/leanness trick lives here: ``get_backend`` lazy-imports
exactly one adapter, so a non-Apple machine never imports the Apple adapter and
never touches MLX — even if it somehow got installed.
"""
from __future__ import annotations

import importlib
import typing
from dataclasses import dataclass

from ara import detect, engines


class UnknownEngine(ValueError):
    """Raised when an explicit engine key is not in the engine catalog."""


class EngineSelection(typing.NamedTuple):
    """The resolved triple for a single engine selection: backend adapter, engine key, package."""
    backend: str
    engine_key: str
    package: str


@dataclass(frozen=True)
class EngineSelectionRecord:
    """Immutable explanation captured from the same selection that controls execution."""

    requested: str
    resolved_engine: str
    backend: str
    mode: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "requested": self.requested,
            "resolved_engine": self.resolved_engine,
            "backend": self.backend,
            "mode": self.mode,
            "reason": self.reason,
        }


_AUTOMATIC_REASONS = {
    "apple": "the machine matched the Apple Silicon backend",
    "cuda": "the machine matched the NVIDIA CUDA backend",
    "cpu": "no supported accelerator was detected, so ARA used the portable CPU fallback",
}


def engine_selection_record(
    requested: str | None,
    selection: EngineSelection,
    *,
    automatic_reason: str | None = None,
) -> EngineSelectionRecord:
    """Describe *selection* without re-reading hardware or resolving the engine again."""
    automatic = requested is None or requested == "auto"
    if automatic:
        reason = automatic_reason or _AUTOMATIC_REASONS.get(
            selection.backend,
            f"automatic resolution selected the {selection.backend} backend",
        )
        return EngineSelectionRecord(
            requested="auto",
            resolved_engine=selection.engine_key,
            backend=selection.backend,
            mode="automatic",
            reason=reason,
        )
    return EngineSelectionRecord(
        requested=selection.engine_key,
        resolved_engine=selection.engine_key,
        backend=selection.backend,
        mode="explicit",
        reason=f"the user selected --engine {selection.engine_key}",
    )


def resolve_engine(engine: str | None) -> EngineSelection:
    """Resolve an ``--engine`` value (or None / 'auto') to a canonical :class:`EngineSelection`.

    ``None`` and ``'auto'`` both defer to the detected hardware backend; an explicit engine key
    is looked up directly in the engine catalog. Raises :exc:`UnknownEngine` for any explicit
    value that isn't a known engine key.
    """
    if engine is None or engine == "auto":
        backend = detect.backend_name()
        engine_key = engines.for_backend(backend)
        package = engines.ENGINES[engine_key]["package"]
    else:
        engine_key = engines.resolve(engine)
        if engine_key is None or engine_key not in engines.ENGINES:
            raise UnknownEngine(engine)
        backend = engines.ENGINES[engine_key]["backend"]
        package = engines.ENGINES[engine_key]["package"]
    return EngineSelection(backend, engine_key, package)


def get_backend(backend: str | None = None):
    """Lazy-import and return a backend module — *backend* if given, else this machine's.

    An explicit *backend* lets a command target a non-detected engine (e.g. ``--engine cpu`` on
    a GPU box to measure the CPU fallback). Only the chosen adapter is imported, so the
    independence guarantee holds either way."""
    name = backend or detect.backend_name()
    return importlib.import_module(f"ara.backends.{name}")


def engine_status(backend: str | None = None) -> tuple[bool, str]:
    """Is a backend's engine installed? Cheap — no import of it. *backend* defaults to the
    detected one; pass it to check a specific (possibly non-detected) backend's engine.

    Returns ``(installed, engine_name)``, both sourced from the engine table so there's one
    place that maps hardware → engine. Presence is an isolated-env existence check, so it never
    imports the engine. Every backend (apple/cuda/cpu) has an engine entry, so the lookup always
    resolves.
    """
    key = engines.for_backend(backend or detect.backend_name())
    labels = {"mlx": "MLX engine", "cuda": "CUDA engine"}
    return engines.is_installed(key), labels.get(key, engines.ENGINES[key]["package"])
