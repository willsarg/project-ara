# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Typed validation for the no-load characterization preflight boundary."""
from __future__ import annotations

import math
from dataclasses import dataclass


class PreflightProtocolError(ValueError):
    """An engine emitted a preflight payload that is unsafe to use."""


@dataclass(frozen=True)
class PreflightError:
    """A valid engine refusal before characterization measurement begins."""

    error: str


@dataclass(frozen=True)
class Estimate:
    """The validated scalar contract consumed by the shared ramp driver."""

    base_gb: float
    ref_baseline_gb: float
    slope_gb_per_k: float
    budget_gb: float
    max_context: int | None
    n_layers: int | None = None
    fit_layers: int | None = None
    vram_budget_gb: float | None = None
    ram_budget_gb: float | None = None


_CORE_FIELDS = {
    "base_gb",
    "ref_baseline_gb",
    "slope_gb_per_k",
    "budget_gb",
    "max_context",
}
_CUDA_GGUF_FIELDS = {
    "n_layers",
    "fit_layers",
    "vram_budget_gb",
    "ram_budget_gb",
}


def _number(payload: dict, field: str) -> float:
    value = payload[field]
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0):
        raise PreflightProtocolError(
            f"preflight '{field}' must be a finite non-negative number")
    return float(value)


def _integer(payload: dict, field: str, *, positive: bool) -> int:
    value = payload[field]
    minimum = 1 if positive else 0
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "positive " if positive else "non-negative "
        raise PreflightProtocolError(
            f"preflight '{field}' must be a {qualifier}integer")
    return value


def parse(payload: object) -> Estimate | PreflightError:
    """Parse one engine preflight response, rejecting it before any model probe."""
    if not isinstance(payload, dict):
        raise PreflightProtocolError("preflight payload must be an object")

    fields = set(payload)
    if "error" in fields:
        unexpected = fields - {"error"}
        if unexpected:
            field = sorted(unexpected)[0]
            raise PreflightProtocolError(
                f"preflight error payload has unexpected field '{field}'")
        error = payload["error"]
        if not isinstance(error, str) or not error.strip():
            raise PreflightProtocolError(
                "preflight 'error' must be a non-empty string")
        return PreflightError(error)

    missing = _CORE_FIELDS - fields
    if missing:
        field = sorted(missing)[0]
        raise PreflightProtocolError(
            f"preflight payload missing required field '{field}'")
    unknown = fields - _CORE_FIELDS - _CUDA_GGUF_FIELDS
    if unknown:
        field = sorted(unknown)[0]
        raise PreflightProtocolError(
            f"preflight payload has unknown field '{field}'")

    extension = fields & _CUDA_GGUF_FIELDS
    if extension and extension != _CUDA_GGUF_FIELDS:
        missing_extension = sorted(_CUDA_GGUF_FIELDS - extension)[0]
        raise PreflightProtocolError(
            f"preflight CUDA-GGUF payload missing required field '{missing_extension}'")

    base_gb = _number(payload, "base_gb")
    ref_baseline_gb = _number(payload, "ref_baseline_gb")
    slope_gb_per_k = _number(payload, "slope_gb_per_k")
    budget_gb = _number(payload, "budget_gb")
    if ref_baseline_gb > base_gb:
        raise PreflightProtocolError(
            "preflight 'ref_baseline_gb' cannot exceed 'base_gb'")

    max_context = payload["max_context"]
    if max_context is not None:
        max_context = _integer(payload, "max_context", positive=True)

    n_layers = fit_layers = None
    vram_budget_gb = ram_budget_gb = None
    if extension:
        n_layers = _integer(payload, "n_layers", positive=True)
        fit_layers = _integer(payload, "fit_layers", positive=False)
        if fit_layers > n_layers:
            raise PreflightProtocolError(
                "preflight 'fit_layers' cannot exceed 'n_layers'")
        vram_budget_gb = _number(payload, "vram_budget_gb")
        ram_budget_gb = _number(payload, "ram_budget_gb")
        if not math.isclose(ram_budget_gb, budget_gb, rel_tol=0.0, abs_tol=1e-9):
            raise PreflightProtocolError(
                "preflight 'ram_budget_gb' must equal the driver's 'budget_gb'")

    return Estimate(
        base_gb=base_gb,
        ref_baseline_gb=ref_baseline_gb,
        slope_gb_per_k=slope_gb_per_k,
        budget_gb=budget_gb,
        max_context=max_context,
        n_layers=n_layers,
        fit_layers=fit_layers,
        vram_budget_gb=vram_budget_gb,
        ram_budget_gb=ram_budget_gb,
    )
