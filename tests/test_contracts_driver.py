# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The engine-agnostic thin-path driver.

This is the antidote to an Apple-shaped abstraction: one ``characterize`` that any
ramp-class backend drives by handing in its own ``preflight``/``measure`` callables and
schedule. The driver owns the methodology (error handling, the L2 post-check, ``ramp.run``,
result shaping) and knows nothing about MLX, llama.cpp, or which engine env it's talking to.

The tests drive it with plain callables — no engine env, no subprocess — to prove the
methodology is independent of any one engine.
"""
from __future__ import annotations

import pytest

from ara import catalog
from ara.contracts import driver


@pytest.fixture(autouse=True)
def _no_catalog_network(monkeypatch):
    """Keep the suite offline: characterize now calls catalog.describe; stub it out."""
    monkeypatch.setattr(catalog, "describe", lambda m: None)


def _est(**kw) -> dict:
    base = {"base_gb": 5.0, "slope_gb_per_k": 1.0, "budget_gb": 36.0,
            "max_context": None, "ref_baseline_gb": 0.0}
    base.update(kw)
    return base


def _linear(intercept: float, slope_per_k: float):
    return lambda model, ctx: {"context": ctx,
                               "mem_gb": intercept + slope_per_k * (ctx / 1000)}


def test_drives_ramp_and_shapes_result():
    # memory would allow ~31k, but the model's window is 16k → capped, window-bound
    est = _est(max_context=16000)
    r = driver.characterize("org/model", preflight=lambda m: est,
                            measure=_linear(5.0, 1.0),
                            schedule=[2000, 4000, 8000, 16000, 32000])
    assert r["model"] == "org/model"
    assert r["safe_context"] == 16000
    assert r["binding"] == "context_window"
    assert r["points"][0] == {"context": 2000, "mem_gb": 7.0}


def test_filters_schedule_above_model_window():
    est = _est(max_context=16000)
    seen: list[int] = []

    def measure(model, ctx):
        seen.append(ctx)
        return {"context": ctx, "mem_gb": 5.0 + ctx / 1000}

    driver.characterize("org/model", preflight=lambda m: est, measure=measure,
                        schedule=[2000, 4000, 8000, 16000, 32000, 65536])
    assert max(seen) <= 16000          # 32000/65536 never dispatched


def test_tiny_context_window_still_reports_window_ceiling():
    # A model whose trained window (2048) is below the 2nd schedule rung: only one standard
    # rung (2000) fits under it. The driver must still probe the window itself so it gets >=2
    # points and reports the model fits its whole window — not "couldn't fit a ceiling".
    est = _est(max_context=2048, slope_gb_per_k=0.1)
    seen: list[int] = []

    def measure(model, ctx):
        seen.append(ctx)
        return {"context": ctx, "mem_gb": 1.0 + 0.1 * (ctx / 1000)}

    r = driver.characterize("m", preflight=lambda m: est, measure=measure,
                            schedule=[2000, 4000, 8000, 16000])
    assert r["safe_context"] == 2048
    assert r["binding"] == "context_window"
    assert 2048 in seen                       # probed the window itself
    assert all(c <= 2048 for c in seen)       # never probed past it


def test_window_below_smallest_rung_gets_a_lower_anchor():
    # An even tinier window (1024) — below every standard rung. Still needs >=2 distinct probes.
    est = _est(max_context=1024, slope_gb_per_k=0.1)
    seen: list[int] = []

    def measure(model, ctx):
        seen.append(ctx)
        return {"context": ctx, "mem_gb": 1.0 + 0.1 * (ctx / 1000)}

    r = driver.characterize("m", preflight=lambda m: est, measure=measure,
                            schedule=[2000, 4000])
    assert r["safe_context"] == 1024 and r["binding"] == "context_window"
    assert len(set(seen)) >= 2 and all(c <= 1024 for c in seen)


def test_degenerate_window_never_probes_above_it():
    # Pathological window of 0 must never produce a probe above the window (the anchor that
    # guarantees ≥2 points is suppressed when it would exceed max_context).
    seen: list[int] = []
    est = _est(max_context=0)

    def measure(model, ctx):
        seen.append(ctx)
        return {"context": ctx, "mem_gb": 1.0}

    driver.characterize("m", preflight=lambda m: est, measure=measure, schedule=[2000, 4000])
    assert all(c <= 0 for c in seen)          # never probed above the model's window


def test_none_when_preflight_errors():
    r = driver.characterize("missing/model",
                            preflight=lambda m: {"error": "not in HF cache"},
                            measure=lambda m, c: {}, schedule=[2000])
    assert r == {"model": "missing/model", "safe_context": None, "points": [],
                 "error": "not in HF cache"}


def test_l2_stops_when_measured_reaches_budget():
    # L1 thinks it's safe (tiny slope), but the ACTUAL measurement is at/over budget
    est = _est(slope_gb_per_k=0.001)
    r = driver.characterize("org/model", preflight=lambda m: est,
                            measure=lambda m, ctx: {"context": ctx, "mem_gb": 40.0},
                            schedule=[2000, 4000])
    assert r["safe_context"] is None       # first rung over budget → <2 usable points


def test_threads_ref_baseline_into_ceiling():
    # delta fit: model base 5, slope 1; live OS baseline 8 → ceiling (36-8-5)/1 = 23k
    est = _est(base_gb=13.0, ref_baseline_gb=8.0)
    r = driver.characterize("m", preflight=lambda m: est, measure=_linear(5.0, 1.0),
                            schedule=[2000, 4000, 8000])
    assert r["safe_context"] == 22_999
    assert r["binding"] == "memory"


def test_engine_refusal_brackets_below_the_abort():
    # A refusal at 8000 is a hard wall: the driver bisects [4000, 8000) and reports a
    # confirmed-safe context strictly under it — never extrapolating the fit past the abort.
    est = _est()

    def measure(model, ctx):
        if ctx >= 8000:
            return {"context": ctx, "refused": True, "reason": "engine veto"}
        return {"context": ctx, "mem_gb": 5.0 + ctx / 1000}

    r = driver.characterize("m", preflight=lambda m: est, measure=measure,
                            schedule=[2000, 4000, 8000, 16000])
    assert 4000 <= r["safe_context"] < 8000
    assert r["binding"] == "memory"
    assert r["safe_context"] in {p["context"] for p in r["points"]}


# --------------------------------------------------------------------------- #
# decode_context — analytic decode ceiling returned from characterize
# --------------------------------------------------------------------------- #
_META_DICT = {"n_layers": 32, "kv_heads": 8, "head_dim": 128}


def test_decode_context_larger_than_safe_context_when_meta_available(monkeypatch):
    # Shallow linear ramp → prefill fit exists; meta provides kv dims; kv slope < prefill slope
    # so decode ceiling > prefill ceiling.
    from ara.contracts import driver as drv
    from ara import catalog
    monkeypatch.setattr(catalog, "describe", lambda m: dict(_META_DICT))
    est = _est(slope_gb_per_k=2.0)   # steep prefill slope to ensure kv slope is smaller
    r = drv.characterize("m", preflight=lambda m: est, measure=_linear(5.0, 2.0),
                         schedule=[2000, 4000, 8000])
    assert "decode_context" in r
    assert r["decode_context"] is not None
    assert isinstance(r["decode_context"], int)
    assert r["decode_context"] > r["safe_context"]


def test_decode_context_none_when_describe_returns_none(monkeypatch):
    from ara.contracts import driver as drv
    from ara import catalog
    monkeypatch.setattr(catalog, "describe", lambda m: None)
    est = _est()
    r = drv.characterize("m", preflight=lambda m: est, measure=_linear(5.0, 1.0),
                         schedule=[2000, 4000, 8000])
    assert r["decode_context"] is None


def test_decode_context_none_when_fit_is_none(monkeypatch):
    # Single safe point → res.fit is None → decode_context must be None
    from ara.contracts import driver as drv
    from ara import catalog
    monkeypatch.setattr(catalog, "describe", lambda m: dict(_META_DICT))
    # base 33+slope 1: 2k→35 < 36 (safe), 4k→37≥36 (L1 gates out), so only 1 point — no fit
    est = _est(base_gb=33.0, slope_gb_per_k=1.0)
    r = drv.characterize("m", preflight=lambda m: est, measure=_linear(33.0, 1.0),
                         schedule=[2000, 4000])
    assert r["safe_context"] == 2000
    assert r["decode_context"] is None


def test_decode_context_does_not_change_prefill_ceiling_or_binding(monkeypatch):
    # Confirm the L2 gate still keys off the measured peak; decode_context is additive only
    from ara.contracts import driver as drv
    from ara import catalog
    monkeypatch.setattr(catalog, "describe", lambda m: dict(_META_DICT))
    est = _est(max_context=16000)
    r = drv.characterize("m", preflight=lambda m: est, measure=_linear(5.0, 1.0),
                         schedule=[2000, 4000, 8000, 16000, 32000])
    assert r["safe_context"] == 16000
    assert r["binding"] == "context_window"
    assert "decode_context" in r


def test_decode_context_computed_in_abort_bisect_regime(monkeypatch):
    # I2: When a ramp rung aborts (e.g. at 8000), the driver bisects below the abort wall
    # and safe_context ends up < 8000.  decode_context is still computed from the fit over
    # the safe measured points — it is NOT suppressed by the abort.  This is intentional:
    # a prefill abort means the prefill transient blew the budget, which is exactly when
    # streaming (decode) headroom is largest.  The decode estimate uses the measured base +
    # conservative analytic KV slope and is unaffected by the prefill abort.
    # decode_context may legally exceed safe_context in this regime.
    from ara.contracts import driver as drv
    from ara import catalog
    monkeypatch.setattr(catalog, "describe", lambda m: dict(_META_DICT))

    # Shallow linear ramp so the fit exists; abort any ctx >= 8000 to trigger bisect.
    def measure(model, ctx):
        if ctx >= 8000:
            return {"context": ctx, "refused": True, "reason": "engine veto"}
        return {"context": ctx, "mem_gb": 5.0 + ctx / 1000}

    # steep prefill slope relative to KV slope → decode ceiling > prefill ceiling
    est = _est(slope_gb_per_k=2.0)
    r = drv.characterize("m", preflight=lambda m: est, measure=measure,
                         schedule=[2000, 4000, 8000, 16000])
    assert 4000 <= r["safe_context"] < 8000         # bisect confirmed below the abort wall
    assert r.get("decode_context") is not None       # decode is still computed, not suppressed
    assert isinstance(r["decode_context"], int)
    assert r["decode_context"] > 0                   # positive — a real estimate was produced
