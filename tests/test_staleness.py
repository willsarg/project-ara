# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""staleness.py — is a stored characterization ceiling stale for the current model revision?

ARA's one seam over the (engine-agnostic, stdlib-only) vendored ``fit_is_stale`` — governance
consumers of a stored ``safe_context`` warn, never block, when the model's cache artifacts have
changed since it was measured (Rule #3: honest about a possibly-outdated measurement).

Slug: 2026-07-02-ara-ceiling-staleness
"""
from __future__ import annotations

from ara import staleness


def test_ceiling_is_stale_delegates_and_passes_args(monkeypatch):
    seen = {}

    def fake(model_id, measured_at):
        seen["args"] = (model_id, measured_at)
        return True

    monkeypatch.setattr(staleness, "fit_is_stale", fake)
    assert staleness.ceiling_is_stale("org/m", "2020-01-01T00:00:00+00:00") is True
    assert seen["args"] == ("org/m", "2020-01-01T00:00:00+00:00")


def test_ceiling_is_stale_passes_through_false(monkeypatch):
    # Conservative default from the vendored helper (no timestamp / uncached) → not stale.
    monkeypatch.setattr(staleness, "fit_is_stale", lambda model_id, measured_at: False)
    assert staleness.ceiling_is_stale("org/m", None) is False
