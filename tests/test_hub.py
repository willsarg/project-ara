# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""hub.py — Hugging Face Hub search via the `hf` CLI (engine-agnostic)."""
from __future__ import annotations

import types

from ara import hub


def _proc(stdout="", returncode=0):
    return types.SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


def test_search_parses_results(monkeypatch):
    payload = '[{"id": "org/A", "downloads": 1000, "likes": 5}, {"id": "org/B"}]'
    seen = {}

    def fake_run(cmd, capture_output, text, timeout):
        seen["cmd"] = cmd
        return _proc(stdout=payload)

    monkeypatch.setattr(hub.subprocess, "run", fake_run)
    out = hub.search("smol", limit=10)
    assert out == [{"id": "org/A", "downloads": 1000, "likes": 5},
                   {"id": "org/B", "downloads": 0, "likes": 0}]
    assert seen["cmd"][:3] == ["hf", "models", "list"]
    assert "--limit" in seen["cmd"] and "10" in seen["cmd"]


def test_search_passes_author_filter(monkeypatch):
    seen = {}

    def fake_run(cmd, capture_output, text, timeout):
        seen["cmd"] = cmd
        return _proc(stdout="[]")

    monkeypatch.setattr(hub.subprocess, "run", fake_run)
    hub.search("q", author="mlx-community")
    assert "--author" in seen["cmd"] and "mlx-community" in seen["cmd"]


def test_search_none_when_hf_missing(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no hf")
    monkeypatch.setattr(hub.subprocess, "run", boom)
    assert hub.search("x") is None


def test_search_none_on_nonzero(monkeypatch):
    monkeypatch.setattr(hub.subprocess, "run", lambda *a, **k: _proc(returncode=1))
    assert hub.search("x") is None


def test_search_empty_on_bad_json(monkeypatch):
    monkeypatch.setattr(hub.subprocess, "run", lambda *a, **k: _proc(stdout="not json"))
    assert hub.search("x") == []
