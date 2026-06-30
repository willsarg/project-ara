# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Node bearer-token auth — generation, storage, and constant-time matching."""
from __future__ import annotations

import pytest

from ara.node import auth


@pytest.fixture(autouse=True)
def _node_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_NODE_DIR", str(tmp_path / "node"))


def test_load_token_is_none_when_absent():
    assert auth.load_token() is None


def test_ensure_token_creates_then_is_idempotent():
    tok = auth.ensure_token()
    assert tok and auth.load_token() == tok
    assert auth.ensure_token() == tok          # second call returns the same token, no regen


def test_rotate_token_replaces_it():
    first = auth.ensure_token()
    second = auth.rotate_token()
    assert second != first and auth.load_token() == second


def test_token_matches_accepts_correct_bearer():
    tok = auth.ensure_token()
    assert auth.token_matches(f"Bearer {tok}") is True
    assert auth.token_matches(f"bearer {tok}") is True       # scheme is case-insensitive


@pytest.mark.parametrize("header", [None, "", "Bearer wrong", "Token abc", "justgarbage", "Bearer "])
def test_token_matches_rejects_bad_headers(header):
    auth.ensure_token()
    assert auth.token_matches(header) is False


def test_token_matches_is_false_when_no_token_configured():
    assert auth.load_token() is None
    assert auth.token_matches("Bearer anything") is False
