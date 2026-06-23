# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""acquire.py — backend-neutral HF downloads + the pre-download disk check."""
from __future__ import annotations

import types

import huggingface_hub
import huggingface_hub.utils as hf_utils

from ara import acquire


# --------------------------------------------------------------------------- #
# repo_size_gb
# --------------------------------------------------------------------------- #
def _fake_api(siblings):
    class FakeApi:
        def model_info(self, repo_id, files_metadata=False):
            return types.SimpleNamespace(siblings=siblings)
    return lambda: FakeApi()


def test_repo_size_sums_siblings(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "HfApi", _fake_api([
        types.SimpleNamespace(size=2_000_000_000),
        types.SimpleNamespace(size=3_000_000_000),
    ]))
    assert acquire.repo_size_gb("org/repo") == 5.0  # decimal GB


def test_repo_size_ignores_missing_sizes(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "HfApi", _fake_api([
        types.SimpleNamespace(size=1_500_000_000),
        types.SimpleNamespace(size=None),
    ]))
    assert acquire.repo_size_gb("org/repo") == 1.5


def test_repo_size_none_when_empty(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "HfApi", _fake_api([]))
    assert acquire.repo_size_gb("org/repo") is None


def test_repo_size_none_on_api_error(monkeypatch):
    class BoomApi:
        def model_info(self, *a, **k):
            raise RuntimeError("offline")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: BoomApi())
    assert acquire.repo_size_gb("org/repo") is None


# --------------------------------------------------------------------------- #
# free_disk_gb
# --------------------------------------------------------------------------- #
def test_free_disk_gb(monkeypatch):
    monkeypatch.setattr("shutil.disk_usage", lambda p: types.SimpleNamespace(free=100_000_000_000))
    assert acquire.free_disk_gb() == 100.0


def test_free_disk_gb_none_on_error(monkeypatch):
    def boom(p):
        raise OSError("no volume")
    monkeypatch.setattr("shutil.disk_usage", boom)
    assert acquire.free_disk_gb() is None


# --------------------------------------------------------------------------- #
# download (silences HF progress bars, restores prior setting)
# --------------------------------------------------------------------------- #
def test_download_calls_snapshot_and_restores_bars(monkeypatch):
    calls, order = [], []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id: calls.append(repo_id))
    monkeypatch.setattr(hf_utils, "are_progress_bars_disabled", lambda: False)
    monkeypatch.setattr(hf_utils, "disable_progress_bars", lambda: order.append("disable"))
    monkeypatch.setattr(hf_utils, "enable_progress_bars", lambda: order.append("enable"))

    acquire.download("org/repo")

    assert calls == ["org/repo"]
    assert order == ["disable", "enable"]  # disabled for the download, then restored


def test_download_leaves_bars_disabled_if_already_disabled(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id: None)
    monkeypatch.setattr(hf_utils, "are_progress_bars_disabled", lambda: True)
    monkeypatch.setattr(hf_utils, "disable_progress_bars", lambda: None)
    enabled = []
    monkeypatch.setattr(hf_utils, "enable_progress_bars", lambda: enabled.append(True))

    acquire.download("org/repo")

    assert enabled == []  # they were already off → not re-enabled


def test_download_restores_bars_even_on_error(monkeypatch):
    monkeypatch.setattr(hf_utils, "are_progress_bars_disabled", lambda: False)
    monkeypatch.setattr(hf_utils, "disable_progress_bars", lambda: None)
    enabled = []
    monkeypatch.setattr(hf_utils, "enable_progress_bars", lambda: enabled.append(True))

    def boom(repo_id):
        raise RuntimeError("network died")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)

    try:
        acquire.download("org/repo")
    except RuntimeError:
        pass
    assert enabled == [True]  # finally-block restored bars despite the error
