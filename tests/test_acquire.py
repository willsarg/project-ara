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


# 2026-06-24-download-progress: progress=True enables bars; prior state always restored.
def test_download_progress_true_enables_bars_when_previously_disabled(monkeypatch):
    """progress=True: bars enabled during download; prior disabled state restored after.

    Slug: 2026-06-24-download-progress
    """
    order = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id: None)
    monkeypatch.setattr(hf_utils, "are_progress_bars_disabled", lambda: True)  # bars were off
    monkeypatch.setattr(hf_utils, "enable_progress_bars", lambda: order.append("enable"))
    monkeypatch.setattr(hf_utils, "disable_progress_bars", lambda: order.append("disable"))

    acquire.download("org/repo", progress=True)

    # enable for the call, then restore (was_disabled=True → disable in finally)
    assert order == ["enable", "disable"]


def test_download_progress_true_enables_bars_when_previously_enabled(monkeypatch):
    """progress=True: bars enabled during download; prior enabled state restored after.

    Slug: 2026-06-24-download-progress
    """
    order = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id: None)
    monkeypatch.setattr(hf_utils, "are_progress_bars_disabled", lambda: False)  # bars were on
    monkeypatch.setattr(hf_utils, "enable_progress_bars", lambda: order.append("enable"))
    monkeypatch.setattr(hf_utils, "disable_progress_bars", lambda: order.append("disable"))

    acquire.download("org/repo", progress=True)

    # enable for the call, restore (was_disabled=False → enable in finally)
    assert order == ["enable", "enable"]


def test_download_progress_false_disables_bars_when_previously_enabled(monkeypatch):
    """progress=False (default): bars disabled during download; prior enabled state restored.

    Slug: 2026-06-24-download-progress
    """
    order = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id: None)
    monkeypatch.setattr(hf_utils, "are_progress_bars_disabled", lambda: False)  # bars were on
    monkeypatch.setattr(hf_utils, "enable_progress_bars", lambda: order.append("enable"))
    monkeypatch.setattr(hf_utils, "disable_progress_bars", lambda: order.append("disable"))

    acquire.download("org/repo", progress=False)

    # disable for the call, restore (was_disabled=False → enable in finally)
    assert order == ["disable", "enable"]


def test_download_progress_false_restores_disabled_state(monkeypatch):
    """progress=False: bars disabled during download; prior disabled state kept after.

    Slug: 2026-06-24-download-progress
    """
    order = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id: None)
    monkeypatch.setattr(hf_utils, "are_progress_bars_disabled", lambda: True)  # bars were off
    monkeypatch.setattr(hf_utils, "enable_progress_bars", lambda: order.append("enable"))
    monkeypatch.setattr(hf_utils, "disable_progress_bars", lambda: order.append("disable"))

    acquire.download("org/repo", progress=False)

    # disable for the call, restore (was_disabled=True → disable in finally)
    assert order == ["disable", "disable"]


def test_download_progress_true_restores_prior_state_even_on_error(monkeypatch):
    """progress=True: prior disabled state restored even when snapshot_download raises.

    Slug: 2026-06-24-download-progress
    """
    order = []
    monkeypatch.setattr(hf_utils, "are_progress_bars_disabled", lambda: True)  # bars were off
    monkeypatch.setattr(hf_utils, "enable_progress_bars", lambda: order.append("enable"))
    monkeypatch.setattr(hf_utils, "disable_progress_bars", lambda: order.append("disable"))

    def boom(repo_id):
        raise RuntimeError("network died")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)

    try:
        acquire.download("org/repo", progress=True)
    except RuntimeError:
        pass
    # enable called at start; finally restores was_disabled=True → disable called
    assert order == ["enable", "disable"]


def test_valid_model_id_accepts_well_formed_repo_ids():
    # org/name and bare name, with the chars HF allows in a segment.
    for ok in ("mlx-community/Qwen3-0.6B-4bit", "meta-llama/Llama-3.2-1B",
               "SmolLM-135M", "org_1/model.v2", "a/b"):
        assert acquire.valid_model_id(ok) is True


def test_valid_model_id_rejects_flag_like_and_malformed():
    # The model becomes a worker argv positional, so flag-like / traversal / malformed ids must be
    # rejected before they leave ARA (the argv-injection sink). Empty, leading '-', '=', spaces,
    # path traversal, and extra path segments all fail.
    for bad in ("", "-rf", "--evil", "a=b", "a b", "../etc/passwd", "a/b/c",
                "/abs", ".hidden", "org/", "/name"):
        assert acquire.valid_model_id(bad) is False


# --------------------------------------------------------------------------- #
# classify_repo_error — maps HF exceptions to a small set of honest reasons
# --------------------------------------------------------------------------- #
def _fake_response(status_code=500):
    """Minimal fake requests.Response for constructing HF HTTP error instances."""
    class FakeResponse:
        headers = {}
        request = None
        def __init__(self, code):
            self.status_code = code
    return FakeResponse(status_code)


def test_classify_gated_repo_error():
    from huggingface_hub.errors import GatedRepoError
    exc = GatedRepoError("gated", response=_fake_response(403))
    assert acquire.classify_repo_error(exc) == "gated"


def test_classify_not_found_error():
    from huggingface_hub.errors import RepositoryNotFoundError
    exc = RepositoryNotFoundError("not found", response=_fake_response(404))
    assert acquire.classify_repo_error(exc) == "not_found"


def test_classify_auth_http_error():
    # An HTTP 401 that isn't a gated/not-found subclass → auth error.
    from huggingface_hub.errors import HfHubHTTPError
    exc = HfHubHTTPError("unauthorized", response=_fake_response(401))
    assert acquire.classify_repo_error(exc) == "auth"


def test_classify_offline_local_entry():
    from huggingface_hub.errors import LocalEntryNotFoundError
    exc = LocalEntryNotFoundError("not cached")
    assert acquire.classify_repo_error(exc) == "offline"


def test_classify_offline_mode_enabled():
    from huggingface_hub.errors import OfflineModeIsEnabled
    exc = OfflineModeIsEnabled("offline mode")
    assert acquire.classify_repo_error(exc) == "offline"


def test_classify_offline_connection_error():
    exc = ConnectionError("connection refused")
    assert acquire.classify_repo_error(exc) == "offline"


def test_classify_unknown_fallback():
    exc = RuntimeError("something weird")
    assert acquire.classify_repo_error(exc) == "unknown"


# --------------------------------------------------------------------------- #
# probe_repo — structured dict with size_gb + reason
# --------------------------------------------------------------------------- #
def test_probe_repo_success(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "HfApi", _fake_api([
        types.SimpleNamespace(size=2_000_000_000),
    ]))
    result = acquire.probe_repo("org/repo")
    assert result == {"size_gb": 2.0, "reason": None}


def test_probe_repo_empty_siblings(monkeypatch):
    # API succeeds but no files → size_gb is None, reason is None (it's reachable, just empty).
    monkeypatch.setattr(huggingface_hub, "HfApi", _fake_api([]))
    result = acquire.probe_repo("org/repo")
    assert result == {"size_gb": None, "reason": None}


def test_probe_repo_gated(monkeypatch):
    from huggingface_hub.errors import GatedRepoError
    exc = GatedRepoError("gated", response=_fake_response(403))
    class BoomApi:
        def model_info(self, *a, **k):
            raise exc
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: BoomApi())
    result = acquire.probe_repo("org/gated-model")
    assert result["size_gb"] is None
    assert result["reason"] == "gated"


def test_probe_repo_not_found(monkeypatch):
    from huggingface_hub.errors import RepositoryNotFoundError
    exc = RepositoryNotFoundError("not found", response=_fake_response(404))
    class BoomApi:
        def model_info(self, *a, **k):
            raise exc
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: BoomApi())
    result = acquire.probe_repo("org/missing")
    assert result == {"size_gb": None, "reason": "not_found"}


def test_probe_repo_offline(monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError
    exc = LocalEntryNotFoundError("not cached")
    class BoomApi:
        def model_info(self, *a, **k):
            raise exc
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: BoomApi())
    result = acquire.probe_repo("org/repo")
    assert result == {"size_gb": None, "reason": "offline"}


# --------------------------------------------------------------------------- #
# repo_size_gb still returns float|None (public contract preserved)
# --------------------------------------------------------------------------- #
def test_repo_size_still_returns_none_on_gated(monkeypatch):
    from huggingface_hub.errors import GatedRepoError
    exc = GatedRepoError("gated", response=_fake_response(403))
    class BoomApi:
        def model_info(self, *a, **k):
            raise exc
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: BoomApi())
    # Contract: repo_size_gb still returns None (not the reason) so existing callers don't break.
    assert acquire.repo_size_gb("org/gated") is None
