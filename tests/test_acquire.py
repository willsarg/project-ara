# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""acquire.py — backend-neutral HF downloads + the pre-download disk check."""
from __future__ import annotations

import types
from pathlib import Path

import huggingface_hub
import huggingface_hub.utils as hf_utils
import pytest

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


def test_free_disk_gb_uses_hf_cache_volume(tmp_path, monkeypatch):
    custom = tmp_path / "cache-volume" / "hf-hub"
    seen = []
    monkeypatch.setenv("HF_HUB_CACHE", str(custom))
    monkeypatch.setattr("shutil.disk_usage",
                        lambda path: seen.append(Path(path)) or types.SimpleNamespace(
                            free=7_000_000_000))

    assert acquire.free_disk_gb() == 7.0
    assert seen == [tmp_path]


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
    order = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id: None)
    monkeypatch.setattr(hf_utils, "are_progress_bars_disabled", lambda: True)
    monkeypatch.setattr(hf_utils, "disable_progress_bars", lambda: order.append("disable"))
    monkeypatch.setattr(hf_utils, "enable_progress_bars", lambda: order.append("enable"))

    acquire.download("org/repo")

    # Asserting only "enable never called" would also pass if the whole finally restore were
    # deleted. Pin that the restore actually ran and chose the disable branch (prior state).
    assert order == ["disable", "disable"]  # silence for the download, then restore to disabled


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


def test_download_gguf_exact_selector_fetches_only_selected_file(monkeypatch):
    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, filename: calls.append((repo, filename)) or "/cache/model.gguf")
    monkeypatch.setattr(hf_utils, "are_progress_bars_disabled", lambda: False)
    monkeypatch.setattr(hf_utils, "disable_progress_bars", lambda: None)
    monkeypatch.setattr(hf_utils, "enable_progress_bars", lambda: None)

    assert acquire.download_gguf("org/repo:Q4/model.gguf") == "/cache/model.gguf"
    assert calls == [("org/repo", "Q4/model.gguf")]


def test_download_gguf_bare_selects_smallest_non_projector(monkeypatch):
    siblings = [
        types.SimpleNamespace(rfilename="mmproj-model.gguf", size=1),
        types.SimpleNamespace(rfilename="model-q8.gguf", size=800),
        types.SimpleNamespace(rfilename="model-q4.gguf", size=400),
    ]
    monkeypatch.setattr(huggingface_hub, "HfApi", _fake_api(siblings))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, filename: f"/cache/{repo}/{filename}")
    monkeypatch.setattr(hf_utils, "are_progress_bars_disabled", lambda: False)
    monkeypatch.setattr(hf_utils, "disable_progress_bars", lambda: None)
    monkeypatch.setattr(hf_utils, "enable_progress_bars", lambda: None)

    assert acquire.download_gguf("org/repo").endswith("model-q4.gguf")


def test_gguf_size_gb_matches_selected_remote_file(monkeypatch):
    siblings = [
        types.SimpleNamespace(rfilename="mmproj-model.gguf", size=1),
        types.SimpleNamespace(rfilename="model-q8.gguf", size=8_000_000_000),
        types.SimpleNamespace(rfilename="model-q4.gguf", size=4_000_000_000),
    ]
    monkeypatch.setattr(huggingface_hub, "HfApi", _fake_api(siblings))
    assert acquire.gguf_size_gb("org/repo") == 4.0
    assert acquire.gguf_size_gb("org/repo:model-q8.gguf") == 8.0


def test_gguf_helpers_cover_local_missing_and_invalid_inputs(tmp_path, monkeypatch):
    local = tmp_path / "model.gguf"
    local.write_bytes(b"weights")
    assert acquire.gguf_size_gb(str(local)) == 0.0
    assert acquire.download_gguf(str(local)) == str(local.resolve())

    monkeypatch.setattr(acquire.os.path, "getsize", lambda _path: (_ for _ in ()).throw(OSError()))
    assert acquire.gguf_size_gb(str(local)) is None
    with pytest.raises(ValueError, match="invalid GGUF"):
        acquire.download_gguf("../bad")


def test_gguf_remote_selection_refuses_missing_weight(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "HfApi", _fake_api([
        types.SimpleNamespace(rfilename="mmproj-only.gguf", size=1),
    ]))
    with pytest.raises(FileNotFoundError, match="no loadable"):
        acquire._remote_gguf("org/repo")
    assert acquire.gguf_size_gb("org/repo") is None


def test_prepared_gguf_download_pins_one_revision_and_selection(monkeypatch):
    siblings = [
        types.SimpleNamespace(rfilename="model-q8.gguf", size=8_000_000_000),
        types.SimpleNamespace(rfilename="model-q4.gguf", size=4_000_000_000),
    ]
    class Api:
        def model_info(self, *_args, **_kwargs):
            return types.SimpleNamespace(sha="a" * 40, siblings=siblings)
    calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: Api())
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, filename, **kwargs: calls.append(
                            (repo, filename, kwargs)) or "/cache/model.gguf")
    monkeypatch.setattr(hf_utils, "are_progress_bars_disabled", lambda: False)
    monkeypatch.setattr(hf_utils, "disable_progress_bars", lambda: None)
    monkeypatch.setattr(hf_utils, "enable_progress_bars", lambda: None)

    plan = acquire.prepare_download("org/repo", gguf=True)
    assert plan.revision == "a" * 40 and plan.filename == "model-q4.gguf"
    assert plan.size_gb == 4.0
    assert acquire.download_prepared(plan) == "/cache/model.gguf"
    assert calls == [("org/repo", "model-q4.gguf", {"revision": "a" * 40})]


def test_prepared_transformer_download_pins_probed_revision(monkeypatch):
    class Api:
        def model_info(self, *_args, **_kwargs):
            return types.SimpleNamespace(
                sha="b" * 40,
                siblings=[types.SimpleNamespace(rfilename="model.safetensors", size=2_000_000_000)])
    calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: Api())
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo, **kwargs: calls.append((repo, kwargs)) or "/cache/snapshot")
    monkeypatch.setattr(hf_utils, "are_progress_bars_disabled", lambda: False)
    monkeypatch.setattr(hf_utils, "disable_progress_bars", lambda: None)
    monkeypatch.setattr(hf_utils, "enable_progress_bars", lambda: None)

    plan = acquire.prepare_download("org/repo", gguf=False)
    assert plan.revision == "b" * 40 and plan.size_gb == 2.0
    assert acquire.download_prepared(plan) == "/cache/snapshot"
    assert calls == [("org/repo", {"revision": "b" * 40})]


def test_prepare_download_covers_local_invalid_revision_and_missing_selection(
        tmp_path, monkeypatch):
    local = tmp_path / "model.gguf"
    local.write_bytes(b"weights")
    local_plan = acquire.prepare_download(str(local), gguf=True)
    assert local_plan.repo_id is None
    assert acquire.download_prepared(local_plan) == str(local.resolve())

    with pytest.raises(ValueError, match="invalid model"):
        acquire.prepare_download("../bad", gguf=False)

    class BadRevisionApi:
        def model_info(self, *_args, **_kwargs):
            return types.SimpleNamespace(sha="main", siblings=[])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: BadRevisionApi())
    with pytest.raises(RuntimeError, match="immutable revision"):
        acquire.prepare_download("org/repo", gguf=False)

    class NoGgufApi:
        def model_info(self, *_args, **_kwargs):
            return types.SimpleNamespace(sha="c" * 40, siblings=[])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: NoGgufApi())
    with pytest.raises(FileNotFoundError, match="no loadable"):
        acquire.prepare_download("org/repo", gguf=True)

    with pytest.raises(ValueError, match="no file"):
        acquire.download_prepared(acquire.AcquisitionPlan(
            "local", None, None, None, None))


def test_prepare_download_refuses_unknown_payload_sizes(monkeypatch):
    class UnknownTransformerApi:
        def model_info(self, *_args, **_kwargs):
            return types.SimpleNamespace(
                sha="a" * 40,
                siblings=[types.SimpleNamespace(rfilename="model.safetensors", size=None)])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: UnknownTransformerApi())
    with pytest.raises(RuntimeError, match="size"):
        acquire.prepare_download("org/repo", gguf=False)

    class UnknownGgufApi:
        def model_info(self, *_args, **_kwargs):
            return types.SimpleNamespace(
                sha="a" * 40,
                siblings=[types.SimpleNamespace(rfilename="model.gguf", size=None)])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: UnknownGgufApi())
    with pytest.raises(RuntimeError, match="size"):
        acquire.prepare_download("org/repo", gguf=True)


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
# is_local_gguf / valid_model_ref — accept loose local GGUF files (the local model library)
# Slug: 2026-06-25-local-gguf-cli-support
# --------------------------------------------------------------------------- #
def test_is_local_gguf_accepts_existing_gguf_file(tmp_path):
    f = tmp_path / "Model-Q4_K_M.gguf"
    f.write_bytes(b"\x00")
    assert acquire.is_local_gguf(str(f)) is True


def test_is_local_gguf_rejects_missing_non_gguf_flaglike_and_nonstr(tmp_path):
    txt = tmp_path / "model.txt"
    txt.write_bytes(b"\x00")
    assert acquire.is_local_gguf(str(tmp_path / "nope.gguf")) is False   # doesn't exist
    assert acquire.is_local_gguf(str(txt)) is False                       # not .gguf
    assert acquire.is_local_gguf("-evil.gguf") is False                   # flag-like (leading dash)
    assert acquire.is_local_gguf(12345) is False                          # not a str


def test_valid_model_ref_accepts_repo_id_and_local_gguf(tmp_path):
    f = tmp_path / "Model-Q4_K_M.gguf"
    f.write_bytes(b"\x00")
    assert acquire.valid_model_ref("org/name") is True       # repo id
    assert acquire.valid_model_ref(str(f)) is True           # local .gguf file


def test_valid_model_ref_accepts_repo_quant_selector():
    # `repo:filename.gguf` pins a specific quant in an HF repo — the workers' _resolve_gguf
    # supports it, so the CLI must too (essential for quant-ladder benchmarking).
    assert acquire.valid_model_ref("bartowski/Qwen2.5-14B-Instruct-GGUF:"
                                   "Qwen2.5-14B-Instruct-Q4_K_M.gguf") is True


def test_repo_quant_selector_excludes_local_colon_and_windows_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    local = Path("repo:Model-Q4_K_M.gguf")
    local.write_bytes(b"weights")
    assert acquire.is_local_gguf(str(local)) is True
    assert acquire.valid_repo_gguf_ref(str(local)) is False
    assert acquire.valid_model_ref(str(local)) is True
    assert acquire.valid_repo_gguf_ref(r"C:\models\M-Q4_K_M.gguf") is False
    assert acquire.valid_repo_gguf_ref("C:/models/M-Q4_K_M.gguf") is False


def test_valid_model_ref_rejects_repo_quant_selector_when_not_gguf():
    # the filename half must be a .gguf, and must not be flag-like (argv-injection guard)
    assert acquire.valid_model_ref("org/name:notes.txt") is False
    assert acquire.valid_model_ref("org/name:-rf.gguf") is False
    assert acquire.valid_model_ref("--evil/x:m.gguf") is False     # repo half must be valid
    assert acquire.valid_model_ref("org/name:../../outside.gguf") is False
    assert acquire.valid_model_ref("org/name:/absolute.gguf") is False
    assert acquire.valid_model_ref(r"org/name:dir\outside.gguf") is False


def test_valid_model_ref_rejects_flag_like_and_malformed():
    for bad in ("--evil", "a=b", "a b", "-rf"):
        assert acquire.valid_model_ref(bad) is False


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
