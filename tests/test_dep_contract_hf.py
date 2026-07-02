# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Dependency contract for ``huggingface_hub`` — layer 4 of the testing architecture.

Slug: 2026-07-01-ara-testing-architecture

ARA's unit suite mocks ``huggingface_hub`` everywhere (deliberately — hermetic + offline: see
``tests/conftest.py`` and the per-module ``_whoami`` / ``_get_token`` seams). That's the right
call for the 100%-coverage unit gate, but it means a ``huggingface_hub`` version bump can pass
the whole gate without the REAL library ever being exercised — a signature rename, a moved
error class, or a changed cache-miss sentinel would go unnoticed until a user hits it.

This file is the fix, in two halves:

- **Offline half** (no marker): imports the real ``huggingface_hub`` and asserts, via
  ``inspect.signature``, that every function/class ARA calls still exists with the parameter
  names ARA passes, and that every error class ARA catches still exists as an ``Exception``
  subclass. No network; runs inside the default hermetic gate (measures the *dependency*, not
  ``ara/`` — coverage is scoped to ``ara/`` so these tests don't affect ``fail_under = 100``).
- **Online half** (``@pytest.mark.dep_contract``): hits the real Hugging Face Hub with a tiny,
  stable public repo to prove the *behavior* ARA depends on (not just the signature) still
  holds. Deselected from the default run exactly like ``integration`` (see ``addopts`` in
  ``pyproject.toml``); opt in via ``pytest -m dep_contract --no-cov``.

The surface below is exactly what ``ara/`` imports from ``huggingface_hub`` today — enumerated
via ``grep -rn "huggingface_hub\\|hf_hub" ara/ --include="*.py"``. Nothing more.
"""
from __future__ import annotations

import inspect

import pytest


# --------------------------------------------------------------------------- #
# offline half — signature / existence contract, hermetic, part of the default gate
# --------------------------------------------------------------------------- #

def _params(fn) -> set[str]:
    return set(inspect.signature(fn).parameters)


def test_hf_hub_download_signature():
    """ara/catalog.py:27  hf_hub_download(model_id, "config.json")
    ara/workers/cpu_llama.py:175  hf_hub_download(repo, fname)
    ara/workers/cuda_gguf_llama.py:264  hf_hub_download(repo, fname)
    ara/workers/vulkan_llama.py:237  hf_hub_download(repo, fname)
    All call sites pass exactly two positionals: repo id, filename.
    """
    from huggingface_hub import hf_hub_download

    params = list(inspect.signature(hf_hub_download).parameters)
    # First two params must accept positional args named/ordered as ARA expects.
    assert params[0] == "repo_id"
    assert params[1] == "filename"


def test_snapshot_download_signature():
    """ara/acquire.py:149  snapshot_download(repo_id)"""
    from huggingface_hub import snapshot_download

    params = list(inspect.signature(snapshot_download).parameters)
    assert params[0] == "repo_id"


def test_scan_cache_dir_exists():
    """ara/catalog.py:49,149,277,311  scan_cache_dir() — called with no args."""
    from huggingface_hub import scan_cache_dir

    sig = inspect.signature(scan_cache_dir)
    # every declared param must have a default — ARA calls it with zero args
    assert all(p.default is not inspect.Parameter.empty for p in sig.parameters.values())


def test_try_to_load_from_cache_signature():
    """ara/backends/cuda.py:56  try_to_load_from_cache(model, "config.json")
    ara/backends/apple.py:45  try_to_load_from_cache(model, "config.json")
    """
    from huggingface_hub import try_to_load_from_cache

    params = list(inspect.signature(try_to_load_from_cache).parameters)
    assert params[0] == "repo_id"
    assert params[1] == "filename"


def test_hfapi_model_info_signature():
    """ara/acquire.py:108  HfApi().model_info(repo_id, files_metadata=True)
    ara/workers/cpu_llama.py:170  HfApi().model_info(repo, files_metadata=True)
    ara/workers/cuda_gguf_llama.py:250  HfApi().model_info(repo, files_metadata=True)
    ara/workers/vulkan_llama.py:223  HfApi().model_info(repo, files_metadata=True)
    """
    from huggingface_hub import HfApi

    assert inspect.isclass(HfApi)
    params = _params(HfApi.model_info)
    assert "repo_id" in params
    assert "files_metadata" in params

    # acquire.py reads ``info.siblings`` -> each with ``.size`` and ``.rfilename``.
    from huggingface_hub.hf_api import RepoSibling

    sibling_fields = _params(RepoSibling.__init__) | set(
        getattr(RepoSibling, "__dataclass_fields__", {})
    )
    assert "size" in sibling_fields
    assert "rfilename" in sibling_fields


def test_progress_bar_helpers_exist():
    """ara/acquire.py:141
    from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars
    """
    from huggingface_hub.utils import (
        are_progress_bars_disabled, disable_progress_bars, enable_progress_bars,
    )

    assert callable(are_progress_bars_disabled)
    assert callable(disable_progress_bars)
    assert callable(enable_progress_bars)
    # ARA calls each with zero args.
    for fn in (are_progress_bars_disabled, disable_progress_bars, enable_progress_bars):
        sig = inspect.signature(fn)
        assert all(p.default is not inspect.Parameter.empty for p in sig.parameters.values())


def test_whoami_signature():
    """ara/hf_auth.py:29  whoami(token=token)"""
    from huggingface_hub import whoami

    assert "token" in _params(whoami)


def test_get_token_signature():
    """ara/hf_auth.py:35  get_token() — called with no args."""
    from huggingface_hub import get_token

    sig = inspect.signature(get_token)
    assert all(p.default is not inspect.Parameter.empty for p in sig.parameters.values())


def test_hf_token_path_constant_exists():
    """ara/hf_auth.py:41  from huggingface_hub.constants import HF_TOKEN_PATH"""
    from huggingface_hub.constants import HF_TOKEN_PATH

    assert isinstance(HF_TOKEN_PATH, str)
    assert HF_TOKEN_PATH  # non-empty


@pytest.mark.parametrize(
    "name",
    [
        "GatedRepoError",
        "HfHubHTTPError",
        "LocalEntryNotFoundError",
        "OfflineModeIsEnabled",
        "RepositoryNotFoundError",
    ],
)
def test_error_classes_exist_as_exceptions(name):
    """ara/acquire.py:78-81  classify_repo_error catches these five error classes."""
    import huggingface_hub.errors as hf_errors

    cls = getattr(hf_errors, name)
    assert inspect.isclass(cls)
    assert issubclass(cls, Exception)


def test_gated_and_not_found_are_distinguishable():
    """ara/acquire.py:83-85 checks isinstance(exc, GatedRepoError) BEFORE
    isinstance(exc, RepositoryNotFoundError) — GatedRepoError subclasses
    RepositoryNotFoundError in huggingface_hub, so ordering matters. Guard that
    subclass relationship so a library refactor can't silently invert it."""
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    assert issubclass(GatedRepoError, RepositoryNotFoundError)


def test_offline_errors_are_connection_errors_or_file_errors():
    """ara/acquire.py:87-88 catches (LocalEntryNotFoundError, OfflineModeIsEnabled) as
    the 'offline' bucket, and separately catches bare ConnectionError. Confirm
    OfflineModeIsEnabled still derives from ConnectionError (why a bare
    ``except ConnectionError`` also catches it in practice elsewhere in the codebase)."""
    from huggingface_hub.errors import OfflineModeIsEnabled

    assert issubclass(OfflineModeIsEnabled, ConnectionError)


def test_hf_hub_http_error_has_response_status_code_shape():
    """ara/acquire.py:91-92 and ara/hf_auth.py:88-89 both read
    ``exc.response.status_code`` off an HfHubHTTPError-shaped exception to detect 401/403.
    Confirm the constructor still accepts/carries a ``response`` (an httpx.Response, which
    itself carries ``.status_code``) — the exact attribute path ARA's classifiers inspect."""
    from huggingface_hub.errors import HfHubHTTPError

    assert issubclass(HfHubHTTPError, Exception)
    init_params = _params(HfHubHTTPError.__init__)
    assert "response" in init_params


# --------------------------------------------------------------------------- #
# online half — real Hub behavior, marked, excluded from the default gate
# --------------------------------------------------------------------------- #

# A tiny, stable, public repo maintained by the HF team for exactly this purpose
# (used across the ecosystem's own test suites) — keeps download bytes minimal.
_TINY_REPO = "hf-internal-testing/tiny-random-gpt2"


@pytest.mark.dep_contract
def test_online_model_info_returns_fields_ara_reads():
    """ara/acquire.py:probe_repo reads info.siblings[*].size / .rfilename."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(_TINY_REPO, files_metadata=True)
    assert info.id == _TINY_REPO
    assert info.siblings, "expected at least one sibling file"
    sizes = [s.size for s in info.siblings if s.size]
    assert sizes, "expected at least one sibling with a known size"
    assert any(s.rfilename == "config.json" for s in info.siblings)


@pytest.mark.dep_contract
def test_online_hf_hub_download_produces_a_file(tmp_path):
    """ara/catalog.py:_read_config and the worker _resolve_gguf helpers download a real
    file via hf_hub_download; prove it actually lands on disk and is readable JSON."""
    import json

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(_TINY_REPO, "config.json", cache_dir=str(tmp_path))
    assert path
    with open(path) as fh:
        cfg = json.load(fh)
    assert isinstance(cfg, dict)


@pytest.mark.dep_contract
def test_online_try_to_load_from_cache_miss_then_hit(tmp_path):
    """ara/backends/cuda.py / ara/backends/apple.py calibration_model_cached():
    a cache miss returns a non-str sentinel; a cache hit returns a str path."""
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    miss = try_to_load_from_cache(_TINY_REPO, "config.json", cache_dir=str(tmp_path))
    assert not isinstance(miss, str)  # None (or the _CACHED_NO_EXIST sentinel), never a path

    hf_hub_download(_TINY_REPO, "config.json", cache_dir=str(tmp_path))
    hit = try_to_load_from_cache(_TINY_REPO, "config.json", cache_dir=str(tmp_path))
    assert isinstance(hit, str)


@pytest.mark.dep_contract
def test_online_whoami_with_invalid_token_raises_classified_error():
    """ara/hf_auth.py:_classify_whoami_error maps a 401/403 whoami response to "invalid".
    Prove the REAL error path: an invalid token raises HfHubHTTPError with response.status_code
    in (401, 403) — the exact shape hf_auth._classify_whoami_error inspects."""
    from huggingface_hub import whoami
    from huggingface_hub.errors import HfHubHTTPError

    with pytest.raises(HfHubHTTPError) as excinfo:
        whoami(token="hf_invalid_dep_contract_probe_0000000000000")

    status = getattr(getattr(excinfo.value, "response", None), "status_code", None)
    assert status in (401, 403)


@pytest.mark.dep_contract
def test_online_model_info_not_found_raises_repository_not_found_error():
    """ara/acquire.py:classify_repo_error catches RepositoryNotFoundError from
    HfApi().model_info() for a repo that doesn't exist — prove the real Hub still raises it."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    with pytest.raises(RepositoryNotFoundError):
        HfApi().model_info("this-org-does-not-exist-ara-dep-contract/this-repo-does-not-exist")
