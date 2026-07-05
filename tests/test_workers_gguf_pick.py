# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""GGUF file selection shared by the three llama.cpp-class workers (_resolve_gguf).

For a bare HF repo id the worker picks a ``.gguf`` sibling to download. The old pick —
``min(files, key=size)`` over *all* ``.gguf`` — silently grabbed a ``mmproj-*`` vision
projector (tiny, not the LM) or the most-crushed quant, confounding whatever ran next. The
workers are self-contained scripts (they never import ``ara``) so the fix is duplicated across
the three; this parametrized test pins the shared contract. Workers are outside the 100% core
coverage gate (pyproject ``omit``), so these tests carry the confidence instead.

Item 2 of the master TODO (2026-07-04).
"""
from __future__ import annotations

import types

import pytest

from ara.workers import cpu_llama, cuda_gguf_llama, vulkan_llama

_WORKERS = [cpu_llama, cuda_gguf_llama, vulkan_llama]
_ids = [m.__name__.rsplit(".", 1)[-1] for m in _WORKERS]


def _sib(name: str, size: int):
    return types.SimpleNamespace(rfilename=name, size=size)


@pytest.fixture
def fake_hf(monkeypatch):
    """Install a fake HfApi + hf_hub_download returning ``/dl/<chosen>`` for the given siblings."""
    def _install(siblings):
        import huggingface_hub

        class FakeApi:
            def model_info(self, repo, files_metadata=False):
                return types.SimpleNamespace(siblings=siblings)

        monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                            lambda repo, fname: f"/dl/{fname}")
    return _install


@pytest.mark.parametrize("w", _WORKERS, ids=_ids)
def test_resolve_gguf_excludes_mmproj_and_discloses(w, fake_hf, capsys):
    fake_hf([_sib("mmproj-model-f16.gguf", 100),   # vision projector: tiny, must NOT be chosen
             _sib("model-Q4_K_M.gguf", 5000),
             _sib("model-Q2_K.gguf", 3000)])
    path = w._resolve_gguf("org/repo")
    assert path == "/dl/model-Q2_K.gguf"           # smallest REAL weight, projector skipped
    assert "model-Q2_K.gguf" in capsys.readouterr().err   # discloses the chosen quant (stderr)


@pytest.mark.parametrize("w", _WORKERS, ids=_ids)
def test_resolve_gguf_all_projectors_raises(w, fake_hf):
    fake_hf([_sib("mmproj-model-f16.gguf", 100)])  # only a projector → nothing loadable
    with pytest.raises(FileNotFoundError):
        w._resolve_gguf("org/repo")
