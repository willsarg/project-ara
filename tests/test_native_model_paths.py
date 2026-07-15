# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Native transformer engines accept immutable local HF snapshot directories."""
from __future__ import annotations

import json
import sys
import types

from ara._engine_packages.cuda.ara_engine_cuda import models as cuda_models
from ara._engine_packages.mlx.ara_engine_mlx import models as mlx_models


def _snapshot(tmp_path):
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 4,
        "hidden_size": 512,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "max_position_embeddings": 4096,
    }))
    with open(snapshot / "model.safetensors", "wb") as weights:
        weights.truncate(10_000_000)
    return snapshot


def test_mlx_models_describe_local_snapshot_directory(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path)
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    utils = types.ModuleType("mlx_lm.utils")
    utils._get_classes = lambda _config: (object, object)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", utils)

    info = mlx_models.describe(str(snapshot))

    assert info is not None
    assert info.hf_id == str(snapshot)
    assert info.n_layers == 4 and info.max_context == 4096
    assert info.weights_gb > 0


def test_cuda_models_describe_local_snapshot_directory(tmp_path):
    snapshot = _snapshot(tmp_path)

    info = cuda_models.describe(str(snapshot))

    assert info is not None
    assert info.hf_id == str(snapshot)
    assert info.n_layers == 4 and info.max_context == 4096
    assert info.weights_gb > 0


def test_native_weight_sizing_uses_only_confined_index_shards(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "model.safetensors").rename(snapshot / "model-00001.safetensors")
    with open(snapshot / "unreferenced.safetensors", "wb") as weights:
        weights.truncate(50_000_000)
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layer.0": "model-00001.safetensors"},
    }))

    assert mlx_models._snapshot_weight_bytes(str(snapshot)) == 10_000_000
    assert cuda_models._snapshot_weight_bytes(str(snapshot)) == 10_000_000

    (snapshot / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layer.0": "../outside.safetensors"},
    }))
    assert mlx_models._snapshot_weight_bytes(str(snapshot)) == 0
    assert cuda_models._snapshot_weight_bytes(str(snapshot)) == 0


def test_native_weight_sizing_refuses_external_direct_symlink(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "model.safetensors").unlink()
    external = tmp_path / "outside.safetensors"
    external.write_bytes(b"outside")
    (snapshot / "model.safetensors").symlink_to(external)

    assert mlx_models._snapshot_weight_bytes(str(snapshot)) == 0
    assert cuda_models._snapshot_weight_bytes(str(snapshot)) == 0


def test_native_weight_sizing_accepts_repository_blob_symlink(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "model.safetensors").unlink()
    blob = tmp_path / "blobs" / ("b" * 40)
    blob.parent.mkdir()
    blob.write_bytes(b"weights")
    (snapshot / "model.safetensors").symlink_to(blob)

    assert mlx_models._snapshot_weight_bytes(str(snapshot)) == 7
    assert cuda_models._snapshot_weight_bytes(str(snapshot)) == 7
