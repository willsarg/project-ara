# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Clean shutdown behavior for the separately packaged MLX HTTP worker."""
from __future__ import annotations

import sys
import types

from ara._engine_packages.mlx.ara_engine_mlx import serve as mlx_serve


def test_worker_ctrl_c_closes_server_without_propagating(monkeypatch):
    state = {"closed": False}

    class _Server:
        def __init__(self, address, handler):
            assert address == ("127.0.0.1", 1234)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            state["closed"] = True

    monkeypatch.setattr(mlx_serve, "_pre_load_gate", lambda *_a, **_k: (None, None))
    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(load=lambda _m: (object(), object())))
    monkeypatch.setattr(mlx_serve, "register_turn_end_tokens", lambda _t: None)
    monkeypatch.setattr(mlx_serve, "_make_handler", lambda *_a: object())
    monkeypatch.setattr(mlx_serve, "HTTPServer", _Server)

    mlx_serve.serve("org/model", 2048, margin_gb=2.0, overhead_gb=1.0, port=1234)
    assert state["closed"] is True
