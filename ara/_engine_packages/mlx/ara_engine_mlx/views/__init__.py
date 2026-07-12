# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Pure render functions for each CLI command — one module per command.

Each module exposes ``render(console, data)``: a PURE function — data in,
styled output emitted via the ``Console`` primitives (``ara_engine_mlx.ui``). No
DB / system / MLX access here; that lives in the ``cmd_*`` gather steps in
``cli.py``. This keeps every screen deterministic and golden-testable, and
keeps the whole CLI speaking one visual language.

Visual source of truth: ``docs/mockups/cli-output-mockup.html``.
"""
