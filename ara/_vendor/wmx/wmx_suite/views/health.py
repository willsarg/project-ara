# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Render function for ``wmx-suite health``.

Data schema
-----------
``data`` is a dict with the following keys:

    wall_gb         : float — crash wall in GB (e.g. 17.18)
    safe_budget_gb  : float — crash wall minus margin (e.g. 15.18)
    free_now_gb     : float — safe_budget_gb minus live baseline (e.g. 12.04)
    swap_free_gb    : float | None — swap available in GB, or None if unknown
    swap_warn       : bool  — True when swap_free_gb is not None and < 2.0
    margin_gb       : float — configured margin in GB (e.g. 2.0)
    baseline_gb     : float — live wired baseline in GB (e.g. 3.14)
    baseline_sample : str   — description of sampling, e.g. "median of 3"
    models          : list[dict]
        Each dict has:
            name         : str   — short model name (mlx-community/ prefix stripped)
            loads_gb     : float — base memory to load (no context), e.g. 8.10
            spare_gb     : float — free_now_gb minus loads_gb, signed (e.g. 7.08 or -0.40)
            safe_ctx     : int   — safe context ceiling in tokens (0 if over budget)
            safe_ctx_str : str   — human display string, e.g. "~26,400 tokens" or
                                   "over budget — won't load"
            ok           : bool  — True if model can load + has useful context
            base_gb      : float — raw base GB from fit (for verbose table)
            slope_gb_per_k: float — GB per 1k tokens from fit (for verbose table)
            safe_cap_tok : int   — raw safe context cap from launcher (for verbose)
"""
from __future__ import annotations


def render(console, data: dict) -> None:
    """Render the health snapshot to *console*.

    Normal output:
        Section headline, glossed budget block (crash wall / safe budget /
        free now / swap), blank line, aligned model table with ✓/✗ glyphs,
        blank line, legend, next_block.

    Verbose appends a raw per-model table (base GB, GB/1k slope, safe cap tok)
    plus margin/baseline metadata.
    """
    c = console
    models = data["models"]

    # ── headline ────────────────────────────────────────────────────────────
    c.emit(c.section("Can each model load and run safely right now?"))
    c.emit()

    # ── budget block ────────────────────────────────────────────────────────
    c.emit(c.field(
        "crash wall",
        f"{data['wall_gb']:.2f} GB",
        "cross it and the Mac can hard-lock",
        value_role="warn",
    ))
    c.emit(c.field(
        "safe budget",
        f"{data['safe_budget_gb']:.2f} GB",
        f"crash wall − {data['margin_gb']:g} GB margin (the planning ceiling)",
        value_role="good",
    ))
    c.emit(c.field(
        "free now",
        f"{data['free_now_gb']:.2f} GB",
        "safe budget − memory already wired",
        value_role="metric",
    ))

    swap = data["swap_free_gb"]
    if swap is None:
        c.emit(c.field("swap free", "unknown"))
    elif data["swap_warn"]:
        swap_val = f"{swap:.2f} GB " + c.glyph("warn") + " tiny"
        c.emit(c.field(
            "swap free",
            swap_val,
            "almost no fallback if you cross the wall",
            value_role="warn",
        ))
    else:
        c.emit(c.field("swap free", f"{swap:.2f} GB"))

    c.emit()

    # ── model table ─────────────────────────────────────────────────────────
    # Compute column widths matching the mockup layout.
    LOADS_W = 9   # "loads at" right-justified
    SPARE_W = 10  # "spare room" right-justified

    name_w = max((len(m["name"]) for m in models), default=5)
    name_w = max(name_w, len("model"))

    # Header (manually aligned to match the mockup, no table primitive — table
    # primitive forces GUTTER which would misalign the glyph column).
    hdr = (
        "   "
        + "   "                              # glyph placeholder "✓ " = 2 + 2 gap
        + c.style("header", "model".ljust(name_w))
        + "   "
        + c.style("header", "loads at".rjust(LOADS_W))
        + "   "
        + c.style("header", "spare room".rjust(SPARE_W))
        + "   "
        + c.style("header", "safe context")
    )
    c.emit(hdr)

    for m in models:
        glyph = c.glyph("ok") if m["ok"] else c.glyph("bad")
        nm = m["name"].ljust(name_w)
        loads_str = f"{m['loads_gb']:.2f} GB".rjust(LOADS_W)
        sign = "+" if m["spare_gb"] >= 0 else "−"
        spare_str = f"{sign}{abs(m['spare_gb']):.2f} GB".rjust(SPARE_W)
        spare_styled = c.style("good", spare_str) if m["ok"] else c.style("bad", spare_str)
        ctx_styled = m["safe_ctx_str"] if m["ok"] else c.style("bad", m["safe_ctx_str"])
        row = (
            "   " + glyph + "  "
            + nm
            + "   " + loads_str
            + "   " + spare_styled
            + "   " + ctx_styled
        )
        c.emit(row)

    # ── verbose raw table ───────────────────────────────────────────────────
    if c.verbose:
        vhdr = (
            c.section("raw (--verbose)")
            + c.style("dim",
                f"   margin {data['margin_gb']:.2f} GB"
                f" · baseline {data['baseline_gb']:.2f} GB"
                f" ({data['baseline_sample']})"
            )
        )
        c.emit()
        c.emit(vhdr)
        vname_w = max((len(m["name"]) for m in models), default=5)
        vhdr2 = c.style("dim",
            "  " + "model".ljust(vname_w)
            + "   " + "base GB".rjust(8)
            + "   " + "GB/1k".rjust(7)
            + "   " + "safe cap tok".rjust(12)
        )
        c.emit(vhdr2)
        for m in models:
            row = c.style("dim",
                "  " + m["name"].ljust(vname_w)
                + "   " + f"{m['base_gb']:.2f}".rjust(8)
                + "   " + f"{m['slope_gb_per_k']:.4f}".rjust(7)
                + "   " + str(m["safe_cap_tok"]).rjust(12)
            )
            c.emit(row)

    # ── legend ──────────────────────────────────────────────────────────────
    c.emit()
    c.emit(c.style("dim",
        "  loads at     memory the model needs just to load (no context yet)\n"
        "  spare room   safe budget left after it loads — bigger is safer\n"
        "  safe context largest prompt+reply we can guarantee won't cross the wall"
    ))

    # ── next block ──────────────────────────────────────────────────────────
    c.emit(c.next_block([
        ("wmx-suite run --model <m> --dry-run", "preview the safe launch plan without running it"),
        ("wmx-suite characterize <model>",      "measure the safe ceiling of a model not listed yet"),
    ]))
