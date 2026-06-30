# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Pure render functions for ``wmx-suite benchmark-embeddings``.

Visual language: matches the house style established in ``docs/mockups/cli-output-mockup.html``
(renderSystemProposed / renderHealthProposed / renderListProposed patterns).

PURE: only ``console`` + ``data`` in; ``console.emit`` out. No DB / system / MLX imports.

## Data schemas

### ``render_surface(console, data)``

``data`` is a dict with the following keys:

    model           : str   — full HF model ID
                               e.g. "mlx-community/ModernBERT-base-4bit"
    margin_gb       : float — safety margin in GB (e.g. 2.0)
    profile_source  : str   — one of "loaded" | "cold" | "ignored"
    mlx_version     : str   — e.g. "0.22.1"
    batches         : list[int]  — batch-size axis (rows), e.g. [1, 2, 4, 8]
    seqs            : list[int]  — sequence-length axis (columns), e.g. [128, 512, 2048, 8192]
    cells           : dict[(int,int), dict]
        Keyed by (batch, seq).  Each value dict has:
            status          : str   — "measured" | "skipped"
            throughput_tps  : float | None  — tokens/second (None if skipped)
            latency_ms      : float | None  — milliseconds  (None if skipped)
            peak_gb         : float | None  — MLX-reported peak GB (None if skipped)
            os_wired_gb     : float | None  — OS-wired peak GB    (None if skipped)
            predicted_gb    : float | None  — pre-flight predicted wired GB
                                              (present for both measured and skipped)
    n_cells_measured : int
    n_cells_skipped  : int

PRIMARY METRIC STRATEGY
    Default (normal) mode: the 2-D grid shows **throughput_tps** (e.g. "4211 tok/s").
    This is the most actionable number for a user picking a (batch, seq) operating point.

    Verbose appends a raw appendix (via ``console.raw``) with per-cell rows that include
    all three secondary metrics: latency_ms, peak_gb (MLX), and predicted_gb so users
    can compare predicted vs measured. The appendix only prints when ``console.verbose``.

### ``render_profile_note(console, data)``

``data`` is a dict with:

    profile_source  : str   — "loaded" | "cold" | "ignored"
    model           : str   — HF model ID
    mlx_version     : str   — MLX version string

### ``render_safeguard(console, data)``

``data`` is a dict with:

    model           : str   — HF model ID
    reason          : str   — human-readable abort reason
    predicted_gb    : float — predicted wired peak in GB
    safe_budget_gb  : float — safe budget at time of refusal
    margin_gb       : float — margin in effect
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_PREFIX = "mlx-community/"


def _short(model: str) -> str:
    """Strip the mlx-community/ prefix for display."""
    return model[len(_PREFIX):] if model.startswith(_PREFIX) else model


def _fmt_tps(tps: float) -> str:
    return f"{tps:,.0f} tok/s"


def _fmt_ms(ms: float) -> str:
    return f"{ms:.1f} ms"


def _fmt_gb(gb: float) -> str:
    return f"{gb:.2f} GB"


# ---------------------------------------------------------------------------
# render_profile_note
# ---------------------------------------------------------------------------

def render_profile_note(console, data: dict) -> None:
    """Render the calibration-profile status line.

    Emits one ``field``-style line glossing the profile state:
    - "loaded"  → gate was seeded from a stored profile (green)
    - "cold"    → no profile yet; gate uses conservative defaults (warn)
    - "ignored" → --ignore-profile was passed; profile exists but is bypassed (warn)
    """
    src = data["profile_source"]

    if src == "loaded":
        value = "loaded"
        gloss = "seeding predictive gate from stored calibration"
        role = "good"
    elif src == "ignored":
        value = "ignored"
        gloss = "--ignore-profile in effect; gate uses conservative defaults"
        role = "warn"
    else:  # "cold"
        value = "cold start"
        gloss = "no stored profile; gate uses conservative defaults"
        role = "warn"

    console.emit(console.field("calibration", console.style(role, value), gloss))


# ---------------------------------------------------------------------------
# render_safeguard
# ---------------------------------------------------------------------------

def render_safeguard(console, data: dict) -> None:
    """Render a pre-flight abort / surface-wide refusal guidance block.

    Uses ``console.guidance`` (headline + why + try suggestions).
    """
    short = _short(data["model"])
    predicted = data["predicted_gb"]
    budget = data["safe_budget_gb"]

    headline = f"Won't run {short} — over budget"

    why = [
        f"predicted peak {predicted:.2f} GB exceeds safe budget {budget:.2f} GB",
        f"crossing the crash wall can hard-lock the Mac (margin {data['margin_gb']:.1f} GB active)",
    ]

    tries = [
        ("wmx-suite health",
         "check current free headroom — another model may fit"),
        ("wmx-suite benchmark-embeddings --margin <GB>",
         "loosen the margin if you know your machine has room"),
        ("wmx-suite system",
         "see the full memory budget breakdown"),
    ]

    console.emit(console.guidance(headline, why, tries))


# ---------------------------------------------------------------------------
# render_surface
# ---------------------------------------------------------------------------

def render_surface(console, data: dict) -> None:
    """Render the batch×seq memory/throughput surface table.

    Output structure (normal mode):
        1. Section header
        2. Status line: model · margin · profile source
        3. Calibration-profile note (delegates to render_profile_note)
        4. Primary-metric table (throughput, tok/s) — rows=batch, columns=seq
           Skipped cells show the glyph("bad") marker + predicted GB.
        5. Measured / skipped summary counts
        6. Next block

    Verbose appends (via console.raw):
        Per-cell raw appendix — all three secondary metrics:
        latency_ms, MLX peak GB, predicted GB (vs measured os_wired_gb).
    """
    model = data["model"]
    short = _short(model)
    margin_gb = data["margin_gb"]
    profile_source = data["profile_source"]
    batches: list[int] = data["batches"]
    seqs: list[int] = data["seqs"]
    cells: dict = data["cells"]
    n_measured = data["n_cells_measured"]
    n_skipped = data["n_cells_skipped"]

    # --- 1. section header ---
    console.emit(console.section("Embedding Memory Surface  (batch × seq → throughput)"))
    console.emit()

    # --- 2. status line ---
    console.emit(console.status_line([
        (short, "accent"),
        (f"margin {margin_gb:.1f} GB", "metric"),
        (f"profile: {profile_source}", "gloss"),
    ]))
    console.emit()

    # --- 3. calibration note ---
    render_profile_note(console, data)
    console.emit()

    # --- 4. primary-metric table ---
    # Columns: batch (row header) + one column per seq length
    # Primary metric: throughput_tps (tok/s); skipped shows ✗ predicted_GB
    skip_glyph = console.glyph("bad")

    # Build columns: ("batch", "l", "label") + one per seq
    columns = [("batch", "l", "label")]
    for s in seqs:
        columns.append((f"seq {s}", "r", "metric"))

    # Build rows
    rows = []
    for b in batches:
        row: list = [(str(b), "value")]
        for s in seqs:
            cell = cells.get((b, s))
            if cell is None or cell["status"] == "skipped":
                pred = cell["predicted_gb"] if cell else None
                if pred is not None:
                    text = f"{skip_glyph} {pred:.2f}GB"
                else:
                    text = f"{skip_glyph} pruned"
                row.append((text, "bad"))
            else:
                tps = cell["throughput_tps"]
                row.append((_fmt_tps(tps), "metric"))
        rows.append(tuple(row))

    console.emit(console.table(columns, rows))
    console.emit()

    # --- 5. summary counts ---
    console.emit(console.field(
        "measured",
        console.style("good", str(n_measured)),
        f"cells  ·  {n_skipped} skipped (predicted over budget)",
    ))
    console.emit()

    # --- 6. next block ---
    console.emit(console.next_block([
        ("wmx-suite benchmark-embeddings --verbose",
         "see latency, MLX peak, and predicted-vs-measured per cell"),
        ("wmx-suite health",
         "check current free headroom before a larger sweep"),
        ("wmx-suite system",
         "full memory budget breakdown"),
    ]))

    # --- verbose raw appendix ---
    raw_lines = _build_raw_appendix(batches, seqs, cells)
    raw_out = console.raw("raw per-cell (--verbose)", raw_lines)
    if raw_out:
        console.emit()
        console.emit(raw_out)


def _build_raw_appendix(
    batches: list[int],
    seqs: list[int],
    cells: dict,
) -> list[str]:
    """Build raw appendix lines: batch, seq, throughput, latency, peak_gb, predicted_gb."""
    # Header
    lines = [
        f"{'batch':>6}  {'seq':>6}  {'tok/s':>10}  {'latency ms':>10}  {'peak GB':>8}  {'pred GB':>8}"
    ]
    lines.append("-" * 60)
    for b in batches:
        for s in seqs:
            cell = cells.get((b, s))
            if cell is None:
                continue
            if cell["status"] == "measured":
                tps_s = f"{cell['throughput_tps']:>10,.0f}"
                lat_s = f"{cell['latency_ms']:>10.1f}"
                pk_s  = f"{cell['peak_gb']:>8.2f}"
            else:
                tps_s = f"{'—':>10}"
                lat_s = f"{'—':>10}"
                pk_s  = f"{'—':>8}"
            pred = cell.get("predicted_gb")
            pr_s = f"{pred:>8.2f}" if pred is not None else f"{'—':>8}"
            lines.append(f"{b:>6}  {s:>6}  {tps_s}  {lat_s}  {pk_s}  {pr_s}")
    return lines
