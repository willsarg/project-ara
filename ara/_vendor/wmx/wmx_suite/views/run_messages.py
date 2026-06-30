# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""PURE render functions for ``wmx-suite run`` messages and shared error messages.

All functions are side-effect-free except writing to *console*. No DB, system,
model, or MLX imports. Each function accepts a ``Console`` instance and a
plain ``data`` dict whose schema is documented in the docstring.

Data schemas
------------

render_refusal(console, data)
    data keys:
        model           : str   — model name (mlx-community/ prefix is stripped if present)
        needs_gb        : float — live_base + model_base (e.g. 15.56)
        budget_gb       : float — safe threshold = wall − margin (e.g. 15.18)
        wall_gb         : float — crash-wall limit in GB (e.g. 17.18)
        live_base_gb    : float — wired memory measured just before launch (e.g. 3.12)
        model_base_gb   : float — model's base weight in GB (e.g. 12.44)
        slope_gb_per_k  : float — GB per 1k context tokens (e.g. 0.36003)
        safe_cap_tok    : int   — computed safe context cap (0 = refused)
        source          : str   — "measured" | "estimated"
        cache_type      : str   — e.g. "RotatingKVCache"
        kv_mode         : str   — e.g. "fp16 (not quantizable)" | "4-bit"

render_plan(console, data)
    data keys:
        model               : str   — model name (mlx-community/ prefix stripped if present)
        source              : str   — "measured" | "estimated"
        cache_type          : str   — e.g. "KVCache"
        kv_mode             : str   — e.g. "4-bit" | "fp16 (not quantizable)"
        live_base_gb        : float — wired base in GB
        model_base_gb       : float — model weight in GB
        budget_gb           : float — safe threshold in GB
        wall_gb             : float — crash-wall in GB
        slope_gb_per_k      : float — GB per 1k tokens
        max_kv_size         : int   — safe context cap in tokens
        model_max           : int   — model's architecture context limit
        max_kv_size_enforced: bool  — whether --max-kv-size is runtime-enforced

render_not_found(console, data)
    data keys:
        model           : str   — full HF model ID (e.g. "mlx-community/Llama-3-70B")
        cache_path      : str   — searched path (e.g. "~/.cache/huggingface/hub")
        hf_home_set     : bool  — True if HF_HOME env var is set

render_no_models(console, data)
    data keys: (empty dict accepted — no data required)
"""
from __future__ import annotations

_PREFIX = "mlx-community/"


def _strip(name: str) -> str:
    """Remove the mlx-community/ prefix for display."""
    return name[len(_PREFIX):] if name.startswith(_PREFIX) else name


def _fmt_gb(gb: float) -> str:
    """Format a GB value with 2 decimal places."""
    return f"{gb:.2f} GB"


# ---------------------------------------------------------------------------
# render_refusal
# ---------------------------------------------------------------------------

def render_refusal(console, data: dict) -> None:
    """Render a RULE #1 refusal.

    Normal output:
        - Bad (red) headline: "Won't run <model> — it can't fit safely on this Mac."
        - why: needs X GB vs safe budget Y GB + crash-wall context
        - try: wmx-suite health (first), free up memory, run --force (last)

    Verbose appends a ``raw`` section: source/cache/kv, budget math, slope,
    wall/threshold/safe-cap.
    """
    model = _strip(data["model"])
    needs = _fmt_gb(data["needs_gb"])
    budget = _fmt_gb(data["budget_gb"])
    wall = _fmt_gb(data["wall_gb"])

    headline = f"✗  Won't run {model} — it can't fit safely on this Mac."

    why_lines = [
        f"loading it needs {needs}, but your safe budget is only {budget}.",
        "That leaves no room for context — even a short prompt would push past",
        f"the crash wall ({wall}), which can hard-lock the machine.",
    ]

    tries = [
        ("wmx-suite health",  "see which models DO fit right now"),
        ("free up memory",    "close other apps, then re-check health"),
        ("run --force",       "override at your own risk — may crash the Mac"),
    ]

    console.emit(console.guidance(headline, why_lines, tries))

    raw_lines = [
        f"source={data['source']}  cache={data['cache_type']}  kv={data['kv_mode']}",
        (f"live_base {data['live_base_gb']:.2f} + model {data['model_base_gb']:.2f} = "
         f"{data['needs_gb']:.2f} GB  ·  slope {data['slope_gb_per_k']} GB/1k"),
        (f"wall {data['wall_gb']:.2f}  ·  threshold {data['budget_gb']:.2f}"
         f"  ·  safe cap {data['safe_cap_tok']} tok < MIN_USEFUL_CTX 512"),
    ]
    appendix = console.raw("   raw (--verbose)", raw_lines)
    if appendix:
        console.emit()
        console.emit(appendix)


# ---------------------------------------------------------------------------
# render_plan
# ---------------------------------------------------------------------------

def render_plan(console, data: dict) -> None:
    """Render the informational launch-plan block (non-refusal path).

    Emits a tidy glossed block with model, source, cache type, KV mode,
    safe budget, and max-kv-size cap. Printed before the real launch.

    Verbose appends a ``raw`` section with slope and budget math.
    """
    model = _strip(data["model"])
    enforcement = "" if data.get("max_kv_size_enforced", True) else " (NOT runtime-enforced)"
    cap_str = f"{data['max_kv_size']:,} tokens{enforcement}"

    lines = [
        console.field("model",      model,               "the model you're launching"),
        console.field("source",     data["source"],      "how the memory profile was derived"),
        console.field("cache",      data["cache_type"],  None),
        console.field("KV mode",    data["kv_mode"],     None),
        console.field("safe budget", _fmt_gb(data["budget_gb"]),
                      f"crash wall {_fmt_gb(data['wall_gb'])} − margin"),
        console.field("max-kv-size", cap_str,
                      f"model arch limit {data['model_max']:,} tok"),
    ]
    console.emit(console.section("run plan"))
    for line in lines:
        console.emit(line)

    raw_lines = [
        (f"live_base {data['live_base_gb']:.2f} GB + model {data['model_base_gb']:.2f} GB"
         f" = {data['live_base_gb'] + data['model_base_gb']:.2f} GB"),
        f"slope {data['slope_gb_per_k']} GB/1k tokens",
        f"wall {data['wall_gb']:.2f} GB  ·  threshold {data['budget_gb']:.2f} GB",
    ]
    appendix = console.raw("   raw (--verbose)", raw_lines)
    if appendix:
        console.emit()
        console.emit(appendix)


# ---------------------------------------------------------------------------
# render_not_found
# ---------------------------------------------------------------------------

def render_not_found(console, data: dict) -> None:
    """Render a model-not-in-cache guidance block.

    Normal output:
        - Bad headline: "<full model id> isn't in your Hugging Face cache."
        - Context: the suite only sees models you've already downloaded.
        - try: hf download <model>, wmx-suite list.

    Verbose appends the searched cache path (and HF_HOME note if unset).
    """
    model = data["model"]

    headline = f"✗  {model} isn't in your Hugging Face cache."

    why_lines = [
        "The suite only sees models you've already downloaded.",
    ]

    dl_cmd = f"hf download {model}"
    tries = [
        (dl_cmd,          "get it first"),
        ("wmx-suite list", "see what's already measured"),
    ]

    console.emit(console.guidance(headline, why_lines, tries))

    raw_lines = [f"searched: {data['cache_path']}"]
    if not data.get("hf_home_set", True):
        raw_lines.append("HF_HOME is unset — using the default path above")
    appendix = console.raw("   raw (--verbose)", raw_lines)
    if appendix:
        console.emit()
        console.emit(appendix)


# ---------------------------------------------------------------------------
# render_no_models
# ---------------------------------------------------------------------------

def render_no_models(console, data: dict) -> None:  # noqa: ARG001
    """Render the "no characterized models yet" guidance block.

    data: empty dict (no keys required).

    Points the user at downloading a model and ``characterize``
    (measure safe context ceiling).
    """
    headline = "✗  No characterized models yet."

    why_lines = [
        "You haven't measured any models on this machine yet.",
    ]

    tries = [
        ("hf download <model>",            "download an MLX model into your HF cache"),
        ("wmx-suite characterize <model>", "measure a model's safe context ceiling"),
    ]

    console.emit(console.guidance(headline, why_lines, tries))
