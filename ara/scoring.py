# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Use-case capability lookup for ``recommend --use-case`` — engine-free.

``recommend`` never runs inference (closed decision, spec 2026-06-23-capability-pipeline). This
module only *reads* a capability score and labels its provenance:

  * **measured** — a benchmark ARA ran locally on this machine, on the model's actual quant
    (stored by the separate ``benchmark`` step). Captures the quant×capability degradation an
    imported score hides.
  * **imported** — a published score, clearly *not measured here*.

A measured score always wins over an imported one. Absent → ``None`` (the caller renders
``unknown``; never a guessed rank — Rule #3). Spec 2026-06-28-recommend-use-case-and-serve-selection.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_IMPORTED_PATH = Path(__file__).parent / "data" / "usecase_scores.json"

# Trailing quantization / format tokens to strip so a quant variant matches its base model
# (e.g. mlx-community/Llama-3.2-3B-Instruct-4bit → llama-3.2-3b-instruct).
_QUANT_RE = re.compile(
    r"-(?:[qf]?\d+bit|q\d(?:_[a-z0-9]+)*|bf16|fp16|fp32|f16|int[48]|gguf|mlx|awq|gptq)$", re.I)


def base_key(model_id: str) -> str:
    """Normalise a model id to a base key: drop the org, lowercase, and strip trailing quant /
    format tokens. Quant variants and their base resolve to the same key, so an imported base
    score matches a locally-cataloged quant."""
    name = model_id.rsplit("/", 1)[-1].lower()
    prev = None
    while prev != name:                 # strip repeated tails: ...-instruct-4bit-gguf
        prev, name = name, _QUANT_RE.sub("", name)
    return name


# Genuine quantization tokens only — bit-widths, llama.cpp quant classes (Q/IQ/TQ), and the
# awq/gptq/float classes. The container FORMATS `gguf`/`mlx` (present in _QUANT_RE for base_key
# stripping) are deliberately excluded: they describe packaging, not precision, so they're never
# a quant (Rule #3).
_QUANT_ONLY_RE = re.compile(
    r"[qf]?\d+bit|[it]?q\d(?:_[a-z0-9]+)*|bf16|fp16|fp32|f16|f32|int[48]|awq|gptq", re.I)


def quant_key(model_id: str) -> str | None:
    """The first genuine quant token in *model_id* (lowercased), or None if none is present.

    A capturing counterpart to :data:`_QUANT_RE`: it inspects each ``-``-delimited segment of the
    model name (org dropped) and returns the first that is *entirely* a quant token — e.g.
    ``4bit``, ``q4_k_m``, ``int8``, ``awq``, ``fp16``. Container formats (``gguf``/``mlx``) are
    never returned. Used when the catalog doesn't record the quant, so a measured score can still
    carry the precision it was taken at."""
    name = model_id.rsplit("/", 1)[-1].lower()
    for tok in name.split("-"):
        if _QUANT_ONLY_RE.fullmatch(tok):
            return tok
    return None

# The use cases v1 supports as `--use-case` values (design §2.1).
USE_CASES = ("coding", "reasoning", "agentic", "extraction", "rag", "chat")


@dataclass(frozen=True)
class Score:
    """One capability reading. ``value`` is normalised 0..1; ``source`` is the citation
    (imported) or the run conditions (measured); ``tier`` is ``measured`` | ``imported``.

    Measured readings also carry the run's honesty metadata so the caller can annotate a depressed
    or shaky score (Rule #3): ``sample_size`` (probe count; <100 → low-confidence) and the
    per-prompt ``refused_n`` / ``errored_n`` counts. All default None (unknown / not-measured)."""
    tier: str
    value: float
    source: str
    sample_size: int | None = None
    refused_n: int | None = None
    errored_n: int | None = None


def score_for(model_id: str, use_case: str, *,
              measured: dict | None = None, imported: dict | None = None) -> Score | None:
    """The capability score for *model_id* at *use_case*, or ``None`` if unknown.

    ``measured`` is keyed by ``(model_id, use_case)``; ``imported`` by ``model_id`` →
    ``use_case`` → ``{score, source}``. Measured wins over imported.
    """
    if measured:
        hit = measured.get((model_id, use_case))
        if hit is not None:
            return Score("measured", hit["score"], hit["source"],
                         sample_size=hit.get("sample_size"),
                         refused_n=hit.get("refused_n"), errored_n=hit.get("errored_n"))
    if imported:
        bucket = imported.get(model_id) or imported.get(base_key(model_id))
        if bucket and use_case in bucket:
            entry = bucket[use_case]
            return Score("imported", entry["score"], entry["source"])
    return None


def load_imported() -> dict:
    """The shipped table of *imported* (published, not-measured-here) capability scores, keyed
    ``model_id`` → ``use_case`` → ``{score, source}``. Each leaf carries its citation; a
    locally-measured score overrides these when present (see :func:`score_for`)."""
    with _IMPORTED_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)["scores"]
    return {base_key(model_id): by_uc for model_id, by_uc in raw.items()}


def rank(recs: list[dict], use_case: str, *,
         measured: dict | None = None, imported: dict | None = None) -> list[dict]:
    """Annotate each rec with a ``score`` (a :class:`Score` or ``None``) and rank by capability:
    scored models first (highest value), unknowns last (never dropped — honest). Within a group,
    larger ``est_context`` breaks ties, preserving the no-use-case ordering."""
    out = [{**r, "score": score_for(r["model_id"], use_case, measured=measured, imported=imported)}
           for r in recs]
    out.sort(key=lambda r: (r["score"] is None,
                            -(r["score"].value if r["score"] is not None else 0.0),
                            -r.get("est_context", 0)))
    return out
