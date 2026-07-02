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

import itertools
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
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

# llama.cpp's ternary quants aren't an integer bit-width — these are their published
# bits-per-weight (bpw), used only to ORDER them against ordinary quants.
_TERNARY_BPW = {"tq1": 1.69, "tq2": 2.06}


def quant_bits(quant: str | None) -> float | None:
    """Effective bit-width of a quant token, *for ordering inversions only* — never a quality
    claim. Maps float labels (``f32``/``fp32``→32, ``f16``/``fp16``/``bf16``→16), the int/bit
    families (``int8``/``8bit``/``q8_*``→8, ``int4``/``4bit``/``awq``/``gptq``→4, generic
    ``<N>bit``→N), llama.cpp ``q<N>_*``/``iq<N>_*``→N, and the ternary quants
    (``tq1_*``→1.69, ``tq2_*``→2.06). Unknown token or ``None`` → ``None`` (never guessed —
    Rule #3). Spec 2026-07-02-recommend-inversion-guard."""
    if quant is None:
        return None
    q = quant.lower()
    if q in ("f32", "fp32"):
        return 32.0
    if q in ("f16", "fp16", "bf16"):
        return 16.0
    if q in ("int8", "8bit"):
        return 8.0
    if q in ("int4", "4bit", "awq", "gptq"):
        return 4.0
    m = re.fullmatch(r"(tq[12])(?:_.*)?", q)
    if m:
        return _TERNARY_BPW[m.group(1)]
    m = re.fullmatch(r"i?q(\d+)(?:_.*)?", q)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"(\d+)bit", q)
    if m:
        return float(m.group(1))
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
    # Set by :func:`flag_inversions` when a same-base quant of a different precision upset this
    # reading's expected precision ordering — a short disclosure, never a re-ranking (Rule #3).
    inversion: str | None = None


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
    flag_inversions(out, use_case)
    return out


def flag_inversions(recs: list[dict], use_case: str) -> None:
    """Disclose (never silently rank on) quant *inversions*: within a base model, a
    lower-precision quant that measured HIGHER than a higher-precision one. Rule #3 — the fix is
    disclosure, so this only annotates ``score.inversion``; it does NOT reorder *recs*.

    Only measured-tier entries with a known quant (:func:`quant_key` → :func:`quant_bits`, both
    non-``None``) and a non-``None`` ``sample_size`` are eligible; anything else is skipped. For
    each eligible same-base pair whose bit-widths differ and where the lower-bits reading's value
    is *strictly* greater, the two-proportion standard error ``se`` decides confidence:
    ``within_noise`` iff ``|pa-pb| < 1.96*se`` (``se == 0`` counts as within noise). Both entries
    get a short message naming the *other* quant. An entry in several inversions keeps the FIRST
    (by ranked order); *use_case* is accepted for API symmetry — *recs* are already scored for it.
    Spec 2026-07-02-recommend-inversion-guard."""
    del use_case                                  # recs are already scored for this use_case
    groups: dict[str, list[tuple[dict, str, float]]] = defaultdict(list)
    for r in recs:
        s = r["score"]
        if s is None or s.tier != "measured" or s.sample_size is None:
            continue
        quant = quant_key(r["model_id"])
        bits = quant_bits(quant)
        if quant is None or bits is None:
            continue
        groups[base_key(r["model_id"])].append((r, quant, bits))

    for members in groups.values():
        for (ra, qa, ba), (rb, qb, bb) in itertools.combinations(members, 2):
            if ba == bb:
                continue
            (low, low_q), (high, high_q) = ((ra, qa), (rb, qb)) if ba < bb else ((rb, qb), (ra, qa))
            sl, sh = low["score"], high["score"]
            if not sl.value > sh.value:            # not an inversion (higher precision won / tied)
                continue
            pa, na, pb, nb = sl.value, sl.sample_size, sh.value, sh.sample_size
            se = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
            noise = " within noise" if se == 0 or abs(pa - pb) < 1.96 * se else ""
            if low["score"].inversion is None:
                low["score"] = replace(sl, inversion=f"outscores {high_q}{noise}")
            if high["score"].inversion is None:
                high["score"] = replace(sh, inversion=f"outscored by {low_q}{noise}")
