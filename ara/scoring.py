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
import warnings
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from ara import acquire

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


def canonical_model_id(model_id: str) -> str:
    """Catalog identity for an exact model selector; preserves local paths and bare repo ids."""
    if (model_id.startswith(("/", "./", "../", "~/"))
            or re.match(r"^[A-Za-z]:[\\/]", model_id)):
        return model_id
    return model_id.partition(":")[0] if acquire.valid_repo_gguf_ref(model_id) else model_id


def durable_model_id(model_id: str) -> str:
    """Stable persistence key for *model_id*.

    Existing local GGUF paths are resolved now, while the caller's working directory is known.
    Repo ids and repo:file selectors retain their public spelling.  The original CLI spelling is
    still used for display and engine invocation; only durable evidence keys use this value.
    """
    if acquire.is_local_gguf(model_id):
        return str(Path(model_id).expanduser().resolve())
    return model_id


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
    name = model_id.partition(":")[2] or model_id.rsplit("/", 1)[-1]
    name = name.lower().removesuffix(".gguf")
    for tok in re.split(r"[-.]", name):
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
    probe_context: int | None = None
    generation_cap: int | None = None
    repeat_count: int | None = None
    total_generations: int | None = None
    run_scores: tuple[float, ...] | None = None
    evidence_warning: str | None = None
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
                         refused_n=hit.get("refused_n"), errored_n=hit.get("errored_n"),
                         probe_context=hit.get("probe_context"),
                         generation_cap=hit.get("generation_cap"),
                         repeat_count=hit.get("repeat_count"),
                         total_generations=hit.get("total_generations"),
                         run_scores=(tuple(hit["run_scores"])
                                     if hit.get("run_scores") is not None else None),
                         evidence_warning=hit.get("evidence_warning"))
    if imported:
        bucket = imported.get(model_id) or imported.get(base_key(model_id))
        if bucket and use_case in bucket:
            entry = bucket[use_case]
            return Score("imported", entry["score"], entry["source"])
    return None


def decode_run_scores(raw: str | None, repeat_count: int | None
                      ) -> tuple[list[float] | None, str | None]:
    """Decode persisted per-run scores without trusting corrupt local evidence."""
    if raw is None:
        return None, None
    warning = "invalid stored run-score provenance"
    try:
        values = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, warning
    valid_repeat = (isinstance(repeat_count, int) and not isinstance(repeat_count, bool)
                    and repeat_count > 0)
    if not isinstance(values, list) or not valid_repeat or len(values) != repeat_count:
        return None, warning
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        return None, warning
    return [float(value) for value in values], None


def validate_measured_evidence(row: dict) -> tuple[dict | None, str | None]:
    """Validate a local benchmark row before it can influence recommendation ranking."""
    from ara import benchmark

    warning = "invalid stored benchmark evidence"
    model_id = row.get("model_id")
    use_case = row.get("use_case")
    canonical = canonical_model_id(model_id) if isinstance(model_id, str) else None
    if (row.get("tier") != "measured" or not isinstance(use_case, str)
            or row.get("benchmark_id") != use_case
            or row.get("canonical_model_id") != canonical
            or row.get("base_model") != (base_key(canonical) if canonical else None)
            or not isinstance(row.get("artifact_id"), str) or not row["artifact_id"]):
        return None, warning
    if use_case not in benchmark.USE_CASES:
        return None, warning
    expected_probe = benchmark.load_probe(use_case)
    if (row.get("methodology_id") != benchmark.methodology_id(use_case, expected_probe)
            or row.get("sample_size") != len(expected_probe)):
        return None, warning
    if (not isinstance(row.get("engine_key"), str) or not row["engine_key"]
            or not isinstance(row.get("backend"), str) or not row["backend"]):
        return None, warning
    expected_quant = quant_key(model_id)
    if expected_quant is not None and row.get("quant") != expected_quant:
        return None, warning
    score = row.get("score")
    if (isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or not 0 <= score <= 1):
        return None, warning
    if not isinstance(row.get("source"), str) or not row["source"]:
        return None, warning
    max_score = row.get("max_score")
    if (isinstance(max_score, bool) or not isinstance(max_score, (int, float))
            or not math.isfinite(max_score) or max_score != 1.0):
        return None, warning
    measured_at = row.get("measured_at")
    if measured_at is not None:
        if not isinstance(measured_at, str):
            return None, warning
        try:
            datetime.fromisoformat(measured_at)
        except ValueError:
            return None, warning

    def optional_nonnegative_int(name: str, *, positive: bool = False) -> bool:
        value = row.get(name)
        if value is None:
            return True
        return (isinstance(value, int) and not isinstance(value, bool)
                and (value > 0 if positive else value >= 0))

    if not all((
        optional_nonnegative_int("sample_size", positive=True),
        optional_nonnegative_int("refused_n"),
        optional_nonnegative_int("errored_n"),
        optional_nonnegative_int("probe_context", positive=True),
        optional_nonnegative_int("generation_cap", positive=True),
        optional_nonnegative_int("repeat_count", positive=True),
        optional_nonnegative_int("total_generations", positive=True),
    )):
        return None, warning

    sample_size = row.get("sample_size")
    repeat_count = row.get("repeat_count")
    total = row.get("total_generations")
    structured = (row.get("probe_context"), row.get("generation_cap"), repeat_count,
                  total, row.get("run_scores_json"))
    if row.get("refused_n") is None or row.get("errored_n") is None:
        return None, warning
    if any(value is not None for value in structured) and any(value is None for value in structured):
        return None, warning
    if any(value is not None for value in structured) and measured_at is None:
        return None, warning
    if (total is not None and sample_size is not None and repeat_count is not None
            and total != sample_size * repeat_count):
        return None, warning
    if total is not None and sum(
            row.get(name) or 0 for name in ("refused_n", "errored_n")) >= total:
        return None, warning

    run_scores, run_warning = decode_run_scores(row.get("run_scores_json"), repeat_count)
    if run_warning is not None:
        return None, warning
    if run_scores is not None and not math.isclose(
            math.fsum(run_scores) / len(run_scores), float(score), abs_tol=1e-9):
        return None, warning

    return {
        "score": float(score), "source": row["source"],
        "sample_size": sample_size,
        "refused_n": row.get("refused_n"), "errored_n": row.get("errored_n"),
        "probe_context": row.get("probe_context"),
        "generation_cap": row.get("generation_cap"),
        "repeat_count": repeat_count, "total_generations": total,
        "run_scores": run_scores,
    }, None


def _normalise_imported(raw: dict) -> dict:
    """Fold a raw ``model_id`` → ``use_case`` → ``{score, source}`` table into ``base_key(model_id)``
    → ``use_case`` → ``{...}``, warning (never silently) when two distinct model_ids collide on the
    same base key — e.g. an ``mlx-community`` mirror and its ``meta-llama`` original both normalise
    to ``llama-3.2-3b-instruct``. The later entry wins (dict-overwrite order), but a dropped entry
    with no signal would be a lie by omission (Rule #3), so the collision is surfaced."""
    out: dict = {}
    for model_id, by_uc in raw.items():
        key = base_key(model_id)
        if key in out:
            warnings.warn(
                f"load_imported: {model_id!r} normalises to base key {key!r}, which another "
                "imported model_id already maps to — that entry's scores are being overwritten "
                "and dropped.",
                RuntimeWarning, stacklevel=2,
            )
        out[key] = by_uc
    return out


def load_imported() -> dict:
    """The shipped table of *imported* (published, not-measured-here) capability scores, keyed
    ``model_id`` → ``use_case`` → ``{score, source}``. Each leaf carries its citation; a
    locally-measured score overrides these when present (see :func:`score_for`)."""
    with _IMPORTED_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)["scores"]
    return _normalise_imported(raw)


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
