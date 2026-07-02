# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""scoring.py — engine-free use-case capability lookup for `recommend --use-case`.

Spec 2026-06-28-recommend-use-case-and-serve-selection. `recommend` stays engine-free
(closed decision, 2026-06-23-capability-pipeline): it READS a capability score that a
separate benchmark step measured (tier="measured"), or an imported published score
(tier="imported"), and labels the provenance. Absent → None (rendered `unknown`).
"""
from __future__ import annotations

from ara import scoring


def test_score_for_imported_returns_labelled_score():
    # An imported benchmark score for a known model + use-case is returned, labelled imported.
    imported = {"Qwen/Qwen2.5-Coder-7B-Instruct":
                {"coding": {"score": 0.71, "source": "HumanEval (EvalPlus, 2026-05)"}}}
    s = scoring.score_for("Qwen/Qwen2.5-Coder-7B-Instruct", "coding", imported=imported)
    assert s.tier == "imported"
    assert s.value == 0.71
    assert "HumanEval" in s.source


def test_score_for_unknown_model_returns_none():
    # No score anywhere → None (the caller renders `unknown`; never a guessed rank — Rule #3).
    assert scoring.score_for("who/knows", "coding", imported={}) is None


def test_measured_score_beats_imported():
    # A locally-measured score (on the actual quant) wins over an imported leaderboard number.
    imported = {"m": {"coding": {"score": 0.50, "source": "leaderboard"}}}
    measured = {("m", "coding"): {"score": 0.61, "source": "wmx/apple Q4"}}
    s = scoring.score_for("m", "coding", imported=imported, measured=measured)
    assert s.tier == "measured"
    assert s.value == 0.61


def test_rank_orders_by_capability_score_unknowns_last():
    # With --use-case, candidates rank by capability; models with no score sort last (not dropped).
    recs = [
        {"model_id": "a", "est_context": 5000},
        {"model_id": "b", "est_context": 9000},   # no score
        {"model_id": "c", "est_context": 7000},
    ]
    imported = {"a": {"coding": {"score": 0.8, "source": "x"}},
                "c": {"coding": {"score": 0.6, "source": "x"}}}
    ranked = scoring.rank(recs, "coding", imported=imported)
    assert [r["model_id"] for r in ranked] == ["a", "c", "b"]
    assert ranked[0]["score"].tier == "imported"
    assert ranked[-1]["score"] is None


def test_base_key_strips_org_and_quant():
    # A quant variant and its base resolve to the SAME key, so an imported base score matches a
    # locally-cataloged quant (the mlx-community/...-4bit reality on this Mac).
    assert scoring.base_key("mlx-community/Llama-3.2-3B-Instruct-4bit") == "llama-3.2-3b-instruct"
    assert scoring.base_key("meta-llama/Llama-3.2-3B-Instruct") == "llama-3.2-3b-instruct"


def test_score_for_matches_quant_variant_via_base_key():
    imported = {scoring.base_key("meta-llama/Llama-3.2-3B-Instruct"):
                {"coding": {"score": 0.45, "source": "HumanEval"}}}
    s = scoring.score_for("mlx-community/Llama-3.2-3B-Instruct-4bit", "coding", imported=imported)
    assert s is not None and s.value == 0.45


def test_load_imported_keys_are_base_normalised():
    # The shipped table is keyed in base form so quant variants in the local catalog can match.
    table = scoring.load_imported()
    for k in table:
        assert k == scoring.base_key(k)


def test_quant_key_extracts_genuine_quant_token():
    # quant_key returns the first genuine quant token, lowercased. Spec 2026-07-02-benchmark-honesty-persistence.
    assert scoring.quant_key("mlx-community/Llama-3.2-3B-Instruct-4bit") == "4bit"
    assert scoring.quant_key("org/Model-q4_k_m") == "q4_k_m"
    assert scoring.quant_key("org/Model-INT8") == "int8"
    assert scoring.quant_key("org/Model-AWQ") == "awq"
    assert scoring.quant_key("org/Model-fp16") == "fp16"


def test_quant_key_ignores_container_formats_and_returns_none():
    # gguf / mlx are container formats, NOT quants — never returned; no quant → None.
    # Spec 2026-07-02-benchmark-honesty-persistence.
    assert scoring.quant_key("mlx-community/Llama-3.2-3B-Instruct") is None
    assert scoring.quant_key("org/Model-gguf") is None
    assert scoring.quant_key("meta-llama/Llama-3.2-3B-Instruct") is None


def test_quant_key_returns_quant_not_format_when_both_present():
    # A model tagged with both a quant and a container format returns only the quant token.
    # Spec 2026-07-02-benchmark-honesty-persistence.
    assert scoring.quant_key("org/Model-q4_k_m-gguf") == "q4_k_m"


def test_base_key_unchanged_by_quant_key_addition():
    # quant_key must not perturb base_key's stripping behaviour (byte-identical).
    assert scoring.base_key("mlx-community/Llama-3.2-3B-Instruct-4bit-gguf") == "llama-3.2-3b-instruct"


def test_measured_score_carries_sample_size_and_partial_counts():
    # A measured Score surfaces its sample size + refusal/error counts so recommend can annotate
    # partial/low-confidence runs (Rule #3). Spec 2026-07-02-benchmark-honesty-persistence.
    measured = {("m", "coding"): {"score": 0.4, "source": "wmx probe",
                                  "sample_size": 30, "refused_n": 3, "errored_n": 2}}
    s = scoring.score_for("m", "coding", measured=measured)
    assert s.sample_size == 30 and s.refused_n == 3 and s.errored_n == 2


def test_imported_score_has_none_partial_fields():
    # An imported score carries no sample-size / partial-count metadata (defaults None).
    # Spec 2026-07-02-benchmark-honesty-persistence.
    imported = {"m": {"coding": {"score": 0.5, "source": "leaderboard"}}}
    s = scoring.score_for("m", "coding", imported=imported)
    assert s.sample_size is None and s.refused_n is None and s.errored_n is None


def test_measured_score_missing_partial_fields_default_none():
    # A legacy measured hit without the new keys yields None fields (no KeyError).
    # Spec 2026-07-02-benchmark-honesty-persistence.
    measured = {("m", "coding"): {"score": 0.6, "source": "wmx probe"}}
    s = scoring.score_for("m", "coding", measured=measured)
    assert s.sample_size is None and s.refused_n is None and s.errored_n is None


def test_load_imported_ships_cited_normalised_scores():
    # The shipped imported-score table is non-empty and every leaf carries a citation + a
    # normalised 0..1 score — provenance is mandatory (never an unattributed number).
    table = scoring.load_imported()
    assert table
    leaves = [(m, uc, e) for m, by_uc in table.items() for uc, e in by_uc.items()]
    assert leaves
    for _m, uc, entry in leaves:
        assert uc in scoring.USE_CASES
        assert "source" in entry and entry["source"]
        assert 0.0 <= entry["score"] <= 1.0
