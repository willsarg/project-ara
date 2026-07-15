# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""scoring.py — engine-free use-case capability lookup for `recommend --use-case`.

Spec 2026-06-28-recommend-use-case-and-serve-selection. `recommend` stays engine-free
(closed decision, 2026-06-23-capability-pipeline): it READS a capability score that a
separate benchmark step measured (tier="measured"), or an imported published score
(tier="imported"), and labels the provenance. Absent → None (rendered `unknown`).
"""
from __future__ import annotations

import warnings

import pytest

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


def test_gguf_selector_has_canonical_repo_and_exact_quant():
    selector = "org/repo:Model-Q4_K_M.gguf"
    assert scoring.canonical_model_id(selector) == "org/repo"
    assert scoring.quant_key(selector) == "q4_k_m"
    assert scoring.canonical_model_id("org/plain") == "org/plain"


def test_measured_evidence_rejects_selector_quant_mismatch():
    selector = "org/repo:Model-Q4_K_M.gguf"
    row = {
        "model_id": selector, "use_case": "coding", "tier": "measured",
        "benchmark_id": "coding", "canonical_model_id": "org/repo",
        "base_model": scoring.base_key("org/repo"), "artifact_id": "artifact",
        "engine_key": "cpu", "backend": "cpu", "quant": "q8_0",
        "score": 0.5, "source": "probe", "max_score": None, "measured_at": None,
    }
    assert scoring.validate_measured_evidence(row)[0] is None
    row["quant"] = "q4_k_m"
    assert scoring.validate_measured_evidence(row)[0]["score"] == 0.5


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


def test_normalise_imported_warns_on_base_key_collision():
    # Two model_ids that normalise to the same base_key (e.g. an mlx-community mirror and its
    # meta-llama original) must not silently overwrite one another with no signal — Rule #3 says
    # unknown/dropped data must be surfaced, never hidden. The later entry wins (last write), but
    # a warning names the collision so it isn't a silent loss.
    raw = {
        "mlx-community/Llama-3.2-3B-Instruct": {"coding": {"score": 0.5, "source": "a"}},
        "meta-llama/Llama-3.2-3B-Instruct": {"coding": {"score": 0.7, "source": "b"}},
    }
    with pytest.warns(RuntimeWarning, match="Llama-3.2-3B-Instruct"):
        table = scoring._normalise_imported(raw)
    key = scoring.base_key("meta-llama/Llama-3.2-3B-Instruct")
    assert table.keys() == {key}
    assert table[key]["coding"]["score"] == 0.7  # last entry wins, not silently the first


def test_normalise_imported_no_warning_without_collision():
    # Distinct base keys → no warning, both entries preserved.
    raw = {
        "meta-llama/Llama-3.2-3B-Instruct": {"coding": {"score": 0.7, "source": "b"}},
        "Qwen/Qwen2.5-Coder-7B-Instruct": {"coding": {"score": 0.71, "source": "c"}},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table = scoring._normalise_imported(raw)
    assert len(table) == 2


def test_quant_key_extracts_genuine_quant_token():
    # quant_key returns the first genuine quant token, lowercased. Spec 2026-07-02-benchmark-honesty-persistence.
    assert scoring.quant_key("mlx-community/Llama-3.2-3B-Instruct-4bit") == "4bit"
    assert scoring.quant_key("org/Model-q4_k_m") == "q4_k_m"
    assert scoring.quant_key("org/Model-INT8") == "int8"
    assert scoring.quant_key("org/Model-AWQ") == "awq"
    assert scoring.quant_key("org/Model-fp16") == "fp16"


def test_quant_key_covers_gguf_iq_tq_and_f32_classes():
    # llama.cpp's importance-matrix (IQ*) and ternary (TQ*) quants are real quants the fleet
    # actually runs (IQ2_M, TQ1_0 — the 2026-06-29 campaign's top box reasoner was a TQ1_0),
    # and F32 is a full-precision label alongside F16. Spec 2026-07-02-benchmark-honesty-persistence.
    assert scoring.quant_key("org/DeepSeek-Coder-V2-Lite-IQ2_M") == "iq2_m"
    assert scoring.quant_key("org/Qwen3-Coder-30B-A3B-TQ1_0") == "tq1_0"
    assert scoring.quant_key("org/Model-F32") == "f32"


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


def test_measured_score_carries_structured_execution_provenance():
    measured = {("m", "coding"): {
        "score": 0.4, "source": "mlx probe", "sample_size": 100,
        "probe_context": 4096, "generation_cap": 512, "repeat_count": 3,
        "total_generations": 300, "run_scores": [0.3, 0.4, 0.5],
    }}
    s = scoring.score_for("m", "coding", measured=measured)
    assert s.probe_context == 4096
    assert s.generation_cap == 512
    assert s.repeat_count == 3
    assert s.total_generations == 300
    assert s.run_scores == (0.3, 0.4, 0.5)


def test_imported_score_has_none_partial_fields():
    # An imported score carries no sample-size / partial-count metadata (defaults None).
    # Spec 2026-07-02-benchmark-honesty-persistence.
    imported = {"m": {"coding": {"score": 0.5, "source": "leaderboard"}}}
    s = scoring.score_for("m", "coding", imported=imported)
    assert s.sample_size is None and s.refused_n is None and s.errored_n is None
    assert s.probe_context is None and s.generation_cap is None
    assert s.repeat_count is None and s.total_generations is None and s.run_scores is None


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


# --------------------------------------------------------------------------- #
# quant_bits — effective bit-width for ORDERING quant inversions
# Spec 2026-07-02-recommend-inversion-guard
# --------------------------------------------------------------------------- #
def test_quant_bits_full_and_half_precision_labels():
    # Float labels map to their nominal width; bf16 sits with the 16-bit family.
    assert scoring.quant_bits("f32") == 32
    assert scoring.quant_bits("fp32") == 32
    assert scoring.quant_bits("f16") == 16
    assert scoring.quant_bits("fp16") == 16
    assert scoring.quant_bits("bf16") == 16


def test_quant_bits_int_and_bit_families():
    # int8/8bit/q8_* → 8; int4/4bit/awq/gptq → 4; generic <N>bit → N.
    assert scoring.quant_bits("int8") == 8
    assert scoring.quant_bits("8bit") == 8
    assert scoring.quant_bits("q8_0") == 8
    assert scoring.quant_bits("int4") == 4
    assert scoring.quant_bits("4bit") == 4
    assert scoring.quant_bits("awq") == 4
    assert scoring.quant_bits("gptq") == 4
    assert scoring.quant_bits("6bit") == 6


def test_quant_bits_llama_cpp_q_and_iq_classes():
    # llama.cpp q<N>_* / iq<N>_* take their N as the effective width.
    assert scoring.quant_bits("q4_k_m") == 4
    assert scoring.quant_bits("iq2_m") == 2


def test_quant_bits_ternary_bpw():
    # llama.cpp ternary quants carry a fractional bits-per-weight (not an integer N).
    assert scoring.quant_bits("tq1_0") == 1.69
    assert scoring.quant_bits("tq2_0") == 2.06


def test_quant_bits_unknown_and_none():
    # An unrecognised token or None → None (never a guessed width — Rule #3).
    assert scoring.quant_bits("banana") is None
    assert scoring.quant_bits(None) is None


# --------------------------------------------------------------------------- #
# flag_inversions — disclose (never silently rank on) lower-precision upsets
# Spec 2026-07-02-recommend-inversion-guard
# --------------------------------------------------------------------------- #
def _inv_recs():
    # Two quants of the SAME base model, both measured here on the same probe set.
    return [{"model_id": "org/Model-4bit", "est_context": 8000},
            {"model_id": "org/Model-8bit", "est_context": 8000}]


def test_inversion_flagged_both_directions_within_noise():
    # A lower-precision quant outscoring a higher one within statistical noise is DISCLOSED on
    # both entries, naming the other quant — not reordered.
    measured = {("org/Model-4bit", "coding"): {"score": 0.098, "source": "p", "sample_size": 50},
                ("org/Model-8bit", "coding"): {"score": 0.061, "source": "p", "sample_size": 50}}
    ranked = scoring.rank(_inv_recs(), "coding", measured=measured)
    by_id = {r["model_id"]: r["score"] for r in ranked}
    assert by_id["org/Model-4bit"].inversion == "outscores 8bit within noise"
    assert by_id["org/Model-8bit"].inversion == "outscored by 4bit within noise"


def test_inversion_outside_noise_drops_the_within_noise_wording():
    # The same upset, but with large samples the gap clears the noise band → firmer wording.
    measured = {("org/Model-4bit", "coding"): {"score": 0.098, "source": "p", "sample_size": 10000},
                ("org/Model-8bit", "coding"): {"score": 0.061, "source": "p", "sample_size": 10000}}
    ranked = scoring.rank(_inv_recs(), "coding", measured=measured)
    by_id = {r["model_id"]: r["score"] for r in ranked}
    assert by_id["org/Model-4bit"].inversion == "outscores 8bit"
    assert by_id["org/Model-8bit"].inversion == "outscored by 4bit"


def test_inversion_se_zero_is_within_noise():
    # Degenerate variances (values at 0/1) give se == 0; that is treated as within noise.
    measured = {("org/Model-4bit", "coding"): {"score": 1.0, "source": "p", "sample_size": 20},
                ("org/Model-8bit", "coding"): {"score": 0.0, "source": "p", "sample_size": 20}}
    ranked = scoring.rank(_inv_recs(), "coding", measured=measured)
    by_id = {r["model_id"]: r["score"] for r in ranked}
    assert by_id["org/Model-4bit"].inversion == "outscores 8bit within noise"
    assert by_id["org/Model-8bit"].inversion == "outscored by 4bit within noise"


def test_non_inverted_pair_not_flagged():
    # The expected ordering (higher precision scores higher) is NOT an inversion.
    measured = {("org/Model-4bit", "coding"): {"score": 0.30, "source": "p", "sample_size": 50},
                ("org/Model-8bit", "coding"): {"score": 0.90, "source": "p", "sample_size": 50}}
    ranked = scoring.rank(_inv_recs(), "coding", measured=measured)
    for r in ranked:
        assert r["score"].inversion is None


def test_inversion_skips_unknown_quant_imported_and_missing_sample_size():
    # Eligibility: only measured-tier entries with a known quant AND a sample_size participate.
    recs = [{"model_id": "org/Model-4bit", "est_context": 8000},   # measured, no sample_size
            {"model_id": "org/Model", "est_context": 8000},        # measured, unknown quant
            {"model_id": "org/Model-8bit", "est_context": 8000}]   # imported tier
    measured = {("org/Model-4bit", "coding"): {"score": 0.90, "source": "p"},         # sample None
                ("org/Model", "coding"): {"score": 0.95, "source": "p", "sample_size": 50}}
    imported = {"model": {"coding": {"score": 0.10, "source": "leaderboard"}}}
    ranked = scoring.rank(recs, "coding", measured=measured, imported=imported)
    for r in ranked:
        assert r["score"].inversion is None


def test_inversion_does_not_change_ranking_order():
    # DISCLOSURE, not reordering: the (statistically dubious) 4bit-first order is preserved.
    measured = {("org/Model-4bit", "coding"): {"score": 0.098, "source": "p", "sample_size": 50},
                ("org/Model-8bit", "coding"): {"score": 0.061, "source": "p", "sample_size": 50}}
    ranked = scoring.rank(_inv_recs(), "coding", measured=measured)
    assert [r["model_id"] for r in ranked] == ["org/Model-4bit", "org/Model-8bit"]


def test_inversion_ignores_same_bit_width_pair():
    # Two quants of the same effective width (4bit vs int4) can't invert each other — skipped.
    recs = [{"model_id": "org/Model-4bit", "est_context": 8000},
            {"model_id": "org/Model-int4", "est_context": 8000}]
    measured = {("org/Model-4bit", "coding"): {"score": 0.90, "source": "p", "sample_size": 50},
                ("org/Model-int4", "coding"): {"score": 0.30, "source": "p", "sample_size": 50}}
    ranked = scoring.rank(recs, "coding", measured=measured)
    for r in ranked:
        assert r["score"].inversion is None


def test_inversion_keeps_first_when_entry_in_several():
    # An entry inverting against two higher-precision siblings keeps the FIRST message only.
    recs = [{"model_id": "org/Model-4bit", "est_context": 8000},
            {"model_id": "org/Model-8bit", "est_context": 8000},
            {"model_id": "org/Model-fp16", "est_context": 8000}]
    measured = {("org/Model-4bit", "coding"): {"score": 0.90, "source": "p", "sample_size": 50},
                ("org/Model-8bit", "coding"): {"score": 0.88, "source": "p", "sample_size": 50},
                ("org/Model-fp16", "coding"): {"score": 0.85, "source": "p", "sample_size": 50}}
    ranked = scoring.rank(recs, "coding", measured=measured)
    by_id = {r["model_id"]: r["score"] for r in ranked}
    # 4bit beats both 8bit and fp16; only the first pairing (vs 8bit) is recorded.
    assert by_id["org/Model-4bit"].inversion == "outscores 8bit within noise"
