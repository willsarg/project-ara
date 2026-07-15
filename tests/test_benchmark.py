# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Tests for ara.benchmark — judge-free benchmark data and scoring.

2026-06-28-benchmark-layer
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# ── interface existence ──────────────────────────────────────────────────────

def test_use_cases_constant():
    from ara.benchmark import USE_CASES
    assert set(USE_CASES) == {"coding", "reasoning", "agentic", "extraction", "rag"}


def test_load_probe_returns_nonempty_for_all_use_cases():
    from ara.benchmark import USE_CASES, load_probe
    for uc in USE_CASES:
        items = load_probe(uc)
        assert len(items) > 0, f"load_probe('{uc}') returned empty list"


# ── probe file schema ────────────────────────────────────────────────────────

def test_coding_probe_has_required_keys():
    from ara.benchmark import load_probe
    for item in load_probe("coding"):
        assert "task_id" in item
        assert "prompt" in item
        assert "entry_point" in item
        assert "test" in item
        assert "canonical_solution" not in item  # must NOT be included


def test_reasoning_probe_has_required_keys():
    from ara.benchmark import load_probe
    for item in load_probe("reasoning"):
        assert "question" in item
        assert "answer" in item
        assert "####" in item["answer"]  # GSM8K suffix preserved


def test_agentic_probe_has_required_keys():
    from ara.benchmark import load_probe
    for item in load_probe("agentic"):
        assert "id" in item
        assert "question" in item
        assert "function" in item
        assert "expected" in item
        assert "name" in item["expected"]
        assert "arguments" in item["expected"]


def test_extraction_probe_has_required_keys():
    from ara.benchmark import load_probe
    for item in load_probe("extraction"):
        assert "context" in item
        assert "question" in item
        assert "answers" in item
        for a in item["answers"]:
            assert "text" in a


def test_rag_probe_has_required_keys():
    from ara.benchmark import load_probe
    for item in load_probe("rag"):
        assert "context" in item
        assert "question" in item
        assert "answers" in item


def test_probe_counts():
    from ara.benchmark import load_probe
    assert len(load_probe("coding")) == 164
    assert len(load_probe("reasoning")) == 100
    assert len(load_probe("agentic")) == 100
    assert len(load_probe("extraction")) == 100
    assert len(load_probe("rag")) == 100


# ── prompt_for includes question/prompt text ─────────────────────────────────

def test_prompt_for_coding_includes_prompt_text():
    from ara.benchmark import load_probe, prompt_for
    item = load_probe("coding")[0]
    p = prompt_for("coding", item)
    assert item["prompt"] in p or item["entry_point"] in p


def test_prompt_for_reasoning_includes_question():
    from ara.benchmark import load_probe, prompt_for
    item = load_probe("reasoning")[0]
    p = prompt_for("reasoning", item)
    assert item["question"] in p


def test_prompt_for_agentic_includes_question():
    from ara.benchmark import load_probe, prompt_for
    item = load_probe("agentic")[0]
    p = prompt_for("agentic", item)
    assert item["question"] in p


def test_prompt_for_extraction_includes_question():
    from ara.benchmark import load_probe, prompt_for
    item = load_probe("extraction")[0]
    p = prompt_for("extraction", item)
    assert item["question"] in p


def test_prompt_for_rag_includes_question():
    from ara.benchmark import load_probe, prompt_for
    item = load_probe("rag")[0]
    p = prompt_for("rag", item)
    assert item["question"] in p


# ── reasoning scorer ─────────────────────────────────────────────────────────

def test_reasoning_correct_scores_1():
    from ara.benchmark import score
    item = {"question": "What is 2+2?", "answer": "Some reasoning.\n#### 4"}
    assert score("reasoning", item, "Let me think... Answer: 4") == 1.0


def test_reasoning_wrong_scores_0():
    from ara.benchmark import score
    item = {"question": "What is 2+2?", "answer": "Some reasoning.\n#### 4"}
    assert score("reasoning", item, "Answer: 5") == 0.0


def test_reasoning_answer_with_commas():
    """GSM8K numbers can have commas like '1,234'."""
    from ara.benchmark import score
    item = {"question": "q", "answer": "#### 1234"}
    assert score("reasoning", item, "Answer: 1,234") == 1.0


# ── agentic scorer ───────────────────────────────────────────────────────────

def test_agentic_correct_scores_1():
    from ara.benchmark import score
    item = {
        "id": "t1", "question": "q", "function": {},
        "expected": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}},
    }
    completion = '{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}'
    assert score("agentic", item, completion) == 1.0


def test_agentic_wrong_name_scores_0():
    from ara.benchmark import score
    item = {
        "id": "t1", "question": "q", "function": {},
        "expected": {"name": "get_weather", "arguments": {"city": "Paris"}},
    }
    completion = '{"name": "get_forecast", "arguments": {"city": "Paris"}}'
    assert score("agentic", item, completion) == 0.0


def test_agentic_wrong_arg_scores_0():
    from ara.benchmark import score
    item = {
        "id": "t1", "question": "q", "function": {},
        "expected": {"name": "add", "arguments": {"a": 1, "b": 2}},
    }
    completion = '{"name": "add", "arguments": {"a": 1, "b": 99}}'
    assert score("agentic", item, completion) == 0.0


def test_agentic_numeric_tolerance():
    # Exercise the ±1e-4 band in BOTH directions — identical values would never enter the
    # tolerance branch (an exact-only scorer would pass too, proving nothing about tolerance).
    from ara.benchmark import score
    item = {
        "id": "t1", "question": "q", "function": {},
        "expected": {"name": "calc", "arguments": {"x": 3.14159}},
    }
    within = '{"name": "calc", "arguments": {"x": 3.1416}}'    # diff 1e-5 < 1e-4 → match
    outside = '{"name": "calc", "arguments": {"x": 3.1417}}'   # diff 1.1e-4 > 1e-4 → no match
    assert score("agentic", item, within) == 1.0
    assert score("agentic", item, outside) == 0.0


def test_agentic_case_insensitive_string():
    from ara.benchmark import score
    item = {
        "id": "t1", "question": "q", "function": {},
        "expected": {"name": "search", "arguments": {"query": "Python"}},
    }
    completion = '{"name": "search", "arguments": {"query": "python"}}'
    assert score("agentic", item, completion) == 1.0


def test_agentic_strips_markdown_fences():
    from ara.benchmark import score
    item = {
        "id": "t1", "question": "q", "function": {},
        "expected": {"name": "get_weather", "arguments": {"city": "London"}},
    }
    completion = '```json\n{"name": "get_weather", "arguments": {"city": "London"}}\n```'
    assert score("agentic", item, completion) == 1.0


def test_agentic_invalid_json_scores_0():
    from ara.benchmark import score
    item = {
        "id": "t1", "question": "q", "function": {},
        "expected": {"name": "fn", "arguments": {}},
    }
    assert score("agentic", item, "not json at all") == 0.0


def test_agentic_non_object_arguments_score_0_without_crashing():
    from ara.benchmark import score
    item = {
        "id": "t1", "question": "q", "function": {},
        "expected": {"name": "fn", "arguments": {"x": 1}},
    }
    assert score("agentic", item, '{"name": "fn", "arguments": []}') == 0.0


def test_agentic_extra_arguments_fail_exact_match():
    from ara.benchmark import score
    item = {
        "id": "t1", "question": "q", "function": {},
        "expected": {"name": "fn", "arguments": {"x": 1}},
    }
    completion = '{"name": "fn", "arguments": {"x": 1, "danger": true}}'
    assert score("agentic", item, completion) == 0.0


def test_agentic_null_argument_fails_exact_match():
    from ara.benchmark import score
    item = {
        "id": "t1", "question": "q", "function": {},
        "expected": {"name": "fn", "arguments": {"x": 1}},
    }
    assert score("agentic", item, '{"name": "fn", "arguments": {"x": null}}') == 0.0


@pytest.mark.parametrize("expected,predicted", [
    (1, True),
    (True, 1),
    (True, "true"),
    ([1, 2], "[1, 2]"),
    ({"x": 1}, "{'x': 1}"),
])
def test_agentic_argument_types_must_match(expected, predicted):
    from ara.benchmark import score
    item = {
        "id": "t1", "question": "q", "function": {},
        "expected": {"name": "fn", "arguments": {"value": expected}},
    }
    completion = json.dumps({"name": "fn", "arguments": {"value": predicted}})
    assert score("agentic", item, completion) == 0.0


def test_agentic_nested_arguments_match_structurally():
    from ara.benchmark import score
    value = {"enabled": True, "items": [1, "Two", {"ratio": 3.0, "note": None}]}
    item = {
        "id": "t1", "question": "q", "function": {},
        "expected": {"name": "fn", "arguments": {"value": value}},
    }
    predicted = {"enabled": True, "items": [1.00001, "two", {"ratio": 3, "note": None}]}
    completion = json.dumps({"name": "FN", "arguments": {"value": predicted}})
    assert score("agentic", item, completion) == 1.0


# ── extraction scorer ─────────────────────────────────────────────────────────

def test_extraction_exact_match_scores_1():
    from ara.benchmark import score
    item = {
        "context": "The sky is blue.",
        "question": "What color is the sky?",
        "answers": [{"text": "blue"}],
    }
    assert score("extraction", item, "blue") == 1.0


def test_extraction_wrong_scores_0():
    from ara.benchmark import score
    item = {
        "context": "The sky is blue.",
        "question": "What color is the sky?",
        "answers": [{"text": "blue"}],
    }
    assert score("extraction", item, "green") == 0.0


def test_extraction_article_normalization():
    from ara.benchmark import score
    item = {
        "context": "The cat sat on the mat.",
        "question": "What did the cat sit on?",
        "answers": [{"text": "the mat"}],
    }
    # "the mat" normalized → "mat"; completion "mat" → normalized "mat": exact match
    assert score("extraction", item, "mat") == 1.0


def test_extraction_token_f1_partial():
    from ara.benchmark import score
    item = {
        "context": "Apollo 11 landed on the moon in July 1969.",
        "question": "When did Apollo 11 land?",
        "answers": [{"text": "July 1969"}],
    }
    # partial overlap should give F1 > 0 but < 1
    result = score("extraction", item, "1969")
    assert 0.0 < result < 1.0


# ── rag scorer ────────────────────────────────────────────────────────────────

def test_rag_exact_match_scores_1():
    from ara.benchmark import score
    item = {
        "context": "Paris is the capital of France.",
        "question": "What is the capital of France?",
        "answers": [{"text": "Paris"}],
    }
    assert score("rag", item, "Paris") == 1.0


def test_rag_wrong_scores_0():
    from ara.benchmark import score
    item = {
        "context": "Paris is the capital of France.",
        "question": "What is the capital of France?",
        "answers": [{"text": "Paris"}],
    }
    assert score("rag", item, "Berlin") == 0.0


def test_rag_multiple_accepted_answers():
    from ara.benchmark import score
    item = {
        "context": "The treaty was signed in 1783.",
        "question": "When was the treaty signed?",
        "answers": [{"text": "1783"}, {"text": "seventeen eighty-three"}],
    }
    assert score("rag", item, "1783") == 1.0


# ── coding scorer (subprocess) ────────────────────────────────────────────────

_TINY_CODING_ITEM = {
    "task_id": "test/0",
    "prompt": "def add(a, b):\n",
    "entry_point": "add",
    # Real HumanEval shape: a `check(candidate)` the scorer must invoke (bare asserts here
    # would never exercise the candidate and would mask a scorer that doesn't call check).
    "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n",
}


def test_coding_correct_scores_1():
    from ara.benchmark import score
    assert score("coding", _TINY_CODING_ITEM, "    return a + b\n") == 1.0


def test_coding_wrong_scores_0():
    from ara.benchmark import score
    assert score("coding", _TINY_CODING_ITEM, "    return a - b\n") == 0.0


def test_coding_timeout_scores_0(monkeypatch):
    from ara import benchmark
    monkeypatch.setattr(benchmark, "_CODING_TIMEOUT_SECONDS", 0.05)
    item = {
        "task_id": "test/hang",
        "prompt": "def hang():\n",
        "entry_point": "hang",
        "test": "hang()\n",
    }
    assert benchmark.score("coding", item, "    while True: pass\n") == 0.0


# ── additional edge-case tests for full branch coverage ──────────────────────

def test_reasoning_no_answer_marker_scores_0():
    """Completion with no 'Answer: N' pattern → 0.0."""
    from ara.benchmark import score
    item = {"question": "q", "answer": "#### 42"}
    assert score("reasoning", item, "I cannot determine the answer.") == 0.0


def test_agentic_missing_arg_scores_0():
    """Expected arg key is absent from completion → 0.0."""
    from ara.benchmark import score
    item = {
        "id": "t1", "question": "q", "function": {},
        "expected": {"name": "fn", "arguments": {"required": "value"}},
    }
    assert score("agentic", item, '{"name": "fn", "arguments": {}}') == 0.0


def test_agentic_wrong_string_arg_scores_0():
    """Correct name but wrong string arg value → 0.0."""
    from ara.benchmark import score
    item = {
        "id": "t1", "question": "q", "function": {},
        "expected": {"name": "lookup", "arguments": {"city": "Paris"}},
    }
    assert score("agentic", item, '{"name": "lookup", "arguments": {"city": "Berlin"}}') == 0.0


def test_agentic_nonnumeric_for_numeric_arg_scores_0():
    """Completion supplies a non-numeric string for a numeric arg → 0.0."""
    from ara.benchmark import score
    item = {
        "id": "t1", "question": "q", "function": {},
        "expected": {"name": "fn", "arguments": {"n": 42}},
    }
    assert score("agentic", item, '{"name": "fn", "arguments": {"n": "not_a_number"}}') == 0.0


def test_agentic_no_args_matches_name_scores_1():
    """Function with no required arguments — name match alone → 1.0."""
    from ara.benchmark import score
    item = {
        "id": "t1", "question": "q", "function": {},
        "expected": {"name": "reset", "arguments": {}},
    }
    assert score("agentic", item, '{"name": "reset", "arguments": {}}') == 1.0


# ── score_probe_set ───────────────────────────────────────────────────────────

def test_score_probe_set_averages():
    from ara.benchmark import score_probe_set
    items = [
        {"question": "q", "answer": "#### 1"},
        {"question": "q", "answer": "#### 2"},
    ]
    completions = ["Answer: 1", "Answer: 9"]  # 1 correct, 1 wrong → 0.5
    result = score_probe_set("reasoning", items, completions)
    assert abs(result - 0.5) < 1e-9


def test_score_probe_set_length_mismatch_raises():
    from ara.benchmark import score_probe_set
    with pytest.raises(ValueError):
        score_probe_set("reasoning", [{"question": "q", "answer": "#### 1"}], [])


# ── probe file raw JSON sanity ────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent / "ara" / "data" / "benchmarks"


@pytest.mark.parametrize("filename,expected_count", [
    ("humaneval.json", 164),
    ("gsm8k.json", 100),
    ("bfcl_simple.json", 100),
    ("extraction.json", 100),
    ("rag.json", 100),
])
def test_probe_json_file_exists_and_parses(filename, expected_count):
    path = _DATA_DIR / filename
    assert path.exists(), f"{path} does not exist"
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    assert len(data) == expected_count


# ── audit regression tests (2026-06-28 code audit) ───────────────────────────
# These load the REAL shipped probe data so a scorer that doesn't actually run the
# test (or that false-passes) cannot make them green.
from ara import benchmark as _bm  # noqa: E402

_CORRECT_HAS_CLOSE = (
    "    for i in range(len(numbers)):\n"
    "        for j in range(i + 1, len(numbers)):\n"
    "            if abs(numbers[i] - numbers[j]) < threshold:\n"
    "                return True\n"
    "    return False\n"
)


def _coding_item0():
    it = _bm.load_probe("coding")[0]
    assert it["entry_point"] == "has_close_elements"  # guard: dataset shape assumption
    return it


def test_coding_wrong_completion_scores_zero_on_real_data():
    # CRITICAL: a wrong body must NOT score 1.0. (Bug: check() was never called.)
    assert _bm.score("coding", _coding_item0(), "    return None\n") == 0.0


def test_coding_correct_completion_scores_one_on_real_data():
    assert _bm.score("coding", _coding_item0(), _CORRECT_HAS_CLOSE) == 1.0


def test_coding_strips_code_fences_preserving_indent():
    fenced = "```python\n" + _CORRECT_HAS_CLOSE + "```"
    assert _bm.score("coding", _coding_item0(), fenced) == 1.0


def test_reasoning_does_not_crash_on_range_string():
    assert _bm.score("reasoning", {"answer": "#### 42"}, "Answer: 3-4") == 0.0


def test_reasoning_takes_last_answer_not_first():
    out = "First I guessed Answer: 50, but rechecking: Answer: 42"
    assert _bm.score("reasoning", {"answer": "#### 42"}, out) == 1.0


def test_agentic_parses_json_after_preamble():
    item = {"expected": {"name": "f", "arguments": {"x": 1}}}
    assert _bm.score("agentic", item, 'Sure, here you go: {"name":"f","arguments":{"x":1}}') == 1.0


def test_agentic_scores_first_json_object_when_another_follows():
    item = {"expected": {"name": "f", "arguments": {"x": 1}}}
    completion = (
        '{"name":"f","arguments":{"x":1}}\n'
        '{"name":"different","arguments":{"x":2}}'
    )
    assert _bm.score("agentic", item, completion) == 1.0


def test_agentic_skips_malformed_braced_text_before_first_decodable_object():
    item = {"expected": {"name": "f", "arguments": {"x": 1}}}
    completion = (
        'Analysis {not valid JSON}: use the requested call.\n'
        '```json\n{"name":"f","arguments":{"x":1}}\n```\n'
        'Trailing note with {more prose}.'
    )
    assert _bm.score("agentic", item, completion) == 1.0


def test_score_probe_set_empty_returns_zero_not_crash():
    assert _bm.score_probe_set("coding", [], []) == 0.0


def test_methodology_id_binds_probe_prompt_and_scorer_contract(monkeypatch):
    items = [{"question": "2 + 2?", "answer": "#### 4"}]
    baseline = _bm.methodology_id("reasoning", items)
    assert baseline == _bm.methodology_id("reasoning", items)
    assert baseline.startswith("sha256:")
    assert baseline != _bm.methodology_id(
        "reasoning", [{"question": "3 + 3?", "answer": "#### 6"}])
    monkeypatch.setitem(_bm._SCORER_VERSIONS, "reasoning", "2")
    assert baseline != _bm.methodology_id("reasoning", items)


def test_token_f1_counts_repeated_tokens():
    # "cat cat" vs "cat cat cat": Counter intersection = 2, not set's 1.
    f1 = _bm._token_f1("cat cat", "cat cat cat")
    assert f1 == pytest.approx(2 * (2 / 2) * (2 / 3) / ((2 / 2) + (2 / 3)))


def test_extraction_answers_are_extractable_from_context():
    # Every accepted answer must be a normalized span of the passage (caches a "2" vs "two" bug).
    items = _bm.load_probe("extraction")
    bad = []
    for i, it in enumerate(items):
        ctx = _bm._normalize_text(it["context"])
        for a in it["answers"]:
            if _bm._normalize_text(a["text"]) not in ctx:
                bad.append((i, a["text"]))
    assert not bad, f"non-extractable extraction answers: {bad}"


def _flag_list_wrapped(name, val, schema, item_id, bad):
    # A list value is only a bug when the declared type is SCALAR (a BFCL flatten artifact);
    # genuine `array`-typed args (poker hands, number lists) are correct and must NOT be flagged.
    t = schema.get("type")
    if t in ("array", "list"):
        return
    if t == "dict" and isinstance(val, dict):
        nested = schema.get("properties", {})
        for k, v in val.items():
            _flag_list_wrapped(f"{name}.{k}", v, nested.get(k, {}), item_id, bad)
        return
    if isinstance(val, list):
        bad.append((item_id, name))


def test_bfcl_scalar_args_not_list_wrapped():
    # A scalar param whose expected value is a list = BFCL flatten artifact; the agentic scorer
    # would then compare list-vs-scalar and score every correct call 0. (array-typed args are fine.)
    items = _bm.load_probe("agentic")
    bad = []
    for it in items:
        props = it["function"]["parameters"].get("properties", {})
        for arg, val in it["expected"]["arguments"].items():
            _flag_list_wrapped(arg, val, props.get(arg, {}), it["id"], bad)
    assert not bad, f"scalar args wrapped in lists (flatten artifact): {bad}"


# ── sandbox containment (live, macOS) ────────────────────────────────────────
@pytest.mark.skipif(not _bm._SANDBOX_EXEC, reason="sandbox-exec is macOS-only")
def test_coding_sandbox_blocks_filesystem_write():
    import os as _os
    sentinel = "/tmp/ara_sb_sentinel_fswrite"
    if _os.path.exists(sentinel):
        _os.remove(sentinel)
    item = {"prompt": "def f():\n", "entry_point": "f", "test": "def check(c):\n    c()\n"}
    _bm.score("coding", item, f"    open({sentinel!r}, 'w').write('escaped')\n")
    try:
        assert not _os.path.exists(sentinel), "sandbox FAILED to block a filesystem write"
    finally:
        if _os.path.exists(sentinel):
            _os.remove(sentinel)


# ── canonical scorer swap (2026-06-29-canonical-scorer-swap) ─────────────────
# Failing-first tests for the three grading bugs the validation sweep exposed.

def test_rag_verbose_answer_gets_f1_credit():
    # Bug: _score_rag did normalized EXACT-MATCH only → any verbose answer scored 0 (flat-zero
    # sweep result). Canonical SQuAD token-F1 must give partial credit.
    item = {"context": "Paris is the capital of France.",
            "question": "What is the capital of France?",
            "answers": [{"text": "Paris"}]}
    assert _bm.score("rag", item, "The capital is Paris.") > 0.0


def test_coding_full_function_with_prose_no_fence_scores_1():
    # Bug: a chat model emits a full `def` wrapped in prose with no fence; the naive extractor
    # prepended the prose to the signature → SyntaxError → false 0 (the coding-inversion mechanism:
    # a more verbose/better model is penalised). Robust AST extraction must score it 1.0.
    completion = "Here is the solution:\n\ndef add(a, b):\n    return a + b\n"
    assert _bm.score("coding", _TINY_CODING_ITEM, completion) == 1.0


def test_coding_fenced_full_function_scores_1():
    # A FENCED full function (not a bare body): the naive path re-defined the signature on top of
    # the prompt's → SyntaxError. Must detect the full def and score 1.0.
    completion = "```python\ndef add(a, b):\n    return a + b\n```"
    assert _bm.score("coding", _TINY_CODING_ITEM, completion) == 1.0


def test_reasoning_bare_last_number_scores_1():
    # Canonical GSM8K flexible-extract: a model that reasons to a final number WITHOUT the
    # "Answer:" marker must still be scored (the leaderboard-reported metric).
    item = {"question": "q", "answer": "#### 42"}
    assert _bm.score("reasoning", item, "First 20, then 22, so the total is 42.") == 1.0


def test_extract_json_braces_present_but_unparseable_returns_none():
    # A `{...}` span that isn't valid JSON → None (not a crash), so agentic scores 0 cleanly.
    assert _bm._extract_json("here you go: {name: bad, no quotes} end") is None


def test_coding_without_sandbox_warns_and_still_scores(monkeypatch):
    # Off-macOS fallback (no sandbox-exec): a LOUD RuntimeWarning + process-isolation only — never a
    # silent downgrade. Runs our own trusted "return a + b", so executing un-sandboxed here is safe.
    monkeypatch.setattr(_bm, "_SANDBOX_EXEC", None)
    with pytest.warns(RuntimeWarning, match="WITHOUT an OS sandbox"):
        assert _bm.score("coding", _TINY_CODING_ITEM, "    return a + b\n") == 1.0


def test_canon_f1_both_empty_after_normalization_scores_1():
    # Two answers that normalize to empty (pure articles/punct) are identical → 1.0, not a crash.
    from ara import _canonical_scoring as _c
    assert _c.squad_f1("the", "a") == 1.0
    assert _c.squad_f1("", "Paris") == 0.0


def test_canon_to_number_rejects_garbage():
    # A flexible-extract match that isn't a real number ("$,." → stripped to "") → None, not crash.
    from ara import _canonical_scoring as _c
    assert _c._to_number("$,.") is None
    assert _c.gsm8k_extract_number("the price is $,. today") is None


def test_canon_largest_parseable_empty_for_prose_only():
    # A completion with no parseable Python at all → "" (and a full-def scorer path → score 0).
    from ara import _canonical_scoring as _c
    assert _c._largest_parseable("   \n  ") == ""


def test_reasoning_gold_without_marker_scores_0():
    # An item whose answer has no '#### N' gold → 0.0 (can't grade), never a crash.
    assert _bm.score("reasoning", {"answer": "no gold number here"}, "Answer: 5") == 0.0


@pytest.mark.skipif(not _bm._SANDBOX_EXEC, reason="sandbox-exec is macOS-only")
def test_coding_sandbox_blocks_network():
    # NETWORK egress is the observable — NOT a sentinel file. The profile denies file-write*
    # unconditionally, so a "write a file iff the socket connects" proof passes even with the
    # network rule removed (it never reaches the write). Instead, the body's ONLY statement is a
    # connect to a real local listener: under the sandbox the connect is denied → the program
    # raises → exits nonzero → score 0.0. A local listener guarantees the target is reachable, so
    # a green result means the sandbox blocked egress, not that the host was offline. If
    # `(deny network*)` were dropped from _SB_PROFILE_TMPL, the child would connect and the program
    # would exit 0 → score 1.0, failing this test (verified by flipping the rule).
    import socket as _socket
    srv = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        item = {"prompt": "def f():\n", "entry_point": "f", "test": "def check(c):\n    c()\n"}
        body = (
            "    import socket\n"
            f"    socket.create_connection(('127.0.0.1', {port}), timeout=3).close()\n"
        )
        assert _bm.score("coding", item, body) == 0.0, "sandbox FAILED to block network egress"
    finally:
        srv.close()


def test_coding_killpg_falls_back_to_proc_kill(monkeypatch):
    # When the timed-out child can't be reaped by killpg (already gone / EPERM), fall back to
    # proc.kill() so a stuck child is never leaked. Force the sandbox branch + mock the subprocess
    # so this is host-independent (covers the reap fallback on any OS, instantly — no real 10s hang).
    import subprocess as _sp

    class _FakeProc:
        def __init__(self):
            self.pid = 999999
            self.killed = False

        def wait(self, timeout=None):
            if timeout is not None:
                raise _sp.TimeoutExpired(cmd="x", timeout=timeout)   # the 10s wait "times out"

        def poll(self):
            return None          # still running at the finally → reap path runs

        def kill(self):
            self.killed = True

    fake = _FakeProc()
    monkeypatch.setattr(_bm, "_SANDBOX_EXEC", "/usr/bin/sandbox-exec")   # force the sandbox cmd branch
    monkeypatch.setattr(_bm.subprocess, "Popen", lambda *a, **k: fake)
    # raising=False: os.killpg/os.getpgid don't exist on Windows — there the reap hits AttributeError
    # and falls back to proc.kill() (the production fix); on POSIX the mocked killpg raises instead.
    monkeypatch.setattr(_bm.os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(_bm.os, "killpg",
                        lambda pgid, sig: (_ for _ in ()).throw(ProcessLookupError("gone")),
                        raising=False)
    assert _bm.score("coding", _TINY_CODING_ITEM, "    return a + b\n") == 0.0   # timed out → 0
    assert fake.killed is True          # fell back to proc.kill() when killpg couldn't reap
