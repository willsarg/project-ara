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
    assert len(load_probe("coding")) == 25
    assert len(load_probe("reasoning")) == 30
    assert len(load_probe("agentic")) == 20
    assert len(load_probe("extraction")) == 25
    assert len(load_probe("rag")) == 20


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
    from ara.benchmark import score
    item = {
        "id": "t1", "question": "q", "function": {},
        "expected": {"name": "calc", "arguments": {"x": 3.14159}},
    }
    completion = '{"name": "calc", "arguments": {"x": 3.14159}}'
    assert score("agentic", item, completion) == 1.0


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


def test_coding_timeout_scores_0():
    from ara.benchmark import score
    item = {
        "task_id": "test/hang",
        "prompt": "def hang():\n",
        "entry_point": "hang",
        "test": "hang()\n",
    }
    assert score("coding", item, "    while True: pass\n") == 0.0


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
    ("humaneval_25.json", 25),
    ("gsm8k_30.json", 30),
    ("bfcl_simple_20.json", 20),
    ("extraction_25.json", 25),
    ("rag_20.json", 20),
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


def test_score_probe_set_empty_returns_zero_not_crash():
    assert _bm.score_probe_set("coding", [], []) == 0.0


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
