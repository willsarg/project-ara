# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Judge-free benchmark probe data and scoring for ARA's five use-cases.

SAFETY NOTE — coding scorer: ``score("coding", ...)`` executes model-generated
Python inside a subprocess with a 10-second timeout.  Callers MUST gate execution
on explicit user consent (the ``characterize`` command's consent surface is the
canonical example).  Never invoke it in a non-interactive pipeline without consent.

2026-06-28-benchmark-layer
"""
from __future__ import annotations

import json
import re
import string
import subprocess
import sys
import tempfile
from pathlib import Path

USE_CASES: tuple[str, ...] = ("coding", "reasoning", "agentic", "extraction", "rag")

_DATA_DIR = Path(__file__).parent / "data" / "benchmarks"

_PROBE_FILES: dict[str, str] = {
    "coding":     "humaneval_25.json",
    "reasoning":  "gsm8k_30.json",
    "agentic":    "bfcl_simple_20.json",
    "extraction": "extraction_25.json",
    "rag":        "rag_20.json",
}


def load_probe(use_case: str) -> list[dict]:
    """Return the shipped probe list for *use_case*."""
    path = _DATA_DIR / _PROBE_FILES[use_case]
    return json.loads(path.read_text())


def prompt_for(use_case: str, item: dict) -> str:
    """Format the model prompt for a single probe item."""
    if use_case == "coding":
        return (
            "Complete the following Python function. "
            "Output ONLY the function body, indented, with no extra text.\n\n"
            + item["prompt"]
        )
    if use_case == "reasoning":
        return (
            item["question"]
            + "\n\nThink step by step, then write your final answer as:\nAnswer: <number>"
        )
    if use_case == "agentic":
        fn_json = json.dumps(item["function"], indent=2)
        return (
            f"Given this function definition:\n{fn_json}\n\n"
            f"User request: {item['question']}\n\n"
            'Respond with ONLY a JSON object {"name": "...", "arguments": {...}}.'
        )
    if use_case == "extraction":
        return (
            f"Passage: {item['context']}\n\n"
            f"Question: {item['question']}\n\n"
            "Answer using only words from the passage."
        )
    # rag
    return (
        f"Passage: {item['context']}\n\n"
        f"Question: {item['question']}\n\n"
        "Use ONLY the information in the passage to answer."
    )


def score(use_case: str, item: dict, completion: str) -> float:
    """Judge-free score for a single (item, completion) pair.  Returns 0..1."""
    if use_case == "coding":
        return _score_coding(item, completion)
    if use_case == "reasoning":
        return _score_reasoning(item, completion)
    if use_case == "agentic":
        return _score_agentic(item, completion)
    if use_case == "extraction":
        return _score_extraction(item, completion)
    # rag
    return _score_rag(item, completion)


def score_probe_set(
    use_case: str,
    items: list[dict],
    completions: list[str],
) -> float:
    """Mean per-item score over a matched items/completions list."""
    if len(items) != len(completions):
        raise ValueError(
            f"items ({len(items)}) and completions ({len(completions)}) must have equal length"
        )
    scores = [score(use_case, it, c) for it, c in zip(items, completions)]
    return sum(scores) / len(scores)


# ── private scorers ────────────────────────────────────────────────────────


def _score_coding(item: dict, completion: str) -> float:
    """Execute prompt + completion + tests in a sandboxed subprocess.

    The 10-second timeout and ``capture_output=True`` keep side effects contained.
    """
    code = item["prompt"] + completion + "\n" + item["test"]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            timeout=10,
            capture_output=True,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _score_reasoning(item: dict, completion: str) -> float:
    """Exact numeric match (±1e-6) against the GSM8K '#### N' answer."""
    gt_match = re.search(r"####\s*([\d,.\-]+)", item["answer"])
    pred_match = re.search(r"Answer:\s*([\d,.\-]+)", completion, re.IGNORECASE)
    if not gt_match or not pred_match:
        return 0.0
    gt = float(gt_match.group(1).replace(",", ""))
    pred = float(pred_match.group(1).replace(",", ""))
    return 1.0 if abs(pred - gt) < 1e-6 else 0.0


def _strip_fences(text: str) -> str:
    """Remove markdown code fences (```...```) from text."""
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _score_agentic(item: dict, completion: str) -> float:
    """Name + arg exact match (str case-insensitive, numeric ±1e-4)."""
    try:
        parsed = json.loads(_strip_fences(completion))
    except (json.JSONDecodeError, ValueError):
        return 0.0
    expected = item["expected"]
    if parsed.get("name") != expected["name"]:
        return 0.0
    pred_args = parsed.get("arguments", {})
    for arg, exp_val in expected["arguments"].items():
        pred_val = pred_args.get(arg)
        if pred_val is None:
            return 0.0
        if isinstance(exp_val, (int, float)):
            try:
                if abs(float(pred_val) - float(exp_val)) > 1e-4:
                    return 0.0
            except (TypeError, ValueError):
                return 0.0
        else:
            if str(pred_val).lower() != str(exp_val).lower():
                return 0.0
    return 1.0


def _normalize_text(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip articles and punctuation."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground_truth."""
    pred_tokens = _normalize_text(prediction).split()
    gt_tokens = _normalize_text(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _score_extraction(item: dict, completion: str) -> float:
    """SQuAD normalize → exact-match → 1.0, else max token-F1 over answers."""
    norm_pred = _normalize_text(completion)
    best = 0.0
    for ans in item["answers"]:
        norm_gt = _normalize_text(ans["text"])
        if norm_pred == norm_gt:
            return 1.0
        f1 = _token_f1(completion, ans["text"])
        if f1 > best:
            best = f1
    return best


def _score_rag(item: dict, completion: str) -> float:
    """Normalized exact-match over accepted answers → 1.0 or 0.0."""
    norm_pred = _normalize_text(completion)
    for ans in item["answers"]:
        if norm_pred == _normalize_text(ans["text"]):
            return 1.0
    return 0.0
