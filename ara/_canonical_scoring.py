# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Canonical, reference-implementation scoring math for the measured benchmark tier.

These are vendored copies of the *grading* logic from the standard open-source eval tools, so
ARA grades exactly the way the leaderboards do while keeping its own governed generation loop.
Vendoring (not depending on the full harnesses) is the established practice — bigcode-eval is
itself a vendored copy of OpenAI's ``check_correctness``. All upstreams are permissive:

  * SQuAD F1/EM      — rajpurkar/SQuAD-explorer ``evaluate-v2.0.py`` (MIT) /
                       huggingface/evaluate squad ``compute_score.py`` (Apache-2.0)
  * GSM8K extraction — EleutherAI/lm-evaluation-harness ``gsm8k`` task (MIT) /
                       openai/grade-school-math ``dataset.py`` (MIT)
  * HumanEval        — openai/human-eval ``execution.py`` (MIT); robust chat-completion code
                       extraction after evalplus/evalplus ``sanitize.py`` (MIT)

No engine imports — this is core-side scoring only.
"""
from __future__ import annotations

import ast
import re
import string
from collections import Counter

# ── SQuAD v1.1 token-F1 / exact-match ────────────────────────────────────────
# Verbatim canonical order: lower → strip-punct → remove-articles → fix-whitespace.


def squad_normalize(s: str) -> str:
    """Canonical SQuAD ``normalize_answer``: lowercase, drop punctuation, articles, extra space."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def squad_f1(prediction: str, ground_truth: str) -> float:
    """Canonical SQuAD token-level F1 (Counter overlap; 0.0 on no shared tokens)."""
    pred_tokens = squad_normalize(prediction).split()
    gt_tokens = squad_normalize(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        # v2.0 guard: both empty → 1.0 (identical), only one empty → 0.0.
        return float(pred_tokens == gt_tokens)
    common = sum((Counter(pred_tokens) & Counter(gt_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def squad_em(prediction: str, ground_truth: str) -> bool:
    """Canonical SQuAD exact-match over normalized strings."""
    return squad_normalize(prediction) == squad_normalize(ground_truth)


def max_over_golds(metric_fn, prediction: str, golds: list[str]) -> float:
    """Max of ``metric_fn(prediction, gold)`` over acceptable gold answers."""
    return max((float(metric_fn(prediction, g)) for g in golds), default=0.0)


# ── GSM8K answer extraction ──────────────────────────────────────────────────
# strict: an explicit ``Answer:`` / ``####`` marker; flexible: the LAST number anywhere
# (lm-eval ``flexible-extract`` — the metric leaderboards report).

_GSM8K_STRICT = re.compile(r"(?:answer:|####)\s*(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE)
_GSM8K_FLEXIBLE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def gsm8k_extract_number(text: str) -> float | None:
    """Extract the predicted number from GSM8K output (strict marker, else last number)."""
    strict = _GSM8K_STRICT.findall(text)
    if strict:
        return _to_number(strict[-1])
    flex = _GSM8K_FLEXIBLE.findall(text)
    if flex:
        last = flex[-1]
        return _to_number(last[0] or last[1])
    return None


def _to_number(s: str) -> float | None:
    """Parse a GSM8K-style numeric token: strip ``,`` thousands, ``$``, trailing ``.``."""
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except (ValueError, OverflowError):
        return None


# ── HumanEval code extraction + program assembly ─────────────────────────────


def _strip_fences(text: str) -> str:
    """Return fenced code block(s) joined, or the raw text if there are no fences."""
    blocks = re.findall(r"```[A-Za-z0-9_+\-]*\n?(.*?)```", text, re.DOTALL)
    return "\n".join(blocks) if blocks else text


def _largest_parseable(text: str) -> str:
    """Longest contiguous run of lines that ``ast.parse``-s (drops prose around code)."""
    if not text.strip():
        return ""
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        pass
    lines = text.split("\n")
    n = len(lines)
    best = ""
    for i in range(n):
        for j in range(n, i, -1):
            chunk = "\n".join(lines[i:j])
            try:
                ast.parse(chunk)
            except SyntaxError:
                continue
            if len(chunk) > len(best):
                best = chunk
            break  # longest j for this start found; shorter j can't beat it
    return best


def _prompt_imports(prompt: str) -> str:
    """Import lines from the HumanEval prompt (typing deps a bare completion may omit)."""
    lines = [ln for ln in prompt.split("\n") if re.match(r"\s*(import|from)\s", ln)]
    return ("\n".join(lines) + "\n") if lines else ""


def humaneval_program(prompt: str, completion: str, test: str, entry_point: str) -> str:
    """Assemble the canonical HumanEval check program, robust to chat-style completions.

    Two completion shapes are handled:
      * full function (``def {entry_point}`` present, often fenced and/or wrapped in prose) —
        sanitize to the parseable code and restore the prompt's import lines;
      * bare body continuation (the canonical HumanEval completion format) — graft onto the
        prompt signature (a bare indented body does not parse standalone).

    Canonical tail: ``\\n{test}\\ncheck({entry_point})``. Pass iff the program raises nothing.
    """
    check = f"\n{test}\ncheck({entry_point})\n"
    code = _strip_fences(completion)
    if re.search(rf"(?m)^\s*def\s+{re.escape(entry_point)}\b", code):
        sanitized = _largest_parseable(code) or code
        return _prompt_imports(prompt) + sanitized + check
    return prompt + code + check
