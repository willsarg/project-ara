# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Judge-free benchmark probe data and scoring for ARA's five use-cases.

SAFETY NOTE — coding scorer: ``score("coding", ...)`` executes model-generated Python. On macOS it
runs under a Seatbelt sandbox (``sandbox-exec``): **no network, no filesystem writes**, exec confined
to the Python framework — plus a 10s timeout + process-group kill. Off macOS (no ``sandbox-exec``) it
falls back to process-isolation only and emits a loud ``RuntimeWarning``. Either way, callers MUST
gate execution on explicit consent (cli's ``--exec-consent``); never run it unattended. The residual
risk under the sandbox is read-only (no write/network → nothing read can be exfiltrated). Linux
containment (bubblewrap) is a tracked follow-up.

2026-06-28-benchmark-layer
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

from ara import _canonical_scoring as _canon

USE_CASES: tuple[str, ...] = ("coding", "reasoning", "agentic", "extraction", "rag")

_DATA_DIR = Path(__file__).parent / "data" / "benchmarks"

# macOS Seatbelt sandbox for the coding scorer's untrusted-code execution (deprecated-but-present
# on current macOS; needs no root/entitlement). Verified: denies network + all filesystem writes +
# any exec outside the Python framework, while a normal stdlib HumanEval script still runs. file-read
# stays broad (dyld needs version-specific paths) — safe because no-write + no-network block
# exfiltration. None on Linux/Windows → caller falls back to process-isolation + a loud warning.
_SANDBOX_EXEC: str | None = shutil.which("sandbox-exec")

_SB_PROFILE_TMPL = """\
(version 1)
(deny default)
(deny network*)
(deny file-write*)
(allow process-exec (subpath "{python_base}"))
(allow process-fork)
(allow process-info*)
(allow file-read*)
"""

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
    return sum(scores) / len(scores) if scores else 0.0


# ── private scorers ────────────────────────────────────────────────────────


def _score_coding(item: dict, completion: str) -> float:
    """Run the HumanEval unit tests against prompt + completion. 1.0 iff every assert passes.

    Program assembly is the canonical HumanEval recipe with robust chat-completion code extraction
    (:func:`ara._canonical_scoring.humaneval_program`) — it handles both a bare body continuation
    and a full ``def`` wrapped in fences/prose, so a more verbose model is not falsely failed.

    Executes model-generated code under a macOS Seatbelt sandbox (``sandbox-exec``) when present —
    no network, no filesystem writes, exec confined to the Python framework — plus a 10s timeout and
    a process-group kill. Off macOS (no ``sandbox-exec``) it falls back to process-isolation only and
    emits a ``RuntimeWarning`` (never a silent downgrade). The assembled script ends with
    ``check(<entry_point>)`` — the asserts don't run without that call (the bug the audit caught).
    """
    code = _canon.humaneval_program(
        item["prompt"], completion, item["test"], item["entry_point"]
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    python_bin = str(Path(sys.executable).resolve())
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    sb_path: str | None = None
    if _SANDBOX_EXEC:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sb", delete=False,
                                         encoding="utf-8") as pf:
            pf.write(_SB_PROFILE_TMPL.format(python_base=sys.base_prefix))
            sb_path = pf.name
        cmd = [_SANDBOX_EXEC, "-f", sb_path, python_bin, tmp_path]
    else:
        warnings.warn(
            "sandbox-exec not found — the coding benchmark is running WITHOUT an OS sandbox; "
            "model-generated code has full filesystem + network access.",
            RuntimeWarning, stacklevel=2,
        )
        cmd = [python_bin, tmp_path]

    proc = None
    try:
        # start_new_session: the child leads its own process group, so killpg reaps any
        # processes the model's code spawned in that group (a plain proc.kill would not).
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, env=env,
        )
        try:
            return 1.0 if proc.wait(timeout=10) == 0 else 0.0
        except subprocess.TimeoutExpired:
            return 0.0
    finally:
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait()
        Path(tmp_path).unlink(missing_ok=True)
        if sb_path:
            Path(sb_path).unlink(missing_ok=True)


def _score_reasoning(item: dict, completion: str) -> float:
    """Exact numeric match (±1e-6) against the GSM8K '#### N' answer.

    Prediction is extracted the canonical lm-eval way (an explicit ``Answer:``/``####`` marker if
    present, else the LAST number anywhere — the ``flexible-extract`` metric leaderboards report),
    so a model that reasons to a final number without the marker is still scored. Never crashes
    on a malformed number.
    """
    gt_match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", item["answer"])
    if not gt_match:
        return 0.0
    gt = _canon._to_number(gt_match.group(1))
    pred = _canon.gsm8k_extract_number(completion)
    if gt is None or pred is None:
        return 0.0
    return 1.0 if abs(pred - gt) < 1e-6 else 0.0


def _extract_json(text: str):
    """Parse the first JSON object found anywhere in *text* (tolerates prose + fences).

    Unlike code, JSON is whitespace-insensitive, so a greedy ``{...}`` scan is safe.
    Returns the parsed object, or ``None`` if nothing parses.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except (json.JSONDecodeError, ValueError):
        return None


def _score_agentic(item: dict, completion: str) -> float:
    """Name + arg exact match (name + str args case-insensitive, numeric ±1e-4)."""
    parsed = _extract_json(completion)
    if not isinstance(parsed, dict):
        return 0.0
    expected = item["expected"]
    if str(parsed.get("name", "")).lower() != str(expected["name"]).lower():
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


# Canonical SQuAD normalize / token-F1 (vendored in ``_canonical_scoring``); these aliases keep
# the historical private names other call sites and tests reference.
_normalize_text = _canon.squad_normalize
_token_f1 = _canon.squad_f1


def _score_extraction(item: dict, completion: str) -> float:
    """Canonical SQuAD: exact-match → 1.0, else max token-F1 over accepted answers."""
    golds = [ans["text"] for ans in item["answers"]]
    if _canon.max_over_golds(_canon.squad_em, completion, golds):
        return 1.0
    return _canon.max_over_golds(_canon.squad_f1, completion, golds)


def _score_rag(item: dict, completion: str) -> float:
    """Canonical SQuAD token-F1 over accepted answers (EM short-circuits to 1.0).

    RAG answers are full phrases, so this is F1-scored exactly like extraction — the prior
    exact-match-only scorer returned a false flat-zero for every verbose answer.
    """
    golds = [ans["text"] for ans in item["answers"]]
    if _canon.max_over_golds(_canon.squad_em, completion, golds):
        return 1.0
    return _canon.max_over_golds(_canon.squad_f1, completion, golds)
