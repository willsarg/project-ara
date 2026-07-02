# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Wires the node app to ARA's real verbs by shelling out to the ARA CLI with ``--json``.

The node deliberately does NOT import ``ara.cli`` and call its render functions in-process:
re-running each verb as ``python -m ara.cli <verb> --json`` reuses the CLI exactly as a human
would, so every safety gate (consent prompts, the benchmark exec-gate, the context governor) and
the canonical ``--json`` shape come along for free — zero re-implementation, zero coupling. The
node only translates a job's ``args`` dict into the right flags.

:func:`_run_cli` is the single external boundary (the one subprocess call); tests monkeypatch it to
drive providers/workers without spawning anything. A failed verb is returned as an ``{"error": ...}``
dict rather than raised — the agent loop reports it back as the job's (failed) result, so either way
nothing is lost.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable


def _run_cli(args: list[str]) -> dict:
    """Run ``python -m ara.cli <args> --json`` and return the parsed JSON object.

    The one mockable seam to the outside world. On a non-zero exit or output that isn't valid JSON,
    return a structured ``{"error", "stderr"}`` dict instead of raising — a failed verb is data the
    caller can show, not a crash. ``--json`` is appended here so every call-site stays terse and
    can't forget it (the whole contract depends on machine-readable output)."""
    proc = subprocess.run([sys.executable, "-m", "ara.cli", *args, "--json"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": f"`ara {args[0]}` exited {proc.returncode}",
                "stderr": (proc.stderr or "").strip()}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"error": f"`ara {args[0]}` produced unparseable output",
                "stderr": (proc.stderr or "").strip()}


def default_providers() -> dict[str, Callable[[], dict]]:
    """The read-endpoint providers: each key maps to a zero-arg call of the same-named CLI verb.

    The ``v=verb`` default binds the loop variable per-lambda (else every closure would capture the
    last verb)."""
    return {verb: (lambda v=verb: _run_cli([v]))
            for verb in ("status", "detect", "profile", "models")}


def _safe(value: str, field: str) -> str:
    """Guard a value that becomes a CLI argv token: reject a leading ``-`` so it can't be smuggled in
    as a flag (argv flag-injection — e.g. a model named ``--exec-consent`` flipping a safety gate).

    ARA's CLI uses a hand-rolled parser with NO ``--`` end-of-options sentinel, so a ``--`` separator
    wouldn't work (it'd become a stray positional); validation is the right defense. A job's args come
    from the token-authorized coordinator — which could already set these flags explicitly — so this is
    defense-in-depth + correctness (a model id / engine / use-case starting with ``-`` is invalid
    anyway). Raising → the agent loop reports it back as a failed job result."""
    if not isinstance(value, str) or value.startswith("-"):
        raise ValueError(f"invalid {field}: must be a string not starting with '-'")
    return value


def _characterize(args: dict) -> dict:
    """``ara characterize <model> [--engine E]`` — measure + store a model's safe ceiling."""
    cli = ["characterize", _safe(args["model"], "model")]
    if args.get("engine"):
        cli += ["--engine", _safe(args["engine"], "engine")]
    return _run_cli(cli)


def _run(args: dict) -> dict:
    """``ara run <model> <prompt> [--engine E] --yes`` — one governed generation."""
    cli = ["run", _safe(args["model"], "model")]
    if args.get("engine"):
        cli += ["--engine", _safe(args["engine"], "engine")]
    cli.append("--yes")                          # non-interactive: no TTY to confirm at
    if args.get("prompt"):
        cli.append(str(args["prompt"]))          # the prompt is one trailing free-text positional
    return _run_cli(cli)


def _serve(args: dict) -> dict:
    """``ara serve <model> [--engine E] [--ctx N] [--name X] --yes`` — stand a model up + endpoint."""
    cli = ["serve", _safe(args["model"], "model")]
    if args.get("engine"):
        cli += ["--engine", _safe(args["engine"], "engine")]
    if args.get("ctx") is not None:
        cli += ["--ctx", str(args["ctx"])]
    if args.get("name"):
        cli += ["--name", _safe(args["name"], "name")]
    cli.append("--yes")
    return _run_cli(cli)


def _benchmark(args: dict) -> dict:
    """``ara benchmark <model> --use-case X [--engine E] [--ctx N] [--max-tokens N] --yes``.

    The coding benchmark's code-execution gate (``--exec-consent``) is NOT auto-supplied — that's a
    deliberate per-job opt-in, so it's only added when the caller sets ``exec_consent`` in args."""
    cli = ["benchmark", _safe(args["model"], "model"), "--use-case", _safe(args["use_case"], "use_case")]
    if args.get("engine"):
        cli += ["--engine", _safe(args["engine"], "engine")]
    if args.get("ctx") is not None:
        cli += ["--ctx", str(args["ctx"])]
    if args.get("max_tokens") is not None:
        cli += ["--max-tokens", str(args["max_tokens"])]
    if args.get("exec_consent"):
        cli.append("--exec-consent")
    cli.append("--yes")
    return _run_cli(cli)


def default_workers() -> dict[str, Callable[[dict], dict]]:
    """The job workers: one per action verb, each translating a job ``args`` dict into CLI flags."""
    return {"characterize": _characterize, "run": _run, "serve": _serve, "benchmark": _benchmark}
