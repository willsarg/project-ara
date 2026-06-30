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
dict rather than raised — read endpoints surface it as the body, and the JobRunner records it as the
job's result (it already turns a raised exception into a failed job, so either way nothing is lost).
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from importlib import metadata


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


def _characterize(args: dict) -> dict:
    """``ara characterize <model> [--engine E]`` — measure + store a model's safe ceiling."""
    cli = ["characterize", args["model"]]
    if args.get("engine"):
        cli += ["--engine", args["engine"]]
    return _run_cli(cli)


def _run(args: dict) -> dict:
    """``ara run <model> <prompt> [--engine E] --yes`` — one governed generation."""
    cli = ["run", args["model"]]
    if args.get("engine"):
        cli += ["--engine", args["engine"]]
    cli.append("--yes")                          # non-interactive: no TTY to confirm at
    if args.get("prompt"):
        cli.append(args["prompt"])               # the prompt is positional and trails the flags
    return _run_cli(cli)


def _serve(args: dict) -> dict:
    """``ara serve <model> [--engine E] [--ctx N] [--name X] --yes`` — stand a model up + endpoint."""
    cli = ["serve", args["model"]]
    if args.get("engine"):
        cli += ["--engine", args["engine"]]
    if args.get("ctx") is not None:
        cli += ["--ctx", str(args["ctx"])]
    if args.get("name"):
        cli += ["--name", args["name"]]
    cli.append("--yes")
    return _run_cli(cli)


def _benchmark(args: dict) -> dict:
    """``ara benchmark <model> --use-case X [--engine E] [--ctx N] [--max-tokens N] --yes``.

    The coding benchmark's code-execution gate (``--exec-consent``) is NOT auto-supplied — that's a
    deliberate per-job opt-in, so it's only added when the caller sets ``exec_consent`` in args."""
    cli = ["benchmark", args["model"], "--use-case", args["use_case"]]
    if args.get("engine"):
        cli += ["--engine", args["engine"]]
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


def _ara_version() -> str:
    """The installed project-ara version, or ``"?"`` from an un-installed source tree."""
    try:
        return metadata.version("project-ara")
    except metadata.PackageNotFoundError:
        return "?"


def build_app(version: str | None = None):
    """Assemble the production node app: a JobRunner over the real workers + the real providers.

    ``app``/``jobs`` are imported lazily so merely importing this module doesn't hard-require FastAPI
    (an optional ``[node]`` extra) — same on-demand philosophy as the engines."""
    from ara.node.app import create_app
    from ara.node.jobs import JobRunner
    runner = JobRunner(default_workers())
    return create_app(runner, default_providers(), version=version or _ara_version())
