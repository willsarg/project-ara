# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Wires the node app to ARA's real verbs by shelling out to the ARA CLI with ``--json``.

The node deliberately does NOT import ``ara.cli`` and call its render functions in-process:
re-running each verb as ``python -m ara <verb> --json`` reuses the canonical CLI exactly as a human
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
from typing import cast


def _run_cli_payload(args: list[str], expected_type: type[dict] | type[list],
                     shape_error: str) -> dict | list:
    """Run the canonical JSON CLI and enforce one explicit successful payload shape.

    On a non-zero exit or output that isn't valid JSON,
    return a structured ``{"error", "stderr"}`` dict instead of raising — a failed verb is data the
    caller can show, not a crash. ``--json`` is inserted before any ``--`` separator so every
    call-site stays terse without weakening positional-argument safety."""
    json_at = args.index("--") if "--" in args else len(args)
    command_args = [*args[:json_at], "--json", *args[json_at:]]
    proc = subprocess.run([sys.executable, "-m", "ara", *command_args],
                          capture_output=True, text=True)
    try:
        payload = json.loads(proc.stdout or "")
    except json.JSONDecodeError:
        if proc.returncode != 0:
            return {"error": f"`ara {args[0]}` exited {proc.returncode}",
                    "stderr": (proc.stderr or "").strip()}
        return {"error": f"`ara {args[0]}` produced unparseable output",
                "stderr": (proc.stderr or "").strip()}
    if proc.returncode != 0:
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, str) and error.strip():
            stderr = (proc.stderr or "").strip()
            if stderr and "stderr" not in payload:
                payload["stderr"] = stderr
            return payload
        return {"error": f"`ara {args[0]}` exited {proc.returncode}",
                "stderr": (proc.stderr or "").strip()}
    if not isinstance(payload, expected_type):
        return {"error": f"`ara {args[0]}` produced {shape_error}",
                "stderr": (proc.stderr or "").strip()}
    return payload


def _run_cli(args: list[str]) -> dict:
    """Run ``python -m ara <args> --json`` and return the parsed JSON object.

    Generic read/action providers require an object on success. A nonzero operational error object
    is preserved; every other nonzero or malformed shape becomes a stable generic failure.
    ``--json`` is inserted before any ``--`` separator so positional data stays protected.
    """
    return cast(dict, _run_cli_payload(args, dict, "non-object JSON output"))


def _run_models_inventory() -> dict:
    """Return canonical ``detect --models`` inventory under a stable node object envelope."""
    payload = _run_cli_payload(
        ["detect", "--models"], list, "non-array model inventory")
    if isinstance(payload, dict):
        return payload
    return {"models": payload}


def default_providers() -> dict[str, Callable[[], dict]]:
    """Map read-endpoint keys to their canonical CLI argv.

    Most providers use a same-named verb; physical model inventory is the ``detect --models``
    facet. The ``a=args`` default binds each argv list per lambda (else every closure would capture
    the last mapping)."""
    commands = {
        "status": ["status"],
        "detect": ["detect"],
        "profile": ["profile"],
    }
    providers = {key: (lambda a=args: _run_cli(a)) for key, args in commands.items()}
    providers["models"] = _run_models_inventory
    return providers


def _safe(value: str, field: str) -> str:
    """Guard a value that becomes a CLI argv token: reject a leading ``-`` so it can't be smuggled in
    as a flag (argv flag-injection — e.g. a model named ``--exec-consent`` flipping a safety gate).

    The Click CLI also receives a ``--`` end-of-options sentinel before free positional data, so a
    prompt beginning with ``-`` stays prompt text. Validation remains defense-in-depth for fields that
    must never be options (model id, engine, use case, and served name). Raising lets the agent loop
    report the bad job as a failed result."""
    if not isinstance(value, str) or value.startswith("-"):
        raise ValueError(f"invalid {field}: must be a string not starting with '-'")
    return value


def _characterize(args: dict) -> dict:
    """``ara characterize <model> [--engine E]`` — measure + store a model's safe ceiling."""
    cli = ["characterize"]
    if args.get("engine"):
        cli += ["--engine", _safe(args["engine"], "engine")]
    cli += ["--", _safe(args["model"], "model")]
    return _run_cli(cli)


def _run(args: dict) -> dict:
    """``ara run <model> <prompt> [--engine E] --yes`` — one governed generation."""
    cli = ["run"]
    if args.get("engine"):
        cli += ["--engine", _safe(args["engine"], "engine")]
    cli.append("--yes")                          # non-interactive: no TTY to confirm at
    cli += ["--", _safe(args["model"], "model")]
    if args.get("prompt"):
        cli.append(str(args["prompt"]))          # the prompt is one trailing free-text positional
    return _run_cli(cli)


def _serve(args: dict) -> dict:
    """``ara serve <model> [--engine E] [--ctx N] [--name X] --yes`` — stand a model up + endpoint."""
    cli = ["serve"]
    if args.get("engine"):
        cli += ["--engine", _safe(args["engine"], "engine")]
    if args.get("ctx") is not None:
        cli += ["--ctx", str(args["ctx"])]
    if args.get("name"):
        cli += ["--name", _safe(args["name"], "name")]
    cli.append("--yes")
    cli += ["--", _safe(args["model"], "model")]
    return _run_cli(cli)


def _benchmark(args: dict) -> dict:
    """``ara benchmark <model> --use-case X [--engine E] [--ctx N] [--max-tokens N] --yes``.

    The coding benchmark's code-execution gate (``--exec-consent``) is NOT auto-supplied — that's a
    deliberate per-job opt-in, so it's only added when the caller sets ``exec_consent`` in args."""
    cli = ["benchmark", "--use-case", _safe(args["use_case"], "use_case")]
    if args.get("engine"):
        cli += ["--engine", _safe(args["engine"], "engine")]
    if args.get("ctx") is not None:
        cli += ["--ctx", str(args["ctx"])]
    if args.get("max_tokens") is not None:
        cli += ["--max-tokens", str(args["max_tokens"])]
    if args.get("exec_consent"):
        cli.append("--exec-consent")
    cli.append("--yes")
    cli += ["--", _safe(args["model"], "model")]
    return _run_cli(cli)


def default_workers() -> dict[str, Callable[[dict], dict]]:
    """The job workers: one per action verb, each translating a job ``args`` dict into CLI flags."""
    return {"characterize": _characterize, "run": _run, "serve": _serve, "benchmark": _benchmark}
