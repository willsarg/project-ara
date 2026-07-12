# Contributing to Project ARA

Thanks for your interest! ARA started as a personal tool for honestly assessing a machine
for AI work and running local models safely — but the goal is for it to work across **lots
of hardware and lots of AI software**, so contributions are genuinely wanted, especially:

- recon coverage for tools/interpreters/model stores/apps ARA doesn't know about yet,
- new **backends** (NVIDIA/CUDA is the next frontier),
- and reports from machines unlike the M-series reference box.

Please read [AGENTS.md](./AGENTS.md) first — it's the source of truth for the project's
purpose, the architecture boundary, and the hard rules. This file covers the *human
workflow*.

## Setup

```bash
uv sync                       # install into .venv (incl. dev tools)
uv run ara                    # landing screen
uv run pytest                 # tests (100% statement + branch coverage is the bar)
```

ARA needs **Python 3.12+** and [`uv`](https://github.com/astral-sh/uv). The core stays lean on every
platform. `ara install --engine mlx` or `ara install --engine cuda` installs the selected native
engine package and its heavy dependencies into an isolated environment on demand; recon needs
neither engine.

## The rules that override everything

From [AGENTS.md](./AGENTS.md) — a change must not violate these:

- **Recon is read-only.** `detect` / `status` / `python` / `apps` / `mlx` observe; they never
  stress the machine, load a model, or mutate state.
- **`characterize` is consent-gated.** It measures (and may download model weights) only with
  explicit opt-in; `profile` remains engine-free and read-only.
- **Advisory, never destructive.** ARA surfaces facts; it never runs or prescribes
  state-mutating commands for the user.
- **The core stays engine-free.** No hardware-specific import outside an isolated engine
  subprocess. Native engine sources live under `ara/_engine_packages/{mlx,cuda}` but are never
  imported by the core process.

## Conventions

- **`uv` only** — no `pip install --break-system-packages`. The HF CLI is `hf`.
- **Report the user's environment, not ARA's** — strip ARA's venv when probing tools.
- **Tests land with code** — `fail_under = 100` (statement + branch). Cover engine seams with
  subprocess fakes; the normal suite does not install or import MLX, torch, or CUDA.
- **Match the surrounding style** — semantic console roles (`accent`/`dim`/`good`/`warn`),
  honest copy, and the `--json` / `--include` / `--exclude` flags where a recon command adds
  output.

## Landing a change

1. Branch off `main`.
2. Keep the PR focused on one change; fill in the PR template.
3. `uv run pytest` green (100% coverage), and paste relevant command output (e.g. the
   `ara <command>` before/after) so reviewers can see the behavior.
4. Note any new tool/app/store you added to a curated catalog and how you verified it.

## License

Project ARA is licensed under the **Apache License 2.0** (`LICENSE` / `NOTICE`). By submitting a
contribution you agree it is provided under Apache 2.0 — the "inbound = outbound" rule of Apache §5
(no separate CLA or DCO). Contribute only code you have the right to license this way: code you
wrote, or code under an Apache-2.0-**compatible** permissive license (MIT/BSD/ISC/Apache-2.0) with
its notice preserved in `NOTICE`. Never paste in GPL/LGPL/AGPL, proprietary, or unknown-provenance
code. Start every new source file with `# SPDX-License-Identifier: Apache-2.0` and
`# Copyright 2026 Will Sarg`.

**AI coding agents** are welcome under the same rules; the human who opens the PR is responsible
for the assurances above. See "License & AI agents" in [AGENTS.md](./AGENTS.md).

## Reporting issues

Use the issue templates (bug / feature). For anything security-related, see
[SECURITY.md](./SECURITY.md) — please don't open a public issue.
