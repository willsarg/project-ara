# AGENTS.md — purpose, boundaries, and conventions for ARA

This is the source of truth for *what ARA is and how to work on it*. Read it before
contributing (human or agent). [CONTRIBUTING.md](./CONTRIBUTING.md) covers the human
workflow (setup, landing a change); this covers the **why** and the **rules**.

## What ARA is

**Project ARA — "AI Runs Anywhere."** A tool you reach for to **honestly assess any machine
with a Python runtime for AI workloads**, then run local models safely on whatever hardware
is present. Apple Silicon (MLX), NVIDIA (CUDA), and any CPU all run models today; recon works everywhere.

### The three rules (the invariant core)

ARA's mission — *"AI Runs Anywhere: safely, reliably, and accurately — train, run, and govern AI
workloads on any infrastructure"* — **is** three numbered rules. Every change, in every part of the
system, answers to all three. Canonical statement: the private vault's `ARA - Product` note.

1. **Safety** — *don't crash the system.* Never exceed the memory wall; run right up to the safe
   edge and no further. In ARA's core this means **recon is read-only** and **`characterize` is
   consent-gated** (see Hard rules); the engines (wmx/wcx) enforce the wall when they measure and
   launch.
2. **Reliability** — *every component is properly tested.* `fail_under = 100` (statement + branch);
   new code lands with tests (see Conventions). A component you can't trust isn't shipped.
3. **Accuracy** — *report true data; never lie to the user.* For deterministic recon **and**
   non-deterministic model output alike: report the user's *real* environment, never ARA's
   internals; never claim something ARA didn't observe; `unknown` is a first-class answer
   (distinguish measured / curated / unknown); never surface a model's hallucination as fact.

### ARA-specific design values

- **Well-scoped tools.** Each command does one clear job with a predictable boundary:
  - `detect` — **read-only recon**. Observes the machine; never stresses, benchmarks, or
    loads an ML engine.
  - `status` — running AI/ML processes, right now.
  - `python` — every interpreter + its AI libraries + install cautions.
  - `apps` — installed AI/ML apps, versions, source, and Homebrew drift.
  - `mlx` — the MLX ecosystem + Apple readiness.
  - `profile` — **engine-free** analytic capability assessment: estimates the safe memory
    budget from recon facts; never loads an engine or a model.
  - `characterize` — **the command that measures**: opt-in; crosses the seam into the engine to
    find a model's real safe context ceiling (refusing before it risks the memory wall).
  - `recommend` — ranks cached models that fit this machine's budget, by estimated usable context.
  - `run` — governed one-shot inference, capped under the measured safe ceiling (CPU · MLX · CUDA).
  - `models` / `search` — catalog cached models (with measured ceilings) / search the HF Hub.
- **Broad compatibility.** Cover the open-source AI ecosystem widely — engines (MLX,
  llama.cpp, Ollama, LM Studio, vLLM), model stores (HF, Ollama, LM Studio, Jan, GPT4All),
  frameworks (PyTorch, transformers, TensorFlow), and apps — not one vendor's corner.

## The architecture boundary (don't break this)

- **Pure-Python core, swappable backend adapters.** The core (`ara/detect.py`, `cli.py`,
  recon modules) must **never import a hardware-specific engine**. Backends live behind a
  registry and are loaded lazily — only the one chosen for the machine.
- **Apple backend wraps [`wmx-suite`](https://github.com/willsarg/wmx-suite).** The engine
  import happens *inside* the adapter's functions, not at module load — so nothing
  MLX-shaped loads until ARA actually runs the engine.
- **Engines install on demand, not as dependencies.** The hardware engine is **not** in
  `pyproject.toml`. ARA probes the machine and installs the matched suite at runtime via
  `ara install` (`ara/engines.py` is the catalog + `uv pip install git+<spec>` logic). This
  keeps the core universal, the lock engine-free, and `uv sync` identical on every OS — and
  never ships MLX to a non-Apple machine. `--engine {wmx|wcx|cpu|auto}` is the consent surface
  (the flag itself authorizes the install, so it stays scriptable).

## Hard rules

These are how **Rule #1 (Safety)** and **Rule #3 (Accuracy)** are enforced in the recon core:

- **Recon is read-only.** Nothing under `detect`/`status`/`python`/`apps`/`mlx` may stress
  the machine, load a model, or mutate state.
- **Measuring is consent-gated.** `characterize` is the only command that loads an engine and a
  model (downloading weights on demand) — it runs only with explicit user opt-in. `profile` stays
  engine-free and read-only.
- **Advisory, never destructive.** ARA surfaces facts and considerations. It does **not**
  run or prescribe state-mutating commands on the user's behalf. (A flag may *describe* a
  fix; it must not tell the user to run something that silently destroys state.)
- **Honest about the user's environment, not ARA's.** When probing tools/interpreters, strip
  ARA's own virtualenv so results reflect what the *user* has, not ARA's bundled deps.

## Conventions

- **`uv` only.** No `pip install --break-system-packages`. The HF CLI is `hf`, not the
  deprecated `huggingface-cli`.
- **Tests are the bar.** `fail_under = 100` (statement + branch). New code lands with tests.
  The suite runs **without** `wmx-suite` on purpose — it proves the core stays engine-free;
  the seam is covered via a fake `wmx_suite`.
- **Planning/design docs live in the private vault, not the repo.** This repo is code +
  standard community files. Don't add design specs or logs here.
- **Write portable; claim only what's tested.** Shared layers (the engine env, worker IPC,
  paths) must be OS-agnostic — use `pathlib`/`os.path`, branch interpreter/venv layout on
  `os.name` (`Scripts\python.exe` vs `bin/python`), keep OS-specific recon (sysctl, Homebrew
  paths) behind a platform guard. ARA is developed on macOS (Apple Silicon) and tested green on
  macOS (CPU + MLX), Windows (CPU + CUDA, RTX 2070), and Linux (CPU); **claim a platform only once
  the suite is green there** — CUDA-on-Linux shares the Windows code path but isn't claimed yet
  (no NVIDIA-on-Linux box has run it).

## License & AI agents (Apache 2.0)

Project ARA is licensed **Apache-2.0** (`LICENSE`, `NOTICE`). Abide by it in both directions.

**Contributing here (inbound — Apache §5, inbound = outbound; no CLA/DCO):**
- All contributions are under Apache-2.0.
- Add only original code, or code under an Apache-2.0-compatible permissive license (MIT/BSD/ISC/Apache-2.0); preserve its copyright/license and record third-party components in `NOTICE`.
- Never introduce GPL/LGPL/AGPL, proprietary, or unknown-provenance code.
- Start every new source file with: `# SPDX-License-Identifier: Apache-2.0` and `# Copyright 2026 Will Sarg`.
- Do not alter or remove `LICENSE`, `NOTICE`, or existing SPDX headers.

**Cloning / forking / redistributing (outbound — Apache §4):**
- Keep `LICENSE` and `NOTICE` intact in any copy or fork.
- Retain all SPDX headers and copyright notices in files you carry.
- State significant changes you make.
- You may relicense *your own* additions; the Apache-2.0-covered files stay Apache-2.0.
