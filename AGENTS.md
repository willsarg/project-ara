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
   consent-gated** (see Hard rules); the engines (`mlx`/`cuda` and the GGUF lanes) enforce the wall when they measure and
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
    loads an ML engine. Its `--python`, `--apps`, `--runtime`, and `--models` facets report
    interpreter/library inventory, installed apps, runtime/backend readiness, and physical model
    caches respectively.
  - `status` — live ARA-owned activity, right now; never a generic process monitor.
  - `profile` — **engine-free** analytic capability assessment: estimates the safe memory
    budget from recon facts; never loads an engine or a model.
  - `characterize` — **the command that measures**: opt-in; crosses the seam into the engine to
    find a model's real safe context ceiling (refusing before it risks the memory wall).
  - `benchmark` — **the measured-capability tier**: opt-in; runs probe sets through the model's
    *actual* quant on the engine and scores them judge-free against canonical references
    (HumanEval / GSM8K / SQuAD), storing the score. The deepest measurement — after `characterize`
    proves a model *fits*, `benchmark` measures how *well* it performs. `--max-tokens N` lifts the
    generation cap for thinking models. Coding is execution-consent-gated (and sandboxed only where
    a sandbox exists — macOS Seatbelt; skipped on un-sandboxable hosts).
  - `models recommend` — ranks cached models that fit this machine's budget, by estimated usable context
    (and measured capability, where `benchmark` has run, via `--use-case`).
  - `run` — governed one-shot inference, capped under the measured safe ceiling (CPU · MLX · CUDA).
  - `serve` — governed OpenAI-compatible endpoint, capped under the measured ceiling (on Ollama / MLX).
  - `hub` — the fleet coordinator server. Builds and runs ARA's version-matched coordinator in
    Docker, attached in the foreground, with its SQLite state bind-mounted to the host. Nodes phone
    home to it; it never reaches into nodes.
  - `models show` / `models search` — inspect one model (with measured ceilings) / search the HF Hub.
  - `hf login` / `logout` / `status` — manage the Hugging Face token (needed for gated models). An
    **action** command (writes the standard HF token store, so every fetch + worker reads it), not
    recon; verifies via the Hub and never prints the token.
- **Broad compatibility.** Cover the open-source AI ecosystem widely — engines (MLX,
  llama.cpp, Ollama, LM Studio, vLLM), model stores (HF, Ollama, LM Studio, Jan, GPT4All),
  frameworks (PyTorch, transformers, TensorFlow), and apps — not one vendor's corner.

### Frozen public command tree

The visible tree is `detect` (facets `--python`, `--apps`, `--runtime`, `--models`), `profile`,
`status`, `models {search,recommend,show}`, `run`, `serve`, `characterize`, `benchmark`, `install`,
`uninstall`, `hub`, `node {enroll,install,run,start,status,stop,uninstall}`,
`hf {login,logout,status}`, and `doctor`.
`models list` is deferred. `status` reports only live ARA-owned activity; it is not a generic
process monitor. `detect --runtime` reports common runtime/backend inventory on every platform and
adds MLX ecosystem detail only on Apple Silicon.

For one compatibility release, top-level `python`, `apps`, `mlx`, `search`, and `recommend`, plus
`models MODEL`, remain hidden aliases with deprecation warnings. New code and docs use only the
canonical tree. The console script `ara` and `python -m ara` share the one blessed main; production
subprocesses and service templates use `python -m ara`, never the internal `ara.cli` module.

## The architecture boundary (don't break this)

- **Pure-Python core, swappable backend adapters.** The core (`ara/detect.py`, `cli.py`,
  recon modules) must **never import a hardware-specific engine**. Backends live behind a
  registry and are loaded lazily — only the one chosen for the machine.
- **Apple and CUDA use native ARA engine packages.** Their pure-Python sources ship under
  `ara/_engine_packages/mlx` and `ara/_engine_packages/cuda`. They are independently installable,
  but never imported by ARA core: adapters invoke `ara_engine_mlx.*` / `ara_engine_cuda.*` modules
  only as subprocesses inside isolated engine environments.
  - **The predecessor Apple/MLX and CUDA repositories are retired.** ARA is the sole source of
    truth; do not clone, edit, sync, commit to, or release the retired repositories as part of ARA
    work unless explicitly asked. There is no re-vendoring workflow: change the native package in
    `_engine_packages` directly.
- **Engines install on demand, not as dependencies.** The hardware engine's heavy deps (MLX, torch)
  are **not** in ARA's `pyproject.toml`. ARA probes the machine and installs the matched suite at
  runtime via `ara install` (`ara/engines.py` is the catalog). The native packages install from
  **bundled source** (`uv pip install ara/_engine_packages/<key>`) — no git fetch, so a release
  installs the exact engine code in its wheel, reproducibly and offline. `ARA_MLX_SOURCE` and
  `ARA_CUDA_SOURCE` can override those roots with a local checkout installed editable for engine
  development. The predecessor source-variable names remain one-release compatibility aliases and
  must not appear in new instructions. This keeps the core universal, the lock engine-free, and
  `uv sync` identical on every OS — and never installs MLX/torch on a machine that does not select
  it. `--engine {mlx|cuda|cpu|vulkan|cuda-gguf|auto}` is the
  consent surface (the flag itself authorizes the install, so it stays scriptable). `vulkan`
  (GGUF on an AMD APU's iGPU) and `cuda-gguf` (GGUF on NVIDIA via **partial** offload —
  `n_gpu_layers=K`) are opt-in GPU-offload lanes; `cuda-gguf` is ARA's first **two-wall** engine,
  governing discrete VRAM *and* system RAM at once (a model too big for VRAM runs K layers on the
  GPU, the rest on CPU). The full engine matrix is runtime × backend: mlx/torch/llamacpp/ollama ×
  apple/cuda/cpu/vulkan.
- **Compatibility aliases last one release.** New commands and configuration must use the canonical
  names. During the compatibility release only, `wmx` → `mlx` and `wcx` → `cuda` CLI inputs,
  `ARA_WMX_SOURCE` → `ARA_MLX_SOURCE` and `ARA_WCX_SOURCE` → `ARA_CUDA_SOURCE` package-source
  overrides, and `WMX_SUITE_MARGIN_GB` → `ARA_MLX_MARGIN_GB` and `WCX_SUITE_MARGIN_GB` →
  `ARA_CUDA_MARGIN_GB` direct-engine margin overrides remain accepted with lower precedence and
  deprecation warnings. They are scheduled for deletion in the following release.

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
  The suite runs **without** MLX, torch, or CUDA engine dependencies on purpose — it proves the
  core stays engine-free; subprocess seams are covered with fakes.
- **Planning/design docs live in the private vault, not the repo.** This repo is code +
  standard community files. Don't add design specs or logs here.
- **Write portable; claim only what's tested.** Shared layers (the engine env, worker IPC,
  paths) must be OS-agnostic — use `pathlib`/`os.path`, branch interpreter/venv layout on
  `os.name` (`Scripts\python.exe` vs `bin/python`), keep OS-specific recon (sysctl, Homebrew
  paths) behind a platform guard. ARA is developed on macOS (Apple Silicon) and tested green on
  macOS (CPU + MLX), Windows (CPU + CUDA, 8 GB NVIDIA Turing GPU), and Linux (CPU); **claim a platform only once
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
