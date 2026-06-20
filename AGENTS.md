# AGENTS.md — purpose, boundaries, and conventions for ARA

This is the source of truth for *what ARA is and how to work on it*. Read it before
contributing (human or agent). [CONTRIBUTING.md](./CONTRIBUTING.md) covers the human
workflow (setup, landing a change); this covers the **why** and the **rules**.

## What ARA is

**Project ARA — "AI Runs Anywhere."** A tool you reach for to **honestly assess any machine
with a Python runtime for AI workloads**, then run local models safely on whatever hardware
is present. Apple Silicon is the first running backend; recon works everywhere.

### Design values (in priority order)

1. **Honesty.** Report the user's *real* environment, never ARA's own internals. Never claim
   something ARA didn't observe. When detection is uncertain, say so; don't guess.
2. **Well-scoped tools.** Each command does one clear job with a predictable boundary:
   - `detect` — **read-only recon**. Observes the machine; never stresses, benchmarks, or
     loads an ML engine.
   - `status` — running AI/ML processes, right now.
   - `python` — every interpreter + its AI libraries + install cautions.
   - `apps` — installed AI/ML apps, versions, source, and Homebrew drift.
   - `mlx` — the MLX ecosystem + Apple readiness.
   - `profile` — **the only command that measures**; opt-in and consent-gated; crosses the
     seam into the engine.
   - `recommend` / `run` — *planned* (curated catalog × measured wall; safe launch).
3. **Broad compatibility.** Cover the open-source AI ecosystem widely — engines (MLX,
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
  never ships MLX to a non-Apple machine. `--engine {wmx|wcx|auto}` is the consent surface
  (the flag itself authorizes the install, so it stays scriptable).

## Hard rules

- **Recon is read-only.** Nothing under `detect`/`status`/`python`/`apps`/`mlx` may stress
  the machine, load a model, or mutate state.
- **`profile` is consent-gated.** It only measures (and only downloads a calibration model)
  with explicit user opt-in.
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
  paths) behind a platform guard. But ARA is developed and tested on macOS (Apple Silicon),
  with Linux supported; **make no Windows support claims until it's actually been run there.**
