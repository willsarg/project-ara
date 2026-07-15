<div align="center">

# 🌍 Project ARA

### `ara` — **AI Runs Anywhere**: assess any computer for AI workloads, then run local models safely on whatever hardware you've got

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](./pyproject.toml)
[![packaged with uv](https://img.shields.io/badge/packaged_with-uv-DE5FE9.svg?style=for-the-badge)](https://github.com/astral-sh/uv)
[![Backend-agnostic](https://img.shields.io/badge/backend-agnostic-4C9A2A.svg?style=for-the-badge)](#-architecture)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](./CONTRIBUTING.md)
[![Buy me a coffee](https://img.shields.io/badge/Buy_me_a_coffee-FFDD00.svg?style=for-the-badge&logo=buymeacoffee&logoColor=black)](https://www.buymeacoffee.com/willsarg)

</div>

> [!NOTE]
> ARA is the tool you reach for to **honestly assess any machine with a Python runtime for
> AI work** — what hardware it has, what's installed, where your models and interpreters
> actually live — and then run local models right up to the hardware's safe edge, never
> over. Recon is read-only and runs anywhere; running models works today on CPU, Apple
> Silicon (MLX), NVIDIA (CUDA — full-GPU, or a two-wall GGUF partial-offload hybrid), and
> AMD iGPUs (Vulkan).

---

## 🧠 Why this exists

Local AI tooling is a maze. You've got Python installed five different ways (macOS,
Homebrew, python.org, pyenv, conda, uv) and no idea which one has your ML libraries. Apps
get installed through Homebrew and then silently self-update past it. Your models are
scattered across HF cache, Ollama, LM Studio, Jan, and GPT4All. And nothing tells you how
hard you can actually push the hardware before it falls over.

ARA cuts through that with **high-level, honest readouts** and a **safety governor** that
knows your machine's real limits. The design values, in order:

- **Honesty** — report your *real* environment, never ARA's own internals; never claim
  something it didn't observe.
- **Well-scoped tools** — each command does one clear job (`detect` observes, `status`
  watches, `characterize` measures), so the output is predictable.
- **Broad compatibility** — across the open-source AI ecosystem (engines, model stores,
  frameworks, apps), not one vendor's corner of it.

---

## 🚀 Quick start

```bash
uv sync                          # install the pure-Python core into .venv (works on any OS)
uv run ara                       # the landing screen + getting-started path
uv run ara detect                # read-only recon of this machine
uv run ara install               # add the engine matched to this machine (MLX / CUDA / CPU)
uv run ara characterize <model>  # measure that model's safe context ceiling on this machine
uv run ara run <model> "..."     # one-shot inference, governed under the measured ceiling
```

No arguments shows what ARA can do for this machine. Everything below is a subcommand.
The hardware engine isn't a dependency — `ara install` adds it on demand, so `uv sync`
stays lean and works the same on every platform.

Use `ara detect --runtime` for cross-platform runtime/backend readiness and
`ara detect --runtime --json` for its machine-readable form.

The installed `ara` console script and `python -m ara` are the two supported spellings of the
same production entrypoint. In a checkout, prefix either with `uv run`.

---

## 🔭 Commands

| Command | What it does |
|---|---|
| `ara detect` | Read-only machine recon: chip, memory, accelerator, storage, engines, frameworks, cached models, and installed AI/ML apps. Facets: `--python`, `--apps`, `--runtime`, and `--models`. Runtime inventory is cross-platform; MLX ecosystem detail is added only on Apple Silicon. Never stresses the machine or loads an engine. |
| `ara status` | What ARA itself is doing right now: idle, searching, characterizing, benchmarking, running, serving, or hosting the fleet coordinator. It is not a generic process monitor. |
| `ara install` / `ara uninstall` | Add or remove the engine matched to this machine. `--engine {mlx\|cuda\|cpu\|vulkan\|cuda-gguf\|auto}` picks it (`auto` resolves the GPU engine for this hardware — `mlx` on Apple Silicon, `cuda` when an NVIDIA GPU is present; pass `--engine cpu` for the built-in CPU fallback); the flag is the consent, so it's scriptable. Engines: `mlx` (Apple Silicon/MLX), `cuda` (NVIDIA/CUDA full-GPU), the built-in `cpu` (llama.cpp), `vulkan` (GGUF on an AMD iGPU's shared memory), and `cuda-gguf` (GGUF on NVIDIA via **partial offload** — a two-wall hybrid that splits layers across VRAM and system RAM). |
| `ara profile` | **Engine-free** analytic capability assessment: estimates this machine's safe memory budget from `detect` facts (grounded in a measured wall if you've characterized before), and with `--model` checks whether that model's weights + context fit. Never loads an engine or a model. |
| `ara characterize <model>` | **Measures** a model's real safe context ceiling on this machine — an empirical ramp under the engine that refuses before it risks the memory wall, then stores the ceiling for `models recommend` / `run`. The command that crosses into the engine. |
| `ara models search <query>` | Search the Hugging Face Hub for models matching a query (ids, downloads, likes). |
| `ara models recommend` | Rank the models in your local HF cache that fit this machine's estimated budget, ordered by estimated usable context, marking the ones already characterized here. Analytic — no engine or model load. |
| `ara models show <model>` | Show one model's architecture and per-engine measured safe ceiling. |
| `ara run <model> "<prompt>"` | **Governed one-shot inference**: generate a completion capped at the model's characterized safe ceiling, never over the wall. Refuses if the model hasn't been characterized yet. Runs on every engine (CPU, MLX, CUDA, Vulkan, cuda-gguf). |
| `ara benchmark <model> --use-case <coding\|reasoning\|agentic\|extraction\|rag>` | **Governed capability benchmark**: run a probe set for that use-case under the engine (capped at the safe ceiling, like `run`) and store the measured score. The model's own chat template is applied, so template-strict instruct models score honestly. |
| `ara serve <model>` | Stand the model up as a **governed OpenAI-compatible endpoint** — on Ollama, or the MLX server with `--engine mlx` — capped at the model's safe context ceiling, and return the endpoint. |
| `ara hub` | Build and run ARA's version-matched **fleet coordinator** in Docker, attached in the foreground. SQLite state persists in ARA's host data directory; `--bind`, `--port`, `--data-dir`, and `--rebuild` control the host deployment. The default bind is loopback; put HTTPS/TLS termination in front before enrolling remote nodes. |
| `ara hf login` / `logout` / `status` | Manage your Hugging Face token (needed for gated models). `login` reads it from a hidden prompt, piped stdin, or `--token`, verifies it against the Hub, and stores it in the **standard HF token file** — so every fetch and engine worker picks it up. `status` shows who you're logged in as (never the token); `logout` removes it. |
| `ara node enroll` / `install` / `run` / `start` / `status` / `stop` / `uninstall` | Enroll and operate ARA's push-only node daemon. |
| `ara doctor` | Inspect ARA's stored machine identity and local record counts. |

Commands that support structured output expose command-level `--json`; Click owns invalid syntax
and usage errors. `detect` also supports `--include` / `--exclude` for its full report.

For one compatibility release, the old top-level `search`, `recommend`, `python`, `apps`, and `mlx`
spellings and `models MODEL` remain callable with deprecation warnings, but are hidden from help.
Use the canonical tree above in scripts and documentation. `models list` is deferred.

---

## 🧩 Architecture

ARA is a **pure-Python core** with **swappable backend adapters**. The core never imports a
hardware-specific engine; it picks a backend for the machine and loads only that one.

- **Apple Silicon** → ARA's native **MLX engine**,
  which finds each model's safe context ceiling *without crashing the machine*.
- **NVIDIA / CUDA** → ARA's native **CUDA engine** (torch).
- **Everything else** → the built-in **CPU engine** (llama.cpp on system RAM) — the universal
  fallback, so any machine with enough RAM can run models, just not GPU-accelerated.
- **AMD iGPU** → the built-in **Vulkan engine** (llama.cpp GGUF on the integrated GPU's shared
  memory) — one wall, since the iGPU and CPU share RAM.
- **NVIDIA, oversized for VRAM** → the built-in **cuda-gguf** engine (llama.cpp GGUF, partial
  offload `n_gpu_layers=K`) — ARA's first **two-wall** engine, governing discrete VRAM *and*
  system RAM at once so a model too big for VRAM runs K layers on the GPU and the rest on CPU.

The engine is **not a core dependency**. ARA ships the native MLX and CUDA package sources under
`ara/_engine_packages/{mlx,cuda}` and installs only the matched package and its heavy dependencies
into its **own isolated environment** on demand (`ara install`) — so the core stays universal,
`uv sync` is identical on every OS, and you never download MLX onto an NVIDIA box or vice-versa.
The catalog of engines and the install logic live in [`ara/engines.py`](./ara/engines.py). See
[AGENTS.md](./AGENTS.md) for the design boundary and conventions.

**Platform support.** Recon is OS-agnostic and runs anywhere. Running models is verified live on:

| OS | Engines verified |
|---|---|
| **macOS (Apple Silicon)** | CPU + MLX — the primary development machine |
| **Windows** | CPU + CUDA — full test suite green, inference verified on an RTX 2070 |
| **Linux** | CPU — full suite and CPU integration tests green |

CUDA on Linux shares the same native CUDA engine path as Windows, but isn't claimed here until the
full suite is green on an NVIDIA-on-Linux box.

---

## 🛟 Safety

`ara detect` (including `--python`, `--apps`, `--runtime`, and `--models`) and `ara status` are
**strictly read-only** — they observe, never stress, benchmark, or load a model. Measuring is opt-in:
`ara characterize` is the command that crosses into the engine, and it finds a model's safe
context ceiling by **refusing before it ever loads past the memory wall** and aborting a probe
the moment usage approaches the limit. Each engine governs against the right wall — physical RAM
(CPU; swap is reported but never counted), the MLX unified-memory wall (Apple), or VRAM (CUDA) —
and `ara run` stays capped under that measured ceiling. Model ids are validated before they're
ever passed to an engine subprocess. ARA stays **advisory** — it surfaces what's true and what to
consider; it never runs destructive or system-mutating commands on your behalf.

---

## 🤝 Contributing

Issues and PRs are welcome — see [CONTRIBUTING.md](./CONTRIBUTING.md) for setup and
conventions, and [AGENTS.md](./AGENTS.md) for the project's purpose and boundaries.

## 📄 License

[Apache 2.0](./LICENSE) © Will Sarg
