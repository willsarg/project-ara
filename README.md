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
> ARA tells you what local AI work this computer can safely handle, then governs a model while
> it runs. Start with read-only machine inspection, measure one model's safe limit, and generate
> a first answer without having to learn the local-AI toolchain up front.

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

## 🚀 Your first ten minutes

Install the released command with [`uv`](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv tool install project-ara
```

`uv tool install` is the **pipx-style** option: it puts the `ara` command in its own isolated
Python environment instead of adding packages to the Python environment for your application.

Now choose the block that matches what `ara detect` reports. Bare `ara install` automatically
selects only Apple Silicon or NVIDIA acceleration. A CPU installation is explicit because it may
compile llama.cpp locally and can take longer.

### Apple Silicon

```bash
ara detect
ara install
ara characterize mlx-community/SmolLM-135M-Instruct-4bit --engine mlx
ara run mlx-community/SmolLM-135M-Instruct-4bit "Explain local AI simply"
```

### NVIDIA GPU (Windows verified)

This CUDA path is verified on Windows. NVIDIA on Linux is not yet claimed; use the CPU
block below on Linux until that lane has passed ARA's full verification suite.

```bash
ara detect
ara install
ara characterize HuggingFaceTB/SmolLM-135M-Instruct --engine cuda
ara run HuggingFaceTB/SmolLM-135M-Instruct "Explain local AI simply"
```

### CPU (no GPU required)

This is the portable path for Windows or Linux without an NVIDIA GPU, and Intel Macs use this CPU
path too. Windows and Linux CPU execution are verified. Intel Mac has not yet passed ARA's full
verification suite, so that route is available but not currently claimed as verified.

```bash
ara detect
ara install --engine cpu
ara characterize bartowski/SmolLM2-135M-Instruct-GGUF --engine cpu
ara run bartowski/SmolLM2-135M-Instruct-GGUF "Explain local AI simply"
```

Windows uses a prebuilt CPU engine wheel. On Linux and macOS, the engine may compile llama.cpp
locally; when it does, the machine needs working C/C++ build tools. ARA reports a build failure
instead of changing system tooling on your behalf.

If `ara detect` reports Vulkan as usable on an x86_64 Windows or Linux machine, GPU offload is an
optional next step through `ara install --engine vulkan`. The CPU path above remains the simplest
starting point and is selected explicitly.

These are deliberately tiny workflow demonstration models, not quality recommendations. The
`characterize` step downloads missing model weights, then measures the largest context this
machine can run safely. Expect it to use network bandwidth, disk space, and time; the exact cost
depends on the model and machine. The later `run` refuses to exceed the ceiling that was measured.

**Optional preflight:** `ara profile` estimates the machine's safe memory budget without loading
an engine or model. It is useful for orientation, but it is not required before `characterize`.

No arguments shows the same machine-specific first-run path:

```bash
ara
```

### The local-AI words used above

- **model:** the learned weights and configuration that produce an answer.
- **engine:** the software that loads those weights and performs the computation on CPU or GPU.
- **Transformers format:** the common Hugging Face model layout used by ARA's MLX and CUDA engines.
- **GGUF format:** a file format designed for efficient local inference, used by the CPU and
  GPU-offload llama.cpp lanes.
- **quantization:** storing weights with fewer bits to reduce memory and disk use, usually with a
  quality tradeoff.
- **token:** a piece of text processed by a model; it may be a word, part of a word, or punctuation.
- **context:** the tokens a model can consider at once, including the prompt and generated answer.
- **characterization:** ARA's measurement of a particular model artifact and engine on this
  machine, producing the safe context ceiling used to govern later runs.

For fleet nodes, install the outbound HTTP client too:

```bash
uv tool install 'project-ara[node]'
```

From a source checkout, use the same commands through `uv run`:

```bash
uv sync --frozen --group dev
uv run ara detect
uv run pytest                    # 100% statement + branch coverage gate
```

Everything below is a subcommand. The hardware engine isn't a core dependency — the install step
adds the selected engine on demand, so `uv sync` stays lean and works the same on every platform.

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
| `ara install` / `ara uninstall` | Add or remove the engine matched to this machine. `--engine {mlx\|cuda\|cpu\|vulkan\|cuda-gguf\|auto}` picks it (`auto` resolves the GPU engine for this hardware — `mlx` on Apple Silicon, `cuda` when an NVIDIA GPU is present; pass `--engine cpu` for the built-in portable CPU path); the flag is the consent, so it's scriptable. Engines: `mlx` (Apple Silicon/MLX), `cuda` (NVIDIA/CUDA full-GPU), the built-in `cpu` (llama.cpp), `vulkan` (GGUF on a compatible x86_64 Windows/Linux GPU, including integrated GPUs), and `cuda-gguf` (GGUF on NVIDIA via **partial offload** — a two-wall hybrid that splits layers across VRAM and system RAM). |
| `ara profile` | **Engine-free** analytic capability assessment: estimates this machine's safe memory budget from `detect` facts, and with `--model` checks whether that model's weights + context fit. Stored MLX measurements are shown as history until an execution command verifies the live Metal authority. Never loads an engine or a model. |
| `ara characterize <model>` | **Measures** a model's real safe context ceiling on this machine — an empirical ramp under MLX, CUDA, CPU, Vulkan, cuda-gguf, or Ollama that applies the engine's pre-load memory gate, then stores the ceiling for `models recommend` / `run`. Native ARA engines also enforce an in-worker watchdog; the external Ollama daemon cannot provide that guarantee (see Safety). The command that crosses into an engine. |
| `ara models search <query>` | Search the Hugging Face Hub for models matching a query (ids, downloads, likes). |
| `ara models recommend` | Rank the models in your local HF cache that fit this machine's estimated budget. Add `--engine ollama` to rank supported local Ollama artifacts with exact reusable measurements first; analytic estimates remain clearly labeled comparison-only and cannot authorize execution. With `--engine ollama --use-case ...`, only a locally measured score still bound to the current exact manifest, daemon/config authority, request policy, and runtime target can influence rank; legacy or drifted evidence is labeled unknown, and cross-runtime imported scores are never substituted. Read-only — no model load. |
| `ara models show <model>` | Show one model's architecture and per-engine measured safe ceiling. Add `--engine ollama` for the exact cached manifest, Ollama capabilities/parameters, and reusable or display-only characterization evidence. |
| `ara run <model> "<prompt>"` | **Governed one-shot inference**: generate a completion capped at the model's characterized safe ceiling, never over the wall. Refuses if the model hasn't been characterized yet. `--ctx N` may lower the effective context but cannot replace or exceed the measured ceiling. Runs on every engine (CPU, MLX, CUDA, Vulkan, cuda-gguf). Use `-` to read the complete UTF-8 prompt from standard input or `--prompt-file PATH` to read it from a file; every prompt source is limited to 1 MiB. |
| `ara benchmark <model> --use-case <coding\|reasoning\|agentic\|extraction\|rag>` | **Governed capability benchmark**: run a judge-free probe set against the model's actual quant, under its characterized ceiling, and store the measured score. Add `--engine ollama` to benchmark an exact local cached Ollama artifact through the existing loopback daemon; this requires reusable measured Ollama evidence, uses the canonical base model directly, and governs ARA's sequential requests while outside Ollama clients remain concurrent. Ollama stays optional. `--max-tokens N` lifts the generation cap for thinking models. Coding output executes only with `--exec-consent`; ARA uses macOS Seatbelt and skips coding execution on hosts where it cannot provide a sandbox. |
| `ara serve [model]` | Stand up a **governed OpenAI-compatible endpoint** through an existing Ollama installation or the native MLX server. Ollama serving requires an exact reusable characterization; `--ctx` may lower but cannot replace that measured authority, and `serve` never pulls missing weights implicitly. With no model, ARA selects the largest reusable Ollama characterization through the same logic as `models recommend --engine ollama`. The handoff is verified at setup time; later reloads follow the daemon's normal cache/eviction policy and are not claimed as continuously governed. |
| `ara hub` | Build and run ARA's version-matched **fleet coordinator** in Docker, attached in the foreground. SQLite state persists in ARA's host data directory; `--bind`, `--port`, `--data-dir`, and `--rebuild` control the host deployment. The default bind is loopback; put HTTPS/TLS termination in front before enrolling remote nodes. |
| `ara hf login` / `logout` / `status` | Manage your Hugging Face token (needed for gated models). `login` reads it from a hidden prompt, piped stdin, or `--token`, verifies it against the Hub, and stores it in the **standard HF token file** — so every fetch and engine worker picks it up. `status` shows who you're logged in as (never the token); `logout` removes it. |
| `ara node enroll` / `run` | Enroll with a coordinator and run ARA's push-only node loop. Nodes dial out; the coordinator never opens SSH or connects back into them. Install the `project-ara[node]` extra for the HTTP client. |
| `ara node install` / `start` / `status` / `stop` / `uninstall` | Manage the node as a Linux systemd user service. ARA does not silently enable lingering or any other administrator policy. |
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
- **Everything else** → the built-in **CPU engine** (llama.cpp on system RAM) — the portable
  path, so any machine with enough RAM can run models without GPU acceleration.
- **Vulkan-capable GPU on x86_64 Windows/Linux** → the opt-in **Vulkan engine** (llama.cpp
  GGUF with GPU offload), including integrated GPUs that use shared system memory.
- **NVIDIA, oversized for VRAM** → the built-in **cuda-gguf** engine (llama.cpp GGUF, partial
  offload `n_gpu_layers=K`) — ARA's first **two-wall** engine, governing discrete VRAM *and*
  system RAM at once so a model too big for VRAM runs K layers on the GPU and the rest on CPU.
- **Existing Ollama installation** → a first-class, optional adapter for exact model inspection,
  certified recommendation, governed characterization, one-shot inference, and serving. Ollama is
  never required for recon or for ARA's native engines.

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
| **Windows** | CPU + CUDA — full test suite green, inference verified on an 8 GiB NVIDIA Turing GPU |
| **Linux** | CPU — full suite and CPU integration tests green |

CUDA on Linux shares the same native CUDA engine path as Windows, but isn't claimed here until the
full suite is green on an NVIDIA-on-Linux box.

Intel Macs route to the portable CPU engine by design, but are not listed in the verified table
until that path has passed ARA's full suite and live CPU integration on Intel macOS.

---

## 🛟 Safety

`ara detect` (including `--python`, `--apps`, `--runtime`, and `--models`) and `ara status` are
**strictly read-only** — they observe, never stress, benchmark, or load a model. Measuring is opt-in:
`ara characterize` is the command that crosses into the engine. Native ARA engines refuse before
the governed memory boundary and use an in-worker watchdog to abort a probe as usage approaches the
limit. Each native engine governs against the right boundary — physical RAM (CPU; swap is reported
but never counted), MLX's live Metal working-set limit (Apple), or VRAM (CUDA).
CUDA-GGUF keeps both system RAM and VRAM as direct gates; its fitted curve is explicitly absolute
system RAM, while every point records the separately checked VRAM observation and both budgets.

Ollama is an external daemon, so ARA cannot provide a trustworthy active watchdog after a load
begins. Before **each** Ollama probe, ARA instead fails closed unless it can conservatively bound
the model's expanded residency, requested-context KV cache, and runtime overhead against every
possible physical wall. The after-load process and memory snapshots are evidence, not an active
abort mechanism. `ara run` stays capped under the resulting reusable measured ceiling. ARA also
resolves each model to an immutable
artifact, records that identity with the measurement, and verifies it
again before loading; changed or ambiguous weights are refused instead of being presented as the
characterized model. ARA stays
**advisory** — it surfaces what's true and what to consider; it never runs destructive or
system-mutating commands on your behalf.

Fleet control follows the same boundary. `ara hub` binds to loopback by default and runs the
version-matched coordinator in Docker with host-persisted SQLite state. Nodes phone home, work is
ownership-bound and journaled across restarts, and incomplete or failed work is never reported as a
successful model result. Put a trusted TLS reverse proxy in front before enrolling remote nodes.

---

## 🤝 Contributing

Issues and PRs are welcome — see [CONTRIBUTING.md](./CONTRIBUTING.md) for setup and
conventions, and [AGENTS.md](./AGENTS.md) for the project's purpose and boundaries.

## 📄 License

[Apache 2.0](./LICENSE) © Will Sarg
