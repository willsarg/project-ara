<div align="center">

# 🌍 Project ARA

### `ara` — **AI Runs Anywhere**: assess any computer for AI workloads, then run local models safely on whatever hardware you've got

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](./LICENSE)
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
> over. Recon is read-only and runs anywhere; running models is backend-specific (Apple
> Silicon today, more later).

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
  watches, `profile` measures), so the output is predictable.
- **Broad compatibility** — across the open-source AI ecosystem (engines, model stores,
  frameworks, apps), not one vendor's corner of it.

---

## 🚀 Quick start

```bash
uv sync                # install the pure-Python core into .venv (works on any OS)
uv run ara             # the landing screen + getting-started path
uv run ara detect      # read-only recon of this machine
uv run ara install     # add the engine matched to this machine (Apple Silicon today)
```

No arguments shows what ARA can do for this machine. Everything below is a subcommand.
The hardware engine isn't a dependency — `ara install` adds it on demand, so `uv sync`
stays lean and works the same on every platform.

---

## 🔭 Commands

| Command | What it does |
|---|---|
| `ara detect` | Read-only machine recon: chip, memory, accelerator, storage, **engines** (what can launch models), **frameworks** (your Python's AI libs), **models** (HF / Ollama / LM Studio / Jan / GPT4All), and an **AI/ML apps** summary. Never stresses the machine or loads an engine. |
| `ara status` | Live view of AI/ML processes running *right now* — what's holding memory before you launch something. |
| `ara python` | Every Python interpreter on the system (macOS / Homebrew / python.org / pyenv / conda / uv / asdf), which has which AI libraries, and which you shouldn't `pip install` into. |
| `ara apps` | Full inventory of installed AI/ML apps with versions and install source (App / Homebrew cask / formula), flagging real duplicates and self-update-vs-Homebrew drift. |
| `ara mlx` | The MLX ecosystem (Apple Silicon): libraries by modality + readiness (Metal GPU, cached `mlx-community` models, LM Studio's MLX runtime). |
| `ara install` / `ara uninstall` | Add or remove the engine matched to this machine. `--engine {wmx\|wcx\|auto}` picks it (`auto` = whatever fits this hardware); the flag is the consent, so it's scriptable. Today: `wmx` ([`wmx-suite`](https://github.com/willsarg/wmx-suite), Apple Silicon); `wcx` (CUDA) is coming. |
| `ara profile` | Measure this machine's **safe memory limits** — opt-in calibration against a tiny model, crossing into the engine. Installs the engine first if you pass `--engine`. |
| `ara recommend` | *(planned)* Best model per modality that fits this machine — curated catalog × measured wall. |
| `ara run <model>` | *(planned)* Launch a model safely — right up to the edge, never over. |

Recon commands share `--json` (machine-readable) and `--include` / `--exclude` (show only
or hide specific sections, e.g. `ara detect --exclude models`).

---

## 🧩 Architecture

ARA is a **pure-Python core** with **swappable backend adapters**. The core never imports a
hardware-specific engine; it picks a backend for the machine and loads only that one.

- **Apple Silicon** → the [`wmx-suite`](https://github.com/willsarg/wmx-suite) engine (MLX),
  which finds each model's safe context ceiling *without crashing the machine*.
- **Other hardware** → recon works everywhere; running support arrives as backends land
  (NVIDIA / CUDA is next).

The engine is **not a dependency**. ARA probes the machine and installs the matched suite
on demand (`ara install`) — so the core stays universal, `uv sync` is identical on every
OS, and you never download MLX onto an NVIDIA box or vice-versa. The catalog of engines and
the install logic live in [`ara/engines.py`](./ara/engines.py). See [AGENTS.md](./AGENTS.md)
for the design boundary and conventions.

---

## 🛟 Safety

`ara detect` and the other recon commands are **strictly read-only** — they observe, never
stress, profile, or load a model. The only command that measures the machine, `ara profile`,
is **opt-in and consent-gated**, and the safe limits come from `wmx-suite`, which predicts
the hardware wall and stays under it rather than probing into the danger zone. ARA stays
**advisory** — it surfaces what's true and what to consider; it never runs destructive or
system-mutating commands on your behalf.

---

## 🤝 Contributing

Issues and PRs are welcome — see [CONTRIBUTING.md](./CONTRIBUTING.md) for setup and
conventions, and [AGENTS.md](./AGENTS.md) for the project's purpose and boundaries.

## 📄 License

[MIT](./LICENSE) © Will Sarg
