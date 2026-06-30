<div align="center">

# 🧪 Will's MLX Suite

### `wmx-suite` — find each model's safe context ceiling on Apple Silicon, *without crashing the machine*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](./pyproject.toml)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1_to_M4-000000.svg?style=for-the-badge&logo=apple&logoColor=white)](#-why-this-exists)
[![Built with MLX](https://img.shields.io/badge/Built_with-MLX-FF6F61.svg?style=for-the-badge)](https://github.com/ml-explore/mlx)
[![packaged with uv](https://img.shields.io/badge/packaged_with-uv-DE5FE9.svg?style=for-the-badge)](https://github.com/astral-sh/uv)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](./CONTRIBUTING.md)
[![Buy me a coffee](https://img.shields.io/badge/Buy_me_a_coffee-FFDD00.svg?style=for-the-badge&logo=buymeacoffee&logoColor=black)](https://www.buymeacoffee.com/willsarg)

</div>

> [!WARNING]
> **First rule: never crash the laptop.** The suite finds each model's safe context
> ceiling by *extrapolating from measurements taken well below the hardware wall* — it
> never probes into the danger zone.

---

## 🧠 Why this exists

MLX allocates **wired** (non-swappable) Metal buffers. The real crash ceiling isn't
total RAM — it's the GPU working-set limit (`max_recommended_working_set_size`), which the
suite reads **live on each machine** and treats as an exact, measured value (rounding
toward it is how you crash). On the reference M4 Pro testbed it is **17.18 GB** (67% of
25.77 GB). With almost no swap free, crossing it can hard-lock the system rather than fail
gracefully. So we **predict the wall and stay under it.**

### 🔬 Key facts the design is built on

- 📏 **Danger metric is OS-wired high-water** (`vm_stat` "Pages wired down"), *not* MLX's
  `get_peak_memory()`, which undercounts true footprint by ~40% (it excludes the buffer
  cache, which the OS still wires).
- ⚡ **The prefill transient spike** drives the crash — it grows several× faster than the
  steady-state KV cache and is invisible in MLX's reported peak.
- 🧬 **Only `full_attention` layers** grow KV with context; sliding/linear layers don't.

**Two memory classes** the suite classifies and handles differently:

| Class | Models | Cache | KV quantization |
|---|---|---|---|
| 🪟 **Sliding-window** | Gemma, GPT-OSS | `RotatingKVCache` | ❌ can't — `--kv-bits 4` crashes them past 5 k tokens, so they run **fp16** (window caps most layers, growth is gentle) |
| 📈 **Linear + full** | Qwen3.5 9B / 27B | standard | ✅ 4-bit quantizes fine and lowers high-context memory (brief spike at the 5 k threshold) |

---

## 🚀 Quick start

```bash
uv sync                          # install deps into .venv
uv run wmx-suite system          # show the machine's wall, swap, baseline
```

The default safety cushion is 2 GB below the machine's wired-memory wall. Set
`WMX_SUITE_MARGIN_GB` to change it globally for `system`, `health`, `characterize`,
`calibrate`, and `run`; an explicit `--margin` on a command takes precedence:

```bash
export WMX_SUITE_MARGIN_GB=3
uv run wmx-suite health
uv run wmx-suite run --margin 2.5 --dry-run --model <hf_id> --prompt "..."
```

The margin must be finite and non-negative. Lower margins reduce the safety cushion;
use them only when you understand the hard-lock risk.

| Command | What it does |
|---|---|
| `uv run wmx-suite system` | Show the machine's wall, swap, baseline |
| `uv run wmx-suite health` | Live snapshot: current pressure + per-model ✓/✗ go-no-go |
| `uv run wmx-suite characterize <hf_id>` | Safe probe → fitted context ceiling (`--speed quick` is ~3× faster, still conservative; `standard` is the default, `full` is finer) |
| `uv run wmx-suite calibrate` | Measure this machine's cold-start memory overhead so pre-flight estimates are accurate on your Apple Silicon SKU (run once per machine; `characterize` still adapts per model) |
| `uv run wmx-suite list` | Ceilings for everything characterized; warns about stale fits |
| `uv run wmx-suite run --model <hf_id> …` | Safely launch `mlx_lm.generate` |

`characterize` refuses to launch any probe whose pre-flight base estimate already
exceeds the safe threshold — this is how oversized models (like the 27B) are handled:
**predicted, never run into the wall.**

### 🛡️ `run` — the safe launcher

`run` plans a launch and then execs `mlx_lm.generate`. It:

- picks `--kv-bits` by **cache type** — `4` for standard caches, **omitted** for
  RotatingKVCache models (Gemma, GPT-OSS) which can't quantize and would otherwise crash;
- samples the **live settled baseline** and caps `--max-kv-size` at the context where
  `live_base + model_base + slope·c` hits the safe threshold, using the model's
  **measured** curve from `suite.db` (or a conservative estimate, with a warning, if
  uncharacterized);
- **refuses** to launch if the model would breach the wall just to load (e.g. the 27B).
- tokenizes ordinary prompts before launch, warns above 80% of the effective context
  cap, and refuses prompts above it;
- refuses models such as Qwen3.5 whose custom MLX cache does not currently enforce
  `--max-kv-size`, unless `--force` explicitly accepts the unbounded runtime cache.

```bash
# launch safely
uv run wmx-suite run --model mlx-community/Qwen3.5-9B-OptiQ-4bit --prompt "..." --max-tokens 200

# inspect the plan only — no launch
uv run wmx-suite run --dry-run --model <hf_id> --prompt "..."
```

> `--force` overrides a refusal at your own risk; `--dry-run` prints the plan without launching.
> Stdin prompts and prompt-cache files require `--force` because their complete effective
> prompt cannot be verified by the tokenizer preflight.

Every successful run records its prompt/generation **tokens-per-second** to the database
(output still streams live — it runs under a PTY so the experience is unchanged). `list`
then shows the median gen speed per model. Pass `--no-log` for a bare passthrough.

If artifacts in a model's cached Hugging Face snapshots are newer than its latest
characterization, `list` and `run` warn that the fit may be stale. Unused blobs,
mutable refs, and negative-lookup metadata are ignored. The suite does not
automatically re-characterize; review the cache change and run `characterize` again
before relying on the old ceiling.

### 🎛️ Benchmarks (power users)

Beyond context ceilings, the suite is a memory/perf benchmark lab for two model families.
Each command runs under the same RULE #1 safety gating and records to `suite.db`:

| Command | What it measures |
|---|---|
| `uv run wmx-suite benchmark-kokoro` | Kokoro TTS throughput (RTF / chars-per-sec) vs length |
| `uv run wmx-suite benchmark-kokoro-ttfa` | streaming time-to-first-audio latency |
| `uv run wmx-suite benchmark-kokoro-batch` | batch concurrency vs throughput |
| `uv run wmx-suite benchmark-kokoro-voice` | voice-switching latency |
| `uv run wmx-suite benchmark-kokoro-cache` | voice-cache memory overhead |
| `uv run wmx-suite benchmark-kokoro-baseline` | static active-synthesis RAM floor |
| `uv run wmx-suite benchmark-embeddings` | encoder embeddings memory surface (batch × seq_len) |

Run `uv run wmx-suite benchmark-<name> --help` for options.

---

## 🗂️ Layout

```
wmx_suite/
  config.py            # validated runtime defaults (e.g. WMX_SUITE_MARGIN_GB)
  system.py            # device wall, swap, current wired memory
  models.py            # HF-cache config reader + memory-class classifier
  profiles.py          # per-machine cold-start constants (calibration)
  probe.py             # safe characterize/calibrate: ramp + linear fit + ceiling solve
  probe_worker.py      # ONE isolated (model, context) measurement -> JSON
  launcher.py          # safe `run` planning + exec of mlx_lm.generate
  db.py                # SQLite store (context fits, calibration, benchmarks)
  ui.py                # shared console rendering schema
  views/               # per-command output rendering
  cli.py               # core command entry point
  cli_benchmarks.py    # benchmark subcommands (Kokoro TTS + embeddings)
  embeddings_probe.py  # embeddings memory-surface benchmark
  kokoro_safety.py     # RULE #1 safety gating for the Kokoro workers
  probe_worker_kokoro_*.py / probe_worker_embeddings.py   # isolated benchmark workers
data/suite.db          # results (gitignored)
```

The SQLite store holds three families of tables: **context-ceiling** measurement
(`models`, `probe_runs`, `measurements`, `fits`, `generation_log`), **per-machine
calibration** (`system_profiles`, `embedding_profiles`), and **benchmark** results
(`kokoro_*`, `embeddings_*`).

---

## 📊 Status

**v0 scaffold.** Validated methodology: predicted Gemma's ceiling to within **0.5%** from
safe probes. Calibration of the pre-flight base estimate refines as more models are run.

**Headless engine.** wmx-suite is CLI- and JSON-only — no UI. It is the Apple-Silicon
measurement engine behind [Project ARA](https://github.com/willsarg/project-ara), which
wraps it through a thin adapter: the engine *measures and returns*, the caller *persists*.
Visualization — browsing fits, regression curves, side-by-side ceilings, and the benchmark
dashboards — lives in ARA, not here.


## 🤝 Contributing

Contributions are welcome — especially **memory-benchmark results from other Apple
Silicon SKUs**, which is how the suite becomes trustworthy beyond the reference M4 Pro.
See [CONTRIBUTING.md](./CONTRIBUTING.md). The prime directive applies to every change:
*never ship something that can crash a machine.*

## ☕ Support

If wmx-suite saved you a kernel panic (or you just like the idea), you can buy me a coffee:

<div align="center">

<a href="https://www.buymeacoffee.com/willsarg" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" width="217" height="60"></a>

</div>

## ⚖️ Trademarks

MLX, Apple Silicon, Metal, Mac, and macOS are trademarks of Apple Inc. This project is
an independent, community tool — it is **not affiliated with, endorsed by, or sponsored
by Apple Inc.** References to these names are descriptive only, to indicate the
technologies the suite works with. All other trademarks are the property of their
respective owners.
