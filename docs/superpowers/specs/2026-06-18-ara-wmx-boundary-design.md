# ARA ⇄ wmx-suite Boundary Design

**Date:** 2026-06-18
**Status:** Approved design (no code yet)
**Scope:** How Project ARA depends on `wmx-suite` while staying independent of it, and how a fresh clone installs only the libraries the current machine actually needs.

---

## Problem

ARA ("AI Runs Anywhere") is meant to be backend-agnostic — Apple/MLX today, CUDA later, others eventually. Two things must be true at once:

1. **Independence** — `wmx-suite` stays a standalone Apple-Silicon engine that knows nothing about ARA. Swapping engines must not ripple through ARA's core.
2. **Leanness** — a user on an Nvidia box must never download MLX (or any Apple-only library). "M-series drivers don't belong on an Nvidia chip."

This is one problem felt as two boundaries: a **code boundary** and a **dependency boundary**.

## Non-goals

- Not building ARA v1 features here (the `detect → recommend → run` pipeline). This doc only fixes the *structure* those features will live in.
- No PyPI publishing. Distribution is `git clone` + `uv sync`.
- No plugin framework, entry points, or separate distributable packages. Too much ceremony for now.

---

## The two boundaries

### 1. Code boundary — keeps ARA and wmx-suite independent

- `wmx-suite` stays its own repo. It carries **no ARA branding and no ARA imports**.
- ARA **core is pure Python** — hardware detection, the curated model catalog, fit/recommend logic, and the CLI front door. **Core imports zero ML libraries.**
- Each backend is an **adapter** that lives inside ARA. The Apple adapter is the *only* file in the system that imports `wmx_suite`.
- A small **registry** maps detected hardware → backend module and **lazy-imports only that one module at runtime**.

The discipline that makes leanness hold even if a wrong library is somehow present: **core imports no backend at module-load time.** Markers stop the wrong dependency from installing; lazy-loading stops the wrong code from importing. Belt and suspenders.

### 2. Dependency boundary — keeps installs lean

- `wmx-suite` is declared with an **environment marker** so it auto-installs only on Apple Silicon. CUDA deps get the inverse marker.
- `uv sync` evaluates markers against the current machine and installs only the matching slice.
- Extras (`--extra apple` / `--extra cuda`) sit on top as a manual override for cross-target development.

**marker decides → `uv sync` enforces → `uv.lock` records.**

---

## Concrete layout

```
project-ara/
├── pyproject.toml
└── ara/
    ├── __init__.py
    ├── detect.py          # what hardware am I on?
    ├── registry.py        # pick + load the right backend (the only "clever" line)
    ├── cli.py             # the front door
    └── backends/
        ├── __init__.py
        └── apple.py       # wraps wmx-suite (CUDA arrives later as cuda.py)
```

### `detect.py` — stdlib only

```python
import platform

def detect() -> str:
    """Return a backend name for this machine."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "apple"
    # later: NVIDIA check -> "cuda"
    return "unsupported"
```

### `registry.py` — lazy load; the whole mechanism

```python
import importlib
from ara.detect import detect

def get_backend():
    name = detect()
    # ONLY this module gets imported — apple.py (and MLX) never loads on Nvidia
    return importlib.import_module(f"ara.backends.{name}")
```

### `backends/apple.py` — the adapter; only file that knows wmx-suite exists

```python
import wmx_suite  # MLX is pulled in here, and only here

def characterize():
    """ARA asks 'how hard can I push this machine?' -> delegate to wmx-suite."""
    return wmx_suite.characterize()      # map onto whatever wmx-suite actually exposes

def run(model, request):
    ...
```

> The adapter mapping (`wmx_suite.characterize()` etc.) is illustrative. The real adapter
> maps ARA's calls onto wmx-suite's *actual* current API — a small, contained piece of work
> done when ARA v1 is built, not now.

The CLI calls `get_backend().characterize()` and never names "apple" or "mlx" anywhere.

---

## Dependency wiring (`pyproject.toml`)

```toml
[project]
dependencies = [
    # core is pure-python; the Apple engine auto-installs ONLY on Apple Silicon
    "wmx-suite ; sys_platform == 'darwin' and platform_machine == 'arm64'",
]

[project.optional-dependencies]
apple = ["wmx-suite"]                 # manual override / cross-target dev
# cuda = ["...", ...]                 # added when the CUDA backend lands

[tool.uv.sources]
# distribution default: clone-and-go for anyone. uv git-clones + builds wmx-suite.
wmx-suite = { git = "https://github.com/willsarg/wmx-suite" }
# solo-dev alternative (edit the sibling checkout live):
# wmx-suite = { path = "../wmx-suite", editable = true }
```

- Not being on PyPI costs nothing: a **git URL is a first-class uv source**. `wmx-suite` is
  pip-installable because it has its own `pyproject.toml`.
- Pin as it stabilizes: `branch` → `tag` → `rev`. Whatever form, `uv.lock` records the exact
  resolved commit, so clone-and-`uv sync` is reproducible.

## Install behavior — same repo, same command, different machines

```
# Apple Silicon Mac:
$ uv sync
   marker TRUE  → git-clones + installs wmx-suite → pulls MLX, kokoro, etc.
   .venv = core + Apple engine

# Linux + Nvidia:
$ uv sync
   marker FALSE → wmx-suite skipped entirely. MLX never downloaded.
   .venv = core only (CUDA backend installs via its own marker)
```

`uv.lock` locks all platforms; `uv sync` materializes only the current machine's subset.

---

## Why this is the right amount of structure

- The only non-obvious line in the whole system is `importlib.import_module(...)`. Everything
  else is plain functions and a normal `pyproject.toml`.
- Adding CUDA later = add `backends/cuda.py` + one marker line. **Nothing already working is touched.**
- It honors the prior architecture decision (separate repos, adapter-in-ARA, protocol boundary,
  wmx-suite stays ARA-ignorant) and the "keep changes simple" principle.

## Open items (deferred, not blocking)

- Exact shape of the backend interface (`characterize` / `fit` / `run` signatures) — defined
  when ARA v1's `detect → recommend → run` pipeline is built, against wmx-suite's real API.
- Whether `detect.py` needs richer hardware facts (unified-memory size, VRAM) — likely yes, but
  that's a v1 concern, not a boundary concern.
- Switch the `[tool.uv.sources]` default from `path` to `git` at the moment someone other than
  Will first needs to clone.
