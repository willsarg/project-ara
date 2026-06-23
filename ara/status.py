# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Live recon: find AI/ML processes running right now. Read-only — observes, never acts.

The dynamic counterpart to ``detect``: detect reports what's *installed*, status reports
what's *running and consuming memory this moment*. Pure recon — no engine import, no
backend coupling, works on any box. Matching is "balanced": known inference engines/apps
plus obvious ML python, kept specific to avoid flagging unrelated processes.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass

import psutil

from ara import apps as _apps

GB = 1024 ** 3  # binary GiB, matching `detect`


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
# (label, name_substrings, cmdline_substrings). First rule that matches wins, so
# specific engines are listed before the generic python catch. Ports aren't encoded
# here — status only ever reports a port it actually observed listening, never a guess.
_RULES: list[tuple[str, list[str], list[str]]] = [
    ("Ollama", ["ollama"], ["ollama"]),
    ("LM Studio", ["lm studio", "lmstudio", "lms"], ["lmstudio", "lm-studio", ".lmstudio"]),
    ("llama.cpp", ["llama-server", "llama-cli", "llama-bench"],
     ["llama-server", "llama-cli", "llama.cpp"]),
    ("vLLM", [], ["vllm"]),
    ("MLX (mlx_lm)", [], ["mlx_lm", "mlx-lm"]),
    ("TGI", ["text-generation"], ["text-generation-inference", "text_generation_server"]),
    ("SGLang", [], ["sglang"]),
    ("ComfyUI", [], ["comfyui"]),
    ("Stable Diffusion", [],
     ["stable-diffusion", "stable_diffusion", "sdwebui", "automatic1111", "/sd.webui"]),
    ("Jupyter", ["jupyter"],
     ["jupyter-lab", "jupyterlab", "jupyter-notebook", "ipykernel_launcher"]),
]

# Generic ML python: a python process whose cmdline names a major ML library. Specific
# enough to skip ordinary scripts; catches custom `python train.py`-style workloads only
# when the library is on the command line.
_ML_PYTHON_TOKENS = (
    "torch", "transformers", "diffusers", "tensorflow", "keras", "onnxruntime",
    "accelerate", "sentence-transformers", "llama_cpp", "ctransformers", "exllama",
)


def _classify(name: str, cmd: str) -> str | None:
    """Return a friendly label if this looks like an AI/ML process, else None."""
    nl, cl = name.lower(), cmd.lower()
    for label, names, cmds in _RULES:
        if any(s in nl for s in names) or any(s in cl for s in cmds):
            return label
    if ("python" in nl or "python" in cl) and any(tok in cl for tok in _ML_PYTHON_TOKENS):
        return "Python ML"
    return None


# --------------------------------------------------------------------------- #
# AI client apps (a different lane from workloads)
# --------------------------------------------------------------------------- #
# These talk to a remote API — they aren't local ML workloads, so `status` reports them
# in their own section. The set of "client" apps is the apps catalog's assistant + coding
# categories (local runners stay in the workload lane above).
_APP_CATEGORIES = ("assistant", "coding")

# GUI clients matched by their .app bundle path (reused from the shared apps catalog, so
# there's one source of truth for "known AI apps"). Each helper process of an Electron app
# also has the parent bundle in its path, so grouping by label collapses them to one entry.
_APP_BUNDLES: list[tuple[str, str]] = [
    (f"/{bundle.lower()}.app/", label)
    for label, category, bundles, _tokens in _apps.CATALOG if category in _APP_CATEGORIES
    for bundle in bundles
]

# Terminal clients matched by exact process basename — a CLI's process name differs from
# its package token (e.g. `claude` vs the `claude-code` cask), so it's listed explicitly.
_CLI_CLIENTS: dict[str, str] = {
    "claude": "Claude Code",
    "codex": "Codex CLI",
}


def _classify_app(name: str, cmd: str) -> str | None:
    """Return a client-app label if this process belongs to a known AI client app, else None.

    GUI bundles are checked first so a `.app` whose exec basename collides with a CLI name
    (Claude.app's "Claude" vs the `claude` CLI) is attributed to the app, not the CLI.
    """
    cl = cmd.lower()
    for marker, label in _APP_BUNDLES:
        if marker in cl:
            return label
    if ".app/" not in cl and name.lower() in _CLI_CLIENTS:
        return _CLI_CLIENTS[name.lower()]
    return None


def _short(s: str) -> str:
    """Trim a model path/id to something human: basename for paths, org/name for HF ids."""
    s = s.strip().rstrip("/")
    if s.startswith(("/", "~", ".")):
        return s.rsplit("/", 1)[-1]
    return "/".join(s.split("/")[-2:]) if "/" in s else s


def _detail(cmd_tokens: list[str]) -> str | None:
    """Best-effort model hint from a cmdline: --model, -m, or a weights-file path."""
    for i, t in enumerate(cmd_tokens):
        if t in ("--model", "--model-path", "-m") and i + 1 < len(cmd_tokens):
            return _short(cmd_tokens[i + 1])
        if t.startswith("--model="):
            return _short(t.split("=", 1)[1])
    for t in cmd_tokens:
        if t.endswith((".gguf", ".safetensors")):
            return _short(t)
    return None


# --------------------------------------------------------------------------- #
# gpu (NVIDIA per-process; Apple has no clean per-process attribution)
# --------------------------------------------------------------------------- #
def _nvidia_gpu_by_pid() -> dict[int, float]:
    """Map pid -> GPU memory (MiB) from nvidia-smi. Empty on non-NVIDIA / failure."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout
    except Exception:
        return {}
    gpu: dict[int, float] = {}
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                gpu[int(parts[0])] = float(parts[1])
            except ValueError:
                pass
    return gpu


def _listen_port(proc: psutil.Process) -> int | None:
    """First TCP port this process is LISTENing on, best-effort (may be denied)."""
    getter = getattr(proc, "net_connections", None) or getattr(proc, "connections", None)
    if getter is None:
        return None
    try:
        for conn in getter(kind="inet"):
            if conn.status == psutil.CONN_LISTEN and conn.laddr:
                return conn.laddr.port
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Proc:
    pid: int
    label: str
    detail: str | None
    rss_gb: float
    uptime_s: float
    gpu_mb: float | None
    port: int | None


def scan() -> list[Proc]:
    """All running AI/ML processes, largest memory first. Read-only."""
    gpu = _nvidia_gpu_by_pid()
    now = time.time()
    skip = {os.getpid(), os.getppid()}  # don't report ARA itself

    found: list[Proc] = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time", "memory_info"]):
        try:
            info = p.info
            pid = info.get("pid")
            if pid in skip:
                continue
            name = info.get("name") or ""
            tokens = info.get("cmdline") or []
            label = _classify(name, " ".join(tokens))
            if not label:
                continue
            mem = info.get("memory_info")
            rss_gb = (mem.rss / GB) if mem else 0.0
            created = info.get("create_time") or now
            found.append(Proc(
                pid=pid, label=label, detail=_detail(tokens),
                rss_gb=round(rss_gb, 2), uptime_s=round(now - created, 1),
                gpu_mb=gpu.get(pid), port=_listen_port(p),
            ))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    found.sort(key=lambda x: x.rss_gb, reverse=True)
    return found


@dataclass(frozen=True)
class AppProc:
    label: str
    n_procs: int
    rss_gb: float
    uptime_s: float


def scan_apps() -> list[AppProc]:
    """Running AI *client* apps (assistant/coding), one entry per app, largest memory first.

    A counterpart to ``scan``: those are local ML workloads, these are remote-API clients
    (Claude Desktop, ChatGPT, Cursor, Claude Code …). Multi-process apps (Electron helpers)
    collapse into a single entry. RSS is ordinary RAM — these consume no local ML resources.
    Read-only.
    """
    now = time.time()
    skip = {os.getpid(), os.getppid()}  # don't report ARA itself

    agg: dict[str, dict] = {}
    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time", "memory_info"]):
        try:
            info = p.info
            pid = info.get("pid")
            if pid in skip:
                continue
            label = _classify_app(info.get("name") or "", " ".join(info.get("cmdline") or []))
            if not label:
                continue
            mem = info.get("memory_info")
            rss_gb = (mem.rss / GB) if mem else 0.0
            created = info.get("create_time") or now
            a = agg.setdefault(label, {"n": 0, "rss": 0.0, "oldest": created})
            a["n"] += 1
            a["rss"] += rss_gb
            a["oldest"] = min(a["oldest"], created)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    out = [AppProc(label=k, n_procs=v["n"], rss_gb=round(v["rss"], 2),
                   uptime_s=round(now - v["oldest"], 1)) for k, v in agg.items()]
    out.sort(key=lambda x: x.rss_gb, reverse=True)
    return out
