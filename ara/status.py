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
