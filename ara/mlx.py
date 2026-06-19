"""Detect the MLX ecosystem: which MLX libraries are installed (and where), plus the
surrounding readiness picture (Metal GPU, cached mlx-community models, LM Studio's MLX
runtime). MLX is Apple-Silicon only, so this is meaningful only there.

Read-only. Reuses the interpreter discovery from ``pythons`` and probes each (in parallel)
for the MLX package set, organized by what each piece does.
"""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ara import detect, pythons

# (modality label, [pip packages]) — the MLX ecosystem grouped by what it does.
GROUPS: list[tuple[str, list[str]]] = [
    ("core", ["mlx"]),
    ("text / LLM", ["mlx-lm"]),
    ("vision / VLM", ["mlx-vlm"]),
    ("speech / STT", ["mlx-whisper"]),
    ("embeddings", ["mlx-embeddings"]),
    ("audio / TTS", ["mlx-audio"]),
    ("image", ["mflux", "diffusionkit"]),
    ("serving", ["fastmlx"]),
    ("data", ["mlx-data"]),
]
_ALL = tuple(p for _, pkgs in GROUPS for p in pkgs)


@dataclass(frozen=True)
class MlxInterpreter:
    path: str
    origin: str
    version: str | None
    packages: dict[str, str] = field(default_factory=dict)  # present MLX pkg -> version


def _run(cmd: list[str], timeout: float = 8) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return None


def _probe(real: str) -> tuple[str | None, dict[str, str]]:
    """(python version, {present MLX package: version}) for one interpreter."""
    code = (
        "import sys, json\n"
        "import importlib.metadata as m\n"
        f"names = {list(_ALL)!r}\n"
        "out = {}\n"
        "for n in names:\n"
        "    try: out[n] = m.version(n)\n"
        "    except Exception: pass\n"
        "print(json.dumps({'v': '.'.join(map(str, sys.version_info[:3])), 'pkgs': out}))\n"
    )
    raw = _run([real, "-c", code])
    if not raw:
        return None, {}
    try:
        data = json.loads(raw.strip().splitlines()[-1])
        return data.get("v"), data.get("pkgs", {})
    except Exception:
        return None, {}


def scan() -> list[MlxInterpreter]:
    """Interpreters that have at least one MLX package, richest first."""
    ints = pythons.discover(probe=False)
    with ThreadPoolExecutor(max_workers=8) as pool:
        probed = list(pool.map(lambda i: _probe(i.real), ints))
    out = [MlxInterpreter(path=i.path, origin=i.origin, version=ver or i.version, packages=pkgs)
           for i, (ver, pkgs) in zip(ints, probed) if pkgs]
    out.sort(key=lambda m: len(m.packages), reverse=True)
    return out


def mlx_community_model_count() -> int:
    """How many mlx-community models are in the HF cache."""
    hub = detect._hf_hub_dir()
    return len(list(hub.glob("models--mlx-community--*"))) if hub and hub.exists() else 0


def lmstudio_mlx_runtimes() -> list[str]:
    """Versions of LM Studio's bundled MLX runtime, newest first (empty if none)."""
    base = Path.home() / ".lmstudio" / "extensions" / "backends"
    if not base.exists():
        return []
    versions = {d.name.rsplit("-", 1)[-1] for d in base.glob("mlx-llm-*") if d.is_dir()}
    return sorted(versions, key=pythons._ver_desc)
