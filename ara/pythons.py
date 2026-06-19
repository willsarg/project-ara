"""Find every Python interpreter on the system and what AI libraries each one has.

Python provenance is a notorious source of confusion — macOS ships one, Homebrew
manages another, python.org installs a third, then pyenv/conda/uv/asdf each add more,
most of them NOT on PATH unless activated. This module surfaces them all honestly:
where each lives, its version, who installed it, which is your default, and — the part
that matters for AI work — which libraries it actually has.

Read-only. Discovery is cheap (filesystem only); the per-interpreter library probe runs
one short subprocess each, in parallel, and is opt-in via ``probe=True``.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# AI libraries we report per interpreter (presence + version). Ordered by relevance.
_AI_LIBS = ("torch", "transformers", "tensorflow", "jax", "mlx-lm", "vllm", "onnxruntime")

# A python executable name: python, python3, python3.12 — not python3-config etc.
_PY_NAME = re.compile(r"^python(3(\.\d+)?)?$")

# Render order: freely-usable / user-managed first, then the tool-managed and
# system interpreters you shouldn't install into (uv, Homebrew, macOS) clustered last.
_ORIGIN_ORDER = ["python.org", "pyenv", "conda", "asdf", "venv", "other",
                 "uv", "Homebrew", "macOS system"]


def _ver_desc(v: str | None) -> tuple[int, ...]:
    """Sort key putting newer versions first (3.14 before 3.9)."""
    try:
        return tuple(-int(x) for x in (v or "").split(".") if x.isdigit())
    except Exception:
        return ()


@dataclass(frozen=True)
class Interpreter:
    path: str                     # user-facing path (what you'd type / the symlink)
    real: str                     # resolved real executable
    origin: str                   # macOS system | Homebrew | python.org | pyenv | conda | uv | asdf | venv | other
    version: str | None = None    # "3.12.4" (None until probed)
    is_default: bool = False      # your shell's default python3 (ARA's venv excluded)
    externally_managed: bool = False  # PEP 668 marker — pip installs are blocked here
    ai_libs: dict[str, str | None] = field(default_factory=dict)

    @property
    def ai_present(self) -> dict[str, str | None]:
        return {k: v for k, v in self.ai_libs.items() if v is not None}

    @property
    def caution(self) -> str | None:
        """A heads-up for interpreters you shouldn't install into or upgrade directly.

        The macOS system python is Apple-managed by definition (don't touch it), even
        though it predates PEP 668's ``EXTERNALLY-MANAGED`` marker. Everything else warns
        only when that marker is actually present — detected, not assumed.
        """
        return caution_for(self.origin, self.externally_managed)


# All share one rule — "use a venv, not here" — with a truthful per-manager tail.
_CAUTION = {
    "macOS system": "managed by Apple — use a venv, never here; don't upgrade it",
    "Homebrew": "managed by Homebrew — use a venv or pipx, not here; upgrade via brew",
    "uv": "managed by uv — packages go in a venv (uv add), not here; uv may replace it",
}


def caution_for(origin: str, externally_managed: bool) -> str | None:
    """Shared caution rule for interpreters you shouldn't install into directly.
    macOS system is Apple-managed by definition; the rest warn on the PEP 668 marker.
    """
    if origin == "macOS system":
        return _CAUTION["macOS system"]
    if externally_managed:
        return _CAUTION.get(origin, "externally managed — use a venv, not here")
    return None


def _run(cmd: list[str], timeout: float = 8) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return None


def _known_patterns() -> list[str]:
    home = str(Path.home())
    return [
        "/usr/bin/python3",
        "/Library/Developer/CommandLineTools/usr/bin/python3",
        "/usr/local/bin/python3*",
        "/opt/homebrew/bin/python3*",
        "/Library/Frameworks/Python.framework/Versions/*/bin/python3",
        f"{home}/.pyenv/versions/*/bin/python3",
        f"{home}/.pyenv/shims/python3",
        f"{home}/miniconda3/bin/python3", f"{home}/anaconda3/bin/python3",
        f"{home}/miniforge3/bin/python3", f"{home}/mambaforge/bin/python3",
        f"{home}/miniconda3/envs/*/bin/python3", f"{home}/anaconda3/envs/*/bin/python3",
        f"{home}/miniforge3/envs/*/bin/python3",
        f"{home}/.local/share/uv/python/*/bin/python3*",
        f"{home}/.asdf/installs/python/*/bin/python3",
    ]


def _is_venv(real: str) -> bool:
    # a venv interpreter sits in <env>/bin/python; the env has a pyvenv.cfg
    try:
        return (Path(real).resolve().parent.parent / "pyvenv.cfg").exists()
    except Exception:
        return False


def _origin(real: str, invocations: list[str]) -> str:
    joined = " ".join([real, *invocations]).lower()

    def has(*subs: str) -> bool:
        return any(s in joined for s in subs)

    if has("/.pyenv/"):
        return "pyenv"
    if has("conda", "miniforge", "mambaforge", "anaconda"):
        return "conda"
    if has("/.asdf/"):
        return "asdf"
    if has("/uv/python", "/.local/share/uv"):
        return "uv"
    if has("/library/frameworks/python.framework"):
        return "python.org"
    if has("/opt/homebrew/", "/usr/local/cellar", "/homebrew/"):
        return "Homebrew"
    if real.startswith("/usr/bin/") or has("/commandlinetools/", "/usr/libexec/",
                                           "/system/library/frameworks/python.framework"):
        return "macOS system"
    if _is_venv(real):
        return "venv"
    return "other"


def _user_default_real() -> str | None:
    """Resolved path of the user's shell default python3, with ARA's venv stripped."""
    path = os.environ.get("PATH", "")
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        vbin = os.path.normpath(os.path.join(venv, "bin"))
        path = os.pathsep.join(p for p in path.split(os.pathsep)
                               if os.path.normpath(p) != vbin)
    py = shutil.which("python3", path=path) or shutil.which("python", path=path)
    return os.path.realpath(py) if py else None


def _candidates() -> dict[str, set[str]]:
    """Map each interpreter's real path -> the set of paths that resolve to it."""
    cands: list[str] = []
    path_dirs = [d for d in os.environ.get("PATH", "").split(os.pathsep) if d]
    for d in path_dirs:
        try:
            for name in os.listdir(d):
                if _PY_NAME.match(name):
                    cands.append(os.path.join(d, name))
        except Exception:
            pass
    for pat in _known_patterns():
        cands.extend(glob.glob(pat))

    groups: dict[str, set[str]] = {}
    for c in cands:
        try:
            if not _PY_NAME.match(os.path.basename(c)):
                continue  # drop python3-config, python3.13-config, python3-intel64, …
            if not (os.path.isfile(c) or os.path.islink(c)) or not os.access(c, os.X_OK):
                continue
            groups.setdefault(os.path.realpath(c), set()).add(c)
        except Exception:
            pass
    return groups


def _display_path(invocations: set[str], path_dirs: set[str]) -> str:
    """The most user-meaningful path: prefer one on PATH, then the shortest."""
    return min(invocations, key=lambda p: (os.path.dirname(p) not in path_dirs, len(p)))


def _probe(real: str) -> tuple[str | None, dict[str, str | None], bool]:
    code = (
        "import sys, json, os, sysconfig\n"
        "import importlib.metadata as m\n"
        f"names = {list(_AI_LIBS)!r}\n"
        "libs = {}\n"
        "for n in names:\n"
        "    try: libs[n] = m.version(n)\n"
        "    except Exception: libs[n] = None\n"
        "stdlib = sysconfig.get_paths().get('stdlib', '')\n"
        "em = bool(stdlib) and os.path.exists(os.path.join(stdlib, 'EXTERNALLY-MANAGED'))\n"
        "print(json.dumps({'v': '.'.join(map(str, sys.version_info[:3])), 'libs': libs, 'em': em}))\n"
    )
    out = _run([real, "-c", code])
    blank = {n: None for n in _AI_LIBS}
    if not out:
        return None, blank, False
    try:
        data = json.loads(out.strip().splitlines()[-1])
        return data.get("v"), data.get("libs", blank), bool(data.get("em", False))
    except Exception:
        return None, blank, False


def discover(probe: bool = True) -> list[Interpreter]:
    """All python interpreters on the system, deduped by real path.

    ``probe=True`` runs one short subprocess per interpreter (in parallel) to read the
    version and AI-library set. ``probe=False`` skips that — paths/origin only, instant.
    """
    groups = _candidates()
    path_dirs = {d for d in os.environ.get("PATH", "").split(os.pathsep) if d}
    default_real = _user_default_real()

    versions: dict[str, str | None] = {}
    libs: dict[str, dict[str, str | None]] = {}
    managed: dict[str, bool] = {}
    if probe:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for real, (v, lb, em) in zip(groups, pool.map(_probe, groups)):
                versions[real], libs[real], managed[real] = v, lb, em

    out: list[Interpreter] = []
    for real, invocations in groups.items():
        out.append(Interpreter(
            path=_display_path(invocations, path_dirs),
            real=real,
            origin=_origin(real, list(invocations)),
            version=versions.get(real),
            is_default=(real == default_real),
            externally_managed=managed.get(real, False),
            ai_libs=libs.get(real, {}),
        ))

    out.sort(key=lambda i: (_ORIGIN_ORDER.index(i.origin) if i.origin in _ORIGIN_ORDER else 99,
                            not i.is_default, _ver_desc(i.version), i.path))
    return out


def count() -> int:
    """How many distinct interpreters exist (cheap — no subprocess)."""
    return len(_candidates())
