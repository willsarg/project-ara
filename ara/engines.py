# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The engine catalog and hardware-matched resolution.

ARA's core is engine-free; every hardware engine is installed on demand (`ara install`) into
its **own isolated uv env** under the data dir (``ara/engine_env.py``), never into ARA's venv.
That keeps the core universal, the lock engine-free, and incompatible toolchains (torch-CUDA
vs torch-ROCm, MLX vs llama.cpp) from ever colliding. ARA drives an engine over a subprocess
worker in its env — it never imports one in-process.

Two kinds of engine live here:
  * **nested engines** — the MLX and CUDA heavyweights ship under ARA-owned source trees and install
    from that exact source into isolated environments.
  * **built-in engines** — everything else ships in the ARA repo (only the worker's heavy deps
    install into the env); described by a ``packages`` list and ``builtin: True``.

Read-only resolution (``for_hardware``/``resolve``) does no I/O; install/uninstall delegate to
:mod:`ara.engine_env`.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from ara import engine_env, engine_identity


_SOURCE_VERSION = re.compile(
    r"^v?(\d+)\.(\d+)\.(\d+)-(\d+)-g([0-9a-f]+)(-dirty)?$"
)
_SOURCE_ROOT = Path(__file__).resolve().parent.parent


def _source_checkout_version() -> str | None:
    """Derive the live VCS version when this module is running from a Git checkout."""
    root = _SOURCE_ROOT
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            [
                "git", "-C", str(root), "describe", "--tags", "--long", "--dirty",
                "--match", "v[0-9]*", "--abbrev=9",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    match = _SOURCE_VERSION.fullmatch(result.stdout.strip())
    if match is None:
        return None

    major, minor, patch, distance, revision, dirty = match.groups()
    if int(distance) == 0:
        version = f"{major}.{minor}.{patch}"
    else:
        version = f"{major}.{minor}.{int(patch) + 1}.dev{distance}+g{revision}"
    if dirty:
        version += ".dirty" if "+" in version else "+dirty"
    return version


def _ara_version() -> str:
    """Return the current ARA version for display and engine-environment freshness checks.

    Installed wheels trust their immutable distribution metadata. Editable source checkouts refresh
    that metadata from Git so pulling a newer commit also invalidates stale bundled engine code.
    """
    try:
        installed = metadata.version("project-ara")
    except metadata.PackageNotFoundError:
        return "0+unknown"
    return _source_checkout_version() or installed

# Short, stable handles → how to install each. ``backend`` is both the adapter module name and
# the isolated env name. ``available`` is False for engines whose suite isn't shippable yet.
ENGINES: dict[str, dict] = {
    "mlx": {
        "backend": "apple",
        "package": "ara-engine-mlx",
        "purpose": "Native inference and measurement for Apple Silicon.",
        "hardware": "Apple Silicon Macs.",
        "formats": "Hugging Face Transformers models.",
        "install_summary": "ARA's bundled MLX engine plus MLX and Transformers, isolated from ARA.",
        "caution": "Apple-only; the MLX runtime can be large but is installed only on demand.",
        "available": True,
        "source_dir": "_engine_packages/mlx",
        "env_schema": "ara-engine-mlx:ara_engine_mlx:v2",
        "import_package": "ara_engine_mlx",
        "source_env": "ARA_MLX_SOURCE",
        "legacy_source_env": "ARA_WMX_SOURCE",
        # Nested: the native MLX engine source ships under ara/_engine_packages/mlx and installs into
        # the isolated `apple` env from there — no git fetch at install time, so a release is
        # reproducible from the wheel alone. Folded 2026-06-30 from wmx-suite@374c47d (the #107
        # single-BOS + turn-end stop fixes).
        # ARA_WMX_SOURCE still overrides to a local checkout (editable) for engine dev. MLX +
        # transformers stay engine-env-only — never ARA dependencies.
        "python": "3.12",          # ara-engine-mlx requires >=3.12
        "model_kinds": ("transformers",),
        "smoke_model": "mlx-community/SmolLM-135M-Instruct-4bit",
    },
    "cuda": {
        "backend": "cuda",
        "package": "ara-engine-cuda",
        "purpose": "Native full-GPU inference and measurement through PyTorch CUDA.",
        "hardware": "NVIDIA GPUs on Windows; Linux uses the same path but is not yet claimed.",
        "formats": "Hugging Face Transformers models.",
        "install_summary": "ARA's bundled CUDA engine plus PyTorch, CUDA support, and Transformers.",
        "caution": "PyTorch and CUDA wheels are large; uv selects the compatible CUDA wheel.",
        # Converted to the isolated-env worker model (backends/cuda.py drives the native CUDA
        # device + measure_one workers out-of-process; nothing torch-shaped loads in ARA).
        "available": True,
        "source_dir": "_engine_packages/cuda",
        "env_schema": "ara-engine-cuda:ara_engine_cuda:v1",
        "import_package": "ara_engine_cuda",
        "source_env": "ARA_CUDA_SOURCE",
        "legacy_source_env": "ARA_WCX_SOURCE",
        # Nested: the native CUDA engine source ships under ara/_engine_packages/cuda and installs
        # into the isolated `cuda` env from there. Folded 2026-06-30 from wcx-suite@3a43f63.
        "extras": "cuda",                        # pulls torch + transformers (into the env, not ARA)
        # uv auto-detects the GPU and picks the matching CUDA torch wheel (the default
        # PyPI torch on Windows/Linux is CPU-only).
        "pip_args": ["--torch-backend=auto"],
        "python": "3.12",
        "model_kinds": ("transformers",),
        "smoke_model": "HuggingFaceTB/SmolLM-135M-Instruct",
    },
    "cpu": {
        "backend": "cpu",
        "package": "llama.cpp",
        "purpose": "Portable CPU inference for quantized local models.",
        "hardware": "Any supported CPU; no GPU is required.",
        "formats": "GGUF models.",
        "install_summary": "llama-cpp-python, psutil, and Hugging Face Hub in an isolated env.",
        "caution": "Usually slower than GPU lanes; non-Windows hosts may compile llama.cpp locally.",
        "available": True,
        "builtin": True,           # worker ships in ARA; only its deps install into the env
        "packages": ["llama-cpp-python>=0.3", "psutil", "huggingface_hub"],
        "model_kinds": ("gguf",),
        "smoke_model": "bartowski/SmolLM2-135M-Instruct-GGUF",
        # llama-cpp-python ships NO PyPI wheels, so a stock Windows box (no MSVC) can't build
        # it from source. On Windows pull a prebuilt CPU wheel from the project's own index;
        # `--only-binary` (added in _install_targets) makes that deterministic. Scoped to
        # Windows so Linux/macOS/aarch64 (Pi) keep the source build — the abetlen index has no
        # wheel for them, and those platforms ship a toolchain.
        #   max_version: 0.3.19 is the last Windows CPU wheel that actually RUNS via this binding.
        #   The post-0.3.19 /whl/cpu Windows wheels (py3-none-win_amd64) ship a split/runtime-loaded
        #   ggml backend that llama-cpp-python doesn't initialize (upstream #2069, "no backends
        #   loaded"). Verified 2026-06-30 on a Windows CUDA host (Zen-3, cp312): 0.3.21 imports but fails to
        #   load ANY model (SmolLM2 and gemma-4 both "Failed to load model from file"); 0.3.32's
        #   llama.dll fails to load outright (WinError 127). (Historically these were also flagged
        #   as AVX-512 builds that fault on non-AVX-512 CPUs — 0xc000001d.) Native builds elsewhere
        #   are monolithic + host-ISA, so this cap is Windows-only; Linux/macOS get the latest
        #   llama-cpp-python (verified Mac cpu engine ships 0.3.31, runs gemma-4).
        #   RE-CERTIFY before raising this cap — don't guess. `scripts/certify_llama_cpp_cpu.py`
        #   installs a candidate version the same way (--only-binary + this index), loads a tiny
        #   GGUF, and generates a token; exit 0 = safe to bump. Re-run 2026-07-02 on the Windows host
        #   re-confirmed 0.3.32 still fails at llama.dll load (WinError 127 → cap stays).
        "wheel_only": {
            "llama-cpp-python": {
                "index": "https://abetlen.github.io/llama-cpp-python/whl/cpu",
                "max_version": "0.3.19",
            },
        },
        "wheel_platforms": ("Windows",),   # the abetlen CPU index serves only Windows wheels
        "python": "3.12",
    },
    "vulkan": {
        "backend": "vulkan",
        "package": "llama.cpp (Vulkan)",
        "purpose": "GPU-offloaded GGUF inference through llama.cpp's Vulkan backend.",
        "hardware": "Vulkan-capable GPUs on x86_64 Linux or Windows, including AMD APUs.",
        "formats": "GGUF models.",
        "install_summary": "A Vulkan-enabled llama-cpp-python wheel plus lightweight support deps.",
        "caution": "Requires a working Vulkan driver; prebuilt wheels target Linux and Windows.",
        "available": True,
        "builtin": True,           # worker ships in ARA; only its deps install into the env
        "packages": ["llama-cpp-python>=0.3", "psutil", "huggingface_hub"],
        "model_kinds": ("gguf",),
        "smoke_model": "bartowski/SmolLM2-135M-Instruct-GGUF",
        # GPU-offload GGUF via llama.cpp's Vulkan backend (opt-in, --engine vulkan). Prebuilt
        # Vulkan wheels exist for x86_64 Linux + Windows on the project's own index (default PyPI
        # ships no llama-cpp-python wheel at all). We MUST force the prebuilt Vulkan wheel from
        # that index: a plain `llama-cpp-python` install hands back the `cpu` engine's CPU-only
        # wheel from uv's cache (same version, no GGML_VULKAN) — verified. `--only-binary` (added
        # in _builtin_targets) makes it deterministic (no silent source-build fallback); pinned to
        # the newest Vulkan wheel. Kept AFTER `cpu` so engine_for_model's GGUF default stays `cpu`.
        "wheel_only": {
            "llama-cpp-python": {
                "index": "https://abetlen.github.io/llama-cpp-python/whl/vulkan",
                "max_version": "0.3.31",
            },
        },
        "wheel_platforms": ("Linux", "Windows"),
        "python": "3.12",
    },
    "cuda-gguf": {
        "backend": "cuda_gguf",
        "package": "llama.cpp (CUDA)",
        "purpose": "Partial NVIDIA GPU offload for GGUF models through llama.cpp.",
        "hardware": "NVIDIA GPUs on Linux or Windows, with system RAM for overflow layers.",
        "formats": "GGUF models.",
        "install_summary": "A CUDA-enabled llama-cpp-python wheel plus lightweight support deps.",
        "caution": "A two-wall lane: ARA governs discrete VRAM and system RAM together.",
        "available": True,
        "builtin": True,           # worker ships in ARA; only its deps install into the env
        "packages": ["llama-cpp-python>=0.3", "psutil", "huggingface_hub"],
        "model_kinds": ("gguf",),
        "smoke_model": "bartowski/SmolLM2-135M-Instruct-GGUF",
        # CUDA-offload GGUF via llama.cpp's CUDA build (opt-in, --engine cuda-gguf). Enables the
        # two-wall hybrid path: K layers on NVIDIA VRAM, N-K on CPU RAM, auto-fitted each run.
        # NOT a hardware auto-pick — NVIDIA still auto-picks ``cuda`` (the full-GPU transformers
        # engine). Prebuilt CUDA-124 wheels on the project's own index; we MUST force the prebuilt
        # wheel (same reason as vulkan: a plain install gets the CPU wheel from cache). `--only-binary`
        # (added in _builtin_targets) makes that deterministic. Kept AFTER `vulkan` so the GGUF
        # auto-default stays `cpu`. Linux + Windows only (macOS has no NVIDIA discrete GPU target).
        "wheel_only": {
            "llama-cpp-python": {
                "index": "https://abetlen.github.io/llama-cpp-python/whl/cu124",
                "max_version": "0.3.31",
            },
        },
        "wheel_platforms": ("Linux", "Windows"),
        "python": "3.12",
    },
}


@dataclass(frozen=True)
class AutoDecision:
    """One honest automatic-engine decision plus the read-only facts behind it."""
    key: str | None
    reason: str
    system: str
    machine: str
    nvidia_smi: str | None


@dataclass(frozen=True)
class EngineCompatibility:
    """Read-only compatibility verdict for one explicit engine and machine snapshot."""

    status: str
    reason: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {
            "status": self.status,
            "reason": self.reason,
            "detail": self.detail,
        }


def _compatibility(status: str, reason: str, detail: str) -> EngineCompatibility:
    return EngineCompatibility(status, reason, detail)


def _nvidia_facts(machine) -> tuple[bool | None, float | None]:
    """Return NVIDIA presence and observed VRAM from an existing detect snapshot."""
    accel = getattr(machine, "accel", None)
    accel_kind = getattr(accel, "kind", None)
    accel_vram = getattr(accel, "vram_gb", None)
    gpus = list(getattr(machine, "gpus", ()) or ())
    nvidia_gpus = [gpu for gpu in gpus if getattr(gpu, "vendor", None) == "nvidia"]
    vram_values = [
        value
        for value in (
            accel_vram if accel_kind == "nvidia" else None,
            *(getattr(gpu, "vram_gb", None) for gpu in nvidia_gpus),
        )
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
    ]
    if accel_kind == "nvidia" or nvidia_gpus:
        return True, max(vram_values) if vram_values else None
    if accel_kind in {"none", "apple"} and not any(
            getattr(gpu, "vendor", "unknown") == "unknown" for gpu in gpus):
        return False, None
    return None, None


def classify_compatibility(engine: str, machine) -> EngineCompatibility:
    """Classify an engine using only facts already observed by :func:`detect.machine`.

    Compatibility and capacity are intentionally separate. In particular, ``cuda-gguf`` may be
    compatible with the hardware while profile still lacks an engine-free two-wall RAM/VRAM
    capacity estimate.
    """
    system = getattr(machine, "system", None)
    arch = str(getattr(machine, "arch", "") or "").lower()
    x86_64 = arch in {"amd64", "x86_64"}

    if engine == "cpu":
        return _compatibility(
            "compatible", "portable_cpu_fallback",
            "The CPU engine is the portable fallback and does not require a GPU.")

    if engine == "mlx":
        if not system or not arch:
            return _compatibility(
                "unknown", "platform_unknown",
                "ARA could not confirm the operating system and architecture.")
        if system == "Darwin" and arch in {"arm64", "aarch64"}:
            return _compatibility(
                "compatible", "apple_silicon_detected",
                "Darwin on Arm identifies an Apple Silicon host.")
        return _compatibility(
            "incompatible", "requires_apple_silicon",
            "MLX requires an Apple Silicon Mac.")

    if engine in {"cuda", "cuda-gguf"}:
        label = "CUDA" if engine == "cuda" else "CUDA GGUF partial offload"
        supported_systems = {"Windows"} if engine == "cuda" else {"Linux", "Windows"}
        if not system or not arch:
            return _compatibility(
                "unknown", "platform_unknown",
                "ARA could not confirm the operating system and architecture.")
        if system not in supported_systems:
            detail = (
                "CUDA is supported only on NVIDIA GPUs on Windows."
                if engine == "cuda"
                else "CUDA GGUF partial offload is supported only on x86_64 Linux and Windows.")
            return _compatibility("incompatible", "unsupported_platform", detail)
        if not x86_64:
            return _compatibility(
                "incompatible", "unsupported_architecture",
                f"{label} requires an x86_64 host.")
        nvidia_present, vram_gb = _nvidia_facts(machine)
        if nvidia_present is False:
            return _compatibility(
                "incompatible", "nvidia_gpu_unavailable",
                f"{label} requires an NVIDIA GPU, but none was detected.")
        if nvidia_present is None:
            return _compatibility(
                "unknown", "nvidia_gpu_unknown",
                "ARA could not determine whether a compatible NVIDIA GPU is present.")
        if vram_gb is None:
            return _compatibility(
                "unknown", "nvidia_vram_unknown",
                "An NVIDIA GPU was detected, but its VRAM capacity is unknown.")
        if engine == "cuda":
            return _compatibility(
                "compatible", "nvidia_cuda_detected",
                "A supported Windows NVIDIA GPU and its VRAM were detected.")
        return _compatibility(
            "compatible", "nvidia_partial_offload_detected",
            "A supported NVIDIA GPU and system-RAM overflow path were detected.")

    if engine == "vulkan":
        if not system or not arch:
            return _compatibility(
                "unknown", "platform_unknown",
                "ARA could not confirm the operating system and architecture.")
        if system not in {"Linux", "Windows"}:
            return _compatibility(
                "incompatible", "unsupported_platform",
                "Vulkan is supported only on x86_64 Linux and Windows hosts.")
        if not x86_64:
            return _compatibility(
                "incompatible", "unsupported_architecture",
                "The Vulkan engine requires an x86_64 host.")
        gpus = list(getattr(machine, "gpus", ()) or ())
        if any(getattr(gpu, "usable_backend", None) == "vulkan" for gpu in gpus):
            return _compatibility(
                "compatible", "vulkan_runtime_detected",
                "A usable Vulkan GPU runtime was detected.")
        accel_kind = getattr(getattr(machine, "accel", None), "kind", None)
        if not gpus and accel_kind == "none":
            return _compatibility(
                "incompatible", "vulkan_gpu_unavailable",
                "The Vulkan engine requires a compatible GPU, but none was detected.")
        if gpus and all(
                getattr(gpu, "vendor", "unknown") != "unknown" for gpu in gpus):
            return _compatibility(
                "incompatible", "vulkan_runtime_unavailable",
                "No detected GPU has a usable Vulkan runtime.")
        return _compatibility(
            "unknown", "vulkan_runtime_unknown",
            "ARA could not confirm a usable Vulkan GPU runtime.")

    return _compatibility(
        "unknown", "unknown_engine",
        f"ARA has no compatibility classifier for engine {engine!r}.")


def decide_auto(*, system: str, machine: str, nvidia_smi: str | None) -> AutoDecision:
    """Choose an automatic engine from already-observed host facts."""
    if system == "Darwin" and machine == "arm64":
        return AutoDecision(
            "mlx", "Darwin arm64 identifies Apple Silicon, so ARA selects MLX.",
            system, machine, nvidia_smi)
    if nvidia_smi:
        return AutoDecision(
            "cuda", "nvidia-smi is available on PATH, so ARA selects CUDA.",
            system, machine, nvidia_smi)
    return AutoDecision(
        None,
        f"{system} {machine} is not Apple Silicon and nvidia-smi is not available on PATH, "
        "so ARA has no automatic match.",
        system, machine, nvidia_smi,
    )


def auto_decision() -> AutoDecision:
    """Collect the cheap, read-only facts used by automatic engine selection."""
    return decide_auto(
        system=platform.system(), machine=platform.machine(),
        nvidia_smi=shutil.which("nvidia-smi"),
    )


def for_hardware() -> str | None:
    """The engine ARA would pick for this machine from light recon, or None.

    Deliberately cheap — no subprocess: Apple Silicon by ``platform``, NVIDIA by a
    bare ``nvidia-smi`` on PATH. This is the resolution behind ``--engine auto``.
    """
    return auto_decision().key


def for_backend(backend: str) -> str | None:
    """The engine key whose backend matches *backend* (e.g. 'cuda' → 'cuda'), or None.
    One place maps hardware backends to engines, shared by detect and the registry."""
    return next((k for k, e in ENGINES.items() if e["backend"] == backend), None)


def resolve(value: str) -> str | None:
    """Map an ``--engine`` value to a concrete engine key, or None if it doesn't
    name one. ``auto`` defers to :func:`for_hardware`; explicit keys pass through."""
    if value == "auto":
        return for_hardware()
    canonical = engine_identity.canonical_engine(value)
    return canonical if canonical in ENGINES else None


def is_installed(key: str) -> bool:
    """Is engine *key*'s isolated env ready? Cheap and engine-free.

    An env is ready when its Python exists, its ARA release stamp is current, and, if the catalog
    declares an engine-package schema, its schema stamp matches. Unknown keys, older bundled
    engine code, and stale package layouts are not installed-ready.
    """
    engine = ENGINES.get(engine_identity.canonical_engine(key))
    if engine is None or not engine_env.exists(engine["backend"]):
        return False
    if engine_env.stamped_version(engine["backend"]) != _ara_version():
        return False
    schema = engine.get("env_schema")
    return schema is None or engine_env.stamped_schema(engine["backend"]) == schema


def _bundled_source(key: str) -> Path:
    """The catalog-declared nested package source for engine *key*.

    This directory holds the engine's ``pyproject.toml`` and is handed to ``uv pip install``.
    It ships inside ARA's wheel, so the default install needs no engine-source network fetch.
    """
    engine = ENGINES[key]
    return Path(__file__).resolve().parent / engine["source_dir"]


def source_for(key: str) -> str:
    """The install source for an external engine *key*: a dev override, else its nested source.

      * The canonical source variable (for example ``ARA_MLX_SOURCE``) selects a local checkout for
        engine development; a declared legacy variable remains a temporary compatibility fallback.
      * Otherwise ARA installs the exact catalog source shipped in its wheel, reproducibly and
        offline (no git fetch).
    """
    engine = ENGINES[key]
    override = os.environ.get(engine["source_env"])
    if override:
        return override
    legacy_override = os.environ.get(engine["legacy_source_env"])
    if legacy_override:
        print(
            f"ara: {engine['legacy_source_env']} is deprecated; use {engine['source_env']}",
            file=sys.stderr,
        )
        return legacy_override
    return str(_bundled_source(key))


@dataclass(frozen=True)
class InstallResult:
    """Outcome of an install/uninstall attempt — also the shape behind ``--json``."""
    key: str
    status: str       # installed | refreshed | already | coming_soon | unknown | failed | removed | absent
    detail: str = ""


@dataclass(frozen=True)
class InstallPlan:
    """Side-effect-free description consumed by both help and the mutating installer."""
    key: str
    backend: str
    python: str | None
    targets: tuple[str, ...]
    version: str
    schema: str | None
    expected_import: str | None
    source_override: str | None
    platform: str


def _cap_to_wheel_max(req: str, wheels: dict) -> str:
    """Append a wheel-compatible version ceiling to requirement *req* if it's a ``wheel_only``
    package with a ``max_version`` (the newest prebuilt wheel on the engine's index).
    Leaves the requirement's own floor intact (``llama-cpp-python>=0.3`` → ``…>=0.3,<=0.3.19``).
    """
    name = re.match(r"[A-Za-z0-9._-]+", req).group(0)
    spec = wheels.get(name)
    if spec and spec.get("max_version"):
        return f"{req},<={spec['max_version']}"
    return req


def _builtin_targets(engine: dict) -> list[str]:
    """uv-pip args for a builtin engine: its package list, with prebuilt-wheel handling.

    On the platforms an engine lists in ``wheel_platforms``, every ``wheel_only`` package is
    forced to a prebuilt wheel from its index (``--only-binary`` makes that deterministic — no
    silent source-build fallback) and capped at its ``max_version``. This covers two cases: the
    ``cpu`` engine on Windows (no MSVC → can't source-build), and the ``vulkan`` engine on
    x86_64 Linux/Windows (a plain install would resolve to the CPU-only wheel from cache). On any
    other platform the source build picks the host's own ISA, so the list passes through untouched.
    """
    wheels = engine.get("wheel_only") or {}
    if not wheels or platform.system() not in engine.get("wheel_platforms", ()):
        return list(engine["packages"])
    flags: list[str] = []
    for pkg, spec in wheels.items():
        flags += ["--only-binary", pkg, "--extra-index-url", spec["index"]]
    return [*flags, *(_cap_to_wheel_max(req, wheels) for req in engine["packages"])]


def _install_targets(key: str) -> list[str]:
    """The trailing ``uv pip install`` args (pip flags + targets) for engine *key*'s env.

    Built-in engines install a plain package list (with per-platform prebuilt-wheel handling — see
    :func:`_builtin_targets`). Native engine packages install from a local path, with any ``extras``
    group appended (e.g. ``[cuda]``): the bundled source installed plain (read-only inside ARA's
    wheel), or a dev-override local checkout installed editable (``-e``). Any ``pip_args``
    (e.g. ``--torch-backend=auto`` to fetch the right CUDA torch wheel) come first.
    """
    engine = ENGINES[key]
    if engine.get("builtin"):
        return _builtin_targets(engine)
    pip_args = list(engine.get("pip_args", []))
    extras = engine.get("extras")
    suffix = f"[{extras}]" if extras else ""
    # A dev override (ARA_<KEY>_SOURCE) installs editable so engine edits are live; the bundled
    # default installs plain — its source is read-only inside ARA's wheel.
    target = f"{source_for(key)}{suffix}"
    if os.environ.get(engine["source_env"]) or os.environ.get(engine["legacy_source_env"]):
        return [*pip_args, "-e", target]
    return [*pip_args, target]


def install_plan(key: str, *, version: str | None = None) -> InstallPlan:
    """Build the exact current-host install plan without creating or changing an env."""
    engine = ENGINES[key]
    source_override = None
    if source_env := engine.get("source_env"):
        if value := os.environ.get(source_env):
            source_override = f"{source_env}={value}"
        else:
            legacy_env = engine["legacy_source_env"]
            if value := os.environ.get(legacy_env):
                source_override = f"{legacy_env}={value}"
    return InstallPlan(
        key=key,
        backend=engine["backend"],
        python=engine.get("python"),
        targets=tuple(_install_targets(key)),
        version=_ara_version() if version is None else version,
        schema=engine.get("env_schema"),
        expected_import=engine.get("import_package"),
        source_override=source_override,
        platform=f"{platform.system()} {platform.machine()}",
    )


def install(key: str, *, refresh: bool = False) -> InstallResult:
    """Install engine *key* into its own isolated uv env. Idempotent and honest:
    never creates an env for an unknown or not-yet-available engine.

    Version-aware: an installed env is *stale* when its stamped ARA version differs from the
    current one (a newer ARA wheel ships newer nested engine source) or when it carries
    no stamp at all (built by a pre-stamp ARA). A stale env is torn down and reinstalled so the
    shipped engine code actually reaches the box — reported as ``refreshed``. ``refresh=True`` forces
    that reinstall even when the stamp already matches. Every fresh/refresh install stamps the env
    with the current version."""
    if key not in ENGINES:
        return InstallResult(key, "unknown")
    engine = ENGINES[key]
    if not engine["available"]:
        return InstallResult(key, "coming_soon", f"{engine['package']} isn't available yet")
    current = _ara_version()
    present = engine_env.exists(engine["backend"])
    stale = present and (
        refresh
        or engine_env.stamped_version(engine["backend"]) != current
        or (engine.get("env_schema") is not None
            and engine_env.stamped_schema(engine["backend"]) != engine["env_schema"])
    )
    if present and not stale:
        return InstallResult(key, "already")
    if stale:                       # wipe the old env so the reinstall isn't itself a noop
        engine_env.remove(engine["backend"])
    plan = install_plan(key, version=current)
    create_kwargs = {
        "python": plan.python,
        "version": plan.version,
        "schema": plan.schema,
    }
    if plan.expected_import is not None:
        create_kwargs["expected_import"] = plan.expected_import
    try:
        engine_env.create(plan.backend, list(plan.targets), **create_kwargs)
    except engine_env.EngineEnvError as e:
        return InstallResult(key, "failed", str(e))
    return InstallResult(key, "refreshed" if stale else "installed")


def uninstall(key: str) -> InstallResult:
    """Remove engine *key*'s isolated env. No-op when it isn't an engine or isn't present.
    The shared uv cache and other engines' envs are untouched."""
    if key not in ENGINES:
        return InstallResult(key, "unknown")
    try:
        removed = engine_env.remove(ENGINES[key]["backend"])
    except OSError as exc:
        return InstallResult(key, "failed", str(exc))
    return InstallResult(key, "removed" if removed else "absent")


def _is_gguf(ref: str) -> bool:
    """Confident GGUF signal: a .gguf file path, or a repo:file.gguf reference."""
    tail = ref.split(":", 1)[1] if ":" in ref else ref
    return tail.endswith(".gguf")


def engine_for_model(ref: str) -> str | None:
    """The engine key best suited to *ref*, or None when it can't be told cheaply.

    Only a confident GGUF signal classifies; a bare repo id (even one named '...-GGUF')
    returns None so the caller falls back to the engine's own preflight error."""
    if _is_gguf(ref):
        return next((k for k, e in ENGINES.items() if "gguf" in e.get("model_kinds", ())), None)
    return None
