"""Recon: observe the machine and pick a backend. Read-only — no profiling.

Everything here just *reads* what's already true about the host — memory, GPU,
disk, installed runtimes, downloaded models, environment gates. It never
stresses, benchmarks, or loads an ML engine, so `ara detect` works on any box,
even a bare one with no backend for its hardware yet.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path

import psutil

from ara import apps as _apps, versions as _versions

GB = 1024 ** 3


# --------------------------------------------------------------------------- #
# backend choice
# --------------------------------------------------------------------------- #
def backend_name() -> str:
    """Return the backend module name for this machine.

    ``"unsupported"`` means we have no adapter for this hardware yet (detect
    still reports everything else).
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "apple"
    # later: NVIDIA detection -> "cuda"
    return "unsupported"


# --------------------------------------------------------------------------- #
# low-level reads
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], timeout: float = 3) -> str | None:
    """Run a read-only command, return stdout (or None on any failure)."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout
    except Exception:
        return None


def _sysctl(key: str) -> str | None:
    if platform.system() != "Darwin":
        return None
    out = _run(["sysctl", "-n", key], timeout=2)
    return out.strip() if out else None


def chip_name() -> str:
    """Best-effort human chip name, e.g. 'Apple M4 Pro'."""
    if platform.system() == "Darwin":
        brand = _sysctl("machdep.cpu.brand_string")
        if brand:
            return brand
    return platform.processor() or platform.machine() or "unknown"


def os_version() -> str:
    if platform.system() == "Darwin":
        ver = platform.mac_ver()[0]
        return f"macOS {ver}" if ver else "macOS"
    return platform.system() or "unknown"


def _memory_gb() -> tuple[float | None, float | None]:
    try:
        vm = psutil.virtual_memory()
        return vm.total / GB, vm.available / GB
    except Exception:
        return None, None


def _swap_gb() -> float | None:
    try:
        return psutil.swap_memory().total / GB
    except Exception:
        return None


def _cpu_counts() -> tuple[int | None, int | None]:
    try:
        return psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)
    except Exception:
        return None, None


def _cpu_features() -> list[str]:
    """SIMD features relevant to CPU inference. Read from sysctl / /proc."""
    feats: list[str] = []
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        feats.append("NEON")
        if _sysctl("hw.optional.arm.FEAT_BF16") == "1":
            feats.append("BF16")
        return feats
    # x86
    flags = ""
    try:
        flags = Path("/proc/cpuinfo").read_text().lower()
    except Exception:
        flags = ((_sysctl("machdep.cpu.leaf7_features") or "")
                 + " " + (_sysctl("machdep.cpu.features") or "")).lower()
    for f in ("avx512f", "avx2", "avx", "sse4_2"):
        if f in flags:
            feats.append(f.upper().replace("AVX512F", "AVX-512").replace("SSE4_2", "SSE4.2"))
    return feats


def _disk_free_gb() -> float | None:
    try:
        return shutil.disk_usage(Path.home()).free / GB
    except Exception:
        return None


def _user_python() -> str | None:
    """The python3 a user reaches for in their OWN shell — not ARA's venv.

    Under ``uv run`` the active venv (``VIRTUAL_ENV``) shadows PATH, so a naive
    ``which python3`` returns ARA's interpreter and reports ARA's bundled deps as if
    they were the user's. Strip the venv's bin and re-resolve to find the real one.
    Returns None when the only python available is ARA's own.
    """
    path = os.environ.get("PATH", "")
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        vbin = os.path.normpath(os.path.join(venv, "bin"))
        path = os.pathsep.join(p for p in path.split(os.pathsep)
                               if os.path.normpath(p) != vbin)
    py = shutil.which("python3", path=path) or shutil.which("python", path=path)
    if py and os.path.realpath(py) == os.path.realpath(sys.executable):
        return None  # resolved straight back to ARA's interpreter
    return py


def _python_packages(py: str | None, names: tuple[str, ...]) -> dict[str, str | None]:
    """Versions of *names* installed in interpreter *py* — metadata only, no heavy
    import. Reflects whatever environment *py* belongs to."""
    blank = {n: None for n in names}
    if not py:
        return blank
    code = (
        "import importlib.metadata as m, json\n"
        f"names = {list(names)!r}\n"
        "out = {}\n"
        "for n in names:\n"
        "    try: out[n] = m.version(n)\n"
        "    except Exception: out[n] = None\n"
        "print(json.dumps(out))\n"
    )
    raw = _run([py, "-c", code], timeout=6)
    if not raw:
        return blank
    try:
        return json.loads(raw)
    except Exception:
        return blank


def _ara_pkg_version(name: str) -> str | None:
    """Version of a package in ARA's OWN environment (for engines ARA bundles)."""
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def _python_version(py: str | None = None) -> str | None:
    """Version string of interpreter *py* (defaults to whatever's first on PATH)."""
    py = py or shutil.which("python3") or shutil.which("python")
    if py:
        out = _run([py, "--version"])
        if out:
            return out.strip().replace("Python ", "") or None
    return platform.python_version()


def _power() -> str:
    try:
        bat = psutil.sensors_battery()
    except Exception:
        bat = None
    if bat is None:
        return "AC (no battery)"
    return "AC power" if bat.power_plugged else f"battery {bat.percent:.0f}%"


# --------------------------------------------------------------------------- #
# accelerator
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Accelerator:
    kind: str                  # "apple" | "nvidia" | "none"
    name: str
    vram_gb: float | None      # discrete VRAM; None when unified (Apple)/unknown
    api: str | None            # "Metal" | "CUDA" | None
    count: int = 1             # number of GPUs
    cores: int | None = None   # GPU core count (Apple)
    compute: str | None = None # NVIDIA compute capability, e.g. "7.5"
    cuda_version: str | None = None   # max CUDA the driver supports, e.g. "13.1"
    driver_version: str | None = None # NVIDIA driver version, e.g. "591.86"


def _apple_gpu_cores() -> int | None:
    out = _run(["system_profiler", "SPDisplaysDataType"], timeout=4)
    if not out:
        return None
    for line in out.splitlines():
        if "Total Number of Cores" in line:
            try:
                return int(line.split(":")[1].strip())
            except Exception:
                return None
    return None


def _nvidia_cuda_version(smi: str) -> str | None:
    """Max CUDA runtime the driver supports, from the nvidia-smi header.

    ``--query-gpu`` exposes no CUDA-version field — it appears only in the table
    header as ``CUDA Version: 13.1``. This is distinct from the driver version.
    """
    out = _run([smi])
    if not out:
        return None
    m = re.search(r"CUDA Version:\s*([0-9.]+)", out)
    return m.group(1) if m else None


def accelerator(chip: str) -> Accelerator:
    """Identify the GPU from what the system already reports — no probing."""
    smi = shutil.which("nvidia-smi")
    if smi:
        out = _run([smi, "--query-gpu=name,memory.total,compute_cap,driver_version",
                    "--format=csv,noheader,nounits"])
        if out and out.strip():
            lines = out.strip().splitlines()
            try:
                name, mem_mib, cc, drv = (x.strip() for x in lines[0].split(","))
                return Accelerator(
                    "nvidia", name, round(float(mem_mib) / 1024, 1), "CUDA",
                    count=len(lines), compute=cc or None,
                    cuda_version=_nvidia_cuda_version(smi), driver_version=drv or None,
                )
            except Exception:
                pass
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return Accelerator("apple", f"{chip} GPU", None, "Metal", cores=_apple_gpu_cores())
    return Accelerator("none", "none detected", None, None)


# --------------------------------------------------------------------------- #
# installed runtimes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Runtime:
    name: str
    present: bool
    version: str | None = None
    kind: str = "engine"             # "engine" (launch target) | "framework" (library)
    accels: tuple[str, ...] = ()     # accelerator kinds it's built for; () = cross-platform
    usable: bool | None = None       # resolved against this machine; None = not gated

    @property
    def requires(self) -> str | None:
        """Human accelerator requirement when this runtime can't accelerate here."""
        if self.usable is not False:
            return None
        return "needs " + " / ".join(_ACCEL_LABEL.get(a, a) for a in self.accels)


_ACCEL_LABEL = {"nvidia": "CUDA", "apple": "Apple Silicon"}

# Probed in the USER's python (frameworks/libraries) — engines also fall back to CLIs/ARA.
_FRAMEWORK_PKGS = ("torch", "transformers", "tensorflow")
_ENGINE_PKGS = ("mlx-lm", "vllm")


def runtimes(accel_kind: str = "none", user_py: str | None = None) -> list[Runtime]:
    """Inference engines (launch targets) and ML frameworks (libraries underneath).

    Frameworks are probed in the *user's* python (``user_py``) so they report the user's
    environment, not ARA's bundled deps. Engines also consult PATH CLIs and — for the MLX
    engine ARA ships — ARA's own env, since that's a real capability on this machine.
    """
    pkgs = _python_packages(user_py, _FRAMEWORK_PKGS + _ENGINE_PKGS)
    llama = shutil.which("llama-cli") or shutil.which("llama-server")
    lms = shutil.which("lms") or Path("/Applications/LM Studio.app").exists()

    mlx_ver = pkgs.get("mlx-lm") or _ara_pkg_version("mlx-lm")  # ARA bundles the MLX engine
    mlx_present = mlx_ver is not None or find_spec("mlx_lm") is not None
    # vLLM ships a `vllm` CLI; check PATH too — the user may have it outside any probed env.
    vllm_present = pkgs.get("vllm") is not None or shutil.which("vllm") is not None

    # Versions for the non-python engines, from the reliable sources (brew / .app plist).
    _, lms_ver = _versions.find_app(["LM Studio"])

    # (name, present, version, kind, accels) — accels=() means cross-platform.
    specs: list[tuple[str, bool, str | None, str, tuple[str, ...]]] = [
        ("MLX", mlx_present, mlx_ver, "engine", ("apple",)),
        ("llama.cpp", llama is not None, _versions.brew_version("llama.cpp"), "engine", ()),
        ("Ollama", shutil.which("ollama") is not None or (Path.home() / ".ollama").exists(),
         _versions.brew_version("ollama"), "engine", ()),
        ("LM Studio", bool(lms), lms_ver, "engine", ()),
        ("vLLM", vllm_present, pkgs.get("vllm"), "engine", ("nvidia",)),
        ("PyTorch", pkgs.get("torch") is not None, pkgs.get("torch"), "framework", ()),
        ("transformers", pkgs.get("transformers") is not None, pkgs.get("transformers"),
         "framework", ()),
        ("TensorFlow", pkgs.get("tensorflow") is not None, pkgs.get("tensorflow"),
         "framework", ()),
    ]
    out: list[Runtime] = []
    for name, present, version, kind, accels in specs:
        usable = None if not accels else (accel_kind in accels)
        out.append(Runtime(name, present, version, kind=kind, accels=accels, usable=usable))
    return out


# --------------------------------------------------------------------------- #
# model inventory
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelStore:
    name: str
    present: bool
    count: int = 0
    size_gb: float = 0.0


def _hf_hub_dir() -> Path | None:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _dir_size_gb(d: Path) -> float:
    total = 0
    if d.exists():
        for f in d.iterdir():
            try:
                if f.is_file():
                    total += f.stat().st_size
            except Exception:
                pass
    return total / GB


def _hf_inventory() -> ModelStore:
    hub = _hf_hub_dir()
    if not hub or not hub.exists():
        return ModelStore("HF cache", False)
    models = [p for p in hub.glob("models--*") if p.is_dir()]
    # sum only blobs/ (real files); snapshots/ are symlinks into blobs.
    size = sum(_dir_size_gb(p / "blobs") for p in models)
    return ModelStore("HF cache", True, len(models), round(size, 1))


def _ollama_inventory() -> ModelStore:
    base = Path.home() / ".ollama" / "models"
    present = shutil.which("ollama") is not None or base.exists()
    if not base.exists():
        return ModelStore("Ollama", present)
    manifests = base / "manifests"
    count = sum(1 for f in manifests.rglob("*") if f.is_file()) if manifests.exists() else 0
    size = round(_dir_size_gb(base / "blobs"), 1)
    return ModelStore("Ollama", present, count, size)


_WEIGHT_SUFFIXES = (".gguf", ".safetensors")


def _scan_weight_store(name: str, dirs: list[Path], *, group_depth: int,
                       app_present: bool = False) -> ModelStore:
    """Generic on-disk model store: the first existing dir in *dirs* is scanned for weight
    files. ``group_depth`` decides what counts as one model: N groups weights by their first
    N path components (e.g. publisher/repo at depth 2, model-folder at depth 1); 0 counts
    each weight file individually (flat stores). Sizes sum all weights, binary GiB.
    """
    base = next((d for d in dirs if d.exists()), None)
    present = base is not None or app_present
    if base is None:
        return ModelStore(name, present)

    models: set[tuple[str, ...]] = set()
    total = 0
    for f in base.rglob("*"):
        try:
            if f.is_file() and f.suffix in _WEIGHT_SUFFIXES:
                rel = f.relative_to(base).parts
                if group_depth == 0:
                    models.add(rel)            # each weight file is its own model
                elif len(rel) >= group_depth:
                    models.add(rel[:group_depth])
                total += f.stat().st_size
        except Exception:
            pass
    return ModelStore(name, True, len(models), round(total / GB, 1))


def _lmstudio_inventory() -> ModelStore:
    # ~/.lmstudio/models/<publisher>/<repo>/<weights> (older: ~/.cache/lm-studio/models).
    # Bundled models under .internal/ are excluded — like HF/Ollama, count user downloads.
    return _scan_weight_store(
        "LM Studio",
        [Path.home() / ".lmstudio" / "models",
         Path.home() / ".cache" / "lm-studio" / "models"],
        group_depth=2,
        app_present=Path("/Applications/LM Studio.app").exists(),
    )


def _jan_inventory() -> ModelStore:
    # Jan keeps one folder per model: <data>/models/<model-id>/<weights> + model.json.
    return _scan_weight_store(
        "Jan",
        [Path.home() / "jan" / "models",
         Path.home() / "Library" / "Application Support" / "Jan" / "data" / "models",
         Path.home() / ".jan" / "models"],
        group_depth=1,
        app_present=Path("/Applications/Jan.app").exists(),
    )


def _gpt4all_inventory() -> ModelStore:
    # GPT4All stores flat .gguf files in its model directory (no per-repo nesting).
    return _scan_weight_store(
        "GPT4All",
        [Path.home() / "Library" / "Application Support" / "nomic.ai" / "GPT4All",
         Path.home() / ".cache" / "gpt4all"],
        group_depth=0,
        app_present=Path("/Applications/GPT4All.app").exists(),
    )


def model_stores() -> list[ModelStore]:
    return [_hf_inventory(), _ollama_inventory(), _lmstudio_inventory(),
            _jan_inventory(), _gpt4all_inventory()]


# --------------------------------------------------------------------------- #
# environment gates
# --------------------------------------------------------------------------- #
def _hf_token_present() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    candidates = [Path.home() / ".cache" / "huggingface" / "token"]
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home) / "token")
    return any(p.exists() for p in candidates)


def _user_which(cmd: str) -> str | None:
    """Resolve *cmd* on the user's PATH, with ARA's active venv stripped (so we report the
    user's tool, not ARA's bundled one — same reasoning as _user_python)."""
    path = os.environ.get("PATH", "")
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        vbin = os.path.normpath(os.path.join(venv, "bin"))
        path = os.pathsep.join(p for p in path.split(os.pathsep)
                               if os.path.normpath(p) != vbin)
    return shutil.which(cmd, path=path)


def _hf_cli() -> tuple[bool, str | None]:
    """Is the Hugging Face CLI (`hf` / `huggingface-cli`) on the user's PATH, and its version."""
    exe = _user_which("hf") or _user_which("huggingface-cli")
    if not exe:
        return False, None
    out = _run([exe, "version"]) or ""
    m = re.search(r"\d+\.\d+(?:\.\d+)?", out)
    return True, (m.group(0) if m else None)


# --------------------------------------------------------------------------- #
# the machine snapshot
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Machine:
    system: str
    os_version: str
    chip: str
    arch: str
    cpu_physical: int | None
    cpu_logical: int | None
    cpu_features: list[str]
    python_version: str | None
    ram_total_gb: float | None
    ram_available_gb: float | None
    swap_gb: float | None
    accel: Accelerator
    disk_free_gb: float | None
    runtimes: list[Runtime] = field(default_factory=list)
    framework_python: str | None = None  # interpreter the FRAMEWORKS group was probed in
    model_stores: list[ModelStore] = field(default_factory=list)
    apps: list = field(default_factory=list)  # installed AI/ML apps (ara.apps.App)
    hf_token: bool = False
    hf_cli: bool = False
    hf_cli_version: str | None = None
    power: str = "unknown"
    backend: str = "unsupported"
    engine: str = "unsupported"
    engine_ready: bool = False

    @property
    def supported(self) -> bool:
        return self.backend != "unsupported"


def profile() -> Machine:
    """Observe the host and choose a backend. Read-only — no engine import."""
    chip = chip_name()
    backend = backend_name()
    engine = "wmx-suite" if backend == "apple" else backend
    engine_ready = backend == "apple" and find_spec("wmx_suite") is not None
    total, available = _memory_gb()
    physical, logical = _cpu_counts()
    accel = accelerator(chip)
    user_py = _user_python()
    hf_cli, hf_cli_version = _hf_cli()
    return Machine(
        system=platform.system(),
        os_version=os_version(),
        chip=chip,
        arch=platform.machine() or "unknown",
        cpu_physical=physical,
        cpu_logical=logical,
        cpu_features=_cpu_features(),
        python_version=_python_version(user_py),
        ram_total_gb=total,
        ram_available_gb=available,
        swap_gb=_swap_gb(),
        accel=accel,
        disk_free_gb=_disk_free_gb(),
        runtimes=runtimes(accel.kind, user_py),
        framework_python=user_py,
        model_stores=model_stores(),
        apps=_apps.scan(),
        hf_token=_hf_token_present(),
        hf_cli=hf_cli,
        hf_cli_version=hf_cli_version,
        power=_power(),
        backend=backend,
        engine=engine,
        engine_ready=engine_ready,
    )
