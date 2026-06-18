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
import shutil
import subprocess
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path

import psutil

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


def _python_version() -> str | None:
    """Ambient python3 version (what a user in this shell would reach for)."""
    py = shutil.which("python3") or shutil.which("python")
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
    cuda_version: str | None = None


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
                    count=len(lines), compute=cc or None, cuda_version=drv or None,
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


_PY_PKGS = ("torch", "transformers", "vllm", "mlx-lm")


def _ambient_python_packages() -> dict[str, str | None]:
    """Versions of known ML packages in the ambient python3 — metadata only,
    no heavy import. Reflects this shell's python3, best-effort."""
    py = shutil.which("python3") or shutil.which("python")
    if not py:
        return {}
    code = (
        "import importlib.metadata as m, json\n"
        f"names = {list(_PY_PKGS)!r}\n"
        "out = {}\n"
        "for n in names:\n"
        "    try: out[n] = m.version(n)\n"
        "    except Exception: out[n] = None\n"
        "print(json.dumps(out))\n"
    )
    raw = _run([py, "-c", code], timeout=6)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def runtimes() -> list[Runtime]:
    amb = _ambient_python_packages()
    llama = shutil.which("llama-cli") or shutil.which("llama-server")
    lms = shutil.which("lms") or Path("/Applications/LM Studio.app").exists()
    mlx_ver = amb.get("mlx-lm")
    return [
        Runtime("PyTorch", amb.get("torch") is not None, amb.get("torch")),
        Runtime("transformers", amb.get("transformers") is not None, amb.get("transformers")),
        Runtime("vLLM", amb.get("vllm") is not None, amb.get("vllm")),
        Runtime("MLX", mlx_ver is not None or find_spec("mlx_lm") is not None, mlx_ver),
        Runtime("llama.cpp", llama is not None),
        Runtime("Ollama", shutil.which("ollama") is not None or (Path.home() / ".ollama").exists()),
        Runtime("LM Studio", bool(lms)),
    ]


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


def model_stores() -> list[ModelStore]:
    return [_hf_inventory(), _ollama_inventory()]


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
    model_stores: list[ModelStore] = field(default_factory=list)
    hf_token: bool = False
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
    return Machine(
        system=platform.system(),
        os_version=os_version(),
        chip=chip,
        arch=platform.machine() or "unknown",
        cpu_physical=physical,
        cpu_logical=logical,
        cpu_features=_cpu_features(),
        python_version=_python_version(),
        ram_total_gb=total,
        ram_available_gb=available,
        swap_gb=_swap_gb(),
        accel=accelerator(chip),
        disk_free_gb=_disk_free_gb(),
        runtimes=runtimes(),
        model_stores=model_stores(),
        hf_token=_hf_token_present(),
        power=_power(),
        backend=backend,
        engine=engine,
        engine_ready=engine_ready,
    )
