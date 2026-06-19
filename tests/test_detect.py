"""detect.py — read-only host recon: backend choice, parsers, inventories."""
from __future__ import annotations

import ara.detect as detect
from ara.detect import (
    Accelerator,
    Machine,
    Runtime,
    accelerator,
    backend_name,
)


# --------------------------------------------------------------------------- #
# backend choice
# --------------------------------------------------------------------------- #
def test_backend_name_apple(set_platform):
    set_platform("Darwin", "arm64")
    assert backend_name() == "apple"


def test_backend_name_unsupported_on_linux(set_platform):
    set_platform("Linux", "x86_64")
    assert backend_name() == "unsupported"


def test_backend_name_unsupported_on_intel_mac(set_platform):
    set_platform("Darwin", "x86_64")
    assert backend_name() == "unsupported"


# --------------------------------------------------------------------------- #
# _run / _sysctl
# --------------------------------------------------------------------------- #
def test_run_returns_stdout_for_real_command():
    assert detect._run(["echo", "hello"]) == "hello\n"


def test_run_returns_none_on_failure():
    assert detect._run(["definitely-not-a-real-binary-xyz"]) is None


def test_sysctl_none_off_darwin(set_platform):
    set_platform("Linux", "x86_64")
    assert detect._sysctl("anything") is None


def test_sysctl_reads_value_on_darwin(set_platform, run_stub):
    set_platform("Darwin", "arm64")
    run_stub.add("machdep.cpu.brand_string", "Apple M4 Pro\n")
    assert detect._sysctl("machdep.cpu.brand_string") == "Apple M4 Pro"


# --------------------------------------------------------------------------- #
# chip / os
# --------------------------------------------------------------------------- #
def test_chip_name_from_sysctl_on_darwin(set_platform, run_stub):
    set_platform("Darwin", "arm64")
    run_stub.add("machdep.cpu.brand_string", "Apple M4 Pro\n")
    assert detect.chip_name() == "Apple M4 Pro"


def test_chip_name_falls_back_to_machine(set_platform, monkeypatch):
    set_platform("Linux", "x86_64")
    monkeypatch.setattr(detect.platform, "processor", lambda: "")
    assert detect.chip_name() == "x86_64"


def test_os_version_darwin(set_platform):
    set_platform("Darwin", "arm64")
    assert detect.os_version().startswith("macOS")


def test_os_version_other(set_platform):
    set_platform("Linux", "x86_64")
    assert detect.os_version() == "Linux"


# --------------------------------------------------------------------------- #
# cpu features
# --------------------------------------------------------------------------- #
def test_cpu_features_arm_neon_only(set_platform, run_stub):
    set_platform("Darwin", "arm64")  # FEAT_BF16 unset → None
    assert detect._cpu_features() == ["NEON"]


def test_cpu_features_arm_with_bf16(set_platform, run_stub):
    set_platform("Darwin", "arm64")
    run_stub.add("FEAT_BF16", "1\n")
    assert detect._cpu_features() == ["NEON", "BF16"]


def test_cpu_features_x86_flags(set_platform, run_stub, monkeypatch):
    # Intel Mac: no /proc/cpuinfo, so flags come from sysctl (via the run_stub).
    set_platform("Darwin", "x86_64")
    run_stub.add("machdep.cpu", "fpu avx2 avx avx512f sse4_2 bmi1")
    feats = detect._cpu_features()
    assert feats == ["AVX-512", "AVX2", "AVX", "SSE4.2"]


def test_cpu_features_x86_from_proc_cpuinfo(set_platform, monkeypatch):
    # Linux: flags are read from /proc/cpuinfo.
    set_platform("Linux", "x86_64")
    monkeypatch.setattr(
        detect.Path, "read_text", lambda self: "flags : fpu avx2 sse4_2\n"
    )
    feats = detect._cpu_features()
    # "avx" matches as a substring of "avx2", so AVX2 implies AVX in the report.
    assert feats == ["AVX2", "AVX", "SSE4.2"]


# --------------------------------------------------------------------------- #
# accelerator
# --------------------------------------------------------------------------- #
def test_accelerator_nvidia_single(monkeypatch, run_stub):
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/nvidia-smi" if n == "nvidia-smi" else None)
    run_stub.add("nvidia-smi", "NVIDIA GeForce RTX 4090, 24576, 8.9, 550.00\n")
    a = accelerator("ignored")
    assert a.kind == "nvidia"
    assert a.name == "NVIDIA GeForce RTX 4090"
    assert a.vram_gb == 24.0
    assert a.api == "CUDA"
    assert a.count == 1
    assert a.compute == "8.9"
    assert a.cuda_version == "550.00"


def test_accelerator_nvidia_multi_gpu(monkeypatch, run_stub):
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/nvidia-smi" if n == "nvidia-smi" else None)
    run_stub.add("nvidia-smi", "A100, 81920, 8.0, 535\nA100, 81920, 8.0, 535\n")
    a = accelerator("ignored")
    assert a.kind == "nvidia" and a.count == 2


def test_accelerator_apple(monkeypatch, run_stub, set_platform):
    set_platform("Darwin", "arm64")
    monkeypatch.setattr("shutil.which", lambda n: None)
    run_stub.add("system_profiler", "Graphics/Displays:\n    Apple M4 Pro:\n      Total Number of Cores: 16\n")
    a = accelerator("Apple M4 Pro")
    assert a.kind == "apple"
    assert a.name == "Apple M4 Pro GPU"
    assert a.api == "Metal"
    assert a.cores == 16
    assert a.vram_gb is None


def test_accelerator_none(monkeypatch, set_platform):
    set_platform("Linux", "x86_64")
    monkeypatch.setattr("shutil.which", lambda n: None)
    a = accelerator("whatever")
    assert a.kind == "none" and a.api is None


def test_accelerator_dataclass_defaults():
    a = Accelerator("none", "x", None, None)
    assert a.count == 1 and a.cores is None and a.compute is None


# --------------------------------------------------------------------------- #
# runtimes + usability resolution
# --------------------------------------------------------------------------- #
def test_runtimes_usability_resolution(monkeypatch, fake_home):
    monkeypatch.setattr(detect, "_ambient_python_packages", lambda: {
        "torch": "2.1.0", "transformers": "4.40.0", "vllm": "0.5.0", "mlx-lm": "0.18",
    })
    monkeypatch.setattr("shutil.which", lambda n: None)
    monkeypatch.setattr(detect, "find_spec", lambda n: None)

    rts = {rt.name: rt for rt in detect.runtimes(accel_kind="apple")}

    # cross-platform runtime: no accelerator gate
    assert rts["PyTorch"].present is True
    assert rts["PyTorch"].usable is None
    assert rts["PyTorch"].requires is None

    # vLLM needs CUDA → not usable on an Apple box, with a human reason
    assert rts["vLLM"].present is True
    assert rts["vLLM"].usable is False
    assert rts["vLLM"].requires == "needs CUDA"

    # MLX needs Apple Silicon → usable here
    assert rts["MLX"].present is True
    assert rts["MLX"].usable is True
    assert rts["MLX"].requires is None


def test_runtime_requires_property():
    assert Runtime("vLLM", True, usable=False, accels=("nvidia",)).requires == "needs CUDA"
    assert Runtime("MLX", True, usable=True, accels=("apple",)).requires is None
    assert Runtime("PyTorch", True, usable=None).requires is None
    assert Runtime("x", True, usable=False, accels=("nvidia", "apple")).requires == (
        "needs CUDA / Apple Silicon"
    )


# --------------------------------------------------------------------------- #
# model store inventories
# --------------------------------------------------------------------------- #
def _write(path, nbytes=1024):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * nbytes)


def test_hf_inventory_counts_blobs(fake_home):
    hub = fake_home / ".cache" / "huggingface" / "hub"
    _write(hub / "models--org--repo" / "blobs" / "abc123", detect.GB)  # 1 GiB
    _write(hub / "models--org--other" / "blobs" / "def456", detect.GB // 2)
    store = detect._hf_inventory()
    assert store.name == "HF cache" and store.present is True
    assert store.count == 2
    assert store.size_gb == 1.5


def test_hf_inventory_absent(fake_home):
    store = detect._hf_inventory()
    assert store.present is False and store.count == 0


def test_ollama_inventory(fake_home):
    base = fake_home / ".ollama" / "models"
    _write(base / "manifests" / "registry" / "library" / "llama3" / "latest", 10)
    _write(base / "blobs" / "sha256-abc", detect.GB)
    store = detect._ollama_inventory()
    assert store.name == "Ollama" and store.present is True
    assert store.count == 1
    assert store.size_gb == 1.0


def test_scan_weight_store_depth_two_groups_by_publisher_repo(fake_home, tmp_path):
    base = tmp_path / "store"
    _write(base / "pubA" / "repo1" / "model.gguf", detect.GB)
    _write(base / "pubA" / "repo1" / "extra.safetensors", detect.GB)  # same model
    _write(base / "pubB" / "repo2" / "model.gguf", detect.GB)
    store = detect._scan_weight_store("S", [base], group_depth=2)
    assert store.count == 2  # (pubA/repo1) and (pubB/repo2)
    assert store.size_gb == 3.0


def test_scan_weight_store_depth_one(fake_home, tmp_path):
    base = tmp_path / "jan"
    _write(base / "modelA" / "w.gguf", detect.GB)
    _write(base / "modelB" / "w.gguf", detect.GB)
    store = detect._scan_weight_store("Jan", [base], group_depth=1)
    assert store.count == 2


def test_scan_weight_store_depth_zero_counts_each_file(fake_home, tmp_path):
    base = tmp_path / "flat"
    _write(base / "a.gguf", detect.GB)
    _write(base / "b.gguf", detect.GB)
    _write(base / "notes.txt", 5)  # ignored — not a weight suffix
    store = detect._scan_weight_store("GPT4All", [base], group_depth=0)
    assert store.count == 2


def test_scan_weight_store_missing_dir_uses_app_present_flag(tmp_path):
    missing = tmp_path / "nope"
    store = detect._scan_weight_store("X", [missing], group_depth=1, app_present=True)
    assert store.present is True and store.count == 0


def test_model_stores_returns_all_five(fake_home):
    stores = detect.model_stores()
    names = [s.name for s in stores]
    assert names == ["HF cache", "Ollama", "LM Studio", "Jan", "GPT4All"]


# --------------------------------------------------------------------------- #
# hf token gate
# --------------------------------------------------------------------------- #
def test_hf_token_from_env(fake_home, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    assert detect._hf_token_present() is True


def test_hf_token_from_file(fake_home):
    tok = fake_home / ".cache" / "huggingface" / "token"
    tok.parent.mkdir(parents=True)
    tok.write_text("hf_xxx")
    assert detect._hf_token_present() is True


def test_hf_token_absent(fake_home):
    assert detect._hf_token_present() is False


# --------------------------------------------------------------------------- #
# profile() assembly
# --------------------------------------------------------------------------- #
def test_profile_on_apple(set_platform, run_stub, fake_home, monkeypatch):
    set_platform("Darwin", "arm64")
    monkeypatch.setattr("shutil.which", lambda n: None)
    monkeypatch.setattr(detect, "find_spec", lambda n: None)  # wmx_suite "not installed"
    run_stub.add("machdep.cpu.brand_string", "Apple M4 Pro\n")

    m = detect.profile()
    assert isinstance(m, Machine)
    assert m.backend == "apple"
    assert m.engine == "wmx-suite"
    assert m.engine_ready is False
    assert m.supported is True
    assert m.arch == "arm64"
    assert len(m.model_stores) == 5
    assert m.runtimes  # non-empty


def test_profile_engine_ready_when_spec_found(set_platform, run_stub, fake_home, monkeypatch):
    set_platform("Darwin", "arm64")
    monkeypatch.setattr("shutil.which", lambda n: None)
    monkeypatch.setattr(detect, "find_spec", lambda n: object())
    m = detect.profile()
    assert m.engine_ready is True


# --------------------------------------------------------------------------- #
# ambient python package probe
# --------------------------------------------------------------------------- #
def test_ambient_python_packages_parses_json(run_stub, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/python3")
    run_stub.add("importlib", '{"torch": "2.1.0", "vllm": null}')
    out = detect._ambient_python_packages()
    assert out == {"torch": "2.1.0", "vllm": None}


def test_ambient_python_packages_empty_without_python(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda n: None)
    assert detect._ambient_python_packages() == {}


def test_ambient_python_packages_empty_on_bad_output(run_stub, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/python3")
    run_stub.add("importlib", "not json")
    assert detect._ambient_python_packages() == {}


def test_profile_unsupported(set_platform, run_stub, fake_home, monkeypatch):
    set_platform("Linux", "x86_64")
    monkeypatch.setattr("shutil.which", lambda n: None)
    m = detect.profile()
    assert m.backend == "unsupported"
    assert m.supported is False
    assert m.engine == "unsupported"
