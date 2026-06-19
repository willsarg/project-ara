"""detect.py — read-only host recon: backend choice, parsers, inventories."""
from __future__ import annotations

import sys
import types

import ara.detect as detect
from ara.detect import (
    Accelerator,
    Machine,
    Runtime,
    accelerator,
    backend_name,
)


def _raise(exc=RuntimeError("boom")):
    """A callable that ignores its args and raises — for forcing except branches."""
    def _f(*a, **k):
        raise exc
    return _f


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


def test_accelerator_nvidia_garbage_falls_through(monkeypatch, run_stub, set_platform):
    # nvidia-smi present but unparseable → don't crash; fall back to the host's real GPU.
    set_platform("Darwin", "arm64")
    monkeypatch.setattr("shutil.which", lambda n, path=None: "/usr/bin/nvidia-smi" if n == "nvidia-smi" else None)
    run_stub.add("nvidia-smi", "garbage with no commas\n")
    a = accelerator("Apple M4 Pro")
    assert a.kind == "apple"  # not nvidia — the bad row was swallowed


def test_apple_gpu_cores_non_integer_returns_none(run_stub):
    run_stub.add("system_profiler", "      Total Number of Cores: many\n")
    assert detect._apple_gpu_cores() is None


def test_apple_gpu_cores_no_cores_line_returns_none(run_stub):
    run_stub.add("system_profiler", "Graphics/Displays:\n    Apple M4 Pro:\n      Vendor: Apple\n")
    assert detect._apple_gpu_cores() is None


def test_accelerator_nvidia_blank_output_falls_through(monkeypatch, run_stub, set_platform):
    set_platform("Darwin", "arm64")
    monkeypatch.setattr("shutil.which", lambda n, path=None: "/usr/bin/nvidia-smi" if n == "nvidia-smi" else None)
    run_stub.add("nvidia-smi", "   \n")  # present but whitespace-only → skip, don't crash
    a = accelerator("Apple M4 Pro")
    assert a.kind == "apple"


def test_accelerator_dataclass_defaults():
    a = Accelerator("none", "x", None, None)
    assert a.count == 1 and a.cores is None and a.compute is None


# --------------------------------------------------------------------------- #
# runtimes + usability resolution
# --------------------------------------------------------------------------- #
def test_runtimes_usability_and_kind_split(monkeypatch, fake_home):
    monkeypatch.setattr(detect, "_python_packages", lambda py, names: {
        "torch": "2.1.0", "transformers": "4.40.0", "tensorflow": None,
        "vllm": "0.5.0", "mlx-lm": "0.18",
    })
    monkeypatch.setattr(detect, "_ara_pkg_version", lambda name: None)
    monkeypatch.setattr("shutil.which", lambda n, path=None: None)
    monkeypatch.setattr(detect, "find_spec", lambda n: None)

    rts = {rt.name: rt for rt in detect.runtimes(accel_kind="apple", user_py="/usr/bin/python3")}

    # frameworks are libraries: no accelerator gate, tagged "framework"
    assert rts["PyTorch"].kind == "framework"
    assert rts["PyTorch"].present is True
    assert rts["PyTorch"].usable is None
    assert rts["PyTorch"].requires is None
    assert rts["TensorFlow"].kind == "framework"
    assert rts["TensorFlow"].present is False  # version was None

    # vLLM is an engine that needs CUDA → not usable on an Apple box, with a reason
    assert rts["vLLM"].kind == "engine"
    assert rts["vLLM"].present is True
    assert rts["vLLM"].usable is False
    assert rts["vLLM"].requires == "needs CUDA"

    # MLX engine needs Apple Silicon → usable here
    assert rts["MLX"].kind == "engine"
    assert rts["MLX"].present is True
    assert rts["MLX"].usable is True
    assert rts["MLX"].requires is None


def test_runtimes_detected_via_second_or_signal(monkeypatch, fake_home):
    # Each engine is detectable via either of two signals; here only the SECOND is
    # present, pinning the `or` (an `and` mutation would drop them).
    monkeypatch.setattr(detect, "_python_packages", lambda py, names: {n: None for n in names})
    monkeypatch.setattr(detect, "_ara_pkg_version", lambda name: None)
    present = {"llama-server", "vllm"}   # llama-cli absent / llama-server present; vllm via CLI
    monkeypatch.setattr("shutil.which", lambda n, path=None: f"/x/{n}" if n in present else None)
    monkeypatch.setattr(detect, "find_spec", lambda n: object() if n == "mlx_lm" else None)

    rts = {rt.name: rt for rt in detect.runtimes("apple")}
    assert rts["llama.cpp"].present is True   # via llama-server (2nd operand of the or)
    assert rts["vLLM"].present is True         # via the vllm CLI (2nd operand)
    assert rts["MLX"].present is True          # via find_spec("mlx_lm") (2nd operand)


def test_runtimes_mlx_falls_back_to_ara_env(monkeypatch, fake_home):
    # The user's python has no mlx-lm, but ARA bundles the MLX engine → still present.
    monkeypatch.setattr(detect, "_python_packages", lambda py, names: {n: None for n in names})
    monkeypatch.setattr(detect, "_ara_pkg_version", lambda name: "0.18" if name == "mlx-lm" else None)
    monkeypatch.setattr("shutil.which", lambda n, path=None: None)
    monkeypatch.setattr(detect, "find_spec", lambda n: None)

    rts = {rt.name: rt for rt in detect.runtimes("apple", user_py=None)}
    assert rts["MLX"].present is True
    assert rts["MLX"].version == "0.18"


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
    # Sparse file: stat().st_size reports nbytes, but no blocks are actually written,
    # so "1 GiB" inventory fixtures cost ~nothing on disk (and don't fill it under
    # repeated runs like mutation testing).
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(nbytes)


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


def test_scan_weight_store_depth_equals_path_length(fake_home, tmp_path):
    # boundary: a weight file whose path depth == group_depth still forms a group
    # (the check is >=, not >). rel = ("pub", "model.gguf") has length 2 at depth 2.
    base = tmp_path / "store"
    _write(base / "pub" / "model.gguf", detect.GB)
    store = detect._scan_weight_store("S", [base], group_depth=2)
    assert store.count == 1


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
    monkeypatch.setattr("shutil.which", lambda n, path=None: None)
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
    # which → None means no user python resolved; framework probe falls back to none
    assert m.framework_python is None
    assert {rt.kind for rt in m.runtimes} == {"engine", "framework"}


def test_profile_engine_ready_when_spec_found(set_platform, run_stub, fake_home, monkeypatch):
    set_platform("Darwin", "arm64")
    monkeypatch.setattr("shutil.which", lambda n, path=None: None)
    monkeypatch.setattr(detect, "find_spec", lambda n: object())
    m = detect.profile()
    assert m.engine_ready is True


# --------------------------------------------------------------------------- #
# user-python resolution (the real shell python, not ARA's venv)
# --------------------------------------------------------------------------- #
def test_user_python_strips_venv_bin(monkeypatch):
    monkeypatch.setenv("PATH", "/venv/bin:/usr/bin")
    monkeypatch.setenv("VIRTUAL_ENV", "/venv")
    monkeypatch.setattr(detect.sys, "executable", "/venv/bin/python3")
    monkeypatch.setattr(detect.os.path, "realpath", lambda p, *a, **k: p)  # identity
    seen = {}

    def fake_which(name, path=None):
        seen["path"] = path
        return "/usr/bin/python3" if name == "python3" else None

    monkeypatch.setattr("shutil.which", fake_which)
    assert detect._user_python() == "/usr/bin/python3"
    assert "/venv/bin" not in seen["path"]   # the venv's bin was stripped
    assert "/usr/bin" in seen["path"]


def test_user_python_none_when_resolves_back_to_ara(monkeypatch):
    monkeypatch.setenv("PATH", "/venv/bin:/usr/bin")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(detect.sys, "executable", "/venv/bin/python3")
    monkeypatch.setattr(detect.os.path, "realpath", lambda p, *a, **k: p)
    monkeypatch.setattr("shutil.which", lambda name, path=None: "/venv/bin/python3")
    assert detect._user_python() is None


def test_user_python_none_when_no_python(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr("shutil.which", lambda name, path=None: None)
    assert detect._user_python() is None


# --------------------------------------------------------------------------- #
# python package probes
# --------------------------------------------------------------------------- #
def test_python_packages_parses_json(run_stub):
    run_stub.add("importlib", '{"torch": "2.1.0", "tensorflow": null}')
    out = detect._python_packages("/usr/bin/python3", ("torch", "tensorflow"))
    assert out == {"torch": "2.1.0", "tensorflow": None}


def test_python_packages_blank_without_interpreter():
    assert detect._python_packages(None, ("torch", "vllm")) == {"torch": None, "vllm": None}


def test_python_packages_blank_on_run_failure(run_stub):
    # run_stub returns None for an unmatched command → all-None dict, not a crash.
    assert detect._python_packages("/usr/bin/python3", ("torch",)) == {"torch": None}


def test_python_packages_blank_on_bad_output(run_stub):
    run_stub.add("importlib", "not json at all")
    assert detect._python_packages("/usr/bin/python3", ("torch", "vllm")) == {
        "torch": None, "vllm": None}


def test_ara_pkg_version_reads_own_env():
    # psutil is a real dependency in ARA's own environment.
    assert detect._ara_pkg_version("psutil") is not None
    assert detect._ara_pkg_version("definitely-not-installed-xyz") is None


def test_profile_unsupported(set_platform, run_stub, fake_home, monkeypatch):
    set_platform("Linux", "x86_64")
    monkeypatch.setattr("shutil.which", lambda n, path=None: None)
    m = detect.profile()
    assert m.backend == "unsupported"
    assert m.supported is False
    assert m.engine == "unsupported"


# --------------------------------------------------------------------------- #
# defensive fallbacks — system reads that return None/empty when the OS throws
# --------------------------------------------------------------------------- #
def test_memory_gb_none_on_error(monkeypatch):
    monkeypatch.setattr(detect.psutil, "virtual_memory", _raise())
    assert detect._memory_gb() == (None, None)


def test_swap_gb_none_on_error(monkeypatch):
    monkeypatch.setattr(detect.psutil, "swap_memory", _raise())
    assert detect._swap_gb() is None


def test_cpu_counts_none_on_error(monkeypatch):
    monkeypatch.setattr(detect.psutil, "cpu_count", _raise())
    assert detect._cpu_counts() == (None, None)


def test_disk_free_none_on_error(monkeypatch):
    monkeypatch.setattr("shutil.disk_usage", _raise(OSError("no volume")))
    assert detect._disk_free_gb() is None


def test_power_no_battery_when_unavailable(monkeypatch):
    monkeypatch.setattr(detect.psutil, "sensors_battery", lambda: None, raising=False)
    assert detect._power() == "AC (no battery)"


def test_power_exception_treated_as_no_battery(monkeypatch):
    monkeypatch.setattr(detect.psutil, "sensors_battery", _raise(), raising=False)
    assert detect._power() == "AC (no battery)"


def test_power_ac_vs_battery(monkeypatch):
    monkeypatch.setattr(detect.psutil, "sensors_battery",
                        lambda: types.SimpleNamespace(power_plugged=True, percent=80.0),
                        raising=False)
    assert detect._power() == "AC power"
    monkeypatch.setattr(detect.psutil, "sensors_battery",
                        lambda: types.SimpleNamespace(power_plugged=False, percent=80.0),
                        raising=False)
    assert detect._power() == "battery 80%"


def test_dir_size_gb_swallows_stat_errors(tmp_path, monkeypatch):
    (tmp_path / "f").write_bytes(b"x")
    monkeypatch.setattr(detect.Path, "is_file", _raise(OSError()))
    assert detect._dir_size_gb(tmp_path) == 0.0


def test_scan_weight_store_swallows_stat_errors(tmp_path, monkeypatch):
    _write(tmp_path / "a.gguf", 10)
    monkeypatch.setattr(detect.Path, "is_file", _raise(OSError()))
    store = detect._scan_weight_store("X", [tmp_path], group_depth=0)
    assert store.count == 0


def test_scan_weight_store_file_shallower_than_group_depth(tmp_path):
    # rel length 1 but group_depth 2 → too shallow to form a group, yet bytes still count.
    _write(tmp_path / "loose.gguf", detect.GB)
    store = detect._scan_weight_store("X", [tmp_path], group_depth=2)
    assert store.count == 0
    assert store.size_gb == 1.0


# --------------------------------------------------------------------------- #
# HF_HOME overrides + live python-version probe
# --------------------------------------------------------------------------- #
def test_hf_hub_dir_uses_hf_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert detect._hf_hub_dir() == tmp_path / "hf" / "hub"


def test_hf_token_from_hf_home_token_file(fake_home, monkeypatch, tmp_path):
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "token").write_text("hf_xxx")
    monkeypatch.setenv("HF_HOME", str(hf))
    assert detect._hf_token_present() is True


def test_python_version_live_against_real_interpreter():
    # No run_stub here: actually shells out to `<py> --version` and parses it.
    ver = detect._python_version(sys.executable)
    assert ver and ver[0].isdigit()


# --------------------------------------------------------------------------- #
# remaining branch corners
# --------------------------------------------------------------------------- #
def test_python_version_falls_back_when_unreadable(run_stub):
    # py given but `<py> --version` yields nothing (run_stub → None) → platform fallback.
    ver = detect._python_version("/usr/bin/python3")
    assert ver and ver[0].isdigit()


def test_dir_size_gb_missing_dir_is_zero(tmp_path):
    assert detect._dir_size_gb(tmp_path / "nope") == 0.0


def test_dir_size_gb_skips_subdirectories(tmp_path):
    (tmp_path / "sub").mkdir()                 # not a file → skipped, loop continues
    (tmp_path / "f").write_bytes(b"x" * 10)
    assert detect._dir_size_gb(tmp_path) == 10 / detect.GB
