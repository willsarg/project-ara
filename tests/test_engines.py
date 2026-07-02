# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""engines.py — the engine catalog + isolated-env install lifecycle (`ara install`)."""
from __future__ import annotations

import ara.engines as engines


# --------------------------------------------------------------------------- #
# for_hardware() — the light "what would ARA pick here?" probe behind `auto`
# --------------------------------------------------------------------------- #
def test_for_hardware_picks_wmx_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(engines.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(engines.platform, "machine", lambda: "arm64")
    assert engines.for_hardware() == "wmx"


def test_for_hardware_picks_wcx_when_nvidia_smi_present(monkeypatch):
    monkeypatch.setattr(engines.platform, "system", lambda: "Windows")
    monkeypatch.setattr(engines.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(engines.shutil, "which",
                        lambda n: "C:/Windows/System32/nvidia-smi.exe" if n == "nvidia-smi" else None)
    assert engines.for_hardware() == "wcx"


def test_for_hardware_none_when_no_known_accelerator(monkeypatch):
    monkeypatch.setattr(engines.platform, "system", lambda: "Linux")
    monkeypatch.setattr(engines.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(engines.shutil, "which", lambda n: None)
    assert engines.for_hardware() is None


# --------------------------------------------------------------------------- #
# resolve() — map an --engine value (wmx | wcx | cpu | auto) to a concrete key
# --------------------------------------------------------------------------- #
def test_resolve_passes_through_explicit_engine():
    assert engines.resolve("wmx") == "wmx"
    assert engines.resolve("wcx") == "wcx"
    assert engines.resolve("cpu") == "cpu"
    assert engines.resolve("vulkan") == "vulkan"
    assert engines.resolve("cuda-gguf") == "cuda-gguf"


def test_resolve_auto_uses_hardware_pick(monkeypatch):
    monkeypatch.setattr(engines, "for_hardware", lambda: "wcx")
    assert engines.resolve("auto") == "wcx"


def test_resolve_auto_none_when_no_match(monkeypatch):
    monkeypatch.setattr(engines, "for_hardware", lambda: None)
    assert engines.resolve("auto") is None


def test_resolve_unknown_is_none():
    assert engines.resolve("nonsense") is None


def test_for_backend_maps_backend_to_engine():
    assert engines.for_backend("apple") == "wmx"
    assert engines.for_backend("cuda") == "wcx"          # NVIDIA still auto-picks wcx
    assert engines.for_backend("cpu") == "cpu"
    assert engines.for_backend("vulkan") == "vulkan"
    assert engines.for_backend("cuda_gguf") == "cuda-gguf"
    assert engines.for_backend("unsupported") is None


def test_for_backend_cuda_still_returns_wcx_not_cuda_gguf():
    # cuda-gguf is opt-in only; hardware auto-pick for NVIDIA is wcx (backend="cuda").
    # for_backend("cuda") must NOT return "cuda-gguf" (its backend is "cuda_gguf", distinct).
    assert engines.for_backend("cuda") == "wcx"
    assert engines.for_backend("cuda") != "cuda-gguf"


# --------------------------------------------------------------------------- #
# is_installed() — is the engine's isolated env present? (no import of the engine)
# --------------------------------------------------------------------------- #
def test_is_installed_true_when_env_exists(monkeypatch):
    monkeypatch.setattr(engines.engine_env, "exists", lambda name: name == "apple")
    assert engines.is_installed("wmx") is True


def test_is_installed_false_when_env_absent(monkeypatch):
    monkeypatch.setattr(engines.engine_env, "exists", lambda name: False)
    assert engines.is_installed("wmx") is False


def test_is_installed_false_for_unknown_engine():
    assert engines.is_installed("nonsense") is False


# --------------------------------------------------------------------------- #
# source_for() — the install source, with a dev env-var override (external suites)
# --------------------------------------------------------------------------- #
def test_source_for_defaults_to_vendored_path(monkeypatch):
    # A folded engine (no git spec) installs from the package source ARA ships in its wheel,
    # under ara/_vendor/<key> — reproducible and offline. The dir (with its pyproject) must exist.
    monkeypatch.delenv("ARA_WMX_SOURCE", raising=False)
    assert "spec" not in engines.ENGINES["wmx"]            # folded in — no git source
    src = engines.source_for("wmx")
    assert src == str(engines._vendored_source("wmx"))
    assert (engines._vendored_source("wmx") / "pyproject.toml").is_file()


def test_source_for_uses_env_override(monkeypatch):
    # The dev override is a local checkout — used verbatim, with no SHA pin appended.
    monkeypatch.setenv("ARA_WMX_SOURCE", "../wmx-suite")
    assert engines.source_for("wmx") == "../wmx-suite"


# --------------------------------------------------------------------------- #
# _install_targets() — the uv-pip args per engine kind
# --------------------------------------------------------------------------- #
def test_install_targets_builtin_is_the_package_list(monkeypatch):
    # Off Windows the builtin engine installs its plain package list (source build is the
    # universal path — Linux/macOS/aarch64/Pi, where no prebuilt wheel index serves them).
    monkeypatch.setattr(engines.platform, "system", lambda: "Linux")
    assert engines._install_targets("cpu") == engines.ENGINES["cpu"]["packages"]


def test_install_targets_cpu_forces_prebuilt_wheel_on_windows(monkeypatch):
    # llama-cpp-python ships NO PyPI wheels; a stock Windows box has no MSVC, so a source
    # build fails. On Windows ARA must pull a prebuilt CPU wheel from the project's own index,
    # and `--only-binary` makes that deterministic (never silently falls back to building).
    monkeypatch.setattr(engines.platform, "system", lambda: "Windows")
    targets = engines._install_targets("cpu")
    spec = engines.ENGINES["cpu"]["wheel_only"]["llama-cpp-python"]
    assert targets[:4] == [
        "--only-binary", "llama-cpp-python", "--extra-index-url", spec["index"]]
    # the llama-cpp-python requirement gets the AVX2-baseline ceiling appended (post-0.3.19
    # abetlen wheels are AVX-512-only and fault on CPUs without it); other deps pass through.
    assert targets[4:] == [
        f"llama-cpp-python>=0.3,<={spec['max_version']}", "psutil", "huggingface_hub"]


def test_install_targets_vulkan_forces_prebuilt_wheel_on_linux(monkeypatch):
    # The Vulkan engine MUST pull the prebuilt Vulkan wheel from the project's own index on
    # x86_64 Linux: a plain install would resolve llama-cpp-python to the `cpu` engine's
    # CPU-only wheel from uv's cache (same version, no GGML_VULKAN). `--only-binary` makes it
    # deterministic. (Slug: 2026-06-25-vulkan-amd-engine-lane)
    monkeypatch.setattr(engines.platform, "system", lambda: "Linux")
    targets = engines._install_targets("vulkan")
    spec = engines.ENGINES["vulkan"]["wheel_only"]["llama-cpp-python"]
    assert spec["index"].endswith("/whl/vulkan")
    assert targets[:4] == [
        "--only-binary", "llama-cpp-python", "--extra-index-url", spec["index"]]
    assert targets[4:] == [
        f"llama-cpp-python>=0.3,<={spec['max_version']}", "psutil", "huggingface_hub"]


def test_install_targets_vulkan_forces_prebuilt_wheel_on_windows(monkeypatch):
    # Windows also has prebuilt Vulkan wheels on the index → same forced-wheel path.
    monkeypatch.setattr(engines.platform, "system", lambda: "Windows")
    targets = engines._install_targets("vulkan")
    spec = engines.ENGINES["vulkan"]["wheel_only"]["llama-cpp-python"]
    assert targets[:4] == [
        "--only-binary", "llama-cpp-python", "--extra-index-url", spec["index"]]


def test_install_targets_vulkan_plain_on_macos(monkeypatch):
    # macOS isn't in vulkan's wheel_platforms (no Vulkan there — it's Metal/MLX country), so the
    # package list passes through untouched rather than forcing a non-existent wheel.
    monkeypatch.setattr(engines.platform, "system", lambda: "Darwin")
    assert engines._install_targets("vulkan") == engines.ENGINES["vulkan"]["packages"]


def test_install_targets_vendored_is_plain_path(monkeypatch):
    # A folded engine installs from its vendored dir — a plain (non-editable) path, since the source
    # is read-only inside ARA's wheel. No extras for wmx.
    monkeypatch.delenv("ARA_WMX_SOURCE", raising=False)
    assert engines._install_targets("wmx") == [str(engines._vendored_source("wmx"))]


def test_install_targets_external_local_is_editable(monkeypatch):
    monkeypatch.setenv("ARA_WMX_SOURCE", "../wmx-suite")
    assert engines._install_targets("wmx") == ["-e", "../wmx-suite"]


def test_install_targets_wcx_folds_extra_and_torch_backend(monkeypatch):
    # Vendored wcx installs from its path with the [cuda] extra appended and the torch-backend
    # selector leading — plain (non-editable), since it's read-only inside ARA's wheel.
    monkeypatch.delenv("ARA_WCX_SOURCE", raising=False)
    assert engines._install_targets("wcx") == [
        "--torch-backend=auto", f"{engines._vendored_source('wcx')}[cuda]"]


def test_install_targets_wcx_local_is_editable_with_extra(monkeypatch):
    monkeypatch.setenv("ARA_WCX_SOURCE", "../wcx-suite")
    assert engines._install_targets("wcx") == [
        "--torch-backend=auto", "-e", "../wcx-suite[cuda]"]


# --------------------------------------------------------------------------- #
# install() — create the isolated env (engine_env injected)
# --------------------------------------------------------------------------- #
def test_wcx_is_available_and_installs_into_its_cuda_env(monkeypatch):
    # wcx is converted to the isolated-env worker model, so it installs like any other engine —
    # into the `cuda` env, folding the [cuda] extra + the auto torch-backend selector.
    monkeypatch.setattr(engines, "is_installed", lambda k: False)
    monkeypatch.delenv("ARA_WCX_SOURCE", raising=False)
    seen = {}

    def fake_create(name, packages, *, python=None, **kw):
        seen.update(name=name, packages=packages, python=python)

    monkeypatch.setattr(engines.engine_env, "create", fake_create)
    assert engines.ENGINES["wcx"]["available"] is True
    assert engines.install("wcx").status == "installed"
    assert seen["name"] == "cuda" and seen["python"] == "3.12"
    assert seen["packages"] == [
        "--torch-backend=auto", f"{engines._vendored_source('wcx')}[cuda]"]


def test_install_unknown_engine_reports_unknown():
    assert engines.install("nonsense").status == "unknown"


def test_install_unavailable_engine_is_coming_soon(monkeypatch):
    monkeypatch.setitem(engines.ENGINES["wcx"], "available", False)   # force coming-soon
    created = []
    monkeypatch.setattr(engines.engine_env, "create",
                        lambda *a, **k: created.append(a))
    r = engines.install("wcx")
    assert r.status == "coming_soon"
    assert created == []   # never built an env for a not-yet-available engine


def test_install_already_present_is_noop(monkeypatch):
    # Present AND current (stamp matches) → noop: don't rebuild.
    monkeypatch.setattr(engines, "is_installed", lambda k: True)
    monkeypatch.setattr(engines.engine_env, "stamped_version", lambda n: engines._ara_version())
    created = []
    monkeypatch.setattr(engines.engine_env, "create", lambda *a, **k: created.append(a))
    r = engines.install("wmx")
    assert r.status == "already"
    assert created == []   # already there + current → don't rebuild


def test_install_stamps_env_with_current_version(monkeypatch):
    # A fresh install stamps the env with the current ARA version (so the next install can tell
    # whether it's stale). version= is threaded into engine_env.create.
    monkeypatch.setattr(engines, "is_installed", lambda k: False)
    monkeypatch.setattr(engines, "_ara_version", lambda: "3.1.4")
    seen = {}
    monkeypatch.setattr(engines.engine_env, "create",
                        lambda name, packages, *, python=None, version=None:
                        seen.update(version=version))
    assert engines.install("wmx").status == "installed"
    assert seen["version"] == "3.1.4"


def test_install_reinstalls_on_stamp_mismatch(monkeypatch):
    # Present but a DIFFERENT stamp (an older ARA wheel built this env) → tear down + reinstall,
    # reported as "refreshed", stamped with the current version.
    monkeypatch.setattr(engines, "is_installed", lambda k: True)
    monkeypatch.setattr(engines, "_ara_version", lambda: "2.0.0")
    monkeypatch.setattr(engines.engine_env, "stamped_version", lambda n: "1.0.0")
    removed, created = [], {}
    monkeypatch.setattr(engines.engine_env, "remove", lambda n: removed.append(n))
    monkeypatch.setattr(engines.engine_env, "create",
                        lambda name, packages, *, python=None, version=None:
                        created.update(name=name, version=version))
    r = engines.install("wmx")
    assert r.status == "refreshed"
    assert removed == ["apple"]                 # old env wiped first
    assert created == {"name": "apple", "version": "2.0.0"}


def test_install_reinstalls_on_missing_stamp(monkeypatch):
    # Present but UNSTAMPED (a pre-stamp ARA built this env) → treated as stale → refreshed.
    monkeypatch.setattr(engines, "is_installed", lambda k: True)
    monkeypatch.setattr(engines, "_ara_version", lambda: "2.0.0")
    monkeypatch.setattr(engines.engine_env, "stamped_version", lambda n: None)
    removed = []
    monkeypatch.setattr(engines.engine_env, "remove", lambda n: removed.append(n))
    monkeypatch.setattr(engines.engine_env, "create", lambda *a, **k: None)
    assert engines.install("wmx").status == "refreshed"
    assert removed == ["apple"]


def test_install_refresh_forces_reinstall_even_when_current(monkeypatch):
    # refresh=True: reinstall even though the stamp already matches the current version.
    monkeypatch.setattr(engines, "is_installed", lambda k: True)
    monkeypatch.setattr(engines, "_ara_version", lambda: "2.0.0")
    monkeypatch.setattr(engines.engine_env, "stamped_version", lambda n: "2.0.0")
    removed = []
    monkeypatch.setattr(engines.engine_env, "remove", lambda n: removed.append(n))
    monkeypatch.setattr(engines.engine_env, "create", lambda *a, **k: None)
    assert engines.install("wmx", refresh=True).status == "refreshed"
    assert removed == ["apple"]


def test_install_creates_env_with_targets_and_python_pin(monkeypatch):
    monkeypatch.setattr(engines, "is_installed", lambda k: False)
    monkeypatch.delenv("ARA_WMX_SOURCE", raising=False)
    seen = {}

    def fake_create(name, packages, *, python=None, **kw):
        seen.update(name=name, packages=packages, python=python)

    monkeypatch.setattr(engines.engine_env, "create", fake_create)
    r = engines.install("wmx")
    assert r.status == "installed"
    assert seen == {"name": "apple",
                    "packages": [str(engines._vendored_source("wmx"))],
                    "python": "3.12"}


def test_install_builtin_cpu_creates_env_with_packages(monkeypatch):
    # Force the non-Windows branch deterministically on any host: on Windows the product
    # takes the prebuilt-wheel path (already covered by test_install_targets_cpu_forces_prebuilt_wheel_on_windows),
    # and this test exercises the plain source-build path.
    monkeypatch.setattr(engines.platform, "system", lambda: "Linux")
    monkeypatch.setattr(engines, "is_installed", lambda k: False)
    seen = {}
    monkeypatch.setattr(engines.engine_env, "create",
                        lambda name, packages, **kw: seen.update(name=name, packages=packages))
    assert engines.install("cpu").status == "installed"
    assert seen["name"] == "cpu"
    assert "llama-cpp-python>=0.3" in seen["packages"]


def test_install_reports_failed_on_engine_env_error(monkeypatch):
    monkeypatch.setattr(engines, "is_installed", lambda k: False)
    monkeypatch.delenv("ARA_WMX_SOURCE", raising=False)

    def boom(*a, **k):
        raise engines.engine_env.EngineEnvError("resolution impossible")

    monkeypatch.setattr(engines.engine_env, "create", boom)
    r = engines.install("wmx")
    assert r.status == "failed"
    assert "resolution impossible" in r.detail


# --------------------------------------------------------------------------- #
# uninstall() — symmetric: remove the engine's env
# --------------------------------------------------------------------------- #
def test_uninstall_unknown_engine_reports_unknown():
    assert engines.uninstall("nonsense").status == "unknown"


def test_uninstall_absent_engine_is_noop(monkeypatch):
    monkeypatch.setattr(engines, "is_installed", lambda k: False)
    removed = []
    monkeypatch.setattr(engines.engine_env, "remove", lambda name: removed.append(name))
    assert engines.uninstall("wmx").status == "absent"
    assert removed == []   # nothing installed → nothing to remove


def test_uninstall_removes_the_env(monkeypatch):
    monkeypatch.setattr(engines, "is_installed", lambda k: True)
    removed = []
    monkeypatch.setattr(engines.engine_env, "remove", lambda name: removed.append(name))
    r = engines.uninstall("wmx")
    assert r.status == "removed"
    assert removed == ["apple"]   # the backend/env name, not the dist


# --------------------------------------------------------------------------- #
# model_kinds — every shipping engine declares which model formats it accepts
# --------------------------------------------------------------------------- #
def test_every_shipping_engine_has_model_kinds():
    for key in ("wmx", "wcx", "cpu", "vulkan", "cuda-gguf"):
        assert "model_kinds" in engines.ENGINES[key], f"{key!r} missing model_kinds"


def test_vulkan_is_gguf_capable_but_cpu_stays_the_gguf_default():
    # vulkan also accepts GGUF, but it's ordered AFTER cpu so the cheap classifier still defaults
    # a bare .gguf to the CPU engine (vulkan is opt-in via --engine). (2026-06-25-vulkan-amd-engine-lane)
    assert "gguf" in engines.ENGINES["vulkan"]["model_kinds"]
    assert engines.engine_for_model("model-Q4_K_M.gguf") == "cpu"


def test_cuda_gguf_is_gguf_capable_but_cpu_stays_the_gguf_default():
    # cuda-gguf also accepts GGUF, ordered AFTER cpu (and vulkan) so the default is still cpu.
    # Opt-in via --engine cuda-gguf. (2026-06-29-cuda-gguf-hybrid-two-wall-engine)
    assert "gguf" in engines.ENGINES["cuda-gguf"]["model_kinds"]
    assert engines.engine_for_model("model-Q4_K_M.gguf") == "cpu"


# --------------------------------------------------------------------------- #
# cuda-gguf install targets — prebuilt CUDA-124 wheel on Linux/Windows
# --------------------------------------------------------------------------- #
def test_install_targets_cuda_gguf_forces_prebuilt_wheel_on_linux(monkeypatch):
    # CUDA wheels only exist for Linux + Windows; must be forced with --only-binary.
    # (Slug: 2026-06-29-cuda-gguf-hybrid-two-wall-engine)
    monkeypatch.setattr(engines.platform, "system", lambda: "Linux")
    targets = engines._install_targets("cuda-gguf")
    spec = engines.ENGINES["cuda-gguf"]["wheel_only"]["llama-cpp-python"]
    assert spec["index"].endswith("/whl/cu124")
    assert "--only-binary" in targets
    assert "--extra-index-url" in targets


def test_install_targets_cuda_gguf_forces_prebuilt_wheel_on_windows(monkeypatch):
    monkeypatch.setattr(engines.platform, "system", lambda: "Windows")
    targets = engines._install_targets("cuda-gguf")
    assert "--only-binary" in targets
    assert "--extra-index-url" in targets


def test_install_targets_cuda_gguf_plain_on_macos(monkeypatch):
    # macOS isn't in cuda-gguf's wheel_platforms (no NVIDIA discrete GPU target on Mac).
    monkeypatch.setattr(engines.platform, "system", lambda: "Darwin")
    assert engines._install_targets("cuda-gguf") == engines.ENGINES["cuda-gguf"]["packages"]


# --------------------------------------------------------------------------- #
# engine_for_model() — cheap classifier: confident GGUF signal only
# --------------------------------------------------------------------------- #
def test_engine_for_model_gguf_file_path():
    assert engines.engine_for_model("x.gguf") == "cpu"


def test_engine_for_model_repo_colon_gguf_file():
    assert engines.engine_for_model("org/repo:Model-Q4_K_M.gguf") == "cpu"


def test_engine_for_model_bare_repo_is_none():
    assert engines.engine_for_model("org/Model") is None


def test_engine_for_model_repo_name_contains_gguf_but_no_suffix_is_none():
    # Repo *name* has GGUF in it but there's no .gguf file reference — not a confident signal.
    assert engines.engine_for_model("bartowski/SmolLM2-135M-Instruct-GGUF") is None
