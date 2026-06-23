# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""status.py — live AI/ML process recon: classification + scan."""
from __future__ import annotations

import os
import types

import psutil
import pytest

import ara.status as status
from ara.status import GB, Proc, _classify, _detail, _short


# --------------------------------------------------------------------------- #
# _classify
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,cmd,expected", [
    ("ollama", "ollama serve", "Ollama"),
    ("lms", "/Applications/LM Studio.app/lmstudio", "LM Studio"),
    ("llama-server", "llama-server -m model.gguf", "llama.cpp"),
    ("python3.12", "python -m vllm.entrypoints.openai", "vLLM"),
    ("python", "python -m mlx_lm.server", "MLX (mlx_lm)"),
    ("text-generation", "text-generation-inference", "TGI"),
    ("python", "python -m sglang.launch_server", "SGLang"),
    ("python", "main.py --comfyui", "ComfyUI"),
    ("python", "launch.py automatic1111", "Stable Diffusion"),
    ("jupyter-lab", "jupyter-lab", "Jupyter"),
    ("python", "python train.py --use torch", "Python ML"),
])
def test_classify_matches(name, cmd, expected):
    assert _classify(name, cmd) == expected


def test_classify_matches_by_name_only():
    # name matches the rule but the cmdline doesn't → pins the `or` (not `and`).
    assert _classify("ollama", "serve --port 11434") == "Ollama"


def test_classify_python_ml_by_name_only():
    # "python" appears in the process name but not the cmdline; an ML token is on cmd.
    assert _classify("python3.12", "train.py --backend torch") == "Python ML"


def test_classify_returns_none_for_unrelated():
    assert _classify("bash", "bash -c ls") is None
    assert _classify("python", "python manage.py runserver") is None


def test_classify_specific_engine_wins_over_generic_python():
    # cmdline names both vllm and torch; the specific vLLM rule precedes Python ML.
    assert _classify("python", "python -m vllm --model x torch") == "vLLM"


# --------------------------------------------------------------------------- #
# _short / _detail
# --------------------------------------------------------------------------- #
def test_short_path_basename():
    assert _short("/models/llama/model.gguf") == "model.gguf"
    assert _short("~/weights/foo.safetensors") == "foo.safetensors"


def test_short_hf_id_keeps_org_and_name():
    assert _short("mlx-community/SmolLM-135M-Instruct-4bit") == "mlx-community/SmolLM-135M-Instruct-4bit"
    assert _short("a/b/c/d") == "c/d"


def test_short_plain_token():
    assert _short("llama3") == "llama3"


def test_detail_flag_with_separate_value():
    assert _detail(["llama-server", "--model", "/m/foo.gguf"]) == "foo.gguf"
    assert _detail(["x", "-m", "org/repo"]) == "org/repo"


def test_detail_flag_equals_form():
    assert _detail(["x", "--model=mlx-community/Model"]) == "mlx-community/Model"


def test_detail_weights_path_fallback():
    assert _detail(["python", "run.py", "/data/weights.safetensors"]) == "weights.safetensors"


def test_detail_none_when_no_hint():
    assert _detail(["python", "server.py"]) is None


def test_detail_flag_as_last_token_is_safe():
    # boundary: `--model` with nothing after it must not index past the end → no hint.
    assert _detail(["llama-server", "--model"]) is None


def test_detail_dash_m_collides_with_python_module_flag():
    # Known quirk: `-m` (model shorthand) also matches python's module flag, so
    # `python -m vllm ...` reports "vllm" as the model. Pinned to document it.
    assert _detail(["python", "-m", "vllm", "--model", "x"]) == "vllm"


# --------------------------------------------------------------------------- #
# nvidia per-pid gpu memory
# --------------------------------------------------------------------------- #
def test_nvidia_gpu_by_pid_parses(monkeypatch):
    monkeypatch.setattr(status.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        stdout="1234, 512\n5678, 2048\n"))
    assert status._nvidia_gpu_by_pid() == {1234: 512.0, 5678: 2048.0}


def test_nvidia_gpu_by_pid_empty_on_failure(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no nvidia-smi")
    monkeypatch.setattr(status.subprocess, "run", boom)
    assert status._nvidia_gpu_by_pid() == {}


def test_nvidia_gpu_by_pid_skips_garbage_lines(monkeypatch):
    monkeypatch.setattr(status.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        stdout="header junk\n1234, 512\n\nbad, line\n"))
    assert status._nvidia_gpu_by_pid() == {1234: 512.0}


def test_nvidia_gpu_by_pid_skips_non_numeric_memory(monkeypatch):
    # pid is a digit but memory isn't a float → that row is dropped, not fatal.
    monkeypatch.setattr(status.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        stdout="1234, [N/A]\n5678, 256\n"))
    assert status._nvidia_gpu_by_pid() == {5678: 256.0}


# --------------------------------------------------------------------------- #
# _listen_port
# --------------------------------------------------------------------------- #
def test_listen_port_returns_first_listening(monkeypatch):
    conn = types.SimpleNamespace(
        status=psutil.CONN_LISTEN, laddr=types.SimpleNamespace(port=11434))
    proc = types.SimpleNamespace(net_connections=lambda kind: [conn])
    assert status._listen_port(proc) == 11434


def test_listen_port_none_when_not_listening(monkeypatch):
    conn = types.SimpleNamespace(
        status=psutil.CONN_ESTABLISHED, laddr=types.SimpleNamespace(port=80))
    proc = types.SimpleNamespace(net_connections=lambda kind: [conn])
    assert status._listen_port(proc) is None


def test_listen_port_handles_access_denied(monkeypatch):
    def denied(kind):
        raise psutil.AccessDenied()
    proc = types.SimpleNamespace(net_connections=denied)
    assert status._listen_port(proc) is None


# --------------------------------------------------------------------------- #
# scan()
# --------------------------------------------------------------------------- #
class FakeProc:
    def __init__(self, info, raise_on_info=False):
        self._info = info
        self._raise = raise_on_info

    @property
    def info(self):
        if self._raise:
            raise psutil.NoSuchProcess(self._info.get("pid", 0))
        return self._info


def _proc_info(pid, name, cmdline, rss_gb=1.0, create_time=None):
    return {
        "pid": pid,
        "name": name,
        "cmdline": cmdline,
        "memory_info": types.SimpleNamespace(rss=int(rss_gb * GB)),
        "create_time": create_time,
    }


def test_scan_finds_and_sorts_by_memory(monkeypatch):
    monkeypatch.setattr(status, "_nvidia_gpu_by_pid", lambda: {})
    procs = [
        FakeProc(_proc_info(1001, "ollama", ["ollama", "serve"], rss_gb=1.0)),
        FakeProc(_proc_info(1002, "python", ["vllm", "serve", "--model", "x"], rss_gb=4.0)),
        FakeProc(_proc_info(1003, "bash", ["bash"], rss_gb=9.0)),  # not ML → excluded
    ]
    monkeypatch.setattr(status.psutil, "process_iter", lambda fields: iter(procs))

    found = status.scan()
    assert [p.label for p in found] == ["vLLM", "Ollama"]  # 4 GB before 1 GB
    assert all(isinstance(p, Proc) for p in found)
    assert found[0].detail == "x"


def test_scan_skips_self_and_parent(monkeypatch):
    monkeypatch.setattr(status, "_nvidia_gpu_by_pid", lambda: {})
    me, parent = os.getpid(), os.getppid()
    procs = [
        FakeProc(_proc_info(me, "ollama", ["ollama"])),
        FakeProc(_proc_info(parent, "ollama", ["ollama"])),
        FakeProc(_proc_info(2002, "ollama", ["ollama"])),
    ]
    monkeypatch.setattr(status.psutil, "process_iter", lambda fields: iter(procs))
    found = status.scan()
    assert [p.pid for p in found] == [2002]


def test_scan_survives_dead_process(monkeypatch):
    monkeypatch.setattr(status, "_nvidia_gpu_by_pid", lambda: {})
    procs = [
        FakeProc(_proc_info(3001, "ollama", ["ollama"])),
        FakeProc(_proc_info(3002, "x", []), raise_on_info=True),
    ]
    monkeypatch.setattr(status.psutil, "process_iter", lambda fields: iter(procs))
    found = status.scan()
    assert [p.pid for p in found] == [3001]


def test_scan_attaches_gpu_memory(monkeypatch):
    monkeypatch.setattr(status, "_nvidia_gpu_by_pid", lambda: {4001: 8192.0})
    procs = [FakeProc(_proc_info(4001, "python", ["python", "-m", "vllm"]))]
    monkeypatch.setattr(status.psutil, "process_iter", lambda fields: iter(procs))
    found = status.scan()
    assert found[0].gpu_mb == 8192.0


# --------------------------------------------------------------------------- #
# _classify_app — AI *client* apps (assistant + coding), distinct from workloads
# --------------------------------------------------------------------------- #
def test_classify_app_gui_bundle_match():
    # GUI clients match their .app bundle path (reused from the apps catalog).
    assert status._classify_app(
        "ChatGPT", "/Applications/ChatGPT.app/Contents/MacOS/ChatGPT") == "ChatGPT"
    assert status._classify_app(
        "Cursor", "/Applications/Cursor.app/Contents/MacOS/Cursor") == "Cursor"


def test_classify_app_cli_basename_match():
    # terminal client: matched by exact process basename, no .app in the cmdline.
    assert status._classify_app("claude", "claude --resume abc123") == "Claude Code"


def test_classify_app_gui_wins_over_cli_collision():
    # Claude.app's main exec basename is "Claude" → collides with the `claude` CLI.
    # The .app path must win so Claude Desktop isn't mislabeled "Claude Code".
    assert status._classify_app(
        "Claude", "/Applications/Claude.app/Contents/MacOS/Claude") == "Claude"


def test_classify_app_cli_ignored_inside_app_bundle():
    # a `claude`-named process living inside some unrelated .app must NOT be taken as the
    # CLI client (the .app guard), and it isn't a known client bundle → None.
    assert status._classify_app("claude", "/Applications/SomeEditor.app/bin/claude") is None


def test_classify_app_excludes_runners_and_unrelated():
    # Ollama is a 'runner' (a local workload), not a client app → not in this section.
    assert status._classify_app(
        "Ollama", "/Applications/Ollama.app/Contents/MacOS/Ollama") is None
    assert status._classify_app("bash", "bash -c ls") is None


# --------------------------------------------------------------------------- #
# scan_apps — running AI client apps, one entry per app
# --------------------------------------------------------------------------- #
def _app_info(pid, name, cmdline, rss_gb=0.5, create_time=None):
    return {
        "pid": pid, "name": name, "cmdline": cmdline,
        "memory_info": types.SimpleNamespace(rss=int(rss_gb * GB)),
        "create_time": create_time,
    }


def test_scan_apps_collapses_multiprocess_app(monkeypatch):
    monkeypatch.setattr(status.time, "time", lambda: 1000.0)
    base = "/Applications/Claude.app/Contents"
    helper = f"{base}/Frameworks/Claude Helper.app/Contents/MacOS/Claude Helper"
    procs = [
        FakeProc(_app_info(1, "Claude", [f"{base}/MacOS/Claude"], rss_gb=0.4, create_time=100.0)),
        FakeProc(_app_info(2, "Claude Helper", [helper, "--type=renderer"], rss_gb=0.3, create_time=200.0)),
        FakeProc(_app_info(3, "Claude Helper", [helper, "--type=gpu-process"], rss_gb=0.2, create_time=150.0)),
    ]
    monkeypatch.setattr(status.psutil, "process_iter", lambda fields: iter(procs))
    found = status.scan_apps()
    assert len(found) == 1                       # 8-helper Electron app -> one line
    assert found[0].label == "Claude" and found[0].n_procs == 3
    assert found[0].rss_gb == 0.9                # summed RSS
    assert found[0].uptime_s == 900.0            # 1000 - oldest create_time (100)


def test_scan_apps_sorts_by_memory_and_includes_cli(monkeypatch):
    monkeypatch.setattr(status.time, "time", lambda: 1000.0)
    procs = [
        FakeProc(_app_info(10, "claude", ["claude", "--resume", "x"], rss_gb=0.2, create_time=500.0)),
        FakeProc(_app_info(11, "ChatGPT", ["/Applications/ChatGPT.app/Contents/MacOS/ChatGPT"], rss_gb=1.5, create_time=500.0)),
        FakeProc(_app_info(12, "bash", ["bash"], rss_gb=9.0)),  # unrelated -> excluded
    ]
    monkeypatch.setattr(status.psutil, "process_iter", lambda fields: iter(procs))
    found = status.scan_apps()
    assert [a.label for a in found] == ["ChatGPT", "Claude Code"]   # 1.5 GB before 0.2 GB
    assert all(isinstance(a, status.AppProc) for a in found)


def test_scan_apps_skips_self_and_parent(monkeypatch):
    monkeypatch.setattr(status.time, "time", lambda: 1000.0)
    me, parent = os.getpid(), os.getppid()
    cc = ["/Applications/ChatGPT.app/Contents/MacOS/ChatGPT"]
    procs = [
        FakeProc(_app_info(me, "ChatGPT", cc)),
        FakeProc(_app_info(parent, "ChatGPT", cc)),
        FakeProc(_app_info(7777, "ChatGPT", cc, create_time=500.0)),
    ]
    monkeypatch.setattr(status.psutil, "process_iter", lambda fields: iter(procs))
    found = status.scan_apps()
    assert len(found) == 1 and found[0].n_procs == 1     # only pid 7777


def test_scan_apps_survives_dead_process(monkeypatch):
    monkeypatch.setattr(status.time, "time", lambda: 1000.0)
    procs = [
        FakeProc(_app_info(8001, "ChatGPT", ["/Applications/ChatGPT.app/Contents/MacOS/ChatGPT"], create_time=500.0)),
        FakeProc(_app_info(8002, "x", []), raise_on_info=True),
    ]
    monkeypatch.setattr(status.psutil, "process_iter", lambda fields: iter(procs))
    found = status.scan_apps()
    assert [a.label for a in found] == ["ChatGPT"]


def test_scan_apps_handles_missing_mem_and_time(monkeypatch):
    monkeypatch.setattr(status.time, "time", lambda: 1000.0)
    info = {"pid": 9001, "name": "Cursor",
            "cmdline": ["/Applications/Cursor.app/Contents/MacOS/Cursor"],
            "memory_info": None, "create_time": None}
    monkeypatch.setattr(status.psutil, "process_iter", lambda fields: iter([FakeProc(info)]))
    found = status.scan_apps()
    assert found[0].label == "Cursor"
    assert found[0].rss_gb == 0.0          # no memory_info -> 0
    assert found[0].uptime_s == 0.0        # no create_time -> now -> 0
