#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Certify whether a candidate ``llama-cpp-python`` version actually RUNS on this platform's
prebuilt CPU wheel — the runtime check behind the ``cpu`` engine's Windows ``max_version`` cap.

Why this exists
---------------
``ara/engines.py`` caps the Windows ``cpu`` engine's ``llama-cpp-python`` at ``max_version``
(``0.3.19`` as of 2026-07-02). Post-0.3.19 Windows CPU wheels from the abetlen ``/whl/cpu`` index
ship a split / runtime-loaded ggml backend that the Python binding does not initialize — a load
fails with a "no backend"/"failed to load model" error at runtime (upstream abetlen/llama-cpp-python
#2069). The cap is therefore a *runtime* fact, not a packaging preference: a unit test can pin that
the cap is APPLIED, but only a real install-and-run on the platform can certify whether the cap can
be *raised*. This script is that certification — the evidence a ``max_version`` bump must be backed by.

What it does
------------
Installs ``llama-cpp-python==<version>`` into a throwaway env EXACTLY as ARA would for a wheel
platform (``--only-binary`` + the project ``--extra-index-url`` — never a silent source build), then
loads a tiny GGUF and generates one token. Exit 0 = certified working (safe to raise the cap to this
version); non-zero = failed (the cap must stay), with the failure text captured so you can confirm
it's the #2069 backend symptom vs. an unrelated error.

Usage (run ON the platform in question — e.g. the Windows box for a Windows cap):
    uv run python scripts/certify_llama_cpp_cpu.py --version 0.3.31 --model C:\\path\\tiny.gguf
    uv run python scripts/certify_llama_cpp_cpu.py --version 0.3.19 --model ./tiny.gguf   # baseline

``--model`` should be the smallest GGUF you have cached (e.g. SmolLM2-135M IQ3). Use ``--keep`` to
leave the throwaway env for inspection. This is a manual/live certification, NOT part of the
engine-free ``uv run pytest`` gate (it downloads a wheel and loads a model).
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile

DEFAULT_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"

# Run inside the throwaway env: load the model + generate 1 token, print a sentinel on success.
# The construction is wrapped so the REAL load error surfaces — llama-cpp-python's __del__/close
# raises a masking ``'LlamaModel' object has no attribute 'sampler'`` when __init__ fails partway,
# which would otherwise bury the actual "unsupported architecture / bad DLL" cause.
_PROBE = (
    "import sys\n"
    "from llama_cpp import Llama\n"       # ImportError here = the #2069 DLL-load failure
    "try:\n"
    "    m = Llama(sys.argv[1], n_ctx=256, verbose=False)\n"
    "except BaseException as e:\n"
    "    print('LOAD_ERROR: ' + type(e).__name__ + ': ' + str(e)[:400])\n"
    "    sys.exit(4)\n"
    "out = m('Hi', max_tokens=1)['choices'][0]['text']\n"
    "print('OUTPUT=' + repr(out))\n"
    "print('CERT_OK')\n"
)


def _venv_python(venv: str) -> str:
    sub = "Scripts" if platform.system() == "Windows" else "bin"
    exe = "python.exe" if platform.system() == "Windows" else "python"
    return os.path.join(venv, sub, exe)


def certify(version: str, model: str, index: str, keep: bool) -> int:
    if shutil.which("uv") is None:
        print("FAIL: uv not found on PATH — install uv and retry", file=sys.stderr)
        return 3
    if not os.path.exists(model):
        print(f"FAIL: model not found: {model}", file=sys.stderr)
        return 3

    workdir = tempfile.mkdtemp(prefix="llamacert_")
    venv = os.path.join(workdir, "venv")
    try:
        subprocess.run(["uv", "venv", venv], check=True, capture_output=True, text=True)
        py = _venv_python(venv)
        # Mirror ARA's wheel-platform install: forced prebuilt wheel from the project index.
        install = subprocess.run(
            ["uv", "pip", "install", "--python", py,
             "--only-binary", "llama-cpp-python", "--extra-index-url", index,
             f"llama-cpp-python=={version}"],
            capture_output=True, text=True,
        )
        if install.returncode != 0:
            print(f"FAIL: could not install llama-cpp-python=={version} from {index}")
            print((install.stderr or install.stdout)[-1500:])
            return 2
        run = subprocess.run([py, "-c", _PROBE, model], capture_output=True, text=True)
        ok = run.returncode == 0 and "CERT_OK" in run.stdout
        print(run.stdout.strip())
        if ok:
            print(f"\nCERTIFIED: llama-cpp-python=={version} loads + runs on "
                  f"{platform.system()} (CPU wheel). Safe to raise the cap to {version}.")
            return 0
        tail = (run.stderr or run.stdout)[-1500:]
        sig = tail.lower()
        # The #2069 signature is specifically a SHARED-LIBRARY load failure: the split /
        # runtime-loaded ggml backend DLL can't be dlopen'd (Windows: WinError 127 "procedure
        # could not be found"; or a "no backend" once the lib is up). A plain "failed to load
        # model" is NOT this — that's a model problem and must not be misread as a cap reason.
        dll = any(s in sig for s in (
            "failed to load shared library", "winerror 127", "procedure could not be found",
            "module could not be found", "no backend"))
        print(f"\nNOT CERTIFIED: llama-cpp-python=={version} failed to run on "
              f"{platform.system()}. Cap must stay.")
        print("  (matches the #2069 shared-library/backend symptom — the cap's exact rationale)"
              if dll else
              "  (NOTE: NOT the #2069 DLL symptom — looks like a model/other error; inspect below)")
        print(tail)
        return 1
    finally:
        if keep:
            print(f"\n(kept throwaway env: {workdir})")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True, help="llama-cpp-python version to certify, e.g. 0.3.31")
    ap.add_argument("--model", required=True, help="path to a small local .gguf to load")
    ap.add_argument("--index", default=DEFAULT_INDEX, help=f"wheel index (default {DEFAULT_INDEX})")
    ap.add_argument("--keep", action="store_true", help="keep the throwaway env for inspection")
    args = ap.parse_args(argv)
    return certify(args.version, args.model, args.index, args.keep)


if __name__ == "__main__":
    raise SystemExit(main())
