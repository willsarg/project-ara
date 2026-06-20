"""CPU/llama.cpp measurement worker — built into ARA, runs in the isolated ``cpu`` env.

The CPU engine is built into ARA (only the huge CUDA/MLX suites get their own repos), but its
heavy dep (``llama-cpp-python``) must never enter ARA core's lock. So this is a **self-contained
script**: it NEVER imports ``ara``, and it imports the engine (``llama_cpp``) only *inside*
functions — top-level imports are stdlib only, so its pure logic is unit-testable in any venv
(see tests/test_workers_cpu_llama.py). ARA core stays engine-free; this file runs under the
``cpu`` env's own python via ``engine_env.run_worker``.

It mirrors the canonical worker contract (ara/contracts/worker.py) so ARA's engine-agnostic
driver treats it identically to the Apple worker:

    preflight: {base_gb, ref_baseline_gb, slope_gb_per_k, budget_gb, max_context}
    safe:      {"context": <int>, "mem_gb": <process RSS-delta high-water, GB>}
    refused:   {"context": <int>, "refused": true, "reason": "<why>"}

Differences from the MLX/Apple worker, all physical: the wall is **system RAM** (not unified
GPU memory), the metric is the worker process's **peak RSS delta** (not os-wired), and model
facts come from **GGUF metadata** read with ``vocab_only=True`` (no weights loaded).

Usage:
    python cpu_llama.py <model> <ctx> --margin G --overhead G [--preflight]
    python cpu_llama.py --probe <gguf_path> <ctx> --abort-gb G      (internal: one child probe)

``<model>`` is a local ``*.gguf`` path, an HF repo id (smallest ``.gguf`` is picked), or
``repo:filename.gguf``.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

GIB = 1024 ** 3
DEFAULT_REPEATS = 3
KV_BYTES_F16 = 2          # llama.cpp default KV cache element size (no KV quantization)
RESIDENT_FACTOR = 1.0     # mmap'd GGUF weights fault fully resident under inference


# --------------------------------------------------------------------------- #
# Pure logic (no engine import) — unit-tested in ARA's venv.
# --------------------------------------------------------------------------- #
def kv_slope_gb_per_k(meta: dict, *, kv_bytes: int = KV_BYTES_F16) -> float:
    """GB of KV cache added per 1000 tokens, from GGUF metadata.

    KV cache = 2 (K and V) × layers × n_embd_kv × bytes, where ``n_embd_kv`` accounts for
    grouped-query attention (fewer KV heads than query heads). All values are strings in GGUF
    metadata; missing KV-head count means no GQA (KV heads == query heads).
    """
    arch = meta["general.architecture"]
    n_layers = int(meta[f"{arch}.block_count"])
    n_embd = int(meta[f"{arch}.embedding_length"])
    n_head = int(meta[f"{arch}.attention.head_count"])
    n_head_kv = int(meta.get(f"{arch}.attention.head_count_kv", n_head))
    head_dim = n_embd // n_head
    n_embd_kv = head_dim * n_head_kv
    bytes_per_token = 2 * n_layers * n_embd_kv * kv_bytes
    return bytes_per_token * 1000 / GIB


def max_context_from(meta: dict) -> int:
    """The model's trained context window from GGUF metadata."""
    arch = meta["general.architecture"]
    return int(meta[f"{arch}.context_length"])


def safe_threshold_gb(total_gb: float, margin_gb: float) -> float:
    """The safe RAM budget: physical RAM minus the policy margin (the cushion below the wall)."""
    return total_gb - margin_gb


def safety_gate(*, base_gb: float, slope_gb_per_k: float, ctx: int,
                budget_gb: float) -> str | None:
    """Refuse-before-load (L4). ``base_gb`` is the absolute footprint at ctx→0 (live RAM +
    model). Two conservative ``>=`` checks: the model must load at all, and the prediction at
    *ctx* must stay under budget. Returns a reason to refuse, or None when safe."""
    if base_gb >= budget_gb:
        return f"base estimate {base_gb:.2f}GB >= safe budget {budget_gb:.2f}GB — won't load"
    predicted = base_gb + slope_gb_per_k * (ctx / 1000)
    if predicted >= budget_gb:
        return (f"predicted {predicted:.2f}GB at {ctx} tok >= safe budget {budget_gb:.2f}GB")
    return None


# --------------------------------------------------------------------------- #
# Engine-touching helpers (import llama_cpp / psutil / huggingface_hub inside).
# --------------------------------------------------------------------------- #
def _used_gb() -> float:
    """Live system RAM in use right now (GB) — the ambient baseline."""
    import psutil

    return min(psutil.virtual_memory().used for _ in range(3)) / GIB


def _total_gb() -> float:
    import psutil

    return psutil.virtual_memory().total / GIB


def _resolve_gguf(model: str) -> str:
    """Resolve *model* to a local GGUF file path, downloading from HF if needed.

    Accepts a local ``*.gguf`` path, ``repo:filename.gguf``, or a bare HF repo id (the
    smallest ``.gguf`` sibling is chosen — typically the most aggressively quantized).
    """
    if model.endswith(".gguf") and os.path.exists(model):
        return model
    from huggingface_hub import HfApi, hf_hub_download

    if ":" in model:
        repo, _, fname = model.partition(":")
    else:
        repo = model
        files = [s for s in HfApi().model_info(repo, files_metadata=True).siblings
                 if s.rfilename.endswith(".gguf")]
        if not files:
            raise FileNotFoundError(f"no .gguf in {repo}")
        fname = min(files, key=lambda s: s.size or 1 << 62).rfilename
    return hf_hub_download(repo, fname)


def _read_meta(gguf_path: str) -> dict:
    """Read GGUF metadata without loading weights (``vocab_only=True``)."""
    from llama_cpp import Llama

    llm = Llama(model_path=gguf_path, vocab_only=True, verbose=False)
    return dict(llm.metadata)


def _probe(gguf_path: str, ctx: int, abort_gb: float) -> dict:
    """Load the model at *ctx* in this (fresh child) process, return its RSS-delta footprint.

    A watchdog thread (L5) aborts the process if live system RAM reaches *abort_gb* mid-load,
    so an under-estimate can never run the machine out of memory. Returns the process RSS
    high-water minus its pre-load baseline — the model's marginal footprint at this context.
    """
    import threading

    import psutil

    proc = psutil.Process()
    baseline = proc.memory_info().rss / GIB
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(0.05):
            if psutil.virtual_memory().used / GIB >= abort_gb:
                os._exit(3)        # L5: hard abort before the wall

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    try:
        from llama_cpp import Llama

        # n_ctx allocates the full KV cache up front, so this footprint IS the cost at ctx.
        llm = Llama(model_path=gguf_path, n_ctx=ctx, verbose=False)
        llm.eval([llm.token_bos()])         # fault weights resident + touch the KV cache
        peak = proc.memory_info().rss / GIB
        return {"status": "ok", "delta_gb": round(peak - baseline, 4)}
    except Exception as e:                   # OOM, unsupported model, llama.cpp error
        return {"status": "error", "note": str(e)}
    finally:
        stop.set()


def _run_probe_child(gguf_path: str, ctx: int, abort_gb: float) -> dict:
    """Run one ``--probe`` in a fresh child process (clean RSS baseline per repeat)."""
    cmd = [sys.executable, os.path.abspath(__file__), "--probe", gguf_path, str(ctx),
           "--abort-gb", str(abort_gb)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    line = next((ln for ln in out.stdout.splitlines() if ln.lstrip().startswith("{")), None)
    if line is None:
        code = out.returncode
        note = "aborted at memory wall (L5)" if code == 3 else (out.stderr.strip() or "no output")
        return {"status": "error", "note": note}
    return json.loads(line)


# --------------------------------------------------------------------------- #
# Contract entry points.
# --------------------------------------------------------------------------- #
def _model_base_gb(gguf_path: str, overhead_gb: float) -> float:
    """Model's resident footprint at ctx→0: GGUF weights (mmap, ~fully resident) + overhead."""
    weights_gb = os.path.getsize(gguf_path) / GIB
    return weights_gb * RESIDENT_FACTOR + overhead_gb


def preflight(model: str, *, margin_gb: float, overhead_gb: float) -> dict:
    """No-load estimate for ARA's scheduler: absolute base, a-priori slope, budget, window."""
    try:
        gguf = _resolve_gguf(model)
        meta = _read_meta(gguf)
    except Exception as e:
        return {"error": str(e)}
    live_base = _used_gb()
    model_base = _model_base_gb(gguf, overhead_gb)
    return {
        "base_gb": round(live_base + model_base, 4),    # absolute, for ARA's a-priori gate
        "ref_baseline_gb": round(live_base, 4),         # live RAM, added back at solve time
        "slope_gb_per_k": kv_slope_gb_per_k(meta),
        "budget_gb": safe_threshold_gb(_total_gb(), margin_gb),
        "max_context": max_context_from(meta),
    }


def _refused(ctx: int, reason: str) -> dict:
    return {"context": ctx, "refused": True, "reason": reason}


def run(model: str, ctx: int, *, margin_gb: float, overhead_gb: float,
        repeats: int = DEFAULT_REPEATS) -> dict:
    """Gate (L4) then, if safe, measure the RSS-delta footprint at *ctx* (median of repeats)."""
    try:
        gguf = _resolve_gguf(model)
        meta = _read_meta(gguf)
    except Exception as e:
        return _refused(ctx, str(e))
    budget = safe_threshold_gb(_total_gb(), margin_gb)
    base_gb = _used_gb() + _model_base_gb(gguf, overhead_gb)
    reason = safety_gate(base_gb=base_gb, slope_gb_per_k=kv_slope_gb_per_k(meta),
                         ctx=ctx, budget_gb=budget)
    if reason is not None:
        return _refused(ctx, reason)
    deltas = []
    for _ in range(max(1, repeats)):
        raw = _run_probe_child(gguf, ctx, budget)
        if raw.get("status") != "ok":
            return _refused(ctx, f"probe failed: {raw.get('note', 'no output')}")
        deltas.append(raw["delta_gb"])
    return {"context": ctx, "mem_gb": round(statistics.median(deltas), 3)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Safe single-context CPU memory measurement.")
    ap.add_argument("model", help="local .gguf, HF repo id, or repo:filename.gguf")
    ap.add_argument("ctx", type=int)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--overhead", type=float, default=1.0)
    ap.add_argument("--preflight", action="store_true",
                    help="print the no-load estimate and exit")
    ap.add_argument("--probe", action="store_true",
                    help="internal: load once and print this process's RSS-delta footprint")
    ap.add_argument("--abort-gb", type=float, default=None,
                    help="internal: L5 watchdog wall for --probe")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = ap.parse_args(argv)

    if args.probe:
        result = _probe(args.model, args.ctx, args.abort_gb)
    elif args.preflight:
        result = preflight(args.model, margin_gb=args.margin, overhead_gb=args.overhead)
    else:
        result = run(args.model, args.ctx, margin_gb=args.margin,
                     overhead_gb=args.overhead, repeats=args.repeats)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
