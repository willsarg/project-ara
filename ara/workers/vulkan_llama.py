# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Vulkan/llama.cpp measurement worker — built into ARA, runs in the isolated ``vulkan`` env.

The Vulkan engine runs GGUF models on a GPU (an AMD APU's integrated Radeon today) via
llama.cpp's Vulkan backend — the same llama.cpp the ``cpu`` engine uses, just offloaded
(``n_gpu_layers=-1``). Like the CPU worker it's **built into ARA** (only the huge CUDA/MLX
suites get their own repos), so it's a **self-contained script**: it NEVER imports ``ara``, and
it imports the engine (``llama_cpp``) only *inside* functions — top-level imports are stdlib
only, so its pure logic is unit-testable in any venv (see tests/test_workers_vulkan_llama.py).
ARA core stays engine-free; this runs under the ``vulkan`` env's own python via
``engine_env.run_worker``.

It mirrors the canonical worker contract (ara/contracts/worker.py) so ARA's engine-agnostic
driver treats it identically to the CPU/Apple workers:

    preflight: {base_gb, ref_baseline_gb, slope_gb_per_k, budget_gb, max_context}
    safe:      {"context": <int>, "mem_gb": <GPU+CPU footprint delta, GB>}
    refused:   {"context": <int>, "refused": true, "reason": "<why>"}

Three differences from ``cpu_llama.py``, all physical to a shared-memory APU:

1. **Offload.** Every probe loads with ``n_gpu_layers=-1`` (all layers on the GPU).
2. **Memory metric = amdgpu GTT sysfs delta, not process RSS.** GPU allocations page through
   the GTT pool (carved from system RAM) and do NOT appear in the worker's RSS. The footprint is
   ``Δ(GTT+VRAM used)`` from ``/sys/class/drm/card*/device/mem_info_*`` plus the CPU-side
   ``Δ(RSS)`` (llama.cpp still keeps a small ``CPU_Mapped`` buffer). The wall is still **system
   RAM** (the GPU's memory is carved from it — GPU OOM is whole-box OOM), so the L4/L5 gates and
   budget reuse the CPU arithmetic verbatim.
3. **Honest offload check (Rule #3).** The parent captures the child's stderr and parses
   llama.cpp's ``offloaded N/M layers to GPU`` + Vulkan device line. If nothing offloaded (e.g.
   a CPU-only wheel got installed, or only a software rasterizer is present), the probe **refuses**
   rather than silently measuring a CPU run as if it were the GPU.

Usage:
    python vulkan_llama.py <model> <ctx> --margin G --overhead G [--preflight]
    python vulkan_llama.py --probe <gguf_path> <ctx> --abort-gb G      (internal: one child probe)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys

GIB = 1024 ** 3
DEFAULT_REPEATS = 3
KV_BYTES_F16 = 2          # llama.cpp default KV cache element size (no KV quantization)
RESIDENT_FACTOR = 1.0     # GGUF weights are fully resident once offloaded (GTT-backed)

# KV-cache quantization → the ggml type id passed to Llama(type_k/type_v). Symmetric only (K==V):
# asymmetric types drop off the fused-FA fast path and hit an offload bug on Vulkan. q8_0 is the
# near-lossless ~half-KV sweet spot; q4_0 is ~quarter-KV at a real quality cost. ids match
# llama_cpp.GGML_TYPE_{F16,Q8_0,Q4_0} (verified on the engine wheel). Quantized KV requires FA.
_KV_GGML_TYPE = {"f16": 1, "q8_0": 8, "q4_0": 2}

# Effective bytes per KV element by quant type, for the a-priori memory slope. The L1/L4 gate must
# predict the SAME memory the quantized cache actually uses — otherwise it refuses contexts that
# fit and KV-quant buys no extra ceiling (caught on rog-ubuntu). f16=2; q8_0=34B/32 (int8 + one
# f16 scale per 32-block); q4_0=18B/32 (int4 + one f16 scale). Measurement (L2/L5) still backstops
# these against the real wall, so a small inaccuracy can't breach safety.
_KV_BYTES = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}

# amdgpu exposes live memory accounting here, readable WITHOUT root. Module-level so tests can
# point it at a fixture tree. GTT is the pool weights/KV/compute buffers actually land in on a
# RADV APU; VRAM is the small BIOS carveout (usually barely moves) — sum both to be safe.
DRM_DEVICE_GLOB = "/sys/class/drm/card*/device"


# --------------------------------------------------------------------------- #
# Pure logic (no engine import) — unit-tested in ARA's venv.
# --------------------------------------------------------------------------- #
def kv_slope_gb_per_k(meta: dict, *, kv_bytes: int = KV_BYTES_F16) -> float:
    """GB of KV cache added per 1000 tokens, from GGUF metadata (identical to the CPU worker —
    KV lives in the same shared pool whether the math runs on CPU or GPU)."""
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


def effective_margin_gb(total_gb: float, cap_gb: float) -> float:
    """Safety margin to actually use: ~10% of RAM, capped at *cap_gb*, floored at 0.5 GB (the
    small shared-memory APUs this targets — e.g. ~11 GB — would be zeroed by a flat 2 GB cap)."""
    return min(cap_gb, max(0.5, total_gb * 0.1))


def safe_threshold_gb(total_gb: float, margin_gb: float) -> float:
    """Safe budget: physical RAM minus the margin, clamped at 0."""
    return max(0.0, total_gb - margin_gb)


def limits_from(total_gb: float, used_gb: float, swap_free_gb: float, device: str,
                margin_gb: float) -> dict:
    """The memory wall + safe budget as a plain dict. The wall is shared **system RAM** (the GPU's
    memory is carved from it), so it's read exactly — like the CPU engine, not Apple's hidden
    cold-start overhead. *device* is the GPU/runtime label for display."""
    safe = safe_threshold_gb(total_gb, margin_gb)
    return {
        "device": device,
        "total_gb": round(total_gb, 3),
        "wall_gb": round(total_gb, 3),
        "safe_budget_gb": round(safe, 3),
        "margin_gb": round(margin_gb, 3),
        "headroom_gb": round(max(0.0, safe - used_gb), 3),
        "swap_free_gb": round(swap_free_gb, 3),
    }


def safety_gate(*, base_gb: float, slope_gb_per_k: float, ctx: int,
                budget_gb: float) -> str | None:
    """Refuse-before-load (L4): two conservative ``>=`` checks against the shared budget."""
    if base_gb >= budget_gb:
        return f"base estimate {base_gb:.2f}GB >= safe budget {budget_gb:.2f}GB — won't load"
    predicted = base_gb + slope_gb_per_k * (ctx / 1000)
    if predicted >= budget_gb:
        return (f"predicted {predicted:.2f}GB at {ctx} tok >= safe budget {budget_gb:.2f}GB")
    return None


def parse_offloaded(stderr: str) -> tuple[int, int] | None:
    """Parse llama.cpp's ``load_tensors: offloaded N/M layers to GPU`` → (N, M), or None.

    Absence means no GPU offload happened (a CPU-only build prints no such line) — the caller
    treats that as 'Vulkan not active' and refuses (Rule #3)."""
    m = re.search(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU", stderr)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_vulkan_device(stderr: str) -> dict | None:
    """Parse the ``ggml_vulkan: 0 = <name> (radv) | ... | matrix cores: <coopmat>`` line into
    ``{name, coopmat}`` (best-effort; coopmat None if absent), or None if no Vulkan device line."""
    m = re.search(r"ggml_vulkan:\s*\d+\s*=\s*(.+?)\s*(?:\||$)", stderr)
    if not m:
        return None
    name = m.group(1).strip()
    cm = re.search(r"matrix cores:\s*(\S+)", stderr)
    return {"name": name, "coopmat": cm.group(1) if cm else None}


def offload_ok(device: dict | None, offloaded: tuple[int, int] | None) -> str | None:
    """Return a refusal reason if Vulkan offload did NOT actually happen, else None.

    Honest guard (Rule #3): a CPU-only wheel (no Vulkan device / no offload line) or a software
    rasterizer being selected must never be reported as a GPU run."""
    if offloaded is None or offloaded[0] == 0:
        return "Vulkan offload not active (model ran on CPU — is this a Vulkan llama.cpp build?)"
    if device is not None and "llvmpipe" in device["name"].lower():
        return f"Vulkan selected a software rasterizer ({device['name']}), not a GPU"
    return None


# --------------------------------------------------------------------------- #
# Engine-touching helpers (import llama_cpp / psutil / huggingface_hub inside).
# --------------------------------------------------------------------------- #
def _used_gb() -> float:
    """Live system RAM in use right now (GB) — the ambient baseline (GTT counts here too).

    Takes the MAX of a few reads (Rule #1): the baseline is added to the model footprint and
    checked against the safe budget, so under-reporting it over-states headroom — a crash-wall
    trap. The conservative read is the highest sample, never the lowest.
    """
    import psutil

    return max(psutil.virtual_memory().used for _ in range(3)) / GIB


def _total_gb() -> float:
    import psutil

    return psutil.virtual_memory().total / GIB


def _gpu_used_gb() -> float:
    """Live GPU memory in use (GB): Δ-able GTT + VRAM 'used' summed across amdgpu cards.

    Reads ``mem_info_gtt_used`` + ``mem_info_vram_used`` from sysfs (no root). Returns 0.0 if the
    files are absent (non-amdgpu host) — the RSS delta still captures CPU-side buffers, and the
    offload check independently refuses a non-GPU run, so a 0 here can't mask a CPU fallback."""
    total = 0
    for dev in glob.glob(DRM_DEVICE_GLOB):
        for kind in ("mem_info_gtt_used", "mem_info_vram_used"):
            try:
                with open(os.path.join(dev, kind)) as fh:
                    total += int(fh.read().strip())
            except (OSError, ValueError):
                pass
    return total / GIB


def _resolve_gguf(model: str) -> str:
    """Resolve *model* to a local GGUF path, downloading from HF if needed (same rules as the CPU
    worker: a ``*.gguf`` path, ``repo:filename.gguf``, or a bare repo id → smallest ``.gguf``)."""
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


def _probe(gguf_path: str, ctx: int, abort_gb: float, flash_attn: bool = True,
           kv_quant: str = "f16") -> dict:
    """Load *gguf_path* at *ctx* fully offloaded to the GPU; return its footprint.

    Footprint = Δ(GPU used, from GTT/VRAM sysfs) + Δ(process RSS) — the total system memory the
    load consumed, since the GPU pool is carved from the same RAM. A watchdog (L5) aborts if live
    system RAM reaches *abort_gb* mid-load. ``verbose=True`` so llama.cpp logs the offload/device
    lines to **stderr**, which the parent captures to verify offload actually happened.
    """
    if abort_gb is None:
        return {"status": "error", "note": "refusing to probe without an L5 abort limit"}
    if kv_quant != "f16":
        flash_attn = True            # quantized KV cache requires flash-attention

    import threading

    import psutil

    proc = psutil.Process()
    rss0 = proc.memory_info().rss / GIB
    gpu0 = _gpu_used_gb()
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(0.05):
            if psutil.virtual_memory().used / GIB >= abort_gb:
                os._exit(3)        # L5: hard abort before the wall (GPU OOM == whole-box OOM)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    try:
        from llama_cpp import Llama

        kv = ({} if kv_quant == "f16"
              else {"type_k": _KV_GGML_TYPE[kv_quant], "type_v": _KV_GGML_TYPE[kv_quant]})
        llm = Llama(model_path=gguf_path, n_ctx=ctx, n_gpu_layers=-1,
                    flash_attn=flash_attn, verbose=True, **kv)
        llm.eval([llm.token_bos()])         # fault weights resident + touch the KV cache
        gpu_delta = max(0.0, _gpu_used_gb() - gpu0)
        rss_delta = max(0.0, proc.memory_info().rss / GIB - rss0)
        return {"status": "ok", "delta_gb": round(gpu_delta + rss_delta, 4),
                "gpu_delta_gb": round(gpu_delta, 4), "rss_delta_gb": round(rss_delta, 4)}
    except Exception as e:                   # OOM, unsupported model, llama.cpp error
        return {"status": "error", "note": str(e)}
    finally:
        stop.set()


def _run_probe_child(gguf_path: str, ctx: int, abort_gb: float,
                     flash_attn: bool = True, kv_quant: str = "f16") -> dict:
    """Run one ``--probe`` child (clean baseline per repeat) and capture its stderr.

    The child's stderr carries llama.cpp's device + offload logs (C-level fd-2 writes, captured
    reliably via the subprocess pipe). The parent parses them and **refuses** if offload didn't
    happen, attaching the observed Vulkan device fact (Rule #3)."""
    cmd = [sys.executable, os.path.abspath(__file__), "--probe", gguf_path, str(ctx),
           "--abort-gb", str(abort_gb)]
    if not flash_attn:
        cmd.append("--no-flash-attn")
    if kv_quant != "f16":
        cmd += ["--kv-quant", kv_quant]
    out = subprocess.run(cmd, capture_output=True, text=True)
    line = next((ln for ln in out.stdout.splitlines() if ln.lstrip().startswith("{")), None)
    if line is None:
        code = out.returncode
        note = "aborted at memory wall (L5)" if code == 3 else (out.stderr.strip() or "no output")
        return {"status": "error", "note": note}
    result = json.loads(line)
    if result.get("status") == "ok":
        device = parse_vulkan_device(out.stderr)
        offloaded = parse_offloaded(out.stderr)
        bad = offload_ok(device, offloaded)
        if bad is not None:
            return {"status": "error", "note": bad}
        result["vulkan_device"] = device
        result["offloaded"] = offloaded
    return result


# --------------------------------------------------------------------------- #
# Contract entry points (identical shape to cpu_llama.py — the adapter is a twin).
# --------------------------------------------------------------------------- #
def _model_base_gb(gguf_path: str, overhead_gb: float) -> float:
    """Model's resident footprint at ctx→0: GGUF weights (GTT-resident) + overhead. Weights count
    against the shared wall whether they sit in GTT or RAM."""
    weights_gb = os.path.getsize(gguf_path) / GIB
    return weights_gb * RESIDENT_FACTOR + overhead_gb


def preflight(model: str, *, margin_gb: float, overhead_gb: float,
              kv_quant: str = "f16") -> dict:
    """No-load estimate for ARA's scheduler: absolute base, a-priori slope, budget, window. The
    slope is KV-quant-aware so the a-priori gate predicts the cache size actually in use."""
    try:
        gguf = _resolve_gguf(model)
        meta = _read_meta(gguf)
    except Exception as e:
        return {"error": str(e)}
    live_base = _used_gb()
    model_base = _model_base_gb(gguf, overhead_gb)
    total = _total_gb()
    return {
        "base_gb": round(live_base + model_base, 4),
        "ref_baseline_gb": round(live_base, 4),
        "slope_gb_per_k": kv_slope_gb_per_k(meta, kv_bytes=_KV_BYTES[kv_quant]),
        "budget_gb": safe_threshold_gb(total, effective_margin_gb(total, margin_gb)),
        "max_context": max_context_from(meta),
    }


def limits(*, margin_gb: float) -> dict:
    """The memory wall + safe budget, read live (system RAM, swap, GPU device label)."""
    import platform

    import psutil

    total = _total_gb()
    return limits_from(
        total_gb=total,
        used_gb=_used_gb(),
        swap_free_gb=psutil.swap_memory().free / GIB,
        device=platform.processor() or platform.machine() or "GPU (Vulkan)",
        margin_gb=effective_margin_gb(total, margin_gb),
    )


def _refused(ctx: int, reason: str) -> dict:
    return {"context": ctx, "refused": True, "reason": reason}


def run(model: str, ctx: int, *, margin_gb: float, overhead_gb: float,
        flash_attn: bool = True, kv_quant: str = "f16",
        repeats: int = DEFAULT_REPEATS) -> dict:
    """Gate (L4) then, if safe, measure the GPU+CPU footprint at *ctx* (median of repeats)."""
    try:
        gguf = _resolve_gguf(model)
        meta = _read_meta(gguf)
    except Exception as e:
        return _refused(ctx, str(e))
    total = _total_gb()
    budget = safe_threshold_gb(total, effective_margin_gb(total, margin_gb))
    base_gb = _used_gb() + _model_base_gb(gguf, overhead_gb)
    reason = safety_gate(base_gb=base_gb,
                         slope_gb_per_k=kv_slope_gb_per_k(meta, kv_bytes=_KV_BYTES[kv_quant]),
                         ctx=ctx, budget_gb=budget)
    if reason is not None:
        return _refused(ctx, reason)
    deltas = []
    for _ in range(max(1, repeats)):
        raw = _run_probe_child(gguf, ctx, budget, flash_attn, kv_quant)
        if raw.get("status") != "ok":
            return _refused(ctx, f"probe failed: {raw.get('note', 'no output')}")
        deltas.append(raw["delta_gb"])
    return {"context": ctx, "mem_gb": round(statistics.median(deltas), 3)}


def generate(model: str, ctx: int, prompt: str, *, margin_gb: float, overhead_gb: float,
             max_tokens: int, flash_attn: bool = True, kv_quant: str = "f16") -> dict:
    """Gate (L4) then, if safe, load fully offloaded with the KV cache capped at *ctx* (the
    governed safe ceiling) and return a one-shot completion. ``ctx`` is ARA's characterized
    ceiling, so generation never allocates past it."""
    try:
        gguf = _resolve_gguf(model)
        meta = _read_meta(gguf)
    except Exception as e:
        return _refused(ctx, str(e))
    total = _total_gb()
    budget = safe_threshold_gb(total, effective_margin_gb(total, margin_gb))
    base_gb = _used_gb() + _model_base_gb(gguf, overhead_gb)
    reason = safety_gate(base_gb=base_gb,
                         slope_gb_per_k=kv_slope_gb_per_k(meta, kv_bytes=_KV_BYTES[kv_quant]),
                         ctx=ctx, budget_gb=budget)
    if reason is not None:
        return _refused(ctx, reason)
    if kv_quant != "f16":
        flash_attn = True            # quantized KV cache requires flash-attention
    from llama_cpp import Llama

    kv = ({} if kv_quant == "f16"
          else {"type_k": _KV_GGML_TYPE[kv_quant], "type_v": _KV_GGML_TYPE[kv_quant]})
    llm = Llama(model_path=gguf, n_ctx=ctx, n_gpu_layers=-1,
                flash_attn=flash_attn, verbose=False, **kv)
    out = llm.create_completion(prompt, max_tokens=max_tokens)
    return {"context": ctx, "completion": out["choices"][0]["text"]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Safe single-context Vulkan/GPU memory measurement.")
    ap.add_argument("model", nargs="?", help="local .gguf, HF repo id, or repo:filename.gguf")
    ap.add_argument("ctx", nargs="?", type=int)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--overhead", type=float, default=1.0)
    ap.add_argument("--limits", action="store_true",
                    help="print the memory wall + safe budget and exit (no model)")
    ap.add_argument("--preflight", action="store_true",
                    help="print the no-load estimate and exit")
    ap.add_argument("--probe", action="store_true",
                    help="internal: load once (offloaded) and print this load's footprint")
    ap.add_argument("--abort-gb", type=float, default=None,
                    help="internal: L5 watchdog wall for --probe")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--generate", action="store_true",
                    help="one-shot completion at <ctx> (the governed ceiling); prompt on stdin")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-flash-attn", action="store_true",
                    help="disable Vulkan flash-attention (on by default; FA ~doubles the context "
                         "ceiling at a small prefill cost). Off favours prompt-processing speed.")
    ap.add_argument("--kv-quant", choices=tuple(_KV_GGML_TYPE), default="f16",
                    help="KV-cache quantization (symmetric K=V): q8_0 ~half KV memory near-lossless, "
                         "q4_0 ~quarter at a quality cost. Requires (and forces on) flash-attention.")
    args = ap.parse_args(argv)
    flash_attn = not args.no_flash_attn
    kv_quant = args.kv_quant

    if args.limits:
        result = limits(margin_gb=args.margin)
    elif args.probe:
        result = _probe(args.model, args.ctx, args.abort_gb, flash_attn, kv_quant)
    elif args.preflight:
        result = preflight(args.model, margin_gb=args.margin, overhead_gb=args.overhead,
                           kv_quant=kv_quant)
    elif args.generate:
        result = generate(args.model, args.ctx, sys.stdin.read(),
                          margin_gb=args.margin, overhead_gb=args.overhead,
                          max_tokens=args.max_tokens, flash_attn=flash_attn, kv_quant=kv_quant)
    else:
        result = run(args.model, args.ctx, margin_gb=args.margin, overhead_gb=args.overhead,
                     flash_attn=flash_attn, kv_quant=kv_quant, repeats=args.repeats)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
