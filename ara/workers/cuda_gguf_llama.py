# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""CUDA-GGUF hybrid worker — partial GPU offload of a GGUF on NVIDIA, runs in the ``cuda-gguf`` env.

The first **two-wall** engine: it runs a GGUF model too big for VRAM by offloading *K* of *N*
layers to the GPU (llama.cpp ``n_gpu_layers=K``, CUDA build) and the rest on the CPU — so it must
stay under **both** the discrete VRAM wall *and* the system-RAM wall at once. Every other engine
governs a single wall.

Self-contained (never imports ``ara``; ``llama_cpp`` only inside functions), like the cpu/vulkan
workers — so its novel pure logic (the per-layer split, the K auto-fit, the two-wall gate, the
load-log buffer parse, the partial-offload honest check) is unit-testable in ARA's venv
(tests/test_workers_cuda_gguf_llama.py). It mirrors the canonical worker contract so ARA's
engine-agnostic driver treats it like the others, but reports two budgets and certifies the real
footprint from llama.cpp's **parseable load log** (CUDA0 buffers → VRAM, CPU_Mapped/CPU KV → RAM),
not RSS (mmap leaves offloaded pages in the page cache → RSS lies).

Design: Designs/specs/2026-06-29-cuda-gguf-hybrid-two-wall-engine.md.

Usage:
    python cuda_gguf_llama.py <model> <ctx> --vram-margin G --ram-margin G [--preflight]
    python cuda_gguf_llama.py <model> <ctx> --generate  --max-tokens N        (prompt on stdin)
    python cuda_gguf_llama.py <model> <ctx> --benchmark --max-tokens N        (JSON prompts stdin)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys

GIB = 1024 ** 3
DEFAULT_REPEATS = 3
KV_BYTES_F16 = 2
RESIDENT_FACTOR = 1.0
# Fixed CUDA-context/compute floor (cuBLAS + kernels), ≈1516 MiB per the oobabooga empirical fit.
CUDA_FLOOR_GB = 1.5


# --------------------------------------------------------------------------- #
# Pure logic (no engine import) — unit-tested in ARA's venv.
# --------------------------------------------------------------------------- #
def kv_slope_gb_per_k(meta: dict, *, kv_bytes: int = KV_BYTES_F16) -> float:
    """GB of KV cache added per 1000 tokens, from GGUF metadata (GQA-aware). Total across all
    layers — the per-layer share is this divided by the layer count."""
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
    arch = meta["general.architecture"]
    return int(meta[f"{arch}.context_length"])


def n_layers_from(meta: dict) -> int:
    arch = meta["general.architecture"]
    return int(meta[f"{arch}.block_count"])


def effective_margin_gb(total_gb: float, cap_gb: float) -> float:
    """Margin to use: ~10% of the wall, capped at *cap_gb*, floored at 0.5 GB (small-box safe)."""
    return min(cap_gb, max(0.5, total_gb * 0.1))


def safe_threshold_gb(total_gb: float, margin_gb: float) -> float:
    """Safe budget = wall − margin, never negative."""
    return max(0.0, total_gb - margin_gb)


def per_layer_gb(weights_gb: float, n_layers: int) -> float:
    """Coarse per-layer weight footprint — the GGUF file size split evenly across layers. The
    a-priori split is intentionally coarse; the load-log measurement is the real certifier."""
    return weights_gb / n_layers


def vram_estimate(k: int, n_layers: int, weights_gb: float, kv_slope_gb_per_k_: float,
                  ctx: int, cuda_floor_gb: float = CUDA_FLOOR_GB) -> float:
    """A-priori VRAM for *k* offloaded layers at *ctx*: fixed CUDA floor + k layers' weights +
    those k layers' share of the KV cache. Conservative + coarse (see the design spec §4)."""
    per_w = per_layer_gb(weights_gb, n_layers)
    kv_per_layer = kv_slope_gb_per_k_ / n_layers
    return round(cuda_floor_gb + k * per_w + k * kv_per_layer * (ctx / 1000), 4)


def ram_estimate(k: int, n_layers: int, weights_gb: float, kv_slope_gb_per_k_: float,
                 ctx: int, live_base_gb: float) -> float:
    """A-priori system-RAM for the (N−k) CPU-resident layers at *ctx*: live baseline + those
    layers' weights + their share of the KV cache."""
    rem = n_layers - k
    per_w = per_layer_gb(weights_gb, n_layers)
    kv_per_layer = kv_slope_gb_per_k_ / n_layers
    return round(live_base_gb + rem * per_w + rem * kv_per_layer * (ctx / 1000), 4)


def fit_layers(n_layers: int, weights_gb: float, kv_slope_gb_per_k_: float, ctx: int,
               vram_budget_gb: float, cuda_floor_gb: float = CUDA_FLOOR_GB) -> int:
    """The largest K in [0, n_layers] whose VRAM estimate stays under *vram_budget_gb* at *ctx*.

    VRAM rises monotonically with K, so we stop at the first K that busts the budget. Returns 0 when
    even the CUDA floor alone exceeds the budget (the GPU can't host a single layer safely)."""
    best = 0
    for k in range(0, n_layers + 1):
        if vram_estimate(k, n_layers, weights_gb, kv_slope_gb_per_k_, ctx, cuda_floor_gb) \
                <= vram_budget_gb:
            best = k
        else:
            break
    return best


def two_wall_gate(k: int, n_layers: int, weights_gb: float, kv_slope_gb_per_k_: float, ctx: int,
                  *, vram_budget_gb: float, ram_budget_gb: float, live_base_gb: float,
                  cuda_floor_gb: float = CUDA_FLOOR_GB) -> str | None:
    """Refuse-before-load (L4) on BOTH walls (Rule #1). Returns a reason to refuse, or None when
    *k* offloaded layers fit VRAM AND the (N−k) remainder fits system RAM at *ctx*."""
    v = vram_estimate(k, n_layers, weights_gb, kv_slope_gb_per_k_, ctx, cuda_floor_gb)
    if v >= vram_budget_gb:
        return (f"VRAM: estimated {v:.2f}GB for K={k} layers at {ctx} tok "
                f">= safe VRAM budget {vram_budget_gb:.2f}GB")
    r = ram_estimate(k, n_layers, weights_gb, kv_slope_gb_per_k_, ctx, live_base_gb)
    if r >= ram_budget_gb:
        return (f"RAM: estimated {r:.2f}GB for {n_layers - k} CPU layers at {ctx} tok "
                f">= safe RAM budget {ram_budget_gb:.2f}GB")
    return None


def governed_max_tokens(prompt_tokens: int, requested_max_tokens: int,
                        ceiling: int) -> int | None:
    """Allowed max_tokens for one prompt under *ceiling*, or None to refuse. Identical contract to
    the cpu/vulkan/MLX workers (Rule #1)."""
    if prompt_tokens >= ceiling:
        return None
    if prompt_tokens + requested_max_tokens > ceiling:
        return None
    return min(requested_max_tokens, ceiling - prompt_tokens)


def parse_offloaded(stderr: str) -> tuple[int, int] | None:
    """Parse ``offloaded K/N layers to GPU`` → (K, N), or None if absent (CPU-only / fallback)."""
    m = re.search(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU", stderr)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_cuda_device(stderr: str) -> dict | None:
    """Parse llama.cpp's CUDA device line ``... Device 0: <name>, ...`` → {name}, or None."""
    m = re.search(r"Device\s+\d+:\s*([^,\n]+)", stderr)
    return {"name": m.group(1).strip()} if m else None


def parse_cuda_buffers(stderr: str) -> dict:
    """The post-load TRUTH from the load log. Sum every ``CUDA0 … buffer size = N MiB`` line into
    the VRAM side and every ``CPU[_Mapped] … buffer size = N MiB`` line into the RAM side (weights +
    KV + compute). Returns ``{vram_gb, ram_gb}``. Trusted over RSS (mmap page-cache lies)."""
    def _sum(pattern: str) -> float:
        return sum(float(x) for x in re.findall(pattern, stderr))
    vram_mib = _sum(r"CUDA0[^\n=]*buffer size\s*=\s*([\d.]+)\s*MiB")
    ram_mib = _sum(r"CPU(?:_Mapped)?[^\n=]*buffer size\s*=\s*([\d.]+)\s*MiB")
    return {"vram_gb": round(vram_mib / 1024, 4), "ram_gb": round(ram_mib / 1024, 4)}


def offload_ok_partial(device: dict | None, offloaded: tuple[int, int] | None) -> str | None:
    """Honest-offload guard (Rule #3). PARTIAL (0 < K ≤ N) on a real CUDA device is the expected
    hybrid state — accepted (unlike Vulkan, which demands full offload). Refuse the
    silent-CPU-fallback (#2079): no offload line at all, K==0, or a software device."""
    if offloaded is None:
        return ("GPU offload not active (no 'offloaded N/M layers to GPU' line — the CUDA "
                "llama.cpp wheel may have fallen back to CPU, see abetlen #2079)")
    k, _n = offloaded
    if k == 0:
        return "model ran on CPU (0 layers offloaded) — not a hybrid run"
    name = (device or {}).get("name", "") or ""
    if "llvmpipe" in name.lower() or "software" in name.lower():
        return f"refusing a software rasterizer device ({name!r}) — not a real CUDA GPU"
    return None


# --------------------------------------------------------------------------- #
# Engine-touching helpers (import llama_cpp / psutil / nvidia-smi inside).
# Not unit-covered here (workers are omitted from the gate); validated LIVE on willw11.
# --------------------------------------------------------------------------- #
def _used_ram_gb() -> float:
    """Live system RAM in use (GB) — MAX of a few reads (Rule #1: never under-report the base)."""
    import psutil
    return max(psutil.virtual_memory().used for _ in range(3)) / GIB


def _total_ram_gb() -> float:
    import psutil
    return psutil.virtual_memory().total / GIB


def _vram_total_used_gb() -> tuple[float, float]:
    """(total, used) VRAM in GB via nvidia-smi. Used is the gross, all-process figure."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    total_mib, used_mib = (float(x) for x in out.stdout.strip().splitlines()[0].split(","))
    return total_mib / 1024, used_mib / 1024


def _resolve_gguf(model: str) -> str:
    """Resolve *model* to a local GGUF path, downloading from HF if needed (smallest .gguf)."""
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
    from llama_cpp import Llama
    return dict(Llama(model_path=gguf_path, vocab_only=True, verbose=False).metadata)


def _budgets(vram_margin_gb: float, ram_margin_gb: float) -> dict:
    """Both walls, read exactly (no calibration): VRAM via nvidia-smi, RAM via psutil."""
    vram_total, vram_used = _vram_total_used_gb()
    ram_total = _total_ram_gb()
    ram_used = _used_ram_gb()
    return {
        "vram_total_gb": round(vram_total, 3),
        "vram_used_gb": round(vram_used, 3),
        "vram_budget_gb": safe_threshold_gb(vram_total - vram_used, vram_margin_gb),
        "ram_total_gb": round(ram_total, 3),
        "ram_used_gb": round(ram_used, 3),        # ambient base added to OUR measured CPU buffers
        "ram_budget_gb": safe_threshold_gb(ram_total, effective_margin_gb(ram_total, ram_margin_gb)),
    }


def limits(*, vram_margin_gb: float, ram_margin_gb: float) -> dict:
    b = _budgets(vram_margin_gb, ram_margin_gb)
    return {"device": "GPU+CPU (CUDA hybrid)", **b}


def _model_weights_gb(gguf_path: str) -> float:
    return os.path.getsize(gguf_path) / GIB * RESIDENT_FACTOR


def preflight(model: str, *, vram_margin_gb: float, ram_margin_gb: float) -> dict:
    """No-load estimate for ARA's scheduler, plus the auto-fit K at the model's full window."""
    try:
        gguf = _resolve_gguf(model)
        meta = _read_meta(gguf)
    except Exception as e:
        return {"error": str(e)}
    b = _budgets(vram_margin_gb, ram_margin_gb)
    weights = _model_weights_gb(gguf)
    n = n_layers_from(meta)
    slope = kv_slope_gb_per_k(meta)
    live = _used_ram_gb()
    win = max_context_from(meta)
    k = fit_layers(n, weights, slope, min(win, 4000), b["vram_budget_gb"])
    return {
        "base_gb": round(live + ram_estimate(k, n, weights, slope, 0, 0.0), 4),
        "ref_baseline_gb": round(live, 4),
        "slope_gb_per_k": slope,
        "budget_gb": b["ram_budget_gb"],   # the driver ramps ctx on the RAM wall (the binding one
                                           # as K shrinks at high ctx); the worker guards VRAM per rung
        "n_layers": n,
        "fit_layers": k,
        "vram_budget_gb": b["vram_budget_gb"],
        "ram_budget_gb": b["ram_budget_gb"],
        "max_context": win,
    }


def _refused(ctx: int, reason: str) -> dict:
    return {"context": ctx, "refused": True, "reason": reason}


def _gate(gguf: str, meta: dict, ctx: int, vram_margin_gb: float,
          ram_margin_gb: float) -> tuple[int, dict] | dict:
    """Shared L4: auto-fit K, then two-wall gate. Returns (K, budgets) if safe, else a refusal."""
    b = _budgets(vram_margin_gb, ram_margin_gb)
    weights = _model_weights_gb(gguf)
    n = n_layers_from(meta)
    slope = kv_slope_gb_per_k(meta)
    live = _used_ram_gb()
    k = fit_layers(n, weights, slope, ctx, b["vram_budget_gb"])
    reason = two_wall_gate(k, n, weights, slope, ctx, vram_budget_gb=b["vram_budget_gb"],
                           ram_budget_gb=b["ram_budget_gb"], live_base_gb=live)
    if reason is not None:
        return _refused(ctx, reason)
    return k, b


def _load(gguf: str, ctx: int, n_gpu_layers: int):
    """Load offloaded, capturing llama.cpp's stderr so we can verify offload + read buffers."""
    import contextlib
    import io

    from llama_cpp import Llama
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        llm = Llama(model_path=gguf, n_ctx=ctx, n_gpu_layers=n_gpu_layers, verbose=True)
    return llm, buf.getvalue()


def _verify_offload(stderr: str, b: dict) -> str | None:
    """Honest-offload (Rule #3) + measured two-wall check (L2) from the load log."""
    reason = offload_ok_partial(parse_cuda_device(stderr), parse_offloaded(stderr))
    if reason is not None:
        return reason
    m = parse_cuda_buffers(stderr)
    if m["vram_gb"] >= b["vram_budget_gb"]:
        return f"measured VRAM {m['vram_gb']:.2f}GB >= safe budget {b['vram_budget_gb']:.2f}GB"
    if m["ram_gb"] + b.get("ram_used_gb", 0.0) >= b["ram_budget_gb"]:
        return f"measured RAM {m['ram_gb']:.2f}GB >= safe budget {b['ram_budget_gb']:.2f}GB"
    return None


def run(model: str, ctx: int, *, vram_margin_gb: float, ram_margin_gb: float,
        repeats: int = DEFAULT_REPEATS) -> dict:
    """Gate (L4) then, if safe, load offloaded and certify the measured two-wall footprint (L2)."""
    try:
        gguf = _resolve_gguf(model)
        meta = _read_meta(gguf)
    except Exception as e:
        return _refused(ctx, str(e))
    gated = _gate(gguf, meta, ctx, vram_margin_gb, ram_margin_gb)
    if isinstance(gated, dict):
        return gated
    k, b = gated
    vrams = []
    for _ in range(max(1, repeats)):
        llm, stderr = _load(gguf, ctx, k)
        bad = _verify_offload(stderr, b)
        if bad is not None:
            return _refused(ctx, bad)
        vrams.append(parse_cuda_buffers(stderr)["vram_gb"])
        del llm
    return {"context": ctx, "gpu_layers": k, "mem_gb": round(statistics.median(vrams), 3)}


def generate(model: str, ctx: int, prompt: str, *, vram_margin_gb: float, ram_margin_gb: float,
             max_tokens: int) -> dict:
    try:
        gguf = _resolve_gguf(model)
        meta = _read_meta(gguf)
    except Exception as e:
        return _refused(ctx, str(e))
    gated = _gate(gguf, meta, ctx, vram_margin_gb, ram_margin_gb)
    if isinstance(gated, dict):
        return gated
    k, b = gated
    llm, stderr = _load(gguf, ctx, k)
    bad = _verify_offload(stderr, b)
    if bad is not None:
        return _refused(ctx, bad)
    # create_chat_completion applies the GGUF's embedded chat template (instruct models need it —
    # raw create_completion yields empty/garbage on template-strict models like gemma). Rule #3.
    out = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                     max_tokens=max_tokens)
    return {"context": ctx, "gpu_layers": k, "completion": out["choices"][0]["message"]["content"]}


def benchmark(model: str, ctx: int, prompts: list, *, vram_margin_gb: float, ram_margin_gb: float,
              max_tokens: int) -> dict:
    """Gate + honest-offload once, then complete every prompt with the model loaded a SINGLE time,
    governing each prompt's max_tokens under the ceiling (Rule #1)."""
    try:
        gguf = _resolve_gguf(model)
        meta = _read_meta(gguf)
    except Exception as e:
        return _refused(ctx, str(e))
    gated = _gate(gguf, meta, ctx, vram_margin_gb, ram_margin_gb)
    if isinstance(gated, dict):
        return gated
    k, b = gated
    llm, stderr = _load(gguf, ctx, k)
    bad = _verify_offload(stderr, b)
    if bad is not None:
        return _refused(ctx, bad)
    results = []
    for i, prompt in enumerate(prompts):
        n_prompt = len(llm.tokenize(prompt.encode("utf-8")))
        allowed = governed_max_tokens(n_prompt, max_tokens, ctx)
        if allowed is None:
            results.append({"prompt_index": i, "refused": True,
                            "reason": f"prompt fills context ceiling {ctx}"})
            continue
        out = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                         max_tokens=allowed)
        results.append({"prompt_index": i, "completion": out["choices"][0]["message"]["content"]})
    return {"context": ctx, "gpu_layers": k, "results": results}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Safe two-wall CUDA-GGUF hybrid (partial offload).")
    ap.add_argument("model", nargs="?")
    ap.add_argument("ctx", nargs="?", type=int)
    ap.add_argument("--vram-margin", type=float, default=1.0)
    ap.add_argument("--ram-margin", type=float, default=2.0)
    ap.add_argument("--limits", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = ap.parse_args(argv)

    if args.limits:
        result = limits(vram_margin_gb=args.vram_margin, ram_margin_gb=args.ram_margin)
    elif args.preflight:
        result = preflight(args.model, vram_margin_gb=args.vram_margin,
                           ram_margin_gb=args.ram_margin)
    elif args.generate:
        result = generate(args.model, args.ctx, sys.stdin.read(),
                          vram_margin_gb=args.vram_margin, ram_margin_gb=args.ram_margin,
                          max_tokens=args.max_tokens)
    elif args.benchmark:
        result = benchmark(args.model, args.ctx, json.loads(sys.stdin.read()),
                           vram_margin_gb=args.vram_margin, ram_margin_gb=args.ram_margin,
                           max_tokens=args.max_tokens)
    else:
        result = run(args.model, args.ctx, vram_margin_gb=args.vram_margin,
                     ram_margin_gb=args.ram_margin, repeats=args.repeats)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
