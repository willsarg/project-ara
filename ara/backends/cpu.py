"""CPU backend — system-RAM inference — STUB (no implementation yet).

Contract class: **ramp** (safe context ceiling).
Wall source: system RAM + swap (psutil). One adapter for *all* CPU-only inference
regardless of ISA — x86 (Intel/AMD), arm64, Raspberry Pi, riscv-when-it-matters.
The ISA is metadata reported by ``detect``, not a separate backend, because it does
not change the wall. Engine on-demand (llama.cpp / GGUF, onnxruntime). This is the
universal fallback: nearly every machine matches it, which is why ``detect`` needs a
priority order (discrete GPU > unified/iGPU > NPU-where-applicable > cpu).
"""
