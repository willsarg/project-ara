"""Intel GPU backend (Arc / iGPU) — STUB (no implementation yet).

Contract class: **ramp** (safe context ceiling).
Wall source: Level Zero / ``xpu-smi``. Discrete Arc reads an exact VRAM wall (no
calibration, like cuda); an integrated Xe shares system memory (unified, calibrate
like apple). Engine on-demand (oneAPI/SYCL, IPEX-LLM, or the shared Vulkan path).
This is the Intel *GPU* — the Intel NPU is a separate accelerator (see intel_npu).
"""
