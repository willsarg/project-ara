"""AMD GPU backend — STUB (no implementation yet).

Contract class: **ramp** (safe context ceiling, like cuda/apple).
Wall source: ``rocm-smi`` (or sysfs). Covers two memory shapes in one adapter:
discrete Radeon (VRAM is an *exact* wall, no calibration — like cuda) and the
Ryzen APU / Strix Halo line (*unified* memory shared with the system, hidden
cold-start overhead — calibrate like apple). Engine is on-demand (ROCm/HIP, or a
shared Vulkan probe path); ARA owns persistence.
"""
