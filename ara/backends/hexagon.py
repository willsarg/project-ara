# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Qualcomm Hexagon NPU (Snapdragon X / mobile) — STUB (no implementation yet).

Contract class: **graph-fit** (not a context ramp). The Hexagon DSP/NPU, programmed
via the Qualcomm AI Engine (QNN) / SNPE stack; assessment is "does this quantized
graph map onto the accelerator + its memory slice." Covers Snapdragon X laptops and
mobile SoCs. The Adreno GPU on the same chip would be a separate GPU backend (likely
served by the shared Vulkan engine path).
Wall source: NPU's slice of shared system memory.
"""
