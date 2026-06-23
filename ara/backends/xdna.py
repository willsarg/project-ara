# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""AMD Ryzen AI NPU (XDNA) — STUB (no implementation yet).

Contract class: **graph-fit** (not a context ramp). Fixed-function INT8 accelerator
programmed via the Ryzen AI Software stack (ONNX Runtime + VitisAI execution
provider); assessment is "does this quantized graph map onto the NPU tiles + its
memory slice," not a KV-cache ceiling. Distinct silicon from the Radeon/APU GPU
(see rocm.py) — a Strix Halo box carries both backends at once.
Wall source: NPU's slice of unified system memory.
"""
