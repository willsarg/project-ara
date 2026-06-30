# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Will's CUDA Suite (wcx-suite) — a custom VRAM/stress bench for local CUDA inference on NVIDIA GPUs.

The sibling of wmx-suite, for the other side of the hardware fence. Same discipline:
find each model's safe context ceiling by extrapolating from measurements taken below the
hardware wall — never probe into it. Here the wall is the GPU's VRAM rather than Apple's
unified-memory working set.
"""

__version__ = "0.1.0"
