# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""ARA's native CUDA engine — safe local CUDA inference on NVIDIA GPUs.

It follows the same discipline as ARA's native MLX engine:
find each model's safe context ceiling by extrapolating from measurements taken below the
hardware wall — never probe into it. Here the wall is the GPU's VRAM rather than Apple's
unified-memory working set.
"""

__version__ = "0.1.0"
