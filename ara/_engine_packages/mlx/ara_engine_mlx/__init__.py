# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""ARA's native MLX engine — safe local MLX inference on Apple Silicon.

First rule: never crash the laptop. The suite finds each model's safe context ceiling
by extrapolating from measurements taken well below the hardware wall — it never probes
into the danger zone.
"""

__version__ = "0.1.0"
