"""Apple Neural Engine (ANE / CoreML) — STUB (no implementation yet).

Contract class: **graph-fit** (not a context ramp). The question here isn't "how far
can KV-cache grow" but "does this fixed, quantized graph map onto the accelerator and
its memory slice." A different assessment from apple.py, which targets the *GPU*
(MLX/Metal) on the same chip — a modern Mac carries both backends at once.
Wall source: unified memory shared with the system; programmed via CoreML.
"""
