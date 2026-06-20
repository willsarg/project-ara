"""Microcontroller / TinyML (ESP32, Arduino, RP2040 …) — STUB (no implementation yet).

Contract class: **static-fit** (not a context ramp, not even runtime probing). The
assessment is a *static* question answered without executing anything on-device: does
this quantized TFLite-Micro / CMSIS-NN model fit in the board's flash + SRAM at all,
and what's the headroom? No KV cache, no safe-ramp — a binary fit + margin.
Wall source: board flash size and SRAM (from a board profile, not a live read).
This is the far edge of "AI Runs Anywhere": the same product question, a different
measurement contract.
"""
