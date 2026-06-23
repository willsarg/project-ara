# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Google Coral / Edge TPU — STUB (no implementation yet).

Contract class: **graph-fit** (not a context ramp). A fixed-function edge accelerator
(USB/PCIe/SoM) that runs only INT8 TFLite models compiled by the Edge TPU compiler;
assessment is "does this compiled graph fit the device's on-chip SRAM + its model
budget," with overflow spilling to host. Closest neighbour to the MCU/TinyML class.
Wall source: Edge TPU on-chip memory (+ host RAM for the runtime).
"""
