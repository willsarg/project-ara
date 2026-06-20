"""Backend namespace stubs — placeholders, no implementation yet.

These modules stake out where each device/runtime *will* live in ARA's hierarchy
("AI Runs Anywhere"). They carry only a docstring: the device, its measurement
contract class, and the memory-wall source. This test asserts the namespace resolves
and every stub declares its contract class, so the organization stays honest as the
real adapters get filled in — without committing to any executable behavior.
"""
from __future__ import annotations

import importlib

import pytest

# (module, expected contract class named in the docstring)
_STUBS = [
    ("ara.backends.rocm", "ramp"),        # AMD GPU (Radeon + APU/Strix)
    ("ara.backends.oneapi", "ramp"),      # Intel Arc / iGPU
    ("ara.backends.ane", "graph-fit"),    # Apple Neural Engine
    ("ara.backends.xdna", "graph-fit"),   # AMD Ryzen AI NPU
    ("ara.backends.intel_npu", "graph-fit"),  # Intel NPU
    ("ara.backends.hexagon", "graph-fit"),    # Qualcomm Hexagon
    ("ara.backends.coral", "graph-fit"),  # Google Edge TPU
    ("ara.backends.esp32", "static-fit"),  # MCU / TinyML
    ("ara.backends.webgpu", "ramp"),      # browser runtime
]


@pytest.mark.parametrize("module, contract", _STUBS)
def test_stub_resolves_and_declares_contract(module, contract):
    mod = importlib.import_module(module)
    doc = mod.__doc__ or ""
    assert "STUB" in doc, f"{module} should mark itself a stub"
    assert contract in doc, f"{module} should name its '{contract}' contract class"
