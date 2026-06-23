# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Portable browser runtime (WebGPU / WASM) — STUB (no implementation yet).

Contract class: **ramp** (safe context ceiling), measured against a portable runtime
rather than a physical device — the literal end of "AI Runs Anywhere." Engines run
in the browser via transformers.js / WebLLM over WebGPU (WASM/CPU fallback).
Wall source: the WebGPU device's reported limits (``maxBufferSize`` /
``maxStorageBufferBindingSize``) and the JS heap budget — a software wall the host
browser enforces, not a hardware register.
"""
