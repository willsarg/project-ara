# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Measurement contracts — the methodology ARA owns, shared across backends.

A *contract class* defines what "assessment" means for a family of hardware with the
same physics: ``ramp`` (safe context ceiling, KV grows), and later graph-fit / static-fit.
The methodology lives here, in ARA, so a given class produces comparable numbers on every
backend that shares it — engines only supply safe measurements, not the math.
"""
