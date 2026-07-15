# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Static contracts for CI jobs that local actionlint and runtime probes exercise."""
from __future__ import annotations

import re
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOWS = _ROOT / ".github" / "workflows"


def test_coordinator_ci_uses_the_same_node_major_as_the_production_image():
    workflow = (_WORKFLOWS / "dependabot-ci.yml").read_text(encoding="utf-8")
    dockerfile = (_ROOT / "coordinator" / "Dockerfile").read_text(encoding="utf-8")

    workflow_major = re.search(r'node-version: "(\d+)"', workflow)
    image_majors = set(re.findall(r"^FROM node:(\d+)-", dockerfile, flags=re.MULTILINE))

    assert workflow_major is not None
    assert {workflow_major.group(1)} == image_majors


def test_docker_smoke_loop_does_not_name_an_unused_counter():
    workflow = (_WORKFLOWS / "docker-scan.yml").read_text(encoding="utf-8")

    assert "for _ in $(seq 1 15); do" in workflow


def test_mutation_summary_uses_one_atomic_redirect_block():
    workflow = (_WORKFLOWS / "mutation.yml").read_text(encoding="utf-8")

    assert workflow.count('>> "$GITHUB_STEP_SUMMARY"') == 1
