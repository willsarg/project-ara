# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Static contracts for CI jobs that local actionlint and runtime probes exercise."""
from __future__ import annotations

import re
import tomllib
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


def test_cross_os_suite_uses_bounded_parallelism_and_reports_slowest_tests():
    workflow = (_WORKFLOWS / "test.yml").read_text(encoding="utf-8")

    assert "uv run pytest -n 4 --dist=loadfile --durations=25" in workflow


def test_parallel_pytest_runner_is_locked_as_a_dev_dependency():
    config = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert any(
        requirement.startswith("pytest-xdist>=")
        for requirement in config["dependency-groups"]["dev"]
    )


def test_human_ci_gates_pull_requests_and_main_without_duplicating_dependabot():
    workflow = (_WORKFLOWS / "ci.yml").read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert "branches: [main]" in workflow
    assert "github.event.pull_request.user.login != 'dependabot[bot]'" in workflow


def test_pr_gates_cancel_stale_runs_for_the_same_branch():
    for name in ("ci.yml", "dependabot-ci.yml"):
        workflow = (_WORKFLOWS / name).read_text(encoding="utf-8")
        assert "concurrency:" in workflow
        assert "github.event.pull_request.number || github.ref" in workflow
        assert "cancel-in-progress: true" in workflow
