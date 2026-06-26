#!/usr/bin/env python3
"""Pin ARA's external engine refs to each engine repo's current main HEAD.

Run before tagging an ARA release so the release ships reproducible, immutable engine commits
(see ara/engines.py `ref` fields):

    uv run python scripts/pin_engines.py

It git-ls-remotes each external engine's repo, rewrites the `ref` SHAs in ara/engines.py in place,
and prints the changes. Review the diff, bump the ara version, then tag. Maintainer tooling — not
shipped in the wheel and outside the coverage gate.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

from ara import engines

ENGINES_PY = pathlib.Path(__file__).resolve().parent.parent / "ara" / "engines.py"


def _main_head(spec: str) -> str:
    """The current main-branch HEAD SHA of the git repo behind an engine `spec`."""
    url = spec.removeprefix("git+")
    out = subprocess.run(["git", "ls-remote", url, "refs/heads/main"],
                         capture_output=True, text=True, check=True).stdout
    if not out.strip():
        raise SystemExit(f"no main branch at {url}")
    return out.split()[0]


def main() -> int:
    text = ENGINES_PY.read_text()
    changed = False
    for key, engine in engines.ENGINES.items():
        old, spec = engine.get("ref"), engine.get("spec", "")
        if not old or not spec.startswith("git+"):
            continue                       # builtin / unpinned engine — skip
        new = _main_head(spec)
        if new == old:
            print(f"{key}: already at {new[:12]}")
            continue
        if old not in text:
            raise SystemExit(f"{key}: current ref {old} not found in {ENGINES_PY}")
        text = text.replace(old, new)      # each ref SHA is unique → unambiguous
        changed = True
        print(f"{key}: {old[:12]} -> {new[:12]}")
    if changed:
        ENGINES_PY.write_text(text)
        print("\nengines.py updated — review the diff, bump the ara version, then tag.")
    else:
        print("\nall engines already pinned to main HEAD — nothing to do.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
