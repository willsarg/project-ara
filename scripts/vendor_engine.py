#!/usr/bin/env python3
"""Re-vendor a folded engine suite's source into ARA's wheel.

The Apple (`wmx`) and CUDA (`wcx`) engines are vendored under ``ara/_vendor/<key>`` and ship in
ARA's wheel (see ara/engines.py — they have no git ``spec``). To bump an engine to a newer commit,
re-run this against a clean checkout of its repo:

    uv run python scripts/vendor_engine.py wmx ~/Documents/Github/willsarg/wmx-suite

It copies the package source (minus caches) plus the build-required sidecars
(``pyproject.toml``, ``README.md``, ``LICENSE``, ``NOTICE``) verbatim — it never edits the suite.
Review the diff, run the gate (`uv run pytest`), then commit. Maintainer tooling — not shipped in
the wheel and outside the coverage gate.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

from ara import engines

VENDOR_ROOT = pathlib.Path(__file__).resolve().parent.parent / "ara" / "_vendor"
SIDECARS = ("pyproject.toml", "README.md", "LICENSE", "NOTICE")


def vendor(key: str, checkout: pathlib.Path) -> int:
    engine = engines.ENGINES.get(key)
    if engine is None or "spec" in engine:
        raise SystemExit(f"{key!r} is not a vendored engine (must exist and have no git spec)")
    pkg = engine["package"].replace("-", "_")            # wmx-suite -> wmx_suite
    src_pkg = checkout / pkg
    if not src_pkg.is_dir():
        raise SystemExit(f"package dir not found: {src_pkg}")

    dst = VENDOR_ROOT / key
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    shutil.copytree(src_pkg, dst / pkg,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in SIDECARS:
        s = checkout / name
        if s.is_file():
            shutil.copy2(s, dst / name)

    try:
        sha = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "(not a git checkout)"
    print(f"{key}: vendored {pkg} from {checkout}@{sha[:12]} -> {dst.relative_to(VENDOR_ROOT.parent.parent)}")
    print("review the diff, run `uv run pytest`, update the 'Folded … from <repo>@<sha>' note in "
          "ara/engines.py, then commit.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise SystemExit("usage: vendor_engine.py <key> <path-to-suite-checkout>")
    return vendor(args[0], pathlib.Path(args[1]).expanduser().resolve())


if __name__ == "__main__":
    sys.exit(main())
