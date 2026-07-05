#!/usr/bin/env python3
"""Re-vendor a folded engine suite's source into ARA's wheel.

The Apple (`wmx`) and CUDA (`wcx`) engines are vendored under ``ara/_vendor/<key>`` and ship in
ARA's wheel (see ara/engines.py — they have no git ``spec``). To bump an engine to a newer commit,
re-run this against a clean checkout of its repo:

    uv run python scripts/vendor_engine.py wmx ~/Documents/Github/willsarg/wmx-suite

The vendored tree is now ARA-owned and trimmed (Spec 2026-07-05-refold-engines-to-adapter-surface):
the standalone-app scaffolding was removed and the manifest is ARA-maintained. So re-vendor is a
**reconcile, not a verbatim copy** — it copies newer module source, then **prunes back to the module
set currently vendored** (never re-dragging the removed cli/db/ui/launcher/views), **keeps ARA's
trimmed ``pyproject.toml``** (reconcile upstream dep changes by hand), and copies only the stable
``LICENSE``/``NOTICE`` sidecars. Review the diff, run the gate (`uv run pytest` — the engine-boundary
guard included), then commit. Maintainer tooling — not shipped in the wheel, outside the coverage gate.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

from ara import engines

VENDOR_ROOT = pathlib.Path(__file__).resolve().parent.parent / "ara" / "_vendor"
# Only the stable attribution files are copied verbatim. pyproject.toml is NOT — ARA now maintains a
# trimmed manifest (base deps + the `[nonllm]`/`[cuda]` extras, no console-scripts); re-vendor must
# not clobber it. README.md was dropped from the vendored trees. Spec
# 2026-07-05-refold-engines-to-adapter-surface.
SIDECARS = ("LICENSE", "NOTICE")


def vendor(key: str, checkout: pathlib.Path) -> int:
    engine = engines.ENGINES.get(key)
    if engine is None or "spec" in engine:
        raise SystemExit(f"{key!r} is not a vendored engine (must exist and have no git spec)")
    pkg = engine["package"].replace("-", "_")            # wmx-suite -> wmx_suite
    src_pkg = checkout / pkg
    if not src_pkg.is_dir():
        raise SystemExit(f"package dir not found: {src_pkg}")

    dst = VENDOR_ROOT / key
    dst_pkg = dst / pkg
    # Preserve the trim (Spec 2026-07-05): the vendored tree was reduced to the headless
    # measurement/governance modules (the standalone-app scaffolding — cli/db/ui/launcher/views —
    # was removed). Re-vendor must bring NEWER versions of the modules we keep, NOT re-drag the
    # deleted ones. The current vendored tree IS the allow-list: snapshot its module set, then prune
    # the fresh copy back to it. (A genuinely-new module we want must be added to the tree by hand —
    # deliberately, not smuggled in by a bulk copy.)
    kept = {q.relative_to(dst_pkg) for q in dst_pkg.rglob("*.py")} if dst_pkg.is_dir() else None
    manifest = dst / "pyproject.toml"
    keep_manifest = manifest.read_text(encoding="utf-8") if manifest.is_file() else None
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    shutil.copytree(src_pkg, dst_pkg,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if kept is not None:
        for q in list(dst_pkg.rglob("*.py")):
            if q.relative_to(dst_pkg) not in kept:
                q.unlink()
        for d in sorted((p for p in dst_pkg.rglob("*") if p.is_dir()), reverse=True):
            if not any(d.iterdir()):
                d.rmdir()
        print(f"{key}: pruned re-vendor to the {len(kept)} vendored modules (trim preserved)")
    if keep_manifest is not None:                    # ARA maintains the trimmed manifest — restore it
        manifest.write_text(keep_manifest, encoding="utf-8")
        print(f"{key}: kept ARA's trimmed pyproject.toml (reconcile any upstream dep changes by hand)")
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
