# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Shared version lookups for installed AI/ML software — from the reliable, non-fragile
sources: a macOS .app's Info.plist (CFBundleShortVersionString) and `brew list --versions`.

Cached per process (lru_cache) so the Homebrew calls run once even though both the apps
inventory and the engines list ask for versions.
"""
from __future__ import annotations

import functools
import json
import os
import plistlib
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

_APP_DIRS = (Path("/Applications"), Path.home() / "Applications")


def _brew_versions(kind: str) -> dict[str, str | None]:
    """{package: version} for installed Homebrew formulae or casks. Keys lowercased and
    stripped of any @version suffix; value is None if a version can't be parsed."""
    if not shutil.which("brew"):
        return {}
    try:
        out = subprocess.run(["brew", "list", kind, "--versions"],
                             capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return {}
    versions: dict[str, str | None] = {}
    for line in (out or "").splitlines():
        parts = line.split()
        if parts:
            name = parts[0].lower().split("@", 1)[0]
            versions.setdefault(name, parts[1] if len(parts) > 1 else None)
    return versions


@functools.lru_cache(maxsize=1)
def brew_formulae() -> dict[str, str | None]:
    return _brew_versions("--formula")


@functools.lru_cache(maxsize=1)
def brew_casks() -> dict[str, str | None]:
    return _brew_versions("--cask")


def brew_version(*tokens: str) -> str | None:
    """Version of the first of *tokens* installed as a formula or cask, else None."""
    for t in tokens:
        v = brew_formulae().get(t) or brew_casks().get(t)
        if v:
            return v
    return None


def cask_auto_updates(tokens: Iterable[str]) -> dict[str, bool]:
    """{cask token: auto_updates} for the given cask *tokens* — True when the cask declares
    that the app updates itself, so brew defers version management (drift is expected, not a
    conflict). One batched `brew info` call scoped to *tokens* (never the full installed-cask
    list — that's the caller's job, e.g. the tokens actually present in the scanned AI/ML app
    inventory) so this can't balloon into a full-catalog fetch. Tokens lowercased, @version
    stripped."""
    key = tuple(sorted({t.lower().split("@", 1)[0] for t in tokens if t}))
    return _cask_auto_updates_cached(key)


@functools.lru_cache(maxsize=1)
def _cask_auto_updates_cached(tokens: tuple[str, ...]) -> dict[str, bool]:
    if not tokens:
        return {}
    try:
        env = {**os.environ, "HOMEBREW_NO_AUTO_UPDATE": "1"}
        out = subprocess.run(["brew", "info", "--json=v2", "--cask", *tokens],
                             capture_output=True, text=True, timeout=30, env=env).stdout
        data = json.loads(out)
    except Exception:
        return {}
    result: dict[str, bool] = {}
    for c in data.get("casks", []):
        token = (c.get("token") or "").lower().split("@", 1)[0]
        if token:
            result[token] = bool(c.get("auto_updates"))
    return result


def _plist_version(app: Path) -> str | None:
    try:
        with open(app / "Contents" / "Info.plist", "rb") as f:
            data = plistlib.load(f)
        return data.get("CFBundleShortVersionString") or data.get("CFBundleVersion")
    except Exception:
        return None


def find_app(bundles: list[str]) -> tuple[bool, str | None]:
    """(present, version) for the first of *bundles* (names without .app) found in an
    Applications folder. Version may be None even when present (unreadable Info.plist)."""
    for b in bundles:
        for base in _APP_DIRS:
            app = base / f"{b}.app"
            if app.is_dir():
                return True, _plist_version(app)
    return False, None
