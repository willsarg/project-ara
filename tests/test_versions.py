# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""versions.py — version lookups from brew + .app Info.plist (lru-cached)."""
from __future__ import annotations

import plistlib
import types

from ara import versions


# --------------------------------------------------------------------------- #
# brew formula/cask version parsing
# --------------------------------------------------------------------------- #
def test_brew_versions_empty_without_brew(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda n: None)
    assert versions._brew_versions("--formula") == {}


def test_brew_versions_parses_and_normalizes(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda n: "/opt/homebrew/bin/brew")
    out = "ollama 0.1.2\nllama.cpp 4567\npython@3.12 3.12.4\nbare\n"
    monkeypatch.setattr(versions.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=out))
    got = versions._brew_versions("--formula")
    assert got["ollama"] == "0.1.2"
    assert got["llama.cpp"] == "4567"
    assert got["python"] == "3.12.4"   # @version suffix stripped, lowercased
    assert got["bare"] is None         # no version column → None


def test_brew_versions_empty_on_subprocess_error(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/brew")
    def boom(*a, **k):
        raise OSError("brew exploded")
    monkeypatch.setattr(versions.subprocess, "run", boom)
    assert versions._brew_versions("--cask") == {}


def test_brew_versions_first_occurrence_wins(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/brew")
    out = "foo@1 1.0\nfoo@2 2.0\n"   # both normalize to "foo"; setdefault keeps the first
    monkeypatch.setattr(versions.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=out))
    assert versions._brew_versions("--formula") == {"foo": "1.0"}


# --------------------------------------------------------------------------- #
# brew_version (token lookup across formula + cask)
# --------------------------------------------------------------------------- #
def test_brew_version_prefers_formula_then_cask(monkeypatch):
    monkeypatch.setattr(versions, "brew_formulae", lambda: {"ollama": "0.1.2"})
    monkeypatch.setattr(versions, "brew_casks", lambda: {"lm-studio": "0.3.0"})
    assert versions.brew_version("ollama") == "0.1.2"
    assert versions.brew_version("lm-studio") == "0.3.0"       # only in casks
    assert versions.brew_version("missing", "lm-studio") == "0.3.0"  # first present wins
    assert versions.brew_version("nope") is None


# --------------------------------------------------------------------------- #
# cask_auto_updates — takes explicit tokens (not the whole brew cask list), and
# never blocks on brew's background auto-update check.
# --------------------------------------------------------------------------- #
def test_cask_auto_updates_empty_without_tokens():
    assert versions.cask_auto_updates([]) == {}


def test_cask_auto_updates_parses_json(monkeypatch):
    payload = '{"casks":[{"token":"lm-studio","auto_updates":true},{"token":"cursor@2","auto_updates":false}]}'
    monkeypatch.setattr(versions.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=payload))
    got = versions.cask_auto_updates(["lm-studio", "cursor"])
    assert got == {"lm-studio": True, "cursor": False}   # token @-suffix stripped


def test_cask_auto_updates_empty_on_error(monkeypatch):
    monkeypatch.setattr(versions.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout="not json"))
    assert versions.cask_auto_updates(["x"]) == {}


def test_cask_auto_updates_scopes_call_to_passed_tokens_only(monkeypatch):
    # Regression: cask_auto_updates() used to call brew_casks() internally and run
    # `brew info` against EVERY installed cask (a 15s+ freeze on a big Applications
    # folder). It must now hit only the tokens the caller passes, and must disable
    # brew's own network auto-update check so the call can't stall on that either.
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return types.SimpleNamespace(stdout='{"casks":[]}')

    monkeypatch.setattr(versions.subprocess, "run", fake_run)
    versions.cask_auto_updates(["Cursor", "lm-studio@1"])

    assert captured["cmd"] == ["brew", "info", "--json=v2", "--cask", "cursor", "lm-studio"]
    assert captured["env"] is not None
    assert captured["env"]["HOMEBREW_NO_AUTO_UPDATE"] == "1"


def test_cask_auto_updates_ignores_installed_casks_not_in_tokens(monkeypatch):
    monkeypatch.setattr(versions, "brew_casks", lambda: {"unrelated-cask": "9.0"})
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return types.SimpleNamespace(stdout='{"casks":[]}')

    monkeypatch.setattr(versions.subprocess, "run", fake_run)
    versions.cask_auto_updates(["cursor"])
    assert "unrelated-cask" not in captured["cmd"]
    assert captured["cmd"][-1] == "cursor"


# --------------------------------------------------------------------------- #
# .app Info.plist version + find_app
# --------------------------------------------------------------------------- #
def _make_app(base, name, plist=None):
    app = base / f"{name}.app"
    (app / "Contents").mkdir(parents=True)
    if plist is not None:
        with open(app / "Contents" / "Info.plist", "wb") as f:
            plistlib.dump(plist, f)
    return app


def test_plist_version_prefers_short_string(tmp_path):
    app = _make_app(tmp_path, "Foo", {"CFBundleShortVersionString": "1.2.3", "CFBundleVersion": "1203"})
    assert versions._plist_version(app) == "1.2.3"


def test_plist_version_falls_back_to_bundle_version(tmp_path):
    app = _make_app(tmp_path, "Foo", {"CFBundleVersion": "1203"})
    assert versions._plist_version(app) == "1203"


def test_plist_version_none_when_unreadable(tmp_path):
    app = _make_app(tmp_path, "Foo", plist=None)  # no Info.plist written
    assert versions._plist_version(app) is None


def test_find_app_present_with_version(tmp_path, monkeypatch):
    _make_app(tmp_path, "LM Studio", {"CFBundleShortVersionString": "0.3.0"})
    monkeypatch.setattr(versions, "_APP_DIRS", (tmp_path,))
    present, ver = versions.find_app(["LM Studio"])
    assert present is True and ver == "0.3.0"


def test_find_app_tries_multiple_bundle_names(tmp_path, monkeypatch):
    _make_app(tmp_path, "gpt4all", {"CFBundleShortVersionString": "2.0"})
    monkeypatch.setattr(versions, "_APP_DIRS", (tmp_path,))
    present, ver = versions.find_app(["GPT4All", "gpt4all"])  # second name matches
    assert present is True and ver == "2.0"


def test_find_app_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(versions, "_APP_DIRS", (tmp_path,))
    assert versions.find_app(["Nonexistent"]) == (False, None)


def test_brew_versions_skips_blank_lines(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/brew")
    out = "ollama 0.1.2\n\n   \nllama.cpp 4567\n"   # blank / whitespace lines → skipped
    monkeypatch.setattr(versions.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=out))
    assert versions._brew_versions("--formula") == {"ollama": "0.1.2", "llama.cpp": "4567"}


def test_cask_auto_updates_skips_empty_token(monkeypatch):
    payload = '{"casks":[{"token":"","auto_updates":true},{"token":"cursor","auto_updates":false}]}'
    monkeypatch.setattr(versions.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=payload))
    assert versions.cask_auto_updates(["cursor"]) == {"cursor": False}   # empty-token entry dropped
