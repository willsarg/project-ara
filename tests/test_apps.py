# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""apps.py — curated AI/ML app inventory + drift/duplicate classification."""
from __future__ import annotations

from ara import apps
from ara.apps import App


def _app(**over) -> App:
    base = dict(label="X", category="runner", in_app=False, cask=False, formula=False,
                version=None, brew_recorded=None, cask_token=None, installed_at=None)
    base.update(over)
    return App(**base)


# --------------------------------------------------------------------------- #
# homebrew / source
# --------------------------------------------------------------------------- #
def test_homebrew_property():
    assert _app(cask=True).homebrew is True
    assert _app(formula=True).homebrew is True
    assert _app(in_app=True).homebrew is False


def test_source_variants():
    assert _app(cask=True, formula=True).source == "Homebrew (cask + formula)"
    assert _app(cask=True).source == "Homebrew (cask)"
    assert _app(formula=True, in_app=True).source == "Homebrew (formula) + separate app"
    assert _app(formula=True).source == "Homebrew (formula)"
    assert _app(in_app=True).source == "app (not via Homebrew)"


# --------------------------------------------------------------------------- #
# drift (cask GUI self-updated past brew's frozen receipt)
# --------------------------------------------------------------------------- #
def test_drift_true_when_app_version_diverges():
    a = _app(cask=True, in_app=True, version="0.3.5", brew_recorded="0.3.0")
    assert a.drift is True


def test_drift_false_when_versions_match():
    a = _app(cask=True, in_app=True, version="0.3.0", brew_recorded="0.3.0")
    assert a.drift is False


def test_drift_false_for_cli_only_cask():
    # no .app present → a CLI cask can't "drift"
    a = _app(cask=True, in_app=False, version="0.3.5", brew_recorded="0.3.0")
    assert a.drift is False


def test_drift_false_for_plain_formula():
    a = _app(formula=True, version="1.0", brew_recorded=None).drift
    assert a is False


# --------------------------------------------------------------------------- #
# duplicate (two independent installs of the same tool)
# --------------------------------------------------------------------------- #
def test_duplicate_formula_plus_cask():
    assert _app(formula=True, cask=True).duplicate is True


def test_duplicate_formula_plus_handapp():
    assert _app(formula=True, in_app=True).duplicate is True


def test_not_duplicate_cask_with_its_own_app():
    assert _app(cask=True, in_app=True).duplicate is False


def test_not_duplicate_formula_only():
    assert _app(formula=True).duplicate is False


# --------------------------------------------------------------------------- #
# _install_time
# --------------------------------------------------------------------------- #
def test_install_time_from_app_bundle(tmp_path, monkeypatch):
    app = tmp_path / "Foo.app"
    app.mkdir()
    monkeypatch.setattr(apps, "_APP_DIRS", (tmp_path,))
    t = apps._install_time(["Foo"], [], in_app=True)
    assert isinstance(t, float) and t > 0


def test_install_time_from_brew_dir(tmp_path, monkeypatch):
    caskroom = tmp_path / "Caskroom" / "ollama"
    caskroom.mkdir(parents=True)
    (caskroom / "0.1.0").mkdir()
    monkeypatch.setattr(apps, "_BREW_PREFIX", tmp_path)
    monkeypatch.setattr(apps, "_APP_DIRS", (tmp_path / "none",))
    t = apps._install_time([], ["ollama"], in_app=False)
    assert isinstance(t, float) and t > 0


def test_install_time_none_when_nothing_found(tmp_path, monkeypatch):
    monkeypatch.setattr(apps, "_APP_DIRS", (tmp_path / "none",))
    monkeypatch.setattr(apps, "_BREW_PREFIX", tmp_path / "none")
    assert apps._install_time(["Foo"], ["bar"], in_app=False) is None


# --------------------------------------------------------------------------- #
# scan() — catalog matching + version/drift assembly
# --------------------------------------------------------------------------- #
def test_scan_assembles_cask_formula_and_handapp(monkeypatch):
    # LM Studio: cask whose .app self-updated past brew → drift.
    # llama.cpp: formula CLI (no app). Draw Things: hand-installed .app (no brew).
    monkeypatch.setattr(apps.versions, "brew_formulae", lambda: {"llama.cpp": "4567"})
    monkeypatch.setattr(apps.versions, "brew_casks", lambda: {"lm-studio": "0.3.0"})

    def fake_find_app(bundles):
        if "LM Studio" in bundles:
            return True, "0.3.5"          # installed app ahead of brew's 0.3.0
        if "Draw Things" in bundles:
            return True, "1.1"
        return False, None

    monkeypatch.setattr(apps.versions, "find_app", fake_find_app)
    monkeypatch.setattr(apps, "_install_time", lambda *a: 100.0)

    by = {a.label: a for a in apps.scan()}

    lms = by["LM Studio"]
    assert lms.cask and lms.in_app and lms.version == "0.3.5" and lms.brew_recorded == "0.3.0"
    assert lms.drift is True and lms.cask_token == "lm-studio"

    llama = by["llama.cpp"]
    assert llama.formula and not llama.in_app and llama.version == "4567"
    assert llama.brew_recorded is None and llama.source == "Homebrew (formula)"

    dt = by["Draw Things"]
    assert dt.in_app and not dt.homebrew and dt.version == "1.1"
    assert dt.source == "app (not via Homebrew)"


def test_scan_skips_absent_and_orders_by_category(monkeypatch):
    monkeypatch.setattr(apps.versions, "brew_formulae", lambda: {})
    monkeypatch.setattr(apps.versions, "brew_casks", lambda: {"cursor": "2.0"})
    monkeypatch.setattr(apps.versions, "find_app",
                        lambda bundles: (True, "1.0") if "Ollama" in bundles else (False, None))
    monkeypatch.setattr(apps, "_install_time", lambda *a: 100.0)

    result = apps.scan()
    labels = [a.label for a in result]
    assert "Ollama" in labels and "Cursor" in labels
    assert all(a.label not in ("ChatGPT", "MLX") for a in result)  # absent → skipped
    # ordered by category: runner (Ollama) before coding (Cursor)
    cats = [a.category for a in result]
    assert cats.index("runner") < cats.index("coding")


def test_install_time_in_app_not_found_falls_through_to_brew(tmp_path, monkeypatch):
    # in_app=True but the .app isn't in any Applications dir → fall through to a brew dir.
    cellar = tmp_path / "Cellar" / "llama.cpp"
    cellar.mkdir(parents=True)
    (cellar / "1.0").mkdir()
    monkeypatch.setattr(apps, "_APP_DIRS", (tmp_path / "Applications",))   # empty
    monkeypatch.setattr(apps, "_BREW_PREFIX", tmp_path)
    t = apps._install_time(["NotInstalled"], ["llama.cpp"], in_app=True)
    assert isinstance(t, float) and t > 0


def test_install_time_brew_dir_iterdir_error_is_none(tmp_path, monkeypatch):
    token = tmp_path / "Caskroom" / "ollama"
    token.mkdir(parents=True)
    monkeypatch.setattr(apps, "_BREW_PREFIX", tmp_path)
    monkeypatch.setattr(apps, "_APP_DIRS", (tmp_path / "none",))
    monkeypatch.setattr(apps.Path, "iterdir", _raise_oserror)
    assert apps._install_time([], ["ollama"], in_app=False) is None


def _raise_oserror(self):
    raise OSError("iterdir blew up")
