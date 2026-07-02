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


def test_source_label_overrides_homebrew_logic():
    # source_label set → returned verbatim, bypassing the cask/formula/in_app logic entirely.
    assert _app(source_label="Flatpak", cask=True, formula=True).source == "Flatpak"
    # source_label unset (the default) → existing Homebrew logic still applies unchanged.
    assert _app(cask=True).source == "Homebrew (cask)"


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
    monkeypatch.setattr(apps.sys, "platform", "darwin")
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
    monkeypatch.setattr(apps.sys, "platform", "darwin")
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


# --------------------------------------------------------------------------- #
# scan() dispatch
# --------------------------------------------------------------------------- #
def test_scan_dispatch_unknown_platform_returns_empty(monkeypatch):
    monkeypatch.setattr(apps.sys, "platform", "win32")
    assert apps.scan() == []


def test_scan_dispatch_linux_routes_to_scan_linux(monkeypatch):
    monkeypatch.setattr(apps.sys, "platform", "linux")
    monkeypatch.setattr(apps, "_scan_linux", lambda: ["sentinel"])
    assert apps.scan() == ["sentinel"]


# --------------------------------------------------------------------------- #
# _linux_desktop_names
# --------------------------------------------------------------------------- #
def test_linux_desktop_names_reads_name_field(tmp_path, monkeypatch):
    present = tmp_path / "apps1"
    present.mkdir()
    (present / "ollama.desktop").write_text("[Desktop Entry]\nName=Ollama\nExec=ollama\n")
    # no Name= line at all — the inner loop must run to completion without breaking.
    (present / "noname.desktop").write_text("[Desktop Entry]\nExec=mystery\n")
    absent = tmp_path / "does-not-exist"       # never created — glob just finds nothing
    monkeypatch.setattr(apps, "_LINUX_DESKTOP_DIRS", (present, absent))
    assert apps._linux_desktop_names() == ["Ollama"]


def test_linux_desktop_names_dir_glob_oserror_skipped(monkeypatch):
    class _RaisingDir:
        def glob(self, pattern):
            raise OSError("boom")

    monkeypatch.setattr(apps, "_LINUX_DESKTOP_DIRS", (_RaisingDir(),))
    assert apps._linux_desktop_names() == []


def test_linux_desktop_names_file_read_oserror_skipped(tmp_path, monkeypatch):
    d = tmp_path / "apps"
    d.mkdir()
    (d / "bad.desktop").write_text("Name=Bad\n")
    monkeypatch.setattr(apps, "_LINUX_DESKTOP_DIRS", (d,))
    monkeypatch.setattr(apps.Path, "read_text", lambda self, **k: (_ for _ in ()).throw(OSError("nope")))
    assert apps._linux_desktop_names() == []


# --------------------------------------------------------------------------- #
# _flatpak_apps
# --------------------------------------------------------------------------- #
def test_flatpak_apps_absent_tool(monkeypatch):
    monkeypatch.setattr(apps.shutil, "which", lambda name: None)
    assert apps._flatpak_apps() == {}


def test_flatpak_apps_parses_output(monkeypatch):
    monkeypatch.setattr(apps.shutil, "which", lambda name: "/usr/bin/flatpak")

    class FakeResult:
        stdout = (
            "com.ollama.Ollama\tOllama\t0.5.0\n"
            "org.x.NoVersion\tNoVersion\t\n"
            "malformed-line-no-tabs\n"
            "org.x.NoName\t\t1.0\n"
        )

    monkeypatch.setattr(apps.subprocess, "run", lambda *a, **k: FakeResult())
    assert apps._flatpak_apps() == {"Ollama": "0.5.0", "NoVersion": None}


def test_flatpak_apps_subprocess_error(monkeypatch):
    monkeypatch.setattr(apps.shutil, "which", lambda name: "/usr/bin/flatpak")

    def raise_err(*a, **k):
        raise OSError("boom")

    monkeypatch.setattr(apps.subprocess, "run", raise_err)
    assert apps._flatpak_apps() == {}


# --------------------------------------------------------------------------- #
# _snap_apps
# --------------------------------------------------------------------------- #
def test_snap_apps_absent_tool(monkeypatch):
    monkeypatch.setattr(apps.shutil, "which", lambda name: None)
    assert apps._snap_apps() == {}


def test_snap_apps_parses_output(monkeypatch):
    monkeypatch.setattr(apps.shutil, "which", lambda name: "/usr/bin/snap")

    class FakeResult:
        stdout = (
            "Name    Version  Rev  Tracking       Publisher  Notes\n"
            "ollama  0.5.0    123  latest/stable  ollama     -\n"
            "\n"
        )

    monkeypatch.setattr(apps.subprocess, "run", lambda *a, **k: FakeResult())
    assert apps._snap_apps() == {"ollama": "0.5.0"}


def test_snap_apps_subprocess_error(monkeypatch):
    monkeypatch.setattr(apps.shutil, "which", lambda name: "/usr/bin/snap")

    def raise_err(*a, **k):
        raise OSError("boom")

    monkeypatch.setattr(apps.subprocess, "run", raise_err)
    assert apps._snap_apps() == {}


# --------------------------------------------------------------------------- #
# _scan_linux — catalog matching against .desktop / Flatpak / Snap
# --------------------------------------------------------------------------- #
def _linux_discovery(monkeypatch, desktop=(), flatpak=None, snap=None):
    monkeypatch.setattr(apps, "_linux_desktop_names", lambda: list(desktop))
    monkeypatch.setattr(apps, "_flatpak_apps", lambda: dict(flatpak or {}))
    monkeypatch.setattr(apps, "_snap_apps", lambda: dict(snap or {}))


def test_scan_linux_desktop_match(monkeypatch):
    _linux_discovery(monkeypatch, desktop=["LM Studio"])
    by = {a.label: a for a in apps._scan_linux()}
    lms = by["LM Studio"]
    assert lms.source_label == "app (.desktop)"
    assert lms.source == "app (.desktop)"
    assert lms.version is None


def test_scan_linux_flatpak_match_with_version(monkeypatch):
    _linux_discovery(monkeypatch, flatpak={"Ollama": "0.5.1"})
    by = {a.label: a for a in apps._scan_linux()}
    o = by["Ollama"]
    assert o.source_label == "Flatpak" and o.source == "Flatpak"
    assert o.version == "0.5.1"


def test_scan_linux_snap_match_with_version(monkeypatch):
    _linux_discovery(monkeypatch, snap={"cursor": "1.2.3"})
    by = {a.label: a for a in apps._scan_linux()}
    c = by["Cursor"]
    assert c.source_label == "Snap" and c.source == "Snap"
    assert c.version == "1.2.3"


def test_scan_linux_case_insensitive_matching(monkeypatch):
    _linux_discovery(monkeypatch, desktop=["chatgpt"])   # catalog label is "ChatGPT"
    labels = [a.label for a in apps._scan_linux()]
    assert "ChatGPT" in labels


def test_scan_linux_no_match_skipped(monkeypatch):
    _linux_discovery(monkeypatch)   # nothing installed anywhere
    assert apps._scan_linux() == []


def test_scan_linux_sort_order(monkeypatch):
    _linux_discovery(monkeypatch, desktop=["Ollama", "LM Studio", "Cursor"])
    result = apps._scan_linux()
    order = [(a.category, a.label) for a in result]
    # runner category before coding, and alphabetical within a category
    assert order.index(("runner", "LM Studio")) < order.index(("runner", "Ollama"))
    assert order.index(("runner", "Ollama")) < order.index(("coding", "Cursor"))


def test_scan_linux_via_scan_dispatch(monkeypatch):
    # confirms scan() itself routes to the real _scan_linux implementation on Linux.
    monkeypatch.setattr(apps.sys, "platform", "linux")
    _linux_discovery(monkeypatch, desktop=["Ollama"])
    labels = [a.label for a in apps.scan()]
    assert labels == ["Ollama"]
