# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""pythons.py — discover every interpreter on the system + its AI libraries."""
from __future__ import annotations

import os
import types

import ara.pythons as pythons
from ara.pythons import Interpreter


def _raise(exc=RuntimeError("boom")):
    """A callable that ignores its args and raises — for forcing except branches."""
    def _f(*a, **k):
        raise exc
    return _f


# --------------------------------------------------------------------------- #
# version sort key
# --------------------------------------------------------------------------- #
def test_ver_desc_orders_newer_first():
    assert pythons._ver_desc("3.14") < pythons._ver_desc("3.9")  # 3.14 sorts before 3.9
    assert pythons._ver_desc("3.12.4") < pythons._ver_desc("3.12.1")


def test_ver_desc_handles_none_and_garbage():
    assert pythons._ver_desc(None) == ()
    assert pythons._ver_desc("not.a.version") == ()


# --------------------------------------------------------------------------- #
# Interpreter properties
# --------------------------------------------------------------------------- #
def test_ai_present_filters_absent():
    i = Interpreter("p", "p", "venv", ai_libs={"torch": "2.1", "vllm": None})
    assert i.ai_present == {"torch": "2.1"}


def test_caution_macos_system_always():
    i = Interpreter("/usr/bin/python3", "/usr/bin/python3", "macOS system")
    assert i.caution == pythons._CAUTION["macOS system"]


def test_caution_externally_managed_known_origin():
    i = Interpreter("p", "p", "Homebrew", externally_managed=True)
    assert i.caution == pythons._CAUTION["Homebrew"]


def test_caution_externally_managed_unknown_origin_uses_default():
    i = Interpreter("p", "p", "other", externally_managed=True)
    assert i.caution == "externally managed — use a venv, not here"


def test_caution_none_when_unmanaged():
    i = Interpreter("p", "p", "pyenv", externally_managed=False)
    assert i.caution is None


# --------------------------------------------------------------------------- #
# origin classification
# --------------------------------------------------------------------------- #
def test_origin_classification():
    home = os.path.expanduser("~")
    cases = {
        f"{home}/.pyenv/versions/3.12.0/bin/python3": "pyenv",
        f"{home}/miniconda3/bin/python3": "conda",
        f"{home}/.asdf/installs/python/3.12.0/bin/python3": "asdf",
        f"{home}/.local/share/uv/python/cpython-3.12/bin/python3": "uv",
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3": "python.org",
        "/opt/homebrew/bin/python3.12": "Homebrew",
        "/usr/bin/python3": "macOS system",
    }
    for real, expected in cases.items():
        assert pythons._origin(real, [real]) == expected, real


def test_origin_classification_windows_paths():
    # Backslash paths normalize to '/', so the same rules classify Windows installs.
    cases = {
        r"C:\Users\dev\AppData\Local\Programs\Python\Python312\python.exe": "python.org",
        r"C:\Program Files\Python313\python.exe": "python.org",
        r"C:\Program Files (x86)\Python311\python.exe": "python.org",
        r"C:\Users\dev\.pyenv\pyenv-win\versions\3.12.0\python.exe": "pyenv",
        r"C:\Users\dev\miniconda3\python.exe": "conda",
        r"C:\Users\dev\AppData\Roaming\uv\python\cpython-3.12\python.exe": "uv",
    }
    for real, expected in cases.items():
        assert pythons._origin(real, [real]) == expected, real


def test_origin_venv_via_pyvenv_cfg(tmp_path):
    env = tmp_path / "myenv"
    (env / "bin").mkdir(parents=True)
    (env / "pyvenv.cfg").write_text("home = /usr\n")
    py = env / "bin" / "python"
    py.write_text("")
    assert pythons._origin(str(py), [str(py)]) == "venv"


def test_origin_other_fallback(tmp_path):
    py = tmp_path / "weird" / "python3"
    py.parent.mkdir(parents=True)
    py.write_text("")
    assert pythons._origin(str(py), [str(py)]) == "other"


def test_is_venv_true_and_false(tmp_path):
    env = tmp_path / "v"
    (env / "bin").mkdir(parents=True)
    (env / "pyvenv.cfg").write_text("")
    assert pythons._is_venv(str(env / "bin" / "python")) is True
    assert pythons._is_venv("/usr/bin/python3") is False


# --------------------------------------------------------------------------- #
# default-interpreter resolution (ARA's venv stripped)
# --------------------------------------------------------------------------- #
def test_user_default_real_strips_venv(monkeypatch):
    # Build fixtures with host conventions so the product's os.pathsep / os.name logic
    # strips the right dir on both POSIX and Windows.
    sub = "Scripts" if os.name == "nt" else "bin"
    venv = os.path.normpath("/tmp/venv")
    vbin = os.path.join(venv, sub)
    other = os.path.normpath("/usr/bin")
    monkeypatch.setenv("VIRTUAL_ENV", venv)
    monkeypatch.setenv("PATH", os.pathsep.join([vbin, other]))
    monkeypatch.setattr(pythons.os.path, "realpath", lambda p, *a, **k: p)
    seen = {}

    def fake_which(name, path=None):
        seen["path"] = path
        py_name = "python" if os.name == "nt" else "python3"
        return os.path.join(other, py_name) if name == py_name else None

    monkeypatch.setattr("shutil.which", fake_which)
    py_name = "python" if os.name == "nt" else "python3"
    assert pythons._user_default_real() == os.path.join(other, py_name)
    parts = seen["path"].split(os.pathsep)
    assert vbin not in parts   # the venv's bin was stripped


def test_user_default_real_strips_windows_scripts_dir(monkeypatch):
    # On Windows the venv interpreter lives in <env>\Scripts, not <env>/bin.
    monkeypatch.setattr(pythons.os, "name", "nt")
    sep = os.pathsep
    monkeypatch.setenv("PATH", sep.join(["/venv/Scripts", "/usr/bin"]))
    monkeypatch.setenv("VIRTUAL_ENV", "/venv")
    monkeypatch.setattr(pythons.os.path, "realpath", lambda p, *a, **k: p)
    seen = {}

    def fake_which(name, path=None):
        seen["path"] = path
        return "/usr/bin/python" if name == "python" else None

    monkeypatch.setattr("shutil.which", fake_which)
    assert pythons._user_default_real() == "/usr/bin/python"
    assert "/venv/Scripts" not in seen["path"]


def test_user_default_real_none_when_no_python(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr("shutil.which", lambda name, path=None: None)
    assert pythons._user_default_real() is None


# --------------------------------------------------------------------------- #
# display path selection
# --------------------------------------------------------------------------- #
def test_display_path_prefers_on_path_then_shortest():
    invocations = {"/opt/homebrew/bin/python3.12", "/usr/local/bin/python3"}
    path_dirs = {"/usr/local/bin"}
    # the on-PATH one wins even though it's not the shortest string
    assert pythons._display_path(invocations, path_dirs) == "/usr/local/bin/python3"


def test_display_path_shortest_when_none_on_path():
    invocations = {"/a/very/long/path/python3", "/short/python3"}
    assert pythons._display_path(invocations, set()) == "/short/python3"


# --------------------------------------------------------------------------- #
# per-interpreter probe (subprocess)
# --------------------------------------------------------------------------- #
def test_probe_parses_output(monkeypatch):
    payload = '{"v": "3.12.4", "libs": {"torch": "2.1.0", "vllm": null}, "em": true}'
    monkeypatch.setattr(pythons, "_run", lambda cmd, timeout=8: "noise\n" + payload + "\n")
    ver, libs, em = pythons._probe("/usr/bin/python3")
    assert ver == "3.12.4"
    assert libs == {"torch": "2.1.0", "vllm": None}
    assert em is True


def test_probe_blank_on_no_output(monkeypatch):
    monkeypatch.setattr(pythons, "_run", lambda cmd, timeout=8: None)
    ver, libs, em = pythons._probe("/usr/bin/python3")
    assert ver is None and em is False
    assert set(libs) == set(pythons._AI_LIBS) and all(v is None for v in libs.values())


def test_probe_blank_on_bad_json(monkeypatch):
    monkeypatch.setattr(pythons, "_run", lambda cmd, timeout=8: "not json")
    ver, libs, em = pythons._probe("/usr/bin/python3")
    assert ver is None and all(v is None for v in libs.values())


# --------------------------------------------------------------------------- #
# candidate discovery on a controlled PATH
# --------------------------------------------------------------------------- #
def test_run_none_on_failure():
    assert pythons._run(["definitely-not-a-real-binary-xyz"]) is None


def test_known_patterns_covers_standard_homes():
    # The product returns host-appropriate patterns: Windows globs on nt, POSIX paths
    # elsewhere.  Assert representative entries for each host to keep the test portable.
    pats = pythons._known_patterns()
    if os.name == "nt":
        # Windows: every pattern is a python.exe glob; check a representative one.
        assert any(r"Programs\Python" in p and p.endswith("python.exe") for p in pats)
        assert any("pyenv-win" in p for p in pats)
    else:
        assert "/usr/bin/python3" in pats
        assert any("/.pyenv/" in p for p in pats)
        assert any("/opt/homebrew/" in p for p in pats)
        assert any(("conda" in p or "miniforge" in p) for p in pats)


def test_py_name_matches_windows_exe():
    assert pythons._PY_NAME.match("python.exe")
    assert pythons._PY_NAME.match("python3.exe")
    assert pythons._PY_NAME.match("python3.12.exe")
    assert not pythons._PY_NAME.match("pythonw.exe")     # windowed interpreter — excluded
    assert not pythons._PY_NAME.match("python3-config")


def test_windows_patterns_content(monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\dev\AppData\Local")
    monkeypatch.setenv("APPDATA", r"C:\Users\dev\AppData\Roaming")
    monkeypatch.setenv("ProgramFiles", r"C:\Program Files")
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)   # unset → its entry is dropped
    pats = pythons._windows_patterns(r"C:\Users\dev")
    assert all(p.endswith("python.exe") for p in pats)
    assert any(r"Programs\Python" in p for p in pats)
    assert any(r"pyenv-win" in p for p in pats)
    assert any(r"uv\python" in p for p in pats)
    # no entry collapsed to a leading backslash from an empty env var
    assert all(not p.startswith("\\") for p in pats)
    # the ProgramFiles(x86) entry was dropped entirely
    assert not any("(x86)" in p for p in pats)


def test_known_patterns_dispatches_to_windows(monkeypatch):
    # os.name='nt' flips pathlib to WindowsPath (uninstantiable on posix), so stub Path.
    monkeypatch.setattr(pythons.os, "name", "nt")
    monkeypatch.setattr(pythons, "Path",
                        types.SimpleNamespace(home=lambda: r"C:\Users\dev"))
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\dev\AppData\Local")
    pats = pythons._known_patterns()
    assert pats and all(p.endswith("python.exe") for p in pats)


def test_known_patterns_dispatches_to_posix(monkeypatch):
    # Mirror of the windows-dispatch test: force the non-nt branch regardless of host so the POSIX
    # return (pythons.py:142) is covered when the suite runs ON Windows too, not only on a POSIX box.
    monkeypatch.setattr(pythons.os, "name", "posix")
    monkeypatch.setattr(pythons, "Path",
                        types.SimpleNamespace(home=lambda: "/home/dev"))
    pats = pythons._known_patterns()
    assert "/usr/bin/python3" in pats
    assert any("/.pyenv/" in p for p in pats)


def test_candidates_skips_a_candidate_that_errors(monkeypatch):
    # A candidate that blows up mid-inspection must be skipped, not crash discovery — covers the
    # per-candidate except (pythons.py:241-242) deterministically on every OS, rather than relying
    # on the host filesystem happening to raise.
    monkeypatch.setenv("PATH", "/fakebin")
    monkeypatch.setattr(pythons.os, "listdir", lambda d: ["python3"])   # one candidate enters
    monkeypatch.setattr(pythons, "_known_patterns", lambda: [])         # no glob candidates
    monkeypatch.setattr(pythons.os.path, "basename", _raise())          # inspecting it raises
    assert pythons._candidates() == {}                                  # swallowed → empty, no crash


# --------------------------------------------------------------------------- #
# _is_executable — Windows PATHEXT gate vs. POSIX exec-bit
# --------------------------------------------------------------------------- #
def test_is_executable_windows_uses_pathext(monkeypatch):
    monkeypatch.setattr(pythons.os, "name", "nt")
    monkeypatch.setenv("PATHEXT", ".EXE;.BAT;.CMD")
    assert pythons._is_executable(r"C:\x\python.exe") is True
    assert pythons._is_executable(r"C:\x\python3") is False       # no extension
    assert pythons._is_executable(r"C:\x\python3.11") is False    # ".11" not an exec ext
    assert pythons._is_executable(r"C:\x\run.bat") is True        # .bat is allowed


def test_is_executable_posix_uses_access(monkeypatch):
    monkeypatch.setattr(pythons.os, "name", "posix")
    monkeypatch.setattr(pythons.os, "access", lambda p, mode: p == "/x/good" and mode == os.X_OK)
    assert pythons._is_executable("/x/good") is True
    assert pythons._is_executable("/x/bad") is False


def test_candidates_filters_globs_and_skips(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    good = bindir / "python3"
    good.write_text("#!/bin/sh\n")
    good.chmod(0o755)
    noexec = bindir / "python3.11"            # matches the name but isn't executable → skip
    noexec.write_text("")
    noexec.chmod(0o644)
    (bindir / "python3-config").write_text("")  # name mismatch → never a candidate
    globdir = tmp_path / "glob"
    globdir.mkdir()
    globpy = globdir / "python3"               # found via _known_patterns glob, off PATH
    globpy.write_text("#!/bin/sh\n")
    globpy.chmod(0o755)

    # PATH includes a bogus dir → os.listdir raises and is swallowed.
    monkeypatch.setenv("PATH", os.pathsep.join([str(bindir), str(tmp_path / "missing")]))
    # glob returns a real interpreter AND a non-interpreter (drops at the basename re-check).
    monkeypatch.setattr(pythons, "_known_patterns",
                        lambda: [str(globpy), str(bindir / "python3-config")])

    # Route through the new _is_executable helper — platform-independent mock.
    noexec_real = os.path.realpath(str(noexec))
    monkeypatch.setattr(pythons, "_is_executable", lambda p: os.path.realpath(p) != noexec_real)

    groups = pythons._candidates()
    reals = set(groups)
    assert os.path.realpath(str(good)) in reals       # PATH executable found
    assert os.path.realpath(str(globpy)) in reals      # glob-found, off PATH
    assert os.path.realpath(str(noexec)) not in reals  # non-executable skipped
    assert all("config" not in p for grp in groups.values() for p in grp)


def test_count_matches_candidate_groups(monkeypatch):
    monkeypatch.setattr(pythons, "_candidates", lambda: {"/a": {"/a"}, "/b": {"/b"}})
    assert pythons.count() == 2


# --------------------------------------------------------------------------- #
# discover() assembly + ordering
# --------------------------------------------------------------------------- #
def test_discover_assembles_and_orders(monkeypatch):
    groups = {
        "/usr/bin/python3": {"/usr/bin/python3"},                       # macOS system
        "/opt/homebrew/bin/python3.12": {"/opt/homebrew/bin/python3.12"},  # Homebrew (default)
        "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3":
            {"/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"},  # python.org
    }
    monkeypatch.setattr(pythons, "_candidates", lambda: groups)
    monkeypatch.setattr(pythons, "_user_default_real", lambda: "/opt/homebrew/bin/python3.12")

    probes = {
        "/usr/bin/python3": ("3.9.6", {"torch": None}, False),
        "/opt/homebrew/bin/python3.12": ("3.12.4", {"torch": "2.1.0"}, True),
        "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3":
            ("3.13.0", {"torch": None}, False),
    }
    monkeypatch.setattr(pythons, "_probe", lambda real: probes[real])

    out = pythons.discover(probe=True)
    # python.org sorts before Homebrew before macOS system (per _ORIGIN_ORDER)
    assert [i.origin for i in out] == ["python.org", "Homebrew", "macOS system"]
    hb = next(i for i in out if i.origin == "Homebrew")
    assert hb.is_default is True
    assert hb.version == "3.12.4"
    assert hb.ai_present == {"torch": "2.1.0"}
    assert hb.externally_managed is True


def test_discover_probe_false_skips_subprocess(monkeypatch):
    monkeypatch.setattr(pythons, "_candidates", lambda: {"/usr/bin/python3": {"/usr/bin/python3"}})
    monkeypatch.setattr(pythons, "_user_default_real", lambda: None)

    def _boom(real):
        raise AssertionError("_probe must not run when probe=False")

    monkeypatch.setattr(pythons, "_probe", _boom)
    out = pythons.discover(probe=False)
    assert len(out) == 1
    assert out[0].version is None and out[0].ai_libs == {}


# --------------------------------------------------------------------------- #
# defensive guards — python is fickle; bad input must degrade, not crash
# --------------------------------------------------------------------------- #
def test_ver_desc_swallows_non_string_input():
    # a non-str version (type says str|None, but reality leaks) → () not a crash
    assert pythons._ver_desc(3.14) == ()


def test_is_venv_false_when_path_raises(monkeypatch):
    monkeypatch.setattr(pythons, "Path", _raise(OSError("symlink loop")))
    assert pythons._is_venv("/x/bin/python") is False


def test_candidates_swallows_per_entry_errors(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    py = bindir / "python3"
    py.write_text("#!/bin/sh\n")
    py.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setattr(pythons, "_known_patterns", lambda: [])
    # realpath blows up on our candidate specifically (scoped so we don't break
    # unrelated path resolution) → the entry is dropped silently, no crash.
    real_realpath = pythons.os.path.realpath

    def boom_on_candidate(p, *a, **k):
        if str(p) == str(py):
            raise OSError("boom")
        return real_realpath(p, *a, **k)

    monkeypatch.setattr(pythons.os.path, "realpath", boom_on_candidate)
    assert pythons._candidates() == {}


# --------------------------------------------------------------------------- #
# manager_of (who owns the interpreter, for "don't install here" guidance)
# --------------------------------------------------------------------------- #
def test_manager_of():
    assert pythons.manager_of("macOS system", False) == "Apple"
    assert pythons.manager_of("Homebrew", True) == "Homebrew"
    assert pythons.manager_of("uv", True) == "uv"
    assert pythons.manager_of("other", True) == "the system"   # unknown managed origin
    assert pythons.manager_of("pyenv", False) is None          # user-managed → free to use
