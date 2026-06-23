# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Console: color gating and semantic styling."""
from __future__ import annotations

import io

from ara.ui import RESET, ROLES, Console, _ensure_utf8


class FakeReconfigurable:
    """A stream stand-in that records reconfigure() calls and a writable encoding.

    (Real io.StringIO has a read-only ``encoding``, so it can't model a legacy
    codepage; ``_ensure_utf8`` only touches ``.encoding`` and ``.reconfigure``.)
    """

    def __init__(self, encoding: str = "cp1252", raises: Exception | None = None):
        self.encoding = encoding
        self._raises = raises
        self.reconfigured: dict | None = None

    def reconfigure(self, **kwargs):
        if self._raises is not None:
            raise self._raises
        self.reconfigured = kwargs
        self.encoding = kwargs.get("encoding", self.encoding)


class FakeTTY(io.StringIO):
    def __init__(self, isatty: bool):
        super().__init__()
        self._isatty = isatty

    def isatty(self) -> bool:
        return self._isatty


def test_from_env_colors_on_a_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    c = Console.from_env(stream=FakeTTY(True))
    assert c.color is True


def test_from_env_no_color_when_not_a_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    c = Console.from_env(stream=FakeTTY(False))
    assert c.color is False


def test_from_env_respects_no_color_env(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    c = Console.from_env(stream=FakeTTY(True))
    assert c.color is False


def test_style_wraps_known_role_when_color():
    c = Console(color=True, stream=io.StringIO())
    out = c.style("good", "ok")
    assert out == f"\033[{ROLES['good']}m" + "ok" + RESET


def test_style_is_noop_without_color():
    c = Console(color=False, stream=io.StringIO())
    assert c.style("good", "ok") == "ok"


def test_style_unknown_role_is_noop_even_with_color():
    c = Console(color=True, stream=io.StringIO())
    assert c.style("nonsense", "ok") == "ok"


def test_field_alignment_and_gloss():
    c = Console(color=False, stream=io.StringIO())
    line = c.field("chip", "M4 Pro", "fast", label_width=12)
    assert "chip" in line and "M4 Pro" in line and "fast" in line
    # label is left-padded to label_width inside the leading two spaces
    assert line.startswith("  chip" + " " * 8)


def test_field_without_gloss_has_no_trailing_separator():
    c = Console(color=False, stream=io.StringIO())
    line = c.field("os", "macOS")
    assert "macOS" in line
    assert "   " not in line.split("macOS")[-1]


def test_section_uses_header_role_when_color():
    c = Console(color=True, stream=io.StringIO())
    assert c.section("SYSTEM") == f"\033[{ROLES['header']}mSYSTEM{RESET}"


def test_ensure_utf8_reconfigures_a_legacy_codepage_stream():
    s = FakeReconfigurable(encoding="cp1252")
    _ensure_utf8(s)
    assert s.reconfigured == {"encoding": "utf-8", "errors": "replace"}


def test_ensure_utf8_skips_a_stream_already_on_utf8():
    s = FakeReconfigurable(encoding="utf-8")
    _ensure_utf8(s)
    assert s.reconfigured is None


def test_ensure_utf8_noop_when_stream_cannot_reconfigure():
    # Plain StringIO has no reconfigure attribute — must not raise.
    _ensure_utf8(io.StringIO())


def test_ensure_utf8_swallows_reconfigure_failure():
    s = FakeReconfigurable(encoding="cp1252", raises=ValueError("nope"))
    _ensure_utf8(s)  # must not propagate
    assert s.reconfigured is None


def test_console_reconfigures_its_stream_to_utf8_on_construction():
    s = FakeReconfigurable(encoding="cp1252")
    Console(color=False, stream=s)
    assert s.reconfigured == {"encoding": "utf-8", "errors": "replace"}


def test_emit_writes_line_to_stream():
    buf = io.StringIO()
    c = Console(color=False, stream=buf)
    c.emit("hello")
    c.emit()
    assert buf.getvalue() == "hello\n\n"
