"""Console: color gating and semantic styling."""
from __future__ import annotations

import io

from ara.ui import RESET, ROLES, Console


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


def test_emit_writes_line_to_stream():
    buf = io.StringIO()
    c = Console(color=False, stream=buf)
    c.emit("hello")
    c.emit()
    assert buf.getvalue() == "hello\n\n"
