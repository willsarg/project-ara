# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Minimal CLI rendering for ARA — semantic roles over raw ANSI. Stdlib only.

Uses the established engine-console feel (accent/dim/gloss/section) but trimmed to
exactly what ARA's front door needs. Color only on a TTY without NO_COLOR;
otherwise plain text so piping stays clean.
"""
from __future__ import annotations

import os
import sys

RESET = "\033[0m"


def _ensure_utf8(stream) -> None:
    """Best-effort: make *stream* emit UTF-8 without ever crashing on a glyph.

    Windows consoles default to a legacy codepage (e.g. cp1252), where ARA's
    bullets and separators (``●``, ``·``, ``▸``) raise ``UnicodeEncodeError`` on
    a plain ``print``. Reconfiguring to UTF-8 with ``errors="replace"`` makes
    output crash-proof; on a capable terminal (Windows Terminal, modern SSH) the
    glyphs also render correctly. No-op on streams that don't support it (e.g.
    an already-UTF-8 stream, or a plain buffer with no ``reconfigure``).
    """
    reconfigure = getattr(stream, "reconfigure", None)
    encoding = (getattr(stream, "encoding", "") or "").lower()
    if reconfigure is None or encoding.startswith("utf"):
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, LookupError):
        # A stream may refuse mid-flight reconfiguration; leave it as-is.
        pass

# Semantic role -> ANSI SGR code.
ROLES: dict[str, str] = {
    "accent": "35",   # magenta
    "dim": "2",       # dim
    "gloss": "2",     # dim/grey
    "header": "36",   # cyan
    "good": "32",     # green
    "warn": "33",     # yellow
    "bad": "31",      # red
    "metric": "36",   # cyan
}


class Console:
    """Holds color/verbose state and a stream; styles and emits text."""

    def __init__(self, color: bool, verbose: bool = False, stream=None):
        self.color = color
        self.verbose = verbose
        self.stream = stream if stream is not None else sys.stdout
        _ensure_utf8(self.stream)

    @classmethod
    def from_env(cls, *, stream=None, verbose: bool = False) -> "Console":
        stream = stream if stream is not None else sys.stdout
        is_tty = bool(getattr(stream, "isatty", lambda: False)())
        # NO_COLOR convention: disabled when the var is present. https://no-color.org
        color = is_tty and "NO_COLOR" not in os.environ
        return cls(color=color, verbose=verbose, stream=stream)

    def style(self, role: str, text: str) -> str:
        """Wrap *text* in the role's ANSI code iff color; unknown role = no-op."""
        if not self.color:
            return text
        code = ROLES.get(role)
        return f"\033[{code}m{text}{RESET}" if code else text

    def section(self, title: str) -> str:
        return self.style("header", title)

    def field(self, label: str, value: str, gloss: str | None = None, *,
              label_width: int = 12, value_role: str = "metric") -> str:
        """One aligned 'label   value   gloss' row."""
        line = "  " + self.style("dim", f"{label:<{label_width}}") + self.style(value_role, value)
        if gloss:
            line += "   " + self.style("gloss", gloss)
        return line

    def emit(self, text: str = "") -> None:
        print(text, file=self.stream)
