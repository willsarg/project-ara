"""Minimal CLI rendering for ARA — semantic roles over raw ANSI. Stdlib only.

Mirrors the wmx-suite Console feel (accent/dim/gloss/section) but trimmed to
exactly what ARA's front door needs. Color only on a TTY without NO_COLOR;
otherwise plain text so piping stays clean.
"""
from __future__ import annotations

import os
import sys

RESET = "\033[0m"

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

    def emit(self, text: str = "") -> None:
        print(text, file=self.stream)
