# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Shared CLI rendering schema for ARA's native MLX engine.

One ``Console`` owns all presentation: color/verbose policy plus a small set of
layout *primitives* (``field``, ``section``, ``table``, ``glyph``, ``next_block``,
``guidance``, ``status_line``, ``raw``) built on semantic *roles* rather than raw
ANSI. Primitives return strings; callers compose them and ``emit`` the result.

The visual source of truth is ``docs/mockups/cli-output-mockup.html`` (approved
inline-gloss default + Normal/Verbose). Color is applied only on a TTY with
neither ``--no-color`` nor ``NO_COLOR``; otherwise output is stable plain text so
goldens and piping stay clean.

Stdlib only — no new dependencies.
"""
from __future__ import annotations

import os
import sys

RESET = "\033[0m"

# Semantic role -> ANSI SGR code. Color is emitted only when console.color.
ROLES: dict[str, str] = {
    "label": "2",     # dim
    "value": "1",     # bold (default-ish, bold for emphasis)
    "gloss": "2",     # dim/grey
    "header": "36",   # cyan
    "accent": "35",   # magenta
    "good": "32",     # green
    "warn": "33",     # yellow
    "bad": "31",      # red
    "metric": "36",   # cyan
    "dim": "2",       # dim
}

# Field label column width (matches the mockup's pad(label, 16)).
LABEL_WIDTH = 16
# Gap between a value and its inline gloss, and the table column gutter.
GLOSS_GAP = "   "
GUTTER = "   "


class Console:
    """Holds color/verbose state and a stream; renders schema primitives."""

    def __init__(self, color: bool, verbose: bool, stream=None):
        self.color = color
        self.verbose = verbose
        self.stream = stream if stream is not None else sys.stdout

    @classmethod
    def from_args(cls, *, stream=None, no_color: bool = False,
                  verbose: bool = False) -> "Console":
        stream = stream if stream is not None else sys.stdout
        is_tty = bool(getattr(stream, "isatty", lambda: False)())
        # NO_COLOR convention: disabled when the var is PRESENT, regardless of
        # value (including empty string). https://no-color.org
        color = is_tty and not no_color and "NO_COLOR" not in os.environ
        return cls(color=color, verbose=verbose, stream=stream)

    # ------------------------------------------------------------------ #
    # styling
    # ------------------------------------------------------------------ #
    def style(self, role: str, text: str) -> str:
        """Wrap *text* in the role's ANSI code iff color; else return unchanged.

        An unknown role is a no-op (text returned unchanged) even with color.
        """
        if not self.color:
            return text
        code = ROLES.get(role)
        if code is None:
            return text
        return f"\033[{code}m{text}{RESET}"

    # ------------------------------------------------------------------ #
    # primitives (all return strings)
    # ------------------------------------------------------------------ #
    def field(self, label: str, value: str, gloss: str | None = None, *,
              indent: int = 0, value_role: str = "value") -> str:
        pad = max(0, LABEL_WIDTH - indent)
        lbl = (" " * indent) + self.style("label", f"{label:<{pad}}")
        val = self.style(value_role, value)
        # Mockup form: 16-wide label, then ": " (colon at a fixed column).
        line = f"{lbl}: {val}"
        if gloss:
            line += GLOSS_GAP + self.style("gloss", gloss)
        return line

    def section(self, title: str) -> str:
        return self.style("header", title)

    def table(self, columns, rows) -> str:
        """Render an aligned table.

        ``columns``: list of ``(header, align, role)`` where align in {"l","r"}.
        ``rows``: list of tuples; each cell is a plain ``str`` or ``(text, role)``.

        Column widths are computed from header + cell text (unstyled), the
        single source of all alignment. The header row uses the ``header`` role;
        per-cell roles are applied on top of that width.
        """
        headers = [c[0] for c in columns]
        aligns = [c[1] for c in columns]
        col_roles = [c[2] for c in columns]

        def cell_text(cell) -> str:
            return cell[0] if isinstance(cell, tuple) else cell

        widths = []
        for ci, (header, _, _) in enumerate(columns):
            w = len(header)
            for row in rows:
                if ci < len(row):
                    w = max(w, len(cell_text(row[ci])))
            widths.append(w)

        def lay(text: str, ci: int) -> str:
            w = widths[ci]
            if aligns[ci] == "r":
                return text.rjust(w)
            return text.ljust(w)

        lines = []
        header_cells = [self.style("header", lay(headers[ci], ci))
                        for ci in range(len(columns))]
        lines.append(GUTTER.join(header_cells))
        for row in rows:
            out_cells = []
            for ci in range(len(columns)):
                cell = row[ci] if ci < len(row) else ""
                if isinstance(cell, tuple):
                    text, role = cell
                else:
                    text, role = cell, col_roles[ci]
                out_cells.append(self.style(role, lay(text, ci)))
            lines.append(GUTTER.join(out_cells))
        return "\n".join(lines)

    def glyph(self, status: str) -> str:
        sym = {"ok": "✓", "bad": "✗", "warn": "⚠"}[status]
        role = {"ok": "good", "bad": "bad", "warn": "warn"}[status]
        return self.style(role, sym)

    def next_block(self, items) -> str:
        """A "next" section header + aligned accent-cmd + gloss rows."""
        if not items:
            return "\n" + self.section("next")
        w = max(len(cmd) for cmd, _ in items)
        lines = ["", self.section("next")]
        for cmd, why in items:
            row = "  " + self.style("accent", f"{cmd:<{w}}")
            if why:
                row += GLOSS_GAP + self.style("gloss", why)
            lines.append(row)
        return "\n".join(lines)

    def guidance(self, headline: str, why, tries) -> str:
        """The failure block: a bad headline, ``why`` lines, ``try`` options."""
        lines = [self.style("bad", headline), ""]
        why = list(why)
        if why:
            lines.append("   " + self.style("label", "why ") + "  " + why[0])
            for extra in why[1:]:
                lines.append("          " + extra)
            lines.append("")
        if tries:
            w = max(len(cmd) for cmd, _ in tries)
            first_cmd, first_why = tries[0]
            row = "   " + self.style("label", "try ") + "  " + self.style(
                "accent", f"{first_cmd:<{w}}")
            if first_why:
                row += GLOSS_GAP + self.style("gloss", first_why)
            lines.append(row)
            for cmd, gl in tries[1:]:
                row = "          " + self.style("accent", f"{cmd:<{w}}")
                if gl:
                    row += GLOSS_GAP + self.style("gloss", gl)
                lines.append(row)
        return "\n".join(lines)

    def status_line(self, parts) -> str:
        """Join ``(text, role)`` parts with ' · '."""
        return " · ".join(self.style(role, text) for text, role in parts)

    def raw(self, title: str, lines) -> str:
        """A verbose-only appendix. Returns "" unless ``self.verbose``."""
        if not self.verbose:
            return ""
        out = [self.section(title)]
        for line in lines:
            out.append(self.style("dim", line))
        return "\n".join(out)

    def emit(self, text: str = "") -> None:
        print(text, file=self.stream)
