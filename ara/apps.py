# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Inventory of AI/ML applications installed on the machine — GUI apps in /Applications
plus Homebrew packages on macOS, .desktop entries/Flatpak/Snap on Linux, registry Uninstall
keys on Windows — matched against a curated catalog of known AI/ML software.

A different lens from ENGINES (what ARA can launch) and FRAMEWORKS (python libraries):
this is "what AI software is installed here," organized by what it's for. Read-only.
scan() dispatches on sys.platform; degrades to [] on platforms without a scanner yet.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ara import versions

# Category keys in display order, with their section sub-headers.
CATEGORY_LABEL = {
    "runner": "model runners",
    "image": "image generation",
    "speech": "speech / audio",
    "toolkit": "ML toolkits",
    "assistant": "AI assistants",
    "coding": "AI coding",
}
_ORDER = list(CATEGORY_LABEL)
_NO_MATCH = object()  # sentinel: distinguishes "no match" from "matched, version is None"

# (label, category, [.app bundle names], [brew formula/cask tokens]). Curated — matched
# exactly (case-insensitive), no keyword guessing, so a hit is always a real known app.
CATALOG: list[tuple[str, str, list[str], list[str]]] = [
    # local model runners / chat frontends
    ("LM Studio", "runner", ["LM Studio"], ["lm-studio"]),
    ("Ollama", "runner", ["Ollama"], ["ollama"]),
    ("GPT4All", "runner", ["GPT4All", "gpt4all"], ["gpt4all"]),
    ("Jan", "runner", ["Jan"], ["jan"]),
    ("Msty", "runner", ["Msty"], ["msty"]),
    ("Enchanted", "runner", ["Enchanted"], []),
    ("Ollamac", "runner", ["Ollamac"], []),
    ("Pinokio", "runner", ["Pinokio"], ["pinokio"]),
    ("Transformer Lab", "runner", ["Transformer Lab"], []),
    # image generation
    ("DiffusionBee", "image", ["DiffusionBee"], ["diffusionbee"]),
    ("Draw Things", "image", ["Draw Things"], []),
    ("ComfyUI", "image", ["ComfyUI"], []),
    ("InvokeAI", "image", ["InvokeAI"], []),
    ("Diffusers", "image", ["Diffusers"], []),
    ("Fooocus", "image", ["Fooocus"], []),
    # speech / audio
    ("MacWhisper", "speech", ["MacWhisper"], ["macwhisper"]),
    ("superwhisper", "speech", ["superwhisper"], ["superwhisper"]),
    ("VoiceInk", "speech", ["VoiceInk"], ["voiceink"]),
    ("Aiko", "speech", ["Aiko"], []),
    ("Whisper Transcription", "speech", ["Whisper Transcription"], []),
    # ML toolkits / CLIs (largely Homebrew)
    ("llama.cpp", "toolkit", [], ["llama.cpp"]),
    ("whisper.cpp", "toolkit", [], ["whisper-cpp"]),
    ("MLX", "toolkit", [], ["mlx", "mlx-c"]),
    ("ggml", "toolkit", [], ["ggml"]),
    ("ONNX Runtime", "toolkit", [], ["onnxruntime"]),
    ("PyTorch", "toolkit", [], ["pytorch"]),
    ("TensorFlow", "toolkit", [], ["tensorflow"]),
    ("Hugging Face CLI", "toolkit", [], ["huggingface-cli"]),
    # AI assistants (cloud clients) and AI coding tools
    ("ChatGPT", "assistant", ["ChatGPT"], ["chatgpt"]),
    ("Claude", "assistant", ["Claude"], ["claude"]),
    ("Perplexity", "assistant", ["Perplexity"], ["perplexity"]),
    ("Cursor", "coding", ["Cursor"], ["cursor"]),
    ("Windsurf", "coding", ["Windsurf"], ["windsurf"]),
    ("Antigravity", "coding", ["Antigravity"], []),
    # Codex ships as two distinct artifacts — keep them separate so their independent
    # versions aren't compared as "drift" (the .app is com.openai.codex; the cask is the CLI).
    ("Codex", "coding", ["Codex"], []),
    ("Codex CLI", "coding", [], ["codex"]),
    ("CodexBar", "coding", ["CodexBar"], ["codexbar"]),
    ("Claude Code", "coding", [], ["claude-code"]),
    ("GitHub Copilot", "coding", ["GitHub Copilot"], ["copilot"]),
    ("Warp", "coding", ["Warp"], ["warp"]),
]


@dataclass(frozen=True)
class App:
    label: str
    category: str
    in_app: bool        # a .app bundle is present in an Applications folder
    cask: bool          # installed as a Homebrew cask (which IS how its .app got there)
    formula: bool       # installed as a Homebrew formula (CLI)
    version: str | None = None          # what's actually installed (.app plist for GUIs)
    brew_recorded: str | None = None     # Homebrew's receipt version, when it manages this
    cask_token: str | None = None        # the matched brew cask token (for drift remediation)
    installed_at: float | None = None    # epoch mtime/birthtime, for "recently installed"
    source_label: str | None = None      # non-Homebrew provenance (e.g. a Linux packaging system)

    @property
    def homebrew(self) -> bool:
        return self.cask or self.formula

    @property
    def drift(self) -> bool:
        """A cask GUI app whose installed (.app) version has self-updated past Homebrew's
        frozen receipt — so `brew` no longer reflects reality (and `brew upgrade` may clobber).
        Requires an actual installed .app, so a CLI-only cask never counts as drift."""
        return bool(self.cask and self.in_app and self.brew_recorded and self.version
                    and self.version != self.brew_recorded)

    @property
    def duplicate(self) -> bool:
        """Two independent installs of the same tool: a CLI formula alongside a GUI
        install (cask or a hand-dropped .app), or both a cask and a formula. A cask plus
        its own .app is NOT a duplicate — the cask is what put the .app there."""
        return self.formula and (self.cask or self.in_app)

    @property
    def source(self) -> str:
        if self.source_label is not None:
            return self.source_label
        if self.cask and self.formula:
            return "Homebrew (cask + formula)"
        if self.cask:
            return "Homebrew (cask)"
        if self.formula and self.in_app:
            return "Homebrew (formula) + separate app"
        if self.formula:
            return "Homebrew (formula)"
        return "app (not via Homebrew)"


_APP_DIRS = (Path("/Applications"), Path.home() / "Applications")
_BREW_PREFIX = Path("/opt/homebrew") if Path("/opt/homebrew").exists() else Path("/usr/local")


def _install_time(bundles: list[str], tokens: list[str], in_app: bool) -> float | None:
    """Best-effort install/update time: a .app's filesystem time, else a Homebrew dir's.
    Uses max(mtime, birthtime) — some bundles report a bogus birthtime."""
    if in_app:
        for b in bundles:
            for base in _APP_DIRS:
                app = base / f"{b}.app"
                if app.is_dir():
                    st = app.stat()
                    return max(st.st_mtime, getattr(st, "st_birthtime", 0) or 0)
    for t in tokens:
        for sub in ("Caskroom", "Cellar"):
            d = _BREW_PREFIX / sub / t
            if d.is_dir():
                try:
                    return max((p.stat().st_mtime for p in d.iterdir()), default=d.stat().st_mtime)
                except Exception:
                    return None
    return None


def scan() -> list[App]:
    """Installed AI/ML apps from the curated catalog, ordered by category then name.
    Dispatches on sys.platform so it stays mockable in tests."""
    if sys.platform == "darwin":
        return _scan_macos()
    if sys.platform.startswith("linux"):
        return _scan_linux()
    if sys.platform == "win32":
        return _scan_windows()
    return []


def _scan_macos() -> list[App]:
    formulae, casks = versions.brew_formulae(), versions.brew_casks()
    out: list[App] = []
    for label, category, bundles, tokens in CATALOG:
        in_app, app_ver = versions.find_app(bundles)
        cask_ver = next((casks[t] for t in tokens if casks.get(t)), None)
        formula_ver = next((formulae[t] for t in tokens if formulae.get(t)), None)
        cask = any(t in casks for t in tokens)
        formula = any(t in formulae for t in tokens)
        if not (in_app or cask or formula):
            continue
        # The installed truth is the .app's own version; show that. For a cask we also keep
        # brew's receipt so we can flag self-update drift. A formula (CLI) has no .app, so
        # its brew version IS the truth.
        cask_token = next((t for t in tokens if t in casks), None)
        if cask:
            version, brew_recorded = (app_ver or cask_ver), cask_ver
        elif formula:
            version, brew_recorded = formula_ver, None
        else:  # hand-installed .app
            version, brew_recorded = app_ver, None
        out.append(App(label, category, in_app=in_app, cask=cask, formula=formula,
                       version=version, brew_recorded=brew_recorded, cask_token=cask_token,
                       installed_at=_install_time(bundles, tokens, in_app)))
    out.sort(key=lambda a: (_ORDER.index(a.category), a.label.lower()))
    return out


_LINUX_DESKTOP_DIRS = (
    Path("/usr/share/applications"),
    Path.home() / ".local" / "share" / "applications",
    Path("/var/lib/flatpak/exports/share/applications"),
)


def _linux_desktop_names() -> list[str]:
    """Display names (the `Name=` field) of every *.desktop entry found in the standard
    Linux application directories. Missing/unreadable dirs and files degrade to nothing —
    never raise."""
    names: list[str] = []
    for base in _LINUX_DESKTOP_DIRS:
        try:
            entries = list(base.glob("*.desktop"))
        except OSError:
            continue
        for entry in entries:
            try:
                text = entry.read_text(errors="ignore")
            except OSError:
                continue
            for line in text.splitlines():
                if line.startswith("Name="):
                    names.append(line[len("Name="):].strip())
                    break
    return names


def _flatpak_apps() -> dict[str, str | None]:
    """{app name: version} for installed Flatpaks, keyed by their display name. Empty if
    flatpak isn't installed or the call fails."""
    if not shutil.which("flatpak"):
        return {}
    try:
        out = subprocess.run(
            ["flatpak", "list", "--columns=application,name,version"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    result: dict[str, str | None] = {}
    for line in (out or "").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            name = parts[1].strip()
            version = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else None
            if name:
                result[name] = version
    return result


def _snap_apps() -> dict[str, str | None]:
    """{snap name: version} for installed Snaps. Empty if snap isn't installed or the call
    fails."""
    if not shutil.which("snap"):
        return {}
    try:
        out = subprocess.run(
            ["snap", "list"], capture_output=True, text=True, timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    result: dict[str, str | None] = {}
    for line in (out or "").splitlines()[1:]:  # skip the header row
        parts = line.split()
        if len(parts) >= 2:
            result[parts[0]] = parts[1]
    return result


def _scan_linux() -> list[App]:
    """Installed AI/ML apps on Linux, matched against the curated catalog by label and
    bundle-name (which double as display names) against .desktop entries, Flatpaks, and
    Snaps. Ordered identically to _scan_macos."""
    desktop_names = {n.lower() for n in _linux_desktop_names()}
    flatpaks_lower = {n.lower(): v for n, v in _flatpak_apps().items()}
    snaps_lower = {n.lower(): v for n, v in _snap_apps().items()}

    out: list[App] = []
    for label, category, bundles, _tokens in CATALOG:
        candidates_lower = [c.lower() for c in (label, *bundles)]

        match_desktop = any(c in desktop_names for c in candidates_lower)
        flatpak_match = next((c for c in candidates_lower if c in flatpaks_lower), None)
        snap_match = next((c for c in candidates_lower if c in snaps_lower), None)

        if not (match_desktop or flatpak_match or snap_match):
            continue

        if flatpak_match is not None:
            source_label, version = "Flatpak", flatpaks_lower[flatpak_match]
        elif snap_match is not None:
            source_label, version = "Snap", snaps_lower[snap_match]
        else:
            source_label, version = "app (.desktop)", None

        out.append(App(label, category, in_app=False, cask=False, formula=False,
                       version=version, source_label=source_label, cask_token=None,
                       installed_at=None))
    out.sort(key=lambda a: (_ORDER.index(a.category), a.label.lower()))
    return out


# (winreg hive attribute name, subkey path) — the three standard Uninstall locations that
# together cover 64-bit, 32-bit-on-64-bit (WOW6432Node), and per-user installs.
_WINDOWS_UNINSTALL_KEYS = (
    ("HKEY_LOCAL_MACHINE", r"Software\Microsoft\Windows\CurrentVersion\Uninstall"),
    ("HKEY_LOCAL_MACHINE", r"Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
    ("HKEY_CURRENT_USER", r"Software\Microsoft\Windows\CurrentVersion\Uninstall"),
)


def _windows_installed_programs() -> dict[str, str | None]:
    """{DisplayName: DisplayVersion} scraped from the registry's Uninstall keys (HKLM,
    HKLM\\WOW6432Node, and HKCU) — the canonical Windows "installed programs" list. Every
    key/subkey/value read is individually guarded so a missing key or value degrades cleanly
    rather than raising. Imports winreg lazily so this module stays importable — and this
    function stays callable, returning {} — on non-Windows platforms."""
    try:
        import winreg  # noqa: PLC0415 — intentional lazy import, Windows-only module
    except ImportError:
        return {}

    programs: dict[str, str | None] = {}
    for hive_name, subkey_path in _WINDOWS_UNINSTALL_KEYS:
        hive = getattr(winreg, hive_name)
        try:
            root = winreg.OpenKey(hive, subkey_path)
        except OSError:
            continue
        with root:
            index = 0
            while True:
                try:
                    entry_name = winreg.EnumKey(root, index)
                except OSError:
                    break
                index += 1
                try:
                    with winreg.OpenKey(root, entry_name) as entry:
                        try:
                            display_name, _ = winreg.QueryValueEx(entry, "DisplayName")
                        except OSError:
                            continue
                        try:
                            display_version, _ = winreg.QueryValueEx(entry, "DisplayVersion")
                        except OSError:
                            display_version = None
                        programs[display_name] = display_version
                except OSError:
                    continue
    return programs


def _scan_windows() -> list[App]:
    """Installed AI/ML apps on Windows, matched against the curated catalog by SUBSTRING
    (case-insensitive) against registry DisplayNames — Windows DisplayNames routinely carry
    version/edition suffixes ("Ollama version 0.23.0", "Cursor 0.45.11", "Cursor (User)"),
    so an exact match (as macOS/Linux use) would miss real installs. A catalog entry that
    substring-matches multiple DisplayNames (e.g. two Cursor installs) is still emitted once
    — the first match wins. Ordered identically to _scan_macos/_scan_linux."""
    programs_lower = {name.lower(): version for name, version in _windows_installed_programs().items()}

    out: list[App] = []
    for label, category, bundles, _tokens in CATALOG:
        candidates_lower = [c.lower() for c in (label, *bundles)]
        match = next(
            (v for n, v in programs_lower.items() if any(c in n for c in candidates_lower)),
            _NO_MATCH,
        )
        if match is _NO_MATCH:
            continue
        out.append(App(label, category, in_app=False, cask=False, formula=False,
                       version=match, source_label="registry", cask_token=None,
                       installed_at=None))
    out.sort(key=lambda a: (_ORDER.index(a.category), a.label.lower()))
    return out
