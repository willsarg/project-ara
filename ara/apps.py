# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Inventory of AI/ML applications installed on the machine — GUI apps in /Applications
plus Homebrew packages — matched against a curated catalog of known AI/ML software.

A different lens from ENGINES (what ARA can launch) and FRAMEWORKS (python libraries):
this is "what AI software is installed here," organized by what it's for. Read-only.
macOS-focused (scans /Applications + Homebrew); degrades to whatever it can find elsewhere.
"""
from __future__ import annotations

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
    """Installed AI/ML apps from the curated catalog, ordered by category then name."""
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
