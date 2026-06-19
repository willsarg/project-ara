"""Inventory of AI/ML applications installed on the machine — GUI apps in /Applications
plus Homebrew packages — matched against a curated catalog of known AI/ML software.

A different lens from ENGINES (what ARA can launch) and FRAMEWORKS (python libraries):
this is "what AI software is installed here," organized by what it's for. Read-only.
macOS-focused (scans /Applications + Homebrew); degrades to whatever it can find elsewhere.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

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
    ("Codex", "coding", ["Codex"], ["codex"]),
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

    @property
    def homebrew(self) -> bool:
        return self.cask or self.formula

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


def _app_stems() -> set[str]:
    """Lowercased names (without .app) of bundles in the Applications folders."""
    found: set[str] = set()
    for d in (Path("/Applications"), Path.home() / "Applications"):
        try:
            for p in d.iterdir():
                if p.name.endswith(".app"):
                    found.add(p.name[:-4].lower())
        except Exception:
            pass
    return found


def _brew_lists() -> tuple[set[str], set[str]]:
    """(formulae, casks) installed via Homebrew — lowercased, @version stripped. Kept
    separate because casks install GUI apps into /Applications, formulae don't."""
    if not shutil.which("brew"):
        return set(), set()

    def listing(kind: str) -> set[str]:
        try:
            out = subprocess.run(["brew", "list", kind, "-1"],
                                 capture_output=True, text=True, timeout=15).stdout
        except Exception:
            out = ""
        return {ln.strip().lower().split("@", 1)[0] for ln in (out or "").splitlines() if ln.strip()}

    return listing("--formula"), listing("--cask")


def scan() -> list[App]:
    """Installed AI/ML apps from the curated catalog, ordered by category then name."""
    apps = _app_stems()
    formulae, casks = _brew_lists()
    out: list[App] = []
    for label, category, bundles, tokens in CATALOG:
        in_app = any(b.lower() in apps for b in bundles)
        cask = any(t in casks for t in tokens)
        formula = any(t in formulae for t in tokens)
        if in_app or cask or formula:
            out.append(App(label, category, in_app=in_app, cask=cask, formula=formula))
    out.sort(key=lambda a: (_ORDER.index(a.category), a.label.lower()))
    return out
