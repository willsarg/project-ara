# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Run the push-only node on boot (systemd --user).

The ``install``/``start``/``stop``/``status``/``uninstall`` functions manage a **user** systemd unit
(``systemctl --user``, no root): the node comes back after a reboot or a crash without touching
system-wide config, matching ARA's install-into-your-own-space philosophy. ExecStart runs the
phone-home work loop (``ara node run``) — the node is a pure client with no inbound socket.

Linux first — systemd is the only init covered today; the other platforms raise a clear error via
:func:`_require_linux` rather than silently no-op'ing (Rule #3). The one external boundary is
:func:`_run` (every ``systemctl`` call), which tests stub.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import unicodedata
from enum import Enum
from pathlib import Path

UNIT_NAME = "ara-node.service"


def _run(cmd: list[str]) -> tuple[int, str, str]:
    """Run *cmd*, return (returncode, stdout, stderr). The one external boundary (mirrors
    :func:`ara.engine_env._run`); tests stub it so no real ``systemctl`` ever runs."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


class _SystemctlResult(Enum):
    SUCCESS = "success"
    ABSENT = "absent"
    FAILURE = "failure"


def _classify_systemctl(rc: int, out: str, err: str) -> _SystemctlResult:
    """Classify only explicit systemd absence as idempotent; everything else is a failure."""
    if rc == 0:
        return _SystemctlResult.SUCCESS
    diagnostics = tuple(
        part.strip().casefold() for part in (out, err) if part.strip()
    )
    message = "\n".join(diagnostics)
    fatal_markers = (
        "permission denied", "access denied", "failed to connect to bus",
        "authentication", "transport", "i/o error", "input/output error",
    )
    if any(marker in message for marker in fatal_markers):
        return _SystemctlResult.FAILURE
    known_absent = frozenset({
        f"unit {UNIT_NAME} does not exist.",
        f"failed to disable unit: unit file {UNIT_NAME} does not exist.",
        f"unit {UNIT_NAME} not loaded.",
        f"unit file {UNIT_NAME} not found.",
    })
    if rc == 1 and len(diagnostics) == 1 and diagnostics[0] in known_absent:
        return _SystemctlResult.ABSENT
    return _SystemctlResult.FAILURE


def _checked(cmd: list[str], *, allowed: tuple[int, ...] = (0,),
             allow_absent: bool = False) -> tuple[str, str]:
    """Run one systemctl command, raising with all daemon diagnostics on failure."""
    rc, out, err = _run(cmd)
    outcome = _classify_systemctl(rc, out, err)
    if rc not in allowed and not (allow_absent and outcome is _SystemctlResult.ABSENT):
        details = "; ".join(part.strip() for part in (out, err) if part.strip())
        suffix = details or "no diagnostic output"
        raise RuntimeError(f"`{' '.join(cmd)}` exited {rc}: {suffix}")
    return out, err


def _require_linux() -> None:
    """Guard the systemd path. Checks ``platform.system()`` (not ``hasattr``) so tests can mock the
    OS per the project's cross-OS rule — the branch stays exercisable on any host."""
    if platform.system() != "Linux":
        raise RuntimeError(
            f"`ara node install` uses systemd --user and is Linux-only (this is "
            f"{platform.system()}); run `ara node run` to launch it in the foreground instead")


def _unit_dir() -> Path:
    """Where the unit file is written — ``ARA_NODE_SYSTEMD_DIR`` if set (tests), else the standard
    per-user systemd directory."""
    override = os.environ.get("ARA_NODE_SYSTEMD_DIR")
    if override:
        return Path(override)
    return Path.home() / ".config" / "systemd" / "user"


def _unit_path() -> Path:
    """Full path of the node's systemd unit file."""
    return _unit_dir() / UNIT_NAME


# Watchdog ceiling: systemd kills+restarts the node if it misses a WATCHDOG=1 beat this long. The
# agent loop pets it every poll iteration (health.heartbeat), so 30s comfortably clears the cadence.
WATCHDOG_SEC = 30


def _systemd_quote(value: str) -> str:
    """Quote one ExecStart argv item using systemd's double-quoted string rules."""
    if any(unicodedata.category(char).startswith("C") for char in value):
        raise ValueError("systemd ExecStart values must not contain control characters")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return f'"{escaped}"'


def _unit_text() -> str:
    """The systemd unit for the push-only node. ExecStart runs the phone-home agent loop
    (``ara node run``) — the loop that pets systemd's watchdog via sd_notify, hence ``Type=notify``
    and ``WatchdogSec``. ExecStart uses the *current* interpreter (``sys.executable -m ara``) so
    it points at whichever uv-managed environment ARA is installed in rather than assuming a fixed
    ``~/.local/bin/ara`` that may not exist there."""
    return (
        "[Unit]\n"
        "Description=ARA node daemon\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=notify\n"
        f"ExecStart={_systemd_quote(sys.executable)} -m ara node run\n"
        f"WatchdogSec={WATCHDOG_SEC}\n"
        "Restart=on-failure\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def install() -> None:
    """Write the unit, reload systemd, and enable+start it so the node survives reboots."""
    _require_linux()
    path = _unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_unit_text())
    _checked(["systemctl", "--user", "daemon-reload"])
    _checked(["systemctl", "--user", "enable", "--now", UNIT_NAME])


def start() -> None:
    """Start the installed node unit now."""
    _require_linux()
    _checked(["systemctl", "--user", "start", UNIT_NAME])


def stop() -> None:
    """Stop the running node unit."""
    _require_linux()
    _checked(["systemctl", "--user", "stop", UNIT_NAME])


def status() -> str:
    """Return ``systemctl status`` output for the node unit (the human-readable state block)."""
    _require_linux()
    out, _err = _checked(
        ["systemctl", "--user", "status", UNIT_NAME], allowed=(0, 3))
    return out


def uninstall() -> None:
    """Disable+stop the unit, remove its file, and reload so systemd forgets it."""
    _require_linux()
    path = _unit_path()
    _checked(["systemctl", "--user", "disable", "--now", UNIT_NAME], allow_absent=True)
    path.unlink(missing_ok=True)
    _checked(["systemctl", "--user", "daemon-reload"])
