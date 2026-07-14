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
from pathlib import Path

UNIT_NAME = "ara-node.service"


def _run(cmd: list[str]) -> tuple[int, str, str]:
    """Run *cmd*, return (returncode, stdout, stderr). The one external boundary (mirrors
    :func:`ara.engine_env._run`); tests stub it so no real ``systemctl`` ever runs."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _require_linux() -> None:
    """Guard the systemd path. Checks ``platform.system()`` (not ``hasattr``) so tests can mock the
    OS per the project's cross-OS rule — the branch stays exercisable on any host."""
    if platform.system() != "Linux":
        raise RuntimeError(
            f"`ara node install` uses systemd --user and is Linux-only (this is "
            f"{platform.system()}); run `ara node start` to launch it in the foreground instead")


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
        f"ExecStart={sys.executable} -m ara node run\n"
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
    _run(["systemctl", "--user", "daemon-reload"])
    _run(["systemctl", "--user", "enable", "--now", UNIT_NAME])


def start() -> None:
    """Start the installed node unit now."""
    _require_linux()
    _run(["systemctl", "--user", "start", UNIT_NAME])


def stop() -> None:
    """Stop the running node unit."""
    _require_linux()
    _run(["systemctl", "--user", "stop", UNIT_NAME])


def status() -> str:
    """Return ``systemctl status`` output for the node unit (the human-readable state block)."""
    _require_linux()
    _rc, out, _err = _run(["systemctl", "--user", "status", UNIT_NAME])
    return out


def uninstall() -> None:
    """Disable+stop the unit, remove its file, and reload so systemd forgets it."""
    _require_linux()
    _run(["systemctl", "--user", "disable", "--now", UNIT_NAME])
    _unit_path().unlink(missing_ok=True)         # idempotent: fine if it was never installed
    _run(["systemctl", "--user", "daemon-reload"])
