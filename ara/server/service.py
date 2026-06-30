# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Run the server on boot (systemd --user) + the uvicorn launcher + Django migrations.

Mirrors :mod:`ara.node.service` exactly: :func:`serve` is the foreground launcher (uvicorn over the
Django ASGI app), :func:`migrate` runs Django's migrations, and ``install``/``start``/``stop``/
``status``/``uninstall`` manage a **user** systemd unit whose ExecStart runs ``ara server serve``.
The systemd path is Linux-only (:func:`_require_linux`); the one external boundary is :func:`_run`
(every ``systemctl`` call), which tests stub. ``uvicorn``/``django`` import lazily so this module
imports without the optional ``[server]`` extra present.
"""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

UNIT_NAME = "ara-server.service"
SETTINGS_MODULE = "ara.server.settings"


def _run(cmd: list[str]) -> tuple[int, str, str]:
    """Run *cmd*, return (returncode, stdout, stderr). The one external boundary (mirrors
    :func:`ara.node.service._run`); tests stub it so no real ``systemctl`` ever runs."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _require_linux() -> None:
    """Guard the systemd path. Checks ``platform.system()`` (not ``hasattr``) so tests can mock the
    OS per the project's cross-OS rule — the branch stays exercisable on any host."""
    if platform.system() != "Linux":
        raise RuntimeError(
            f"`ara server install` uses systemd --user and is Linux-only (this is "
            f"{platform.system()}); run `ara server serve` to launch it in the foreground instead")


def _unit_dir() -> Path:
    """Where the unit file is written — ``ARA_SERVER_SYSTEMD_DIR`` if set (tests), else the standard
    per-user systemd directory."""
    override = os.environ.get("ARA_SERVER_SYSTEMD_DIR")
    if override:
        return Path(override)
    return Path.home() / ".config" / "systemd" / "user"


def _unit_path() -> Path:
    """Full path of the server's systemd unit file."""
    return _unit_dir() / UNIT_NAME


def _unit_text(host: str, port: int) -> str:
    """The systemd unit. ExecStart uses the ``ara`` console-script at the ``%h``-relative pip
    ``--user`` location so the unit needs no absolute, machine-specific path."""
    return (
        "[Unit]\n"
        "Description=ARA server coordinator\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart=%h/.local/bin/ara server serve --host {host} --port {port}\n"
        "Restart=on-failure\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def install(host: str, port: int) -> None:
    """Write the unit, reload systemd, and enable+start it so the server survives reboots."""
    _require_linux()
    path = _unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_unit_text(host, port))
    _run(["systemctl", "--user", "daemon-reload"])
    _run(["systemctl", "--user", "enable", "--now", UNIT_NAME])


def start() -> None:
    """Start the installed server unit now."""
    _require_linux()
    _run(["systemctl", "--user", "start", UNIT_NAME])


def stop() -> None:
    """Stop the running server unit."""
    _require_linux()
    _run(["systemctl", "--user", "stop", UNIT_NAME])


def status() -> str:
    """Return ``systemctl status`` output for the server unit (the human-readable state block)."""
    _require_linux()
    _rc, out, _err = _run(["systemctl", "--user", "status", UNIT_NAME])
    return out


def uninstall() -> None:
    """Disable+stop the unit, remove its file, and reload so systemd forgets it."""
    _require_linux()
    _run(["systemctl", "--user", "disable", "--now", UNIT_NAME])
    _unit_path().unlink(missing_ok=True)         # idempotent: fine if it was never installed
    _run(["systemctl", "--user", "daemon-reload"])


def migrate() -> None:
    """Run Django migrations (creates/updates the SQLite registry). ``django`` imports lazily so this
    module loads without the optional ``[server]`` extra; tests monkeypatch this whole function."""
    import django
    from django.core.management import call_command

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", SETTINGS_MODULE)
    django.setup()
    call_command("migrate", interactive=False)


def serve(host: str, port: int) -> None:
    """Foreground launcher: run the Django ASGI app under uvicorn.

    ``uvicorn`` imports lazily so this module imports without the optional ``[server]`` extra; tests
    monkeypatch ``uvicorn.run`` to avoid binding a port."""
    import uvicorn

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", SETTINGS_MODULE)
    uvicorn.run("ara.server.asgi:application", host=host, port=port)


def _django_setup() -> None:
    """Initialize Django against the server settings (idempotent) — the ORM-touching ops share it."""
    import django

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", SETTINGS_MODULE)
    django.setup()


def create_admin(username: str) -> dict:
    """Create — or reset the password of — a Django superuser for the dashboard, returning
    ``{username, password, created}``. The password is GENERATED and returned once (the CLI has no
    interactive prompt). ``django`` imports lazily; tests monkeypatch this whole function."""
    import secrets

    _django_setup()
    from django.contrib.auth import get_user_model

    password = secrets.token_urlsafe(12)
    user_model = get_user_model()
    user, created = user_model.objects.get_or_create(username=username)
    user.is_staff = True
    user.is_superuser = True
    user.set_password(password)
    user.save()
    return {"username": username, "password": password, "created": created}


def add_node(name: str, base_url: str, token: str) -> dict:
    """Register (or update) a node in the server's registry, returning ``{name, base_url}``.
    Idempotent on the node name. ``django`` imports lazily; tests monkeypatch this whole function."""
    _django_setup()
    from ara.server.nodes.models import Node

    node, _created = Node.objects.update_or_create(
        name=name, defaults={"base_url": base_url.rstrip("/"), "token": token, "enabled": True})
    return {"name": node.name, "base_url": node.base_url}
