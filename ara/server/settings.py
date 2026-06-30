# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Django settings for the ARA server coordinator.

Single-host v1: SQLite in the ARA data dir, a SECRET_KEY generated to a file there on first run,
and the Django admin as the dashboard. Everything is env-overridable so the same settings module
works on a dev box and a LAN host. DEBUG is off by default; the admin login gates the dashboard.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

import platformdirs


def _data_dir() -> Path:
    """The server's state directory (db, secret key) — under the OS data dir, created on demand."""
    d = Path(platformdirs.user_data_dir("ara")) / "server"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _db_path() -> Path:
    """SQLite location — ``ARA_SERVER_DB`` if set (tests/ops), else the ARA data dir."""
    override = os.environ.get("ARA_SERVER_DB")
    return Path(override) if override else _data_dir() / "server.db"


def _secret_key() -> str:
    """The Django SECRET_KEY — ``ARA_SERVER_SECRET_KEY`` if set, else read/generate a file (0600)
    in the data dir so it's stable across restarts without being committed."""
    env = os.environ.get("ARA_SERVER_SECRET_KEY")
    if env:
        return env
    path = _data_dir() / "secret_key"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    key = secrets.token_urlsafe(50)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Atomic owner-only create (like the node token): the SECRET_KEY is a credential, so it must
    # never be world/group-readable even for the instant between write and chmod.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(key)
    return key


SECRET_KEY = _secret_key()
DEBUG = os.environ.get("ARA_SERVER_DEBUG", "") == "1"
# Default "*" because the coordinator is a LAN service behind the admin login (token/credential
# gated), not an internet-exposed app; narrow it with ARA_SERVER_ALLOWED_HOSTS in real deployments.
ALLOWED_HOSTS = os.environ.get("ARA_SERVER_ALLOWED_HOSTS", "*").split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "ara.server.nodes",
]

MIDDLEWARE = [
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

ROOT_URLCONF = "ara.server.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "ara.server.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": str(_db_path()),
    }
}

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
USE_TZ = True
