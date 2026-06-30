# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""ASGI entry point — what ``ara server serve`` hands to uvicorn."""
from __future__ import annotations

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ara.server.settings")

from django.core.asgi import get_asgi_application  # noqa: E402 — must follow the env default

application = get_asgi_application()
