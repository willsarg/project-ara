# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""WSGI entry point (for a generic WSGI server); ``ara server serve`` uses the ASGI app instead."""
from __future__ import annotations

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ara.server.settings")

from django.core.wsgi import get_wsgi_application  # noqa: E402 — must follow the env default

application = get_wsgi_application()
