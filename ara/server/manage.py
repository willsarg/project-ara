#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Django's manage.py — `python -m ara.server.manage <cmd>` for admin/migration tasks.

``ara server migrate`` shells through :mod:`ara.server.service`; this exists for the full Django
management surface (createsuperuser, makemigrations, …) when operating the coordinator directly.
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ara.server.settings")
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
