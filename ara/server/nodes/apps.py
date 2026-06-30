# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Django app config for the node registry."""
from __future__ import annotations

from django.apps import AppConfig


class NodesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ara.server.nodes"
    label = "nodes"
