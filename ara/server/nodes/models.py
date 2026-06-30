# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The :class:`Node` model — one row per node the coordinator drives over HTTP."""
from __future__ import annotations

from django.db import models


class Node(models.Model):
    """A registered ARA node: where it lives (``base_url``), how to authenticate (``token``), and a
    cache of the last status the coordinator pulled (``last_status``/``last_seen``)."""

    name = models.CharField(max_length=200, unique=True)
    base_url = models.URLField(help_text="e.g. http://192.168.1.50:8473")
    token = models.CharField(max_length=512, blank=True,
                             help_text="the node's bearer token (`ara node token`)")
    enabled = models.BooleanField(default=True)
    last_status = models.JSONField(null=True, blank=True)
    last_seen = models.DateTimeField(null=True, blank=True)

    def __str__(self) -> str:
        return self.name
