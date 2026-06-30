# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The Django admin registration for :class:`Node` — the v1 dashboard.

``list_display`` is the fleet view; the two admin actions drive nodes through :mod:`ara.server.client`:
"refresh status" pulls each node's ``/status`` and caches it on the row, and "submit detect job"
posts a job to each selected node. Errors surface as admin messages (Rule #3: never swallow them).
"""
from __future__ import annotations

from django.contrib import admin, messages
from django.utils import timezone

from ara.server import client
from ara.server.nodes.models import Node


@admin.register(Node)
class NodeAdmin(admin.ModelAdmin):
    list_display = ("name", "base_url", "enabled", "last_seen")
    list_filter = ("enabled",)
    search_fields = ("name", "base_url")
    actions = ["refresh_status", "submit_detect_job"]

    @admin.action(description="Refresh status from the node")
    def refresh_status(self, request, queryset):
        for node in queryset:
            try:
                node.last_status = client.status(node)
                node.last_seen = timezone.now()
                node.save(update_fields=["last_status", "last_seen"])
                self.message_user(request, f"{node.name}: status refreshed")
            except Exception as exc:                  # noqa: BLE001 — surface any client error
                self.message_user(request, f"{node.name}: {exc}", level=messages.ERROR)

    @admin.action(description="Submit a detect job to the node")
    def submit_detect_job(self, request, queryset):
        for node in queryset:
            try:
                resp = client.submit_job(node, "detect", {})
                self.message_user(request, f"{node.name}: job {resp.get('job_id')}")
            except Exception as exc:                  # noqa: BLE001 — surface any client error
                self.message_user(request, f"{node.name}: {exc}", level=messages.ERROR)
