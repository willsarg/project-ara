# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""URL routing — ``/`` is the custom fleet dashboard; the Django admin stays at ``/admin/``."""
from __future__ import annotations

from django.contrib import admin
from django.urls import path

from ara.server.nodes.views import dashboard

urlpatterns = [
    path("", dashboard, name="dashboard"),
    path("admin/", admin.site.urls),
]
