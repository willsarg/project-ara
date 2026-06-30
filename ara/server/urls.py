# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""URL routing — the admin IS the v1 dashboard, so ``/`` just redirects into it."""
from __future__ import annotations

from django.contrib import admin
from django.urls import path
from django.views.generic import RedirectView

urlpatterns = [
    path("", RedirectView.as_view(url="admin/", permanent=False)),
    path("admin/", admin.site.urls),
]
