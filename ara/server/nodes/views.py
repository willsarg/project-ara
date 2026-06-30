# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""The fleet dashboard — a live view of every registered node.

Each registered node is polled on render (short per-node timeout, so an offline box can't stall the
page): /detect for the silicon, /status for what's running. The server holds the tokens; the browser
only ever sees rendered HTML.
"""
from __future__ import annotations

from django.shortcuts import render

from ara.server import client
from ara.server.nodes.models import Node

_POLL_TIMEOUT = 2.5          # seconds per node — keep the dashboard snappy when a box is down


def _probe(node) -> dict:
    """A render-ready row for one node: identity + (if reachable) silicon and live activity."""
    row = {"name": node.name, "base_url": node.base_url, "enabled": node.enabled,
           "online": False, "ram_pct": 0}
    if not node.enabled:
        return row
    try:
        d = client.get(node, "/detect", timeout=_POLL_TIMEOUT)
        s = client.get(node, "/status", timeout=_POLL_TIMEOUT)
    except Exception:
        return row               # unreachable / timed out / refused → render as offline
    accel = d.get("accel") or {}
    total = d.get("ram_total_gb") or 0
    avail = d.get("ram_available_gb")
    used = (total - avail) if (avail is not None and total) else None
    row.update(
        online=True,
        system=d.get("system"),
        chip=d.get("chip") or d.get("system") or "—",
        cores=d.get("cpu_logical"),
        ram_total=round(total) if total else None,
        ram_used=round(used) if used is not None else None,
        ram_pct=int(round(100 * used / total)) if (used is not None and total) else 0,
        accel=accel.get("name") if accel.get("vendor") not in (None, "none") else None,
        accel_vendor=accel.get("vendor"),
        running=len((s or {}).get("workloads") or []) if isinstance(s, dict) else 0,
    )
    return row


def dashboard(request):
    nodes = [_probe(n) for n in Node.objects.all().order_by("name")]
    online = sum(1 for n in nodes if n["online"])
    running = sum(1 for n in nodes if n.get("running"))
    return render(request, "nodes/dashboard.html", {
        "nodes": nodes, "total": len(nodes), "online": online, "running": running,
    })
