// SPDX-License-Identifier: Apache-2.0
// Turn a registered node into a render-ready row: identity + (if reachable) silicon and live
// activity. SERVER-ONLY. Mirrors the Django dashboard's _probe so the fleet view matches.
import "server-only";
import type { Node } from "./db";
import { nodeGet, type Detect, type Status } from "./node-client";

export interface NodeRow {
  id: number;
  name: string;
  base_url: string;
  enabled: boolean;
  online: boolean;
  system?: string;
  chip?: string;
  cores?: number;
  ram_total?: number;
  ram_used?: number;
  ram_pct: number;
  accel?: string; // null/undefined => "cpu only"
  running: number;
}

export async function probe(node: Node): Promise<NodeRow> {
  const base: NodeRow = {
    id: node.id,
    name: node.name,
    base_url: node.base_url,
    enabled: !!node.enabled,
    online: false,
    ram_pct: 0,
    running: 0,
  };
  if (!node.enabled) return base;

  let d: Detect, s: Status;
  try {
    [d, s] = await Promise.all([
      nodeGet<Detect>(node, "/detect"),
      nodeGet<Status>(node, "/status"),
    ]);
  } catch {
    return base; // unreachable / timed out / refused → render as offline
  }

  const accel = d.accel || {};
  const total = d.ram_total_gb || 0;
  const avail = d.ram_available_gb;
  const used = avail != null && total ? total - avail : null;
  const hasAccel = accel.vendor != null && accel.vendor !== "none";

  return {
    ...base,
    online: true,
    system: d.system,
    chip: d.chip || d.system || "—",
    cores: d.cpu_logical,
    ram_total: total ? Math.round(total) : undefined,
    ram_used: used != null ? Math.round(used) : undefined,
    ram_pct: used != null && total ? Math.round((100 * used) / total) : 0,
    accel: hasAccel ? accel.name : undefined,
    running: Array.isArray(s.workloads) ? s.workloads.length : 0,
  };
}

export async function probeAll(nodes: Node[]): Promise<NodeRow[]> {
  return Promise.all(nodes.map(probe));
}
