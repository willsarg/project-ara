// SPDX-License-Identifier: Apache-2.0
// The HTTP client to ARA nodes. SERVER-ONLY: it is the single place node tokens are attached to a
// request. Nothing here is ever sent to the browser — callers return rendered rows, not raw tokens.
import "server-only";
import type { Node } from "./db";

const POLL_TIMEOUT_MS = 2500; // ~2.5s per node — an offline box can't stall the dashboard

export interface Detect {
  system?: string;
  chip?: string;
  cpu_logical?: number;
  ram_total_gb?: number;
  ram_available_gb?: number;
  accel?: { vendor?: string; name?: string; kind?: string } | null;
}

export interface Status {
  workloads?: unknown[];
  apps?: unknown[];
}

/** GET <node.base_url><path> with the node's bearer token and a hard timeout. Throws on failure. */
export async function nodeGet<T>(node: Node, path: string, timeoutMs = POLL_TIMEOUT_MS): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${node.base_url}${path}`, {
      headers: { Authorization: `Bearer ${node.token}` },
      signal: ctrl.signal,
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`${path} → ${res.status}`);
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}
