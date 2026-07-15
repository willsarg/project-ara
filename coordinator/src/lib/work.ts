// SPDX-License-Identifier: Apache-2.0
// The server-owned work dispatch queue. The node long-polls GET /api/work; the admin enqueues jobs.
// better-sqlite3 is SYNCHRONOUS, so `nextForAgent` must never busy-loop the event loop — it does a
// cheap sync read, and if there's nothing yet it AWAITS a short timer before the next read. SERVER-ONLY.
import "server-only";
import { randomBytes } from "node:crypto";
import {
  acknowledgeWorkForAgent, claimNextWorkForAgent, insertWork, listRecentWork as listRecentWorkRows,
  recordWorkResult, type WorkAckResult, type WorkResultWrite,
} from "./db";
import { assertAllowedJobKind } from "./job-kinds";

/** The dispatched job as it goes on the wire (work.response `job`). */
export interface DispatchedJob {
  id: string;
  kind: string;
  args: Record<string, unknown>;
}

export class AgentNotActiveError extends Error {
  constructor() {
    super("work can only be queued for an active agent");
  }
}

export interface AdminWorkSummary {
  id: string;
  machine_key: string;
  kind: string;
  model: string | null;
  status: string;
  result_json: string | null;
  error: string | null;
  result_environment_json: string | null;
  created_at: string;
  dispatched_at: string | null;
  finished_at: string | null;
}

function modelFromArgs(argsJson: string | null): string | null {
  if (!argsJson) return null;
  try {
    const args: unknown = JSON.parse(argsJson);
    if (typeof args === "object" && args !== null && "model" in args) {
      return typeof args.model === "string" ? args.model : null;
    }
  } catch {
    // Preserve the job in the view; only the optional model label is unavailable.
  }
  return null;
}

export function listRecentWork(): AdminWorkSummary[] {
  return listRecentWorkRows().map((row) => ({
    id: row.id,
    machine_key: row.machine_key,
    kind: row.kind,
    model: modelFromArgs(row.args_json),
    status: row.status,
    result_json: row.result_json,
    error: row.error,
    result_environment_json: row.result_environment_json,
    created_at: row.created_at,
    dispatched_at: row.dispatched_at,
    finished_at: row.finished_at,
  }));
}

const POLL_STEP_MS = 250; // re-check cadence between sync reads during a long-poll

/** Queue a job for an agent. Returns the job id (also the wire `job.id`). */
export function enqueue(agentId: number, kind: string, args: Record<string, unknown>): string {
  assertAllowedJobKind(kind);
  const id = `job_${randomBytes(12).toString("hex")}`;
  if (!insertWork(id, agentId, kind, JSON.stringify(args ?? {}))) {
    throw new AgentNotActiveError();
  }
  return id;
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/** Long-poll for the next queued/expired-offer job, up to `waitMs`. The returned offer remains
 * unexecutable until the node journals it and calls acknowledge(). */
export async function nextForAgent(agentId: number, waitMs: number): Promise<DispatchedJob | null> {
  const deadline = Date.now() + Math.max(0, waitMs);
  for (;;) {
    const row = claimNextWorkForAgent(agentId);
    if (row) {
      return {
        id: row.id,
        kind: row.kind,
        args: row.args_json ? (JSON.parse(row.args_json) as Record<string, unknown>) : {},
      };
    }
    const remaining = deadline - Date.now();
    if (remaining <= 0) return null;
    await sleep(Math.min(POLL_STEP_MS, remaining));
  }
}

export function acknowledge(jobId: string, agentId: number): WorkAckResult {
  return acknowledgeWorkForAgent(jobId, agentId);
}

/** Atomically record the first terminal result for an owned, dispatched job. */
export function recordResult(
  jobId: string,
  agentId: number,
  payload: {
    status: string;
    result?: unknown;
    error?: unknown;
    measurement?: unknown;
    environment: Record<string, unknown>;
  },
): WorkResultWrite {
  return recordWorkResult(jobId, agentId, {
    status: payload.status,
    result_json: payload.result != null ? JSON.stringify(payload.result) : null,
    error: typeof payload.error === "string" ? payload.error : payload.error != null ? String(payload.error) : null,
    measurement_json: payload.measurement != null ? JSON.stringify(payload.measurement) : null,
    result_environment_json: JSON.stringify(payload.environment),
  });
}
