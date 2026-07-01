// SPDX-License-Identifier: Apache-2.0
// The server-owned work dispatch queue. The node long-polls GET /api/work; the admin enqueues jobs.
// better-sqlite3 is SYNCHRONOUS, so `nextForAgent` must never busy-loop the event loop — it does a
// cheap sync read, and if there's nothing yet it AWAITS a short timer before the next read. SERVER-ONLY.
import "server-only";
import { randomBytes } from "node:crypto";
import {
  getQueuedWorkForAgent,
  getWorkById,
  insertWork,
  markWorkDispatched,
  recordWorkResult,
} from "./db";

/** The dispatched job as it goes on the wire (work.response `job`). */
export interface DispatchedJob {
  id: string;
  kind: string;
  args: Record<string, unknown>;
}

const POLL_STEP_MS = 250; // re-check cadence between sync reads during a long-poll

/** Queue a job for an agent. Returns the job id (also the wire `job.id`). */
export function enqueue(agentId: number, kind: string, args: Record<string, unknown>): string {
  const id = `job_${randomBytes(12).toString("hex")}`;
  insertWork(id, agentId, kind, JSON.stringify(args ?? {}));
  return id;
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/** Long-poll for the next queued job, up to `waitMs`. Returns the job (marking it dispatched) as
 *  soon as one is available, or null once the window elapses. Awaits between cheap sync reads so
 *  the synchronous DB never blocks the loop. */
export async function nextForAgent(agentId: number, waitMs: number): Promise<DispatchedJob | null> {
  const deadline = Date.now() + Math.max(0, waitMs);
  for (;;) {
    const row = getQueuedWorkForAgent(agentId);
    if (row) {
      markWorkDispatched(row.id);
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

/** Record a job's result. Returns false if the job id is unknown (→ the route answers 404). */
export function recordResult(
  jobId: string,
  payload: {
    status: string;
    result?: unknown;
    error?: unknown;
    measurement?: unknown;
  },
): boolean {
  if (!getWorkById(jobId)) return false;
  recordWorkResult(jobId, {
    status: payload.status,
    result_json: payload.result != null ? JSON.stringify(payload.result) : null,
    error: typeof payload.error === "string" ? payload.error : payload.error != null ? String(payload.error) : null,
    measurement_json: payload.measurement != null ? JSON.stringify(payload.measurement) : null,
  });
  return true;
}
