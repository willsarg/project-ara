// SPDX-License-Identifier: Apache-2.0
// Enrollment lifecycle for the push channel: mint enrollment tokens, take a node's self-description
// into a PENDING agent, let an admin approve/deny, and recoverably deliver the session token when
// the node polls. SERVER-ONLY. Session plaintext is retained only until successful node auth.
import "server-only";
import { randomBytes } from "node:crypto";
import { hashToken, verifyEnrollmentToken } from "./node-auth";
import {
  activateAgent,
  denyPendingAgent,
  enrollAgentAtomically,
  getAgentByEnrollmentId,
  getPendingSessionToken,
  insertEnrollmentToken,
  listAgents,
  listAgentsByStatus,
  revokeAgent,
  type AgentRow,
} from "./db";

/** The node's enroll payload (enroll.request wire shape). Stored verbatim as JSON on the agent. */
export interface SelfDescription {
  machine_key?: unknown;
  identity?: unknown;
  capabilities?: unknown;
  environment?: unknown;
  profile_projection?: unknown;
}

const newToken = () => randomBytes(32).toString("base64url"); // high-entropy secret, shown once
const newId = (prefix: string) => `${prefix}_${randomBytes(12).toString("hex")}`;

/** Mint an enrollment token. Returns the PLAINTEXT once; only its hash is persisted. */
export function issueEnrollmentToken(): { token: string } {
  const token = newToken();
  insertEnrollmentToken(hashToken(token));
  return { token };
}

/** Enroll a node. A used token retries its original enrollment idempotently. Every fresh generic
 * token creates a distinct pending owner; it cannot rotate or inherit an existing node's work. */
export function enroll(
  token: string,
  self: SelfDescription,
): { enrollment_id: string; status: "pending" } | null {
  const tokRow = verifyEnrollmentToken(token, { allowUsed: true });
  if (!tokRow) return null;

  const enrollment_id = newId("enr");
  const machine_key = typeof self?.machine_key === "string" ? self.machine_key : "";
  const stored = {
    enrollment_id,
    identity_json: self?.identity != null ? JSON.stringify(self.identity) : null,
    caps_json: self?.capabilities != null ? JSON.stringify(self.capabilities) : null,
    environment_json: self?.environment != null ? JSON.stringify(self.environment) : null,
  };
  const agent = enrollAgentAtomically({ token_id: tokRow.id, machine_key, ...stored });
  return agent ? { enrollment_id: agent.enrollment_id, status: "pending" } : null;
}

/** Result of an enrollment poll. The route maps each to an HTTP response / wire body. */
export type PollResult =
  | { kind: "unauthorized" }
  | { kind: "not_found" }
  | { kind: "pending" }
  | { kind: "active"; session_token: string }
  | { kind: "consumed" } // active, and successful session auth acknowledged token receipt
  | { kind: "denied" };

/** Poll for approval. Auth: the enrollment token bound to this enrollment. After approval, repeats
 *  the session token until successful session authentication acknowledges durable receipt. */
export function pollApproval(enrollmentId: string, token: string): PollResult {
  const tokRow = verifyEnrollmentToken(token, { allowUsed: true });
  if (!tokRow) return { kind: "unauthorized" };
  const agent = getAgentByEnrollmentId(enrollmentId);
  if (!agent) return { kind: "not_found" };
  // A token may only poll the enrollment IT created — a valid-but-different token can't lift
  // another node's pending session token (IDOR guard).
  if (agent.enrollment_token_id !== tokRow.id) return { kind: "unauthorized" };
  if (agent.status === "denied") return { kind: "denied" };
  if (agent.status !== "active") return { kind: "pending" };

  const pendingToken = getPendingSessionToken(agent.id);
  if (pendingToken == null) return { kind: "consumed" };
  return { kind: "active", session_token: pendingToken };
}

/** Approve a PENDING agent: mint the session token, persist its hash, and stash plaintext until the
 *  node authenticates. `activateAgent` only flips a still-pending row, so double-approve is safe. */
export function approveAgent(id: number): void {
  const token = newToken();
  activateAgent(id, hashToken(token), token);
}

export function denyAgent(id: number): boolean {
  return denyPendingAgent(id);
}

/** Revoke an approved agent: deny it and invalidate its session token (see db.revokeAgent). */
export function revoke(id: number): void {
  revokeAgent(id);
}

export function listPending(): AgentRow[] {
  return listAgentsByStatus("pending");
}

export function listActive(): AgentRow[] {
  return listAgentsByStatus("active");
}

/** A render-ready, token-free view of an enrolled agent for the dashboard. Secrets (session token,
 *  its hash, the transient plaintext) are deliberately excluded — this shape is safe to render. */
export interface AgentSummary {
  id: number;
  machine_key: string;
  status: string;
  last_seen: string | null;
  recently_seen: boolean;
  caps_count: number;
  serve_models: { id: string; engine: string; authority?: string }[];
}

export const RECENT_HEARTBEAT_WINDOW_MS = 60_000;

export function wasAgentSeenRecently(lastSeen: string | null, now: number = Date.now()): boolean {
  if (!lastSeen) return false;
  const timestamp = /(?:Z|[+-]\d\d:\d\d)$/.test(lastSeen) ? lastSeen : `${lastSeen}Z`;
  const seenAt = Date.parse(timestamp);
  const age = now - seenAt;
  return Number.isFinite(seenAt) && age >= 0 && age <= RECENT_HEARTBEAT_WINDOW_MS;
}

const canonicalEngine = (engine: string) =>
  engine === "wmx" ? "mlx" : engine === "wcx" ? "cuda" : engine;

/** Count an agent's advertised capabilities from its stored caps_json (a JSON array), 0 if absent
 *  or malformed, and extract the `serve_model` entries (the models this node can serve). Never
 *  throws — a bad blob just yields 0 / []. */
export function summarizeAgent(a: AgentRow): AgentSummary {
  let caps_count = 0;
  let serve_models: { id: string; engine: string; authority?: string }[] = [];
  if (a.caps_json) {
    try {
      const parsed = JSON.parse(a.caps_json);
      if (Array.isArray(parsed)) {
        caps_count = parsed.length;
        serve_models = parsed
          .filter(
            (c): c is { kind: unknown; id: string; engine?: unknown; authority?: unknown } =>
              typeof c === "object" && c !== null && c.kind === "serve_model" && typeof c.id === "string",
          )
          .map((c) => ({
            id: c.id,
            engine: canonicalEngine(typeof c.engine === "string" ? c.engine : "?"),
            ...(typeof c.authority === "string" ? { authority: c.authority } : {}),
          }));
      }
    } catch {
      /* malformed caps_json → 0 / [] */
    }
  }
  return {
    id: a.id,
    machine_key: a.machine_key,
    status: a.status,
    last_seen: a.last_seen,
    recently_seen: a.status === "active" && wasAgentSeenRecently(a.last_seen),
    caps_count,
    serve_models,
  };
}

export interface AuthorizedServeModel {
  id: string;
  engine: string;
  authority: string;
}

/** Resolve an admin selection against the capability blob enrolled by this exact node. The
 * coordinator echoes this opaque authority; the node independently re-derives it before runtime
 * execution, so dashboard input cannot substitute a model or target. */
export function authorizedServeModel(
  agent: AgentRow, authority: string,
): AuthorizedServeModel | null {
  if (!agent.caps_json || !authority) return null;
  try {
    const parsed: unknown = JSON.parse(agent.caps_json);
    if (!Array.isArray(parsed)) return null;
    const cap = parsed.find((candidate): candidate is Record<string, unknown> => (
      typeof candidate === "object" && candidate !== null
      && candidate.kind === "serve_model"
      && candidate.evidence === "characterized"
      && candidate.authority === authority
      && typeof candidate.id === "string" && candidate.id.length > 0
      && typeof candidate.engine === "string" && candidate.engine.length > 0
    ));
    return cap ? {
      id: cap.id as string,
      engine: canonicalEngine(cap.engine as string),
      authority,
    } : null;
  } catch {
    return null;
  }
}

/** Every enrolled agent, newest first, as token-free summaries for the dashboard. */
export function listAgentSummaries(): AgentSummary[] {
  return listAgents().map(summarizeAgent);
}
