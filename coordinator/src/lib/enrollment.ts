// SPDX-License-Identifier: Apache-2.0
// Enrollment lifecycle for the push channel: mint enrollment tokens, take a node's self-description
// into a PENDING agent, let an admin approve/deny, and deliver the one-time session token when the
// node polls. SERVER-ONLY. All secrets are minted here and only their hashes reach the DB.
import "server-only";
import { randomBytes } from "node:crypto";
import { hashToken, verifyEnrollmentToken } from "./node-auth";
import {
  activateAgent,
  createPendingAgent,
  getAgentByEnrollmentId,
  insertEnrollmentToken,
  listAgents,
  listAgentsByStatus,
  markEnrollmentTokenUsed,
  revokeAgent,
  setAgentStatus,
  takePendingSessionToken,
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

/** Enroll a node: verify the (unused) enrollment token, consume it, and create a PENDING agent.
 *  Returns the enrollment handle, or null if the token is invalid/used (→ the route answers 401). */
export function enroll(
  token: string,
  self: SelfDescription,
): { enrollment_id: string; status: "pending" } | null {
  const tokRow = verifyEnrollmentToken(token);
  if (!tokRow) return null;

  const enrollment_id = newId("enr");
  createPendingAgent({
    machine_key: typeof self?.machine_key === "string" ? self.machine_key : "",
    enrollment_id,
    enrollment_token_id: tokRow.id, // bind: only this token may poll this agent (IDOR guard)
    identity_json: self?.identity != null ? JSON.stringify(self.identity) : null,
    caps_json: self?.capabilities != null ? JSON.stringify(self.capabilities) : null,
    environment_json: self?.environment != null ? JSON.stringify(self.environment) : null,
  });
  markEnrollmentTokenUsed(tokRow.id); // single-use for enroll; the token can still poll (allowUsed)
  return { enrollment_id, status: "pending" };
}

/** Result of an enrollment poll. The route maps each to an HTTP response / wire body. */
export type PollResult =
  | { kind: "unauthorized" }
  | { kind: "not_found" }
  | { kind: "pending" }
  | { kind: "active"; session_token: string }
  | { kind: "consumed" } // active, but the one-time token was already delivered
  | { kind: "denied" };

/** Poll for approval. Auth: the enrollment token (allowUsed — enroll already consumed it). On the
 *  first poll after approval, hands back the session token exactly once, then it's gone. */
export function pollApproval(enrollmentId: string, token: string): PollResult {
  const tokRow = verifyEnrollmentToken(token, { allowUsed: true });
  if (!tokRow) return { kind: "unauthorized" };
  const agent = getAgentByEnrollmentId(enrollmentId);
  if (!agent) return { kind: "not_found" };
  // A token may only poll the enrollment IT created — a valid-but-different token can't lift
  // another node's one-time session token (IDOR guard).
  if (agent.enrollment_token_id !== tokRow.id) return { kind: "unauthorized" };
  if (agent.status === "denied") return { kind: "denied" };
  if (agent.status !== "active") return { kind: "pending" };

  const token_once = takePendingSessionToken(agent.id);
  if (token_once == null) return { kind: "consumed" };
  return { kind: "active", session_token: token_once };
}

/** Approve a PENDING agent: mint the session token, persist its hash, stash the plaintext for one
 *  poll. `activateAgent` only flips a still-pending row, so a double-approve is a safe no-op. */
export function approveAgent(id: number): void {
  const token = newToken();
  activateAgent(id, hashToken(token), token);
}

export function denyAgent(id: number): void {
  setAgentStatus(id, "denied");
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
  caps_count: number;
}

/** Count an agent's advertised capabilities from its stored caps_json (a JSON array), 0 if absent
 *  or malformed. Never throws — a bad blob just yields 0. */
export function summarizeAgent(a: AgentRow): AgentSummary {
  let caps_count = 0;
  if (a.caps_json) {
    try {
      const parsed = JSON.parse(a.caps_json);
      if (Array.isArray(parsed)) caps_count = parsed.length;
    } catch {
      /* malformed caps_json → 0 */
    }
  }
  return {
    id: a.id,
    machine_key: a.machine_key,
    status: a.status,
    last_seen: a.last_seen,
    caps_count,
  };
}

/** Every enrolled agent, newest first, as token-free summaries for the dashboard. */
export function listAgentSummaries(): AgentSummary[] {
  return listAgents().map(summarizeAgent);
}
