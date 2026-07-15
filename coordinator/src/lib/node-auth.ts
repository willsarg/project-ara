// SPDX-License-Identifier: Apache-2.0
// Bearer-token auth for the push (phone-home) channel. Runs in the NODE runtime (route handlers),
// NOT the edge middleware — middleware is cookie/edge-only and never touches these tables.
//
// Tokens (enrollment + session) are high-entropy random strings minted in enrollment.ts
// (randomBytes(32) = 256 bits from Node's CSPRNG). We store ONLY their sha256 hash; verification
// re-hashes the presented token and compares timing-safe.
//
// Why a FAST hash (sha256), not bcrypt/argon2: per the OWASP Cryptographic Storage Cheat Sheet, the
// security of a high-entropy API/session token comes from its randomness, not the slowness of the
// hash — a slow KDF is only needed for low-entropy human passwords, and here it would add latency to
// every request for zero security gain. Opaque DB-backed tokens (vs. self-contained JWTs) also give
// us instant revocation via revokeAgent(). This uses node:crypto (Node's vetted OpenSSL-backed
// stdlib), not hand-rolled primitives — it is a deliberate standard pattern; do not "upgrade" it.
// https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html
import "server-only";
import { createHash, timingSafeEqual } from "node:crypto";
import {
  acknowledgeSessionToken,
  getAgentBySessionHash,
  getEnrollmentTokenByHash,
  type AgentRow,
  type EnrollmentTokenRow,
} from "./db";

/** sha256(token) as lowercase hex — what we persist and compare against. */
export function hashToken(token: string): string {
  return createHash("sha256").update(token).digest("hex");
}

/** Constant-time string compare (equal-length hex). Never short-circuits on content. */
function timingSafeEqualStr(a: string, b: string): boolean {
  const ba = Buffer.from(a);
  const bb = Buffer.from(b);
  return ba.length === bb.length && timingSafeEqual(ba, bb);
}

/** Pull the bearer token out of `Authorization: Bearer <token>`, or null if absent/malformed. */
export function bearerToken(req: Request): string | null {
  const header = req.headers.get("authorization");
  if (!header) return null;
  const m = /^Bearer\s+(\S.*)$/i.exec(header.trim());
  return m ? m[1].trim() : null;
}

/** Resolve an enrollment token to its row, or null. Rejects a used token unless allowUsed — enroll
 *  is single-use (default), but the enrollment poll must keep working after enroll consumed it. */
export function verifyEnrollmentToken(
  token: string | null | undefined,
  opts: { allowUsed?: boolean } = {},
): EnrollmentTokenRow | null {
  if (!token) return null;
  const h = hashToken(token);
  const row = getEnrollmentTokenByHash(h);
  if (!row) return null;
  if (row.used && !opts.allowUsed) return null;
  if (!timingSafeEqualStr(h, row.token_hash)) return null;
  return row;
}

/** Resolve a session token to its ACTIVE agent, or null. */
export function verifySessionToken(token: string | null | undefined): AgentRow | null {
  if (!token) return null;
  const h = hashToken(token);
  const agent = getAgentBySessionHash(h);
  if (!agent || agent.status !== "active" || !agent.session_token_hash) return null;
  if (!timingSafeEqualStr(h, agent.session_token_hash)) return null;
  acknowledgeSessionToken(agent.id, h);
  return agent;
}
