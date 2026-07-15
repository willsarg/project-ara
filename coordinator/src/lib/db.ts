// SPDX-License-Identifier: Apache-2.0
// The agent registry — SQLite, volume-mountable. Holds phone-home agents, enrollment tokens, and the
// work queue; this module is SERVER-ONLY and must never be imported from a Client Component.
import "server-only";
import Database from "better-sqlite3";
import { randomBytes, scryptSync, timingSafeEqual } from "node:crypto";
import { mkdirSync } from "node:fs";
import path from "node:path";

// Statically scoped to cwd/data so Turbopack's Node File Trace (for the standalone build output)
// can see this resolves under the project root and doesn't fall back to tracing the whole project.
// ARA_COORDINATOR_DB still overrides it — that value is only ever consumed at RUNTIME (below), never
// at build time, so the override can't influence what NFT decides to include.
const DB_PATH = process.env.ARA_COORDINATOR_DB || path.join(process.cwd(), "data", "coordinator.db");

let _db: Database.Database | null = null;

function open(): Database.Database {
  if (_db) return _db;
  // Ensure the parent dir exists (e.g. ./data on a fresh checkout or empty volume). The resolved
  // path is only known at runtime (ARA_COORDINATOR_DB may override it), so tell the build-time
  // tracer not to try to statically follow it — see the comment on DB_PATH above.
  mkdirSync(path.dirname(path.resolve(/* turbopackIgnore: true */ DB_PATH)), { recursive: true });
  const db = new Database(DB_PATH);
  db.pragma("journal_mode = WAL");
  db.exec(`
    CREATE TABLE IF NOT EXISTS meta (
      key   TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );

    -- Phone-home (push) model: the node initiates everything. An 'agent' is a node that enrolled
    -- through the push channel — the coordinator never connects back to it.
    CREATE TABLE IF NOT EXISTS agents (
      id                    INTEGER PRIMARY KEY AUTOINCREMENT,
      machine_key           TEXT NOT NULL,
      enrollment_id         TEXT NOT NULL UNIQUE,
      status                TEXT NOT NULL DEFAULT 'pending',
      session_token_hash    TEXT,          -- permanent: sha256(session token); the plaintext is never stored
      pending_session_token TEXT,          -- transient: plaintext held until the new session authenticates
      identity_json         TEXT,
      caps_json             TEXT,
      environment_json      TEXT,
      enrollment_token_id   INTEGER,        -- binds the poll: ONLY this token may poll this agent (IDOR guard)
      created_at            TEXT NOT NULL DEFAULT (datetime('now')),
      last_seen             TEXT
    );
    -- Enrollment tokens: we store only the sha256 hash. Single-use for enroll (used flag).
    CREATE TABLE IF NOT EXISTS enrollment_tokens (
      id         INTEGER PRIMARY KEY AUTOINCREMENT,
      token_hash TEXT NOT NULL UNIQUE,
      used       INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    -- The server-owned dispatch queue. queued -> dispatched (picked up via long-poll) -> done/failed.
    CREATE TABLE IF NOT EXISTS work (
      id               TEXT PRIMARY KEY,
      agent_id         INTEGER NOT NULL,
      kind             TEXT NOT NULL,
      args_json        TEXT,
      status           TEXT NOT NULL DEFAULT 'queued',
      result_json      TEXT,
      error            TEXT,
      measurement_json TEXT,
      created_at       TEXT NOT NULL DEFAULT (datetime('now')),
      dispatched_at    TEXT,
      finished_at      TEXT
    );
  `);
  _db = db;
  return db;
}

// --- meta: small persisted settings (generated admin password, session secret) -----------------

function getMeta(key: string): string | null {
  const row = open().prepare("SELECT value FROM meta WHERE key = ?").get(key) as
    | { value: string }
    | undefined;
  return row?.value ?? null;
}

function setMeta(key: string, value: string): void {
  open()
    .prepare("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value")
    .run(key, value);
}

// Admin password is NEVER stored in plaintext. When no env password is set we generate one, log it
// ONCE for the operator, and persist only a salted scrypt hash — so a leaked DB doesn't leak the
// credential. Login verifies by hashing the submitted password with the stored salt (timing-safe).

function _scrypt(password: string, salt: Buffer): Buffer {
  return scryptSync(password, salt, 32);
}

function _eq(a: Buffer, b: Buffer): boolean {
  return a.length === b.length && timingSafeEqual(a, b);
}

/** First-run setup of the generated admin password: hash+persist, log the plaintext exactly once.
 *  No-op when ARA_COORDINATOR_PASSWORD is set (that path stores nothing) or a hash already exists. */
export function ensureAdminPassword(): void {
  if (process.env.ARA_COORDINATOR_PASSWORD) return;
  if (getMeta("admin_pw_hash")) return;
  const generated = randomBytes(15).toString("base64url");
  const salt = randomBytes(16);
  setMeta("admin_pw_salt", salt.toString("hex"));
  setMeta("admin_pw_hash", _scrypt(generated, salt).toString("hex"));
  open().prepare("DELETE FROM meta WHERE key = 'admin_password'").run(); // scrub any legacy plaintext
  console.log(
    `\n[ara-coordinator] No ARA_COORDINATOR_PASSWORD set — generated an admin password:\n` +
      `[ara-coordinator]   ${generated}\n` +
      `[ara-coordinator] Set ARA_COORDINATOR_PASSWORD to override. Logged once; only its hash is stored.\n`,
  );
}

/** Verify a submitted admin password, constant-time. Env password → direct compare; otherwise the
 *  salted scrypt hash. Never reveals or returns the stored credential. */
export function verifyAdminPassword(submitted: string): boolean {
  const fromEnv = process.env.ARA_COORDINATOR_PASSWORD;
  if (fromEnv && fromEnv.length > 0) {
    return _eq(Buffer.from(submitted), Buffer.from(fromEnv));
  }
  ensureAdminPassword();
  const saltHex = getMeta("admin_pw_salt");
  const hashHex = getMeta("admin_pw_hash");
  if (!saltHex || !hashHex) return false;
  return _eq(_scrypt(submitted, Buffer.from(saltHex, "hex")), Buffer.from(hashHex, "hex"));
}

// =================================================================================================
// Phone-home (push) CRUD. These are raw row helpers — hashing, token minting, and auth live in
// node-auth/enrollment/work.
// =================================================================================================

export interface AgentRow {
  id: number;
  machine_key: string;
  enrollment_id: string;
  status: string; // 'pending' | 'active' | 'denied'
  session_token_hash: string | null;
  pending_session_token: string | null;
  identity_json: string | null;
  caps_json: string | null;
  environment_json: string | null;
  enrollment_token_id: number | null;
  created_at: string;
  last_seen: string | null;
}

export interface EnrollmentTokenRow {
  id: number;
  token_hash: string;
  used: number; // 0/1
  created_at: string;
}

export interface WorkRow {
  id: string;
  agent_id: number;
  kind: string;
  args_json: string | null;
  status: string; // 'queued' | 'dispatched' | 'done' | 'failed'
  result_json: string | null;
  error: string | null;
  measurement_json: string | null;
  created_at: string;
  dispatched_at: string | null;
  finished_at: string | null;
}

// --- enrollment tokens ---------------------------------------------------------------------------

export function insertEnrollmentToken(tokenHash: string): void {
  open().prepare("INSERT INTO enrollment_tokens (token_hash) VALUES (?)").run(tokenHash);
}

export function getEnrollmentTokenByHash(tokenHash: string): EnrollmentTokenRow | null {
  return (open()
    .prepare("SELECT * FROM enrollment_tokens WHERE token_hash = ?")
    .get(tokenHash) as EnrollmentTokenRow | undefined) ?? null;
}

export function markEnrollmentTokenUsed(id: number): void {
  open().prepare("UPDATE enrollment_tokens SET used = 1 WHERE id = ?").run(id);
}

// --- agents --------------------------------------------------------------------------------------

export function createPendingAgent(a: {
  machine_key: string;
  enrollment_id: string;
  enrollment_token_id: number;
  identity_json: string | null;
  caps_json: string | null;
  environment_json: string | null;
}): AgentRow {
  const info = open()
    .prepare(
      `INSERT INTO agents (machine_key, enrollment_id, enrollment_token_id, identity_json, caps_json, environment_json)
       VALUES (@machine_key, @enrollment_id, @enrollment_token_id, @identity_json, @caps_json, @environment_json)`,
    )
    .run(a);
  return getAgentById(Number(info.lastInsertRowid))!;
}

export function getAgentById(id: number): AgentRow | null {
  return (open().prepare("SELECT * FROM agents WHERE id = ?").get(id) as AgentRow | undefined) ?? null;
}

export function getAgentByEnrollmentId(enrollmentId: string): AgentRow | null {
  return (open()
    .prepare("SELECT * FROM agents WHERE enrollment_id = ?")
    .get(enrollmentId) as AgentRow | undefined) ?? null;
}

export function getAgentByEnrollmentTokenId(enrollmentTokenId: number): AgentRow | null {
  return (open()
    .prepare("SELECT * FROM agents WHERE enrollment_token_id = ? ORDER BY id DESC LIMIT 1")
    .get(enrollmentTokenId) as AgentRow | undefined) ?? null;
}

export function getAgentByMachineKey(machineKey: string): AgentRow | null {
  return (open()
    .prepare("SELECT * FROM agents WHERE machine_key = ? ORDER BY id DESC LIMIT 1")
    .get(machineKey) as AgentRow | undefined) ?? null;
}

export function getAgentBySessionHash(sessionTokenHash: string): AgentRow | null {
  return (open()
    .prepare("SELECT * FROM agents WHERE session_token_hash = ?")
    .get(sessionTokenHash) as AgentRow | undefined) ?? null;
}

/** Re-enroll a known machine on its existing identity. Rotating the enrollment handle/token binding
 *  prevents the old enrollment token from polling, while clearing session state revokes the old
 *  bearer immediately. Work remains attached to the stable agent id. */
export function reenrollAgent(
  id: number,
  a: {
    enrollment_id: string;
    enrollment_token_id: number;
    identity_json: string | null;
    caps_json: string | null;
    environment_json: string | null;
  },
): AgentRow {
  open()
    .prepare(
      `UPDATE agents
       SET enrollment_id = @enrollment_id, enrollment_token_id = @enrollment_token_id,
           status = 'pending', session_token_hash = NULL, pending_session_token = NULL,
           identity_json = @identity_json, caps_json = @caps_json,
           environment_json = @environment_json, last_seen = NULL
       WHERE id = @id`,
    )
    .run({ id, ...a });
  return getAgentById(id)!;
}

/** Approve: flip to active, persist the session-token HASH, and stash plaintext until session auth. */
export function activateAgent(id: number, sessionTokenHash: string, plaintext: string): void {
  open()
    .prepare(
      `UPDATE agents SET status = 'active', session_token_hash = ?, pending_session_token = ?
       WHERE id = ? AND status = 'pending'`,
    )
    .run(sessionTokenHash, plaintext, id);
}

export function setAgentStatus(id: number, status: string): void {
  open().prepare("UPDATE agents SET status = ? WHERE id = ?").run(status, id);
}

/** Revoke an agent: deny it and erase both the live hash and any unacknowledged plaintext, so no
 *  session token can authenticate and no revoked credential remains recoverable from storage. */
export function revokeAgent(id: number): void {
  open()
    .prepare(
      `UPDATE agents
       SET status = 'denied', session_token_hash = NULL, pending_session_token = NULL
       WHERE id = ?`,
    )
    .run(id);
}

/** Read the transient plaintext without consuming it. Poll retries return this same value until the
 *  node proves receipt by authenticating with the matching session token. */
export function getPendingSessionToken(id: number): string | null {
  const row = open().prepare("SELECT pending_session_token FROM agents WHERE id = ?").get(id) as
    | { pending_session_token: string | null }
    | undefined;
  return row?.pending_session_token ?? null;
}

/** A successful bearer authentication acknowledges durable session-token receipt. The hash guard
 *  ensures an old concurrent auth can never clear plaintext for a newly rotated session. */
export function acknowledgeSessionToken(id: number, sessionTokenHash: string): void {
  open()
    .prepare(
      `UPDATE agents SET pending_session_token = NULL
       WHERE id = ? AND status = 'active' AND session_token_hash = ?`,
    )
    .run(id, sessionTokenHash);
}

export function touchAgentLastSeen(id: number): void {
  open().prepare("UPDATE agents SET last_seen = datetime('now') WHERE id = ?").run(id);
}

export function listAgentsByStatus(status: string): AgentRow[] {
  return open()
    .prepare("SELECT * FROM agents WHERE status = ? ORDER BY created_at DESC, id DESC")
    .all(status) as AgentRow[];
}

/** Every agent, newest first — regardless of status. Backs the dashboard's enrolled-agents view. */
export function listAgents(): AgentRow[] {
  return open()
    .prepare("SELECT * FROM agents ORDER BY created_at DESC, id DESC")
    .all() as AgentRow[];
}

// --- work queue ----------------------------------------------------------------------------------

export function insertWork(id: string, agentId: number, kind: string, argsJson: string | null): void {
  open()
    .prepare("INSERT INTO work (id, agent_id, kind, args_json) VALUES (?, ?, ?, ?)")
    .run(id, agentId, kind, argsJson);
}

/** The claimed job as returned by the atomic dispatch (RETURNING projection). */
export interface ClaimedWork {
  id: string;
  kind: string;
  args_json: string | null;
}

/** Atomically claim the oldest queued job for this agent: flip queued→dispatched and return the
 *  claimed row in ONE statement (SQLite RETURNING), so two concurrent polls can never double-dispatch
 *  the same job. Returns the claimed row, or null when nothing is queued. */
export function claimNextWorkForAgent(agentId: number): ClaimedWork | null {
  return (open()
    .prepare(
      `UPDATE work SET status = 'dispatched', dispatched_at = datetime('now')
       WHERE id = (
         SELECT id FROM work WHERE agent_id = ? AND status = 'queued'
         ORDER BY created_at, id LIMIT 1
       )
       RETURNING id, kind, args_json`,
    )
    .get(agentId) as ClaimedWork | undefined) ?? null;
}

export function getWorkById(id: string): WorkRow | null {
  return (open().prepare("SELECT * FROM work WHERE id = ?").get(id) as WorkRow | undefined) ?? null;
}

export function recordWorkResult(
  id: string,
  p: { status: string; result_json: string | null; error: string | null; measurement_json: string | null },
): void {
  open()
    .prepare(
      `UPDATE work
       SET status = @status, result_json = @result_json, error = @error,
           measurement_json = @measurement_json, finished_at = datetime('now')
       WHERE id = @id`,
    )
    .run({ id, ...p });
}
