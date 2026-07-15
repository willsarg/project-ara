// SPDX-License-Identifier: Apache-2.0
// The agent registry — SQLite, volume-mountable. Holds phone-home agents, enrollment tokens, and the
// work queue; this module is SERVER-ONLY and must never be imported from a Client Component.
import "server-only";
import Database from "better-sqlite3";
import { randomBytes, scryptSync, timingSafeEqual } from "node:crypto";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { ALLOWED_JOB_KINDS } from "./job-kinds";

// Statically scoped to cwd/data so Turbopack's Node File Trace (for the standalone build output)
// can see this resolves under the project root and doesn't fall back to tracing the whole project.
// ARA_COORDINATOR_DB still overrides it — that value is only ever consumed at RUNTIME (below), never
// at build time, so the override can't influence what NFT decides to include.
const DB_PATH = process.env.ARA_COORDINATOR_DB || path.join(process.cwd(), "data", "coordinator.db");

let _db: Database.Database | null = null;

/** Open/migrate the registry and prove it can execute a read. Used by readiness health checks. */
export function checkDatabaseReady(): void {
  open().prepare("SELECT 1").get();
}

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
    -- The server-owned dispatch queue. queued -> offered -> dispatched (durably acked) -> done/failed.
    CREATE TABLE IF NOT EXISTS work (
      id               TEXT PRIMARY KEY,
      agent_id         INTEGER NOT NULL,
      kind             TEXT NOT NULL,
      args_json        TEXT,
      status           TEXT NOT NULL DEFAULT 'queued',
      result_json      TEXT,
      error            TEXT,
      measurement_json TEXT,
      result_environment_json TEXT,
      created_at       TEXT NOT NULL DEFAULT (datetime('now')),
      dispatched_at    TEXT,
      finished_at      TEXT
    );
  `);
  // Pre-transaction databases could contain duplicate token bindings after a race. Preserve every
  // agent/job, keep only the newest poll binding, then enforce one bound enrollment per token.
  migrateEnrollmentBindings(db);
  migrateWorkResultEnvironment(db);
  const allowedKindsSql = ALLOWED_JOB_KINDS.map((kind) => `'${kind}'`).join(", ");
  // Triggers migrate existing databases without rebuilding the work table or rejecting historical
  // rows. From this version onward, both new inserts and kind updates are defended in SQLite too.
  db.exec(`
    CREATE TRIGGER IF NOT EXISTS work_kind_insert_guard
    BEFORE INSERT ON work
    WHEN NEW.kind NOT IN (${allowedKindsSql})
    BEGIN
      SELECT RAISE(ABORT, 'invalid job kind');
    END;
    CREATE TRIGGER IF NOT EXISTS work_kind_update_guard
    BEFORE UPDATE OF kind ON work
    WHEN NEW.kind NOT IN (${allowedKindsSql})
    BEGIN
      SELECT RAISE(ABORT, 'invalid job kind');
    END;
  `);
  _db = db;
  return db;
}

/** Safely upgrade pre-transaction databases before installing the unique binding index. */
export function migrateEnrollmentBindings(db: Database.Database): void {
  db.transaction(() => {
    db.exec(`
      UPDATE agents AS older
      SET enrollment_token_id = NULL
      WHERE enrollment_token_id IS NOT NULL
        AND EXISTS (
          SELECT 1 FROM agents AS newer
          WHERE newer.enrollment_token_id = older.enrollment_token_id AND newer.id > older.id
        );
      CREATE UNIQUE INDEX IF NOT EXISTS agents_one_enrollment_per_token
      ON agents (enrollment_token_id) WHERE enrollment_token_id IS NOT NULL;
    `);
  }).immediate();
}

/** Add result-environment storage to databases created before result provenance was persisted. */
export function migrateWorkResultEnvironment(db: Database.Database): void {
  db.transaction(() => {
    const columns = db.prepare("PRAGMA table_info(work)").all() as { name: string }[];
    if (!columns.some((column) => column.name === "result_environment_json")) {
      db.exec("ALTER TABLE work ADD COLUMN result_environment_json TEXT");
    }
  }).immediate();
}

// --- meta: small persisted settings (generated admin password, session auth state) --------------

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

const SESSION_SECRET_KEY = "session_secret";
const SESSION_EPOCH_KEY = "session_epoch";

/** Resolve the session signing secret with the documented precedence. Explicit env secret wins;
 *  otherwise an env admin password supplies the stable derived key. With neither configured, a
 *  cryptographically random key is created once in SQLite and reused across bundles/restarts. */
export function ensureSessionSecret(): string {
  const configured = process.env.ARA_COORDINATOR_SECRET;
  if (configured) return configured;
  const password = process.env.ARA_COORDINATOR_PASSWORD;
  if (password) return `pw:${password}`;

  const candidate = randomBytes(32).toString("base64url");
  open()
    .prepare("INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)")
    .run(SESSION_SECRET_KEY, candidate);
  const persisted = getMeta(SESSION_SECRET_KEY);
  if (!persisted) throw new Error("ARA coordinator: failed to persist the generated session secret");
  return persisted;
}

function parseSessionEpoch(value: string | null): number {
  if (value === null) return 0;
  if (!/^(?:0|[1-9]\d*)$/.test(value)) {
    throw new Error("ARA coordinator: persisted session epoch is invalid");
  }
  const epoch = Number(value);
  if (!Number.isSafeInteger(epoch)) {
    throw new Error("ARA coordinator: persisted session epoch exceeds the safe integer range");
  }
  return epoch;
}

/** Current durable revocation generation. Missing state is the initial generation zero. */
export function getSessionEpoch(): number {
  return parseSessionEpoch(getMeta(SESSION_EPOCH_KEY));
}

/** Atomically invalidate all sessions issued under the current generation. BEGIN IMMEDIATE makes
 *  read+increment+write safe across separate Next bundles, processes, and SQLite connections. */
export function advanceSessionEpoch(): number {
  const db = open();
  return db.transaction(() => {
    const current = parseSessionEpoch(getMeta(SESSION_EPOCH_KEY));
    if (current === Number.MAX_SAFE_INTEGER) {
      throw new Error("ARA coordinator: session epoch is exhausted");
    }
    const next = current + 1;
    setMeta(SESSION_EPOCH_KEY, String(next));
    return next;
  }).immediate();
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
  status: string; // 'queued' | 'offered' | 'dispatched' | 'done' | 'failed'
  result_json: string | null;
  error: string | null;
  measurement_json: string | null;
  result_environment_json: string | null;
  created_at: string;
  dispatched_at: string | null;
  finished_at: string | null;
}

export interface AdminWorkRow extends WorkRow {
  machine_key: string;
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

export function getAgentById(id: number): AgentRow | null {
  return (open().prepare("SELECT * FROM agents WHERE id = ?").get(id) as AgentRow | undefined) ?? null;
}

export function getAgentByEnrollmentId(enrollmentId: string): AgentRow | null {
  return (open()
    .prepare("SELECT * FROM agents WHERE enrollment_id = ?")
    .get(enrollmentId) as AgentRow | undefined) ?? null;
}

export function getAgentBySessionHash(sessionTokenHash: string): AgentRow | null {
  return (open()
    .prepare("SELECT * FROM agents WHERE session_token_hash = ?")
    .get(sessionTokenHash) as AgentRow | undefined) ?? null;
}

/** Consume one enrollment token and create/re-enroll its machine as one BEGIN IMMEDIATE unit.
 *  The write lock is acquired before reading `used` or the machine row, so another coordinator
 *  process cannot observe both as fresh and race us to a second binding. Used-token retries return
 *  the single row already bound to that token. */
export function enrollAgentAtomically(a: {
  token_id: number;
  machine_key: string;
  enrollment_id: string;
  identity_json: string | null;
  caps_json: string | null;
  environment_json: string | null;
}): AgentRow | null {
  const db = open();
  const complete = db.transaction((input: typeof a): AgentRow | null => {
    const token = db.prepare("SELECT used FROM enrollment_tokens WHERE id = ?").get(input.token_id) as
      | { used: number }
      | undefined;
    if (!token) return null;
    if (token.used) {
      return (db
        .prepare("SELECT * FROM agents WHERE enrollment_token_id = ? ORDER BY id DESC LIMIT 1")
        .get(input.token_id) as AgentRow | undefined) ?? null;
    }

    const existing = input.machine_key
      ? ((db
          .prepare("SELECT * FROM agents WHERE machine_key = ? ORDER BY id DESC LIMIT 1")
          .get(input.machine_key) as AgentRow | undefined) ?? null)
      : null;
    let id: number;
    if (existing) {
      db.prepare(
        `UPDATE agents
         SET enrollment_id = @enrollment_id, enrollment_token_id = @token_id,
             status = 'pending', session_token_hash = NULL, pending_session_token = NULL,
             identity_json = @identity_json, caps_json = @caps_json,
             environment_json = @environment_json, last_seen = NULL
         WHERE id = @id`,
      ).run({ ...input, id: existing.id });
      id = existing.id;
    } else {
      const inserted = db.prepare(
        `INSERT INTO agents
           (machine_key, enrollment_id, enrollment_token_id, identity_json, caps_json, environment_json)
         VALUES (@machine_key, @enrollment_id, @token_id, @identity_json, @caps_json, @environment_json)`,
      ).run(input);
      id = Number(inserted.lastInsertRowid);
    }
    db.prepare("UPDATE enrollment_tokens SET used = 1 WHERE id = ? AND used = 0").run(input.token_id);
    return db.prepare("SELECT * FROM agents WHERE id = ?").get(id) as AgentRow;
  });
  return complete.immediate(a);
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

export function insertWork(id: string, agentId: number, kind: string, argsJson: string | null): boolean {
  const inserted = open()
    .prepare(
      `INSERT INTO work (id, agent_id, kind, args_json)
       SELECT ?, ?, ?, ? WHERE EXISTS (
         SELECT 1 FROM agents WHERE id = ? AND status = 'active'
       )`,
    )
    .run(id, agentId, kind, argsJson, agentId);
  return inserted.changes === 1;
}

/** The job offer returned by the atomic queue claim (RETURNING projection). */
export interface ClaimedWork {
  id: string;
  kind: string;
  args_json: string | null;
}

/** Offer lease: the node must durably journal + ack within this window. Only unacknowledged offers
 *  expire; a dispatched long-running model job never expires merely because execution takes hours. */
export const OFFER_LEASE_SECONDS = 30;

/** Atomically offer the oldest queued job, or reclaim an expired unacknowledged offer. */
export function claimNextWorkForAgent(
  agentId: number, offerLeaseSeconds: number = OFFER_LEASE_SECONDS,
): ClaimedWork | null {
  return (open()
    .prepare(
      `UPDATE work SET status = 'offered', dispatched_at = datetime('now')
       WHERE id = (
         SELECT id FROM work
         WHERE agent_id = ? AND (
           status = 'queued'
           OR (status = 'offered' AND dispatched_at <= datetime('now', ?))
         )
         ORDER BY CASE status WHEN 'offered' THEN 0 ELSE 1 END, created_at, rowid LIMIT 1
       )
       RETURNING id, kind, args_json`,
    )
    .get(agentId, `-${offerLeaseSeconds} seconds`) as ClaimedWork | undefined) ?? null;
}

export type WorkAckResult = "ok" | "unknown" | "conflict";

/** Move an owned durable offer to dispatched. Repeated acks are idempotent. */
export function acknowledgeWorkForAgent(id: string, agentId: number): WorkAckResult {
  const db = open();
  const changed = db.prepare(
    `UPDATE work SET status = 'dispatched'
     WHERE id = ? AND agent_id = ? AND status = 'offered'`,
  ).run(id, agentId);
  if (changed.changes === 1) return "ok";
  const row = db.prepare("SELECT agent_id, status FROM work WHERE id = ?").get(id) as
    | { agent_id: number; status: string }
    | undefined;
  if (!row || row.agent_id !== agentId) return "unknown";
  return row.status === "dispatched" ? "ok" : "conflict";
}

export function getWorkById(id: string): WorkRow | null {
  return (open().prepare("SELECT * FROM work WHERE id = ?").get(id) as WorkRow | undefined) ?? null;
}

/** Newest persisted jobs for the administrator. Exact stored payload/provenance stays available
 * so the UI never invents a friendlier outcome than the node actually reported. */
export function listRecentWork(): AdminWorkRow[] {
  return open()
    .prepare(
      `SELECT work.*, agents.machine_key
       FROM work JOIN agents ON agents.id = work.agent_id
       ORDER BY work.created_at DESC, work.rowid DESC LIMIT 50`,
    )
    .all() as AdminWorkRow[];
}

export type WorkResultWrite = "recorded" | "already_recorded" | "unknown" | "conflict";

/** Atomically claim the dispatched -> terminal transition for an owned job. */
export function recordWorkResult(
  id: string,
  agentId: number,
  p: {
    status: string;
    result_json: string | null;
    error: string | null;
    measurement_json: string | null;
    result_environment_json: string;
  },
): WorkResultWrite {
  const db = open();
  const changed = db
    .prepare(
      `UPDATE work
       SET status = @status, result_json = @result_json, error = @error,
           measurement_json = @measurement_json,
           result_environment_json = @result_environment_json,
           finished_at = datetime('now')
       WHERE id = @id AND agent_id = @agent_id AND status = 'dispatched'
         AND @status IN ('done', 'failed')`,
    )
    .run({ id, agent_id: agentId, ...p });
  if (changed.changes === 1) return "recorded";
  const row = db.prepare("SELECT agent_id, status FROM work WHERE id = ?").get(id) as
    | { agent_id: number; status: string }
    | undefined;
  if (!row || row.agent_id !== agentId) return "unknown";
  return row.status === "done" || row.status === "failed" ? "already_recorded" : "conflict";
}
