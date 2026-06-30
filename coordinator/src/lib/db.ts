// SPDX-License-Identifier: Apache-2.0
// The node registry — SQLite, volume-mountable. Holds node tokens; this module is SERVER-ONLY
// and must never be imported from a Client Component.
import "server-only";
import Database from "better-sqlite3";
import { randomBytes, scryptSync, timingSafeEqual } from "node:crypto";
import { mkdirSync } from "node:fs";
import path from "node:path";

export interface Node {
  id: number;
  name: string;
  base_url: string;
  token: string;
  enabled: number; // SQLite has no bool — 0/1
}

const DB_PATH = process.env.ARA_COORDINATOR_DB || "./data/coordinator.db";

let _db: Database.Database | null = null;

function open(): Database.Database {
  if (_db) return _db;
  // Ensure the parent dir exists (e.g. ./data on a fresh checkout or empty volume).
  mkdirSync(path.dirname(path.resolve(DB_PATH)), { recursive: true });
  const db = new Database(DB_PATH);
  db.pragma("journal_mode = WAL");
  db.exec(`
    CREATE TABLE IF NOT EXISTS nodes (
      id        INTEGER PRIMARY KEY AUTOINCREMENT,
      name      TEXT NOT NULL UNIQUE,
      base_url  TEXT NOT NULL,
      token     TEXT NOT NULL,
      enabled   INTEGER NOT NULL DEFAULT 1
    );
    CREATE TABLE IF NOT EXISTS meta (
      key   TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );
  `);
  _db = db;
  return db;
}

export function listNodes(): Node[] {
  return open().prepare("SELECT * FROM nodes ORDER BY name").all() as Node[];
}

export function addNode(name: string, base_url: string, token: string): void {
  open()
    .prepare("INSERT INTO nodes (name, base_url, token, enabled) VALUES (?, ?, ?, 1)")
    .run(name.trim(), base_url.trim().replace(/\/+$/, ""), token.trim());
}

export function deleteNode(id: number): void {
  open().prepare("DELETE FROM nodes WHERE id = ?").run(id);
}

export function toggleNode(id: number): void {
  open().prepare("UPDATE nodes SET enabled = 1 - enabled WHERE id = ?").run(id);
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
