// SPDX-License-Identifier: Apache-2.0
// The node registry — SQLite, volume-mountable. Holds node tokens; this module is SERVER-ONLY
// and must never be imported from a Client Component.
import "server-only";
import Database from "better-sqlite3";
import { randomBytes } from "node:crypto";
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

/** The admin password: from env if set, else a stable one generated + persisted on first run. */
export function getAdminPassword(): string {
  const fromEnv = process.env.ARA_COORDINATOR_PASSWORD;
  if (fromEnv && fromEnv.length > 0) return fromEnv;
  const existing = getMeta("admin_password");
  if (existing) return existing;
  const generated = randomBytes(15).toString("base64url");
  setMeta("admin_password", generated);
  // Log it exactly once (when we generate it), so an operator can read it from stdout.
  console.log(
    `\n[ara-coordinator] No ARA_COORDINATOR_PASSWORD set — generated an admin password:\n` +
      `[ara-coordinator]   ${generated}\n` +
      `[ara-coordinator] Set ARA_COORDINATOR_PASSWORD to override. This was logged once.\n`,
  );
  return generated;
}
