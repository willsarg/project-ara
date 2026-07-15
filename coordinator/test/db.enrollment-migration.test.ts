// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Will Sarg
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import Database from "better-sqlite3";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const dir = mkdtempSync(path.join(tmpdir(), "ara-enrollment-migration-"));
const dbPath = path.join(dir, "coordinator.db");

beforeAll(async () => {
  const old = new Database(dbPath);
  old.exec(`
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE agents (
      id INTEGER PRIMARY KEY AUTOINCREMENT, machine_key TEXT NOT NULL,
      enrollment_id TEXT NOT NULL UNIQUE, status TEXT NOT NULL DEFAULT 'pending',
      session_token_hash TEXT, pending_session_token TEXT, identity_json TEXT, caps_json TEXT,
      environment_json TEXT, enrollment_token_id INTEGER, created_at TEXT NOT NULL DEFAULT (datetime('now')),
      last_seen TEXT
    );
    CREATE TABLE enrollment_tokens (
      id INTEGER PRIMARY KEY AUTOINCREMENT, token_hash TEXT NOT NULL UNIQUE,
      used INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE work (
      id TEXT PRIMARY KEY, agent_id INTEGER NOT NULL, kind TEXT NOT NULL, args_json TEXT,
      status TEXT NOT NULL DEFAULT 'queued', result_json TEXT, error TEXT, measurement_json TEXT,
      created_at TEXT NOT NULL DEFAULT (datetime('now')), dispatched_at TEXT, finished_at TEXT
    );
    INSERT INTO agents (machine_key, enrollment_id, enrollment_token_id)
    VALUES ('old-a', 'enr_old_a', 7), ('old-b', 'enr_old_b', 7);
    INSERT INTO work (id, agent_id, kind, args_json, status, result_json, dispatched_at)
    VALUES ('job_legacy', 2, 'run', '{"model":"qwen"}', 'done', '{"output":"kept"}',
            '2026-07-01 12:00:00');
  `);
  const {
    migrateEnrollmentBindings, migrateWorkOfferTimestamps, migrateWorkResultEnvironment,
  } = await import("@/lib/db");
  migrateEnrollmentBindings(old);
  migrateWorkResultEnvironment(old);
  migrateWorkOfferTimestamps(old);
  old.close();
});

afterAll(() => rmSync(dir, { recursive: true, force: true }));

describe("existing enrollment-token binding migration", () => {
  it("keeps the newest legacy binding and enforces one bound enrollment going forward", () => {
    const migrated = new Database(dbPath);
    const bindings = migrated
      .prepare("SELECT id, enrollment_token_id FROM agents ORDER BY id")
      .all() as { id: number; enrollment_token_id: number | null }[];
    expect(bindings).toEqual([
      { id: 1, enrollment_token_id: null },
      { id: 2, enrollment_token_id: 7 },
    ]);
    expect(() =>
      migrated
        .prepare("INSERT INTO agents (machine_key, enrollment_id, enrollment_token_id) VALUES (?, ?, ?)")
        .run("new", "enr_new", 7),
    ).toThrow(/unique/i);
    migrated.close();
  });

  it("adds nullable result environment storage without rebuilding legacy work rows", () => {
    const migrated = new Database(dbPath);
    const columns = migrated.prepare("PRAGMA table_info(work)").all() as { name: string }[];
    expect(columns.map((column) => column.name)).toContain("result_environment_json");
    expect(migrated.prepare(
      "SELECT id, status, result_json, result_environment_json FROM work WHERE id = 'job_legacy'",
    ).get()).toEqual({
      id: "job_legacy",
      status: "done",
      result_json: '{"output":"kept"}',
      result_environment_json: null,
    });
    migrated.close();
  });

  it("preserves legacy offer evidence without claiming an unobserved acknowledgement time", () => {
    const migrated = new Database(dbPath);
    const columns = migrated.prepare("PRAGMA table_info(work)").all() as { name: string }[];
    expect(columns.map((column) => column.name)).toContain("offered_at");
    expect(migrated.prepare(
      "SELECT offered_at, dispatched_at FROM work WHERE id = 'job_legacy'",
    ).get()).toEqual({ offered_at: "2026-07-01 12:00:00", dispatched_at: null });
    migrated.close();
  });
});
