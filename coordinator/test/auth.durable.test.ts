// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Will Sarg
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import Database from "better-sqlite3";

let scratch: string;

beforeEach(() => {
  scratch = mkdtempSync(path.join(tmpdir(), "ara-auth-durable-"));
  process.env.ARA_COORDINATOR_DB = path.join(scratch, "coordinator.db");
  vi.stubEnv("ARA_COORDINATOR_SECRET", "");
  vi.stubEnv("ARA_COORDINATOR_PASSWORD", "");
  vi.resetModules();
});

afterEach(() => {
  vi.unstubAllEnvs();
  rmSync(scratch, { recursive: true, force: true });
});

describe("durable coordinator session state", () => {
  it("shares the generated signing secret and epoch across isolated module instances", async () => {
    const actionBundle = await import("@/lib/auth");
    const beforeLogout = await actionBundle.createSession();

    vi.resetModules();
    const proxyBundle = await import("@/lib/auth");
    expect(await proxyBundle.verifySession(beforeLogout)).toBe(true);

    proxyBundle.invalidateSessions();
    expect(await actionBundle.verifySession(beforeLogout)).toBe(false);

    const afterLogout = await actionBundle.createSession();
    expect(await proxyBundle.verifySession(afterLogout)).toBe(true);
  });

  it("preserves no-env sessions and logout revocation across a restart", async () => {
    const firstProcess = await import("@/lib/auth");
    const revoked = await firstProcess.createSession();
    firstProcess.invalidateSessions();

    vi.resetModules();
    const restartedProcess = await import("@/lib/auth");
    expect(await restartedProcess.verifySession(revoked)).toBe(false);
    const fresh = await restartedProcess.createSession();
    expect(await restartedProcess.verifySession(fresh)).toBe(true);
  });

  it("atomically advances the epoch from independent database connections", async () => {
    const first = await import("@/lib/db");
    vi.resetModules();
    const second = await import("@/lib/db");

    expect(first.getSessionEpoch()).toBe(0);
    expect(first.advanceSessionEpoch()).toBe(1);
    expect(second.advanceSessionEpoch()).toBe(2);
    expect(first.getSessionEpoch()).toBe(2);
  });

  it("fails closed on malformed, unsafe, or exhausted persisted epochs", async () => {
    const db = await import("@/lib/db");
    expect(db.getSessionEpoch()).toBe(0);
    const raw = new Database(process.env.ARA_COORDINATOR_DB!);
    const write = raw.prepare(
      "INSERT INTO meta (key, value) VALUES ('session_epoch', ?) " +
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
    );

    write.run("-1");
    expect(() => db.getSessionEpoch()).toThrow(/epoch is invalid/);
    write.run(String(Number.MAX_SAFE_INTEGER + 1));
    expect(() => db.getSessionEpoch()).toThrow(/safe integer range/);
    write.run(String(Number.MAX_SAFE_INTEGER));
    expect(() => db.advanceSessionEpoch()).toThrow(/epoch is exhausted/);
    raw.close();
  });

  it("fails closed if SQLite refuses to persist a generated signing secret", async () => {
    const db = await import("@/lib/db");
    expect(db.getSessionEpoch()).toBe(0); // initialize the schema before adding the fault trigger
    const raw = new Database(process.env.ARA_COORDINATOR_DB!);
    raw.exec(`
      CREATE TRIGGER reject_session_secret
      BEFORE INSERT ON meta WHEN NEW.key = 'session_secret'
      BEGIN SELECT RAISE(IGNORE); END;
    `);
    expect(() => db.ensureSessionSecret()).toThrow(/failed to persist/);
    raw.close();
  });

  it("initializes the generated admin password and signing secret without logging the secret", async () => {
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    const { registerNode } = await import("@/instrumentation-node");
    registerNode();

    expect(logSpy).toHaveBeenCalledTimes(1);
    const message = String(logSpy.mock.calls[0][0]);
    const match = /generated an admin password:\n\[ara-coordinator\]\s+(\S+)/.exec(message);
    expect(match).toBeTruthy();

    const db = await import("@/lib/db");
    expect(db.verifyAdminPassword(match![1])).toBe(true);
    expect(message).not.toContain(db.ensureSessionSecret());
    logSpy.mockRestore();
  });
});
