// SPDX-License-Identifier: Apache-2.0
// Defensive guard: verifyAdminPassword must fail SAFE (return false, never throw) if the persisted
// salt/hash pair is ever incomplete — e.g. a prior process crashed between the two setMeta() writes
// in ensureAdminPassword(). That's not reproducible against a healthy DB, so this file fakes
// better-sqlite3's meta table to silently drop the salt write and simulate exactly that corruption.
import { describe, it, expect, vi, beforeAll, afterEach } from "vitest";

const metaStore = new Map<string, string>();

vi.mock("better-sqlite3", () => {
  class FakeStatement {
    constructor(private sql: string) {}
    get(key?: string) {
      if (this.sql.includes("SELECT value FROM meta")) {
        const value = metaStore.get(key!);
        return value === undefined ? undefined : { value };
      }
      return undefined;
    }
    run(...args: unknown[]) {
      if (this.sql.includes("INSERT INTO meta")) {
        const [key, value] = args as [string, string];
        // Simulate a crash-between-writes: the salt write never lands, the hash write does.
        if (key !== "admin_pw_salt") metaStore.set(key, value);
      }
      return { changes: 1, lastInsertRowid: 0 };
    }
  }
  class FakeDatabase {
    pragma() {}
    exec() {}
    prepare(sql: string) {
      return new FakeStatement(sql);
    }
  }
  return { default: FakeDatabase };
});

process.env.ARA_COORDINATOR_DB = ":memory:";

let db: typeof import("@/lib/db");

beforeAll(async () => {
  db = await import("@/lib/db");
});

afterEach(() => vi.unstubAllEnvs());

describe("verifyAdminPassword with a corrupted meta table (hash present, salt missing)", () => {
  it("returns false instead of throwing", () => {
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "");
    expect(db.verifyAdminPassword("anything")).toBe(false);
  });
});
