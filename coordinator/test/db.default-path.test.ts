// SPDX-License-Identifier: Apache-2.0
// db.ts resolves its SQLite file from ARA_COORDINATOR_DB, falling back to ./data/coordinator.db
// (relative to cwd) when unset. Exercises that fallback in an isolated scratch cwd so the test
// never touches the real project's data/coordinator.db.
import { describe, it, expect } from "vitest";
import { mkdtempSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

describe("db.ts default path fallback (ARA_COORDINATOR_DB unset)", () => {
  it("opens ./data/coordinator.db relative to cwd when no env override is set", async () => {
    const scratch = mkdtempSync(path.join(tmpdir(), "ara-coordinator-db-default-"));
    const originalCwd = process.cwd();
    delete process.env.ARA_COORDINATOR_DB;
    process.chdir(scratch);
    try {
      const db = await import("@/lib/db");
      db.getAgentById(1); // any call triggers open() and the file-path fallback
      expect(existsSync(path.join(scratch, "data", "coordinator.db"))).toBe(true);
    } finally {
      process.chdir(originalCwd);
      rmSync(scratch, { recursive: true, force: true });
    }
  });
});
