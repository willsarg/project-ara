// SPDX-License-Identifier: Apache-2.0
// Smoke test proving the coordinator's server-only DB layer is testable under vitest:
// the `server-only` alias resolves, better-sqlite3 loads, and an in-memory registry opens.
import { describe, it, expect, beforeAll } from "vitest";

describe("coordinator db harness", () => {
  beforeAll(() => {
    // db.ts reads ARA_COORDINATOR_DB at module load, so set it BEFORE the dynamic import below.
    process.env.ARA_COORDINATOR_DB = ":memory:";
  });

  it("opens an in-memory registry and starts with no agents", async () => {
    const { listActive } = await import("@/lib/enrollment");
    const agents = listActive();
    expect(Array.isArray(agents)).toBe(true);
    expect(agents.length).toBe(0);
  });
});
