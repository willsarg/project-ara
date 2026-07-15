// SPDX-License-Identifier: Apache-2.0
// GET /api/health — public, unauthenticated liveness probe used by the Docker healthcheck.
import { describe, it, expect, vi, beforeAll } from "vitest";

const checkDatabaseReady = vi.fn();
vi.mock("@/lib/db", () => ({ checkDatabaseReady }));

let GET: typeof import("@/app/api/health/route").GET;

beforeAll(async () => {
  ({ GET } = await import("@/app/api/health/route"));
});

describe("GET /api/health", () => {
  it("reports ok with the service name", async () => {
    const res = GET();
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ service: "ara-coordinator", status: "ok" });
  });

  it("reports unavailable when the registry cannot open, migrate, or read", async () => {
    checkDatabaseReady.mockImplementationOnce(() => { throw new Error("database is read-only"); });
    const res = GET();
    expect(res.status).toBe(503);
    expect(await res.json()).toEqual({ service: "ara-coordinator", status: "unavailable" });
  });
});
