// SPDX-License-Identifier: Apache-2.0
// GET /api/health — public, unauthenticated liveness probe used by the Docker healthcheck.
import { describe, it, expect } from "vitest";
import { GET } from "@/app/api/health/route";

describe("GET /api/health", () => {
  it("reports ok with the service name", async () => {
    const res = GET();
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ service: "ara-coordinator", status: "ok" });
  });
});
