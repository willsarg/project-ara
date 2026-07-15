// SPDX-License-Identifier: Apache-2.0
// POST /api/enroll is rate-limited per-IP (src/lib/rate-limit.ts wired in at src/app/api/enroll/
// route.ts). Uses a dedicated X-Forwarded-For so this file's bursts can't spill into (or be masked
// by) the "unknown" bucket other enroll tests in this same worker share.
import { describe, it, expect, beforeAll } from "vitest";

process.env.ARA_COORDINATOR_DB = ":memory:";
process.env.ARA_COORDINATOR_TRUST_PROXY = "1";

let enrollRoute: typeof import("@/app/api/enroll/route");

beforeAll(async () => {
  enrollRoute = await import("@/app/api/enroll/route");
});

const post = (ip: string) =>
  enrollRoute.POST(
    new Request("http://x/api/enroll", {
      method: "POST",
      headers: { authorization: "Bearer nope", "x-forwarded-for": ip },
      body: JSON.stringify({ machine_key: "box", environment: {} }),
    }),
  );

describe("POST /api/enroll rate limiting", () => {
  it("401s under the cap, then 429s once the per-IP window is exceeded", async () => {
    const ip = "203.0.113.42";
    let sawUnauthorized = 0;
    let firstLimitedAt = -1;
    for (let i = 0; i < 40; i++) {
      const res = await post(ip);
      if (res.status === 401) sawUnauthorized++;
      if (res.status === 429 && firstLimitedAt === -1) firstLimitedAt = i;
    }
    expect(sawUnauthorized).toBeGreaterThan(0); // under-the-cap calls still run the real auth check
    expect(firstLimitedAt).toBeGreaterThan(-1); // the cap was hit within 40 calls
  });

  it("429 response carries a Retry-After header and a distinct IP is unaffected", async () => {
    const ip = "203.0.113.43";
    let res = await post(ip);
    for (let i = 0; i < 40 && res.status !== 429; i++) res = await post(ip);
    expect(res.status).toBe(429);
    expect(await res.json()).toEqual({ error: "too many requests" });
    expect(Number(res.headers.get("Retry-After"))).toBeGreaterThan(0);

    // A different IP has its own, unexhausted bucket.
    const other = await post("203.0.113.44");
    expect(other.status).toBe(401); // unauthorized (bad token), not 429 (not rate limited)
  });
});
