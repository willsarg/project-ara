// SPDX-License-Identifier: Apache-2.0
// Defense-in-depth: POST /api/enroll checks the bearer token, then calls enroll() which re-checks
// it. If enroll() ever reports failure AFTER the route's own check passed (e.g. a token consumed in
// the gap between the two checks), the route must still answer 401 — never fall through to a 500 or
// a 201 with a null handle. enroll() itself is mocked here; its real behavior is covered elsewhere
// (test/phone-home.lib.test.ts).
import { describe, it, expect, vi, beforeAll } from "vitest";

process.env.ARA_COORDINATOR_DB = ":memory:";

vi.mock("@/lib/enrollment", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/enrollment")>();
  return { ...actual, enroll: vi.fn(() => null) };
});

let enrollRoute: typeof import("@/app/api/enroll/route");
let enrollLib: typeof import("@/lib/enrollment");

beforeAll(async () => {
  enrollLib = await import("@/lib/enrollment");
  enrollRoute = await import("@/app/api/enroll/route");
});

describe("POST /api/enroll — enroll() returns null after the route's own token check passes", () => {
  it("answers 401, not 201/500", async () => {
    const { token } = enrollLib.issueEnrollmentToken();
    const res = await enrollRoute.POST(
      new Request("http://x/api/enroll", {
        method: "POST",
        headers: { authorization: `Bearer ${token}` },
        body: JSON.stringify({ machine_key: "box", environment: {} }),
      }),
    );
    expect(res.status).toBe(401);
    expect(await res.json()).toEqual({ error: "unauthorized" });
  });
});
