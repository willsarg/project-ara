// SPDX-License-Identifier: Apache-2.0
// Route-handler tests for the push channel: drive the real Next route handlers end-to-end over an
// in-memory DB — enroll → approve → poll (token delivered once) → long-poll work → report result,
// plus the 401 (bad/missing token) and 204 (long-poll timeout) paths.
import { describe, it, expect, beforeAll, vi } from "vitest";

process.env.ARA_COORDINATOR_DB = ":memory:";

let enrollRoute: typeof import("@/app/api/enroll/route");
let pollRoute: typeof import("@/app/api/enroll/[id]/route");
let workRoute: typeof import("@/app/api/work/route");
let resultRoute: typeof import("@/app/api/work/[id]/result/route");
let db: typeof import("@/lib/db");
let enroll: typeof import("@/lib/enrollment");

const ENV = {
  platform: "linux",
  accel: "vulkan",
  containerized: false,
  virtualization_layer: null,
  wall_source: "physical",
};
const enrollBody = {
  machine_key: "rog-9f3a1c",
  identity: { hostname: "rog", os: "linux", arch: "x86_64" },
  capabilities: [{ kind: "serve_model", id: "qwen", engine: "vulkan", evidence: "characterized" }],
  environment: ENV,
};

const req = (url: string, init?: RequestInit & { bearer?: string }) => {
  const headers = new Headers(init?.headers);
  if (init?.bearer) headers.set("authorization", `Bearer ${init.bearer}`);
  return new Request(url, { ...init, headers });
};
const params = (id: string) => ({ params: Promise.resolve({ id }) });

beforeAll(async () => {
  process.env.ARA_COORDINATOR_DB = ":memory:";
  db = await import("@/lib/db");
  enroll = await import("@/lib/enrollment");
  enrollRoute = await import("@/app/api/enroll/route");
  pollRoute = await import("@/app/api/enroll/[id]/route");
  workRoute = await import("@/app/api/work/route");
  resultRoute = await import("@/app/api/work/[id]/result/route");
});

describe("POST /api/enroll", () => {
  it("401 without a token, 401 with a bad token", async () => {
    const noTok = await enrollRoute.POST(
      req("http://x/api/enroll", { method: "POST", body: JSON.stringify(enrollBody) }),
    );
    expect(noTok.status).toBe(401);

    const badTok = await enrollRoute.POST(
      req("http://x/api/enroll", { method: "POST", bearer: "nope", body: JSON.stringify(enrollBody) }),
    );
    expect(badTok.status).toBe(401);
  });

  it("creates a PENDING agent with a valid enrollment token", async () => {
    const { token } = enroll.issueEnrollmentToken();
    const res = await enrollRoute.POST(
      req("http://x/api/enroll", { method: "POST", bearer: token, body: JSON.stringify(enrollBody) }),
    );
    expect(res.status).toBe(201);
    const json = await res.json();
    expect(json).toEqual({ enrollment_id: expect.stringMatching(/^enr_/), status: "pending" });
    expect(db.getAgentByEnrollmentId(json.enrollment_id)!.status).toBe("pending");
  });
});

describe("full enroll → approve → poll → work → result flow", () => {
  it("delivers the session token once and dispatches/records a job", async () => {
    // enroll
    const { token } = enroll.issueEnrollmentToken();
    const enrRes = await enrollRoute.POST(
      req("http://x/api/enroll", { method: "POST", bearer: token, body: JSON.stringify(enrollBody) }),
    );
    const { enrollment_id } = await enrRes.json();

    // poll while pending
    const pendingRes = await pollRoute.GET(
      req(`http://x/api/enroll/${enrollment_id}`, { bearer: token }),
      params(enrollment_id),
    );
    expect(pendingRes.status).toBe(200);
    expect(await pendingRes.json()).toEqual({ status: "pending" });

    // poll with no/bad token → 401
    const unauth = await pollRoute.GET(
      req(`http://x/api/enroll/${enrollment_id}`),
      params(enrollment_id),
    );
    expect(unauth.status).toBe(401);

    // admin approves
    const agent = db.getAgentByEnrollmentId(enrollment_id)!;
    enroll.approveAgent(agent.id);

    // poll → active, session token delivered once
    const activeRes = await pollRoute.GET(
      req(`http://x/api/enroll/${enrollment_id}`, { bearer: token }),
      params(enrollment_id),
    );
    expect(activeRes.status).toBe(200);
    const active = await activeRes.json();
    expect(active.status).toBe("active");
    const sessionToken: string = active.session_token;
    expect(sessionToken.length).toBeGreaterThan(10);

    // re-poll → 409 (already delivered)
    const again = await pollRoute.GET(
      req(`http://x/api/enroll/${enrollment_id}`, { bearer: token }),
      params(enrollment_id),
    );
    expect(again.status).toBe(409);

    // work poll with bad session token → 401
    const badWork = await workRoute.GET(req("http://x/api/work?wait=0", { bearer: "nope" }));
    expect(badWork.status).toBe(401);

    // work poll, nothing queued → 204
    const empty = await workRoute.GET(req("http://x/api/work?wait=0", { bearer: sessionToken }));
    expect(empty.status).toBe(204);

    // enqueue a job, then the node long-poll picks it up → 200 with the wire job shape
    const jobId = (await import("@/lib/work")).enqueue(agent.id, "run", {
      model: "qwen",
      prompt: "hi",
    });
    const workRes = await workRoute.GET(req("http://x/api/work?wait=0", { bearer: sessionToken }));
    expect(workRes.status).toBe(200);
    expect(await workRes.json()).toEqual({
      job: { id: jobId, kind: "run", args: { model: "qwen", prompt: "hi" } },
    });
    expect(db.getWorkById(jobId)!.status).toBe("dispatched");

    // report a result → 200, recorded; bad session → 401; unknown/foreign job → 404
    const badResult = await resultRoute.POST(
      req(`http://x/api/work/${jobId}/result`, { method: "POST", bearer: "nope", body: "{}" }),
      params(jobId),
    );
    expect(badResult.status).toBe(401);

    const notFound = await resultRoute.POST(
      req(`http://x/api/work/job_missing/result`, {
        method: "POST",
        bearer: sessionToken,
        body: JSON.stringify({ status: "done", result: {}, environment: ENV }),
      }),
      params("job_missing"),
    );
    expect(notFound.status).toBe(404);

    const ok = await resultRoute.POST(
      req(`http://x/api/work/${jobId}/result`, {
        method: "POST",
        bearer: sessionToken,
        body: JSON.stringify({
          status: "done",
          result: { output: "Paris." },
          measurement: { peak_mem_gb: 3.2 },
          environment: ENV,
        }),
      }),
      params(jobId),
    );
    expect(ok.status).toBe(200);
    const row = db.getWorkById(jobId)!;
    expect(row.status).toBe("done");
    expect(JSON.parse(row.result_json!)).toEqual({ output: "Paris." });
  });
});

describe("GET /api/work long-poll timeout", () => {
  it("returns 204 after the wait window with no work (fake timers)", async () => {
    const { token } = enroll.issueEnrollmentToken();
    const enrRes = await enrollRoute.POST(
      req("http://x/api/enroll", { method: "POST", bearer: token, body: JSON.stringify(enrollBody) }),
    );
    const { enrollment_id } = await enrRes.json();
    const agent = db.getAgentByEnrollmentId(enrollment_id)!;
    enroll.approveAgent(agent.id);
    const sessionToken = (
      enroll.pollApproval(enrollment_id, token) as { kind: "active"; session_token: string }
    ).session_token;

    vi.useFakeTimers();
    try {
      const p = workRoute.GET(req("http://x/api/work?wait=2", { bearer: sessionToken }));
      await vi.advanceTimersByTimeAsync(2000);
      const res = await p;
      expect(res.status).toBe(204);
    } finally {
      vi.useRealTimers();
    }
  });
});
