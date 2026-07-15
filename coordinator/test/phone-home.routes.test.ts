// SPDX-License-Identifier: Apache-2.0
// Route-handler tests for the push channel: drive the real Next route handlers end-to-end over an
// in-memory DB — enroll → approve → poll (token delivered once) → long-poll work → report result,
// plus the 401 (bad/missing token) and 204 (long-poll timeout) paths.
import { describe, it, expect, beforeAll, vi } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";

process.env.ARA_COORDINATOR_DB = ":memory:";

let enrollRoute: typeof import("@/app/api/enroll/route");
let pollRoute: typeof import("@/app/api/enroll/[id]/route");
let workRoute: typeof import("@/app/api/work/route");
let ackRoute: typeof import("@/app/api/work/[id]/ack/route");
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
const canonicalEnrollBody = JSON.parse(
  readFileSync(path.resolve(__dirname, "../../contracts/wire/fixtures/enroll.request.valid.json"), "utf8"),
) as Record<string, unknown>;

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
  ackRoute = await import("@/app/api/work/[id]/ack/route");
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

  it("returns the same enrollment handle when the used token retries after response loss", async () => {
    const { token } = enroll.issueEnrollmentToken();
    const request = () =>
      enrollRoute.POST(
        req("http://x/api/enroll", {
          method: "POST",
          bearer: token,
          body: JSON.stringify({ ...enrollBody, machine_key: "box-enroll-retry" }),
        }),
      );

    const first = await request();
    expect(first.status).toBe(201);
    const firstBody = await first.json();
    const retry = await request();
    expect(retry.status).toBe(201);
    expect(await retry.json()).toEqual(firstBody);
  });
});

describe("full enroll → approve → poll → work → result flow", () => {
  it("delivers the session token until auth acknowledgement and dispatches/records a job", async () => {
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

    // poll → active, then a response-loss re-poll returns the same session token
    const activeRes = await pollRoute.GET(
      req(`http://x/api/enroll/${enrollment_id}`, { bearer: token }),
      params(enrollment_id),
    );
    expect(activeRes.status).toBe(200);
    const active = await activeRes.json();
    expect(active.status).toBe("active");
    const sessionToken: string = active.session_token;
    expect(sessionToken.length).toBeGreaterThan(10);

    // re-poll before session auth proves receipt → same token
    const again = await pollRoute.GET(
      req(`http://x/api/enroll/${enrollment_id}`, { bearer: token }),
      params(enrollment_id),
    );
    expect(again.status).toBe(200);
    expect(await again.json()).toEqual(active);

    // work poll with bad session token → 401
    const badWork = await workRoute.GET(req("http://x/api/work?wait=0", { bearer: "nope" }));
    expect(badWork.status).toBe(401);

    // work poll, nothing queued → 204
    const empty = await workRoute.GET(req("http://x/api/work?wait=0", { bearer: sessionToken }));
    expect(empty.status).toBe(204);

    // Successful bearer auth acknowledges durable receipt, so enrollment polling is now consumed.
    const acknowledged = await pollRoute.GET(
      req(`http://x/api/enroll/${enrollment_id}`, { bearer: token }),
      params(enrollment_id),
    );
    expect(acknowledged.status).toBe(409);

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
    expect(db.getWorkById(jobId)!.status).toBe("offered");
    const ack = await ackRoute.POST(
      req(`http://x/api/work/${jobId}/ack`, { method: "POST", bearer: sessionToken }),
      params(jobId),
    );
    expect(ack.status).toBe(200);
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
    expect(row.result_environment_json).toBe(JSON.stringify(ENV));

    // Result and acknowledgement response loss are both idempotent/terminal-safe.
    const repeatedResult = await resultRoute.POST(
      req(`http://x/api/work/${jobId}/result`, {
        method: "POST", bearer: sessionToken,
        body: JSON.stringify({ status: "done", result: { different: true }, environment: ENV }),
      }),
      params(jobId),
    );
    expect(repeatedResult.status).toBe(200);
    expect(JSON.parse(db.getWorkById(jobId)!.result_json!)).toEqual({ output: "Paris." });
    expect(db.getWorkById(jobId)!.result_environment_json).toBe(JSON.stringify(ENV));
    const lateAck = await ackRoute.POST(
      req(`http://x/api/work/${jobId}/ack`, { method: "POST", bearer: sessionToken }),
      params(jobId),
    );
    expect(lateAck.status).toBe(409);
  });
});

describe("POST /api/work/[id]/ack", () => {
  it("authenticates, hides foreign/unknown jobs, and acknowledges an offer idempotently", async () => {
    const owner = await activate("box-ack-route-owner");
    const other = await activate("box-ack-route-other");
    const { enqueue } = await import("@/lib/work");
    const jobId = enqueue(owner.agentId, "run", {});

    expect((await ackRoute.POST(
      req(`http://x/api/work/${jobId}/ack`, { method: "POST" }), params(jobId))).status).toBe(401);
    expect((await ackRoute.POST(
      req("http://x/api/work/missing/ack", { method: "POST", bearer: owner.sessionToken }),
      params("missing"))).status).toBe(404);
    expect((await ackRoute.POST(
      req(`http://x/api/work/${jobId}/ack`, { method: "POST", bearer: other.sessionToken }),
      params(jobId))).status).toBe(404);
    expect((await ackRoute.POST(
      req(`http://x/api/work/${jobId}/ack`, { method: "POST", bearer: owner.sessionToken }),
      params(jobId))).status).toBe(409); // queued, not yet offered

    expect((await workRoute.GET(
      req("http://x/api/work?wait=0", { bearer: owner.sessionToken }))).status).toBe(200);
    expect((await ackRoute.POST(
      req(`http://x/api/work/${jobId}/ack`, { method: "POST", bearer: owner.sessionToken }),
      params(jobId))).status).toBe(200);
    expect((await ackRoute.POST(
      req(`http://x/api/work/${jobId}/ack`, { method: "POST", bearer: owner.sessionToken }),
      params(jobId))).status).toBe(200);
  });
});

// Enroll → approve → deliver session token; returns the agent id and its live session token.
async function activate(machineKey: string): Promise<{ agentId: number; sessionToken: string }> {
  const { token } = enroll.issueEnrollmentToken();
  const enrRes = await enrollRoute.POST(
    req("http://x/api/enroll", {
      method: "POST",
      bearer: token,
      body: JSON.stringify({ ...enrollBody, machine_key: machineKey }),
    }),
  );
  const { enrollment_id } = await enrRes.json();
  const agent = db.getAgentByEnrollmentId(enrollment_id)!;
  enroll.approveAgent(agent.id);
  const sessionToken = (
    enroll.pollApproval(enrollment_id, token) as { kind: "active"; session_token: string }
  ).session_token;
  return { agentId: agent.id, sessionToken };
}

describe("POST /api/enroll boundary validation", () => {
  it("400 when machine_key is missing/blank", async () => {
    const { token } = enroll.issueEnrollmentToken();
    const missing = await enrollRoute.POST(
      req("http://x/api/enroll", {
        method: "POST",
        bearer: token,
        body: JSON.stringify({ identity: {}, environment: ENV }),
      }),
    );
    expect(missing.status).toBe(400);

    const { token: t2 } = enroll.issueEnrollmentToken();
    const blank = await enrollRoute.POST(
      req("http://x/api/enroll", {
        method: "POST",
        bearer: t2,
        body: JSON.stringify({ machine_key: "", environment: ENV }),
      }),
    );
    expect(blank.status).toBe(400);
  });

  it("400 when environment is missing or not an object", async () => {
    const { token } = enroll.issueEnrollmentToken();
    const noEnv = await enrollRoute.POST(
      req("http://x/api/enroll", {
        method: "POST",
        bearer: token,
        body: JSON.stringify({ machine_key: "box-x" }),
      }),
    );
    expect(noEnv.status).toBe(400);

    const { token: t2 } = enroll.issueEnrollmentToken();
    const badEnv = await enrollRoute.POST(
      req("http://x/api/enroll", {
        method: "POST",
        bearer: t2,
        body: JSON.stringify({ machine_key: "box-x", environment: "nope" }),
      }),
    );
    expect(badEnv.status).toBe(400);
  });

  it("400 on malformed JSON body (not 500)", async () => {
    const { token } = enroll.issueEnrollmentToken();
    const res = await enrollRoute.POST(
      req("http://x/api/enroll", { method: "POST", bearer: token, body: "{not json" }),
    );
    expect(res.status).toBe(400);
  });

  it("enforces the complete pinned enroll.request shape before consuming the token or mutating agents", async () => {
    const invalidBodies: unknown[] = [
      null,
      [],
      "not an object",
      { ...canonicalEnrollBody, machine_key: undefined },
      { ...canonicalEnrollBody, machine_key: "" },
      { ...canonicalEnrollBody, machine_key: 7 },
      { ...canonicalEnrollBody, identity: undefined },
      { ...canonicalEnrollBody, identity: null },
      { ...canonicalEnrollBody, identity: [] },
      { ...canonicalEnrollBody, identity: {} },
      { ...canonicalEnrollBody, identity: { hostname: "" } },
      { ...canonicalEnrollBody, identity: { hostname: 7 } },
      { ...canonicalEnrollBody, identity: { hostname: "box", os: 7 } },
      { ...canonicalEnrollBody, identity: { hostname: "box", arch: 7 } },
      { ...canonicalEnrollBody, profile_projection: [] },
      { ...canonicalEnrollBody, capabilities: undefined },
      { ...canonicalEnrollBody, capabilities: {} },
      { ...canonicalEnrollBody, capabilities: [null] },
      { ...canonicalEnrollBody, capabilities: [{}] },
      { ...canonicalEnrollBody, capabilities: [{ kind: "chat", id: "m", engine: "cpu", evidence: "none" }] },
      { ...canonicalEnrollBody, capabilities: [{ kind: "serve_model", id: "", engine: "cpu", evidence: "none" }] },
      { ...canonicalEnrollBody, capabilities: [{ kind: "serve_model", id: "m", engine: "", evidence: "none" }] },
      { ...canonicalEnrollBody, capabilities: [{ kind: "serve_model", id: "m", engine: "cpu", evidence: "guessed" }] },
      { ...canonicalEnrollBody, capabilities: [{ kind: "serve_model", id: "m", engine: "cpu", evidence: "none", extra: true }] },
      { ...canonicalEnrollBody, environment: undefined },
      { ...canonicalEnrollBody, environment: [] },
      { ...canonicalEnrollBody, environment: { ...ENV, platform: "plan9" } },
      { ...canonicalEnrollBody, environment: { ...ENV, accel: "tpu" } },
      { ...canonicalEnrollBody, environment: { ...ENV, containerized: "false" } },
      { ...canonicalEnrollBody, environment: { ...ENV, virtualization_layer: 7 } },
      { ...canonicalEnrollBody, environment: { ...ENV, wall_source: "guessed" } },
      { ...canonicalEnrollBody, environment: { ...ENV, surprise: true } },
      { ...canonicalEnrollBody, environment: { ...ENV, platform: undefined } },
      { ...canonicalEnrollBody, environment: { ...ENV, accel: undefined } },
      { ...canonicalEnrollBody, environment: { ...ENV, containerized: undefined } },
      { ...canonicalEnrollBody, environment: { ...ENV, wall_source: undefined } },
      { ...canonicalEnrollBody, sneaky_extra: true },
    ];
    const before = enroll.listAgentSummaries().length;
    const { token } = enroll.issueEnrollmentToken();

    for (const [index, body] of invalidBodies.entries()) {
      const res = await enrollRoute.POST(
        req("http://x/api/enroll", {
          method: "POST",
          bearer: token,
          headers: { "x-forwarded-for": `192.0.2.${index + 1}` },
          body: JSON.stringify(body),
        }),
      );
      expect(res.status, JSON.stringify(body)).toBe(400);
      expect(enroll.listAgentSummaries()).toHaveLength(before);
    }

    const valid = await enrollRoute.POST(
      req("http://x/api/enroll", {
        method: "POST",
        bearer: token,
        headers: { "x-forwarded-for": "192.0.2.254" },
        body: JSON.stringify(canonicalEnrollBody),
      }),
    );
    expect(valid.status).toBe(201);
    expect(enroll.listAgentSummaries()).toHaveLength(before + 1);
  });

  it("accepts contract-permitted open identity and profile projection objects", async () => {
    const { token } = enroll.issueEnrollmentToken();
    const res = await enrollRoute.POST(
      req("http://x/api/enroll", {
        method: "POST",
        bearer: token,
        body: JSON.stringify({
          ...canonicalEnrollBody,
          machine_key: "box-open-enroll-fields",
          identity: { hostname: "box", vendor_detail: { serial: 7 } },
          profile_projection: { nested: ["is", "allowed"] },
        }),
      }),
    );
    expect(res.status).toBe(201);
  });
});

describe("POST /api/work/[id]/result boundary + error paths", () => {
  it("lets one of two requests past the terminal gate without allowing the loser to overwrite", async () => {
    const { agentId, sessionToken } = await activate("box-result-race");
    const jobId = (await import("@/lib/work")).enqueue(agentId, "run", {});
    expect(db.claimNextWorkForAgent(agentId)!.id).toBe(jobId);
    expect(db.acknowledgeWorkForAgent(jobId, agentId)).toBe("ok");

    let arrivals = 0;
    let releaseJson!: () => void;
    let bothReading!: () => void;
    const release = new Promise<void>((resolve) => { releaseJson = resolve; });
    const bothAtAwait = new Promise<void>((resolve) => { bothReading = resolve; });
    const bodies = [
      { status: "done", result: { winner: "first" }, environment: ENV },
      {
        status: "failed", error: "second", environment: { ...ENV, accel: "cpu" },
      },
    ];
    const delayedRequest = (body: (typeof bodies)[number]) => {
      const request = req(`http://x/api/work/${jobId}/result`, {
        method: "POST", bearer: sessionToken,
      });
      Object.defineProperty(request, "json", { value: async () => {
        arrivals += 1;
        if (arrivals === 2) bothReading();
        await release;
        return body;
      } });
      return request;
    };

    const pending = bodies.map((body) =>
      resultRoute.POST(delayedRequest(body), params(jobId)));
    await bothAtAwait;
    releaseJson();
    const responses = await Promise.all(pending);
    const responseBodies = await Promise.all(responses.map((response) => response.json()));

    expect(responses.map((response) => response.status)).toEqual([200, 200]);
    const recordedIndex = responseBodies.findIndex((body) => !("already_recorded" in body));
    expect(recordedIndex).toBeGreaterThanOrEqual(0);
    expect(responseBodies.filter((body) => body.already_recorded === true)).toHaveLength(1);
    const row = db.getWorkById(jobId)!;
    const winningBody = bodies[recordedIndex];
    expect(row.status).toBe(winningBody.status);
    expect(row.result_json).toBe(
      "result" in winningBody ? JSON.stringify(winningBody.result) : null,
    );
    expect(row.error).toBe("error" in winningBody ? winningBody.error : null);
    expect(row.result_environment_json).toBe(JSON.stringify(winningBody.environment));
  });

  it("409s a valid result for work that was never offered and acknowledged", async () => {
    const { agentId, sessionToken } = await activate("box-result-unacked");
    const jobId = (await import("@/lib/work")).enqueue(agentId, "run", {});
    const res = await resultRoute.POST(
      req(`http://x/api/work/${jobId}/result`, {
        method: "POST", bearer: sessionToken,
        body: JSON.stringify({ status: "done", result: {}, environment: ENV }),
      }),
      params(jobId),
    );
    expect(res.status).toBe(409);
    expect(db.getWorkById(jobId)!.status).toBe("queued");
  });

  it("404 on an unknown job id (no leak of existence)", async () => {
    const { sessionToken } = await activate("box-r1");
    const res = await resultRoute.POST(
      req("http://x/api/work/job_does_not_exist/result", {
        method: "POST",
        bearer: sessionToken,
        body: JSON.stringify({ status: "done", environment: ENV }),
      }),
      params("job_does_not_exist"),
    );
    expect(res.status).toBe(404);
  });

  it("404s if an owned job disappears after the ownership read", async () => {
    const { agentId, sessionToken } = await activate("box-result-disappeared");
    const workLib = await import("@/lib/work");
    const jobId = workLib.enqueue(agentId, "run", {});
    const record = vi.spyOn(workLib, "recordResult").mockReturnValueOnce("unknown");
    try {
      const res = await resultRoute.POST(
        req(`http://x/api/work/${jobId}/result`, {
          method: "POST", bearer: sessionToken,
          body: JSON.stringify({ status: "done", result: {}, environment: ENV }),
        }),
        params(jobId),
      );
      expect(res.status).toBe(404);
    } finally {
      record.mockRestore();
    }
  });

  it("404 when reporting on ANOTHER agent's job (still 404, no leak)", async () => {
    const a = await activate("box-owner");
    const b = await activate("box-intruder");
    const { enqueue } = await import("@/lib/work");
    const jobId = enqueue(a.agentId, "run", { model: "qwen" });

    // b holds a valid session token but the job belongs to a → 404, not 403/200.
    const res = await resultRoute.POST(
      req(`http://x/api/work/${jobId}/result`, {
        method: "POST",
        bearer: b.sessionToken,
        body: JSON.stringify({ status: "done", environment: ENV }),
      }),
      params(jobId),
    );
    expect(res.status).toBe(404);
    expect(db.getWorkById(jobId)!.status).toBe("queued"); // untouched
  });

  it("400 on a bad status and on a missing environment", async () => {
    const { agentId, sessionToken } = await activate("box-r2");
    const { enqueue } = await import("@/lib/work");

    const jobId1 = enqueue(agentId, "run", { model: "qwen" });
    const badStatus = await resultRoute.POST(
      req(`http://x/api/work/${jobId1}/result`, {
        method: "POST",
        bearer: sessionToken,
        body: JSON.stringify({ status: "weird", environment: ENV }),
      }),
      params(jobId1),
    );
    expect(badStatus.status).toBe(400);

    const jobId2 = enqueue(agentId, "run", { model: "qwen" });
    const noEnv = await resultRoute.POST(
      req(`http://x/api/work/${jobId2}/result`, {
        method: "POST",
        bearer: sessionToken,
        body: JSON.stringify({ status: "done" }),
      }),
      params(jobId2),
    );
    expect(noEnv.status).toBe(400);
  });

  it("validates the complete pinned result.request shape before mutating the job", async () => {
    const { agentId, sessionToken } = await activate("box-result-schema");
    const { enqueue } = await import("@/lib/work");
    const jobId = enqueue(agentId, "benchmark", {});
    const invalidBodies: unknown[] = [
      null,
      [],
      { status: "done", environment: ENV },
      { status: "failed", environment: ENV },
      { status: "done", result: [], environment: ENV },
      { status: "failed", error: {}, environment: ENV },
      { status: "done", result: {}, measurement: [], environment: ENV },
      { status: "done", result: {}, environment: { ...ENV, platform: "plan9" } },
      { status: "done", result: {}, environment: { ...ENV, accel: "tpu" } },
      { status: "done", result: {}, environment: { ...ENV, containerized: "false" } },
      { status: "done", result: {}, environment: { ...ENV, virtualization_layer: 7 } },
      { status: "done", result: {}, environment: { ...ENV, wall_source: "guessed" } },
      { status: "done", result: {}, environment: { ...ENV, surprise: true } },
      { status: "done", result: {}, environment: { ...ENV, platform: undefined } },
      { status: "done", result: {}, environment: { ...ENV, accel: undefined } },
      { status: "done", result: {}, environment: { ...ENV, containerized: undefined } },
      { status: "done", result: {}, environment: { ...ENV, wall_source: undefined } },
      { status: "done", result: {}, environment: ENV, surprise: true },
    ];

    for (const body of invalidBodies) {
      const res = await resultRoute.POST(
        req(`http://x/api/work/${jobId}/result`, {
          method: "POST",
          bearer: sessionToken,
          body: JSON.stringify(body),
        }),
        params(jobId),
      );
      expect(res.status, JSON.stringify(body)).toBe(400);
      const row = db.getWorkById(jobId)!;
      expect(row.status).toBe("queued");
      expect(row.result_json).toBeNull();
      expect(row.error).toBeNull();
      expect(row.measurement_json).toBeNull();
    }
  });

  it("accepts nullable schema fields when the status-required property is present", async () => {
    const { agentId, sessionToken } = await activate("box-result-schema-nullable");
    const { enqueue } = await import("@/lib/work");
    const doneId = enqueue(agentId, "detect", {});
    const failedId = enqueue(agentId, "run", {});
    expect(db.claimNextWorkForAgent(agentId)!.id).toBe(doneId);
    expect(db.acknowledgeWorkForAgent(doneId, agentId)).toBe("ok");
    expect(db.claimNextWorkForAgent(agentId)!.id).toBe(failedId);
    expect(db.acknowledgeWorkForAgent(failedId, agentId)).toBe("ok");

    const done = await resultRoute.POST(
      req(`http://x/api/work/${doneId}/result`, {
        method: "POST",
        bearer: sessionToken,
        body: JSON.stringify({
          status: "done",
          result: null,
          measurement: null,
          environment: ENV,
        }),
      }),
      params(doneId),
    );
    const failed = await resultRoute.POST(
      req(`http://x/api/work/${failedId}/result`, {
        method: "POST",
        bearer: sessionToken,
        body: JSON.stringify({ status: "failed", error: null, environment: ENV }),
      }),
      params(failedId),
    );

    expect(done.status).toBe(200);
    expect(failed.status).toBe(200);
  });

  it("400 on malformed JSON body (not 500)", async () => {
    const { agentId, sessionToken } = await activate("box-r3");
    const { enqueue } = await import("@/lib/work");
    const jobId = enqueue(agentId, "run", { model: "qwen" });
    const res = await resultRoute.POST(
      req(`http://x/api/work/${jobId}/result`, {
        method: "POST",
        bearer: sessionToken,
        body: "{ broken",
      }),
      params(jobId),
    );
    expect(res.status).toBe(400);
  });
});

describe("GET /api/enroll/[id] on an unknown enrollment id", () => {
  it("404s with a valid (but unrelated) enrollment token", async () => {
    const { token } = enroll.issueEnrollmentToken();
    const res = await pollRoute.GET(
      req("http://x/api/enroll/enr_does_not_exist", { bearer: token }),
      params("enr_does_not_exist"),
    );
    expect(res.status).toBe(404);
    expect(await res.json()).toEqual({ error: "unknown enrollment" });
  });
});

describe("GET /api/enroll/[id] on a denied enrollment", () => {
  it("403s (denied)", async () => {
    const { token } = enroll.issueEnrollmentToken();
    const enrRes = await enrollRoute.POST(
      req("http://x/api/enroll", {
        method: "POST",
        bearer: token,
        body: JSON.stringify({ ...enrollBody, machine_key: "box-denied-route" }),
      }),
    );
    const { enrollment_id } = await enrRes.json();
    const agent = db.getAgentByEnrollmentId(enrollment_id)!;
    enroll.denyAgent(agent.id);

    const res = await pollRoute.GET(
      req(`http://x/api/enroll/${enrollment_id}`, { bearer: token }),
      params(enrollment_id),
    );
    expect(res.status).toBe(403);
    expect(await res.json()).toEqual({ error: "enrollment denied" });
  });
});

describe("GET /api/work with a non-numeric or absent wait param", () => {
  it("treats a NaN wait as 0 (immediate 204 when nothing queued)", async () => {
    const { sessionToken } = await activate("box-wait-nan");
    const res = await workRoute.GET(req("http://x/api/work?wait=not-a-number", { bearer: sessionToken }));
    expect(res.status).toBe(204);
  });

  it("treats a fully absent wait param as 0 (?? \"0\" fallback)", async () => {
    const { sessionToken } = await activate("box-wait-absent");
    const res = await workRoute.GET(req("http://x/api/work", { bearer: sessionToken }));
    expect(res.status).toBe(204);
  });
});

describe("GET /api/work with no Authorization header at all", () => {
  it("401s (verifySessionToken(null) short-circuit)", async () => {
    const res = await workRoute.GET(req("http://x/api/work?wait=0"));
    expect(res.status).toBe(401);
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
