// SPDX-License-Identifier: Apache-2.0
// Unit tests for the push (phone-home) lib layer: db CRUD, node-auth token verification, the
// enrollment lifecycle, and the work long-poll. Runs against an in-memory SQLite (see db.smoke).
import { describe, it, expect, beforeAll, vi } from "vitest";

// db.ts reads ARA_COORDINATOR_DB at module load — set it BEFORE the dynamic imports in beforeAll.
process.env.ARA_COORDINATOR_DB = ":memory:";

type DB = typeof import("@/lib/db");
type Auth = typeof import("@/lib/node-auth");
type Enroll = typeof import("@/lib/enrollment");
type Work = typeof import("@/lib/work");

let db: DB, auth: Auth, enroll: Enroll, work: Work;

const VALID_ENV = {
  platform: "linux",
  accel: "vulkan",
  containerized: false,
  virtualization_layer: null,
  wall_source: "physical",
};

function selfDesc(machineKey: string) {
  return {
    machine_key: machineKey,
    identity: { hostname: machineKey, os: "linux", arch: "x86_64" },
    capabilities: [{ kind: "serve_model", id: "qwen", engine: "vulkan", evidence: "characterized" }],
    environment: VALID_ENV,
  };
}

beforeAll(async () => {
  process.env.ARA_COORDINATOR_DB = ":memory:";
  db = await import("@/lib/db");
  auth = await import("@/lib/node-auth");
  enroll = await import("@/lib/enrollment");
  work = await import("@/lib/work");
});

describe("node-auth.hashToken", () => {
  it("is deterministic sha256 hex (64 chars) and differs per input", () => {
    expect(auth.hashToken("abc")).toBe(auth.hashToken("abc"));
    expect(auth.hashToken("abc")).toMatch(/^[0-9a-f]{64}$/);
    expect(auth.hashToken("abc")).not.toBe(auth.hashToken("abd"));
  });
});

describe("enrollment tokens", () => {
  it("verifies a freshly issued token and rejects garbage", () => {
    const { token } = enroll.issueEnrollmentToken();
    expect(auth.verifyEnrollmentToken(token)).toBeTruthy();
    expect(auth.verifyEnrollmentToken("not-a-real-token")).toBeNull();
    expect(auth.verifyEnrollmentToken("")).toBeNull();
    expect(auth.verifyEnrollmentToken(null)).toBeNull();
  });

  it("is single-use for enroll but still authorizes the poll (allowUsed)", () => {
    const { token } = enroll.issueEnrollmentToken();
    const out = enroll.enroll(token, selfDesc("box-a"));
    expect(out).toEqual({ enrollment_id: expect.stringMatching(/^enr_/), status: "pending" });

    // consumed for enroll…
    expect(auth.verifyEnrollmentToken(token)).toBeNull();
    expect(enroll.enroll(token, selfDesc("box-a2"))).toBeNull();
    // …but a poll may still present it.
    expect(auth.verifyEnrollmentToken(token, { allowUsed: true })).toBeTruthy();
  });

  it("stores only the hash, never the plaintext", () => {
    const { token } = enroll.issueEnrollmentToken();
    const row = db.getEnrollmentTokenByHash(auth.hashToken(token));
    expect(row).toBeTruthy();
    expect(row!.token_hash).toBe(auth.hashToken(token));
    expect(row!.token_hash).not.toBe(token);
  });
});

describe("approval + session token delivery", () => {
  it("pending → approve → active token delivered ONCE, then consumed", () => {
    const { token } = enroll.issueEnrollmentToken();
    const { enrollment_id } = enroll.enroll(token, selfDesc("box-b"))!;

    // pending
    expect(enroll.pollApproval(enrollment_id, token)).toEqual({ kind: "pending" });

    const agent = db.getAgentByEnrollmentId(enrollment_id)!;
    enroll.approveAgent(agent.id);

    // first poll delivers the session token
    const first = enroll.pollApproval(enrollment_id, token);
    expect(first.kind).toBe("active");
    const sessionToken = (first as { kind: "active"; session_token: string }).session_token;
    expect(sessionToken.length).toBeGreaterThan(10);

    // the plaintext is now gone from the DB; only the hash remains
    const after = db.getAgentByEnrollmentId(enrollment_id)!;
    expect(after.pending_session_token).toBeNull();
    expect(after.session_token_hash).toBe(auth.hashToken(sessionToken));

    // second poll: consumed
    expect(enroll.pollApproval(enrollment_id, token)).toEqual({ kind: "consumed" });

    // the session token authenticates its agent; a bad one does not
    expect(auth.verifySessionToken(sessionToken)!.id).toBe(agent.id);
    expect(auth.verifySessionToken("wrong")).toBeNull();
  });

  it("unknown enrollment id → not_found; wrong token → unauthorized; denied → denied", () => {
    const { token } = enroll.issueEnrollmentToken();
    const { enrollment_id } = enroll.enroll(token, selfDesc("box-c"))!;
    expect(enroll.pollApproval("enr_nope", token).kind).toBe("not_found");
    expect(enroll.pollApproval(enrollment_id, "bad").kind).toBe("unauthorized");

    const agent = db.getAgentByEnrollmentId(enrollment_id)!;
    enroll.denyAgent(agent.id);
    expect(enroll.pollApproval(enrollment_id, token).kind).toBe("denied");
    expect(auth.verifySessionToken(agent.session_token_hash ?? "x")).toBeNull(); // denied ≠ active
  });

  it("a DIFFERENT valid token cannot poll (or steal the session token of) another agent (IDOR guard)", () => {
    const victim = enroll.issueEnrollmentToken();
    const { enrollment_id } = enroll.enroll(victim.token, selfDesc("victim"))!;
    const agent = db.getAgentByEnrollmentId(enrollment_id)!;
    enroll.approveAgent(agent.id); // session token now waiting to be delivered ONCE

    // An attacker holds their own valid (even used) enrollment token and knows the enrollment_id.
    const attacker = enroll.issueEnrollmentToken();
    enroll.enroll(attacker.token, selfDesc("attacker")); // marks it used → still allowUsed for polling
    expect(enroll.pollApproval(enrollment_id, attacker.token).kind).toBe("unauthorized");

    // The bound (victim's own) token still works and gets the token exactly once.
    expect(enroll.pollApproval(enrollment_id, victim.token).kind).toBe("active");
  });

  it("listPending / listActive reflect status", () => {
    const before = enroll.listPending().length;
    const { token } = enroll.issueEnrollmentToken();
    const { enrollment_id } = enroll.enroll(token, selfDesc("box-d"))!;
    expect(enroll.listPending().length).toBe(before + 1);

    const agent = db.getAgentByEnrollmentId(enrollment_id)!;
    enroll.approveAgent(agent.id);
    expect(enroll.listPending().some((a) => a.id === agent.id)).toBe(false);
    expect(enroll.listActive().some((a) => a.id === agent.id)).toBe(true);
  });
});

describe("work queue", () => {
  async function activeAgent(name: string) {
    const { token } = enroll.issueEnrollmentToken();
    const { enrollment_id } = enroll.enroll(token, selfDesc(name))!;
    const agent = db.getAgentByEnrollmentId(enrollment_id)!;
    enroll.approveAgent(agent.id);
    return agent.id;
  }

  it("enqueue → nextForAgent returns the job once, then it is dispatched", async () => {
    const agentId = await activeAgent("box-w1");
    const jobId = work.enqueue(agentId, "run", { model: "qwen", prompt: "hi" });
    expect(jobId).toMatch(/^job_/);

    const job = await work.nextForAgent(agentId, 0);
    expect(job).toEqual({ id: jobId, kind: "run", args: { model: "qwen", prompt: "hi" } });

    // already dispatched → nothing left to hand out on an immediate re-poll
    expect(await work.nextForAgent(agentId, 0)).toBeNull();
    expect(db.getWorkById(jobId)!.status).toBe("dispatched");
  });

  it("claims a job exactly once under two back-to-back polls (atomic dispatch)", async () => {
    const agentId = await activeAgent("box-race");
    const jobId = work.enqueue(agentId, "run", { model: "qwen", prompt: "hi" });

    // Two polls fired back-to-back (no await between them) race for the single queued job. The
    // atomic queued→dispatched claim must hand it to exactly one; the other sees nothing → null.
    const [a, b] = await Promise.all([
      work.nextForAgent(agentId, 0),
      work.nextForAgent(agentId, 0),
    ]);

    const claimed = [a, b].filter((j) => j !== null);
    expect(claimed).toHaveLength(1);
    expect(claimed[0]!.id).toBe(jobId);
    expect(db.getWorkById(jobId)!.status).toBe("dispatched");
  });

  it("nextForAgent long-poll resolves null after the wait window (fake timers)", async () => {
    vi.useFakeTimers();
    try {
      const agentId = await activeAgent("box-w2");
      const p = work.nextForAgent(agentId, 1000);
      await vi.advanceTimersByTimeAsync(1000);
      expect(await p).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it("recordResult writes status + result; unknown job → false", () => {
    const agentIdP = activeAgent("box-w3");
    return agentIdP.then((agentId) => {
      const jobId = work.enqueue(agentId, "run", { model: "qwen" });
      expect(work.recordResult(jobId, { status: "done", result: { output: "ok" } })).toBe(true);
      const row = db.getWorkById(jobId)!;
      expect(row.status).toBe("done");
      expect(JSON.parse(row.result_json!)).toEqual({ output: "ok" });
      expect(work.recordResult("job_missing", { status: "failed", error: "x" })).toBe(false);
    });
  });
});
