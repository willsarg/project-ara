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

  it("revoke: an active agent's session token no longer authenticates", () => {
    const { token } = enroll.issueEnrollmentToken();
    const { enrollment_id } = enroll.enroll(token, selfDesc("box-revoke"))!;
    const agent = db.getAgentByEnrollmentId(enrollment_id)!;
    enroll.approveAgent(agent.id);

    // deliver + capture the one-time session token; it authenticates while active
    const first = enroll.pollApproval(enrollment_id, token);
    const sessionToken = (first as { kind: "active"; session_token: string }).session_token;
    expect(auth.verifySessionToken(sessionToken)!.id).toBe(agent.id);

    // revoke → denied AND session_token_hash NULLed → the token stops authing, agent drops off active
    enroll.revoke(agent.id);
    const after = db.getAgentById(agent.id)!;
    expect(after.status).toBe("denied");
    expect(after.session_token_hash).toBeNull();
    expect(auth.verifySessionToken(sessionToken)).toBeNull();
    expect(enroll.listActive().some((a) => a.id === agent.id)).toBe(false);
  });
});

describe("enroll() with an absent/non-string self-description", () => {
  it("tolerates a bare self-description: empty machine_key, null identity/caps/environment JSON", () => {
    const { token } = enroll.issueEnrollmentToken();
    const out = enroll.enroll(token, {});
    expect(out).toEqual({ enrollment_id: expect.stringMatching(/^enr_/), status: "pending" });
    const agent = db.getAgentByEnrollmentId(out!.enrollment_id)!;
    expect(agent.machine_key).toBe("");
    expect(agent.identity_json).toBeNull();
    expect(agent.caps_json).toBeNull();
    expect(agent.environment_json).toBeNull();
  });

  it("rejects a non-string machine_key the same way (coerced to empty)", () => {
    const { token } = enroll.issueEnrollmentToken();
    const out = enroll.enroll(token, { machine_key: 12345 as unknown as string });
    expect(db.getAgentByEnrollmentId(out!.enrollment_id)!.machine_key).toBe("");
  });
});

describe("bearerToken parsing", () => {
  it("returns null for a missing header, a malformed scheme, and an empty-token 'Bearer'", () => {
    const noHeader = new Request("http://x", {});
    expect(auth.bearerToken(noHeader)).toBeNull();

    const wrongScheme = new Request("http://x", { headers: { authorization: "Basic dXNlcjpwYXNz" } });
    expect(auth.bearerToken(wrongScheme)).toBeNull();

    const noToken = new Request("http://x", { headers: { authorization: "Bearer" } });
    expect(auth.bearerToken(noToken)).toBeNull();
  });

  it("extracts and trims the token from a well-formed Bearer header", () => {
    const ok = new Request("http://x", { headers: { authorization: "Bearer   abc123  " } });
    expect(auth.bearerToken(ok)).toBe("abc123");
  });
});

describe("work.recordResult ternary branches (result/error/measurement presence + type)", () => {
  it("string error is stored verbatim", async () => {
    const agentId = await activeAgentHelper("box-err-string");
    const jobId = work.enqueue(agentId, "run", { model: "qwen" });
    work.recordResult(jobId, { status: "failed", error: "boom" });
    expect(db.getWorkById(jobId)!.error).toBe("boom");
  });

  it("non-string, non-null error is stringified", async () => {
    const agentId = await activeAgentHelper("box-err-object");
    const jobId = work.enqueue(agentId, "run", { model: "qwen" });
    work.recordResult(jobId, { status: "failed", error: { code: 42 } });
    expect(db.getWorkById(jobId)!.error).toBe("[object Object]");
  });

  it("absent error/result/measurement all persist as null", async () => {
    const agentId = await activeAgentHelper("box-err-absent");
    const jobId = work.enqueue(agentId, "run", { model: "qwen" });
    work.recordResult(jobId, { status: "failed" });
    const row = db.getWorkById(jobId)!;
    expect(row.error).toBeNull();
    expect(row.result_json).toBeNull();
    expect(row.measurement_json).toBeNull();
  });
});

async function activeAgentHelper(name: string) {
  const { token } = enroll.issueEnrollmentToken();
  const { enrollment_id } = enroll.enroll(token, selfDesc(name))!;
  const agent = db.getAgentByEnrollmentId(enrollment_id)!;
  enroll.approveAgent(agent.id);
  return agent.id;
}

describe("node-auth defensive hash-mismatch guards (simulated DB tampering)", () => {
  it("verifyEnrollmentToken rejects a row whose stored hash doesn't match the lookup hash", () => {
    const spy = vi.spyOn(db, "getEnrollmentTokenByHash").mockReturnValueOnce({
      id: 999,
      token_hash: "not-the-real-hash",
      used: 0,
      created_at: "now",
    });
    expect(auth.verifyEnrollmentToken("whatever-token")).toBeNull();
    spy.mockRestore();
  });

  it("verifySessionToken rejects an agent whose stored hash doesn't match the lookup hash", () => {
    const spy = vi.spyOn(db, "getAgentBySessionHash").mockReturnValueOnce({
      id: 999,
      machine_key: "tampered",
      enrollment_id: "enr_tampered",
      status: "active",
      session_token_hash: "not-the-real-hash",
      pending_session_token: null,
      identity_json: null,
      caps_json: null,
      environment_json: null,
      enrollment_token_id: null,
      created_at: "now",
      last_seen: null,
    });
    expect(auth.verifySessionToken("whatever-token")).toBeNull();
    spy.mockRestore();
  });
});

describe("dashboard agent-listing helper", () => {
  it("summarizes agents newest-first, token-free, with a capabilities count from caps_json", () => {
    const { token } = enroll.issueEnrollmentToken();
    const { enrollment_id } = enroll.enroll(token, selfDesc("box-dash"))!;
    const agent = db.getAgentByEnrollmentId(enrollment_id)!;
    enroll.approveAgent(agent.id);

    const summaries = enroll.listAgentSummaries();
    const row = summaries.find((s) => s.id === agent.id)!;
    expect(row).toBeTruthy();
    // selfDesc advertises exactly one capability
    expect(row.caps_count).toBe(1);
    expect(row.machine_key).toBe("box-dash");
    expect(row.status).toBe("active");
    // token-free shape: no secret fields ever leak into the summary
    expect(row).not.toHaveProperty("session_token_hash");
    expect(row).not.toHaveProperty("pending_session_token");
    // newest-first ordering: this fresh agent is at the front
    expect(summaries[0].id).toBe(agent.id);
  });

  it("summarizeAgent yields caps_count 0 for absent or malformed caps_json (never throws)", () => {
    const base = { id: 1, machine_key: "m", status: "pending", last_seen: null } as const;
    expect(enroll.summarizeAgent({ ...base, caps_json: null } as never).caps_count).toBe(0);
    expect(enroll.summarizeAgent({ ...base, caps_json: "not json" } as never).caps_count).toBe(0);
    expect(enroll.summarizeAgent({ ...base, caps_json: "{}" } as never).caps_count).toBe(0);
    expect(
      enroll.summarizeAgent({ ...base, caps_json: '[{"a":1},{"b":2}]' } as never).caps_count,
    ).toBe(2);
  });

  it("summarizeAgent extracts serve_model entries into serve_models: {id, engine}", () => {
    const base = { id: 1, machine_key: "m", status: "pending", last_seen: null } as const;
    const caps = JSON.stringify([
      { kind: "serve_model", id: "qwen3:0.6b", engine: "ollama", evidence: "characterized" },
      { kind: "serve_model", id: "llama3:8b", engine: "wcx", evidence: "characterized" },
    ]);
    expect(enroll.summarizeAgent({ ...base, caps_json: caps } as never).serve_models).toEqual([
      { id: "qwen3:0.6b", engine: "ollama" },
      { id: "llama3:8b", engine: "wcx" },
    ]);
  });

  it("summarizeAgent keeps only serve_model entries out of a mixed capabilities list", () => {
    const base = { id: 1, machine_key: "m", status: "pending", last_seen: null } as const;
    const caps = JSON.stringify([
      { kind: "serve_model", id: "qwen3:0.6b", engine: "ollama" },
      { kind: "quantize", id: "q4_k_m" },
    ]);
    expect(enroll.summarizeAgent({ ...base, caps_json: caps } as never).serve_models).toEqual([
      { id: "qwen3:0.6b", engine: "ollama" },
    ]);
  });

  it("summarizeAgent defaults engine to '?' when missing or non-string", () => {
    const base = { id: 1, machine_key: "m", status: "pending", last_seen: null } as const;
    const caps = JSON.stringify([
      { kind: "serve_model", id: "qwen3:0.6b" },
      { kind: "serve_model", id: "llama3:8b", engine: 42 },
    ]);
    expect(enroll.summarizeAgent({ ...base, caps_json: caps } as never).serve_models).toEqual([
      { id: "qwen3:0.6b", engine: "?" },
      { id: "llama3:8b", engine: "?" },
    ]);
  });

  it("summarizeAgent skips non-object entries and entries with a missing/non-string id", () => {
    const base = { id: 1, machine_key: "m", status: "pending", last_seen: null } as const;
    const caps = JSON.stringify([
      "not-an-object",
      42,
      null,
      { kind: "serve_model" },
      { kind: "serve_model", id: 123 },
      { kind: "serve_model", id: "ok-model", engine: "ollama" },
    ]);
    expect(enroll.summarizeAgent({ ...base, caps_json: caps } as never).serve_models).toEqual([
      { id: "ok-model", engine: "ollama" },
    ]);
  });

  it("summarizeAgent yields serve_models [] for absent or malformed caps_json (never throws)", () => {
    const base = { id: 1, machine_key: "m", status: "pending", last_seen: null } as const;
    expect(enroll.summarizeAgent({ ...base, caps_json: null } as never).serve_models).toEqual([]);
    expect(
      enroll.summarizeAgent({ ...base, caps_json: "not json" } as never).serve_models,
    ).toEqual([]);
    expect(enroll.summarizeAgent({ ...base, caps_json: "{}" } as never).serve_models).toEqual([]);
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

  it("enqueue tolerates a nullish args (?? {} fallback) — stored/dispatched as an empty object", async () => {
    const agentId = await activeAgent("box-w4");
    const jobId = work.enqueue(agentId, "run", null as unknown as Record<string, unknown>);
    const job = await work.nextForAgent(agentId, 0);
    expect(job).toEqual({ id: jobId, kind: "run", args: {} });
  });

  it("nextForAgent yields {} args for a row with no args_json (: {} fallback)", async () => {
    const agentId = await activeAgent("box-w5");
    const jobId = "job_no_args";
    db.insertWork(jobId, agentId, "run", null); // bypass enqueue() — args_json genuinely absent
    const job = await work.nextForAgent(agentId, 0);
    expect(job).toEqual({ id: jobId, kind: "run", args: {} });
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
