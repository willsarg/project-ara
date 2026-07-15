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
    capabilities: [{ kind: "serve_model", id: "qwen", engine: "mlx", evidence: "characterized" }],
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

  it("retries enrollment idempotently with the same used token", () => {
    const { token } = enroll.issueEnrollmentToken();
    const out = enroll.enroll(token, selfDesc("box-a"));
    expect(out).toEqual({ enrollment_id: expect.stringMatching(/^enr_/), status: "pending" });

    // The token is consumed for a different enrollment, but a response-loss retry resolves to the
    // original handle instead of rejecting or creating another agent.
    expect(auth.verifyEnrollmentToken(token)).toBeNull();
    expect(enroll.enroll(token, selfDesc("box-a"))).toEqual(out);
    expect(auth.verifyEnrollmentToken(token, { allowUsed: true })).toBeTruthy();
  });

  it("binds concurrent uses of one enrollment token to exactly one enrollment", async () => {
    const { token } = enroll.issueEnrollmentToken();
    const attempts = await Promise.all(
      Array.from({ length: 8 }, () =>
        Promise.resolve().then(() => enroll.enroll(token, selfDesc("box-token-race"))),
      ),
    );

    expect(new Set(attempts.map((attempt) => attempt?.enrollment_id)).size).toBe(1);
    const matching = db.listAgents().filter((agent) => agent.machine_key === "box-token-race");
    expect(matching).toHaveLength(1);
    expect(matching[0].enrollment_token_id).toBe(
      db.getEnrollmentTokenByHash(auth.hashToken(token))!.id,
    );
  });

  it("serializes concurrent fresh-token enrollment for one machine onto its stable row", async () => {
    const tokens = Array.from({ length: 8 }, () => enroll.issueEnrollmentToken().token);
    await Promise.all(
      tokens.map((token) =>
        Promise.resolve().then(() => enroll.enroll(token, selfDesc("box-machine-race"))),
      ),
    );

    const matching = db.listAgents().filter((agent) => agent.machine_key === "box-machine-race");
    expect(matching).toHaveLength(1);
    expect(matching[0].status).toBe("pending");
  });

  it("stores only the hash, never the plaintext", () => {
    const { token } = enroll.issueEnrollmentToken();
    const row = db.getEnrollmentTokenByHash(auth.hashToken(token));
    expect(row).toBeTruthy();
    expect(row!.token_hash).toBe(auth.hashToken(token));
    expect(row!.token_hash).not.toBe(token);
  });

  it("rejects a used token that was never bound to an enrollment", () => {
    expect(enroll.enroll("not-a-real-token", selfDesc("box-invalid-token"))).toBeNull();
    const { token } = enroll.issueEnrollmentToken();
    const row = db.getEnrollmentTokenByHash(auth.hashToken(token))!;
    db.markEnrollmentTokenUsed(row.id);
    expect(enroll.enroll(token, selfDesc("box-orphan-token"))).toBeNull();
  });

  it("returns null when the atomic DB enrollment is given an unknown token row", () => {
    expect(
      db.enrollAgentAtomically({
        token_id: -1,
        machine_key: "missing-token",
        enrollment_id: "enr_missing_token",
        identity_json: null,
        caps_json: null,
        environment_json: null,
      }),
    ).toBeNull();
  });
});

describe("approval + session token delivery", () => {
  it("repeats the approved token until successful session auth proves durable receipt", () => {
    const { token } = enroll.issueEnrollmentToken();
    const { enrollment_id } = enroll.enroll(token, selfDesc("box-b"))!;

    // pending
    expect(enroll.pollApproval(enrollment_id, token)).toEqual({ kind: "pending" });

    const agent = db.getAgentByEnrollmentId(enrollment_id)!;
    enroll.approveAgent(agent.id);

    // Polls repeat the same token while the node may still be recovering from response loss.
    const first = enroll.pollApproval(enrollment_id, token);
    expect(first.kind).toBe("active");
    const sessionToken = (first as { kind: "active"; session_token: string }).session_token;
    expect(sessionToken.length).toBeGreaterThan(10);
    expect(enroll.pollApproval(enrollment_id, token)).toEqual(first);

    // Successful session authentication is the receipt acknowledgement and clears plaintext.
    expect(db.getAgentByEnrollmentId(enrollment_id)!.pending_session_token).toBe(sessionToken);
    expect(auth.verifySessionToken(sessionToken)!.id).toBe(agent.id);
    const after = db.getAgentByEnrollmentId(enrollment_id)!;
    expect(after.pending_session_token).toBeNull();
    expect(after.session_token_hash).toBe(auth.hashToken(sessionToken));
    expect(enroll.pollApproval(enrollment_id, token)).toEqual({ kind: "consumed" });

    // The token remains valid after acknowledging receipt; a bad one does not authenticate.
    expect(auth.verifySessionToken(sessionToken)!.id).toBe(agent.id);
    expect(auth.verifySessionToken("wrong")).toBeNull();
  });

  it("re-enrolls a known nonempty machine on the same agent and invalidates its old session", () => {
    const firstEnrollment = enroll.issueEnrollmentToken();
    const first = enroll.enroll(firstEnrollment.token, selfDesc("box-rotate"))!;
    const original = db.getAgentByEnrollmentId(first.enrollment_id)!;
    enroll.approveAgent(original.id);
    const oldSession = (
      enroll.pollApproval(first.enrollment_id, firstEnrollment.token) as {
        kind: "active";
        session_token: string;
      }
    ).session_token;
    expect(auth.verifySessionToken(oldSession)!.id).toBe(original.id);
    const jobId = work.enqueue(original.id, "run", { model: "qwen" });

    const freshEnrollment = enroll.issueEnrollmentToken();
    const rotated = enroll.enroll(freshEnrollment.token, {
      ...selfDesc("box-rotate"),
      identity: { hostname: "box-rotate-new", os: "linux", arch: "x86_64" },
    })!;
    const reenrolled = db.getAgentByEnrollmentId(rotated.enrollment_id)!;

    expect(rotated.enrollment_id).not.toBe(first.enrollment_id);
    expect(reenrolled.id).toBe(original.id);
    expect(reenrolled.status).toBe("pending");
    expect(reenrolled.session_token_hash).toBeNull();
    expect(JSON.parse(reenrolled.identity_json!)).toMatchObject({ hostname: "box-rotate-new" });
    expect(auth.verifySessionToken(oldSession)).toBeNull();
    expect(db.getWorkById(jobId)!.agent_id).toBe(original.id);
    expect(enroll.pollApproval(rotated.enrollment_id, firstEnrollment.token)).toEqual({
      kind: "unauthorized",
    });

    enroll.approveAgent(reenrolled.id);
    const newSession = (
      enroll.pollApproval(rotated.enrollment_id, freshEnrollment.token) as {
        kind: "active";
        session_token: string;
      }
    ).session_token;
    expect(newSession).not.toBe(oldSession);
    expect(auth.verifySessionToken(newSession)!.id).toBe(original.id);
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
    enroll.approveAgent(agent.id); // session token now waiting for its bound enrollment to poll

    // An attacker holds their own valid (even used) enrollment token and knows the enrollment_id.
    const attacker = enroll.issueEnrollmentToken();
    enroll.enroll(attacker.token, selfDesc("attacker")); // marks it used → still allowUsed for polling
    expect(enroll.pollApproval(enrollment_id, attacker.token).kind).toBe("unauthorized");

    // The bound (victim's own) token still works and gets the pending session token.
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

    // deliver + capture the session token; it authenticates while active
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

  it("revoke clears an approved session plaintext that was never acknowledged", () => {
    const { token } = enroll.issueEnrollmentToken();
    const { enrollment_id } = enroll.enroll(token, selfDesc("box-revoke-before-auth"))!;
    const agent = db.getAgentByEnrollmentId(enrollment_id)!;
    enroll.approveAgent(agent.id);
    expect(db.getAgentById(agent.id)!.pending_session_token).not.toBeNull();

    enroll.revoke(agent.id);

    const after = db.getAgentById(agent.id)!;
    expect(after.session_token_hash).toBeNull();
    expect(after.pending_session_token).toBeNull();
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
    db.claimNextWorkForAgent(agentId);
    db.acknowledgeWorkForAgent(jobId, agentId);
    work.recordResult(jobId, agentId, { status: "failed", error: "boom", environment: VALID_ENV });
    expect(db.getWorkById(jobId)!.error).toBe("boom");
  });

  it("non-string, non-null error is stringified", async () => {
    const agentId = await activeAgentHelper("box-err-object");
    const jobId = work.enqueue(agentId, "run", { model: "qwen" });
    db.claimNextWorkForAgent(agentId);
    db.acknowledgeWorkForAgent(jobId, agentId);
    work.recordResult(jobId, agentId, {
      status: "failed", error: { code: 42 }, environment: VALID_ENV,
    });
    expect(db.getWorkById(jobId)!.error).toBe("[object Object]");
  });

  it("absent error/result/measurement all persist as null", async () => {
    const agentId = await activeAgentHelper("box-err-absent");
    const jobId = work.enqueue(agentId, "run", { model: "qwen" });
    db.claimNextWorkForAgent(agentId);
    db.acknowledgeWorkForAgent(jobId, agentId);
    work.recordResult(jobId, agentId, { status: "failed", environment: VALID_ENV });
    const row = db.getWorkById(jobId)!;
    expect(row.error).toBeNull();
    expect(row.result_json).toBeNull();
    expect(row.measurement_json).toBeNull();
    expect(row.result_environment_json).toBe(JSON.stringify(VALID_ENV));
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
    expect(row.recently_seen).toBe(false); // approval alone is not a heartbeat observation
    // token-free shape: no secret fields ever leak into the summary
    expect(row).not.toHaveProperty("session_token_hash");
    expect(row).not.toHaveProperty("pending_session_token");
    // newest-first ordering: this fresh agent is at the front
    expect(summaries[0].id).toBe(agent.id);
  });

  it("reports recent presence only for a fresh heartbeat from an approved node", () => {
    const now = Date.parse("2026-07-15T12:00:00Z");
    expect(enroll.wasAgentSeenRecently(null, now)).toBe(false);
    expect(enroll.wasAgentSeenRecently("not-a-time", now)).toBe(false);
    expect(enroll.wasAgentSeenRecently("2026-07-15 11:59:01", now)).toBe(true);
    expect(enroll.wasAgentSeenRecently("2026-07-15T11:58:59Z", now)).toBe(false);
    expect(enroll.wasAgentSeenRecently("2026-07-15T12:00:01+00:00", now)).toBe(false);
    const recentTimestamp = new Date(Date.now() - 1_000).toISOString()
      .replace("T", " ").replace(/\.\d{3}Z$/, "");
    const recent = enroll.summarizeAgent({
      id: 1, machine_key: "m", status: "active", last_seen: recentTimestamp,
      caps_json: null,
    } as never);
    expect(recent.recently_seen).toBe(true);
    expect(enroll.summarizeAgent({ ...recent, status: "denied" } as never).recently_seen).toBe(false);
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
      { kind: "serve_model", id: "qwen3:0.6b", engine: "mlx", evidence: "characterized" },
      { kind: "serve_model", id: "llama3:8b", engine: "cuda", evidence: "characterized" },
    ]);
    expect(enroll.summarizeAgent({ ...base, caps_json: caps } as never).serve_models).toEqual([
      { id: "qwen3:0.6b", engine: "mlx" },
      { id: "llama3:8b", engine: "cuda" },
    ]);
  });

  it("renders historical wmx/wcx caps canonically without rewriting stored caps_json", () => {
    const historical = selfDesc("box-historical");
    historical.capabilities = [
      { kind: "serve_model", id: "apple-model", engine: "wmx", evidence: "characterized" },
      { kind: "serve_model", id: "nvidia-model", engine: "wcx", evidence: "characterized" },
    ];
    const { token } = enroll.issueEnrollmentToken();
    const { enrollment_id } = enroll.enroll(token, historical)!;
    const stored = db.getAgentByEnrollmentId(enrollment_id)!;
    const originalCapsJson = stored.caps_json;

    expect(enroll.summarizeAgent(stored).serve_models).toEqual([
      { id: "apple-model", engine: "mlx" },
      { id: "nvidia-model", engine: "cuda" },
    ]);
    expect(db.getAgentByEnrollmentId(enrollment_id)!.caps_json).toBe(originalCapsJson);
    expect(originalCapsJson).toContain('"engine":"wmx"');
    expect(originalCapsJson).toContain('"engine":"wcx"');
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
  async function pendingAgent(name: string) {
    const { token } = enroll.issueEnrollmentToken();
    const { enrollment_id } = enroll.enroll(token, selfDesc(name))!;
    return db.getAgentByEnrollmentId(enrollment_id)!.id;
  }

  async function activeAgent(name: string) {
    const { token } = enroll.issueEnrollmentToken();
    const { enrollment_id } = enroll.enroll(token, selfDesc(name))!;
    const agent = db.getAgentByEnrollmentId(enrollment_id)!;
    enroll.approveAgent(agent.id);
    return agent.id;
  }

  it("offers once, then becomes dispatched only after durable acknowledgement", async () => {
    const agentId = await activeAgent("box-w1");
    const jobId = work.enqueue(agentId, "run", { model: "qwen", prompt: "hi" });
    expect(jobId).toMatch(/^job_/);

    const job = await work.nextForAgent(agentId, 0);
    expect(job).toEqual({ id: jobId, kind: "run", args: { model: "qwen", prompt: "hi" } });

    // A live unacknowledged offer is not duplicated by an immediate re-poll.
    expect(await work.nextForAgent(agentId, 0)).toBeNull();
    expect(db.getWorkById(jobId)!.status).toBe("offered");
    expect(work.acknowledge(jobId, agentId)).toBe("ok");
    expect(db.getWorkById(jobId)!.status).toBe("dispatched");
    expect(work.acknowledge(jobId, agentId)).toBe("ok"); // idempotent after response loss
  });

  it("accepts exactly the four coordinator job kinds", async () => {
    const agentId = await activeAgent("box-job-kinds");
    for (const kind of ["run", "characterize", "detect", "benchmark"]) {
      const jobId = work.enqueue(agentId, kind, {});
      expect(db.getWorkById(jobId)!.kind).toBe(kind);
    }
  });

  it("rejects an invalid job kind before insertion and at the DB boundary", async () => {
    const agentId = await activeAgent("box-invalid-kind");
    expect(() => work.enqueue(agentId, "shell", {})).toThrow(/invalid job kind/i);
    expect(() => db.insertWork("job_invalid_kind", agentId, "shell", "{}")).toThrow(
      /invalid job kind/i,
    );
    expect(db.getWorkById("job_invalid_kind")).toBeNull();
  });

  it("rejects queued work for missing, pending, and denied agents", async () => {
    const pendingId = await pendingAgent("box-work-pending");
    expect(() => work.enqueue(pendingId, "run", {})).toThrow(work.AgentNotActiveError);
    enroll.denyAgent(pendingId);
    expect(() => work.enqueue(pendingId, "run", {})).toThrow(work.AgentNotActiveError);
    expect(() => work.enqueue(999_999, "run", {})).toThrow(work.AgentNotActiveError);
  });

  it("offers a job exactly once under two back-to-back polls", async () => {
    const agentId = await activeAgent("box-race");
    const jobId = work.enqueue(agentId, "run", { model: "qwen", prompt: "hi" });

    // Two polls fired back-to-back (no await between them) race for the single queued job. The
    // atomic queued→offered claim must hand it to exactly one; the other sees nothing → null.
    const [a, b] = await Promise.all([
      work.nextForAgent(agentId, 0),
      work.nextForAgent(agentId, 0),
    ]);

    const claimed = [a, b].filter((j) => j !== null);
    expect(claimed).toHaveLength(1);
    expect(claimed[0]!.id).toBe(jobId);
    expect(db.getWorkById(jobId)!.status).toBe("offered");
  });

  it("preserves FIFO when multiple jobs share SQLite's one-second timestamp", async () => {
    const agentId = await activeAgent("box-fifo-tie");
    // Reverse lexical IDs prove the tie-breaker is insertion order, not the random public ID.
    db.insertWork("job_z_first", agentId, "detect", "{}");
    db.insertWork("job_a_second", agentId, "detect", "{}");
    expect(db.claimNextWorkForAgent(agentId)!.id).toBe("job_z_first");
    expect(db.acknowledgeWorkForAgent("job_z_first", agentId)).toBe("ok");
    expect(db.claimNextWorkForAgent(agentId)!.id).toBe("job_a_second");
  });

  it("reoffers an expired unacknowledged offer but never an acknowledged long job", async () => {
    const agentId = await activeAgent("box-offer-recovery");
    const jobId = work.enqueue(agentId, "benchmark", {});
    expect(await work.nextForAgent(agentId, 0)).toMatchObject({ id: jobId });
    expect(db.claimNextWorkForAgent(agentId, 0)).toMatchObject({ id: jobId });
    expect(work.acknowledge(jobId, agentId)).toBe("ok");
    expect(db.claimNextWorkForAgent(agentId, 0)).toBeNull();
  });

  it("acknowledgement hides foreign jobs and rejects jobs that were never offered", async () => {
    const owner = await activeAgent("box-ack-owner");
    const other = await activeAgent("box-ack-other");
    const jobId = work.enqueue(owner, "run", {});
    expect(work.acknowledge("job_missing", owner)).toBe("unknown");
    expect(work.acknowledge(jobId, other)).toBe("unknown");
    expect(work.acknowledge(jobId, owner)).toBe("conflict");
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

  it("atomically records only the first terminal result and preserves its exact environment", () => {
    const agentIdP = activeAgent("box-w3");
    return agentIdP.then((agentId) => {
      const jobId = work.enqueue(agentId, "run", { model: "qwen" });
      db.claimNextWorkForAgent(agentId);
      db.acknowledgeWorkForAgent(jobId, agentId);
      expect(work.recordResult(jobId, agentId, {
        status: "weird", result: { output: "bad" }, environment: VALID_ENV,
      })).toBe("conflict");
      expect(db.getWorkById(jobId)!.status).toBe("dispatched");
      expect(work.recordResult(jobId, agentId, {
        status: "done", result: { output: "ok" }, environment: VALID_ENV,
      })).toBe("recorded");
      expect(work.recordResult(jobId, agentId, {
        status: "failed", error: "late", environment: { ...VALID_ENV, accel: "cpu" },
      })).toBe("already_recorded");
      const row = db.getWorkById(jobId)!;
      expect(row.status).toBe("done");
      expect(JSON.parse(row.result_json!)).toEqual({ output: "ok" });
      expect(row.error).toBeNull();
      expect(row.result_environment_json).toBe(JSON.stringify(VALID_ENV));
      expect(work.recordResult("job_missing", agentId, {
        status: "failed", error: "x", environment: VALID_ENV,
      })).toBe("unknown");
    });
  });

  it("lists submitted work with its exact lifecycle outcome and execution environment", async () => {
    const agentId = await activeAgent("box-admin-work");
    const jobId = work.enqueue(agentId, "run", { model: "qwen-small", prompt: "hi" });

    expect(work.listRecentWork().find((job) => job.id === jobId)).toMatchObject({
      machine_key: "box-admin-work",
      kind: "run",
      model: "qwen-small",
      status: "queued",
      result_json: null,
      error: null,
      result_environment_json: null,
      dispatched_at: null,
      finished_at: null,
    });

    expect(await work.nextForAgent(agentId, 0)).toMatchObject({ id: jobId });
    expect(work.acknowledge(jobId, agentId)).toBe("ok");
    expect(work.recordResult(jobId, agentId, {
      status: "done",
      result: { output: "hello" },
      environment: VALID_ENV,
    })).toBe("recorded");

    expect(work.listRecentWork().find((job) => job.id === jobId)).toMatchObject({
      status: "done",
      result_json: JSON.stringify({ output: "hello" }),
      error: null,
      result_environment_json: JSON.stringify(VALID_ENV),
      dispatched_at: expect.any(String),
      finished_at: expect.any(String),
    });
  });

  it("keeps jobs visible when their optional model label cannot be decoded", async () => {
    const agentId = await activeAgent("box-admin-odd-args");
    const cases: Array<[string, string | null, string | null]> = [
      ["job_admin_no_args", null, null],
      ["job_admin_bad_json", "not-json", null],
      ["job_admin_null_json", "null", null],
      ["job_admin_scalar_json", "7", null],
      ["job_admin_no_model", "{}", null],
      ["job_admin_nonstring_model", '{"model":7}', null],
      ["job_admin_model", '{"model":"tiny"}', "tiny"],
    ];
    for (const [id, argsJson] of cases) db.insertWork(id, agentId, "run", argsJson);

    const listed = new Map(work.listRecentWork().map((job) => [job.id, job]));
    for (const [id, , model] of cases) {
      expect(listed.get(id)).toMatchObject({ machine_key: "box-admin-odd-args", model });
    }
  });
});
