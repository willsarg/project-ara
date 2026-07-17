// SPDX-License-Identifier: Apache-2.0
// Server actions (src/app/actions.ts): admin login/logout session-cookie lifecycle, plus the
// phone-home admin mutations (approve/deny/revoke/issue-token/submit-job). These run outside a
// request context under vitest, so next/headers, next/navigation, and next/cache are mocked — the
// real cookies()/redirect() implementations require Next's request-scoped runtime.
import { describe, it, expect, vi, beforeAll, afterEach } from "vitest";

process.env.ARA_COORDINATOR_DB = ":memory:";

const cookieStore = { set: vi.fn(), delete: vi.fn() };
// No X-Forwarded-For by default -> loginAction's clientKey() falls back to the shared "unknown"
// bucket. Individual rate-limit tests override this per-call with mockResolvedValueOnce to get
// their OWN bucket, so they can safely exhaust it without perturbing other tests in this file.
const headersMock = vi.fn(async () => new Headers());
vi.mock("next/headers", () => ({ cookies: vi.fn(async () => cookieStore), headers: () => headersMock() }));

const redirectMock = vi.fn((url: string) => {
  throw new Error(`NEXT_REDIRECT:${url}`);
});
vi.mock("next/navigation", () => ({ redirect: (url: string) => redirectMock(url) }));

const revalidatePathMock = vi.fn();
vi.mock("next/cache", () => ({ revalidatePath: revalidatePathMock }));

let actions: typeof import("@/app/actions");
let enroll: typeof import("@/lib/enrollment");
let db: typeof import("@/lib/db");
let work: typeof import("@/lib/work");

beforeAll(async () => {
  actions = await import("@/app/actions");
  enroll = await import("@/lib/enrollment");
  db = await import("@/lib/db");
  work = await import("@/lib/work");
});

afterEach(() => {
  cookieStore.set.mockClear();
  cookieStore.delete.mockClear();
  redirectMock.mockClear();
  revalidatePathMock.mockClear();
  headersMock.mockClear();
  vi.unstubAllEnvs();
});

const TARGET_AUTHORITY = `node-target:v1:${"a".repeat(64)}`;
const EXACT_OLLAMA_CAPABILITY = {
  kind: "serve_model", id: "qwen3:0.6b", engine: "ollama", evidence: "characterized",
  runtime: "ollama", backend: "apple",
  artifact_id: `ollama-manifest-sha256:${"b".repeat(64)}`,
  config_key: "cfg:v1:{}", safe_context: 2048, authority: TARGET_AUTHORITY,
};

async function pendingAgent(machineKey: string, capabilities: unknown[] = []) {
  const { token } = enroll.issueEnrollmentToken();
  const { enrollment_id } = enroll.enroll(token, {
    machine_key: machineKey, environment: {}, capabilities,
  })!;
  return db.getAgentByEnrollmentId(enrollment_id)!;
}

describe("loginAction — env ARA_COORDINATOR_PASSWORD (direct-compare) path", () => {
  it("correct password sets the session cookie and redirects to /", async () => {
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "hunter2");
    const form = new FormData();
    form.set("password", "hunter2");
    await expect(actions.loginAction(undefined, form)).rejects.toThrow("NEXT_REDIRECT:/");

    expect(cookieStore.set).toHaveBeenCalledTimes(1);
    const [name, token, opts] = cookieStore.set.mock.calls[0];
    expect(name).toBe("ara_coord_session");
    expect(typeof token).toBe("string");
    expect(opts).toMatchObject({ httpOnly: true, sameSite: "lax", path: "/" });
    expect(redirectMock).toHaveBeenCalledWith("/");
  });

  it("wrong password of the SAME length is rejected (equal-length compare branch)", async () => {
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "hunter2");
    const form = new FormData();
    form.set("password", "hunter1"); // same length, different bytes
    const result = await actions.loginAction(undefined, form);
    expect(result).toEqual({ error: "Incorrect password." });
    expect(cookieStore.set).not.toHaveBeenCalled();
  });

  it("wrong password of a DIFFERENT length is rejected (length-mismatch branch)", async () => {
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "hunter2");
    const form = new FormData();
    form.set("password", "nope");
    const result = await actions.loginAction(undefined, form);
    expect(result).toEqual({ error: "Incorrect password." });
  });
});

// Runs BEFORE the loginAction generated-password tests below so the admin password hash does not
// yet exist — ensureAdminPassword's "generate + log once" branch only fires on the FIRST call.
describe("loginAction — per-client rate limiting", () => {
  it("limits repeated attempts from the SAME client, and reports a wait time (not a crash)", async () => {
    vi.stubEnv("ARA_COORDINATOR_TRUST_PROXY", "1");
    headersMock.mockResolvedValue(new Headers({ "x-forwarded-for": "198.51.100.7" }));
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "hunter2");

    const form = new FormData();
    form.set("password", "wrong-every-time");

    let sawIncorrect = false;
    let sawRateLimited = false;
    for (let i = 0; i < 20 && !sawRateLimited; i++) {
      const result = await actions.loginAction(undefined, form);
      if (result.error === "Incorrect password.") sawIncorrect = true;
      if (/^Too many attempts\. Try again in \d+s\.$/.test(result.error ?? "")) sawRateLimited = true;
    }
    expect(sawIncorrect).toBe(true); // under-the-cap calls still ran the real password check
    expect(sawRateLimited).toBe(true); // the cap was hit within 20 attempts
  });

  it("a DIFFERENT client (different X-Forwarded-For) has its own, unexhausted bucket", async () => {
    vi.stubEnv("ARA_COORDINATOR_TRUST_PROXY", "1");
    headersMock.mockResolvedValue(new Headers({ "x-forwarded-for": "198.51.100.9" }));
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "hunter2");
    const form = new FormData();
    form.set("password", "still-wrong");
    for (let i = 0; i < 15; i++) await actions.loginAction(undefined, form); // exhaust THIS client's cap

    headersMock.mockResolvedValue(new Headers({ "x-forwarded-for": "198.51.100.10" }));
    const result = await actions.loginAction(undefined, form);
    expect(result).toEqual({ error: "Incorrect password." }); // not rate-limited — different key
  });
});

describe("db.ensureAdminPassword / verifyAdminPassword — generated-password lifecycle", () => {
  it("generates a password once (logged), is idempotent on re-run, verifies correctly", () => {
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "");
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});

    db.ensureAdminPassword();
    expect(logSpy).toHaveBeenCalledTimes(1);
    const message = logSpy.mock.calls[0][0] as string;
    const match = /generated an admin password:\n\[ara-coordinator\]\s+(\S+)/.exec(message);
    expect(match).toBeTruthy();
    const generated = match![1];

    logSpy.mockClear();
    db.ensureAdminPassword(); // hash already persisted -> no-op, no log
    expect(logSpy).not.toHaveBeenCalled();

    expect(db.verifyAdminPassword(generated)).toBe(true);
    // same length, different bytes -> exercises the equal-length-but-mismatched compare branch
    const flipped = generated.slice(0, -1) + (generated.at(-1) === "A" ? "B" : "A");
    expect(db.verifyAdminPassword(flipped)).toBe(false);
    // different length -> exercises the length-mismatch compare branch
    expect(db.verifyAdminPassword("short")).toBe(false);

    logSpy.mockRestore();
  });

  it("no-ops (does not generate/log) when ARA_COORDINATOR_PASSWORD IS set", () => {
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "explicit-pw");
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    db.ensureAdminPassword();
    expect(logSpy).not.toHaveBeenCalled();
    logSpy.mockRestore();
  });
});

describe("db.getAgentById on an unknown id", () => {
  it("returns null (?? null fallback) rather than undefined", () => {
    expect(db.getAgentById(999_999)).toBeNull();
  });
});

describe("loginAction — generated-password (hash) path, no env password set", () => {
  it("wrong password against the generated/hashed admin password is rejected", async () => {
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "");
    const form = new FormData();
    form.set("password", "definitely-wrong");
    const result = await actions.loginAction(undefined, form);
    expect(result).toEqual({ error: "Incorrect password." });
    expect(cookieStore.set).not.toHaveBeenCalled();
  });

  it("empty submitted password (form field absent) is rejected", async () => {
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "");
    const result = await actions.loginAction(undefined, new FormData());
    expect(result).toEqual({ error: "Incorrect password." });
  });
});

describe("logoutAction", () => {
  it("deletes the session cookie and redirects to /login", async () => {
    await expect(actions.logoutAction()).rejects.toThrow("NEXT_REDIRECT:/login");
    expect(cookieStore.delete).toHaveBeenCalledWith("ara_coord_session");
  });

  it("invalidates the session server-side — a copied cookie can't be replayed after logout", async () => {
    vi.stubEnv("ARA_COORDINATOR_SECRET", "s3cret-for-logout-test");
    const auth = await import("@/lib/auth");
    const token = await auth.createSession();
    expect(await auth.verifySession(token)).toBe(true);

    await expect(actions.logoutAction()).rejects.toThrow("NEXT_REDIRECT:/login");

    expect(await auth.verifySession(token)).toBe(false); // stateless JWT alone would still verify here
  });
});

describe("issueEnrollmentTokenAction", () => {
  it("mints a token, revalidates /nodes, and returns the plaintext", async () => {
    const result = await actions.issueEnrollmentTokenAction(undefined, new FormData());
    expect(result.token.length).toBeGreaterThan(10);
    expect(revalidatePathMock).toHaveBeenCalledWith("/nodes");
  });
});

describe("approveAgentAction", () => {
  it("activates a pending agent when id is present", async () => {
    const agent = await pendingAgent("box-approve");
    const form = new FormData();
    form.set("id", String(agent.id));
    await expect(actions.approveAgentAction(form)).rejects.toThrow("NEXT_REDIRECT:/nodes");
    expect(db.getAgentById(agent.id)!.status).toBe("active");
    expect(revalidatePathMock).toHaveBeenCalledWith("/nodes");
  });

  it("no-ops when id is missing/falsy", async () => {
    await expect(actions.approveAgentAction(new FormData())).rejects.toThrow("NEXT_REDIRECT:/nodes");
  });
});

describe("denyAgentAction", () => {
  it("denies a pending agent when id is present", async () => {
    const agent = await pendingAgent("box-deny");
    const form = new FormData();
    form.set("id", String(agent.id));
    await expect(actions.denyAgentAction(form)).rejects.toThrow("NEXT_REDIRECT:/nodes");
    expect(db.getAgentById(agent.id)!.status).toBe("denied");
  });

  it("no-ops when id is missing/falsy", async () => {
    await expect(actions.denyAgentAction(new FormData())).rejects.toThrow("NEXT_REDIRECT:/nodes");
  });

  it("rejects a stale pending-page denial after the node was approved", async () => {
    const agent = await pendingAgent("box-stale-deny");
    enroll.approveAgent(agent.id);
    const approved = db.getAgentById(agent.id)!;
    expect(approved.pending_session_token).toEqual(expect.any(String));

    const form = new FormData();
    form.set("id", String(agent.id));
    await expect(actions.denyAgentAction(form)).rejects.toThrow(
      "NEXT_REDIRECT:/nodes?agent=not-pending",
    );

    expect(db.getAgentById(agent.id)).toMatchObject({
      status: "active",
      session_token_hash: approved.session_token_hash,
      pending_session_token: approved.pending_session_token,
    });
  });
});

describe("revokeAgentAction", () => {
  it("revokes an active agent when id is present", async () => {
    const agent = await pendingAgent("box-revoke-action");
    enroll.approveAgent(agent.id);
    const form = new FormData();
    form.set("id", String(agent.id));
    await expect(actions.revokeAgentAction(form)).rejects.toThrow("NEXT_REDIRECT:/nodes");
    expect(db.getAgentById(agent.id)!.status).toBe("denied");
    expect(db.getAgentById(agent.id)!.session_token_hash).toBeNull();
  });

  it("no-ops when id is missing/falsy", async () => {
    await expect(actions.revokeAgentAction(new FormData())).rejects.toThrow("NEXT_REDIRECT:/nodes");
  });
});

describe("submitJobAction", () => {
  it("enqueues a job bound to an exact advertised node target", async () => {
    const agent = await pendingAgent("box-job", [EXACT_OLLAMA_CAPABILITY]);
    enroll.approveAgent(agent.id);
    const form = new FormData();
    form.set("agentId", String(agent.id));
    form.set("authority", TARGET_AUTHORITY);
    form.set("prompt", "hello");
    await expect(actions.submitJobAction(form)).rejects.toThrow(
      /^NEXT_REDIRECT:\/nodes\?job=queued&jobId=job_[0-9a-f]{24}$/,
    );

    const job = await work.nextForAgent(agent.id, 0);
    expect(job).toMatchObject({
      kind: "run",
      args: {
        model: "qwen3:0.6b", prompt: "hello", engine: "ollama",
        target_authority: TARGET_AUTHORITY,
      },
    });
  });

  it("rejects when authority is missing or blank", async () => {
    const agent = await pendingAgent("box-job-nomodel");
    const form = new FormData();
    form.set("agentId", String(agent.id));
    form.set("authority", "   "); // trims to empty -> falsy
    await expect(actions.submitJobAction(form)).rejects.toThrow(
      "NEXT_REDIRECT:/nodes?job=invalid",
    );
    expect(await work.nextForAgent(agent.id, 0)).toBeNull();
  });

  it("rejects when agentId is missing/falsy", async () => {
    const form = new FormData();
    form.set("authority", TARGET_AUTHORITY);
    form.set("prompt", "hello");
    await expect(actions.submitJobAction(form)).rejects.toThrow(
      "NEXT_REDIRECT:/nodes?job=invalid",
    );
  });

  it("rejects when the authority field is entirely absent (?? \"\" fallback)", async () => {
    const agent = await pendingAgent("box-job-nofield");
    const form = new FormData();
    form.set("agentId", String(agent.id)); // no "authority" key at all
    form.set("prompt", "hello");
    await expect(actions.submitJobAction(form)).rejects.toThrow(
      "NEXT_REDIRECT:/nodes?job=invalid",
    );
    expect(await work.nextForAgent(agent.id, 0)).toBeNull();
  });

  it("rejects a missing or blank prompt before enqueueing", async () => {
    const agent = await pendingAgent("box-job-noprompt", [EXACT_OLLAMA_CAPABILITY]);
    enroll.approveAgent(agent.id);
    for (const prompt of [undefined, "   "]) {
      const form = new FormData();
      form.set("agentId", String(agent.id));
      form.set("authority", TARGET_AUTHORITY);
      if (prompt !== undefined) form.set("prompt", prompt);
      await expect(actions.submitJobAction(form)).rejects.toThrow(
        "NEXT_REDIRECT:/nodes?job=invalid",
      );
    }
    expect(await work.nextForAgent(agent.id, 0)).toBeNull();
  });

  it("does not create work for pending, denied, or nonexistent agents", async () => {
    const pending = await pendingAgent("box-job-inactive");
    for (const id of [pending.id, 999_999]) {
      const form = new FormData();
      form.set("agentId", String(id));
      form.set("authority", TARGET_AUTHORITY);
      form.set("prompt", "hello");
      await expect(actions.submitJobAction(form)).rejects.toThrow(
        "NEXT_REDIRECT:/nodes?job=not-active",
      );
    }
    enroll.denyAgent(pending.id);
    const denied = new FormData();
    denied.set("agentId", String(pending.id));
    denied.set("authority", TARGET_AUTHORITY);
    denied.set("prompt", "hello");
    await expect(actions.submitJobAction(denied)).rejects.toThrow(
      "NEXT_REDIRECT:/nodes?job=not-active",
    );
  });

  it("does not hide unexpected enqueue failures", async () => {
    const active = await pendingAgent("box-job-unexpected-error", [EXACT_OLLAMA_CAPABILITY]);
    enroll.approveAgent(active.id);
    const broken = vi.spyOn(work, "enqueue").mockImplementationOnce(() => {
      throw new Error("database unavailable");
    });
    const form = new FormData();
    form.set("agentId", String(active.id));
    form.set("authority", TARGET_AUTHORITY);
    form.set("prompt", "hello");
    await expect(actions.submitJobAction(form)).rejects.toThrow("database unavailable");
    broken.mockRestore();
  });

  it("reports a node that becomes inactive during enqueue", async () => {
    const active = await pendingAgent("box-job-race", [EXACT_OLLAMA_CAPABILITY]);
    enroll.approveAgent(active.id);
    const raced = vi.spyOn(work, "enqueue").mockImplementationOnce(() => {
      throw new work.AgentNotActiveError();
    });
    const form = new FormData();
    form.set("agentId", String(active.id));
    form.set("authority", TARGET_AUTHORITY);
    form.set("prompt", "hello");
    await expect(actions.submitJobAction(form)).rejects.toThrow(
      "NEXT_REDIRECT:/nodes?job=not-active",
    );
    raced.mockRestore();
  });

  it("rejects an authority that the selected node did not advertise", async () => {
    const active = await pendingAgent("box-job-drift", [EXACT_OLLAMA_CAPABILITY]);
    enroll.approveAgent(active.id);
    const form = new FormData();
    form.set("agentId", String(active.id));
    form.set("authority", `node-target:v1:${"c".repeat(64)}`);
    form.set("prompt", "hello");
    await expect(actions.submitJobAction(form)).rejects.toThrow(
      "NEXT_REDIRECT:/nodes?job=invalid",
    );
    expect(await work.nextForAgent(active.id, 0)).toBeNull();
  });
});
