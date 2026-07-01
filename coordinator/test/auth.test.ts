// SPDX-License-Identifier: Apache-2.0
// The admin session cookie: sign/verify roundtrip, tamper/expiry rejection, and — the security
// fix — that with NO secret configured we fail closed (never sign with a default/forgeable key).
import { describe, it, expect, vi, afterEach } from "vitest";
import { createSession, verifySession } from "@/lib/auth";

afterEach(() => vi.unstubAllEnvs());

describe("admin session cookie", () => {
  it("signs and verifies a fresh session", async () => {
    vi.stubEnv("ARA_COORDINATOR_SECRET", "s3cret");
    const token = await createSession();
    expect(await verifySession(token)).toBe(true);
  });

  it("rejects a tampered or garbage token", async () => {
    vi.stubEnv("ARA_COORDINATOR_SECRET", "s3cret");
    const token = await createSession();
    expect(await verifySession(token + "x")).toBe(false);
    expect(await verifySession("not.a.token")).toBe(false);
    expect(await verifySession(undefined)).toBe(false);
  });

  it("a token signed with a DIFFERENT secret does not verify (forgery guard)", async () => {
    vi.stubEnv("ARA_COORDINATOR_SECRET", "secret-A");
    const token = await createSession();
    vi.stubEnv("ARA_COORDINATOR_SECRET", "secret-B");
    expect(await verifySession(token)).toBe(false);
  });

  it("derives the secret from ARA_COORDINATOR_PASSWORD when no explicit secret", async () => {
    vi.stubEnv("ARA_COORDINATOR_SECRET", "");
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "hunter2");
    const token = await createSession();
    expect(await verifySession(token)).toBe(true);
  });

  it("FAILS CLOSED with no secret: createSession throws, verifySession is false", async () => {
    vi.stubEnv("ARA_COORDINATOR_SECRET", "");
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "");
    await expect(createSession()).rejects.toThrow(/no session secret/i);
    expect(await verifySession("anything")).toBe(false); // never trusts a token without a secret
  });
});
