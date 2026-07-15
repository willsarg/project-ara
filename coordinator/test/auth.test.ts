// SPDX-License-Identifier: Apache-2.0
// The admin session cookie: sign/verify roundtrip, tamper/expiry rejection, and — the security
// fix — that with NO secret configured we fail closed (never sign with a default/forgeable key).
import { describe, it, expect, vi, afterEach } from "vitest";
import { createSession, verifySession, invalidateSessions } from "@/lib/auth";

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

  it("generates a durable secret when no env credential is configured", async () => {
    vi.stubEnv("ARA_COORDINATOR_SECRET", "");
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "");
    const token = await createSession();
    expect(await verifySession(token)).toBe(true);
    expect(await verifySession("anything")).toBe(false);
  });

  it("invalidateSessions() revokes every previously issued token (real logout, not just cookie-clear)", async () => {
    vi.stubEnv("ARA_COORDINATOR_SECRET", "s3cret");
    const before = await createSession();
    expect(await verifySession(before)).toBe(true);

    invalidateSessions();
    expect(await verifySession(before)).toBe(false); // a copied/stolen cookie can no longer be replayed

    const after = await createSession(); // freshly minted AFTER the invalidation still works
    expect(await verifySession(after)).toBe(true);
  });
});
