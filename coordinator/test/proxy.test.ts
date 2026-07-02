// SPDX-License-Identifier: Apache-2.0
// The admin auth gate itself (src/proxy.ts) — layer-2 integration per the testing-architecture
// spec: an unauthenticated request is BLOCKED (redirected to /login), a valid session passes, and
// the matcher exempts exactly the intended paths. The session primitives (auth.test.ts) prove
// sign/verify; THIS file proves the gate uses them to actually gate requests.
import { describe, it, expect, vi, afterEach } from "vitest";
import { NextRequest } from "next/server";
import { proxy, config } from "@/proxy";
import { createSession, SESSION_COOKIE } from "@/lib/auth";

afterEach(() => vi.unstubAllEnvs());

/** Build a NextRequest for a path, optionally carrying a session cookie. */
function req(path: string, cookie?: string): NextRequest {
  const r = new NextRequest(`http://coordinator.test${path}`);
  if (cookie !== undefined) r.cookies.set(SESSION_COOKIE, cookie);
  return r;
}

describe("admin auth gate (middleware)", () => {
  it("redirects an unauthenticated request to /login and strips the query", async () => {
    vi.stubEnv("ARA_COORDINATOR_SECRET", "s3cret");
    const res = await proxy(req("/nodes?tab=pending"));
    expect(res.status).toBe(307);
    const dest = new URL(res.headers.get("location")!);
    expect(dest.pathname).toBe("/login");
    expect(dest.search).toBe(""); // no query leakage onto the login URL
  });

  it("passes a request carrying a valid session cookie", async () => {
    vi.stubEnv("ARA_COORDINATOR_SECRET", "s3cret");
    const res = await proxy(req("/nodes", await createSession()));
    expect(res.status).toBe(200); // NextResponse.next()
    expect(res.headers.get("location")).toBeNull();
  });

  it("blocks a tampered/garbage cookie exactly like a missing one", async () => {
    vi.stubEnv("ARA_COORDINATOR_SECRET", "s3cret");
    const forged = (await createSession()) + "x";
    for (const bad of [forged, "not.a.token", ""]) {
      const res = await proxy(req("/", bad));
      expect(res.status).toBe(307);
      expect(new URL(res.headers.get("location")!).pathname).toBe("/login");
    }
  });

  it("a cookie signed under a DIFFERENT secret is rejected (forgery through the gate)", async () => {
    vi.stubEnv("ARA_COORDINATOR_SECRET", "secret-A");
    const stolen = await createSession();
    vi.stubEnv("ARA_COORDINATOR_SECRET", "secret-B");
    const res = await proxy(req("/nodes", stolen));
    expect(res.status).toBe(307);
  });

  it("fails CLOSED when no secret is configured — requests are blocked, not admitted", async () => {
    vi.stubEnv("ARA_COORDINATOR_SECRET", "");
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "");
    const res = await proxy(req("/", "anything"));
    expect(res.status).toBe(307); // verifySession → false → redirect; never next()
  });
});

describe("matcher scope (config)", () => {
  // The matcher is enforced by Next's ROUTER, not by middleware() — middleware() never sees an
  // exempt path. So pin the matcher's behavior by exercising its regex against pathnames the way
  // path-to-regexp does (full match). Honest scope note: this proves the PATTERN's semantics;
  // that Next applies it is proven only by `next build` (syntax) + a running server (P2 e2e).
  const matches = (p: string) => new RegExp(`^${config.matcher[0]}$`).test(p);

  it("gates dashboard pages", () => {
    for (const page of ["/", "/nodes", "/nodes/abc123"]) {
      expect(matches(page), `${page} must be gated`).toBe(true);
    }
  });

  it("exempts exactly: /login, /api/*, Next internals, favicon", () => {
    for (const open of ["/login", "/login/", "/api/enroll", "/api/work", "/api/health",
                        "/_next/static/x.css", "/_next/image", "/favicon.ico"]) {
      expect(matches(open), `${open} must be exempt`).toBe(false);
    }
  });

  it("does NOT exempt lookalike paths (prefix confusion)", () => {
    // Paths that merely START with an exempt word must still be gated — a future /apikeys page
    // must not silently inherit /api's exemption and ship unauthenticated.
    for (const sneaky of ["/loginX", "/apikeys", "/api2"]) {
      expect(matches(sneaky), `${sneaky} must be gated`).toBe(true);
    }
  });
});
