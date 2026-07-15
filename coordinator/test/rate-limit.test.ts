// SPDX-License-Identifier: Apache-2.0
// The tiny in-process fixed-window limiter (src/lib/rate-limit.ts) used to throttle login attempts
// and /api/enroll POSTs. Pure logic — no Next/DB involved — so this is exercised directly.
import { describe, it, expect, vi, afterEach } from "vitest";
import {
  MAX_RATE_LIMIT_WINDOWS,
  clientRateLimitKey,
  rateLimit,
} from "@/lib/rate-limit";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllEnvs();
});

describe("rateLimit (fixed window)", () => {
  it("allows up to `max` calls in the window, then limits the next one", () => {
    const key = "k-allow-then-limit";
    for (let i = 0; i < 3; i++) {
      expect(rateLimit(key, 3, 60_000).limited).toBe(false);
    }
    const res = rateLimit(key, 3, 60_000);
    expect(res.limited).toBe(true);
    expect(res.retryAfterS).toBeGreaterThan(0);
  });

  it("resets once the window elapses", () => {
    vi.useFakeTimers();
    const key = "k-reset";
    expect(rateLimit(key, 1, 1_000).limited).toBe(false);
    expect(rateLimit(key, 1, 1_000).limited).toBe(true); // over the cap, still in-window

    vi.advanceTimersByTime(1_001);
    expect(rateLimit("k-after-expiry-prunes-old", 1, 1_000).limited).toBe(false);
    expect(rateLimit(key, 1, 1_000).limited).toBe(false); // fresh window
  });

  it("tracks independent keys independently (per-IP isolation)", () => {
    expect(rateLimit("k-a", 1, 60_000).limited).toBe(false);
    expect(rateLimit("k-b", 1, 60_000).limited).toBe(false); // different key, unaffected by k-a
    expect(rateLimit("k-a", 1, 60_000).limited).toBe(true); // k-a's own cap is now exhausted
  });

  it("ignores caller-controlled forwarding headers unless a trusted proxy is configured", () => {
    const headers = new Headers({ "x-forwarded-for": "203.0.113.5" });
    expect(clientRateLimitKey(headers)).toBe("direct");
    vi.stubEnv("ARA_COORDINATOR_TRUST_PROXY", "1");
    expect(clientRateLimitKey(headers)).toBe("203.0.113.5");
    expect(clientRateLimitKey(new Headers({ "x-forwarded-for": "spoofed" }))).toBe("unknown");
    expect(clientRateLimitKey(new Headers())).toBe("unknown");
  });

  it("bounds live identity state and sends excess identities through one limited bucket", () => {
    let last = { limited: false, retryAfterS: 0 };
    for (let i = 0; i < MAX_RATE_LIMIT_WINDOWS + 10; i++) {
      last = rateLimit(`bounded-${i}`, 1, 60_000);
    }
    expect(last.limited).toBe(true);
  });
});
