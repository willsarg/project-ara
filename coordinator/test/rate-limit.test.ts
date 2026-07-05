// SPDX-License-Identifier: Apache-2.0
// The tiny in-process fixed-window limiter (src/lib/rate-limit.ts) used to throttle login attempts
// and /api/enroll POSTs. Pure logic — no Next/DB involved — so this is exercised directly.
import { describe, it, expect, vi, afterEach } from "vitest";
import { rateLimit } from "@/lib/rate-limit";

afterEach(() => vi.useRealTimers());

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
    expect(rateLimit(key, 1, 1_000).limited).toBe(false); // fresh window
  });

  it("tracks independent keys independently (per-IP isolation)", () => {
    expect(rateLimit("k-a", 1, 60_000).limited).toBe(false);
    expect(rateLimit("k-b", 1, 60_000).limited).toBe(false); // different key, unaffected by k-a
    expect(rateLimit("k-a", 1, 60_000).limited).toBe(true); // k-a's own cap is now exhausted
  });
});
