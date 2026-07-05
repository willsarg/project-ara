// SPDX-License-Identifier: Apache-2.0
// A tiny in-process fixed-window rate limiter — no new dependency, hand-rolled deliberately (this
// is generic-enough infra that a mature library would be overkill for two call sites). State is an
// in-memory Map, so a process restart clears it; that's an accepted trade-off consistent with the
// rest of this coordinator's posture (e.g. the long-poll dispatcher is also purely in-process) — a
// single admin-facing Node process, not a distributed/serverless deployment.
//
// Fixed-window (not token-bucket/sliding-log): simpler, O(1) per check, and "good enough" to blunt
// brute-force/DoS attempts against login and /api/enroll — the two call sites that use this.
import "server-only";

interface Window {
  count: number;
  resetAt: number; // epoch ms when this key's window resets
}

const windows = new Map<string, Window>();

export interface RateLimitResult {
  /** True once this key has exceeded `max` calls within the current window. */
  limited: boolean;
  /** Seconds until the window resets; 0 when not limited. */
  retryAfterS: number;
}

/** Check-and-increment a fixed window for `key`. The first call for a fresh (or expired) window
 *  starts a new `windowMs`-long window; each call within it increments the count. Once the count
 *  exceeds `max`, every further call in that window reports `limited: true` until it resets. */
export function rateLimit(key: string, max: number, windowMs: number): RateLimitResult {
  const now = Date.now();
  const w = windows.get(key);

  if (!w || now >= w.resetAt) {
    windows.set(key, { count: 1, resetAt: now + windowMs });
    return { limited: false, retryAfterS: 0 };
  }

  w.count += 1;
  if (w.count > max) {
    return { limited: true, retryAfterS: Math.ceil((w.resetAt - now) / 1000) };
  }
  return { limited: false, retryAfterS: 0 };
}
