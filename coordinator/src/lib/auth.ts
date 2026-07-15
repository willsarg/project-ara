// SPDX-License-Identifier: Apache-2.0
// Session cookies — a small signed JWT (HS256). Next 16 runs proxy.ts in the Node runtime, so both
// the proxy and Server Actions can resolve the SAME durable signing/revocation state from SQLite.
// This is intentionally not module-local: production compiles those entrypoints into independent
// bundles, and module memory cannot provide logout revocation across that boundary or a restart.

// Narrow subpath imports (not the "jose" barrel): keeps jose's JWE/deflate code — which references
// CompressionStream, a Node API the Edge Runtime flags — out of the middleware bundle. We only do
// JWS (sign/verify), never JWE, so nothing is lost.
import { SignJWT } from "jose/jwt/sign";
import { jwtVerify } from "jose/jwt/verify";
import { advanceSessionEpoch, ensureSessionSecret, getSessionEpoch } from "./db";

export const SESSION_COOKIE = "ara_coord_session";
const SESSION_TTL_S = 60 * 60 * 24 * 7; // 7 days
const ALG = "HS256";

function secretKey(): Uint8Array {
  return new TextEncoder().encode(ensureSessionSecret());
}

/** Invalidate every session minted so far — this IS logout. */
export function invalidateSessions(): void {
  advanceSessionEpoch();
}

/** Mint a signed session token. Payload carries the current epoch (for logout revocation) and the
 *  standard exp claim — no secrets. */
export async function createSession(): Promise<string> {
  return new SignJWT({ ep: getSessionEpoch() })
    .setProtectedHeader({ alg: ALG })
    .setIssuedAt()
    .setExpirationTime(`${SESSION_TTL_S}s`)
    .sign(secretKey());
}

/** True iff the token's signature is valid, it has not expired, AND it was minted under the CURRENT
 *  epoch (i.e. no logout has happened since). jose enforces `exp` and — via the pinned `algorithms`
 *  — rejects alg-confusion (e.g. a forged `none`/RS256 header). Any failure (tamper, garbage, expiry,
 *  stale epoch, or no configured secret) resolves to false; never throws. */
export async function verifySession(token: string | undefined): Promise<boolean> {
  if (!token) return false;
  try {
    const { payload } = await jwtVerify(token, secretKey(), { algorithms: [ALG] });
    return payload.ep === getSessionEpoch();
  } catch {
    return false;
  }
}
