// SPDX-License-Identifier: Apache-2.0
// Session cookies — a small signed JWT (HS256), verified with `jose` so the SAME code runs in both
// the Node runtime (login route) and the Edge runtime (middleware). jose is Web Crypto-based and
// zero-dependency (MIT), which is exactly why it works edge-side; we do NOT hand-roll the signing.
//
// IMPORTANT: this module is Edge-safe — it must not import node:crypto, better-sqlite3, or db.ts.
// The signing secret comes ONLY from env, never from the SQLite registry, so middleware (Edge) can
// verify without a DB read. There is deliberately NO default/fallback secret: signing sessions with
// a known key would let anyone forge a session cookie, so we fail closed (see the startup guard in
// instrumentation.ts, which turns this into a clean boot error instead of a per-request throw).

// Narrow subpath imports (not the "jose" barrel): keeps jose's JWE/deflate code — which references
// CompressionStream, a Node API the Edge Runtime flags — out of the middleware bundle. We only do
// JWS (sign/verify), never JWE, so nothing is lost.
import { SignJWT } from "jose/jwt/sign";
import { jwtVerify } from "jose/jwt/verify";

export const SESSION_COOKIE = "ara_coord_session";
const SESSION_TTL_S = 60 * 60 * 24 * 7; // 7 days
const ALG = "HS256";

// Resolve the signing secret edge-safely: an explicit ARA_COORDINATOR_SECRET, else derived from the
// admin password. If NEITHER is set we throw rather than fall back to a guessable key — no forgeable
// sessions, ever. (The Edge runtime can't generate+persist a random secret, so it must be env-given.)
function secretKey(): Uint8Array {
  const secret =
    process.env.ARA_COORDINATOR_SECRET ||
    (process.env.ARA_COORDINATOR_PASSWORD ? `pw:${process.env.ARA_COORDINATOR_PASSWORD}` : "");
  if (!secret) {
    throw new Error(
      "ARA coordinator: no session secret configured — set ARA_COORDINATOR_SECRET or " +
        "ARA_COORDINATOR_PASSWORD. Refusing to sign sessions with a default (forgeable) key.",
    );
  }
  return new TextEncoder().encode(secret);
}

/** Mint a signed session token. Payload carries only the standard exp claim — no secrets. */
export async function createSession(): Promise<string> {
  return new SignJWT({})
    .setProtectedHeader({ alg: ALG })
    .setIssuedAt()
    .setExpirationTime(`${SESSION_TTL_S}s`)
    .sign(secretKey());
}

/** True iff the token's signature is valid and it has not expired. jose enforces `exp` and — via the
 *  pinned `algorithms` — rejects alg-confusion (e.g. a forged `none`/RS256 header). Any failure
 *  (tamper, garbage, expiry, or no configured secret) resolves to false; never throws. */
export async function verifySession(token: string | undefined): Promise<boolean> {
  if (!token) return false;
  try {
    await jwtVerify(token, secretKey(), { algorithms: [ALG] });
    return true;
  } catch {
    return false;
  }
}
