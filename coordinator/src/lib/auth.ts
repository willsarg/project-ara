// SPDX-License-Identifier: Apache-2.0
// Session cookies — a small signed token, verified with Web Crypto so the SAME code runs in both
// the Node runtime (login route) and the Edge runtime (middleware). No third-party auth lib.
//
// IMPORTANT: this module is Edge-safe — it must not import node:crypto, better-sqlite3, or db.ts.
// The signing secret comes ONLY from env, never from the SQLite registry, so middleware (Edge) can
// verify without a DB read. There is deliberately NO default/fallback secret: signing sessions with
// a known key would let anyone forge a session cookie, so we fail closed (see the startup guard in
// instrumentation.ts, which turns this into a clean boot error instead of a per-request throw).

export const SESSION_COOKIE = "ara_coord_session";
const SESSION_TTL_S = 60 * 60 * 24 * 7; // 7 days

// Resolve the signing secret edge-safely: an explicit ARA_COORDINATOR_SECRET, else derived from the
// admin password. If NEITHER is set we throw rather than fall back to a guessable key — no forgeable
// sessions, ever. (The Edge runtime can't generate+persist a random secret, so it must be env-given.)
function secretMaterial(): string {
  const secret =
    process.env.ARA_COORDINATOR_SECRET ||
    (process.env.ARA_COORDINATOR_PASSWORD ? `pw:${process.env.ARA_COORDINATOR_PASSWORD}` : "");
  if (!secret) {
    throw new Error(
      "ARA coordinator: no session secret configured — set ARA_COORDINATOR_SECRET or " +
        "ARA_COORDINATOR_PASSWORD. Refusing to sign sessions with a default (forgeable) key.",
    );
  }
  return secret;
}

function b64urlEncode(bytes: Uint8Array): string {
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function b64urlDecode(s: string): Uint8Array {
  const pad = s.length % 4 === 0 ? "" : "=".repeat(4 - (s.length % 4));
  const bin = atob(s.replace(/-/g, "+").replace(/_/g, "/") + pad);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// Copy into a fresh ArrayBuffer so Web Crypto gets a plain BufferSource (newer DOM lib types make
// Uint8Array generic over ArrayBufferLike, which subtle.* rejects).
function ab(input: string | Uint8Array): ArrayBuffer {
  const bytes = typeof input === "string" ? new TextEncoder().encode(input) : input;
  const out = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(out).set(bytes);
  return out;
}

async function hmacKey(): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    "raw",
    ab(secretMaterial()),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"],
  );
}

/** Mint a signed session token. Payload carries only an expiry — no secrets, no node tokens. */
export async function createSession(): Promise<string> {
  const payload = { exp: Math.floor(Date.now() / 1000) + SESSION_TTL_S };
  const body = b64urlEncode(new TextEncoder().encode(JSON.stringify(payload)));
  const sig = await crypto.subtle.sign("HMAC", await hmacKey(), ab(body));
  return `${body}.${b64urlEncode(new Uint8Array(sig))}`;
}

/** True iff the token's signature is valid and it has not expired. Constant-time via subtle.verify. */
export async function verifySession(token: string | undefined): Promise<boolean> {
  if (!token || !token.includes(".")) return false;
  const [body, sig] = token.split(".");
  try {
    const ok = await crypto.subtle.verify("HMAC", await hmacKey(), ab(b64urlDecode(sig)), ab(body));
    if (!ok) return false;
    const payload = JSON.parse(new TextDecoder().decode(b64urlDecode(body))) as { exp: number };
    return typeof payload.exp === "number" && payload.exp > Math.floor(Date.now() / 1000);
  } catch {
    return false;
  }
}
