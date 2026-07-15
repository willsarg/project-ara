// SPDX-License-Identifier: Apache-2.0
// POST /api/enroll — a node phones home with its self-description. Auth: enrollment token (Bearer).
// Creates a PENDING agent and returns its enrollment handle. Node runtime (uses node:crypto + sqlite).
import { NextResponse } from "next/server";
import { bearerToken, verifyEnrollmentToken } from "@/lib/node-auth";
import { enroll, type SelfDescription } from "@/lib/enrollment";
import { rateLimit } from "@/lib/rate-limit";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Per-IP fixed window: generous enough for a legitimate fleet's enroll traffic, tight enough to
// blunt a token-guessing/DoS burst against this (bearer-token-guarded) endpoint.
const ENROLL_MAX = 30;
const ENROLL_WINDOW_MS = 60_000;

/** Best-effort client identity for rate-limiting: the first hop of X-Forwarded-For, or a single
 *  shared "unknown" bucket when there's no reverse proxy in front of the coordinator (degrades to a
 *  global — not per-IP — limit in that case, which still blunts a single-source burst). */
function clientKey(req: Request): string {
  const fwd = req.headers.get("x-forwarded-for");
  return fwd ? fwd.split(",")[0].trim() : "unknown";
}

export async function POST(req: Request) {
  const rl = rateLimit(`enroll:${clientKey(req)}`, ENROLL_MAX, ENROLL_WINDOW_MS);
  if (rl.limited) {
    return NextResponse.json(
      { error: "too many requests" },
      { status: 429, headers: { "Retry-After": String(rl.retryAfterS) } },
    );
  }

  const token = bearerToken(req);
  // Used tokens are admitted so enroll() can resolve response-loss retries idempotently. The lib
  // still rejects a used token that is not bound to an existing enrollment.
  if (!verifyEnrollmentToken(token, { allowUsed: true })) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  let body: SelfDescription;
  try {
    body = (await req.json()) as SelfDescription;
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
  }

  // Lightweight boundary validation (defense-in-depth): reject obviously-malformed payloads before
  // they reach the lib. A node must send a non-empty machine_key and an environment object.
  if (typeof body?.machine_key !== "string" || body.machine_key.length === 0) {
    return NextResponse.json({ error: "machine_key must be a non-empty string" }, { status: 400 });
  }
  if (typeof body?.environment !== "object" || body.environment === null || Array.isArray(body.environment)) {
    return NextResponse.json({ error: "environment must be an object" }, { status: 400 });
  }

  const out = enroll(token!, body);
  if (!out) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  return NextResponse.json(out, { status: 201 });
}
