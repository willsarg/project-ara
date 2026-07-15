// SPDX-License-Identifier: Apache-2.0
// POST /api/enroll — a node phones home with its self-description. Auth: enrollment token (Bearer).
// Creates a PENDING agent and returns its enrollment handle. Node runtime (uses node:crypto + sqlite).
import { NextResponse } from "next/server";
import { bearerToken, verifyEnrollmentToken } from "@/lib/node-auth";
import { enroll } from "@/lib/enrollment";
import { clientRateLimitKey, rateLimit } from "@/lib/rate-limit";
import { isEnrollmentRequest } from "@/lib/wire-schema";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Per-IP fixed window: generous enough for a legitimate fleet's enroll traffic, tight enough to
// blunt a token-guessing/DoS burst against this (bearer-token-guarded) endpoint.
const ENROLL_MAX = 30;
const ENROLL_WINDOW_MS = 60_000;

export async function POST(req: Request) {
  const rl = rateLimit(
    `enroll:${clientRateLimitKey(req.headers)}`, ENROLL_MAX, ENROLL_WINDOW_MS,
  );
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

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
  }

  if (!isEnrollmentRequest(body)) {
    return NextResponse.json({ error: "invalid enrollment payload" }, { status: 400 });
  }

  const out = enroll(token!, body);
  if (!out) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  return NextResponse.json(out, { status: 201 });
}
