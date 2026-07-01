// SPDX-License-Identifier: Apache-2.0
// POST /api/enroll — a node phones home with its self-description. Auth: enrollment token (Bearer).
// Creates a PENDING agent and returns its enrollment handle. Node runtime (uses node:crypto + sqlite).
import { NextResponse } from "next/server";
import { bearerToken, verifyEnrollmentToken } from "@/lib/node-auth";
import { enroll, type SelfDescription } from "@/lib/enrollment";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  const token = bearerToken(req);
  if (!verifyEnrollmentToken(token)) {
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
