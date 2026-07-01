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

  const out = enroll(token!, body);
  if (!out) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  return NextResponse.json(out, { status: 201 });
}
