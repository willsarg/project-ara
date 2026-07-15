// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Will Sarg
// POST /api/work/{id}/ack — node confirms the offered job is durable locally before execution.
import { NextResponse } from "next/server";
import { bearerToken, verifySessionToken } from "@/lib/node-auth";
import { acknowledge } from "@/lib/work";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const agent = verifySessionToken(bearerToken(req));
  if (!agent) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const outcome = acknowledge(id, agent.id);
  if (outcome === "unknown") {
    return NextResponse.json({ error: "unknown job" }, { status: 404 });
  }
  if (outcome === "conflict") {
    return NextResponse.json({ error: "job is not awaiting acknowledgement" }, { status: 409 });
  }
  return NextResponse.json({ ok: true });
}
