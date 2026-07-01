// SPDX-License-Identifier: Apache-2.0
// POST /api/work/{id}/result — the node reports a finished job. Auth: session token (Bearer).
// Records status + result/error/measurement/environment against the job it was dispatched.
import { NextResponse } from "next/server";
import { bearerToken, verifySessionToken } from "@/lib/node-auth";
import { getWorkById } from "@/lib/db";
import { recordResult } from "@/lib/work";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const agent = verifySessionToken(bearerToken(req));
  if (!agent) return NextResponse.json({ error: "unauthorized" }, { status: 401 });

  // A session can only report on its OWN jobs (also covers unknown ids → 404).
  const job = getWorkById(id);
  if (!job || job.agent_id !== agent.id) {
    return NextResponse.json({ error: "unknown job" }, { status: 404 });
  }

  let body: {
    status?: string;
    result?: unknown;
    error?: unknown;
    measurement?: unknown;
    environment?: unknown;
  };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
  }

  if (body?.status !== "done" && body?.status !== "failed") {
    return NextResponse.json({ error: "status must be done|failed" }, { status: 400 });
  }

  recordResult(id, {
    status: body.status,
    result: body.result,
    error: body.error,
    measurement: body.measurement,
  });
  return NextResponse.json({ ok: true });
}
