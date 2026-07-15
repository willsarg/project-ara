// SPDX-License-Identifier: Apache-2.0
// POST /api/work/{id}/result — the node reports a finished job. Auth: session token (Bearer).
// Records status + result/error/measurement/environment against the job it was dispatched.
import { NextResponse } from "next/server";
import { bearerToken, verifySessionToken } from "@/lib/node-auth";
import { getWorkById } from "@/lib/db";
import { recordResult } from "@/lib/work";
import { isResultRequest } from "@/lib/result-schema";

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
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
  }

  if (!isResultRequest(body)) {
    return NextResponse.json({ error: "invalid result payload" }, { status: 400 });
  }

  const recorded = recordResult(id, agent.id, {
    status: body.status,
    result: body.result,
    error: body.error,
    measurement: body.measurement,
    environment: body.environment,
  });
  if (recorded === "recorded") return NextResponse.json({ ok: true });
  if (recorded === "already_recorded") {
    return NextResponse.json({ ok: true, already_recorded: true });
  }
  if (recorded === "unknown") {
    return NextResponse.json({ error: "unknown job" }, { status: 404 });
  }
  return NextResponse.json({ error: "job was not acknowledged for execution" }, { status: 409 });
}
