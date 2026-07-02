// SPDX-License-Identifier: Apache-2.0
// GET /api/work?wait=N — the node long-polls for a job. Auth: session token (Bearer). Returns
// 200 {job:{id,kind,args}} when work is queued, or 204 (no body) after N seconds of no work.
import { NextResponse } from "next/server";
import { bearerToken, verifySessionToken } from "@/lib/node-auth";
import { touchAgentLastSeen } from "@/lib/db";
import { nextForAgent } from "@/lib/work";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_WAIT_S = 30; // cap the long-poll window regardless of what the node asks for

export async function GET(req: Request) {
  const agent = verifySessionToken(bearerToken(req));
  if (!agent) return NextResponse.json({ error: "unauthorized" }, { status: 401 });

  touchAgentLastSeen(agent.id); // this poll IS the heartbeat

  const raw = Number(new URL(req.url).searchParams.get("wait") ?? "0");
  const waitS = Number.isFinite(raw) ? Math.min(MAX_WAIT_S, Math.max(0, raw)) : 0;

  const job = await nextForAgent(agent.id, waitS * 1000);
  if (!job) return new NextResponse(null, { status: 204 });
  return NextResponse.json({ job });
}
