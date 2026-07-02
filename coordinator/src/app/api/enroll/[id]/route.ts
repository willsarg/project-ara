// SPDX-License-Identifier: Apache-2.0
// GET /api/enroll/{id} — the node polls for approval. Auth: enrollment token (Bearer). Returns
// {status:"pending"} until an admin approves, then {status:"active", session_token} exactly ONCE.
import { NextResponse } from "next/server";
import { bearerToken } from "@/lib/node-auth";
import { pollApproval } from "@/lib/enrollment";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const token = bearerToken(req);
  const res = pollApproval(id, token ?? "");

  switch (res.kind) {
    case "unauthorized":
      return NextResponse.json({ error: "unauthorized" }, { status: 401 });
    case "not_found":
      return NextResponse.json({ error: "unknown enrollment" }, { status: 404 });
    case "denied":
      return NextResponse.json({ error: "enrollment denied" }, { status: 403 });
    case "consumed":
      // The one-time session token was already delivered; a well-behaved node won't re-poll.
      return NextResponse.json({ error: "session token already delivered" }, { status: 409 });
    case "pending":
      return NextResponse.json({ status: "pending" });
    case "active":
      return NextResponse.json({ status: "active", session_token: res.session_token });
  }
}
