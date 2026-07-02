// SPDX-License-Identifier: Apache-2.0
// Auth gate. Every route is protected except /login and public assets; an unauthenticated request
// is redirected to /login. Next 16 renamed this file convention from "middleware" to "proxy" and
// runs it in the Node.js runtime by default (Next <16 defaulted to Edge); this gate only ever
// verified the signed cookie (Web Crypto), never touched the SQLite registry or any node token, so
// it behaves identically either way.
import { NextResponse, type NextRequest } from "next/server";
import { SESSION_COOKIE, verifySession } from "@/lib/auth";

export async function proxy(req: NextRequest) {
  const ok = await verifySession(req.cookies.get(SESSION_COOKIE)?.value);
  if (ok) return NextResponse.next();

  const url = req.nextUrl.clone();
  url.pathname = "/login";
  url.search = "";
  return NextResponse.redirect(url);
}

export const config = {
  // Protect everything except /login, ALL of /api (the push channel — nodes auth per-route with a
  // Bearer token in the Node runtime, not this edge cookie gate), Next internals, and static files.
  // Exemptions are SEGMENT-anchored ((?:/|$)) so lookalike paths stay gated: /apikeys or /loginX
  // must NOT inherit /api's or /login's exemption (prefix confusion — a future route starting with
  // "api" would otherwise silently skip auth). Covered by test/middleware.test.ts.
  matcher: ["/((?!(?:login|api|_next/static|_next/image|favicon\\.ico)(?:/|$)).*)"],
};
