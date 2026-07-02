// SPDX-License-Identifier: Apache-2.0
// Auth gate. Every route is protected except /login and public assets; an unauthenticated request
// is redirected to /login. Runs in the Edge runtime — it only verifies the signed cookie (Web
// Crypto), never touches the SQLite registry or any node token.
import { NextResponse, type NextRequest } from "next/server";
import { SESSION_COOKIE, verifySession } from "@/lib/auth";

export async function middleware(req: NextRequest) {
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
  matcher: ["/((?!login|api|_next/static|_next/image|favicon.ico).*)"],
};
