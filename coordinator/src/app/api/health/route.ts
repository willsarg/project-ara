// SPDX-License-Identifier: Apache-2.0
// Public, unauthenticated readiness probe (used by the Docker healthcheck).
import { checkDatabaseReady } from "@/lib/db";

export const dynamic = "force-dynamic";

export function GET() {
  try {
    checkDatabaseReady();
    return Response.json({ service: "ara-coordinator", status: "ok" });
  } catch {
    return Response.json(
      { service: "ara-coordinator", status: "unavailable" }, { status: 503 },
    );
  }
}
