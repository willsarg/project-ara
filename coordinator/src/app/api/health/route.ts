// SPDX-License-Identifier: Apache-2.0
// Public, unauthenticated liveness probe (used by the Docker healthcheck).
export const dynamic = "force-dynamic";

export function GET() {
  return Response.json({ service: "ara-coordinator", status: "ok" });
}
