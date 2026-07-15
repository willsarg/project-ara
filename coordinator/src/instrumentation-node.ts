// SPDX-License-Identifier: Apache-2.0
// Node-runtime-only startup initialization, dynamically imported only for NEXT_RUNTIME=nodejs.
// This establishes generated credentials before the first request. The admin password is logged
// once by db.ts; the random session signing secret is persisted but never logged.
import { ensureAdminPassword, ensureSessionSecret } from "./lib/db";

export function registerNode(): void {
  ensureAdminPassword();
  ensureSessionSecret();
}
