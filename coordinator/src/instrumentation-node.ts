// SPDX-License-Identifier: Apache-2.0
// Node-runtime-only half of the startup guard — split out of src/instrumentation.ts so Turbopack
// never has to trace process.exit (not an Edge API) into the Edge runtime bundle; see the comment
// there for why. Dynamically imported ONLY when NEXT_RUNTIME is "nodejs".
//
// The coordinator signs session cookies with ARA_COORDINATOR_SECRET (or a key derived from
// ARA_COORDINATOR_PASSWORD). With NEITHER set, the old code fell back to a KNOWN string and anyone
// could forge a session. We refuse to start instead, with a clear message — rather than surfacing a
// cryptic per-request throw from auth.ts.
export function registerNode(): void {
  if (!process.env.ARA_COORDINATOR_SECRET && !process.env.ARA_COORDINATOR_PASSWORD) {
    console.error(
      "\n[ara-coordinator] FATAL: no session secret.\n" +
        "  Set ARA_COORDINATOR_SECRET (or ARA_COORDINATOR_PASSWORD) before starting — sessions are\n" +
        "  signed with it, and refusing to start beats signing with a default (forgeable) key.\n" +
        "  e.g.  ARA_COORDINATOR_PASSWORD=<choose-one> npm start\n",
    );
    process.exit(1); // refuse to start — a clean exit beats a port that 500s every request
  }
}
