// SPDX-License-Identifier: Apache-2.0
// Startup guard entrypoint — runs once at server boot. Next calls register() in EVERY runtime it
// traces this file into (Node AND Edge). The actual check (src/instrumentation-node.ts) refuses to
// start with no session secret via process.exit, which is not an Edge API — Turbopack statically
// flags process.exit if it's reachable from this file at all, even behind a runtime env-guard. Per
// the documented pattern (https://nextjs.org/docs/app/guides/instrumentation, "Importing
// runtime-specific code"), isolate the Node-only code in its own module and import it ONLY on the
// Node runtime, so it — and process.exit — never end up in the Edge trace.
export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    const { registerNode } = await import("./instrumentation-node");
    registerNode();
  }
}
