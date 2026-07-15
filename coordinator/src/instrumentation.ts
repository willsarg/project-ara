// SPDX-License-Identifier: Apache-2.0
// Startup entrypoint — runs once at server boot. Next may trace this file into multiple runtimes;
// keep SQLite/crypto initialization in the Node-only dynamically imported module.
export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    const { registerNode } = await import("./instrumentation-node");
    registerNode();
  }
}
