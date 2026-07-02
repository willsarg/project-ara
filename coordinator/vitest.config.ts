// SPDX-License-Identifier: Apache-2.0
import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  test: {
    environment: "node",
    include: ["test/**/*.test.ts", "src/**/*.test.ts"],
    // A default session secret so tests that touch auth.ts don't trip the no-secret guard;
    // auth.test.ts overrides it per-case with vi.stubEnv.
    env: { ARA_COORDINATOR_SECRET: "test-secret" },
  },
  resolve: {
    alias: {
      // `import "server-only"` throws outside a Server Component; under vitest we import the
      // server modules (db.ts, etc.) directly, so map it to an empty module. The production
      // guard is untouched — this alias only applies to the test run.
      "server-only": path.resolve(__dirname, "test/stubs/server-only.ts"),
      // Mirror the app's "@/*" -> "src/*" path alias so tests import the same way app code does.
      "@": path.resolve(__dirname, "src"),
    },
  },
});
