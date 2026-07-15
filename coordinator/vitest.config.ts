// SPDX-License-Identifier: Apache-2.0
import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  test: {
    environment: "node",
    include: ["test/**/*.test.ts", "src/**/*.test.ts"],
    // Coordinator tests intentionally exercise process-global env, module reloads, in-memory
    // SQLite, and fixed-window limiter state. Running files concurrently lets one file reset or
    // restub another file's runtime, so serialize files while retaining normal in-file execution.
    fileParallelism: false,
    // A default session secret keeps ordinary unit files on the configured-secret path;
    // auth tests override it to exercise durable generated-secret behavior.
    env: { ARA_COORDINATOR_SECRET: "test-secret" },
    coverage: {
      provider: "v8",
      // SCOPE: the TypeScript LOGIC surface only — libs, middleware, API routes, server actions,
      // and the startup guard. Presentation-layer .tsx (pages/layout/login UI under src/app) is
      // deliberately EXCLUDED: it's exercised by `tsc --noEmit` + `next build`, not this unit gate.
      // This mirrors the Python side's 100% bar, which likewise applies to logic, not view markup.
      include: [
        "src/lib/**",
        "src/proxy.ts",
        "src/app/api/**",
        "src/app/actions.ts",
        "src/instrumentation.ts",
        "src/instrumentation-node.ts",
        "next.config.ts",
      ],
      exclude: ["**/*.tsx"],
      thresholds: {
        statements: 100,
        branches: 100,
        functions: 100,
        lines: 100,
      },
    },
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
