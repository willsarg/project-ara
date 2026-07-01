// SPDX-License-Identifier: Apache-2.0
import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  test: {
    environment: "node",
    include: ["test/**/*.test.ts", "src/**/*.test.ts"],
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
