// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Will Sarg
// Keep operator-facing coordinator instructions on the frozen push-only node CLI.
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";

const root = path.resolve(__dirname, "..");

describe("coordinator operator documentation", () => {
  it("uses the canonical node enrollment command and never describes the retired pull server", () => {
    const readme = readFileSync(path.join(root, "README.md"), "utf8");
    const dashboard = readFileSync(path.join(root, "src/app/page.tsx"), "utf8");
    const compose = readFileSync(path.join(root, "compose.yaml"), "utf8");
    for (const text of [readme, dashboard]) {
      expect(text).toContain("ara node enroll");
      expect(text).not.toContain("ara agent enroll");
      expect(text).not.toContain("ara node token");
    }
    expect(readme).toContain("coordinator never opens SSH or connects back to a node");
    expect(compose).toContain("ARA_COORDINATOR_TRUST_PROXY");
    expect(compose).toContain("ARA_COORDINATOR_BIND:-127.0.0.1");
  });

  it("keeps persisted job outcomes and their provenance visible to the administrator", () => {
    const nodes = readFileSync(path.join(root, "src/app/nodes/page.tsx"), "utf8");
    expect(nodes).toContain("Recent work");
    expect(nodes).toContain("job.result_json");
    expect(nodes).toContain("job.error");
    expect(nodes).toContain("job.result_environment_json");
    expect(nodes).toContain("job.dispatched_at");
    expect(nodes).toContain("job.finished_at");
  });
});
