// SPDX-License-Identifier: Apache-2.0
// The coordinator half of the node<->server wire contract. Validates the SAME golden fixtures
// against the SAME schemas as the node's tests/test_wire_contract.py (via the shared manifest).
// If these two suites ever disagree, the contract has drifted.
import { describe, it, expect } from "vitest";
import Ajv2020 from "ajv/dist/2020";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { ALLOWED_JOB_KINDS } from "@/lib/job-kinds";
import { isResultRequest } from "@/lib/result-schema";

const WIRE = path.resolve(__dirname, "../../contracts/wire");
const SCHEMA_DIR = path.join(WIRE, "schema");
const FIXTURE_DIR = path.join(WIRE, "fixtures");

const readJson = (p: string): unknown => JSON.parse(readFileSync(p, "utf8"));

function buildAjv(): Ajv2020 {
  // strict:false — the contract uses descriptive keywords; we validate shape, not lint the schema.
  const ajv = new Ajv2020({ allErrors: true, strict: false });
  for (const f of readdirSync(SCHEMA_DIR)) {
    if (f.endsWith(".schema.json")) ajv.addSchema(readJson(path.join(SCHEMA_DIR, f)) as object);
  }
  return ajv;
}

const ajv = buildAjv();
const manifest = readJson(path.join(FIXTURE_DIR, "manifest.json")) as {
  cases: { fixture: string; schema: string; valid: boolean }[];
};

describe("wire contract fixtures (shared with the node's test_wire_contract.py)", () => {
  for (const c of manifest.cases) {
    it(`${c.fixture} is ${c.valid ? "valid" : "invalid"}`, () => {
      const validate = ajv.getSchema(c.schema);
      expect(validate, `schema not loaded: ${c.schema}`).toBeTruthy();
      const ok = validate!(readJson(path.join(FIXTURE_DIR, c.fixture)));
      expect(ok).toBe(c.valid);
    });
  }

  it("references every fixture on disk (no orphans)", () => {
    const referenced = new Set(manifest.cases.map((c) => c.fixture));
    const onDisk = readdirSync(FIXTURE_DIR).filter((f) => f.endsWith(".json") && f !== "manifest.json");
    expect(new Set(onDisk)).toEqual(referenced);
  });

  it("keeps the coordinator dispatch allowlist equal to the pinned work schema", () => {
    const schema = readJson(path.join(SCHEMA_DIR, "work.response.schema.json")) as {
      properties: { job: { properties: { kind: { enum: string[] } } } };
    };
    expect([...ALLOWED_JOB_KINDS]).toEqual(schema.properties.job.properties.kind.enum);
  });

  it("keeps production result validation aligned with every pinned result fixture", () => {
    for (const c of manifest.cases.filter((entry) => entry.schema.endsWith("result.request.json"))) {
      expect(isResultRequest(readJson(path.join(FIXTURE_DIR, c.fixture))), c.fixture).toBe(c.valid);
    }
  });
});
