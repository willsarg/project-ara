// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Will Sarg
// Production validation for the pinned result.request and environment wire schemas.
import Ajv2020 from "ajv/dist/2020";

const ENVIRONMENT_SCHEMA = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "https://ara.dev/wire/environment.json",
  type: "object",
  properties: {
    platform: { type: "string", enum: ["linux", "darwin", "windows", "unknown"] },
    accel: { type: "string", enum: ["nvidia", "amd", "metal", "vulkan", "intel", "cpu", "unknown"] },
    containerized: { type: "boolean" },
    virtualization_layer: { type: ["string", "null"] },
    wall_source: { type: "string", enum: ["cgroup", "physical"] },
  },
  required: ["platform", "accel", "containerized", "wall_source"],
  additionalProperties: false,
} as const;

const RESULT_REQUEST_SCHEMA = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "https://ara.dev/wire/result.request.json",
  type: "object",
  properties: {
    status: { type: "string", enum: ["done", "failed"] },
    result: { type: ["object", "null"] },
    error: { type: ["string", "null"] },
    measurement: { type: ["object", "null"] },
    environment: { $ref: ENVIRONMENT_SCHEMA.$id },
  },
  required: ["status", "environment"],
  additionalProperties: false,
  allOf: [
    {
      if: { properties: { status: { const: "done" } } },
      then: { required: ["result"] },
    },
    {
      if: { properties: { status: { const: "failed" } } },
      then: { required: ["error"] },
    },
  ],
} as const;

export interface ResultRequest {
  status: "done" | "failed";
  result?: Record<string, unknown> | null;
  error?: string | null;
  measurement?: Record<string, unknown> | null;
  environment: Record<string, unknown>;
}

const ajv = new Ajv2020({ allErrors: true, strict: false });
ajv.addSchema(ENVIRONMENT_SCHEMA);
const validate = ajv.compile<ResultRequest>(RESULT_REQUEST_SCHEMA);

export function isResultRequest(value: unknown): value is ResultRequest {
  return validate(value);
}
