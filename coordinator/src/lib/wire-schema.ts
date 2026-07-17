// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Will Sarg
// Production validators for coordinator-owned node request payloads. The validation vocabulary in
// these definitions is checked against contracts/wire/schema by test/contracts.test.ts.
import Ajv2020 from "ajv/dist/2020";

export const ENVIRONMENT_SCHEMA = {
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

export const CAPABILITY_SCHEMA = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "https://ara.dev/wire/capability.json",
  type: "object",
  properties: {
    kind: { type: "string", enum: ["serve_model", "embeddings"] },
    id: { type: "string", minLength: 1 },
    engine: { type: "string", minLength: 1 },
    evidence: { type: "string", enum: ["characterized", "estimated", "none"] },
    runtime: { type: "string", minLength: 1 },
    backend: { type: "string", minLength: 1 },
    artifact_id: { type: "string", minLength: 1 },
    config_key: { type: "string", minLength: 1 },
    safe_context: { type: "integer", minimum: 1 },
    authority: { type: "string", pattern: "^node-target:v1:[0-9a-f]{64}$" },
  },
  required: ["kind", "id", "engine", "evidence"],
  allOf: [{
    if: { properties: { engine: { const: "ollama" } }, required: ["engine"] },
    then: { required: ["runtime", "backend", "artifact_id", "config_key", "safe_context", "authority"] },
  }],
  additionalProperties: false,
} as const;

export const ENROLL_REQUEST_SCHEMA = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "https://ara.dev/wire/enroll.request.json",
  type: "object",
  properties: {
    machine_key: { type: "string", minLength: 1 },
    identity: {
      type: "object",
      properties: {
        hostname: { type: "string", minLength: 1 },
        os: { type: "string" },
        arch: { type: "string" },
      },
      required: ["hostname"],
      additionalProperties: true,
    },
    profile_projection: { type: "object" },
    capabilities: {
      type: "array",
      items: { $ref: CAPABILITY_SCHEMA.$id },
    },
    environment: { $ref: ENVIRONMENT_SCHEMA.$id },
  },
  required: ["machine_key", "identity", "capabilities", "environment"],
  additionalProperties: false,
} as const;

export const RESULT_REQUEST_SCHEMA = {
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
      then: { required: ["result"], properties: { error: { type: "null" } } },
    },
    {
      if: { properties: { status: { const: "failed" } } },
      then: {
        required: ["error"],
        properties: { error: { type: "string", minLength: 1 }, result: { type: "null" } },
      },
    },
  ],
} as const;

export interface EnrollmentRequest {
  machine_key: string;
  identity: Record<string, unknown> & { hostname: string; os?: string; arch?: string };
  profile_projection?: Record<string, unknown>;
  capabilities: Array<{
    kind: "serve_model" | "embeddings";
    id: string;
    engine: string;
    evidence: "characterized" | "estimated" | "none";
    runtime?: string;
    backend?: string;
    artifact_id?: string;
    config_key?: string;
    safe_context?: number;
    authority?: string;
  }>;
  environment: Record<string, unknown>;
}

interface ResultRequestBase {
  measurement?: Record<string, unknown> | null;
  environment: Record<string, unknown>;
}

export type ResultRequest = ResultRequestBase & (
  | { status: "done"; result: Record<string, unknown> | null; error?: null }
  | { status: "failed"; error: string; result?: null }
);

const ajv = new Ajv2020({ allErrors: true, strict: false });
ajv.addSchema(ENVIRONMENT_SCHEMA);
ajv.addSchema(CAPABILITY_SCHEMA);
const validateEnrollmentRequest = ajv.compile<EnrollmentRequest>(ENROLL_REQUEST_SCHEMA);
const validateResultRequest = ajv.compile<ResultRequest>(RESULT_REQUEST_SCHEMA);

export function isEnrollmentRequest(value: unknown): value is EnrollmentRequest {
  return validateEnrollmentRequest(value);
}

export function isResultRequest(value: unknown): value is ResultRequest {
  return validateResultRequest(value);
}
