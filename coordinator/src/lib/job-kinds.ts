// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Will Sarg
// The coordinator's complete dispatch vocabulary. Every enqueue and DB write uses this one list.

export const ALLOWED_JOB_KINDS = ["run", "characterize", "detect", "benchmark"] as const;

export type JobKind = (typeof ALLOWED_JOB_KINDS)[number];

export function isAllowedJobKind(kind: string): kind is JobKind {
  return (ALLOWED_JOB_KINDS as readonly string[]).includes(kind);
}

export function assertAllowedJobKind(kind: string): asserts kind is JobKind {
  if (!isAllowedJobKind(kind)) throw new RangeError(`invalid job kind: ${kind}`);
}
