// SPDX-License-Identifier: Apache-2.0
// The startup guard (src/instrumentation.ts): refuses to boot with no session secret rather than
// let auth.ts sign with a default (forgeable) key. Verifies both the fail-closed exit path and the
// two ways a secret can satisfy the guard (explicit secret, or a derived-from-password fallback).
// Also verifies the NEXT_RUNTIME guard: Next calls register() in every runtime it traces this file
// into (Node AND Edge), but process.exit is not an Edge API, so the check must be skipped there.
import { describe, it, expect, vi, afterEach } from "vitest";
import { register } from "@/instrumentation";

afterEach(() => vi.unstubAllEnvs());

describe("startup guard (instrumentation.register)", () => {
  it("exits with a clear message when NEITHER secret nor password is set", async () => {
    vi.stubEnv("NEXT_RUNTIME", "nodejs");
    vi.stubEnv("ARA_COORDINATOR_SECRET", "");
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "");
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const exitSpy = vi.spyOn(process, "exit").mockImplementation(() => undefined as never);

    await register();

    expect(errorSpy).toHaveBeenCalledTimes(1);
    expect(errorSpy.mock.calls[0][0]).toMatch(/FATAL: no session secret/);
    expect(exitSpy).toHaveBeenCalledWith(1);

    errorSpy.mockRestore();
    exitSpy.mockRestore();
  });

  it("does NOT exit when ARA_COORDINATOR_SECRET is set", async () => {
    vi.stubEnv("NEXT_RUNTIME", "nodejs");
    vi.stubEnv("ARA_COORDINATOR_SECRET", "s3cret");
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "");
    const exitSpy = vi.spyOn(process, "exit").mockImplementation(() => undefined as never);

    await register();

    expect(exitSpy).not.toHaveBeenCalled();
    exitSpy.mockRestore();
  });

  it("does NOT exit when only ARA_COORDINATOR_PASSWORD is set (derived-secret fallback)", async () => {
    vi.stubEnv("NEXT_RUNTIME", "nodejs");
    vi.stubEnv("ARA_COORDINATOR_SECRET", "");
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "hunter2");
    const exitSpy = vi.spyOn(process, "exit").mockImplementation(() => undefined as never);

    await register();

    expect(exitSpy).not.toHaveBeenCalled();
    exitSpy.mockRestore();
  });

  it("returns without checking anything on the Edge runtime (process.exit is not an Edge API)", async () => {
    vi.stubEnv("NEXT_RUNTIME", "edge");
    vi.stubEnv("ARA_COORDINATOR_SECRET", "");
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "");
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const exitSpy = vi.spyOn(process, "exit").mockImplementation(() => undefined as never);

    await register();

    expect(errorSpy).not.toHaveBeenCalled();
    expect(exitSpy).not.toHaveBeenCalled();

    errorSpy.mockRestore();
    exitSpy.mockRestore();
  });
});
