// SPDX-License-Identifier: Apache-2.0
// Startup initialization creates the durable generated credentials before the first request.
// Also verifies the NEXT_RUNTIME guard: SQLite initialization must stay out of an Edge bundle.
import { describe, it, expect, vi, afterEach } from "vitest";
import { register } from "@/instrumentation";

const setup = vi.hoisted(() => ({
  ensureAdminPassword: vi.fn(),
  ensureSessionSecret: vi.fn(),
}));
vi.mock("@/lib/db", () => setup);

afterEach(() => {
  vi.unstubAllEnvs();
  vi.clearAllMocks();
});

describe("startup initialization (instrumentation.register)", () => {
  it("initializes generated credentials when NEITHER secret nor password is set", async () => {
    vi.stubEnv("NEXT_RUNTIME", "nodejs");
    vi.stubEnv("ARA_COORDINATOR_SECRET", "");
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "");
    const exitSpy = vi.spyOn(process, "exit").mockImplementation(() => undefined as never);

    await register();

    expect(setup.ensureAdminPassword).toHaveBeenCalledOnce();
    expect(setup.ensureSessionSecret).toHaveBeenCalledOnce();
    expect(exitSpy).not.toHaveBeenCalled();

    exitSpy.mockRestore();
  });

  it("does NOT exit when ARA_COORDINATOR_SECRET is set", async () => {
    vi.stubEnv("NEXT_RUNTIME", "nodejs");
    vi.stubEnv("ARA_COORDINATOR_SECRET", "s3cret");
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "");
    const exitSpy = vi.spyOn(process, "exit").mockImplementation(() => undefined as never);

    await register();

    expect(exitSpy).not.toHaveBeenCalled();
    expect(setup.ensureAdminPassword).toHaveBeenCalledOnce();
    expect(setup.ensureSessionSecret).toHaveBeenCalledOnce();
    exitSpy.mockRestore();
  });

  it("does NOT exit when only ARA_COORDINATOR_PASSWORD is set (derived-secret fallback)", async () => {
    vi.stubEnv("NEXT_RUNTIME", "nodejs");
    vi.stubEnv("ARA_COORDINATOR_SECRET", "");
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "hunter2");
    const exitSpy = vi.spyOn(process, "exit").mockImplementation(() => undefined as never);

    await register();

    expect(exitSpy).not.toHaveBeenCalled();
    expect(setup.ensureAdminPassword).toHaveBeenCalledOnce();
    expect(setup.ensureSessionSecret).toHaveBeenCalledOnce();
    exitSpy.mockRestore();
  });

  it("returns without initializing SQLite on the Edge runtime", async () => {
    vi.stubEnv("NEXT_RUNTIME", "edge");
    vi.stubEnv("ARA_COORDINATOR_SECRET", "");
    vi.stubEnv("ARA_COORDINATOR_PASSWORD", "");
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const exitSpy = vi.spyOn(process, "exit").mockImplementation(() => undefined as never);

    await register();

    expect(errorSpy).not.toHaveBeenCalled();
    expect(exitSpy).not.toHaveBeenCalled();
    expect(setup.ensureAdminPassword).not.toHaveBeenCalled();
    expect(setup.ensureSessionSecret).not.toHaveBeenCalled();

    errorSpy.mockRestore();
    exitSpy.mockRestore();
  });
});
