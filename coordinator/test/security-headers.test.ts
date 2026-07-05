// SPDX-License-Identifier: Apache-2.0
// Security headers applied to every response (next.config.ts `headers()`): a strict-ish CSP, HSTS,
// and the click-jacking/MIME-sniffing/referrer hardening headers. This is plain data (no Next
// runtime needed) so it's exercised directly against the exported config.
import { describe, it, expect } from "vitest";
import nextConfig from "../next.config";

describe("security headers (next.config.ts)", () => {
  it("applies one header set to every route (source matches everything)", async () => {
    const rules = await nextConfig.headers!();
    expect(rules).toHaveLength(1);
    expect(rules[0].source).toBe("/:path*");
  });

  it("sets HSTS with a long max-age and includeSubDomains", async () => {
    const [{ headers }] = await nextConfig.headers!();
    const hsts = headers.find((h) => h.key === "Strict-Transport-Security")?.value;
    expect(hsts).toBeDefined();
    const maxAge = Number(/max-age=(\d+)/.exec(hsts!)?.[1]);
    expect(maxAge).toBeGreaterThanOrEqual(60 * 60 * 24 * 180); // at least ~6 months
    expect(hsts).toMatch(/includeSubDomains/);
  });

  it("sets X-Frame-Options: DENY (clickjacking)", async () => {
    const [{ headers }] = await nextConfig.headers!();
    expect(headers.find((h) => h.key === "X-Frame-Options")?.value).toBe("DENY");
  });

  it("sets X-Content-Type-Options: nosniff and a Referrer-Policy", async () => {
    const [{ headers }] = await nextConfig.headers!();
    expect(headers.find((h) => h.key === "X-Content-Type-Options")?.value).toBe("nosniff");
    expect(headers.find((h) => h.key === "Referrer-Policy")?.value).toBe("strict-origin-when-cross-origin");
  });

  it("sets a CSP that defaults-closed and blocks framing/plugins", async () => {
    const [{ headers }] = await nextConfig.headers!();
    const csp = headers.find((h) => h.key === "Content-Security-Policy")?.value ?? "";
    expect(csp).toContain("default-src 'self'");
    expect(csp).toContain("frame-ancestors 'none'");
    expect(csp).toContain("object-src 'none'");
    expect(csp).toContain("base-uri 'self'");
    expect(csp).toContain("form-action 'self'");
    expect(csp).not.toMatch(/unsafe-eval/);
  });

  it("CSP allows the Google Fonts stylesheet/font origins the layout actually uses", async () => {
    const [{ headers }] = await nextConfig.headers!();
    const csp = headers.find((h) => h.key === "Content-Security-Policy")?.value ?? "";
    expect(csp).toContain("https://fonts.googleapis.com");
    expect(csp).toContain("https://fonts.gstatic.com");
  });

  it("does not disturb the existing build config (standalone output, sqlite external)", () => {
    expect(nextConfig.output).toBe("standalone");
    expect(nextConfig.serverExternalPackages).toEqual(["better-sqlite3"]);
  });
});
