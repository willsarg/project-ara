// SPDX-License-Identifier: Apache-2.0
import type { NextConfig } from "next";

// Applied to EVERY response (pages, /login, and the /api/* push-channel routes) via headers()
// below — Next merges these onto its own routing layer, so they cover routes proxy.ts's auth gate
// deliberately exempts too (login, api, static assets) rather than only the authenticated pages.
//
// CSP: no nonces/hashes (would need per-request wiring through Next's Server Components render) —
// 'unsafe-inline' on script-src is the pragmatic call for Next's own inline hydration bootstrap;
// this is still a real, meaningful restriction (blocks loading/executing any THIRD-PARTY script or
// exfiltrating via a foreign origin) even though it doesn't fully close inline-script XSS. Google
// Fonts origins are allowlisted because src/app/layout.tsx actually loads them.
const CSP = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline'",
  "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
  "font-src 'self' https://fonts.gstatic.com",
  "img-src 'self' data:",
  "connect-src 'self'",
  "frame-ancestors 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  "object-src 'none'",
].join("; ");

const SECURITY_HEADERS = [
  { key: "Content-Security-Policy", value: CSP },
  // 2 years + includeSubDomains: this coordinator is only ever meant to be served over TLS in
  // production; browsers ignore HSTS on a plain-http response anyway, so sending it unconditionally
  // (incl. local dev over http) is harmless.
  { key: "Strict-Transport-Security", value: "max-age=63072000; includeSubDomains" },
  { key: "X-Frame-Options", value: "DENY" }, // clickjacking; belt-and-suspenders with frame-ancestors
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
];

const nextConfig: NextConfig = {
  // Emit a self-contained server (.next/standalone) for a slim Docker image.
  output: "standalone",
  // better-sqlite3 is a native module — keep it external so Next doesn't try to bundle the .node binding.
  serverExternalPackages: ["better-sqlite3"],
  // Next 16.2's file tracer sees sharp 0.35's platform package but misses its native libvips
  // payload. Include the installed platform's library explicitly so the standalone image can load
  // sharp; the wildcard remains portable across macOS/Linux, CPU architectures, and libc variants.
  outputFileTracingIncludes: {
    "/*": ["./node_modules/@img/sharp-libvips-*/lib/**/*"],
  },
  async headers() {
    return [{ source: "/:path*", headers: SECURITY_HEADERS }];
  },
};

export default nextConfig;
