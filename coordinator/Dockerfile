# SPDX-License-Identifier: Apache-2.0
# Multi-stage build for the ARA coordinator (Next.js standalone output).
#
# The base is pinned by digest (node:22-bookworm-slim@sha256:…) for reproducible, immutable builds.
# Dependabot's docker ecosystem bumps the digest when the tag is re-published with patches (one PR,
# gated by the Docker Image Scan workflow). Keep all three stages on the same digest.

# ---- deps: install node modules (incl. compiling better-sqlite3's native binding) -------------
FROM node:22-bookworm-slim@sha256:813a7480f28fdadac1f7f5c824bcdad435b5bc1322a5968bbbdef8d058f9dff4 AS deps
WORKDIR /app
# better-sqlite3 builds from source → needs python3 + a C++ toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 make g++ && rm -rf /var/lib/apt/lists/*
COPY package.json package-lock.json* ./
RUN npm ci

# ---- builder: produce .next/standalone ---------------------------------------------------------
FROM node:22-bookworm-slim@sha256:813a7480f28fdadac1f7f5c824bcdad435b5bc1322a5968bbbdef8d058f9dff4 AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ---- runner: slim runtime image ----------------------------------------------------------------
FROM node:22-bookworm-slim@sha256:813a7480f28fdadac1f7f5c824bcdad435b5bc1322a5968bbbdef8d058f9dff4 AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME=0.0.0.0
# The SQLite registry lives here; compose mounts a volume over it.
ENV ARA_COORDINATOR_DB=/app/data/coordinator.db

RUN useradd -m -u 1001 ara
# Standalone server + its traced node_modules (includes the better-sqlite3 native binding).
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
RUN mkdir -p /app/data && chown -R ara:ara /app
USER ara

EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=4s --start-period=10s \
    CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||3000)+'/api/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
CMD ["node", "server.js"]
