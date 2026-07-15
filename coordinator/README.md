# ARA coordinator

The ARA coordinator is the push-only control plane for a fleet of ARA nodes. Nodes make every
network connection: they enroll, wait for approval, long-poll for work, and report results. The
coordinator never opens SSH or connects back to a node.

The web UI provides the single-admin login, enrollment-token issuance, node approval/revocation,
fleet presence, and governed job submission. Node enrollment/session tokens are random opaque
credentials. SQLite stores enrollment-token hashes and the durable session-token hash. During
approval recovery, it temporarily holds the plaintext session token until that node first
authenticates; database access during this window exposes the credential. The browser sees an
enrollment token only once, when the administrator issues it.

## Develop

```bash
npm ci
npm run dev          # http://localhost:3000
```

On first start without `ARA_COORDINATOR_PASSWORD`, the coordinator generates an admin password,
logs it once, and persists only its salted hash. Open `/login`, then issue a one-time enrollment
token at `/nodes`. On the machine being enrolled, run:

```bash
ara node enroll https://coordinator.example --token <token>
ara node run
```

Use HTTPS whenever the node is not connecting to loopback; ARA refuses to send node credentials
over remote plaintext HTTP.

## Run the coordinator

The public lifecycle is one foreground command:

```bash
ara hub
```

ARA builds the coordinator image paired with that installed ARA version, attaches to its logs, and
persists SQLite state in ARA's host data directory. Ctrl-C stops the coordinator. The secure default
is `http://127.0.0.1:3000`; use `--bind`, `--port`, or `--data-dir` to change the host deployment,
and `--rebuild` to force a no-cache image rebuild. The first start prints the generated administrator
password once in the attached logs.

Remote nodes require HTTPS. Keep the default loopback bind and put a TLS reverse proxy in front, or
bind another interface only when the surrounding network and TLS termination are configured.

## Develop or run Compose directly

```bash
mkdir -p data
ARA_HUB_DATA_DIR="$PWD/data" docker compose up --build -d
docker compose logs coordinator
docker compose ps
docker compose down
```

Compose binds port `3000` to `127.0.0.1` and preserves the SQLite registry in the host directory
named by `ARA_HUB_DATA_DIR`.
A one-shot ownership initializer prepares that bind mount before the unprivileged coordinator
starts, including when Docker creates the directory as root on a clean Linux host. Remote nodes
require a TLS reverse proxy in front of that loopback port;
the production admin cookie is `Secure`, and ARA nodes reject remote plaintext HTTP. Set an explicit
password in the environment when desired:

```bash
ARA_HUB_DATA_DIR="$PWD/data" ARA_COORDINATOR_PASSWORD='choose-a-strong-password' \
  docker compose up --build -d
```

The image is a multi-stage Next.js standalone build. Its public `/api/health` endpoint verifies
that the SQLite registry can open, migrate, and answer a read; Docker uses it as the readiness
healthcheck.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `ARA_COORDINATOR_PASSWORD` | generated once | Single administrator login password. |
| `ARA_COORDINATOR_SECRET` | password-derived or generated/persisted | Optional explicit admin-cookie signing secret. |
| `ARA_COORDINATOR_DB` | `./data/coordinator.db` | SQLite registry path. Use a persistent volume in production. |
| `ARA_COORDINATOR_TRUST_PROXY` | unset | Set to `1` only behind a trusted proxy that overwrites `X-Forwarded-For`; enables per-client rate-limit buckets. |
| `ARA_COORDINATOR_BIND` | `127.0.0.1` | Compose host bind address. Keep loopback when a host TLS proxy fronts the coordinator. |
| `ARA_COORDINATOR_PORT` | `3000` | Compose host port. The container always listens on port 3000. |
| `ARA_HUB_DATA_DIR` | `./data`; selected automatically by `ara hub` | Persistent host directory bind-mounted at `/app/data`. |
| `PORT` | `3000` | Listen port. |

Without trusted-proxy mode, forwarding headers are ignored and login/enrollment use conservative
shared rate-limit buckets. This prevents a directly connected caller from bypassing limits by
spoofing `X-Forwarded-For`.

## Auth and storage

Admin pages are protected by a signed, HTTP-only, `SameSite=Lax` session cookie (`Secure` in
production). Logout advances a SQLite-backed session epoch, invalidating copied cookies across
Next.js bundles and process restarts. Node `/api/*` routes use their own hashed bearer credentials;
the public health route is the only unauthenticated non-enrollment surface.

SQLite stores `meta`, `enrollment_tokens`, `agents`, and `work`. Enrollment-token consumption,
distinct owner creation, work offer/acknowledgement, and first-terminal-result recording are
transactional or single-statement atomic operations. A generic enrollment token never authorizes
takeover of an existing node ID or its work. Every reported result retains its measured environment
provenance.

`ara hub` is intentionally attached and single-purpose. Expanded background lifecycle commands can
be added later without changing the coordinator or node protocols.
