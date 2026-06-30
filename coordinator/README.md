# ARA coordinator

A standalone web coordinator for an ARA fleet. It shows every registered ARA **node** (a headless,
token-authed daemon) live over the network — the silicon it's got, how close it sits to its memory
wall, and what it's running right now.

The coordinator is a pure HTTP **client** of nodes. It holds each node's bearer token **server-side**
(in a SQLite registry) and is the only place that token is attached to a request. The browser only
ever receives rendered HTML — never a token.

Built with Next.js (App Router, TypeScript). All node calls happen in Server Components / Route
Handlers / Server Actions.

## Develop

```bash
npm install
npm run dev          # http://localhost:3000
```

On first load you'll be redirected to `/login`. Set a password (below) or read the one generated on
first run from stdout.

Register nodes at `/nodes` (name, base URL like `http://192.168.1.50:8473`, and the node's token —
get it on the node with `ara node token`). The dashboard (`/`) polls each enabled node's `/detect`
and `/status` server-side with a ~2.5s timeout; an offline node renders as a dim "offline" card and
never stalls the page. Hit **↻ refresh** to re-poll.

## Deploy (Docker)

```bash
ARA_COORDINATOR_PASSWORD=your-password docker compose up --build
```

Publishes port `3000` and mounts `./data` for the SQLite registry (so nodes + secrets survive
restarts). The image is a multi-stage build of Next.js `output: 'standalone'`.

> Note: the Docker daemon was off in the environment this was built in, so the image was not run
> here — the `Dockerfile` / `compose.yaml` are written to be correct (the standalone build, native
> `better-sqlite3` binding, healthcheck, and volume were all verified against the local build).

## Environment variables

| Variable                    | Default                  | Purpose                                                                 |
| --------------------------- | ------------------------ | ----------------------------------------------------------------------- |
| `ARA_COORDINATOR_PASSWORD`  | _generated on first run_ | Single admin login password. If unset, one is generated and logged once to stdout. |
| `ARA_COORDINATOR_SECRET`    | derived from password    | Optional explicit secret for signing the session cookie.                |
| `ARA_COORDINATOR_DB`        | `./data/coordinator.db`  | SQLite registry path. Use an absolute path / volume in production.       |
| `PORT`                      | `3000`                   | Listen port.                                                            |

## Auth

Login-gated by a single admin password. A successful login sets a signed, **httpOnly** session
cookie (HMAC-SHA-256 via Web Crypto). `middleware.ts` verifies that cookie on every route except
`/login`, `/api/health`, and static assets; an unauthenticated request is redirected to `/login`.
No third-party auth library.

## Registry

SQLite via `better-sqlite3` at `ARA_COORDINATOR_DB`. Table `nodes(id, name UNIQUE, base_url, token,
enabled)` plus a small `meta` table (generated password). Manage nodes at `/nodes` — add, enable/
disable, delete. The token column is masked in the UI; the real token never leaves the server.
