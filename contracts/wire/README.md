<!-- SPDX-License-Identifier: Apache-2.0 -->
# ARA node ↔ server wire contract

The **single source of truth** for the messages a node and the coordinator exchange over the
push-only (phone-home) channel. Cross-language: the node is Python, the coordinator is TypeScript, so
this contract is JSON Schema (2020-12) + golden fixtures rather than shared code.

- `schema/*.schema.json` — the message schemas, each with a stable `$id` (`https://ara.dev/wire/…`).
- `fixtures/*.json` — golden example instances (valid **and** deliberately-invalid).
- `fixtures/manifest.json` — the case list `{fixture, schema, valid}` that **both** test suites drive.

**Anti-drift backbone:** both suites validate the *same* fixtures against the *same* schemas:
- Node: `tests/test_wire_contract.py` (jsonschema + referencing).
- Coordinator: `coordinator/test/contracts.test.ts` (ajv 2020).

If the two ever disagree, the contract has drifted. Change a message here first, update fixtures, and
let both suites keep it honest. See the design: `2026-07-01-ara-hybrid-cloud-phone-home-architecture`
and the plan `2026-07-01-ara-phone-home-migration-plan` (in the vault).

## v1 messages (all node → server)

| Message | Auth | Schema |
|---|---|---|
| `POST /api/enroll` | enrollment token | `enroll.request` → `enroll.response` |
| `GET /api/enroll/{id}` | enrollment token | `enroll-poll.response` |
| `GET /api/work?wait=N` | session token | `work.response` (200; 204 has no body) |
| `POST /api/work/{id}/ack` | session token | Node durably accepted the offer; execution may begin |
| `POST /api/work/{id}/result` | session token | `result.request` |

Shared: `environment` (on every measurement), `capability` (advertised at enroll). Capability
*matching*/routing and `/api/nodes/self` refresh are deferred (see the plan's slim-v1 cuts).
