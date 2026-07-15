// SPDX-License-Identifier: Apache-2.0
// Manage the push channel: issue an enrollment token, approve/deny pending agents, and dispatch work
// to (or revoke) active ones. Server Component. Agent secrets (session tokens) live only in SQLite
// as hashes and are never rendered into this page.
import { listActive, listPending, summarizeAgent } from "@/lib/enrollment";
import {
  approveAgentAction,
  denyAgentAction,
  revokeAgentAction,
  submitJobAction,
} from "../actions";
import { IssueToken } from "./issue-token";
import { listRecentWork } from "@/lib/work";

export const dynamic = "force-dynamic";

export default function NodesPage() {
  const pending = listPending();
  const active = listActive();
  const recentWork = listRecentWork();
  return (
    <>
      <header>
        <div className="wrap bar">
          <span className="mark">
            <span className="tick" />
            ARA <small>FLEET</small>
          </span>
          <div className="summary">
            <a className="refresh" href="/">
              ← dashboard
            </a>
          </div>
        </div>
      </header>

      <main className="wrap">
        <section className="lede">
          <h1>Phone-home agents</h1>
          <p>
            Push-only: a node enrolls with a token, waits for approval, then long-polls for work.
            The coordinator never connects back to the node.
          </p>
        </section>

        <section className="panel">
          <h2>Enroll a node</h2>
          <IssueToken />
        </section>

        <section className="panel">
          <h2>Pending approval</h2>
          {pending.length === 0 ? (
            <p className="offline-note">none waiting.</p>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>machine key</th>
                  <th>enrollment id</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {pending.map((a) => (
                  <tr key={a.id}>
                    <td>{a.machine_key || "—"}</td>
                    <td>{a.enrollment_id}</td>
                    <td>
                      <div className="rowacts">
                        <form action={approveAgentAction}>
                          <input type="hidden" name="id" value={a.id} />
                          <button className="btn" type="submit">
                            approve
                          </button>
                        </form>
                        <form action={denyAgentAction}>
                          <input type="hidden" name="id" value={a.id} />
                          <button className="btn ghost" type="submit">
                            deny
                          </button>
                        </form>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        <section className="panel">
          <h2>Approved nodes</h2>
          {active.length === 0 ? (
            <p className="offline-note">none yet.</p>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>machine key</th>
                  <th>last seen</th>
                  <th>serves</th>
                  <th>run a job</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {active.map((a) => {
                  const { serve_models } = summarizeAgent(a);
                  return (
                  <tr key={a.id}>
                    <td>{a.machine_key || "—"}</td>
                    <td>{a.last_seen ?? "never"}</td>
                    <td>
                      {serve_models.length === 0 ? (
                        <span className="offline-note">none yet</span>
                      ) : (
                        serve_models.map((m) => `${m.id} (${m.engine})`).join(", ")
                      )}
                    </td>
                    <td>
                      <form action={submitJobAction} className="rowacts">
                        <input type="hidden" name="agentId" value={a.id} />
                        <input name="model" placeholder="model" required />
                        <input name="prompt" placeholder="prompt" />
                        <button className="btn" type="submit">
                          run
                        </button>
                      </form>
                    </td>
                    <td>
                      <form action={revokeAgentAction}>
                        <input type="hidden" name="id" value={a.id} />
                        <button className="btn ghost" type="submit">
                          revoke
                        </button>
                      </form>
                    </td>
                  </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </section>

        <section className="panel">
          <h2>Recent work</h2>
          {recentWork.length === 0 ? (
            <p className="offline-note">no jobs submitted.</p>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>node</th>
                  <th>job</th>
                  <th>state</th>
                  <th>outcome</th>
                  <th>environment</th>
                  <th>timestamps</th>
                </tr>
              </thead>
              <tbody>
                {recentWork.map((job) => (
                  <tr key={job.id}>
                    <td>{job.machine_key}</td>
                    <td>
                      <code>{job.id}</code><br />
                      {job.kind}{job.model ? ` · ${job.model}` : ""}
                    </td>
                    <td>{job.status}</td>
                    <td className="job-data">
                      {job.error ? `error: ${job.error}` : (job.result_json ?? "—")}
                    </td>
                    <td className="job-data">{job.result_environment_json ?? "—"}</td>
                    <td>
                      queued {job.created_at}<br />
                      {job.dispatched_at ? `dispatched ${job.dispatched_at}` : "not dispatched"}<br />
                      {job.finished_at ? `finished ${job.finished_at}` : "not finished"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </main>
    </>
  );
}
