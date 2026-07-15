// SPDX-License-Identifier: Apache-2.0
// The fleet dashboard. A Server Component: it reads the push-channel registry server-side and renders
// the enrolled agents. Push-only — the coordinator never connects back to a node. Agent secrets
// (session tokens) live only in SQLite as hashes and never cross into this output.
import { listAgentSummaries, type AgentSummary } from "@/lib/enrollment";
import { logoutAction } from "./actions";

export const dynamic = "force-dynamic"; // always re-read on load; never statically prerender

export default function Dashboard() {
  const agents = listAgentSummaries();
  const total = agents.length;
  const recentlySeen = agents.filter((a) => a.recently_seen).length;
  const pending = agents.filter((a) => a.status === "pending").length;

  return (
    <>
      <header>
        <div className="wrap bar">
          <span className="mark">
            <span className="tick" />
            ARA <small>FLEET</small>
          </span>
          <div className="summary">
            <span className="chip">
              <b>{total}</b> agent{total === 1 ? "" : "s"}
            </span>
            <span className="chip on">
              <b>{recentlySeen}</b> seen recently
            </span>
            {pending > 0 && (
              <span className="chip run">
                <b>{pending}</b> pending
              </span>
            )}
            <a className="refresh" href="/">
              ↻ refresh
            </a>
            <a className="refresh" href="/nodes">
              nodes
            </a>
            <form action={logoutAction} style={{ display: "inline" }}>
              <button className="refresh" type="submit">
                log out
              </button>
            </form>
          </div>
        </div>
      </header>

      <main className="wrap">
        <section className="lede">
          <h1>
            Your silicon, <em>anywhere</em>.
          </h1>
          <p>
            Every box running ARA phones home — enrolls, waits for approval, then long-polls for
            work. This is the fleet as the coordinator knows it. No SSH, no reaching back in.
          </p>
        </section>

        {agents.length > 0 ? (
          <section className="panel">
            <h2>Enrolled agents</h2>
            <table className="table">
              <thead>
                <tr>
                  <th>machine key</th>
                  <th>status</th>
                  <th>capabilities</th>
                  <th>last seen</th>
                </tr>
              </thead>
              <tbody>
                {agents.map((a) => (
                  <AgentRow key={a.id} a={a} />
                ))}
              </tbody>
            </table>
          </section>
        ) : (
          <section className="empty">
            <h2>No agents yet.</h2>
            <p>Issue an enrollment token, then enroll a box from its own machine.</p>
            <code>ara node enroll &lt;server_url&gt; --token &lt;token&gt;</code>
          </section>
        )}
      </main>

      <footer>
        <div className="wrap foot">
          <span>ARA — AI Runs Anywhere</span>
          <span className="sep">/</span>
          <span>
            {recentlySeen} of {total} agent{total === 1 ? "" : "s"} seen recently
          </span>
        </div>
      </footer>
    </>
  );
}

function AgentRow({ a }: { a: AgentSummary }) {
  const displayedStatus = a.status === "active"
    ? (a.recently_seen ? "seen recently" : "not recently seen")
    : a.status;
  const stateClass = a.recently_seen ? "state-on" : "state-off";
  return (
    <tr>
      <td>{a.machine_key || "—"}</td>
      <td className={stateClass}>{displayedStatus}</td>
      <td>
        {a.caps_count} capabilit{a.caps_count === 1 ? "y" : "ies"}
      </td>
      <td>{a.last_seen ?? "never"}</td>
    </tr>
  );
}
