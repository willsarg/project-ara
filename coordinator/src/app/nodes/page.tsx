// SPDX-License-Identifier: Apache-2.0
// Manage the registry: add a node, toggle it on/off, delete it. Server Component. The token column
// is masked — the real token lives only in SQLite and is attached to requests server-side; it is
// never rendered into this page.
import { listNodes } from "@/lib/db";
import { addNodeAction, deleteNodeAction, toggleNodeAction } from "../actions";

export const dynamic = "force-dynamic";

export default function NodesPage() {
  const nodes = listNodes();
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
          <h1>Nodes</h1>
          <p>Register a box, then watch it on the dashboard. Tokens are stored server-side only.</p>
        </section>

        <section className="grid" style={{ gridTemplateColumns: "minmax(280px, 360px) 1fr" }}>
          <form className="panel" action={addNodeAction}>
            <h2>Add a node</h2>
            <label className="field">
              <span>name</span>
              <input name="name" placeholder="rog" required />
            </label>
            <label className="field">
              <span>base url</span>
              <input name="base_url" placeholder="http://192.168.1.50:8473" required />
            </label>
            <label className="field">
              <span>token</span>
              <input name="token" type="password" placeholder="bearer token" required />
            </label>
            <button className="btn" type="submit">
              Register node
            </button>
          </form>

          <div className="panel">
            <h2>Registered</h2>
            {nodes.length === 0 ? (
              <p className="offline-note">none yet.</p>
            ) : (
              <table className="table">
                <thead>
                  <tr>
                    <th>name</th>
                    <th>base url</th>
                    <th>token</th>
                    <th>state</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {nodes.map((n) => (
                    <tr key={n.id}>
                      <td>{n.name}</td>
                      <td>{n.base_url}</td>
                      <td className="tokmask">•••••• (server-side)</td>
                      <td className={n.enabled ? "state-on" : "state-off"}>
                        {n.enabled ? "enabled" : "disabled"}
                      </td>
                      <td>
                        <div className="rowacts">
                          <form action={toggleNodeAction}>
                            <input type="hidden" name="id" value={n.id} />
                            <button className="btn ghost" type="submit">
                              {n.enabled ? "disable" : "enable"}
                            </button>
                          </form>
                          <form action={deleteNodeAction}>
                            <input type="hidden" name="id" value={n.id} />
                            <button className="btn ghost" type="submit">
                              delete
                            </button>
                          </form>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>
      </main>
    </>
  );
}
