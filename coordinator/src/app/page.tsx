// SPDX-License-Identifier: Apache-2.0
// The fleet dashboard. A Server Component: it reads the registry and polls each enabled node's
// /detect + /status server-side (short timeout), then renders the ported instrument-panel design.
// Tokens never cross into this output — only probed, render-ready rows do.
import { listNodes } from "@/lib/db";
import { probeAll, type NodeRow } from "@/lib/probe";
import { logoutAction } from "./actions";

export const dynamic = "force-dynamic"; // always re-poll on load; never statically prerender

export default async function Dashboard() {
  const rows = await probeAll(listNodes());
  const total = rows.length;
  const online = rows.filter((n) => n.online).length;
  const running = rows.filter((n) => n.running > 0).length;

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
              <b>{total}</b> node{total === 1 ? "" : "s"}
            </span>
            <span className="chip on">
              <b>{online}</b> online
            </span>
            {running > 0 && (
              <span className="chip run">
                <b>{running}</b> running
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
            Every box running ARA, read live over the network — the hardware it&apos;s got, how close
            it sits to its memory wall, and what it&apos;s running right now. No SSH.
          </p>
        </section>

        {rows.length > 0 ? (
          <section className="grid">
            {rows.map((n, i) => (
              <NodeCard key={n.id} n={n} i={i} />
            ))}
          </section>
        ) : (
          <section className="empty">
            <h2>No nodes yet.</h2>
            <p>Stand a box up as a node, then register it here.</p>
            <code>ara server addnode &lt;name&gt; &lt;base_url&gt; &lt;token&gt;</code>
          </section>
        )}
      </main>

      <footer>
        <div className="wrap foot">
          <span>ARA — AI Runs Anywhere</span>
          <span className="sep">/</span>
          <span>
            {online} of {total} node{total === 1 ? "" : "s"} reporting
          </span>
        </div>
      </footer>
    </>
  );
}

function NodeCard({ n, i }: { n: NodeRow; i: number }) {
  const state = n.online ? (n.running > 0 ? "busy" : "up") : "down";
  return (
    <article className={`node ${state}`} style={{ animationDelay: `${i}0ms` }}>
      <div className="ntop">
        <span className="dot" />
        <span className="name">{n.name}</span>
        {n.system && <span className="sys">{n.system}</span>}
      </div>

      {n.online ? (
        <>
          <div className="spec">
            <div className="row">
              <span className="k">chip</span>
              <span className="v">{n.chip}</span>
            </div>
            <div className="row">
              <span className="k">cores</span>
              <span className="v">{n.cores} threads</span>
            </div>
            <div className="row">
              <span className="k">accel</span>
              <span className="v accel">{n.accel ? n.accel : "— cpu only"}</span>
            </div>
          </div>

          <div className="wall">
            <div className="cap">
              <span>memory wall</span>
              <b>
                {n.ram_used} / {n.ram_total} GB
              </b>
            </div>
            <div className="track">
              <div className="ticks">
                <i style={{ left: "25%" }} />
                <i style={{ left: "50%" }} />
                <i style={{ left: "75%" }} />
              </div>
              <div
                className={`fill ${n.ram_pct >= 85 ? "near" : ""}`}
                style={{ width: `${n.ram_pct}%` }}
              />
              <div className="wallcap" />
            </div>
          </div>

          <div className="nfoot">
            {n.running > 0 ? (
              <span className="running">▶ {n.running} running</span>
            ) : (
              <a className="url" href={n.base_url} target="_blank" rel="noopener noreferrer">
                {n.base_url}
              </a>
            )}
            <a className="manage" href="/nodes">
              manage
            </a>
          </div>
        </>
      ) : (
        <>
          <p className="offline-note">
            unreachable · <b>{n.base_url}</b>
          </p>
          <div className="nfoot">
            <a className="url" href="/">
              ↻ retry
            </a>
            <a className="manage" href="/nodes">
              manage
            </a>
          </div>
        </>
      )}
    </article>
  );
}
