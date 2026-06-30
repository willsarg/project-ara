// SPDX-License-Identifier: Apache-2.0
"use client";
import { useActionState } from "react";
import { loginAction } from "../actions";

export default function LoginPage() {
  const [state, action, pending] = useActionState(loginAction, undefined);
  return (
    <main className="wrap">
      <div className="login-wrap">
        <div className="mark" style={{ justifyContent: "center", marginBottom: 24 }}>
          <span className="tick" />
          ARA <small>FLEET</small>
        </div>
        <form className="panel" action={action}>
          <h2>Sign in</h2>
          {state?.error && <p className="err">{state.error}</p>}
          <label className="field">
            <span>admin password</span>
            <input type="password" name="password" autoFocus autoComplete="current-password" />
          </label>
          <button className="btn" type="submit" disabled={pending}>
            {pending ? "…" : "Enter"}
          </button>
        </form>
      </div>
    </main>
  );
}
