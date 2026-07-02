// SPDX-License-Identifier: Apache-2.0
"use client";
// Minimal client island: issue an enrollment token and show its plaintext ONCE. Uses useActionState
// so the value returned by the server action is rendered (a plain <form action> can't display it).
import { useActionState } from "react";
import { issueEnrollmentTokenAction } from "../actions";

export function IssueToken() {
  const [state, action, pending] = useActionState(issueEnrollmentTokenAction, undefined);
  return (
    <form action={action}>
      <button className="btn" type="submit" disabled={pending}>
        Issue enrollment token
      </button>
      {state?.token && (
        <p className="offline-note">
          Enrollment token (copy now — shown once): <code>{state.token}</code>
        </p>
      )}
    </form>
  );
}
