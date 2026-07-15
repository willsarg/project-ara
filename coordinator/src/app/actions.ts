// SPDX-License-Identifier: Apache-2.0
"use server";
// Server Actions — login/logout and push-channel admin mutations. All run server-side; agent
// secrets minted here go straight into SQLite (as hashes) and are never returned to the caller.
import { cookies, headers } from "next/headers";
import { redirect } from "next/navigation";
import { revalidatePath } from "next/cache";
import { SESSION_COOKIE, createSession, invalidateSessions } from "@/lib/auth";
import { verifyAdminPassword } from "@/lib/db";
import { approveAgent, denyAgent, issueEnrollmentToken, revoke } from "@/lib/enrollment";
import { AgentNotActiveError, enqueue } from "@/lib/work";
import { clientRateLimitKey, rateLimit } from "@/lib/rate-limit";

const SESSION_TTL_S = 60 * 60 * 24 * 7;

// Login attempts are rate-limited per client (see clientKey below). Server Actions have no HTTP
// status of their own to set (Next always answers the action's POST with 200 and an RSC payload,
// even on a thrown error) — so unlike /api/enroll's real 429, the limit here surfaces through the
// SAME { error } shape the form already renders.
const LOGIN_MAX = 10;
const LOGIN_WINDOW_MS = 60_000;

export async function loginAction(_prev: { error?: string } | undefined, form: FormData) {
  const rl = rateLimit(
    `login:${clientRateLimitKey(await headers())}`, LOGIN_MAX, LOGIN_WINDOW_MS,
  );
  if (rl.limited) {
    return { error: `Too many attempts. Try again in ${rl.retryAfterS}s.` };
  }

  const password = String(form.get("password") ?? "");
  if (!verifyAdminPassword(password)) {        // constant-time; compares against the stored hash
    return { error: "Incorrect password." };
  }
  const token = await createSession();
  (await cookies()).set(SESSION_COOKIE, token, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: SESSION_TTL_S,
  });
  redirect("/");
}

export async function logoutAction() {
  invalidateSessions(); // stateless JWTs can't be un-signed — kill them server-side, not just the cookie
  (await cookies()).delete(SESSION_COOKIE);
  redirect("/login");
}

// --- Phone-home (push) admin actions -------------------------------------------------------------

/** Mint an enrollment token and RETURN the plaintext so the page can show it ONCE (useActionState).
 *  This is the only path a token's plaintext ever reaches the browser; only its hash is stored. */
export async function issueEnrollmentTokenAction(
  _prev: { token?: string } | undefined,
  _form: FormData,
): Promise<{ token: string }> {
  const { token } = issueEnrollmentToken();
  revalidatePath("/nodes");
  return { token };
}

export async function approveAgentAction(form: FormData) {
  const id = Number(form.get("id"));
  if (id) approveAgent(id);
  revalidatePath("/nodes");
  redirect("/nodes");
}

export async function denyAgentAction(form: FormData) {
  const id = Number(form.get("id"));
  if (id && !denyAgent(id)) {
    revalidatePath("/nodes");
    redirect("/nodes?agent=not-pending");
  }
  revalidatePath("/nodes");
  redirect("/nodes");
}

/** Revoke an active agent: deny it and invalidate its session token so it can no longer poll work. */
export async function revokeAgentAction(form: FormData) {
  const id = Number(form.get("id"));
  if (id) revoke(id);
  revalidatePath("/nodes");
  redirect("/nodes");
}

/** Enqueue a `run` job (model + prompt) for an active agent to pick up on its next work poll. */
export async function submitJobAction(form: FormData) {
  const agentId = Number(form.get("agentId"));
  const model = String(form.get("model") ?? "").trim();
  const prompt = String(form.get("prompt") ?? "").trim();
  if (!agentId || !model || !prompt) {
    revalidatePath("/nodes");
    redirect("/nodes?job=invalid");
  }
  let jobId: string;
  try {
    jobId = enqueue(agentId, "run", { model, prompt });
  } catch (error) {
    if (!(error instanceof AgentNotActiveError)) throw error;
    revalidatePath("/nodes");
    redirect("/nodes?job=not-active");
  }
  revalidatePath("/nodes");
  redirect(`/nodes?job=queued&jobId=${encodeURIComponent(jobId)}`);
}
