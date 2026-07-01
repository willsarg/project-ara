// SPDX-License-Identifier: Apache-2.0
"use server";
// Server Actions — login/logout and registry mutations. All run server-side; node tokens written
// here go straight into SQLite and are never returned to the caller.
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { revalidatePath } from "next/cache";
import { SESSION_COOKIE, createSession } from "@/lib/auth";
import { addNode, deleteNode, toggleNode, verifyAdminPassword } from "@/lib/db";
import { approveAgent, denyAgent, issueEnrollmentToken } from "@/lib/enrollment";
import { enqueue } from "@/lib/work";

const SESSION_TTL_S = 60 * 60 * 24 * 7;

export async function loginAction(_prev: { error?: string } | undefined, form: FormData) {
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
  (await cookies()).delete(SESSION_COOKIE);
  redirect("/login");
}

export async function addNodeAction(form: FormData) {
  const name = String(form.get("name") ?? "").trim();
  const base_url = String(form.get("base_url") ?? "").trim();
  const token = String(form.get("token") ?? "").trim();
  if (name && base_url && token) {
    addNode(name, base_url, token);
  }
  revalidatePath("/nodes");
  redirect("/nodes");
}

export async function deleteNodeAction(form: FormData) {
  const id = Number(form.get("id"));
  if (id) deleteNode(id);
  revalidatePath("/nodes");
  redirect("/nodes");
}

export async function toggleNodeAction(form: FormData) {
  const id = Number(form.get("id"));
  if (id) toggleNode(id);
  revalidatePath("/nodes");
  redirect("/nodes");
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
  if (id) denyAgent(id);
  revalidatePath("/nodes");
  redirect("/nodes");
}

/** Enqueue a `run` job (model + prompt) for an active agent to pick up on its next work poll. */
export async function submitJobAction(form: FormData) {
  const agentId = Number(form.get("agentId"));
  const model = String(form.get("model") ?? "").trim();
  const prompt = String(form.get("prompt") ?? "").trim();
  if (agentId && model) enqueue(agentId, "run", { model, prompt });
  revalidatePath("/nodes");
  redirect("/nodes");
}
