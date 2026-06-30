// SPDX-License-Identifier: Apache-2.0
"use server";
// Server Actions — login/logout and registry mutations. All run server-side; node tokens written
// here go straight into SQLite and are never returned to the caller.
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { revalidatePath } from "next/cache";
import { SESSION_COOKIE, createSession } from "@/lib/auth";
import { addNode, deleteNode, getAdminPassword, toggleNode } from "@/lib/db";

const SESSION_TTL_S = 60 * 60 * 24 * 7;

export async function loginAction(_prev: { error?: string } | undefined, form: FormData) {
  const password = String(form.get("password") ?? "");
  if (password !== getAdminPassword()) {
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
