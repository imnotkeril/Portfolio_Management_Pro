import { NextRequest, NextResponse } from "next/server";

import { AUTH_COOKIE, getBackendUrl } from "@/lib/auth-cookie";
import { forwardedForHeaders } from "@/lib/client-ip";

export async function GET(request: NextRequest) {
  const token = request.cookies.get(AUTH_COOKIE)?.value;
  if (!token) {
    // Anonymous visitors are not an error: avoids a 401 in every public page load.
    return NextResponse.json(null, { status: 200 });
  }

  const res = await fetch(`${getBackendUrl()}/auth/me`, {
    headers: { Authorization: `Bearer ${token}` },
    cache: "no-store",
  });

  const data = await res.json().catch(() => ({}));
  return NextResponse.json(data, { status: res.status });
}

export async function DELETE(request: NextRequest) {
  const token = request.cookies.get(AUTH_COOKIE)?.value;
  if (!token) {
    return NextResponse.json({ detail: "Not authenticated" }, { status: 401 });
  }
  const raw = await request.json().catch(() => ({}));
  const password = typeof raw?.password === "string" ? raw.password : "";
  if (!password) {
    return NextResponse.json(
      { detail: "Password is required" },
      { status: 400 },
    );
  }

  const res = await fetch(`${getBackendUrl()}/auth/me`, {
    method: "DELETE",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      ...forwardedForHeaders(request),
    },
    body: JSON.stringify({ password }),
    cache: "no-store",
  });

  if (res.status !== 204) {
    const data = await res.json().catch(() => ({}));
    return NextResponse.json(data, { status: res.status });
  }
  const response = NextResponse.json({ ok: true });
  response.cookies.set(AUTH_COOKIE, "", {
    httpOnly: true,
    path: "/",
    maxAge: 0,
  });
  return response;
}
