import { NextRequest, NextResponse } from "next/server";

import {
  AUTH_COOKIE,
  authCookieOptions,
  getBackendUrl,
  normalizeAuthBody,
} from "@/lib/auth-cookie";

export async function POST(request: NextRequest) {
  const raw = await request.json();
  const body = normalizeAuthBody(raw);
  if (!body.email || !body.password) {
    return NextResponse.json(
      { detail: "Email and password are required" },
      { status: 400 },
    );
  }

  const res = await fetch(`${getBackendUrl()}/auth/login/json`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    return NextResponse.json(data, { status: res.status });
  }

  const response = NextResponse.json({ ok: true });
  response.cookies.set(AUTH_COOKIE, data.access_token, authCookieOptions());
  return response;
}
