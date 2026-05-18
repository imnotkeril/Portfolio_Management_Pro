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

  const registerRes = await fetch(`${getBackendUrl()}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!registerRes.ok) {
    const err = await registerRes.json().catch(() => ({}));
    return NextResponse.json(err, { status: registerRes.status });
  }

  const loginRes = await fetch(`${getBackendUrl()}/auth/login/json`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: body.email, password: body.password }),
  });

  const loginData = await loginRes.json().catch(() => ({}));
  if (!loginRes.ok) {
    return NextResponse.json(
      {
        detail:
          typeof loginData?.detail === "string"
            ? `Account created but sign-in failed: ${loginData.detail}. Try logging in.`
            : "Account created but sign-in failed. Try logging in.",
      },
      { status: loginRes.status },
    );
  }

  const response = NextResponse.json({ ok: true, email: body.email });
  response.cookies.set(AUTH_COOKIE, loginData.access_token, authCookieOptions());
  return response;
}
