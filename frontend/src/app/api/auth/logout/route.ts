import { NextResponse } from "next/server";

import { AUTH_COOKIE } from "@/lib/auth-cookie";

export async function POST() {
  const response = NextResponse.json({ ok: true });
  response.cookies.set(AUTH_COOKIE, "", {
    httpOnly: true,
    path: "/",
    maxAge: 0,
  });
  return response;
}
