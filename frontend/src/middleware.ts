import { NextRequest, NextResponse } from "next/server";

import { AUTH_COOKIE } from "@/lib/auth-cookie";

const PUBLIC_PAGES = new Set(["/", "/login", "/register"]);

const PROTECTED_PREFIXES = [
  "/dashboard",
  "/portfolio",
  "/portfolios",
  "/create",
  "/analysis",
  "/optimization",
  "/optimization-studio",
  "/opti",
  "/risk",
  "/forecasting",
  "/settings",
];

function isProtectedPage(pathname: string): boolean {
  return PROTECTED_PREFIXES.some(
    (prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`),
  );
}

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const token = request.cookies.get(AUTH_COOKIE)?.value;

  if (pathname.startsWith("/api/")) {
    return NextResponse.next();
  }

  if (PUBLIC_PAGES.has(pathname)) {
    if (token && (pathname === "/login" || pathname === "/register")) {
      return NextResponse.redirect(new URL("/dashboard", request.url));
    }
    return NextResponse.next();
  }

  if (isProtectedPage(pathname) && !token) {
    const login = new URL("/login", request.url);
    login.searchParams.set("from", pathname);
    return NextResponse.redirect(login);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico|.*\\..*).*)"],
};
