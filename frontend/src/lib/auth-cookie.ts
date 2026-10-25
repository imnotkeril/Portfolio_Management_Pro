/** HttpOnly JWT cookie set by Next.js auth routes (BFF). */

export const AUTH_COOKIE = "access_token";

export const AUTH_COOKIE_MAX_AGE = 60 * 60 * 24;

/** Set COOKIE_SECURE=true only behind HTTPS (local Docker uses false). */
export function authCookieOptions() {
  return {
    httpOnly: true,
    secure: process.env.COOKIE_SECURE === "true",
    sameSite: "lax" as const,
    path: "/",
    maxAge: AUTH_COOKIE_MAX_AGE,
  };
}

export function normalizeAuthBody(body: {
  email?: string;
  password?: string;
}): { email: string; password: string } {
  return {
    email: String(body.email ?? "")
      .trim()
      .toLowerCase(),
    password: String(body.password ?? ""),
  };
}

/** Server-side FastAPI base URL (Docker: http://api:8000). */
export function getBackendUrl(): string {
  return (
    process.env.API_INTERNAL_URL ??
    process.env.NEXT_PUBLIC_API_URL ??
    "http://localhost:8000"
  );
}

/** Paths proxied without a session cookie (public backend routes). */
export function isPublicProxyPath(pathStr: string): boolean {
  if (pathStr === "health" || pathStr === "dashboard/indices") return true;
  if (pathStr.startsWith("ticker-price/")) return true;
  if (pathStr === "validate-tickers") return true;
  if (pathStr === "risk/scenarios") return true;
  return false;
}
