/**
 * Client IP as seen by this BFF, forwarded to the API as `X-Forwarded-For`
 * so the API rate limiter keys on the real visitor instead of the BFF egress
 * address. The API honours it only per `TRUSTED_PROXY_COUNT`.
 *
 * Preference: `x-real-ip` (set by the platform), else the LAST entry of
 * `x-forwarded-for` (the one appended by the proxy closest to us; entries to
 * its left are client-controlled). Returns exactly one address or null.
 */
const IP_RE = /^[0-9a-fA-F:.]{3,45}$/;

export function clientIpFromHeaders(headers: Headers): string | null {
  const real = headers.get("x-real-ip")?.trim();
  if (real && IP_RE.test(real)) return real;
  const parts = (headers.get("x-forwarded-for") ?? "")
    .split(",")
    .map((p) => p.trim())
    .filter(Boolean);
  const last = parts[parts.length - 1];
  return last && IP_RE.test(last) ? last : null;
}

/** Headers to add to an upstream API call so it can identify the visitor. */
export function forwardedForHeaders(request: {
  headers: Headers;
}): Record<string, string> {
  const ip = clientIpFromHeaders(request.headers);
  return ip ? { "X-Forwarded-For": ip } : {};
}
