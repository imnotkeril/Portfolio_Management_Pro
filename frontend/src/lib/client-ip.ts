import { createHmac } from "node:crypto";

/**
 * Signed client IP for the API rate limiter (S-19).
 *
 * The BFF reads the visitor IP ONLY from a header the hosting platform
 * guarantees (Vercel overwrites `x-forwarded-for` / sets
 * `x-vercel-forwarded-for` and never forwards external values), then sends it
 * to the API as `X-Client-IP` + `X-Client-IP-Timestamp` +
 * `X-Client-IP-Signature` = hex HMAC-SHA256(CLIENT_IP_SIGNING_KEY,
 * "<ip>|<timestamp>"). The API trusts the IP only when the signature verifies.
 * Client-sent `x-forwarded-for` / `x-real-ip` are never trusted blindly.
 *
 * Without `CLIENT_IP_SIGNING_KEY`, or off Vercel without
 * `CLIENT_IP_TRUSTED_HEADER`, nothing is sent and the API falls back to the
 * socket peer address.
 */
const IP_RE = /^[0-9a-fA-F:.]{3,45}$/;

/** Name of the header this deployment may trust for the visitor IP. */
function trustedHeaderName(): string | null {
  const custom = process.env.CLIENT_IP_TRUSTED_HEADER?.trim().toLowerCase();
  if (custom) return custom;
  if (process.env.VERCEL) return "x-vercel-forwarded-for";
  return null;
}

export function trustedClientIp(headers: Headers): string | null {
  const name = trustedHeaderName();
  if (!name) return null;
  // Platform headers carry one address; if a list slips through, the first
  // entry is the one the platform recorded for the connecting client.
  const first = headers.get(name)?.split(",")[0]?.trim();
  return first && IP_RE.test(first) ? first : null;
}

export function signClientIp(
  ip: string,
  key: string,
  nowSeconds: number = Math.floor(Date.now() / 1000),
): Record<string, string> {
  const ts = String(nowSeconds);
  const sig = createHmac("sha256", key).update(`${ip}|${ts}`).digest("hex");
  return {
    "X-Client-IP": ip,
    "X-Client-IP-Timestamp": ts,
    "X-Client-IP-Signature": sig,
  };
}

/** Headers to add to an upstream API call so it can identify the visitor. */
export function forwardedForHeaders(request: {
  headers: Headers;
}): Record<string, string> {
  const key = process.env.CLIENT_IP_SIGNING_KEY;
  if (!key) return {};
  const ip = trustedClientIp(request.headers);
  return ip ? signClientIp(ip, key) : {};
}
