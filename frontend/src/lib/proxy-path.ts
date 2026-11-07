/**
 * Path validation for the BFF proxy (/api/proxy/[...path]).
 *
 * Only first path segments that exist as backend routes are forwarded
 * (derived from api/main.py + api/routers). `auth` and `stripe` are
 * intentionally absent: auth has dedicated Next routes that manage the
 * cookie, and the Stripe webhook must never be reachable from the browser.
 */
export const PROXY_ALLOWED_ROOTS: ReadonlySet<string> = new Set([
  "health",
  "dashboard",
  "validate-tickers",
  "ticker-price",
  "portfolios",
  "transactions",
  "analytics",
  "optimization",
  "risk",
  "forecasting",
  "billing",
]);

function decodeFully(segment: string): string | null {
  let current = segment;
  // Decode repeatedly so double-encoded traversal ("%252e%252e") is caught too.
  for (let i = 0; i < 3; i++) {
    let next: string;
    try {
      next = decodeURIComponent(current);
    } catch {
      return null; // malformed escape
    }
    if (next === current) return current;
    current = next;
  }
  return current;
}

/**
 * Returns the validated, re-encoded path ("portfolios/abc/holdings"),
 * or null when the request must be rejected with 404.
 */
export function sanitizeProxyPath(segments: string[]): string | null {
  if (segments.length === 0) return null;
  if (!PROXY_ALLOWED_ROOTS.has(segments[0])) return null;

  const safe: string[] = [];
  for (const raw of segments) {
    const decoded = decodeFully(raw);
    if (decoded === null || decoded === "" || decoded === "." || decoded === "..") {
      return null;
    }
    // No separators, control chars, or query/fragment delimiters inside a segment.
    if (/[\/\u0000-\u001f\u007f?#]/.test(decoded)) return null;
    safe.push(encodeURIComponent(decoded));
  }
  return safe.join("/");
}
