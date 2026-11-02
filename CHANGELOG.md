# Changelog

## Unreleased (branch chore/audit)

### Security
- Production guards: `ENVIRONMENT=production` refuses `AUTH_DISABLED=true`, a default or short (<32) `JWT_SECRET`, and a short `CLIENT_IP_SIGNING_KEY`.
- Rate limiting: signed client IP from the BFF (`CLIENT_IP_SIGNING_KEY`), `TRUSTED_PROXY_COUNT` semantics, per-(email, IP) login failures, soft brake, one Monte Carlo run per user.
- Stripe webhooks: compatible with stripe-python 16 objects, state re-fetched from Stripe, expired periods do not grant Pro, failed handlers can be retried.
- Request size limits (Monte Carlo, tickers), bounded Monte Carlo response, API docs hidden and security headers in production, BFF proxy path allowlist.

### Fixed
- 404 instead of 500 for foreign or missing portfolios (transactions, ledger maintain).
- Buy-and-hold portfolios show positions on the overview; portfolio navigation no longer shows stale data; transaction form price rounding.
- `frontend/src/lib` was git-ignored; local SQLite files no longer baked into Docker images.

### Build and docs
- Pinned Python dependencies (`requirements.lock.txt`) and dev tools, non-root containers, frontend job in CI, Next 16.4.3.
- README rewritten with screenshots; USER_GUIDE for the Next.js app; release checklist `docs/RELEASE.md`.
