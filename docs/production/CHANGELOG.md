# Production changelog

Record of phase deliveries (high level). Detail lives in `phases/*.md`.

## 2026-10-25 — Audit hardening (branch chore/audit)

- Security: production guards (JWT_SECRET >= 32 chars, AUTH_DISABLED refused), trusted-proxy rate limiting (`TRUSTED_PROXY_COUNT`), docs hidden in production, security headers (API + Next.js), BFF proxy path allowlist.
- Fixes: 404 instead of 500 for transactions of foreign/missing portfolios; buy-and-hold positions shown on portfolio overview; `frontend/src/lib` was git-ignored and is now tracked.
- Build: pinned `requirements.lock.txt`, non-root containers, Next 16.4.3 (npm audit clean), frontend job in CI (eslint, tsc, build); core coverage 78%.

## 2026-05-22 — Product Phase 5: Stripe subscriptions

- `Subscription` + `stripe_events` models, Alembic `011_subscriptions_stripe`.
- Checkout, Customer Portal, signed webhooks; `require_pro` on optimization/forecasting.
- Free tier: 1 portfolio; `/billing` page in Next.js.
- Docs: unified `README.md` (product 1–6 vs technical `phases/`).

## 2026-05-22 — Technical: Strategy management (rebalance UI)


- `StrategyService` + `GET/PUT /portfolios/{id}/strategy` + `POST .../strategy/preview`.
- Shared `extract_target_weights`; Next.js and Streamlit Strategies tabs.
- Optimization bundle `rebalance_policy` labels for current vs optimized tracks.
- Tests: `test_strategy_service.py`, `test_strategy_api.py`.

## 2026-05-22 — Phase 4: Optimization ledger parity

- Synthetic optimized ledger for `ledger_mode=transactions` (cashflows + rebalance schedule + optimized targets).
- `plan_rebalance_to_target_weights` and helpers in `rebalance_service.py`.
- `user_id` through optimization / risk / forecasting services and UI bundles.
- Notebook split disabled for transaction portfolios in optimization bundle.
- Max drawdown comparison wording fix in optimization bundle.
- Docs: `phases/phase-4-optimization-ledger-parity.md`, `STATUS.md`.

## Earlier

- **Phase 3:** Transaction ledger, API, ledger analytics, maintenance.
- **Phase 2:** JWT auth, `AUTH_DISABLED`, portfolio scoping.
- **Phase 1:** PostgreSQL + Docker Compose stack.
