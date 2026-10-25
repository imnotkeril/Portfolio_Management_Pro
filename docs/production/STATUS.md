# Production status

**Last updated:** 2026-10-25

---

## Product roadmap (phases 1–6)

| Phase | Status | Notes |
|-------|--------|--------|
| 1 PostgreSQL | ✅ | Docker Compose, Alembic |
| 2 Auth | ✅ | JWT, portfolio `user_id` |
| 3 Transactions | ✅ | Ledger, DIVIDEND/SPLIT, maintenance |
| 4 Frontend | ✅ | BFF cookie, middleware, dashboard, portfolio hub |
| **5 Stripe** | ✅ | See below |
| 6 Deploy | 🟡 | Live on Vercel + HF Spaces + Supabase; code-side blockers closed (2026-10-25), owner checklist open |

**Next:** owner items in [phase-06-deploy.md](./phase-06-deploy.md) (Stripe webhook secret, domain/TLS, backups, monitoring)

---

## Phase 5 — Stripe subscriptions ✅

Runbook: [phase-05-stripe.md](./phase-05-stripe.md)

| Area | Delivered |
|------|-----------|
| **Models** | `subscriptions`, `stripe_events`; migration `011_subscriptions_stripe` |
| **Register** | Free subscription row on signup |
| **API** | `GET /billing/status`, `POST /billing/checkout`, `POST /billing/portal`, `POST /stripe/webhook` |
| **Enforcement** | `require_pro` on `/optimization/*`, `/forecasting/*`; Free max 1 portfolio |
| **Settings** | `STRIPE_*`, `FRONTEND_URL`, `BILLING_ENFORCEMENT` in `.env.example` |
| **Frontend** | `/billing` — plan badge, upgrade, customer portal |
| **Tests** | `test_billing_api.py`, `test_billing_service.py` |

**Manual:** Stripe Dashboard product + `STRIPE_PRICE_ID_PRO`; local webhooks via Stripe CLI.

---

## Technical phases (ledger / strategies)

| Doc | Status |
|-----|--------|
| [phases/phase-4-optimization-ledger-parity.md](phases/phase-4-optimization-ledger-parity.md) | ✅ |
| [phases/phase-5-strategies.md](phases/phase-5-strategies.md) | ✅ (rebalance UI; optional) |

---

## CI / quality

```bash
pytest tests/unit/test_billing_service.py tests/integration/test_billing_api.py -q --no-cov
pytest tests/unit/test_strategy_service.py tests/integration/test_strategy_api.py -q --no-cov
```
