# Production roadmap

Phased delivery for **Portfolio Management Pro**. Use this folder as the source of truth.

## Product phases (1–6) — main launch plan

| Phase | Runbook | Theme | Status |
|-------|---------|--------|--------|
| **1** | [phase-01-postgresql.md](./phase-01-postgresql.md) | PostgreSQL, Docker, Alembic | Done |
| **2** | [phase-02-auth.md](./phase-02-auth.md) | JWT auth, `user_id` isolation | Done |
| **3** | [phase-03-transactions.md](./phase-03-transactions.md) | Transaction ledger, holdings, PnL | Done |
| **4** | [phase-04-frontend.md](./phase-04-frontend.md) | Next.js auth, dashboard, portfolio hub | Done |
| **5** | [phase-05-stripe.md](./phase-05-stripe.md) | **Stripe** Free/Pro, webhooks, limits | Done |
| **6** | [phase-06-deploy.md](./phase-06-deploy.md) | Production deploy: Vercel (web), Hugging Face Spaces (API), Supabase (Postgres); Railway was not used | Checklist (owner deploys) |

Overview: [ROADMAP.md](./ROADMAP.md) · Progress: [STATUS.md](./STATUS.md) · History: [CHANGELOG.md](./CHANGELOG.md)

---

## Technical runbooks (`phases/`) — engine parity & UX depth

Separate from product phase numbers; implement when the ledger/analytics stack needs alignment.

| Doc | Topic | Status |
|-----|--------|--------|
| [phases/phase-3-transaction-ledger.md](phases/phase-3-transaction-ledger.md) | Ledger replay detail | Done |
| [phases/phase-4-optimization-ledger-parity.md](phases/phase-4-optimization-ledger-parity.md) | Optimized vs current returns | Done |
| [phases/phase-5-strategies.md](phases/phase-5-strategies.md) | Rebalance targets UI/API (not Stripe) | Done |

> **Note:** UI copy “Strategy management — coming in Phase 5” referred to the **Strategies tab** feature, not product Phase 5 (Stripe). Stripe is [phase-05-stripe.md](./phase-05-stripe.md).

---

## Principles

1. **`core/`** framework-agnostic; **`services/`** orchestration; **`api/`** HTTP; **`frontend/`** (Next.js UI).
2. **One return engine per `ledger_mode`** (transactions → ledger builder; buy_hold → fixed shares).
3. **`user_id`** from JWT through portfolio and paid features.
4. **CI:** `core/` coverage ≥ 70%; Ruff, Black, isort, Mypy.

---

## Quick links

- [USER_GUIDE.md](../../USER_GUIDE.md)
- [README.md](../../README.md)
- [.env.example](../../.env.example) — includes `STRIPE_*`, `FRONTEND_URL`
