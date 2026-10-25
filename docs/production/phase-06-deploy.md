# Phase 6 — Deploy (release checklist)

**Actual production stack** (per root README): web on Vercel, API on Hugging Face Spaces (Docker SDK, root `Dockerfile`), Postgres on Supabase, Stripe in test mode. Railway (the original plan) is not used. Local: `docker compose up --build` (`Dockerfile.api`, `frontend/Dockerfile`).

This file is a checklist only; deploys are done by the owner.

## Release blockers (code side) — all closed in `chore/audit`
- [x] Production refuses default/short `JWT_SECRET` (<32) and `AUTH_DISABLED=true`.
- [x] Rate limit ignores spoofable `X-Forwarded-For` unless `TRUSTED_PROXY_COUNT` is set.
- [x] `/docs`, `/redoc`, `/openapi.json` hidden when `ENVIRONMENT=production`; security headers on API and Next.js.
- [x] npm audit (prod deps) clean; Python deps pinned in `requirements.lock.txt` (used as pip constraints in both Dockerfiles).
- [x] Containers run as non-root; CI runs ruff/black/isort/mypy/pytest and frontend eslint/tsc/build.

## Owner checklist before / at release
1. [ ] HF Spaces secrets: `ENVIRONMENT=production`, strong `JWT_SECRET` (>=32 chars), `DATABASE_URL` (Supabase), `FRONTEND_URL`, `CORS_ORIGINS` if more origins, `AUTH_DISABLED=false`, `DEBUG=false`, `TRUSTED_PROXY_COUNT` (number of trusted proxies in front of the API; HF Spaces = check actual hop count), Stripe vars.
2. [ ] Vercel env: `NEXT_PUBLIC_API_URL`, `API_INTERNAL_URL`, `COOKIE_SECURE=true`.
3. [ ] Stripe live/test: product + `STRIPE_PRICE_ID_PRO`; webhook -> `https://<api>/stripe/webhook` (checkout.session.completed, invoice.paid, customer.subscription.*), set `STRIPE_WEBHOOK_SECRET`.
4. [ ] Custom domain + TLS (Vercel / HF).
5. [ ] Supabase backups (PITR or scheduled dump) and a restore drill.
6. [ ] Monitoring: uptime check on `/health`, error tracking (e.g. Sentry), log retention.
7. [ ] CI green on `main` before deploy; migrations run on API start (`alembic upgrade head` in the image CMD) — keep migrations forward-compatible.
8. [ ] Smoke test: register -> create portfolio -> add transaction -> analytics -> checkout (test card) -> Pro unlock -> logout/login.
9. [ ] Rollback: redeploy the previous Space revision / Vercel deployment; never run destructive migrations.
