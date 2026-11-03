# RELEASE checklist - Portfolio Management Pro

Date: 2026-11-01. Branch: `chore/audit`. Stack: Vercel (Next.js BFF, `frontend/`) + Hugging Face Spaces (FastAPI, root `Dockerfile`, port 7860) + Supabase Postgres + Stripe.
Legend: `[x]` = verified locally in this step (evidence given); `[ ]` = owner or hosting action, NOT done and NOT verifiable by the agent. Nothing was pushed, deployed or changed on any hosting account. No `.env*` file was read; live keys were never seen.

## 0. Verdict

Code side: release candidate, but the QA verdict is still **RED** (docs/QA.md, "Verdict after the Resolution"): F-5 (no account-deletion endpoint, owner decision) and the F-2/F-3 fixes not yet checked in a browser. Blockers before switching to live money are in section 1; most are owner actions or decisions, not code defects.

## 1. Blockers and owner decisions

| # | Item | Status |
|---|---|---|
| B1 | Live production URLs were unhealthy at the last check (DOCS_CHECK, 2026-10-25): Vercel root returned 404, HF `/health` returned 503. Re-check now; a sleeping or stopped Space or a missing Vercel deploy is the likely cause. Not re-checked in this step (no hosting access). | open |
| B2 | Production secrets on HF/Vercel were never confirmed (strong `JWT_SECRET`, `ENVIRONMENT=production`, `AUTH_DISABLED=false`). The app refuses to boot otherwise (verified, section 3), so a wrong value shows up as a Space that does not start. | owner confirms |
| B3 | Stripe is still test mode. Live setup (section 5) is the owner's. Do not enable live keys before the smoke test (section 9) passes in test mode on the production URLs. | open |
| B4 | No error tracking or uptime alerting exists. Webhook failures are silent unless Stripe dashboard alerts are enabled (section 7). Needed before live money. | open |
| B5 | Supabase backups and a restore drill have not been done (section 6). Required before real customer data. | open |
| D1 | S-19 (fixed in code, a79dec8): the BFF signs the client IP with `CLIENT_IP_SIGNING_KEY`; set the same key on Vercel and HF and keep `TRUSTED_PROXY_COUNT=0`. The API now refuses to boot in production without the key (S-23 closed, fail-closed); a mismatched key makes the API fall back to the socket peer, which (behind the BFF) puts all users into one login bucket. Live proof of S-19 under this config is still to do (QA). | owner sets key |
| D2 | The rate limiter is in-process memory (S-04 residual). The HF Space runs one uvicorn process (CMD has no `--workers`), so limits are correct only with exactly one process and one replica. Do not scale to several workers/replicas before moving the limiter to shared storage (Postgres/Redis). The Monte Carlo concurrency semaphore (2) is also per process. | decide / constraint |
| D3 | Other open P2 in `docs/SECURITY_REVIEW.md`: S-05 user enumeration (register returns 409, login timing 0.18 s vs 0.006 s); S-06 open redirect after login (`from` param); S-07 `/validate-tickers` slow with bogus tickers (retries and sleeps; list capped at 100); S-09 no Origin check on BFF mutations (login CSRF, only `SameSite=Lax`); S-10 cookie `Secure` is opt-in (`COOKIE_SECURE=true` MUST be set on Vercel); S-11 no checkout guard or idempotency, and a double-subscription downgrade case (a double click can double-charge). P3: S-12..S-17 (S-17: `ENVIRONMENT` fails open, a typo such as `prod` means development behaviour; set exactly `production`). Owner decides what to fix before live billing; S-11 is the one to fix before charging real cards. | decide |
| D4 | `docs/` is gitignored; this file and other docs are force-added. Decide whether to un-ignore. | decide |
| D5 | README SQLite quick start does not boot (documented). `/llms.txt` is served since `b5f1a3c` (52 operations; test not run yet). Not release blockers. | note |

Test-suite state (final run on 6bf0aeb): 448 passed, 0 skipped, core coverage 78.35%. S-03 remainder and S-18 (970b581, 338748f) were independently re-checked in SECURITY_REVIEW round 2; S-19 (a79dec8) in round 3 by code review plus QA live rate-limit rows (19/19). After `6bf0aeb`, `b5f1a3c` (fail-closed S-23, `/llms.txt`) added tests that have not been run yet.

## 2. Environment variable table (names only)

Sources: `config/settings.py`, `os.getenv`/`os.environ` in code (`CORS_ORIGINS`, `DATABASE_URL`, `RATE_LIMIT_ENABLED`), `process.env` in `frontend/src`, `docker-compose.yml`, `.env.example`. Values are never recorded here.

### 2.1 API (HF Spaces secrets/variables; local = `.env`; compose = `docker-compose.yml`)

| Name | Local dev | Compose | Production (HF Spaces) | Notes |
|---|---|---|---|---|
| `ENVIRONMENT` | `development` | default | `production` (exact) | Enables boot guards, hides `/docs`, HSTS. A typo falls back to dev behaviour (S-17). |
| `JWT_SECRET` | any | dev value | secret, >= 32 chars, random | Generate: `python -c "import secrets; print(secrets.token_urlsafe(64))"`. Rotating it logs everyone out. |
| `JWT_ALGORITHM` | `HS256` | default | default | |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `1440` | default | default | |
| `AUTH_DISABLED` | `false` | default | `false` | `true` is refused in production. |
| `DATABASE_URL` | SQLite / local PG | compose PG | Supabase string, `postgresql+psycopg2://...` | Secret. See 2.4. SQLite does not boot. |
| `DATABASE_ECHO` | `false` | default | `false` | |
| `DEBUG` | `false` | default | `false` | `true` leaks webhook error text. |
| `TRUSTED_PROXY_COUNT` | `0` | `0` | `0` (with the signed client IP from 2.2; any value >0 re-opens S-19 for direct API callers) | |
| `RATE_LIMIT_ENABLED` | `true` | default | `true` | Only tests turn it off. |
| `CORS_ORIGINS` | unset | unset | optional, comma-separated extra origins | `FRONTEND_URL` and localhost:3000 are always allowed. The browser talks to the BFF, so CORS is a safety net. |
| `FRONTEND_URL` | `http://localhost:3000` | same | `https://<vercel-domain>` (no trailing slash) | Used for Stripe return URLs and CORS. |
| `STRIPE_SECRET_KEY` | `sk_test_...` | pass-through | live secret key (secret) | |
| `STRIPE_WEBHOOK_SECRET` | `whsec_...` (stripe listen) | pass-through | `whsec_...` of the LIVE endpoint (secret) | Missing -> webhook returns 503 (verified). |
| `STRIPE_PRICE_ID_PRO` | `price_...` | pass-through | live `price_...` | Not secret, but differs between test and live. |
| `BILLING_ENFORCEMENT` | `true` | `true` | `true` | `false` removes the Pro gates (everyone gets Pro features). Never false in production. |
| `LOG_LEVEL`, `LOG_FILE` | `INFO`, `logs/app.log` | `INFO` | `INFO` | Container disk is ephemeral on HF. |
| `PRICE_CACHE_DIR` | `data/cache/prices` | default | default | Ephemeral on HF; cache is rebuilt. |
| `PORT` | n/a | n/a | `7860` | Image CMD uses `${PORT:-7860}`; do not change on HF. |
| Unused by code (`[unused]` in `.env.example`) | `APP_NAME`, `APP_VERSION`, `ALPHA_VANTAGE_API_KEY`, `IEX_CLOUD_API_KEY`, `YAHOO_FINANCE_ENABLED`, `CACHE_ENABLED`, `CACHE_TTL_SECONDS`, `MAX_WORKERS`, `CALCULATION_TIMEOUT_SECONDS`, `RISK_FREE_RATE`, `MAX_PORTFOLIO_POSITIONS`, `MIN_POSITION_WEIGHT`, `MAX_POSITION_WEIGHT` | | do not set | Compose also sets `CACHE_DIR`, which nothing reads. |

### 2.2 Web (Vercel project, root directory `frontend/`)

| Name | Local dev | Compose | Production (Vercel) | Notes |
|---|---|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | build arg | the HF API URL (README: `https://imnotkeril-portfolio-management-pro-api.hf.space`; confirm) | Build-time and public: redeploy after changing. Never put a secret here. |
| `API_INTERNAL_URL` | `http://localhost:8000` | `http://api:8000` | same HF URL | Server-side base URL the BFF proxy forwards to. |
| `CLIENT_IP_SIGNING_KEY` | unset | unset | random >= 32 chars, same value as on HF | Server-side secret; required on HF, >= 32 chars (the API refuses to boot in production if it is unset or shorter). Vercel is detected via `VERCEL` and reads `x-vercel-forwarded-for` (Vercel overwrites it; docs: vercel.com/docs/headers/request-headers). Without it the API rate-limits on the BFF address (S-19). |
| `CLIENT_IP_TRUSTED_HEADER` | unset | unset | unset on Vercel | Only for non-Vercel hosts: a header the platform sets and clients cannot forge. |
| `COOKIE_SECURE` | unset / `false` | `false` | `true` | Without it the JWT cookie can travel over HTTP (S-10). |
| `NODE_ENV` | by Next | | `production` (Vercel sets it) | |
| Sentry (module error-monitoring, if added) | | | `NEXT_PUBLIC_SENTRY_DSN`, `SENTRY_AUTH_TOKEN`, `SENTRY_ORG`, `SENTRY_PROJECT` | Not wired in code today. |

Never set on Vercel: any `STRIPE_*`, `JWT_SECRET`, `DATABASE_URL`. The BFF does not need them.

### 2.3 Client IP and TRUSTED_PROXY_COUNT

Set `CLIENT_IP_SIGNING_KEY` to the same random value on Vercel and on HF (and optionally `CLIENT_IP_MAX_AGE_SECONDS`, default 60). The BFF signs the visitor IP; the API trusts `X-Client-IP` only with a valid signature and ignores `X-Forwarded-For` / `X-Real-IP` from clients, so with the key set keep `TRUSTED_PROXY_COUNT=0` on HF. The text below applies only if you rely on the unsigned fallback instead.

The API takes the client IP as the Nth entry from the right of `X-Forwarded-For` (0 = ignore the header and use the socket peer). The Vercel BFF sets `X-Forwarded-For` to the visitor IP; each proxy after it appends one entry.
- BFF -> API directly (compose, self-host): `1`.
- BFF -> HF Spaces proxy -> API (this production): documented as `2`.
- Too low: all users share one proxy IP, so IP limits throttle everyone together.
- Too high: direct callers of the public API can choose their own IP key (S-19).
- The real HF hop count has never been measured in this repo. Procedure after deploy: start with `2`; send 25 bad logins from one IP through the BFF (expect 429 after 20 per minute); then from a direct client send requests with a rotating `X-Forwarded-For` header and see whether the limit key changes (if it does, the value is too high for direct traffic, which is the S-19 tradeoff). Record the measured value: ______.

### 2.4 Supabase connection

Migrations run on every boot (`alembic upgrade head`), so prefer the direct or session-mode connection string in `DATABASE_URL` over the transaction pooler (port 6543). Include `sslmode=require`. Not verified against Supabase (no account access).

## 3. Local verification (this step)

Image: root `Dockerfile` (the HF image), built as `audit-pmp-rel-hf:test` (build exit 0; the only warning is `JSONArgsRecommended` on the shell-form CMD, intentional because of `alembic ... && exec uvicorn`). Throwaway postgres:16-alpine and API containers `audit-pmp-rel-*` on a private docker network, host ports 29589 (pg) and 27272 (api, container port 7860), both on 127.0.0.1. `PORT=7860`, `ENVIRONMENT=production`, a generated 64-char `JWT_SECRET` (not printed).

- [x] Boot against empty Postgres: `GET /health` -> `200`.
- [x] Migrations ran on boot: log shows `001_create_price_history` through `011_subscriptions_stripe`; `select version_num from alembic_version` -> `011_subscriptions_stripe`.
- [x] Runs non-root: `id` -> `uid=1000(user) gid=1000(user)`.
- [x] `/docs`, `/redoc`, `/openapi.json` -> 404 in production.
- [x] Headers on `/health`: `strict-transport-security: max-age=31536000; includeSubDomains`, `x-content-type-options: nosniff`, `x-frame-options: DENY`, `referrer-policy: strict-origin-when-cross-origin`.
- [x] No `*.db` or `.env` file inside the image (`find / -xdev ...` returned nothing).
- [x] Webhook route exists: `GET /stripe/webhook` -> 405 (POST only); unsigned `POST` with no `STRIPE_WEBHOOK_SECRET` -> 503.
- [x] Refuses weak secret: `JWT_SECRET=short-secret` -> `RuntimeError: JWT_SECRET must be at least 32 characters in production.`; the container with the real CMD exited with `exit=1`.
- [x] Refuses default secret: `RuntimeError: JWT_SECRET is still the insecure default...`.
- [x] Refuses `AUTH_DISABLED=true`: `RuntimeError: AUTH_DISABLED=true is not allowed when ENVIRONMENT=production.`
- [ ] Not verified: register/login against this container (one probe returned 422 for my payload and was not retried); Stripe flows (no keys); Supabase connectivity; HF proxy behaviour; Vercel build.
- Cleanup: containers, network and image `audit-pmp-rel-*` removed; `docker ps -a --filter name=audit-pmp-rel` is empty.

Not run in this step (required by `process/release-prep.md`, still `[ ]`): `node scripts/secrets-scan.mjs --project . --history`, `pip-audit`, `npm audit --omit=dev`, license check, Lighthouse, clean-clone full build. Last recorded earlier: npm audit 0 vulns, ruff/black/isort/mypy clean, eslint 0 errors, tsc and `next build` OK (docs/STATE.md).

## 4. HF Spaces and Vercel setup (owner)

HF Spaces (Docker SDK, repo root `Dockerfile`):
1. [ ] Space settings -> Variables and secrets: set the Production column of table 2.1. Mark `JWT_SECRET`, `DATABASE_URL`, `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET` as Secrets.
2. [ ] `app_port: 7860` in the Space README front matter (default for the Docker SDK) if not already set.
3. [ ] The Space must be public: Stripe has to reach `/stripe/webhook` without a token.
4. [ ] Free Spaces sleep after inactivity, so the first request after sleep fails or is slow (and a webhook delivery may time out). Use an uptime ping (section 7) or an always-on hardware tier before taking payments.
5. [ ] After variables change, the Space restarts and runs `alembic upgrade head`, then uvicorn. Watch the logs for the migrations and `Application startup complete`.

Vercel:
1. [ ] Project root directory `frontend/`; set table 2.2 for Production (and Preview if used).
2. [ ] `COOKIE_SECURE=true`, then redeploy.
3. [ ] Confirm `https://<vercel-domain>/` returns 200 and `/login` renders (B1).

## 5. Stripe live setup (owner does all of it; the agent never sees live keys)

1. [ ] Activate the Stripe account: business verification, payout bank details, support email, statement descriptor.
2. [ ] Tax: decide on Stripe Tax or manual VAT/sales tax for the seller country; check Stripe availability for that country on stripe.com/global.
3. [ ] Live mode: create Product "Pro" with a recurring Price. Put the live `price_...` id into HF `STRIPE_PRICE_ID_PRO`.
4. [ ] Customer portal: activate it in LIVE mode (the app creates portal sessions; live portal config is separate from test).
5. [ ] Live API key: restricted key if possible (Checkout Sessions, Customers, Subscriptions, Billing Portal) or the secret key; set `STRIPE_SECRET_KEY` in HF secrets only.
6. [ ] Webhook endpoint (live mode): URL `https://<hf-api-host>/stripe/webhook` (the HF API host, NOT the Vercel URL: the BFF proxy blocks this path). Events to send, exactly those the handler uses:
   - `checkout.session.completed`
   - `invoice.paid`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
   Any other event is logged as unhandled and ignored. Copy the endpoint signing secret `whsec_...` into HF `STRIPE_WEBHOOK_SECRET`.
7. [ ] Terms, refund policy, seller details and privacy policy published and reviewed by a human (the agent's text is not legal advice); linked in the footer and in the Stripe account settings.
8. [ ] Checkout branding (name, logo), receipt emails on.
9. [ ] One live end-to-end purchase with a real card, then refund it, before announcing.
10. [ ] Keep test and live separate. Mixing a live key with a test price (or the reverse) makes checkout fail; `stripe_configured` only checks the key format.

Webhook behaviour (read from `services/billing_service.py`, not run live): signature verified on the raw body; event ids deduped in `stripe_events`; every handler re-fetches the subscription from Stripe; a handler failure un-marks the event and returns 400, so Stripe retries with backoff. Missing secret -> 503.

## 6. Database: Supabase backups and restore test

Owner actions:
1. [ ] Check the Supabase plan. The free plan gives no PITR and no guaranteed downloadable backups; paid plans have daily backups and PITR as an add-on. Choose by tolerated data loss. For real customer money data, use daily backups at minimum.
2. [ ] Independent dump on a schedule outside Supabase (weekly at least, daily once money flows), from the DIRECT connection string in plain `postgresql://` form (not the `+psycopg2` form):
   `pg_dump --format=custom --no-owner --no-privileges "$DATABASE_URL_DIRECT" --file pmp-YYYY-MM-DD.dump`
   Do not redirect binary output with `>` in PowerShell (known trap); use `--file`, or dump inside a container and `docker cp` it out. Store dumps encrypted, off Supabase and outside the repo.
3. [ ] Restore test (once before launch, then quarterly), on a throwaway local Postgres, never on production:
   ```
   docker run -d --name pmp-restore-test -e POSTGRES_PASSWORD=x -p 127.0.0.1:25432:5432 postgres:16-alpine
   docker cp pmp-YYYY-MM-DD.dump pmp-restore-test:/tmp/
   docker exec pmp-restore-test createdb -U postgres pmp
   docker exec pmp-restore-test pg_restore -U postgres -d pmp --no-owner /tmp/pmp-YYYY-MM-DD.dump
   docker exec pmp-restore-test psql -U postgres -d pmp -c "select version_num from alembic_version" -c "select count(*) from users" -c "select count(*) from portfolios" -c "select count(*) from subscriptions"
   ```
   Pass: `version_num` is the current head (`011_subscriptions_stripe`), counts are in the expected range, and the API boots against the restored DB (`/health` 200). Record date, dump size, restore time: ______. Remove the container afterwards.
4. [ ] If PITR is enabled, drill a restore into a separate Supabase project.
5. [ ] Write down RPO / RTO: ______ / ______.

Stripe is the source of truth for billing state. After a DB restore, reconcile: `GET /billing/status` re-syncs per user, or resend webhook events from the Stripe dashboard for the lost window.

## 7. Monitoring and alerting (suggestions; none is set up today)

- [ ] Uptime: external ping (UptimeRobot / Better Stack free tier) on `https://<api>/health` every 5 minutes (also keeps a free Space awake) and on the Vercel root. Alert by email or Telegram.
- [ ] Stripe webhook failures MUST alert: Dashboard -> Developers -> Webhooks -> endpoint -> enable failure email alerts, and review "Event deliveries" weekly. A 503 from the endpoint means `STRIPE_WEBHOOK_SECRET` is missing; 400 means signature or handler failure.
- [ ] Reconciliation (weekly, manual or scripted): Stripe active subscriptions vs `subscriptions` rows with `plan=pro`, `status=active`.
- [ ] Error tracking: Sentry for FastAPI and Next.js; production DSN in hosting env only; the privacy policy names the processor. Not integrated in code today (a code task).
- [ ] Logs: HF Space logs (short retention) and Vercel runtime logs; `logs/app.log` is on an ephemeral disk. Add a log drain if retention is needed. Logs were not audited for PII in this step.
- [ ] Supabase: database size and connection alerts; backup status check.
- [ ] Stripe: Radar defaults, email alerts for disputes and failed payouts.

## 8. Migration policy and rollback

Migration policy:
- Alembic migrations `001..011` run automatically on every API boot (`alembic upgrade head && exec uvicorn ...`), so each deploy changes the production schema before serving.
- Forward-compatible and additive only: new nullable columns and tables, backfills as separate steps; no `DROP` or new `NOT NULL` without a two-step release (expand, migrate, contract). Destructive migrations need the owner's explicit yes and a fresh dump taken first.
- Take a `pg_dump` (section 6) before any deploy that includes a new migration.
- Two replicas booting at once can race on `alembic upgrade`; stay on one Space replica (also required by D2).

Rollback:
- Web: Vercel -> Deployments -> previous deployment -> Promote to Production (instant). `NEXT_PUBLIC_*` values are baked per deployment.
- API: revert the commit in the HF Space repo (or restore the previous revision) and restart. `alembic downgrade` is NOT run automatically, so older code must run on the newer schema (additive rule). If a migration itself is bad: stop traffic, restore the pre-deploy dump (section 6), then redeploy the previous revision.
- Config: note the old value before editing any env var. To disable payments quickly, remove `STRIPE_SECRET_KEY` (checkout reports not configured); do not set `BILLING_ENFORCEMENT=false` (that gives everyone Pro).
- Trigger: `/health` failing for more than 5 minutes after a deploy, webhook 5xx, elevated error rate, or a failed smoke test (section 9).

## 9. Domain, TLS and smoke test

Domain/TLS (owner):
1. [ ] Custom domain on Vercel (DNS record at the registrar); TLS is automatic. Update HF `FRONTEND_URL` (and `CORS_ORIGINS` if used) to the final origin and fix the README links.
2. [ ] The API stays on the `*.hf.space` host (TLS by HF) unless a custom domain is set up elsewhere; the Stripe webhook URL must use the API host.
3. [ ] Change `NEXT_PUBLIC_API_URL` only if the API host changes, then redeploy Vercel.
4. [ ] The API sends HSTS (verified locally); confirm the web host sends it too once the domain is final.

Manual smoke test, in order: register -> login -> create portfolio -> add transaction -> analytics -> checkout (test card `4242 4242 4242 4242` in test mode; a real card then refund in live) -> Pro unlock (optimization/forecasting) -> customer portal opens -> cancel -> back to Free -> logout/login.

API smoke script (bash; replace the two hosts; read-only checks, creates nothing):
```
API=https://<api-host>; WEB=https://<web-host>
curl -fsS $API/health                                                     # 200 {"status":"ok"}
test "$(curl -s -o /dev/null -w '%{http_code}' $API/docs)" = 404 && echo "docs hidden"
curl -sI $API/health | grep -i strict-transport-security
test "$(curl -s -o /dev/null -w '%{http_code}' $WEB/login)" = 200 && echo "web login 200"
test "$(curl -s -o /dev/null -w '%{http_code}' -X POST $WEB/api/proxy/stripe/webhook)" != 200 && echo "webhook not exposed via BFF"
test "$(curl -s -o /dev/null -w '%{http_code}' -X POST $API/stripe/webhook -d '{}')" = 400 && echo "webhook rejects unsigned"   # 503 = STRIPE_WEBHOOK_SECRET missing
# Stripe CLI (test mode only): stripe trigger invoice.paid -> dashboard shows a 2xx delivery
```
The register/login/portfolio steps are not scripted: the one register probe in this step returned 422 and the payload shape was not confirmed.

## 10. Live-key swap list (owner)

| Where | Test -> Live |
|---|---|
| HF `STRIPE_SECRET_KEY` | test secret key -> live secret (or restricted) key |
| HF `STRIPE_PRICE_ID_PRO` | test price -> live price |
| HF `STRIPE_WEBHOOK_SECRET` | test endpoint secret -> live endpoint secret |
| HF `FRONTEND_URL` | vercel.app -> custom domain |
| Vercel `COOKIE_SECURE` | must be `true` |
| Stripe dashboard | webhook endpoint created separately in live mode; portal activated in live mode |

## 11. Other owner items (module inserts)

- [ ] Legal pages (privacy, terms, refund, company details) reviewed by a human; the processors list matches reality: Vercel, Hugging Face, Supabase, Stripe, Yahoo Finance data. **Checked 2026-11-03: `frontend/src/app` has no privacy, terms or refund route, so there is nothing to link yet.** The owner must supply or approve the texts (an agent draft is not legal advice); a footer link task follows.
- [ ] Email: the app sends no transactional email today; Stripe sends receipts. If email is added: SPF/DKIM/DMARC.
- [ ] Analytics and Google OAuth: not present in the app; nothing to do.
- [ ] Data protection: personal portfolio data lives in Supabase; account deletion with Stripe customer cleanup is not implemented. Decide before launch if EU or RU users are expected. For an RU audience the operator obligations (Roskomnadzor notification, data localization) are the owner's to check.
- [ ] Version/CHANGELOG: root `CHANGELOG.md` exists (`34741ec`, "Unreleased" section; `frontend/package.json` is 0.1.0, no Python version is set); the tag and release notes are produced in the `release` step after the owner's yes.

## 12. Push/deploy sequence (prepared, NOT run)

Each needs the owner's explicit yes: push `chore/audit`; open a PR to `main`; CI green; merge; Vercel deploys from `main`; HF Space repo updated from `main` (remote and method chosen by the owner); smoke test (section 9) in test mode; then switch Stripe to live (sections 5 and 10). Rollback per section 8.

## 13. Scans and clean build (2026-11-03)

Commands run from the repo root; exit codes are real.

| Command | Exit | Result |
|---|---|---|
| `node scripts/secrets-scan.mjs --project . --history` | 3 | 68 hits, all `ENV-*` type: values of `APP_NAME`, `LOG_FILE`, `PRICE_CACHE_DIR`, `DATABASE_URL` from the local `.env` that also appear in tracked files as defaults (`config/settings.py`, README, docs) and the local compose DB URL in `.env.example`. No key-pattern hit (no `sk_`, `whsec_`, JWT, AWS or GitHub token). Without `--history` also exit 3 |
| `node scripts/license-check.mjs --project frontend` | 0 | prod 60 packages, 59 ok, 1 review: `@img/sharp-win32-x64@0.35.5` (Apache-2.0 AND LGPL-3.0-or-later; optional native binary for Windows dev, not in the Linux image), 0 block |
| `node scripts/license-check.mjs --project .` | 2 | no root `package.json`; the Python dependencies are not covered by this script |
| `npm audit --omit=dev` (frontend) | 0 | found 0 vulnerabilities |

Not run in this session (the Docker daemon would not start): `pip-audit`, clean-clone Docker build and pytest. Last recorded results: 448 passed, 0 skipped, core cov 78.35%, API and web images build (docs/QA.md Resolution, 2026-11-02).

Owner OK needed for the 68 history hits (all benign defaults, listed above). Do not add `.secrets-allow` without that OK.

## 14. Owner-only actions (Russian summary)

1. Подтвердить, что в HF Space заданы `JWT_SECRET` (>= 32), `ENVIRONMENT=production`, `AUTH_DISABLED=false` и `CLIENT_IP_SIGNING_KEY` (>= 32, то же значение на Vercel). Без последнего API теперь не стартует.
2. Проверить, что боевые URL живы (Vercel `/login`, HF `/health`); раньше были 404 и 503.
3. Stripe live: продукт и Price, портал клиента, webhook на `https://<hf-api-host>/stripe/webhook` с четырьмя событиями (раздел 5), ключи только в секретах HF.
4. Бэкапы Supabase и проверка восстановления (раздел 6); мониторинг и алерты на сбои webhook.
5. Юридические страницы и налоги: проверка человеком.
6. Решения: принять 68 безвредных совпадений secret-scan; нужен ли эндпоинт удаления аккаунта (F-5); трекать ли `docs/` в git.
7. push в `main`, PR и деплой: только по явному «да» владельца.

Read: E:/Main/Dev/coding-brain/process/release-prep.md, E:/Main/Dev/coding-brain/routers/projects/webapp/saas.md (release-prep row), E:/Main/Dev/coding-brain/known-traps.md, docs/AUDIT_DOCS.md, docs/OWNER_GUIDE.md
