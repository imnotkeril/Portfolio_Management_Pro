# QA report - Portfolio Management Pro (branch chore/audit)

**Verdict: RED** (no P0/P1 open; blocked by the "nothing open in tier A/B" rule: 1 open tier-A authz contract bug F-1, 1 broken critical UI step F-2, 1 compliance gap F-5). Money path, webhooks, auth, rate limiting, IDOR data protection: all green.

Code under test: commit `f10bc42` (app code identical to HEAD `e558398`, which only adds docs); fixes S-01..S-04, S-18, S-19, S-21 included. Date 2026-11-02.

## Setup (throwaway, nothing real)

| Piece | Detail |
|---|---|
| Network / DB | docker network `audit-pmp-qa-net`; `audit-pmp-qa-pg` = postgres:16-alpine, database `pmp_qa_test` (test-pattern name), random password, alembic 001..011 applied from scratch by the API image entrypoint |
| API | image built by `docker build -f Dockerfile.api -t audit-pmp-qa-img .` (uid 1000), `ENVIRONMENT=production`, generated 86-char `JWT_SECRET`, three instances: `audit-pmp-qa-api` (direct, `TRUSTED_PROXY_COUNT=0`, signing key set), `audit-pmp-qa-api-xff` (`TRUSTED_PROXY_COUNT=1`, no key), `audit-pmp-qa-api-bff` (`TRUSTED_PROXY_COUNT=0`, signing key; upstream of the BFF) |
| Frontend | image built by `docker build -f frontend/Dockerfile` (standalone `server.js`, user `node`), `audit-pmp-qa-web`, `CLIENT_IP_SIGNING_KEY` same as API, `CLIENT_IP_TRUSTED_HEADER=x-real-ip` |
| Stripe | dummy `sk_test_<random>` + dummy `whsec_<random>` generated at run time; webhooks are signed with HMAC-SHA256 exactly as Stripe does (signer validated against `stripe.Webhook.construct_event`, stripe 15.1 host / 16.0.0 in the image). Because the code now re-fetches the subscription from Stripe, `audit-pmp-qa-fakestripe` (50-line HTTP fake, `stripe.api_base` redirected by a test-only `sitecustomize` mounted at `/shim`, the app image is untouched) answers `GET /v1/subscriptions/{id}`. No call leaves the machine. |
| Ports | random in 20000-39999 on 127.0.0.1 only |
| Secrets | `.env*` never read; keys generated per run, kept outside the repo |
| Harness | `docs/qa/harness/*.txt` (copies of the scripts; rename `.py.txt` -> `.py`; they are kept out of `*.py` so ruff/black do not lint them) |
| Cleanup | all `audit-pmp-qa-*` containers, network and temp images removed at the end |

## Risk map

| Area | Tier | Tests | Result |
|---|---|---|---|
| Auth: register/login/JWT/session | A | live matrix (AUTH, TOKEN, REG, LOGIN), unit | GREEN |
| Authorization / tenant isolation (IDOR) | A | live matrix IDOR (22 endpoints x second user) + state-unchanged proof; BFF IDOR | GREEN except F-1 (500 instead of 404 on one endpoint, no leak, no change) |
| Billing/webhooks: signature, dedupe, ordering, past_due, cancel, resubscribe, expiry | A | 32 live checks + unit | GREEN |
| Free-tier limits / `require_pro` | A | live matrix TIER, EXPIRE | GREEN |
| Rate limiting / brute-force / IP spoofing | A | 19 live checks | GREEN |
| Data deletion (portfolio, transaction, cascade) | A | live DEL-1..5 + DB queries | GREEN; account deletion does not exist (F-5) |
| Input bounds (Monte Carlo, ticker list) | A/B | SIZE-1..4 + pytest | GREEN |
| BFF proxy (allowlist, traversal, cookie flags) | A | live BFF/COOKIE | GREEN |
| Security headers / docs hidden / CORS | B | live HDR/DOCS/CORS | GREEN (notes F-8) |
| Portfolio/analytics/optimization/risk/forecast engines | B | pytest 443 (unit+integration), core coverage 78.35% | GREEN (not re-derived live: numbers) |
| Frontend flows (register -> create -> BUY -> overview -> analysis -> sign out) | C | Playwright (Chrome) on the production image | GREEN with F-2/F-3/F-4 |

## Commands run and results (own re-run, this session)

| Command | Result |
|---|---|
| `docker build -f Dockerfile.api -t audit-pmp-qa-img .` | ok |
| pytest in throwaway container (image `audit-pmp-qa-img` + `requirements-dev.txt` under `requirements.lock.txt`, repo copied via `git archive HEAD` so no `.env`/`data`/`logs`/`frontend`/`.git`) `python -m pytest -q -rs` | **443 passed, 0 failed, 0 skipped**, 131 warnings, 61 s; `core/` coverage **78.35%** (gate 70%) |
| `ruff check .` | All checks passed |
| `black --check .` | 176 files would be left unchanged |
| `isort --check-only .` | isort 8.0.1 (version pinned in `.pre-commit-config.yaml`): clean. isort 9.0.2 (what unpinned `isort>=5.13` resolves to today): 2 files flagged, `services/risk_service.py`, `tests/unit/test_risk_scenario_optimization_core.py` - see F-7 |
| `mypy core/ services/ api/ config/ models/ database/` (as CI) | mypy 1.20.2 and 2.4.1: 1 error `core/forecasting_engine/simple/prophet.py:16: Incompatible types in assignment (None vs type[Prophet])` in 126 files - see F-7 |
| `npx eslint` (frontend) | 0 errors, 18 warnings |
| `npx tsc --noEmit` | 0 errors |
| `npm audit --omit=dev` | found 0 vulnerabilities |
| `docker build --no-cache -f frontend/Dockerfile frontend` (runs `next build`; host `next build` can hit EBUSY on Windows) | Next.js 16.4.3 `Compiled successfully`, `Finished TypeScript`, `Generating static pages (22/22)` |
| Live matrix `qa_live.py` | **108 checks: 106 pass, 2 fail** (F-1, F-5) |
| Live rate-limit `qa_rate.py` | **19 checks: 19 pass, 0 fail** |
| UI E2E (Playwright, installed Chrome, production web image) | 7 steps pass; findings F-2, F-3, F-4 |

Test database: `pmp_qa_test` on the throwaway postgres only (pytest itself uses its own sqlite fixtures; no dev data touched).

The only test change I made: `tests/integration/test_request_size_limits.py` (commit `30065e7`): the VaR size-limit test used `pytest.skip` inside a parametrized case; the impossible combination is now filtered at collection time, so the suite is 0 skipped. No new failing tests were committed.

## Findings

| # | Sev | Area | Status | Detail / repro |
|---|---|---|---|---|
| F-1 | P2 | authz (tier A) | **OPEN** | `POST /portfolios/{id}/ledger/maintain` with another user's (or unknown) portfolio id returns **500** `Internal Server Error` instead of 404 (every other 21 portfolio endpoints return 404). No data leaks and nothing changes (verified: A's portfolio, positions and transactions identical afterwards), but it is an unhandled exception on an authorization path. Repro: user B calls it with A's id (`IDOR-1`). Fix: `_verify_portfolio_access` / map `PortfolioNotFoundError` before the unguarded `get_portfolio` in `maintain_portfolio_ledger` (api/main.py). Regression test belongs in `tests/integration/test_user_isolation.py` (expect 404); not added because it would turn the suite red before the fix. |
| F-2 | P2 | frontend (critical flow) | **OPEN** | Transactions page, BUY/SELL: the auto-filled historical price is a raw float (`279.8399963378906`) while the input has `step=0.01`, so the browser's native validation fails (`checkValidity() == false`, message "nearest valid values 279.83 and 279.84") and **the Add Transaction click silently does nothing** (0 requests, table unchanged). Typing the price by hand works. Fix: round the loaded price to 2 decimals or use `step="any"` in `frontend/src/components/transaction-form.tsx`. Evidence: `BUGCHECK` lines in the E2E log. |
| F-3 | P3 | frontend | OPEN | Portfolio overview has horizontal overflow at 375 px (`scrollWidth > clientWidth`); 768 and 1440 are fine. |
| F-4 | P3 | frontend | OPEN | After sign-out the already-mounted pages keep prefetching protected links; `/login?from=...&_rsc=...` returns 404 and Chrome logs console errors. Harmless (login page itself works, protected pages redirect to `/login?from=`), noise only. |
| F-5 | gap | compliance | OPEN (owner decision) | No account-deletion endpoint (openapi has 52 operations, no `DELETE /auth/*`/users). Portfolio and transaction deletion work and cascade (DEL-1..5). The legal-compliance qa insert (delete-request path end to end) cannot be executed. |
| F-6 | accepted | session | documented | JWT is stateless (24 h): after BFF logout the cookie is cleared in the browser but a copy of the old token still works (`GET /api/proxy/portfolios` with the old cookie -> 200). The auth-module "old cookie rejected after logout" check fails by design; mitigations are short TTL or a denylist. Not a regression. |
| F-7 | P3 | CI | OPEN | Lint toolchain drift: `requirements-dev.txt` leaves `isort`/`mypy` unpinned. With isort 9.0.2 two files are flagged (clean with the pre-commit pin 8.0.1); mypy >= 1.20 reports `prophet.py:16`. A fresh CI install can go red without a code change. |
| F-8 | info | web headers | noted | Web app sends nosniff / DENY / referrer / permissions-policy (no `X-Powered-By`); no HSTS and no CSP on the Next.js side (HSTS expected from the hosting platform; CSP deliberately omitted, see STATE F3). API sends HSTS in production. `localhost:3000` is always in the API CORS allow-list (credentials allowed) even in production. A login request with a 1 MB password is accepted for processing (401), `LoginRequest.password` has no max length. |
| F-9 | info | tooling | noted | On the Windows host the standalone server built by host `next build` returned 500 for `/register` ("client reference manifest does not exist") in an earlier attempt; the Linux image of the same code serves all pages. Not reproducible in the docker image, so not counted. |

Fixed during this QA cycle (by others, re-verified live): stripe 16 `StripeObject` crash (every webhook was a 400 -> Pro never granted; now PASS HOOK-checkout etc.), webhook handler failure left the event marked processed (now un-marked, HOOK-handlerfail), out-of-order events (now authoritative re-fetch, OOO-1..4), lockout DoS (S-18), spoofable IP headers (S-19), response size / work caps (S-03), legacy `DELETE /transactions/{id}` 500 -> 404.

## Tier-A matrix: live results (108 checks)

Executed by `qa_live.py` against `audit-pmp-qa-api` (production env) and, for the BFF rows, through `audit-pmp-qa-web`. DB assertions use `psql` inside the throwaway postgres.

| ID | Area | Check | Result |
|---|---|---|---|
| SETUP | setup | two users, one portfolio each, A has DEPOSIT+BUY | PASS |
| IDOR-1 | authz | user B vs user A resources: 22 endpoints denied (404/403/deleted:false) | **FAIL** |
| IDOR-2 | authz | A's portfolio, positions and transactions unchanged after B's attacks | PASS |
| IDOR-3 | authz | B's list shows only B's portfolio | PASS |
| IDOR-4 | authz | A deleting B's tx id via A's own portfolio route fails and B's tx survives | PASS |
| IDOR-5 | authz | A deleting B's tx via legacy DELETE /transactions/{id} fails, B's tx survives | PASS |
| IDOR-6 | authz | unknown/malformed portfolio id '00000000-0000-0000-0000-000000000000' -> 4xx not 5xx | PASS |
| IDOR-6 | authz | unknown/malformed portfolio id '../../etc/passwd' -> 4xx not 5xx | PASS |
| IDOR-6 | authz | unknown/malformed portfolio id "' OR 1=1 --" -> 4xx not 5xx | PASS |
| UNAUTH-1 | authn | 52 openapi operations: 43 protected all return 401 without token | PASS |
| UNAUTH-2 | authn | malformed/wrong-scheme Authorization headers -> 401 | PASS |
| TOKEN | authn | /auth/me with valid control: expect 200 | PASS |
| TOKEN | authn | /auth/me with expired (-1h): expect 401 | PASS |
| TOKEN | authn | /auth/me with expired (-1s): expect 401 | PASS |
| TOKEN | authn | /auth/me with wrong secret: expect 401 | PASS |
| TOKEN | authn | /auth/me with alg=none: expect 401 | PASS |
| TOKEN | authn | /auth/me with alg=HS512 same secret: expect 401 | PASS |
| TOKEN | authn | /auth/me with no exp claim: expect None | PASS |
| TOKEN | authn | /auth/me with unknown user sub: expect 401 | PASS |
| TOKEN | authn | /auth/me with sub missing: expect 401 | PASS |
| TOKEN | authn | /auth/me with sub non-string: expect 401 | PASS |
| TOKEN | authn | /auth/me with payload swapped to B, old signature: expect 401 | PASS |
| TOKEN-exp | authn | issued token carries exp (<=24h) and sub | PASS |
| AUTH-inactive | authn | deactivated user: existing token and new login both 403 | PASS |
| REG-dup | auth | duplicate email -> 409 | PASS |
| REG-dupcase | auth | duplicate email differing by case -> 409 (no case-variant duplicate) | PASS |
| REG-val | auth | register validation password<8 -> 422 | PASS |
| REG-val | auth | register validation bad email -> 422 | PASS |
| REG-val | auth | register validation password>128 -> 422 | PASS |
| REG-val | auth | register validation missing email -> 422 | PASS |
| REG-val | auth | register validation empty -> 422 | PASS |
| REG-nohash | auth | register response exposes no password/hash | PASS |
| REG-hash | auth | password stored hashed (not plaintext), bcrypt/argon prefix | PASS |
| LOGIN-bad | auth | wrong password and unknown email give identical 401 body (no enumeration at login) | PASS |
| LOGIN-form | auth | OAuth2 form login works | PASS |
| LOGIN-case | auth | login with upper-cased email works or is cleanly refused (observed) | PASS |
| LOGIN-sqli | auth | SQLi-style credentials -> 401/422, not 5xx | PASS |
| TIER-1 | billing | free user creating 2nd portfolio -> 403 limit | PASS |
| TIER-2 | billing | billing/status free/active/is_pro false/limit 1 | PASS |
| TIER-3 | billing | 5 require_pro endpoints return 403 for free user (before body validation) | PASS |
| TIER-4 | billing | checkout/portal without Stripe price config -> 503 (no crash, no client-controlled price) | PASS |
| TIER-5 | billing | client-supplied amount/price in checkout request ignored (endpoint takes no body) | PASS |
| HOOK-lib | webhook | test signer output validated by stripe.Webhook.construct_event | PASS |
| HOOK-forged | webhook | forged signature (wrong secret) -> 400, no upgrade | PASS |
| HOOK-nosig | webhook | missing Stripe-Signature -> 400, no upgrade | PASS |
| HOOK-tamper | webhook | valid signature for original body, body swapped to user B -> 400, B not upgraded | PASS |
| HOOK-stale | webhook | correct signature but timestamp 1h old (replay window) -> 400 | PASS |
| HOOK-garbage | webhook | signed non-JSON body -> 400 (no 500) | PASS |
| HOOK-badhdr | webhook | malformed signature header -> 400 | PASS |
| HOOK-noevent | webhook | no stripe_events row recorded for rejected deliveries | PASS |
| HOOK-bff | webhook | webhook not reachable through the browser BFF proxy (404) | PASS |
| HOOK-unpaid | webhook | checkout.session.completed with payment_status=unpaid does NOT grant Pro | PASS |
| HOOK-checkout | webhook | valid signed checkout.session.completed (Stripe re-fetch says active) -> 200, A is pro/active | PASS |
| HOOK-entitle | webhook | after webhook: A can create 2nd portfolio; require_pro route no longer 403 | PASS |
| HOOK-bnot | webhook | user B untouched (still free) | PASS |
| HOOK-dup | webhook | duplicate delivery (same evt id) -> already_processed, 1 stripe_events row, 1 subscription row | PASS |
| HOOK-concurrent | webhook | 8 simultaneous deliveries of one event: exactly one fulfilment, others already_processed, no 5xx | PASS |
| PASTDUE-1 | webhook | invoice.payment_failed accepted (200), no 5xx (entitlement follows Stripe subscription state) | PASS |
| PASTDUE-2 | webhook | Stripe says past_due -> is_pro false, require_pro 403, new portfolio blocked (403) | PASS |
| PASTDUE-3 | webhook | existing data of past_due user still readable (no data loss on downgrade) | PASS |
| PASTDUE-4 | webhook | recovery: Stripe active + invoice.paid -> pro again | PASS |
| TRUTH-1 | webhook | event body says canceled but Stripe API says active -> stays pro (event body not trusted for entitlement) | PASS |
| CANCEL-1 | webhook | Stripe canceled + subscription.deleted -> downgraded to free, is_pro false | PASS |
| REPLAY-1 | webhook | replaying already-processed checkout / invoice.paid events after cancel does not re-upgrade | PASS |
| OOO-1 | webhook | out-of-order: stale subscription.updated(active), NEW event id, delivered AFTER cancellation does NOT restore Pro | PASS |
| OOO-2 | webhook | out-of-order: stale invoice.paid after cancellation does NOT restore Pro | PASS |
| OOO-3 | webhook | subscription.updated for a customer not yet linked to a user (arrives before checkout) -> ack, nothing granted | PASS |
| OOO-4 | webhook | checkout arrives after the early subscription event -> pro granted | PASS |
| RESUB-1 | webhook | canceled user re-subscribes (new subscription id) -> pro again | PASS |
| RESUB-2 | webhook | late event for the OLD (canceled) subscription id does not downgrade the new subscription | PASS |
| EXPIRE-1 | billing | row pro/active but current_period_end 30 days in the past (missed webhook) -> is_pro false and require_pro 403 | PASS |
| HOOK-unknowncus | webhook | event for unknown customer -> 200, no rows created/changed | PASS |
| HOOK-unknownuser | webhook | checkout for nonexistent user_id -> 200 ack, no orphan subscription rows | PASS |
| HOOK-unknowntype | webhook | unhandled event type -> 200 (ack, no retry storm) | PASS |
| HOOK-handlerfail | webhook | Stripe API 500 during handling -> non-2xx so Stripe retries; event NOT left marked processed (retry is re-handled) | PASS |
| SIZE-1 | input | oversized num_simulations/time_horizon rejected with 422 on the offending fields | PASS |
| SIZE-2 | input | ticker list of 500 rejected (422) | PASS |
| SIZE-3 | input | Monte Carlo work cap: 10000 sims x 250 days (2.5M cells > 2M cap) -> 422 | PASS |
| SIZE-4 | input | default Monte Carlo/full run succeeds and response stays bounded (< 3 MB) | PASS |
| DEL-1 | delete | other user's DELETE /portfolios/{id} does not delete (row still present) | PASS |
| DEL-2 | delete | owner delete -> 200 deleted true; portfolio, positions, transactions rows gone (cascade) | PASS |
| DEL-3 | delete | repeat delete is idempotent-safe (404 or deleted:false, no 5xx) | PASS |
| DEL-4 | delete | deleted portfolio GET -> 404 | PASS |
| DEL-5 | delete | owner deletes own transaction -> deleted true, row gone | PASS |
| DEL-6 | delete | account deletion endpoint exists (GDPR delete-request path; legal-compliance qa insert) | **FAIL** |
| HDR-api | headers | API /health (200): nosniff, DENY, referrer-policy, HSTS | PASS |
| HDR-api | headers | API /portfolios (401): nosniff, DENY, referrer-policy, HSTS | PASS |
| HDR-api | headers | API /nope-404 (404): nosniff, DENY, referrer-policy, HSTS | PASS |
| DOCS | headers | API /docs hidden in production -> 404 | PASS |
| DOCS | headers | API /redoc hidden in production -> 404 | PASS |
| DOCS | headers | API /openapi.json hidden in production -> 404 | PASS |
| DOCS | headers | API /docs/oauth2-redirect hidden in production -> 404 | PASS |
| CORS-1 | headers | preflight from evil origin gets no Access-Control-Allow-Origin | PASS |
| HDR-web | headers | Web /login (200): nosniff, DENY, referrer, permissions-policy, no X-Powered-By | PASS |
| HDR-web | headers | Web /api/proxy/health (200): nosniff, DENY, referrer, permissions-policy, no X-Powered-By | PASS |
| HDR-web | headers | Web /dashboard (307): nosniff, DENY, referrer, permissions-policy, no X-Powered-By | PASS |
| BFF-1 | bff | 20 non-allowlisted / traversal / encoded paths never reach API (404, never 200/401-from-upstream) | PASS |
| BFF-1b | bff | backslash id with a valid session stays inside /portfolios/ (upstream 404); %2e%2e/auth/me resolves to the BFF's own /api/auth/me (intended, not the proxy) | PASS |
| BFF-2 | bff | allowlisted public path /health works through BFF | PASS |
| BFF-3 | bff | BFF protected path without cookie -> 401 | PASS |
| BFF-4 | bff | client-supplied Authorization header NOT forwarded (no cookie -> 401) | PASS |
| BFF-5 | bff | forged cookie token -> 401 from upstream | PASS |
| BFF-6 | bff | expired cookie token -> 401 | PASS |
| BFF-7 | bff | valid cookie -> 200 with only A's portfolios | PASS |
| BFF-8 | bff | IDOR through BFF (B cookie on A's portfolio) -> 404 | PASS |
| BFF-9 | bff | every protected openapi op via BFF without cookie -> 401 or 404 | PASS |
| COOKIE-1 | bff | login cookie: HttpOnly + SameSite=lax + Path=/ (Secure only when COOKIE_SECURE=true, see COOKIE-2) | PASS |
| COOKIE-3 | bff | BFF login wrong password -> 401 and no Set-Cookie | PASS |

Notes on specific rows:
- UNAUTH-1: all 43 non-public operations of the 52 in the OpenAPI schema (dumped from the app inside the image because `/openapi.json` is hidden in production) answer 401 without a token. Public by design: `/health`, `/dashboard/indices`, `/validate-tickers`, `/ticker-price/{t}`, `/risk/scenarios`, the 3 auth entry points and the signature-checked `/stripe/webhook`.
- IDOR-1: 21 of 22 operations return 404; F-1 is the 22nd. IDOR-2 proves no state change after all 22 attacker calls.
- BFF-1: 18 of 20 hostile paths return 404. The two that do not are explained and checked by BFF-1b: `%2e%2e/auth/me` is normalised by Next.js to the BFF's own `/api/auth/me` (not the proxy), and `portfolios/..%5cauth` is forwarded as the literal encoded id and gets an upstream 404 with a valid session.
- COOKIE-1: login cookie `access_token`: `HttpOnly; SameSite=lax; Path=/; Max-Age=86400`. With `COOKIE_SECURE=true` (second web container) the cookie also carries `Secure` (checked by hand, `Secure; HttpOnly; SameSite=lax`).
- HOOK rows 5xx never occurred; the invalid-signature rows prove no `stripe_events` row is written for rejected deliveries.

## Rate limiting and brute force (19 checks, `qa_rate.py`)

| ID | Check | Result | Observed |
|---|---|---|---|
| RL-1 | direct API: unsigned X-Forwarded-For / X-Real-IP / X-Client-IP spoofing ignored; 25 bad logins (distinct emails) trip the IP bucket at the 21st | PASS | {401: 20, 429: 5} |
| RL-2 | forged X-Client-IP signature (wrong key) ignored: falls back to peer, bucket trips at 21 | PASS | {401: 20, 429: 5} |
| RL-3 | validly-signed but STALE (10 min old) X-Client-IP ignored (replay window): bucket trips at 21 | PASS | {401: 20, 429: 5} |
| RL-4 | validly signed client IP is honoured: IP X trips at 21 while signed IP Y keeps its own bucket | PASS | ({401: 20, 429: 2}, 401) |
| RL-5 | /auth/login and /auth/login/json share ONE 20/min bucket (alternating, trips at 21) | PASS | {401: 20, 429: 5} |
| RL-6 | per-(account, IP) lockout: attacker IP gets 5x401 then 429 | PASS | [401, 401, 401, 401, 401, 429, 429] |
| RL-7 | victim logging in with the correct password from ANOTHER IP is NOT locked out by the attacker's failures (S-18) | PASS | 200 |
| RL-8 | from the attacker's own locked IP even the correct password is refused (429) | PASS | 429 |
| RL-9 | lock key is case-insensitive (UPPER-cased email from the locked IP still 429) | PASS | 429 |
| RL-10 | successful login resets that (account, IP) counter (3 fails, success, 4 fails still 401 not 429) | PASS | (200, [401, 401, 401, 401]) |
| RL-11 | soft brake: failures for one account spread over 12 IPs are delayed after the 5th (each later failure >= 0.3s, growing) | PASS | [0.18, 0.18, 0.19, 0.18, 0.18, 0.38, 0.6, 0.78, 0.98, 1.19, 1.38, 1.58] |
| RL-12 | ...but the real owner's CORRECT login is never delayed/blocked by the brake | PASS | (200, 0.18) |
| RL-13 | TRUSTED_PROXY_COUNT=1 (no signing key): spoofed left XFF entries ignored, trusted right entry keys the bucket (trips at 21) | PASS | {401: 20, 429: 5} |
| RL-14 | TRUSTED_PROXY_COUNT=1: a different real client has its own bucket | PASS | 401 |
| RL-15 | X-Client-IP is ignored when no signing key is configured (no crash, 401) | PASS | 401 |
| RL-16 | register: 13 rapid registrations (spoofed XFF) -> first 10 created, rest 429 | PASS | [201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 429, 429, 429] |
| RL-17 | BFF with trusted visitor-IP header: visitor 198.51.100.50 gets 401s then 429 after 20 (limit reaches the browser) | PASS | {401: 20, 429: 3} |
| RL-18 | BFF: a different visitor IP is not throttled by visitor 1's burst (signed IP forwarded, per-visitor bucket) | PASS | 401 |
| RL-19 | BFF: client-supplied X-Client-IP*/X-Forwarded-For are not forwarded as identity (without trusted header everyone shares the BFF peer bucket; trips at 21) | PASS | {401: 20, 429: 5} |

## Webhook / billing behaviours covered (payments module qa insert)

Success (checkout paid), declined/unpaid (no grant, HOOK-unpaid), abandoned (never produces an event; free stays free), duplicate and 8-way concurrent delivery (exactly one fulfilment, one `stripe_events` row), forged / missing / malformed / stale-timestamp signature (400, nothing written), webhook not reachable through the browser BFF (404), event for unknown customer / unknown user / unknown type (200 ack, no orphan rows), Stripe API failure while handling (non-2xx, event un-marked so Stripe retries), failed payment -> `past_due` (Pro revoked, data kept, recovery on `active`), cancel, re-subscribe with a new subscription id, late event for the old subscription id (ignored), out-of-order and stale `active` events after cancellation (cannot re-grant: state is re-fetched from Stripe), event body lying about status (ignored), expired `current_period_end` with a missed webhook (not Pro), checkout takes no client-supplied price/amount (503 without server price config). Refund and trial expiry are not events the code handles; with the authoritative re-fetch a refund-driven cancellation arrives as `customer.subscription.deleted`/`updated` and is covered by the cancel rows. Real card/3DS flows and the Stripe CLI were not run (no keys, by instruction).

## UI E2E (production web image, Playwright + installed Chrome 1440x900)

| Step | Result |
|---|---|
| 1 Register in the UI (`/register`) -> redirected to `/dashboard` | pass |
| 2 Create portfolio through the 5-step wizard (name, USD 10 000, text input `AAPL 60%, MSFT 40%`, validate, mode "With Transactions", create) | pass: portfolio created, DEPOSIT + BUY AAPL 17 + BUY MSFT 7 written |
| 3 Add BUY GOOGL x3 in the Transactions page | pass **only after retyping the price** (F-2); row appears, total 4 transactions |
| 4 Overview shows holdings | pass: AAPL 17, GOOGL 3, MSFT 7 with market prices, no "No holdings yet" (the earlier buy-and-hold overview bug stays fixed) |
| 5 Analysis page loads with the portfolio selectable | pass (parameter form rendered; calculations not triggered) |
| 6 Viewports | 1440 ok, 768 no overflow, 375 horizontal overflow (F-3) |
| 7 Sign out -> `/login`; `/dashboard` without cookie -> `/login?from=%2Fdashboard` | pass |

Console/network during the authenticated flow: only `401 /api/auth/me` on the public `/register` page before login (expected). After sign-out: 404 on RSC prefetch of `/login?from=...` (F-4). No uncaught page errors.

## Residual risk (not tested)

Live Stripe (real checkout, portal cancel, 3DS, refunds, Stripe CLI), real email, Vercel/HF Spaces/Supabase behaviour (signed client IP relies on the platform header: `x-vercel-forwarded-for` on Vercel, `CLIENT_IP_TRUSTED_HEADER` elsewhere), load/performance, analysis/optimization/forecast numbers in the browser, Lighthouse/axe, other browsers, devices, in-process limiter state is per worker. Frontend has no unit test suite (lint, types, build and the E2E above are the only gates).

Read: E:/Main/Dev/coding-brain/process/qa.md, routers/projects/webapp/_family.md, routers/projects/webapp/saas.md (qa row), known-traps.md, modules/auth.md, modules/payments.md, modules/email.md, modules/legal-compliance.md, modules/analytics.md, modules/error-monitoring.md (qa inserts via step-brief output), docs/AUDIT_2026-10-24.md, docs/STATE.md, CLAUDE.md

## Resolution (2026-11-02)

Verified after the fixes: pytest in the API image **448 passed, 0 skipped**; `ruff check .`, `black --check`, `isort --check-only`, and `mypy core/ services/ api/ config/ models/ database/` (126 files, 0 errors) clean on the host; fresh `pip install -r requirements-dev.txt` in the API image resolves black 26.3.1 / isort 8.0.1 / mypy 1.20.2 / ruff 0.15.12 and passes the same checks (ruff there also flags the git-ignored local `fix_linting.py`, which is not in the repo); eslint 0 errors (18 warnings, unchanged); `tsc --noEmit` clean; `docker build -f frontend/Dockerfile` (runs `next build`) OK.

| ID | Status | Commit | Evidence |
|---|---|---|---|
| F-1 | fixed | `00a3888` | `tests/integration/test_user_isolation.py::test_ledger_maintain_foreign_or_unknown_portfolio_is_404` (foreign and unknown id -> 404, own -> 200). `maintain_portfolio_ledger` now verifies access first and maps the error. |
| F-2 | fixed | `e1af502` | The auto-filled price is rounded to cents (at least 0.01) in `transaction-form.tsx`, so `step=0.01` validation passes. tsc/eslint/build clean. No automated or browser re-test of the submit (no frontend test runner). |
| F-3 | fixed, not verified | `d69385e` | Holdings table wrapped in `overflow-x-auto`. Not re-measured at 375 px in a browser; other elements may still overflow. |
| F-7 | fixed | `5cb837e` | `requirements-dev.txt` pins black, isort, ruff, mypy and the two type stubs; `prophet.py` optional import carries `# type: ignore[assignment,misc]` (it is the usual optional-dependency pattern, not a real bug). The reported mypy error does not reproduce on the host (Prophet installed) and is gone in the clean image. |
| S-23 | fixed (fail-closed) | `e833e62`, `b5f1a3c` | Short key refused (`test_production_refuses_short_client_ip_signing_key`, `..._accepts_32_char_...`); since `b5f1a3c` an unset key is refused too (`test_production_refuses_unset_client_ip_signing_key`). Dev does not require it (`test_development_does_not_require_...`). **The `b5f1a3c` tests were written on 2026-11-03 but not run: the Docker daemon was unavailable; rerun before release.** |

## Acceptance criteria (docs/FINISH_PLAN.md; there is no SPEC.md)

| AC | Verified by | Result |
|---|---|---|
| B1 prod guards (AUTH_DISABLED, JWT >= 32) | `tests/unit/test_security_hardening.py` (default, short, 32-char, dev tolerant); live: weak-secret container exits 1 (RELEASE.md section 3) | pass |
| B2 rate-limit client IP (TRUSTED_PROXY_COUNT, signed IP) | `test_forwarded_ip_used_only_when_proxy_trusted`, `test_valid_signature_is_honored_and_separates_users`; live rows RL-1..19 (this file) | pass |
| B3 transactions 404 for foreign/unknown portfolio | `tests/integration/test_user_isolation.py`; live A/B-user matrix (this file) | pass |
| B4 docs hidden and security headers in production | `test_security_hardening.py` (headers, 404 on /docs); live `curl` rows (this file) | pass |
| B5 pinned deps, non-root image | image built from `Dockerfile.api`, uid 1000 (QA setup table) | pass |
| B6 coverage >= 78% on core/ | 448 passed, 0 skipped, core 78.35% (Resolution section) | pass |
| F1 Next 16.4.3, `npm audit` 0 | `npm audit` clean, Docker `next build` 22/22 | pass |
| F2 eslint 0 errors, CI frontend job | `npx eslint` 0 errors / 18 warnings; `.github/workflows/ci.yml` | pass |
| F3 proxy path allowlist | live traversal/encoded-path rows (this file); no unit test (frontend has no runner) | pass (live) |
| F4 frontend image non-root | `docker build -f frontend/Dockerfile frontend`, `uid=1000(node)` | pass |
| F5 buy_hold overview shows $0 / no holdings | `frontend/src/lib/buy-hold-holdings.ts`, `portfolio/[id]/page.tsx`; tsc/eslint/build; `assets/screenshots/03-portfolio-overview.png` shows holdings. No automated test (no frontend runner) | pass by build + screenshot; no regression test |
| F-3 (QA finding, narrow-screen overflow of the holdings table) | `d69385e` `overflow-x-auto`; not re-measured at 375 px | **not verified in browser** |
| D1/D2 production docs and deploy checklist | `docs/RELEASE.md`, `docs/production/*` | pass |

## Verdict after the Resolution (restated)

**RED stands.** Open in tier A/B: F-5 (no account-deletion endpoint) and the unverified browser fixes F-2 and F-3. No P0/P1 open. Items that need the owner or a live stack: real Stripe checkout and portal E2E (no test keys), F-5 decision.

Read: E:/Main/Dev/coding-brain/process/qa.md, docs/AUDIT_DOCS.md (exit checklists), docs/FINISH_PLAN.md, docs/REVIEW.md (AC table)
