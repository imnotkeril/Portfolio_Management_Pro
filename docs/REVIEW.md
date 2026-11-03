# Review - finish phase (FINISH_PLAN) - 345d95f..9b15498 - 2026-10-25

## Summary
| Reviewer | P0 | P1 | P2 | Notes |
|---|---|---|---|---|
| reviewer (correctness/quality, opus) | 0 | 0 open (1 fixed) | 5 open (2 fixed: R-06 `a7bac81`, R-03 via S-02 `47cfaa9`) | this file |
| security-reviewer | - | - | - | separate agent (not merged here) |
| design-reviewer | - | - | - | not run (per orchestrator: design covered by none) |

Scope: `git diff 345d95f..0574fe1` (15 commits, code only) plus targeted reads of pre-existing tier-A paths named in the brief (repository/service owner predicates, Stripe webhook dedupe, ledger/transactions). Two fix commits added by this review: `43b0f04`, `9b15498`.

## Findings
| ID | Sev | Where | Problem / fix | Status | Fix commit |
|---|---|---|---|---|---|
| R-01 | P1 | services/billing_service.py:166-180 (pre-existing) | Stripe event was marked processed in its own committed session (`subscription_repository.py:69`) BEFORE the handler ran; if the handler raised (DB error), the route returned 400, Stripe retried, and the retry got `already_processed` -> subscription state change (upgrade/downgrade) silently lost. Fix: on handler exception call new `SubscriptionRepository.unmark_event()` and re-raise. Tests: `test_webhook_handler_failure_unmarks_event_for_retry`, `test_unmark_event_allows_reprocessing`. | fixed | 43b0f04 |
| R-02 | P2 | api/main.py:797-809 (pre-existing) | Legacy `DELETE /transactions/{id}` had no `_handle_error`: deleting another user's transaction raised `PortfolioNotFoundError` -> 500 (same class as audit B3; no data change, ownership was enforced). Fix: try/`_handle_error` -> 404. Test: `test_legacy_delete_foreign_transaction_is_404_and_keeps_row` (also proves the row survives = IDOR negative test). | fixed | 9b15498 |
| R-03 | P2 | services/billing_service.py:202-224, 226-247 (pre-existing) | Out-of-order events: `invoice.paid` always sets plan=PRO/status=active from the local row, so a late `invoice.paid` after `customer.subscription.deleted` re-grants Pro; handlers never re-fetch the subscription from Stripe (router trap "never rely on event order, provider is source of truth"). Fix: in each handler `stripe.Subscription.retrieve(id)` and sync from that state. Needs Stripe test keys to verify -> logged, not fixed. | fixed | `47cfaa9` (S-02) |
| R-04 | P2 | api/routers/billing.py:59-64 | `async def stripe_webhook` calls the fully synchronous `handle_webhook_event` (Stripe SDK + SQLAlchemy) on the event loop; blocks all requests while it runs. Fix: make the route `def` (threadpool) after reading the body, or `await run_in_threadpool(...)`. | open | - |
| R-05 | P2 | core/data_manager/transaction_repository.py:49-160, 194-233 | Repository has no owner predicate (`find_by_portfolio`, `find_portfolio_id`, `delete`, `_update_transaction` filter by id only). Every API route was traced and all reach it only after `portfolio_service.get_portfolio(pid, current_user.id)` (see IDOR table), so not exploitable today; defence-in-depth gap for future callers (e.g. `analytics_service.py:142,350,408,794`). Fix: join PortfolioORM and filter `user_id`, or require a verified portfolio object. | open | - |
| R-06 | P2 | frontend/src/app/portfolio/[id]/page.tsx:37-73, layout.tsx:21-35 (commit df3e699) | eslint fix removed `setLoading(true)`/`setError(null)` (page) and the reset in layout: on client navigation /portfolio/A -> /portfolio/B the previous portfolio's overview/name stays visible until B loads, and a prior error stays shown until success. Fix: key the state by id (store `loadedId`, derive `loading = loadedId !== id`) as was done for `failedId`. | fixed | `a7bac81` |
| R-07 | P2 | frontend/src/lib/buy-hold-holdings.ts:19-24 + page.tsx:112-115 (commit 1d4bd56) | When a quote fails AND `purchase_price` is null, `market_value` is null and the total falls back to `r.quantity` (shares counted as dollars) in `totalMarketValue`; when only the quote fails the price silently falls back to cost (0% return) under the "live market prices" caption. Fix: exclude null values from the total and mark fallback prices (e.g. "cost" badge). | open | - |

No P0. Other areas checked with no finding: B1 guards (`config/settings.py:122-150`), B2 `_client_ip` N-th-from-right logic and short-header fallback (`api/rate_limit.py:32-51`), B3 404 (`api/main.py:735-747`), B4 middleware (raw ASGI, does not duplicate existing headers, applies to 404s), F3 proxy allowlist (`frontend/src/lib/proxy-path.ts`: first segment checked, every segment fully decoded, `..`/separators/control chars rejected, re-encoded; `auth`/`stripe` roots excluded; all frontend calls use `portfolios`/`transactions` roots), F5 cash row handled separately, Dockerfiles non-root.

Pointers for security-reviewer (not full findings here): public unauthenticated endpoints `/dashboard/indices` (no rate limit, outbound yfinance), `/validate-tickers`, `/ticker-price/{t}`, `/risk/scenarios`; `api/routers/billing.py:71-72` returns raw exception text when `settings.debug`.

## Correctness and quality

### ACs (docs/FINISH_PLAN.md) - 13/13 verified in code (the earlier "15" counted D1 and D2 as separate ACs)
| AC | Code | Proof |
|---|---|---|
| B1 | config/settings.py:122-150 | tests/unit/test_security_hardening.py:7-79 (auth_disabled, default, short, 32-char, dev tolerant) |
| B2 | api/rate_limit.py:32-51, settings.trusted_proxy_count | test_security_hardening.py:98-148 incl. spoofed XFF vs login limit |
| B3 | api/main.py:735-747 | test_security_hardening.py:151-180 (missing + foreign) |
| B4 | api/security_headers.py, api/main.py:392-422 | test_security_hardening.py:186-218 |
| B5 | Dockerfile.api (`-c requirements.lock.txt`, USER app), Dockerfile | image rebuilt by this review (`audit-pmp-api:test`), tests ran as uid app (pip --user path `/home/app/.local`) |
| B6 | tests/unit/test_optimization_constraints.py, test_subscription_repository.py | coverage 78% total core/ (pasted below) |
| F1 | frontend/package.json next 16.4.3 | build OK below (npm audit not re-run: security reviewer's area) |
| F2 | eslint 0 errors; .github/workflows/ci.yml frontend job | eslint output below; disables carry reasons (1 file-wide `no-explicit-any` in analysis/page.tsx with reason) |
| F3 | frontend/src/lib/proxy-path.ts, route.ts:17-52, next.config.ts | code read (no unit test for sanitizeProxyPath - frontend has no test runner; P3) |
| F4 | frontend/Dockerfile USER node | `docker build -f frontend/Dockerfile` OK below |
| F5 | frontend/src/lib/buy-hold-holdings.ts, portfolio/[id]/page.tsx | tsc/eslint/build only; see R-06, R-07 |
| D1/D2 | docs/production/*, docs/AUDIT "After fixes" | files present in diff |

### IDOR grep (owner predicate)
- `portfolio_repository.py`: every query filters `PortfolioORM.user_id == user_id` (lines 88-90, 143-145, 168-170, 185, 205, 233-235); positions are touched only through a portfolio loaded with the owner filter.
- `subscription_repository.py`: all by `user_id` or `stripe_customer_id` (webhook-internal).
- `transaction_repository.py`: no owner predicate -> R-05; all 13 call paths in `transaction_service.py` call `get_portfolio(pid, user_id)` first; `delete_transaction` cross-checks tx portfolio vs path portfolio.
- `api/main.py`: all 40+ portfolio-scoped routes pass `current_user.id` (no user id from body/query); `_resolve_user_id(None)` raises unless AUTH_DISABLED (now refused in production by B1).

### Webhook / ledger
- Signature: `stripe.Webhook.construct_event` on raw `request.body()`; dedupe `stripe_events.stripe_event_id` UNIQUE (model + migration 011); concurrent duplicate -> IntegrityError -> 400 -> retry -> deduped. Handler failure now un-marks (R-01). Ordering/re-fetch R-03.
- Money as float in transactions (`shares/price/amount/fees` floats) - pre-existing design, out of this diff's scope; noted only.

### AI-slop grep (added lines of the diff)
No `as any`, `# type: ignore`, `console.log`/`print`, empty `catch {}` / `except: pass`, TODO stubs, `.skip`, sleeps. `catch {}` with a return in route.ts:49 and proxy-path.ts:28 are deliberate (502 / reject). 8 `eslint-disable-next-line react-hooks/set-state-in-effect` all with a reason.

## Final verification (re-run by this reviewer after fixes, HEAD 9b15498)
- `docker build -f Dockerfile.api -t audit-pmp-api:test .` -> exit 0.
- pytest (Docker, container audit-pmp-review-tests3, command from STATE): `388 passed, 131 warnings in 37.77s`, `Required test coverage of 70% reached. Total coverage: 78.35%` (0 failed, 0 skipped; +3 regression tests).
- Before fixes (HEAD 0574fe1, container audit-pmp-review-tests): `385 passed, 131 warnings in 34.50s`, `Required test coverage of 70% reached. Total coverage: 78.33%` -> confirms prior claim.
- Targeted after fixes: `tests/unit/test_billing_service.py .. tests/unit/test_subscription_repository.py ............ tests/integration/test_transactions_api.py ..... 19 passed`.
- Host (PYTHONIOENCODING=utf-8): `ruff check .` -> `All checks passed!`; `black --check .` -> `175 files would be left unchanged.`; `isort --check-only .` -> `Skipped 329 files` (no diffs); `mypy .` -> `Success: no issues found in 127 source files`. Pre-commit hooks black/isort/ruff passed on both fix commits.
- Frontend: `npx eslint` -> `✖ 18 problems (0 errors, 18 warnings)`, exit 0; `npx tsc --noEmit` -> exit 0.
- `npx next build` on host -> FAILED with `EBUSY: resource busy or locked, rmdir '...frontend\.next\standalone'` (directory held by another process, likely a sibling agent's standalone server; not touched). Re-ran the production build in Docker instead: `docker build -f frontend/Dockerfile -t audit-pmp-web:review frontend` -> `✓ Compiled successfully in 5.3s`, `✓ Generating static pages using 15 workers (22/22)`, exit 0.

## Adaptations (framework vs this stack)
1. SPEC.md/plan.md ACs -> docs/FINISH_PLAN.md (project CLAUDE.md: docs/plan.md is the legacy Streamlit plan; no SPEC.md).
2. Part file `docs/review/correctness.md` -> written directly as docs/REVIEW.md (orchestrator instruction; no merge step).
3. "DAL" owner-predicate grep -> SQLAlchemy repositories `core/data_manager/*_repository.py` + service layer `services/*` + FastAPI routes `api/main.py`.
4. zod `.partial()`/`.default()` PATCH trap -> checked Pydantic PATCH semantics not in diff; n/a for this diff.
5. Drizzle traps (CREATE EXTENSION vector, 22P02 arrays), `webhook_event` table -> `stripe_events` (migration 011).
6. Next.js server actions / `proxy.ts` middleware traps -> n/a (BFF route handler `/api/proxy/[...path]` + FastAPI); auth re-checked in FastAPI per route.
7. `pnpm build/lint/tsc` -> `npx eslint`, `npx tsc --noEmit`, `next build` (host EBUSY -> Docker build).
8. Tests on TEST DB -> pytest in throwaway Docker container (SQLite test DB from conftest), host pip broken by SSL proxy.
9. Design gate (375/768/1440, playwright, design-loop, impeccable, accessibility-review), module qa E2E (Stripe CLI, email, analytics, Sentry, legal) -> not run: no design reviewer in this run, no Stripe keys, modules not implemented in this project beyond Stripe billing.
10. `code-review`/`simplify` skills -> manual diff review (diff is mostly config/tests; findings recorded above).

## Trap-worthy (for known-traps.md)
- 2026-10-25 Webhook dedupe marker committed before the handler runs -> a failing handler makes every Stripe retry a no-op (state change lost). Mark and handle in one transaction, or delete the marker on failure. (PMP)
- 2026-10-25 Fixing `react-hooks/set-state-in-effect` by deleting the "reset on id change" setState leaves stale data visible on client navigation between dynamic ids; derive loading/error from a stored id instead. (PMP)
- 2026-10-25 `next build` on Windows fails with EBUSY on `.next/standalone` while a standalone server from the same folder runs; build in Docker or a copy. (PMP)

Read: E:/Main/Dev/coding-brain/process/review.md, E:/Main/Dev/coding-brain/routers/projects/webapp/_family.md, E:/Main/Dev/coding-brain/routers/projects/webapp/saas.md, E:/Main/Dev/coding-brain/known-traps.md
