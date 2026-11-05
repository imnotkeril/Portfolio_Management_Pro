# STATE
- 2026-10-24: audit done; 3 commits on chore/audit (dockerignore, env.example, README). docs/ is gitignored -> STATE/plan/audit are force-added.
- Started finish phase (FINISH_PLAN.md). Host pip broken by SSL proxy -> tests run in Docker (image audit-pmp-api:test rebuilt from Dockerfile.api, throwaway containers prefixed audit-pmp-).
## Needed from owner
- Stripe keys/webhook secret; confirm prod JWT_SECRET strong, ENVIRONMENT=production, AUTH_DISABLED=false on HF Spaces; Supabase backup plan; decision whether docs/ should be un-ignored; push/PR approval.
- [x] frontend/F1 next 16.2.1->16.4.3 (+postcss/sharp/source-map-js via lock) — npm audit 0 vulns; tsc + next build pass
- [x] backend/B1-B4 prod guards (AUTH_DISABLED, JWT>=32), trusted-proxy rate-limit IP (TRUSTED_PROXY_COUNT), transactions 404, docs hidden + security headers — 350 passed (docker), core cov 70.98%, ruff/black/isort/mypy clean
- [x] backend/B6 +35 tests (subscription_repository, optimization constraints/base) — 385 passed, core cov 78.33% (was 70.98%)
- [x] frontend/F2 eslint 46 errors (26 any, 20 set-state-in-effect after eslint-config bump) -> 0; ci.yml frontend job (npm ci, eslint, tsc, build) — eslint 0 errors/18 warnings, tsc ok, build ok
- [x] frontend/F3 proxy first-segment allowlist (src/lib/proxy-path.ts), traversal/encoded -> 404, upstream down -> 502; next.config security headers (nosniff, DENY, referrer, permissions; no CSP: Next inline scripts need nonce) — live curl check on standalone server
- [x] frontend/F4 Dockerfile runs as USER node (chown'd standalone), HOSTNAME=0.0.0.0 — docker build audit-pmp-web:test ok; container id uid=1000(node), /login 200
- [x] B5 pins + non-root verified: image rebuilt, uid=1000, health ok, 385 passed, cov 78.33%
- [x] D1/D2/F5 done; finish phase complete. Next: owner items only.
- [x] docs/README - 24 commands verified, 1 failing (SQLite quick start: API does not boot), /llms.txt missing; details docs/DOCS_CHECK.md
- [x] security/S-01..S-04,S-08(S-05 in brief),R-06 fixes - P0=P1=0; 429 passed 1 skipped; commits 47cfaa9 78a64dd f4d2099 5f44431 a7bac81
- [x] security/S-03 remainder + S-18 fixed - 436 passed; MC response bounded, login failures keyed by (email, IP)
- [x] security/S-19 signed client IP + soft brake + one MC run per user - 443 passed; commits a79dec8 c20f4cb
- [x] qa/<final> — pytest 443 passed/0 failed/0 skipped (core cov 78.35%); live Tier-A 108 checks (106 pass, F-1/F-5 open), rate 19/19, UI E2E 7 steps; verdict RED (F-1 ledger/maintain 500, F-2 price step bug, F-5 no account deletion); see docs/QA.md
- [x] qa/F-1,F-2,F-3,F-7,S-23 fixed - 448 passed, 0 skipped; commits 00a3888 e1af502 5cb837e e833e62 d69385e
- [x] docs/README rewrite (2026-11-02) - 32 commands run, see docs/DOCS_CHECK.md; 5 screenshots in assets/screenshots (commit 96d9484); open: isort fails with an in-repo .venv (README documents --skip), SQLite does not boot, /llms.txt missing, production URLs unhealthy
- [x] review/correctness + security review: docs/REVIEW.md (a29d108), docs/SECURITY_REVIEW.md rounds 1-3 (P0=P1=0 by code review round 3; S-19 live attack proof by QA rate rows)
- [x] release-prep: docs/RELEASE.md (3a97628, e558398 fix S-22); HF Dockerfile verified locally (non-root, migrations, /docs 404, weak-secret refusal)
- [x] audit/docs: docs/AUDIT_DOCS.md (40ff170) verdict FIX; stale statements corrected (S-22/S-23/R-06/test counts)
- [x] final verify on 6bf0aeb: ruff/black/isort/mypy clean, eslint 0 err/18 warn, tsc clean, npm audit 0, docker builds ok, pytest 448 passed 0 skipped cov 78.35%
## Missing vs framework
- /llms.txt for API; owner guide (plans/prices, refunds, MRR, key rotation); Stripe CLI webhook forwarding doc; design gate screenshots 375/768/1440; docs/checks/*.md artifacts; AC table; known-traps additions; account deletion endpoint (F-5); open P2: S-05,S-06,S-07,S-09,S-10,S-11,S-21,S-23(unset key),S-24; SQLite boot bug; F-3/F-2 unverified in browser

## Finish phase 2 (2026-11-03)
- [x] api/llms.txt generated from routes (52 ops), served at GET /llms.txt, test written (not run: Docker hung) - b5f1a3c e4833ca
- [x] S-23 fail-closed: API refuses to boot in production without CLIENT_IP_SIGNING_KEY - b5f1a3c (test not run)
- [x] docs/OWNER_GUIDE.md, docs/STRIPE_WEBHOOK_LOCAL.md, QA AC table, RELEASE sections 13-14 (scans: secrets-scan exit 3 = 68 benign ENV hits, license frontend 0, npm audit 0)
- [x] known-traps.md: 12 traps appended (review + security)
- [x] audit/review - FIX, 5 findings (5 valid) [shadow]
- [x] audit/qa - FIX, 2 findings (2 valid) [shadow]
- [x] audit/docs - FIX, 5 findings (4 valid, 1 waived) [shadow]
- [x] audit/release-prep - OWNER, 5 findings (5 valid) [shadow]
- [ ] BLOCKED on Docker Desktop (daemon hangs, restart denied by permission classifier): run pytest at HEAD, pip-audit, clean build, design gate screenshots 375/768/1440, F-2/F-3 browser checks
- [x] 2026-11-05 Docker back: pytest 451 passed in API image; fixed 1 test for fail-closed S-23; pyasn1 -> 0.6.5; pip-audit left ecdsa/python-jose waived
- [x] 2026-11-05 design gate: 4 rounds, verdict OK (D-01/02/03/07/08 fixed in 81bd623, 5bc664a, 609b91d); F-2 and F-3 browser-verified; QA RED stands on F-5 only
- [x] 2026-11-05 audit/qa re-check: OWNER (A-03 F-5, A-04 resolved); audit/release-prep re-check: OWNER (Lighthouse not run, legal pages, advisories waiver, 68 scan hits)
