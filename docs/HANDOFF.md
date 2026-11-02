# Handoff
status: active
date: 2026-11-02
from-chat-tokens: 300000

## Goal
Finish Portfolio Management Pro per coding-brain framework (router webapp/saas, branch chore/audit): audit findings fixed, framework artifacts and docs complete. Local commits only; owner decides push/PR/deploy.

## State
- Done: audit (docs/AUDIT_2026-10-24.md); fixes S-01..S-04, S-18, S-19, S-21, S-23(short key), F-1, F-2, F-7, R-06; REVIEW/SECURITY_REVIEW (3 rounds, P0=P1=0 by code review in round 3)/QA/RELEASE/DOCS_CHECK/README+5 screenshots/CHANGELOG written.
- Evidence (HEAD 6bf0aeb+docs): ruff/black/isort/mypy clean, eslint 0 err/18 warn, tsc clean, npm audit 0, pytest in Docker 448 passed 0 skipped core cov 78.35%, docker builds OK.
- Auditor (docs/AUDIT_DOCS.md, 40ff170): verdict FIX, 4 P1/5 P2/1 P3: framework artifacts never produced (below).
- Running: nothing (all audit-pmp-* containers/images removed).
- Branch/commit: chore/audit @ 0b3d20d + handoff commit, tree clean. NOT pushed: everything after 345d95f incl. 6bf0aeb (~60 commits).

## Decisions
- Python/FastAPI stack runs under router webapp/saas with adaptations (no Drizzle/Better Auth/pnpm) - listed in REVIEW.md.
- docs/ is gitignored; docs are committed with `git add -f` (owner to decide un-ignoring).
- Client IP for rate limits: BFF signs IP (CLIENT_IP_SIGNING_KEY), API uses TRUSTED_PROXY_COUNT=0; on non-Vercel hosts set CLIENT_IP_TRUSTED_HEADER (frontend) to the platform header carrying the visitor IP, else no IP is sent.
- Host pip is broken by SSL proxy: tests run in Docker (copy repo without frontend/.git/.env/data/logs).

## Next
1. Produce missing framework artifacts: docs/checks/{review,qa,docs,release-prep}.md, AC table in QA.md, `Read:` lines in RELEASE.md/DOCS_CHECK.md, known-traps additions, design-gate screenshots 375/768/1440 (step-brief + agents per docs/AUDIT_DOCS.md exit-check table).
2. Write owner guide (plans/prices, refunds, MRR, key rotation), Stripe CLI webhook doc, API /llms.txt (52 operations), then rerun docs auditor.
3. Verify F-3 (375px overflow), F-2 (submit BUY with autofilled price) in a browser; make the F-1 test red against old code (git stash fix) once; live-check S-19 attack on a throwaway stack.

## Open questions
- Push chore/audit and open PR to main? (owner yes needed)
- Which stripe version does prod run (16 -> paying users had no Pro before fix)? Prod URLs were unhealthy (web /login 404, API /health 503).
- Un-ignore docs/? Account deletion endpoint (F-5)? Accept history secret-scan hits (66, non-secret .env values)?

## Traps
- Open P2: S-05 enumeration, S-06 open redirect, S-07, S-09, S-10, S-11 checkout idempotency, S-21, S-23 unset key only warns, S-24 sleep in sync route; SQLite does not boot.
- Docker Desktop stops between sessions; start it and wait for the daemon. Containers must be named audit-pmp-*.
- Never read .env*; host `next build` may hit EBUSY, use `docker build -f frontend/Dockerfile frontend`.

## Verify
- `git log --oneline 345d95f..HEAD | wc -l` and `git status --short` (clean)
- `python -m ruff check . && python -m mypy core/ services/ api/ config/ models/ database/`
- `cd frontend && npx eslint && npx tsc --noEmit`
- pytest: see docs/QA.md / docs/DOCS_CHECK.md Docker procedure (expect 448 passed, 0 skipped)
