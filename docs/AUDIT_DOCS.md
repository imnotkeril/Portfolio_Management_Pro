# Audit: finish-phase documents (review, qa, docs, release-prep)
mode: shadow (per `rules/audit.md`; advisory only)
auditor-model: claude-opus-5-5 · HEAD audited: `6bf0aeb` (branch `chore/audit`) · date 2026-11-02

Read-only audit. Claims were checked against git history, the code and scripts, not taken from the documents. Nothing outside this file was edited.

## Summary

| Document | Verdict | Why (one line) |
|---|---|---|
| docs/REVIEW.md | OK (stale) | Scoped to `345d95f..9b15498` and correct for that range. R-06 still says `open`, but `a7bac81` fixed it. Traps not appended to `known-traps.md` |
| docs/SECURITY_REVIEW.md | FIX | Final state lists S-22 and S-23 as open. `e558398` resolved S-22; S-23 is partly fixed (`e833e62`). The latest suite count (443) predates the fixes that reached 448 |
| docs/QA.md | FIX | Verdict RED is applied correctly (F-5 still open). The Resolution row says "S-23 done", which overstates the fix (an unset key still fails open). No `docs/checks/qa.md`, no AC table |
| docs/DOCS_CHECK.md | FIX | The 32 commands are real and consistent. The env-name diff is not empty (2 README names missing from `.env.example`). `/llms.txt` 0 vs 52. The 2026-11-02 section has no `Read:` line |
| docs/RELEASE.md | FIX | Stale suite count (436/1 skipped) and a stale "re-check not in docs" line. No `Read:` line. No Russian owner list. secrets-scan, license-check and the clean build were not run |
| README.md | OK | Env tables match the code (scripted). Links and the 5 screenshots exist and are tracked. One minor self-contradiction about SQLite |
| USER_GUIDE.md | OK | Nothing found (plan limits match `services/billing/plans.py:16`) |
| .env.example | FIX (small) | `CLIENT_IP_TRUSTED_HEADER` is read by code and listed in README but missing here |
| docs/STATE.md | OK (gaps) | Ticks match git. Missing lines: review step, release-prep (D2 / `3a97628`), S-22 doc fix, any `audit/*` line |
| docs/FINISH_PLAN.md | OK | All 15 ticks have a matching commit |
| CLAUDE.md | OK | Nothing found |

Blocking-level items (P1 by rule, checklist FAIL): 9 exit-check FAILs, listed in the tables below. Findings with evidence: 10 (A-01..A-10). The router's design and module Done-when items are not met; most need the owner's input.

## Verified facts (evidence)

- **Commits.** `git log --oneline 345d95f..HEAD` shows 56 commits. Every hash cited in STATE, REVIEW, SECURITY_REVIEW, QA and DOCS_CHECK exists, and its subject matches the claim: `43b0f04` `9b15498` `47cfaa9` `78a64dd` `f4d2099` `5f44431` `a7bac81` `970b581` `338748f` `30065e7` `a79dec8` `c20f4cb` `00a3888` `e1af502` `5cb837e` `e833e62` `d69385e` `96d9484` and the D-commits `f030521` `ad245ed` `576fa4d` `0bac40e` `2680dbb` `9d9cddc` `e0b2c49`.
- **Tests cited as proof of "fixed".** All exist (`git grep "def <name>" tests`): `test_webhook_handler_failure_unmarks_event_for_retry` (test_billing_service.py), `test_unmark_event_allows_reprocessing`, `test_legacy_delete_foreign_transaction_is_404_and_keeps_row`, `test_checkout_completed_grants_pro`, `test_late_invoice_paid_does_not_resurrect_canceled`, `test_is_pro_requires_unexpired_period`, `test_per_email_lockout_after_repeated_failures`, `test_forwarded_ip_used_only_when_proxy_trusted`, `test_attacker_ip_failures_do_not_lock_victim_on_another_ip`, `test_valid_signature_is_honored_and_separates_users`, `test_one_concurrent_monte_carlo_run_per_user`, `test_ledger_maintain_foreign_or_unknown_portfolio_is_404`, `test_production_refuses_short_client_ip_signing_key`, `test_production_accepts_32_char_client_ip_signing_key`, `test_production_warns_loudly_when_client_ip_signing_key_unset`. F-2 and F-3 (frontend) have no test; QA.md says so honestly (QA.md:251-252).
- **Counts.** The latest count is 448 passed, 0 skipped (STATE:19, QA.md:246, DOCS_CHECK.md:28; the orchestrator's own run on `6bf0aeb` also gives 448, core cov 78.35%). DOCS_CHECK.md:28 shows 78.45% (host venv, SQLite fixtures), while the Docker runs show 78.35%. The difference comes from the environment, and no document explains it (P3). Subsets 388+57+3 = 448 (DOCS_CHECK:30-32), which is consistent. Stale counts: RELEASE.md:25 (436/1 skipped) and SECURITY_REVIEW.md:197 (443, last round). REVIEW.md:60 (388) is correct for its range.
- **Skips.** `tests/performance/test_parallel_fetching.py:53,123,133` call `pytest.skip` when the yfinance fetch fails. The "0 skipped" result depends on network access (P3 note).
- **Env names (scripted).** I compared Settings fields in `config/settings.py` (33), `os.getenv`/`os.environ` outside tests (`CORS_ORIGINS` api/main.py:434, `DATABASE_URL` config/settings.py:109) and `process.env` in `frontend/src` (`API_INTERNAL_URL CLIENT_IP_SIGNING_KEY CLIENT_IP_TRUSTED_HEADER COOKIE_SECURE NEXT_PUBLIC_API_URL VERCEL`).
  - Every code name is in README: diff empty.
  - Code names missing from `.env.example`: `CLIENT_IP_TRUSTED_HEADER`, `VERCEL`.
  - README names missing from `.env.example`: `CLIENT_IP_TRUSTED_HEADER`, `VERCEL` (plus the non-names `CMD`, `HS256`, `INFO`).
  - `.env.example` names missing from README: none.
  - `frontend/.env.example` exists on disk but is untracked (`frontend/.gitignore:34 .env*`).
- **Links and screenshots.** Every README link target exists and is tracked: USER_GUIDE.md, docs/RELEASE.md, docs/QA.md, docs/SECURITY_REVIEW.md, docs/DOCS_CHECK.md, frontend/README.md, LICENSE. `git ls-files assets` shows `01-login.png 02-dashboard.png 03-portfolio-overview.png 04-analysis.png 05-risk.png` (199-268 KB each), which matches README.md:38-54.
- **CI claim** (README.md:129). `.github/workflows/ci.yml` runs ruff, black, isort, mypy, pytest, codecov, `npx eslint`, `npx tsc --noEmit` and `npm run build`. The claim is accurate.
- **Secrets (pattern scan, no `.env*` read by me).**
  - Grep for `sk_live/sk_test/whsec_/rk_live/price_/JWT/AKIA/ghp_/hf_/postgres://user:pass@` in the 11 documents: one hit, `.env.example:38 postgresql+psycopg2://portfolio:portfolio@localhost:5432/portfolio`. This is the local compose default, not a secret.
  - `node scripts/secrets-scan.mjs --project . --history` exits **3** with 66 hits, all of type `ENV-*`: APP_NAME 22, LOG_FILE 19, PRICE_CACHE_DIR 22, DATABASE_URL 3. These are values from the local `.env` that also appear in tracked files: default paths, the app name and the local compose DB URL (`.env.example:38`, commit `aaefe1a`). No key-pattern hits.
  - Without `--history`: exit 0, 15 hits of the same kind.
  - Per the release-prep checklist, these hits still need to be documented with the owner's OK.
- **Licenses.** `license-check --project .` exits 2 (no root package.json; the Python deps are not covered). `license-check --project frontend` exits 0: prod 60 packages, 59 ok, 1 review (`@img/sharp-win32-x64@0.35.5 Apache-2.0 AND LGPL-3.0-or-later`), 0 block.
- **S-23 reality.** `config/settings.py:153-167` only logs a warning when `CLIENT_IP_SIGNING_KEY` is unset in production. Commit `e833e62` says "Refuse to boot when set but shorter than 32 chars; warn loudly when unset". So the fail-open case named in S-23 (SECURITY_REVIEW.md:243 "a missing signing key fails open") is only mitigated, not closed.
- **known-traps.md.** None of the 3 REVIEW traps (REVIEW.md:80-82) or the 2026-11-02 SECURITY_REVIEW traps (SECURITY_REVIEW.md:248-249) appear in `E:/Main/Dev/coding-brain/known-traps.md`. I searched for "marker", "EBUSY", "set-state-in-effect", "fail closed", "time.sleep" and "signing key": 0 matches.

## Findings

- A-01 | P2 | source: step:review | at: docs/REVIEW.md:20 | quote: "| R-06 | P2 | frontend/src/app/portfolio/[id]/page.tsx:37-73, layout.tsx:21-35 (commit df3e699)" | status column `open`, but SECURITY_REVIEW.md:107 records "R-06 | fixed | `a7bac81`" | scenario: anyone reading REVIEW.md as the review record schedules work that is already done, and the review and security documents disagree | fix: add a Resolution note to REVIEW.md (R-06 fixed `a7bac81`, R-03 covered by S-02 `47cfaa9`)
- A-02 | P2 | source: step:review (security) | at: docs/SECURITY_REVIEW.md:242 | quote: "S-22: RELEASE.md says `TRUSTED_PROXY_COUNT=2`." | RELEASE.md:43 now says production `0`, changed in commit `e558398` "(S-22)" after round 3 | scenario: the owner sees S-22 as an open P2 and re-audits a value that was already corrected | fix: mark S-22 resolved by `e558398` in SECURITY_REVIEW
- A-03 | P2 | source: trap:shared-secret-fail-closed | at: docs/QA.md:254 | quote: "| S-23 | done | `e833e62` |" | the code only warns when the key is unset (config/settings.py:160-167), and RELEASE.md:19 still calls it "(S-23, open P2)" | scenario: an operator who trusts QA.md deploys without `CLIENT_IP_SIGNING_KEY`, the API boots, and every user shares one login bucket behind the BFF (the S-19 problem) | fix: QA.md and STATE:19 should say "partly fixed (short key refused; unset key warns)", matching RELEASE.md:19
- A-04 | P2 | source: step:release-prep | at: docs/RELEASE.md:25 | quote: "Test-suite state per docs: 436 passed, 1 skipped; round-2 fixes for S-03 remainder and S-18 are committed (970b581, 338748f) and marked fixed by the fix owner; an independent re-check of those two is not in the docs yet." | current count is 448/0 skipped (QA.md:246), and SECURITY_REVIEW.md:163 records the independent re-check ("S-03 and S-18 are fixed") | scenario: the release decision rests on a stale suite state that still shows a skip, which the release gate forbids | fix: update the line to 448 passed / 0 skipped at `6bf0aeb`, with the re-verification reference
- A-05 | P1 | source: checklist:docs-3 | at: README.md:227 | quote: "| `CLIENT_IP_TRUSTED_HEADER` | unset (on Vercel: `x-vercel-forwarded-for`) | `frontend/src/lib/client-ip.ts` |" | the name is absent from `.env.example` (scripted diff). `VERCEL` is also absent, which is acceptable because the platform sets it, but the docs should say so | scenario: a self-hoster copies `.env.example`, never learns about the header, and gets one shared rate-limit bucket (or a spoofable one, S-25) | fix: add a commented `# CLIENT_IP_TRUSTED_HEADER=` line (and a note that `VERCEL` is platform-set) to `.env.example`
- A-06 | P1 | source: checklist:docs-2 | at: docs/DOCS_CHECK.md:8 | quote: "| 8 | `curl <api>/llms.txt` | 404 (0 of 52 operations) | open (`rules/api.md`) |" | scenario: the docs exit item requires the counts to match or "no API"; the API exposes 52 operations, so the item fails | fix: add `/llms.txt` (code task) or get an explicit owner waiver
- A-07 | P1 | source: checklist:docs-4 | at: docs/RELEASE.md:223 | quote: "Version/CHANGELOG: there is no root `CHANGELOG.md`" | `docs/production/CHANGELOG.md` ends at "2026-10-25 — Audit hardening", with no entry for the 2026-11-01/02 security and QA fixes | scenario: the release step has no versioned changelog entry to tag | fix: add a root `CHANGELOG.md` (template) with one dated entry for the release version
- A-08 | P1 | source: checklist:release-prep-2 | at: docs/RELEASE.md:102 | quote: "Not run in this step (required by `process/release-prep.md`, still `[ ]`): `node scripts/secrets-scan.mjs --project . --history`, `pip-audit`, `npm audit --omit=dev`, license check, Lighthouse, clean-clone full build." | my run: secrets-scan `--history` exit 3 (66 ENV-value hits, all benign defaults), license-check root exit 2, frontend exit 0 with 1 review | scenario: release-prep cannot be closed without pasted scan output and owner OK on each hit | fix: paste these outputs into RELEASE.md, classify the 4 hit types, and get the owner's OK; add `.secrets-allow` only with that OK
- A-09 | P2 | source: rule:audit (Read line) | at: docs/RELEASE.md:227 | quote: "Each needs the owner's explicit yes: push `chore/audit`; open a PR to `main`" | this is the last section; the file has no `Read:` line, and neither does the 2026-11-02 section of DOCS_CHECK.md (lines 1-57; the only `Read:` at :162 belongs to the 2026-10-25 run) | scenario: `read-audit` cannot confirm the author read the step files | fix: append `Read: <paths>` to both
- A-10 | P3 | source: step:docs | at: README.md:133 | quote: "The repository has no SQLite option (see the note below)." | README.md:178 says SQLite is the default `DATABASE_URL` and the tests use it | scenario: a reader cannot tell whether SQLite exists | fix: "SQLite is the default URL but the API does not boot on it (see below)"

## Contradictions across documents

| Topic | Doc A | Doc B | Reality |
|---|---|---|---|
| TRUSTED_PROXY_COUNT (production) | RELEASE.md:43: `0` | SECURITY_REVIEW.md:242: "RELEASE.md says `TRUSTED_PROXY_COUNT=2`" | RELEASE is current (`e558398`). RELEASE.md:76-79 still says "documented as `2`" / "start with `2`", but only for the unsigned fallback (stated at :72). Consistent, though easy to misread |
| S-23 status | QA.md:254 "done"; STATE:19 "fixed" | RELEASE.md:19 "(S-23, open P2)"; SECURITY_REVIEW.md:243 open | Partly fixed: short key refused, unset key only warns (config/settings.py:153-167) |
| S-22 status | SECURITY_REVIEW.md:242 open | RELEASE.md:43 fixed | fixed by `e558398` |
| R-06 status | REVIEW.md:20 open | SECURITY_REVIEW.md:107 fixed `a7bac81` | fixed |
| P0/P1 counts | SECURITY_REVIEW.md:7 "P0 = 1, P1 = 3" (first pass) | SECURITY_REVIEW.md:233 "P0 = 0, P1 = 0"; QA.md:3 "no P0/P1 open"; REVIEW.md:6 P1 0 open | Consistent once read as history. The final P1 = 0 is "by code review"; the live re-run is pending (SECURITY_REVIEW.md:233). QA RL-1..19 later covered the S-18/S-19 behaviour live (QA.md:200-218), but SECURITY_REVIEW does not record that |
| Suite count | RELEASE.md:25 436/1 skipped | QA.md:246, STATE:19, DOCS_CHECK:28 448/0 | 448/0 (orchestrator re-run on `6bf0aeb`) |
| core coverage | 78.35% (QA, STATE, Docker) | 78.45% (DOCS_CHECK:28, host venv) | both real; the environment difference is not explained |
| Independent re-check of S-03/S-18 | RELEASE.md:25 "not in the docs yet" | SECURITY_REVIEW.md:159-163 round 2 re-verified | done |

## Exit checklists

Stack adaptation, applied throughout: there is no SPEC.md, and ACs come from `docs/FINISH_PLAN.md` (project CLAUDE.md). The DAL is the SQLAlchemy repositories plus services. `pnpm build/lint/tsc` maps to `npm run build`, `npx eslint` and `npx tsc --noEmit`. The TEST DB is a throwaway Postgres `pmp_qa_test` or the SQLite pytest fixtures.

### review (`process/review.md`)
| # | Item | Result | Evidence |
|---|---|---|---|
| 1 | REVIEW.md scope, reviewers (and skips), commands, findings with status; P0 = P1 = 0 open | met (stale status) | REVIEW.md:4-10 scope and reviewers, :58-65 commands, :13-21 findings; P1 0 open (:6). R-06 status stale (A-01). Security: SECURITY_REVIEW.md:233 "P0 = 0, P1 = 0" |
| 2 | Security categories checked or n.a. with reason; money paths traced (tier A) | met | SECURITY_REVIEW.md:19-31 (10 categories, 6 n.a. with a reason; 1, 2, 5, 7, 8 tier A traced, :5) |
| 3 | UI: design screenshots 375/768/1440 from the running app, zero ban fails | **not met** | REVIEW.md:8 "design-reviewer ... not run". QA.md:233 checked viewports but took no screenshots or ban check. The README screenshots are 1440 only (README.md:35) |
| 4 | Prod build + tests re-run by reviewer after fixes, pasted | met | REVIEW.md:58-65 (388 passed, Docker `next build` 22/22). The security fixes were re-verified with 20 targeted tests (SECURITY_REVIEW.md:211); the full 448 run is in QA.md:246 |
| 5 | New traps appended to `known-traps.md` | **not met** | REVIEW.md:80-82 and SECURITY_REVIEW.md:248-249 list them; 0 matches in `coding-brain/known-traps.md` |

### qa (`process/qa.md`)
| # | Item | Result | Evidence |
|---|---|---|---|
| 1 | Tier-A areas have strict tests incl. error and auth paths (listed) | met | QA.md:22-34 risk map, :78-187 108 checks, :198-218 RL-1..19 |
| 2 | Prod build, typecheck, full suite 0 failed / 0 skipped, pasted | met | QA.md:41 "443 passed, 0 failed, 0 skipped"; :246 "448 passed, 0 skipped", tsc clean, Docker `next build` OK |
| 3 | Every P1 AC verified in an AC table | **not met** | QA.md has no AC table. The FINISH_PLAN ACs are verified in REVIEW.md:29-43, not in QA. Partial substitute: tier-A matrix |
| 4 | Browser pass on prod build: 0 console errors or each explained | met | QA.md:236 (401 on /register explained; F-4 prefetch 404 explained) |
| 5 | Tests on the TEST DB only (name shown) | met | QA.md:11 and :54 `pmp_qa_test` |
| 6 | GREEN only if nothing open in tier A/B | met | QA.md:3 RED. After the Resolution (QA.md:244-254), F-5 is still open, so RED stands; QA.md does not restate the verdict after the resolution (P3) |
| 7 | `docs/checks/qa.md` exists, dispositions, STATE `audit/qa` | **not met** | `docs/checks/` does not exist; STATE has no `audit/qa` line |

### docs (`process/docs.md` + saas row 8)
| # | Item | Result | Evidence |
|---|---|---|---|
| 1 | Every documented command run, results listed | met | DOCS_CHECK.md:7-42 (32 rows, 2 documented failures: SQLite and isort with an in-repo venv) |
| 2 | `/llms.txt` count equals route count, or "no API" | **not met** | DOCS_CHECK.md:8 "0 of 52" (A-06) |
| 3 | README env names == `.env.example` names (diff empty) | **not met** | scripted diff: `CLIENT_IP_TRUSTED_HEADER`, `VERCEL` (A-05). DOCS_CHECK.md:46 checked only README ⊇ code |
| 4 | CHANGELOG entry for the release version, with date | **not met** | no root CHANGELOG; docs/production/CHANGELOG.md:5 last entry 2026-10-25 (A-07) |
| 5 | No aspirational features | met | README features, CI and security claims spot-checked (ci.yml steps; settings guards; plans.py:16). Limitations are stated honestly (README.md:333-340) |
| saas-8 | Owner guide: change plans/prices, refund, read MRR, rotate keys; Stripe CLI webhook forwarding in README | **not met** | DOCS_CHECK.md:157 "Owner guide ... is not written". RELEASE.md covers only JWT rotation side effects (:36) and rollback (:169-177). README does not document Stripe CLI forwarding (README.md:291 only names the script) |

### release-prep (`process/release-prep.md` + saas row 9)
| # | Item | Result | Evidence |
|---|---|---|---|
| 1 | All checklist items ticked or waived by the user (cite) | **not met** | many `[ ]` agent-side items (RELEASE.md:99, :102); owner items are legitimately open |
| 2 | `secrets-scan --history` exit 0 and `license-check` exit 0, pasted, or hits documented with owner OK | **not met** | RELEASE.md:102 not run. My run: exit 3 (66 benign ENV hits) and license root exit 2 / frontend exit 0 (A-08) |
| 3 | Final clean-install prod build + tests green, pasted | **not met** in the document | RELEASE.md:102 "clean-clone full build" not run. The orchestrator's run on `6bf0aeb` (448 passed, builds OK) exists but is not pasted into RELEASE.md |
| 4 | `git status` clean; `.env*` untracked; CHANGELOG and version consistent | partly met | status clean at audit start; only `.env.example` is tracked (`git ls-files`). CHANGELOG missing (A-07) |
| 5 | List of owner-only actions delivered in Russian | **not met** in the repo | RELEASE.md is English only (no Cyrillic). It may have been delivered in chat, which cannot be verified from the documents |
| 6 | `docs/checks/release-prep.md`, dispositions, STATE `audit/release-prep` | **not met** | no `docs/checks/`; no STATE line |
| saas-9 | Owner checklist (Stripe live, DNS, legal human review, tax), backups + restore test, env table per environment, rollback plan, live keys never seen | met as a checklist | RELEASE.md:107-134 Stripe, :141-155 backups/restore, :30-66 env table, :169-177 rollback, :4 "live keys were never seen" |

### saas router "Done when" (adapted)
| Item | Result | Note |
|---|---|---|
| Sign up, pay with a test card, access via webhook, portal cancel, delete account (E2E on prod build) | not met | No Stripe test keys; the webhook path was proven with a signed fake (QA.md:14, HOOK rows); no real checkout or portal E2E (QA.md:240); no account deletion (F-5, QA.md:66) |
| Tier-A tests incl. duplicate/out-of-order webhooks; 0 failed / 0 skipped on the TEST DB | met | QA.md:135-149, :246 |
| Build, lint, typecheck green; no secrets in repo/bundle; `.env.example` complete; no seeded demo creds | partly | Builds and lint green (orchestrator run). `.env.example` is incomplete (A-05). Bundle secret scan not recorded. Demo user only in a throwaway screenshot stack (DOCS_CHECK.md:44), nothing seeded in the repo (not exhaustively verified) |
| Reviewer P0 = P1 = 0; design gate with >= 2 rounds, zero ban fails | partly | P0 = P1 = 0 met; the design gate was never run (review item 3) |
| Module done-when lists (auth, payments, email, legal, analytics, error-monitoring) | not met / n.a. | email, analytics and Sentry are not implemented (RELEASE.md:164, :220-221); legal pages not verified (:219) |
| RELEASE.md owner checklist delivered; live steps only after yes | met | RELEASE.md:4, :227 |
| N/A for this stack | n.a. | Drizzle (SQLAlchemy + Alembic used), Better Auth / Google OAuth (email+password JWT, SECURITY_REVIEW.md:14), `pnpm` (npm), YooKassa (no RUB billing), Supabase MCP (no account access), Server Actions (FastAPI routes) |

## STATE.md / FINISH_PLAN.md vs git

- FINISH_PLAN ticks all have commits: B1-B4 `0fe0ea3`, B5 `3416915`, B6 `d52f054`, F1 `7e6b6d1`, F2 `df3e699`, F3 `467b45c` (+`1e7adf2`/`0c6267c` lib tracking), F4 `99e1212`, F5 `1d4bd56`, D1 `0574fe1`, D2 `0574fe1` (deploy checklist) and later `3a97628`.
- STATE lines 14-20 match the commits they cite.
- No STATE line for: the review step (`a29d108`, with its fixes `43b0f04`/`9b15498`), the release checklist (`3a97628`), the round-3 security docs and the S-22 RELEASE fix (`bd705b6`, `e558398`), or any `audit/<step>`. That breaks the "one STATE line per task" rule (P2).

## Notes (non-blocking)

- N-01 | opinion | The riskiest assumption across these documents is that "P1 = 0" holds in production. It depends on `CLIENT_IP_SIGNING_KEY` being set on both Vercel and HF, and an unset key only warns (A-03). Making the API refuse to boot without the key would remove the dependency on the owner remembering it.
- N-02 | opinion | `frontend/.env.example` exists locally but is ignored by `frontend/.gitignore:34`. Either track it with `!.env.example` or delete it, so frontend env names live in one tracked place.
- N-03 | opinion | Performance tests skip on network failure (`test_parallel_fetching.py:53`). "0 skipped" holds only with network access; a CI run without it would show skips.

Read: E:/Main/Dev/coding-brain/rules/audit.md, E:/Main/Dev/coding-brain/process/review.md, E:/Main/Dev/coding-brain/process/qa.md, E:/Main/Dev/coding-brain/process/docs.md, E:/Main/Dev/coding-brain/process/release-prep.md, E:/Main/Dev/coding-brain/routers/projects/webapp/saas.md, E:/Main/Dev/coding-brain/known-traps.md (grep), E:/Main/Dev/coding-brain/scripts/secrets-scan.mjs (header), docs/REVIEW.md, docs/SECURITY_REVIEW.md, docs/QA.md, docs/DOCS_CHECK.md, docs/RELEASE.md, README.md, USER_GUIDE.md, .env.example (names), docs/STATE.md, docs/FINISH_PLAN.md, CLAUDE.md, config/settings.py (fields, guard), services/billing/plans.py, .github/workflows/ci.yml (grep), tests/ (grep), docs/production/CHANGELOG.md (headings)
