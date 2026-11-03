# Audit: release-prep (docs/RELEASE.md)
mode: shadow
author-model: unknown (not stated in the document) · auditor-model: claude-opus-5-5

## Verdict
OWNER
Upstream QA verdict is RED (F-5 account deletion is an owner decision) while RELEASE calls the code release-candidate ready; the 68 secrets-scan hits need the owner's OK; the clean build/tests are not run; two factual errors (CHANGELOG, legal pages) are author-fixable.

## Checklist
| # | Item | Result | Evidence |
|---|---|---|---|
| 1 | All checklist items ticked or explicitly waived by the user in chat (cite) | FAIL | docs/RELEASE.md:102 "Lighthouse, clean-clone full build" still not run, no waiver cited; docs/RELEASE.md:219 "Whether these pages exist in the app was not verified." Owner-only items (Stripe live sec. 5, DNS sec. 9, Supabase backups sec. 6, monitoring sec. 7) are legitimately open and are not counted as defects. |
| 2 | `secrets-scan --history` exit 0 and `license-check` exit 0 (output pasted), or each hit documented with the user's OK | FAIL (owner) | Re-run by auditor: secrets-scan exit 3, 68 hits, all ENV-* (APP_NAME, LOG_FILE, PRICE_CACHE_DIR, DATABASE_URL), matching docs/RELEASE.md:235; license-check frontend exit 0 (prod 60, review 1 sharp-win32 LGPL, block 0), root exit 2 (no package.json), matching lines 236-237. Hits documented but owner OK pending: docs/RELEASE.md:242 "Owner OK needed for the 68 history hits" |
| 3 | Final clean-install production build + tests green, pasted | FAIL | docs/RELEASE.md:240 "Not run in this session (the Docker daemon would not start): `pip-audit`, clean-clone Docker build and pytest." Stated honestly; b5f1a3c tests unrun (also docs/QA.md:254). |
| 4 | `git status` clean; `.env*` untracked; CHANGELOG and version consistent | FAIL | git status clean and `git ls-files` shows only `.env.example` (auditor run). But docs/RELEASE.md:223 "there is no root `CHANGELOG.md`" is false: CHANGELOG.md is tracked (commit 34741ec, section 'Unreleased'); frontend/package.json version 0.1.0, no bump. |
| 5 | A list of things only the user can do (keys, DNS, accounts) delivered in Russian | PASS | docs/RELEASE.md:244 "## 14. Owner-only actions (Russian summary)" |
| 6 | Audit done per rules/audit.md: docs/checks/release-prep.md exists, every finding has a Disposition, STATE has audit/release-prep | PENDING | This report creates the file; Disposition and the STATE line are the orchestrator's next step. |
| R9 | saas router row 9: owner checklist (Stripe live, email domain, OAuth, prod DB, DNS, legal review, tax), backups + restore test, env table per environment, rollback plan | PASS | Stripe sec. 5, tax docs/RELEASE.md:121, email/OAuth not used sec. 11, env tables sec. 2.1/2.2, restore procedure sec. 6, rollback sec. 8. `module:seo-geo` not addressed (see N-02). |

## Findings
- A-01 | P1 | source: step:release-prep | at: docs/RELEASE.md:8 | quote: "Code side: ready for a release candidate." | scenario: process/release-prep.md requires docs/QA.md GREEN as input, but docs/QA.md:275 says 'RED stands' (F-5 no account deletion, F-2/F-3 browser fixes unverified); RELEASE never mentions the RED verdict, so the owner or the release step proceeds believing QA passed. | fix: state the QA RED verdict and its open items (F-5, F-2, F-3) in sections 0 and 1, and make F-5 an explicit owner decision blocking release.
- A-02 | P1 | source: checklist:3 | at: docs/RELEASE.md:240 | quote: "Not run in this session (the Docker daemon would not start): `pip-audit`, clean-clone Docker build and pytest." | scenario: the fail-closed boot guard and /llms.txt from b5f1a3c ship with tests never executed and Python dependencies never audited; a failing test or vulnerable pin would first surface on the production Space. | fix: run clean-clone Docker build, pytest and pip-audit once Docker is available and paste the output.
- A-03 | P1 | source: checklist:1 | at: docs/RELEASE.md:219 | quote: "Whether these pages exist in the app was not verified." | scenario: frontend/src/app has no privacy, terms or refund route; section 5 step 7 tells the owner to link these pages in the footer and Stripe settings, so the owner reaches Stripe live activation with nothing to review or link. | fix: record that the legal pages do not exist and add creating them as a code task before live billing.
- A-04 | P1 | source: checklist:4 | at: docs/RELEASE.md:223 | quote: "there is no root `CHANGELOG.md`" | scenario: CHANGELOG.md is tracked since commit 34741ec with an 'Unreleased' section; the release step following this line would create a new changelog or skip reconciling it with the version (frontend/package.json 0.1.0), so CHANGELOG and version consistency is never checked. | fix: reference the existing CHANGELOG.md and state the target version for the release.
- A-05 | P2 | source: step:release-prep | at: docs/RELEASE.md:23 | quote: "`/llms.txt` is missing (rule `rules/api.md`; the API has 52 operations)." | scenario: /llms.txt is served since b5f1a3c (api/main.py:483, and docs/RELEASE.md:25 says so); the owner reading section 1 sees a stale open item and may schedule duplicate work. | fix: mark D5 /llms.txt as done (b5f1a3c).

## Notes (non-blocking)
- N-01 | opinion | at: docs/RELEASE.md:102 | Section 3 still says secrets-scan, npm audit and license check were not run, while section 13 (2026-11-03) shows them run; a date qualifier would avoid confusion.
- N-02 | opinion | at: docs/RELEASE.md:102 | SEO/GEO (process item 5) and Lighthouse perf/a11y (item 6) have no section; if the app has no public marketing pages, saying so explicitly would close item 5.
- N-03 | opinion | at: docs/RELEASE.md:237 | Python dependency licenses are not covered by any check (license-check covers only npm); a pip-licenses run would close the gap.
- N-04 | opinion | at: docs/RELEASE.md:79 | Riskiest assumption: the production rate-limit keying relies on CLIENT_IP_SIGNING_KEY being identical on Vercel and HF and on one uvicorn process (D2); neither has been observed live.
- N-05 | opinion | at: docs/RELEASE.md:5 | Owner-only items (Stripe live, DNS, Supabase backups, monitoring, secret confirmation) are correctly marked as open owner actions and are not counted as defects.

## Disposition
| id | verdict | by | note |
|---|---|---|---|
| A-01 | valid | orchestrator | fixed: section 0 now states the QA verdict RED |
| A-02 | valid | orchestrator | pip-audit, clean build, pytest need Docker; pending |
| A-03 | valid | orchestrator | fixed in text: legal pages confirmed missing; owner must supply texts |
| A-04 | valid | orchestrator | fixed: CHANGELOG exists; stays Unreleased until the release step |
| A-05 | valid | orchestrator | fixed: llms.txt noted as served |

Read: E:/Main/Dev/coding-brain/rules/audit.md, E:/Main/Dev/coding-brain/process/release-prep.md, E:/Main/Dev/coding-brain/routers/projects/webapp/saas.md, docs/RELEASE.md, docs/QA.md, docs/REVIEW.md, CHANGELOG.md, frontend/package.json, pyproject.toml, api/main.py (grep), frontend/src/app (listing)
