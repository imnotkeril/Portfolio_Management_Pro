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

## Re-check 2026-11-05
mode: shadow
author-model: sonnet · auditor-model: opus

### Verdict
OWNER
The owner OK for the 68 secrets-scan history hits and the F-5 decision are still open, and QA is RED. The clean build predates 81bd623, Lighthouse has not been run or waived, and several lines in sections 1, 3 and 13 contradict section 15.

### Status of earlier findings
| id | status | evidence |
|---|---|---|
| A-01 | resolved | docs/RELEASE.md:8 "the QA verdict is still **RED**" |
| A-02 | partly resolved | pytest, pip-audit and clean-clone builds are now recorded (docs/RELEASE.md:259-261), but the clean build is at f613e63, before the frontend commit 81bd623 (see A-06) |
| A-03 | resolved as text; item open (owner) | docs/RELEASE.md:219 "has no privacy, terms or refund route, so there is nothing to link yet" |
| A-04 | resolved | docs/RELEASE.md:223 "root `CHANGELOG.md` exists"; CHANGELOG.md has an Unreleased section |
| A-05 | resolved | docs/RELEASE.md:23 "`/llms.txt` is served since `b5f1a3c`" (its 'test not run yet' is now stale, see A-08) |

### Checklist
| # | Item | Result | Evidence |
|---|---|---|---|
| 1 | All checklist items ticked or explicitly waived by the user in chat | FAIL | docs/RELEASE.md:102 "Lighthouse, clean-clone full build"; Lighthouse has no run and no user waiver (see A-07). Owner-only items (sections 4-7, 9) are legitimately open. |
| 2 | secrets-scan --history exit 0 and license-check exit 0, or each hit documented with the user's OK | FAIL (owner) | docs/RELEASE.md:242 "Owner OK needed for the 68 history hits"; no owner OK in docs/STATE.md |
| 3 | Final clean-install production build + tests green, pasted | FAIL | docs/RELEASE.md:261 (clean clone at f613e63); HEAD is 81bd623 with frontend changes after it (see A-06) |
| 4 | git status clean; .env* untracked; CHANGELOG and version consistent | PASS | auditor run: git status --short empty, git ls-files shows only .env.example; docs/RELEASE.md:223 |
| 5 | Owner-only list in Russian | PASS | docs/RELEASE.md:244 "## 14. Owner-only actions (Russian summary)" |
| 6 | Audit done: checks file, Dispositions, STATE audit/release-prep | PASS | Disposition rows A-01..A-05 above; docs/STATE.md:36 "audit/release-prep - OWNER, 5 findings (5 valid) [shadow]" |

### Findings
- A-06 | P1 | source: checklist:3 | at: docs/RELEASE.md:261 | quote: "Clean clone of `chore/audit` at `f613e63` (no `.env`)" | scenario: commit 81bd623 (2026-11-05 14:02, after f613e63 at 13:47) changed four frontend pages incl. +27 lines in create/page.tsx; the release candidate at HEAD has no clean-install build, so a build break would first appear on Vercel | fix: repeat the clean-clone frontend/Dockerfile build (and tsc) at HEAD and record the commit
- A-07 | P1 | source: checklist:1 | at: docs/RELEASE.md:102 | quote: "Lighthouse, clean-clone full build" | scenario: process release-prep step 6 requires Lighthouse scores on the production build; none are recorded and no user waiver is cited, so the perf/a11y item is neither ticked nor waived and the checklist cannot close | fix: run Lighthouse on the production web image and paste the scores, or cite the owner's waiver
- A-08 | P2 | source: step:release-prep | at: docs/RELEASE.md:25 | quote: "added tests that have not been run yet" | scenario: section 1 (also line 23 and line 240 'Not run in this session') contradicts section 15 (451 passed, pip-audit run); the owner reading the blockers first sees an unrun suite | fix: update lines 23, 25, 102 and 240 to point to section 15
- A-09 | P2 | source: step:release-prep | at: docs/RELEASE.md:260 | quote: "ecdsa and python-jose advisories waived (no fix version, not reachable: explicit algorithms, symmetric secret)" | scenario: process release-prep step 2 requires the user's OK to document an audit hit as a false positive; this waiver is the agent's and section 14 item 6 does not ask the owner about it, so the owner never decides on shipping two known advisories | fix: add the two advisories to the owner decisions in section 14

### Notes (non-blocking)
- N-06 | opinion | at: docs/RELEASE.md:3 | The header date 2026-11-01 is older than sections 13 and 15; a 'last updated' line would help.
- N-07 | opinion | riskiest assumption: HF Spaces runs exactly one uvicorn process with the same CLIENT_IP_SIGNING_KEY as Vercel (D1/D2); still not observed live (repeat of N-04).

### Disposition (re-check)
| id | verdict | by | note |
|---|---|---|---|
| A-01 | valid (resolved) | orchestrator | historical, fixed in 18713cd/ccd19d8 |
| A-02 | valid (resolved) | orchestrator | clean-clone rebuilt at HEAD `609b91d`, both images ok (RELEASE.md section 15) |
| A-03 | valid (open, owner) | orchestrator | legal pages need owner texts |
| A-04 | valid (resolved) | orchestrator | historical |
| A-05 | valid (resolved) | orchestrator | historical |
| A-06 | valid (resolved) | orchestrator | same as A-02 |
| A-07 | valid (open, owner) | orchestrator | Lighthouse not run: `npx` download denied by the permission system; owner runs it or waives it |
| A-08 | valid (resolved) | orchestrator | stale lines rewritten in c9c39c9 |
| A-09 | valid (open, owner) | orchestrator | ecdsa / python-jose advisories listed as owner items in RELEASE.md section 15 |

Read (re-check): E:/Main/Dev/coding-brain/rules/audit.md, E:/Main/Dev/coding-brain/process/release-prep.md, docs/RELEASE.md, docs/QA.md, docs/checks/release-prep.md, docs/STATE.md, CHANGELOG.md (head), core/auth/tokens.py, requirements.lock.txt (grep), git status / git ls-files / git log / git show --stat f613e63 332fa3d 81bd623
