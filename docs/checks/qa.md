# Audit: qa (docs/QA.md)
mode: shadow
author-model: sonnet · auditor-model: opus

## Verdict
FIX
The suite at HEAD was not run after b5f1a3c added tests (exit-check 2), and the F5 AC row describes a different item than FINISH_PLAN F5 and is left unverified (exit-check 3). Both can be fixed from the documents plus a Docker re-run.

## Checklist
| # | Item | Result | Evidence |
|---|---|---|---|
| 1 | Every tier-A area has strict tests incl. error and auth paths (list them) | PASS | docs/QA.md:24 "Auth: register/login/JWT/session" and docs/QA.md:26 "Billing/webhooks: signature, dedupe, ordering, past_due, cancel, resubscribe, expiry"; matrix lines 78-187 |
| 2 | Production build, typecheck, full suite green with 0 failed and 0 skipped; output pasted | FAIL | docs/QA.md:254 "The `b5f1a3c` tests were written on 2026-11-03 but not run: the Docker daemon was unavailable; rerun before release." |
| 3 | Every P1 AC is verified (test id or browser step) in the AC table | FAIL | docs/QA.md:270 "F5 holdings table fixes" result "**not verified in browser**" |
| 4 | Browser pass on the production build: 0 console errors on critical flows (or each explained) | PASS | docs/QA.md:236 "No uncaught page errors." (401 and RSC 404 explained, F-4) |
| 5 | Tests ran on the TEST database only (name shown in the log) | PASS | docs/QA.md:54 "Test database: `pmp_qa_test` on the throwaway postgres only" |
| 6 | Verdict GREEN only if nothing open in tier A/B | PASS | docs/QA.md:275 "**RED stands.**" |
| 7 | Audit done: docs/checks/qa.md exists, every finding has a Disposition, STATE has audit/qa | PENDING | this file; Disposition and the STATE line are the orchestrator's part |

## Findings
- A-01 | P1 | source: checklist:2 | at: docs/QA.md:254 | quote: "The `b5f1a3c` tests were written on 2026-11-03 but not run: the Docker daemon was unavailable; rerun before release." | scenario: b5f1a3c changed api/main.py and config/settings.py and added tests/unit/test_llms_txt.py (llms.txt checked against live OpenAPI operations); the last green count (448 passed) predates it, so the suite at HEAD may be red and the release step would rely on an unproven green | fix: re-run pytest in the Dockerfile.api image at HEAD and paste the new counts in the Resolution section
- A-02 | P1 | source: F5 | at: docs/QA.md:270 | quote: "F5 holdings table fixes" | scenario: FINISH_PLAN F5 is "buy_hold portfolio overview shows $0 / no holdings: fix", but the AC row maps F5 to the F-3 overflow fix and ends unverified, so the real AC has no test id or browser step in the table and release reads F5 as unverified for the wrong reason | fix: restate F5 as the buy_hold overview fix and cite a browser step on a buy_hold portfolio (or a test), keep F-3 separately

## Notes (non-blocking)
- N-01 | opinion | at: docs/QA.md:5 | Header "Code under test: commit `f10bc42`" is stale since b5f1a3c changed app code; the new public /llms.txt route is not in the UNAUTH-1 public list (line 190). Updating the header with the re-run in A-01 would close both.
- N-02 | opinion | at: docs/QA.md:268 | FINISH_PLAN F3 also promises next.config security headers; the AC row cites only the allowlist although the HDR-web rows (lines 173-175) already verify the headers. Citing them would make the row complete.
- N-03 | opinion | at: docs/QA.md:265 | B6 row cites 78.35% to the Resolution section, which does not state coverage; the plan target is 75%, not 78%. The value is consistent with STATE, only the citation and threshold wording differ.
- N-04 | opinion | riskiest assumption: that the F-2 price rounding fix (line 251) works without a browser re-test of the Add Transaction submit, a critical flow that silently failed before.

## Disposition
| id | verdict | by | note |
|---|---|---|---|
| A-01 | valid | orchestrator | tests of b5f1a3c not run (Docker daemon hung); rerun pending |
| A-02 | valid | orchestrator | fixed: F5 row now maps to the buy_hold fix, F-3 split out |

Read: E:/Main/Dev/coding-brain/rules/audit.md, E:/Main/Dev/coding-brain/process/qa.md, docs/QA.md, docs/FINISH_PLAN.md, docs/REVIEW.md (grep), docs/RELEASE.md (grep), docs/STATE.md (grep), tests/ (grep for cited test names), frontend/Dockerfile and Dockerfile.api (grep USER), git log / git show --stat b5f1a3c

## Re-check 2026-11-05
mode: shadow
author-model: sonnet · auditor-model: opus

### Verdict
OWNER
QA.md still ends RED: F-5 (account deletion) needs the owner; the F-2/F-3 browser checks and a frontend typecheck/build after 81bd623 can be fixed by the author now that Docker is back.

### Status of earlier findings
| id | status | evidence |
|---|---|---|
| A-01 | resolved | docs/QA.md:285 "**451 passed, 0 failed, 0 skipped**" incl. "Includes `test_llms_txt.py` (3)"; tests exist: tests/unit/test_llms_txt.py (3 test functions), tests/unit/test_security_hardening.py:252 test_production_refuses_unset_client_ip_signing_key; test fix in f613e63 (git show --stat). Pasted output accepted, not reproduced by the auditor. |
| A-02 | resolved | docs/QA.md:270 "F5 buy_hold overview shows $0 / no holdings"; F-3 split out to line 271 |

### Checklist
| # | Item | Result | Evidence |
|---|---|---|---|
| 1 | Every tier-A area has strict tests incl. error and auth paths | PASS | docs/QA.md:24-34 risk map, matrix lines 78-187 (unchanged) |
| 2 | Production build, typecheck, full suite green, 0 failed / 0 skipped, pasted | FAIL | backend: docs/QA.md:285 "**451 passed, 0 failed, 0 skipped**"; frontend: 81bd623 (2026-11-05 14:02, after f613e63 at 13:47) changed 4 tsx files and the re-run table at docs/QA.md:286 lists only backend linters (see A-04) |
| 3 | Every P1 AC verified in the AC table | PASS | docs/QA.md:260-272; the F5 row now maps to the buy_hold fix; F-3 (line 271) is a P3 QA finding, not an AC |
| 4 | Browser pass on the production build, 0 console errors or explained | PASS | docs/QA.md:236 "No uncaught page errors." |
| 5 | Tests on the TEST database only | PASS | docs/QA.md:54 "Test database: `pmp_qa_test` on the throwaway postgres only" |
| 6 | Verdict GREEN only if nothing open in tier A/B | PASS | docs/QA.md:276 "**RED stands.**" (honest RED) |
| 7 | Audit done: checks file, Dispositions, STATE audit/qa | PASS | Disposition rows A-01, A-02 above; docs/STATE.md:34 "audit/qa - FIX, 2 findings (2 valid) [shadow]" |
| R7 | saas router row 7 exit: docs/QA.md GREEN | FAIL | docs/QA.md:276 (see A-03) |

### Findings
- A-03 | P1 | source: step:qa | at: docs/QA.md:276 | quote: "Open in tier A/B: F-5 (no account-deletion endpoint) and the unverified browser fixes F-2 and F-3." | scenario: F-2 was a critical flow where the Add Transaction click silently did nothing; its fix (e1af502) and F-3 (d69385e) were never re-run in a browser, so QA stays RED and release-prep starts from a non-GREEN input; F-5 also needs the owner's decision | fix: re-run Playwright steps 3 and 6 (auto-filled price submit, 375 px overflow) on the production web image and paste the results; record the owner's F-5 decision
- A-04 | P1 | source: checklist:2 | at: docs/QA.md:286 | quote: "`ruff check`, `black --check`, `isort --check-only`, `mypy` (126 files)" | scenario: 81bd623 changed frontend/src/app/create/page.tsx (+27 lines) and the transactions, portfolios and billing pages after the 2026-11-05 re-run; no tsc, eslint or next build is recorded for that code, so a type or build error would first show on the Vercel deploy | fix: run npx tsc --noEmit, eslint and the frontend Docker build at HEAD and add the results to the Re-run table

### Notes (non-blocking)
- N-05 | opinion | at: docs/QA.md:285 | The Re-run row says the copy was made with git archive HEAD but names no commit; the test fix it mentions was committed in f613e63 together with the doc, so the commit under test is ambiguous.
- N-06 | opinion | at: docs/QA.md:5 | The header "Code under test: commit `f10bc42`" and the opening verdict line (F-1 open) are still stale relative to the Resolution and Re-run sections (repeat of N-01).
- N-07 | opinion | riskiest assumption: the python-jose PYSEC-2026-4298 waiver relies on decode passing explicit algorithms; core/auth/tokens.py:28-32 shows algorithms=[settings.jwt_algorithm] with the symmetric secret, so the code supports it.

### Disposition (re-check)
| id | verdict | by | note |
|---|---|---|---|
| A-01 | valid (resolved) | orchestrator | 451 passed, 0 skipped on 2026-11-05 (QA.md re-run); quote text changed by the fix, cite-check INVALID is historical |
| A-02 | valid (resolved) | orchestrator | F5 row maps to the buy_hold fix; historical |
| A-03 | valid (resolved except F-5) | orchestrator | F-2 and F-3 browser-verified 2026-11-05 (QA.md "Browser verification"); F-5 stays an owner decision, RED stands on it alone |
| A-04 | valid (resolved) | orchestrator | tsc clean, eslint 0 errors, `next build` via docker (web image) and clean-clone build at `609b91d`, recorded in QA.md and RELEASE.md section 15 |

Read (re-check): E:/Main/Dev/coding-brain/rules/audit.md, E:/Main/Dev/coding-brain/process/qa.md, docs/QA.md, docs/checks/qa.md, docs/STATE.md, core/auth/tokens.py, requirements.lock.txt (grep), tests/unit/test_security_hardening.py (grep), tests/unit/test_llms_txt.py (grep), git log / git show --stat f613e63 332fa3d 81bd623
