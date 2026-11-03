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
