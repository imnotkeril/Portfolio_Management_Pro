# Audit: review (docs/REVIEW.md + docs/SECURITY_REVIEW.md)
mode: shadow
author-model: opus · auditor-model: claude-opus-5-5 (same family as the author; the pairing table asks for fable or sonnet, so this OK/FIX carries less weight)

## Verdict
FIX
Three exit-check items fail: no design screenshots, no reviewer re-run of build + tests after the last fix rounds, security traps not appended to known-traps.md.

## Checklist
| # | Item | Result | Evidence |
|---|---|---|---|
| 1 | REVIEW.md lists scope, reviewers that ran (and why skipped), commands, every finding with status; P0 = P1 = 0 open | PASS | docs/REVIEW.md:10 "Scope: `git diff 345d95f..0574fe1` (15 commits, code only)"; docs/SECURITY_REVIEW.md:233 "**P0 = 0, P1 = 0** (by code review plus 20 passing targeted tests" (status inconsistencies are A-04, A-05) |
| 2 | Security section records each category as checked or n.a. with reason; money projects show tier-A paths traced | PASS | docs/SECURITY_REVIEW.md:5 "Money project, so categories 1, 2, 5, 7 and 8 are tier A: every path traced, nothing sampled."; table rows 1-10 at docs/SECURITY_REVIEW.md:22-31 |
| 3 | UI projects: design section shows screenshots at 375/768/1440 from the running app and zero ban fails | FAIL | docs/REVIEW.md:8 "not run (per orchestrator: design covered by none)"; no docs/design/ folder exists |
| 4 | Production build + tests re-run by `reviewer` after fixes (output pasted) | FAIL | docs/REVIEW.md:58 "## Final verification (re-run by this reviewer after fixes, HEAD 9b15498)"; fixes 47cfaa9..b5f1a3c came later |
| 5 | New traps appended to `known-traps.md` | FAIL | docs/SECURITY_REVIEW.md:89 "**stripe-python 16: `StripeObject` is not a dict.**" not present in E:/Main/Dev/coding-brain/known-traps.md (only the 3 REVIEW.md traps and 2 round-3 traps are there) |

## Findings
- A-01 | P1 | source: checklist:3 | at: docs/REVIEW.md:8 | quote: "not run (per orchestrator: design covered by none)" | scenario: the Next.js UI ships with no 375/768/1440 check; a layout break on a main page (QA F-3 narrow-screen table overflow was found later and fixed without browser verification) reaches users unseen | fix: run design-reviewer on the running app and add the screenshots and ban results, or record an owner waiver
- A-02 | P1 | source: checklist:4 | at: docs/REVIEW.md:58 | quote: "## Final verification (re-run by this reviewer after fixes, HEAD 9b15498)" | scenario: billing, auth, rate-limit and Monte Carlo fixes (47cfaa9, f4d2099, 970b581, 338748f, a79dec8, c20f4cb, b5f1a3c) were never covered by a reviewer full pytest + production build run; the last reviewer run was 20 targeted tests (docs/SECURITY_REVIEW.md:211), so a regression elsewhere in the suite would pass review | fix: reviewer re-runs the full Docker pytest + both image builds at current HEAD and pastes the output
- A-03 | P1 | source: checklist:5 | at: docs/SECURITY_REVIEW.md:89 | quote: "**stripe-python 16: `StripeObject` is not a dict.**" | scenario: this P0 trap (and .dockerignore depth, BFF IP bucket, unbounded simulation size, hard lockout DoS, plain-header IP forwarding, attacker-chosen rate-limit key, global concurrency cap) is absent from known-traps.md, so the next Stripe project repeats the paid-but-no-access bug | fix: append the Trap-worthy lines of SECURITY_REVIEW.md (rounds 1-2) to known-traps.md
- A-04 | P2 | source: checklist:1 | at: docs/REVIEW.md:20 | quote: "P2 (fixed in a7bac81)" | scenario: the same R-06 row has status open and the summary counts it in '6 open', so QA/release read one finding as both fixed and open | fix: set R-06 status fixed, commit a7bac81, and update the summary counts
- A-05 | P2 | source: checklist:1 | at: docs/SECURITY_REVIEW.md:243 | quote: "an unset key only logs a warning (still open as a P2 decision for the owner: set the key on both sides)" | scenario: config/settings.py:160-162 now raises 'CLIENT_IP_SIGNING_KEY is required when ENVIRONMENT=production' (b5f1a3c), so an owner trusting this line deploys without the key and the API refuses to boot | fix: mark S-23 fixed in b5f1a3c (boot refusal) in the final state list

## Notes (non-blocking)
- N-01 | opinion | at: docs/REVIEW.md:29 | The heading claims 15/15 ACs but docs/FINISH_PLAN.md has 13 (B1-B6, F1-F5, D1, D2); the table itself covers all 13, so nothing is missing, only the count is wrong.
- N-02 | opinion | at: docs/SECURITY_REVIEW.md:213 | Riskiest assumption: S-19 is closed by code review only; the live 40-guess reproduction was not repeated against a79dec8 and is delegated to QA.
- N-03 | opinion | at: docs/SECURITY_REVIEW.md:245 | S-15 is still listed as open P3 although commit 00a3888 (QA F-1) maps ledger/maintain to 404; worth reconciling with QA.md.
- N-04 | opinion | at: docs/REVIEW.md:7 | Security findings are not merged into the REVIEW.md Findings table (process step 5); a reader of REVIEW.md alone sees '-' for security.

## Disposition
| id | verdict | by | note |
|---|---|---|---|
| A-01 | valid | orchestrator | design gate (375/768/1440) not run; needs a running stack, blocked on Docker; carried to owner/next session |
| A-02 | valid | orchestrator | full re-run after fixes impossible without Docker; pending |
| A-03 | valid | orchestrator | fixed: 7 more security traps appended to known-traps.md |
| A-04 | valid | orchestrator | fixed: R-06 and R-03 status set to fixed, summary recounted |
| A-05 | valid | orchestrator | fixed: S-23 line updated (fail-closed since b5f1a3c) |

Read: E:/Main/Dev/coding-brain/rules/audit.md, E:/Main/Dev/coding-brain/process/review.md, E:/Main/Dev/coding-brain/known-traps.md (grep), docs/REVIEW.md, docs/SECURITY_REVIEW.md, docs/FINISH_PLAN.md, config/settings.py (grep), git log 9b15498..HEAD
