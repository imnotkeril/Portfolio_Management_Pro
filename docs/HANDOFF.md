# Handoff
status: active
date: 2026-11-05
from-chat-tokens: 300000

## Goal
Finish Portfolio Management Pro per coding-brain (router webapp/saas, branch chore/audit). Pushes to chore/audit are OK; main, merge, PR, deploy and prod are not without the owner's explicit yes.

## State
- Done: pytest in the Dockerfile.api image 451 passed, 0 skipped (incl. test_llms_txt and fail-closed S-23); ruff/black/isort/mypy clean; pyasn1 pinned 0.6.5; pip-audit leaves ecdsa PYSEC-2026-1325 and python-jose PYSEC-2026-4298 (waived, not reachable: explicit algorithms, symmetric secret); clean-clone --no-cache builds of both images at 609b91d ok.
- Design gate: docs/review/design.md, 4 rounds, verdict OK (P0 0, P1 0). Fixed D-01/02/03/07/08 in 81bd623, 5bc664a, 609b91d. Open: P2 D-04 mobile nav, D-05 touch targets, D-06 label links; P3 x10. Screenshots in docs/design/ (about 14 MB, force-added).
- F-2 and F-3 browser-verified (QA.md "Browser verification"). QA stays RED on F-5 only (owner).
- Auditors (shadow): review FIX (old), qa OWNER, docs FIX (old), release-prep OWNER; re-check dispositions filled for qa and release-prep.
- Test stack containers audit-pmp-qa-* were removed at the end of this session.

## Decisions
- Fail-closed in production for CLIENT_IP_SIGNING_KEY; warn-only elsewhere.
- docs/ is gitignored: commit with `git add -f <specific files>`, never `git add -f docs`.
- Forecasting and Optimization ARE Pro-gated (require_pro).
- CHANGELOG stays Unreleased until the release step (needs owner yes).

## Next
1. Owner decisions below, then close QA (F-5) and release-prep.
2. Optional: fix design P2 D-04/D-05/D-06; llms.txt response shapes and errors (P2); re-run auditors review and docs (their last FIX verdicts predate the fixes).
3. Lighthouse not run (npx download was denied): owner runs or waives it.
4. Push to main / PR only on the owner's yes.

## Open questions
- Account deletion endpoint (F-5): QA stays RED until decided.
- Stripe version in prod and prod URL health (web /login 404, API /health 503 at last check).
- Track docs/ in git (currently force-added; screenshots are 14 MB)? Accept the 68 benign secret-scan hits? Accept the ecdsa and python-jose waivers or move to PyJWT? Legal page texts? Lighthouse waiver?
- CLIENT_IP_SIGNING_KEY must be set identically on HF and Vercel before the next prod deploy (S-23 fail-closed).

## Traps
- Docker Desktop can hang at start; probe with a timeout.
- Host Python cannot import FastAPI and pip has an SSL problem: tests only in Docker (pip-audit and linters via python -m, PATH lacks ~/.local/bin).
- prettier reformats whole files here: do not run it on existing pages.
- Never read .env*.

## Verify
- `git log --oneline origin/chore/audit..HEAD` (unpushed list), `git status --short`
- `python scripts/gen_llms_txt.py --check` (exit 0)
- pytest in Docker: see docs/QA.md "Re-run 2026-11-05"
