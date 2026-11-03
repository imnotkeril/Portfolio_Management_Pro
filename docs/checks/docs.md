# Audit: docs (README.md, USER_GUIDE.md, docs/OWNER_GUIDE.md, docs/STRIPE_WEBHOOK_LOCAL.md, docs/DOCS_CHECK.md, api/llms.txt, CHANGELOG.md, .env.example)
mode: shadow
author-model: sonnet · auditor-model: opus

## Verdict
FIX
Two exit-check items fail (unrun documented commands, CHANGELOG without version/date) and the owner guide states a forecasting gating gap that the code does not have; all repairable from the documents and code.

## Checklist
| # | Item | Result | Evidence |
|---|---|---|---|
| 1 | Every documented command was run; results pasted | FAIL | docs/DOCS_CHECK.md:173 "Commands in it were not executed (no Stripe CLI login, no test keys)"; docs/DOCS_CHECK.md:170 "written but not run: the Docker daemon was unavailable in this session" |
| 2 | `/llms.txt` endpoint count equals the real route count | PASS | api/llms.txt has 52 `### METHOD /path` headings; static count of route decorators in api/main.py + api/routers/*.py = 53, minus `/llms.txt` itself declared with `include_in_schema=False` (api/main.py:483) = 52; `python scripts/gen_llms_txt.py --check` exit 0 (run by auditor) |
| 3 | Env var names in README == names in `.env.example` | PASS | Every README Configuration name appears in .env.example; the only one not set as a key, `VERCEL`, is named in .env.example:28 "Vercel is detected via the platform-set VERCEL variable" |
| 4 | CHANGELOG has an entry for the release version with date, no commit dump | FAIL | CHANGELOG.md:3 "## Unreleased (branch chore/audit)" - no version, no date |
| 5 | No aspirational features described that do not exist | PASS | Pro gating claimed in README.md:90 and USER_GUIDE.md:178 matches `Depends(require_pro)` at api/main.py:1237, 1282, 1557, 1580, 1605; owner guide plan facts match services/billing/plans.py:14-19 |
| saas-8 | README (run, env, test DB, Stripe CLI forwarding, deploy), owner guide (plans/prices, refund, MRR, rotate keys), `/llms.txt` | PASS | README.md:346 "Stripe CLI webhook forwarding for local development"; docs/OWNER_GUIDE.md sections 2-5; api/main.py:483 serves llms.txt (command-run part covered by row 1) |

## Findings
- A-01 | P1 | source: step:docs | at: docs/OWNER_GUIDE.md:16 | quote: "endpoints only require a logged-in user" | scenario: the three forecasting routes use `Depends(require_pro)` (api/main.py:1557, 1580, 1605) and api/llms.txt:81 says 'Auth: pro'; an owner acting on the offered choice 'change the guide' would rewrite USER_GUIDE to tell Free users forecasting is available, and they would then get 403 'Pro subscription required'. The same false claim is in docs/DOCS_CHECK.md:175 (P-01) and in the plan table at docs/OWNER_GUIDE.md:12-13 | fix: state that Free = 1 portfolio, no optimization, no forecasting and remove the P-01 note in OWNER_GUIDE and DOCS_CHECK
- A-02 | P1 | source: checklist:1 | at: docs/DOCS_CHECK.md:173 | quote: "Commands in it were not executed (no Stripe CLI login, no test keys)" | scenario: docs/STRIPE_WEBHOOK_LOCAL.md and the README claim that `tests/unit/test_llms_txt.py` checks llms.txt against live routes are shipped unverified; if the test or the listen/resend flow fails, the first reader hits it, not the author. Also unrun: the key generator and SQL in docs/OWNER_GUIDE.md:50 and :66 | fix: run the Stripe CLI flow with test keys and the llms test in the Docker test image, paste results, or mark those docs explicitly as unverified
- A-03 | P1 | source: checklist:4 | at: CHANGELOG.md:3 | quote: "## Unreleased (branch chore/audit)" | scenario: release-prep and release need a version and dated entry to tag; with only 'Unreleased' there is no release version to tag or announce, and the 'Build and docs' heading at CHANGELOG.md:16 is not a Keep a Changelog category | fix: add a versioned, dated section (SemVer) with Added/Changed/Fixed/Security headings
- A-04 | P2 | source: rule:api.md | at: api/llms.txt:322 | quote: "- body `RiskVarFullRequest` { rolling_window: int }" | scenario: RiskVarFullRequest, RiskMonteCarloFullRequest and OptimizationFullRequest inherit required fields (portfolio_id, start_date, end_date, method) from their parent models (api/main.py:184, 188, 240); a client that builds the body from llms.txt sends only the listed fields and gets 422. Same at api/llms.txt:287 and :106 | fix: make gen_llms_txt.py include inherited model fields and regenerate
- A-05 | P2 | source: rule:api.md | at: api/llms.txt:26 | quote: "Compute per-asset data: correlations, metrics, returns, impact analysis. Auth: user." | scenario: rules/api.md and process/docs.md step 3 require response shapes and errors per endpoint; most entries give only the request body, so an LLM client cannot tell what fields come back and has to guess or call /openapi.json, which is disabled in production | fix: add the response model or a short response shape and endpoint-specific error codes per entry

## Notes (non-blocking)
- N-01 | opinion | at: README.md:182 | Riskiest assumption: that `tests/unit/test_llms_txt.py` passes against `app.openapi()`; it was never run, and the AST-based `--check` cannot catch schema drift such as the inherited-field gap in A-04.
- N-02 | opinion | at: USER_GUIDE.md:156 | The known candidate 'Forecasting is Pro-only but not gated' is not a defect: the guide is correct, the endpoints are gated by require_pro since ff2b108.
- N-03 | opinion | at: docs/STRIPE_WEBHOOK_LOCAL.md:20 | 'shows the four events': `stripe listen` without `--events` forwards every event of a checkout, so the terminal will show more than four lines; low impact.
- N-04 | opinion | at: api/llms.txt:314 | '/risk/var' and '/risk/monte-carlo' (and '/optimization/run' at :110) are described as working 'on a ticker list' while the body takes portfolio_id.
- N-05 | opinion | at: docs/RELEASE.md:223 | Outside this step: RELEASE still says there is no root CHANGELOG.md, which is stale since 34741ec.

## Disposition
| id | verdict | by | note |
|---|---|---|---|
| A-01 | valid | orchestrator | fixed: my claim was wrong; forecasting is require_pro; removed from OWNER_GUIDE, DOCS_CHECK, RELEASE |
| A-02 | valid | orchestrator | Stripe CLI flow and new tests unexecuted (no keys / no Docker); stated honestly |
| A-03 | waived | orchestrator | CHANGELOG stays Unreleased until the release step assigns a version; needs owner yes to release |
| A-04 | valid | orchestrator | fixed: generator now includes inherited request fields |
| A-05 | valid | orchestrator | response shapes/errors per endpoint not documented; open P2, follow-up |

Read: E:/Main/Dev/coding-brain/rules/audit.md, E:/Main/Dev/coding-brain/process/docs.md, E:/Main/Dev/coding-brain/routers/projects/webapp/saas.md, E:/Main/Dev/coding-brain/rules/api.md (grep llms), README.md, USER_GUIDE.md, docs/OWNER_GUIDE.md, docs/STRIPE_WEBHOOK_LOCAL.md, docs/DOCS_CHECK.md, api/llms.txt, CHANGELOG.md, .env.example, api/main.py (routes, schemas 170-250, 1234-1300), api/routers/billing.py (grep), api/dependencies.py (require_pro), services/billing_service.py (grep), services/billing/plans.py, config/settings.py (grep), scripts/start-stripe-listen.ps1, docker-compose.yml (grep), docs/RELEASE.md (grep), docs/AUDIT_DOCS.md (grep)
