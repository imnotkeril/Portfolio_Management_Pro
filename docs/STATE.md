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
