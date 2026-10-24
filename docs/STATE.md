# STATE
- 2026-10-24: audit done; 3 commits on chore/audit (dockerignore, env.example, README). docs/ is gitignored -> STATE/plan/audit are force-added.
- Started finish phase (FINISH_PLAN.md). Host pip broken by SSL proxy -> tests run in Docker (image audit-pmp-api:test rebuilt from Dockerfile.api, throwaway containers prefixed audit-pmp-).
## Needed from owner
- Stripe keys/webhook secret; confirm prod JWT_SECRET strong, ENVIRONMENT=production, AUTH_DISABLED=false on HF Spaces; Supabase backup plan; decision whether docs/ should be un-ignored; push/PR approval.
- [x] frontend/F1 next 16.2.1->16.4.3 (+postcss/sharp/source-map-js via lock) — npm audit 0 vulns; tsc + next build pass
- [x] backend/B1-B4 prod guards (AUTH_DISABLED, JWT>=32), trusted-proxy rate-limit IP (TRUSTED_PROXY_COUNT), transactions 404, docs hidden + security headers — 350 passed (docker), core cov 70.98%, ruff/black/isort/mypy clean
