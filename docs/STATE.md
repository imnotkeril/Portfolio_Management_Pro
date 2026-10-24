# STATE
- 2026-10-24: audit done; 3 commits on chore/audit (dockerignore, env.example, README). docs/ is gitignored -> STATE/plan/audit are force-added.
- Started finish phase (FINISH_PLAN.md). Host pip broken by SSL proxy -> tests run in Docker (image audit-pmp-api:test rebuilt from Dockerfile.api, throwaway containers prefixed audit-pmp-).
## Needed from owner
- Stripe keys/webhook secret; confirm prod JWT_SECRET strong, ENVIRONMENT=production, AUTH_DISABLED=false on HF Spaces; Supabase backup plan; decision whether docs/ should be un-ignored; push/PR approval.
