# Finish plan (from AUDIT_2026-10-24)
Owner: B=backend-dev, F=frontend-dev, D=docs. Check = command proving it.

## Backend (B)
- [ ] B1 production guards: ENVIRONMENT=production refuses AUTH_DISABLED=true and JWT_SECRET shorter than 32 chars; unit tests
- [ ] B2 rate limit: trusted-proxy aware client IP (TRUSTED_PROXY_COUNT setting, default 0 = ignore XFF unless configured); tests incl. spoof attempt
- [ ] B3 500->404 on /portfolios/{id}/transactions for foreign/missing portfolio; test
- [ ] B4 hide /docs,/redoc,/openapi.json in production; security headers middleware (nosniff, frame deny, referrer, HSTS when production); tests
- [ ] B5 pin deps: requirements.lock/constraints for Dockerfile.api + Dockerfile (from working image freeze); non-root USER in Dockerfile.api
- [ ] B6 raise core/ coverage to >=75% (optimization_engine base/constraints, subscription_repository)
## Frontend/CI (F)
- [ ] F1 next/sharp/postcss bump, npm audit --omit=dev clean or documented; build passes
- [ ] F2 fix 30 eslint errors; add eslint + tsc + build job to .github/workflows/ci.yml
- [ ] F3 BFF proxy path allowlist; next.config security headers
- [ ] F4 non-root user in frontend/Dockerfile
- [ ] F5 buy_hold portfolio overview shows $0 / no holdings: fix
## Docs (D)
- [ ] D1 update phase-06-deploy.md, STATUS.md, CHANGELOG.md (actual stack Vercel/HF/Supabase); AUDIT "After fixes"
- [ ] D2 release checklist (no deploy)
