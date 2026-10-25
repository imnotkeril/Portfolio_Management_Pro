# Finish plan (from AUDIT_2026-10-24)
Owner: B=backend-dev, F=frontend-dev, D=docs. Check = command proving it.

## Backend (B)
- [x] B1 production guards: ENVIRONMENT=production refuses AUTH_DISABLED=true and JWT_SECRET shorter than 32 chars; unit tests
- [x] B2 rate limit: trusted-proxy aware client IP (TRUSTED_PROXY_COUNT setting, default 0 = ignore XFF unless configured); tests incl. spoof attempt
- [x] B3 500->404 on /portfolios/{id}/transactions for foreign/missing portfolio; test
- [x] B4 hide /docs,/redoc,/openapi.json in production; security headers middleware (nosniff, frame deny, referrer, HSTS when production); tests
- [x] B5 pin deps: requirements.lock/constraints for Dockerfile.api + Dockerfile (from working image freeze); non-root USER in Dockerfile.api
- [x] B6 raise core/ coverage to >=75% (optimization_engine base/constraints, subscription_repository)
## Frontend/CI (F)
- [x] F1 next/sharp/postcss bump, npm audit --omit=dev clean or documented; build passes
- [x] F2 fix 30 eslint errors; add eslint + tsc + build job to .github/workflows/ci.yml
- [x] F3 BFF proxy path allowlist; next.config security headers
- [x] F4 non-root user in frontend/Dockerfile
- [x] F5 buy_hold portfolio overview shows $0 / no holdings: fix
## Docs (D)
- [x] D1 update phase-06-deploy.md, STATUS.md, CHANGELOG.md (actual stack Vercel/HF/Supabase); AUDIT "After fixes"
- [x] D2 release checklist (no deploy)
