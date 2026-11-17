# Contributing

Thanks for taking a look. Portfolio Management Pro is a portfolio analytics web app (FastAPI + Next.js + Stripe billing); small, focused fixes with a test are the easiest changes to merge.

## Setup
```bash
# backend (from the repository root)
cd backend && pip install -r requirements.txt -r requirements-dev.txt
ruff check . && black --check . && isort --check-only .
mypy core/ services/ api/ config/ models/ database/
pytest -q

# frontend
cd ../frontend && npm ci && npx eslint && npx tsc --noEmit && npm run build
```
Copy `.env.example` to `.env` if you need to change settings. Never commit `.env*` files, Stripe keys or database URLs.

## Pull requests
- Branch from `main`; one topic per pull request.
- Add or update a test for every behaviour change.
- Keep `core/` coverage at or above 70% (enforced by `backend/pytest.ini` and CI).
- Lint, types, tests and build must pass; CI runs the same commands.
- Production guards (auth, JWT and signing-key checks, rate limits) must not be weakened.

## Reporting problems
Use the issue templates. For security problems follow [`SECURITY.md`](SECURITY.md) instead of opening a public issue.
