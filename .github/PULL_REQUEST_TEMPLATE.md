## What and why

## Checks
- [ ] `ruff check .`, `black --check .`, `isort --check-only .` and `pytest` pass (in `backend/`)
- [ ] `mypy core/ services/ api/ config/ models/ database/` passes; `npx eslint`, `npx tsc --noEmit` and `npm run build` pass (in `frontend/`)
- [ ] Tests added or updated for behaviour changes
- [ ] Docs / README updated if behaviour changed
- [ ] No secrets, `.env*` files or database files committed
