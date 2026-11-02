# README rewrite check — 2026-11-02 (branch chore/audit)

Commands from the rewritten `README.md`, run on Windows 11 (Docker 29.x, Compose v5.1.1, Python 3.11.9, Node v24.14.1, npm 11.11.0). Compose runs used project `audit-pmp-readme` (and `audit-pmp-readme-b` for the postgres-only step) with containers `audit-pmp-readme-*`, a port override file in the scratchpad (random ports in 20000-39999) and an empty `--env-file`, so the repo `.env` was not read. Host commands ran in clean clones of `chore/audit` in the scratchpad (no repo `.env` copied). The clone used `.env.example` copied to `.env`, with only the Postgres port changed.

Host port 5432 is held by another process on this machine (PID 6188, not touched), so `docker compose up -d postgres` cannot bind the documented port here. Its port was overridden for the run; the README keeps the documented port.

| # | Command (as in README) | Result | Action |
|---|---|---|---|
| 1 | `docker compose up --build` (with overrides) | exit 0; postgres and api healthy, web up | none |
| 2 | `docker compose config` | services api, postgres, web; ports as overridden | none |
| 3 | `curl <api>/health` | 200 `{"status":"ok"}` | none |
| 4 | `curl <web>/` | 200 | none |
| 5 | `curl <web>/login` | 200 | none |
| 6 | `curl <api>/openapi.json` (development) | 200; 52 operations on 46 paths | none |
| 7 | `curl <api>/docs` and `/redoc` (development) | 200, 200 | none |
| 8 | `curl <api>/llms.txt` | 404 (0 of 52 operations) | open (`rules/api.md`) |
| 9 | `docker exec ... id -u` (api, web) | 1000, 1000 | none |
| 10 | `python -m venv .venv` and activate | ok (README gives the PowerShell activate path too) | none |
| 11 | `pip install -c requirements.lock.txt -r requirements.txt` | exit 0 (full set, incl. TensorFlow 2.21, Prophet 1.5, Playwright 1.64) | none |
| 12 | `pip install -c requirements.lock.txt -r requirements-dev.txt` | exit 0 | added to README Testing |
| 13 | `cp .env.example .env` | ok; Postgres port changed in the clone only | deviation (port busy) |
| 14 | `docker compose up -d postgres` | ok only with the port override | none |
| 15 | `alembic upgrade head` | exit 0; migrations 001 to 011 | none |
| 16 | `uvicorn api.main:app --reload --host 127.0.0.1 --port 8000` | started; `/health` 200 | none |
| 17 | SQLite: `DATABASE_URL=sqlite:///./data/...` then uvicorn | fails: `sqlite3.OperationalError: database is locked` from `ensure_free`, via `init_db` (api/main.py:419) | README keeps the "does not boot" note |
| 18 | `cd frontend && npm install` | exit 0 | none |
| 19 | `npm run dev` | ready; `/login` 200 on port 3000 | none (Next created `frontend/AGENTS.md` in the throwaway clone, not in repo) |
| 20 | `pytest` | 448 passed, exit 0; `core/` coverage 78.45% (gate 70%) | README gives no count (it was 388) |
| 21 | `pytest --cov=core --cov-config=.coveragerc --cov-report=html --cov-report=term-missing` | 448 passed, exit 0 | none |
| 22 | `pytest tests/unit/ --no-cov` | 388 passed, exit 0 | none |
| 23 | `pytest tests/integration/ --no-cov` | 57 passed, exit 0 | README has no count now |
| 24 | `pytest tests/performance/ --no-cov` | 3 passed, exit 0 | none |
| 25 | `ruff check .` | All checks passed | none |
| 26 | `black --check .` | 180 files unchanged | none |
| 27 | `isort --check-only .` | exit 1 with the venv inside the repo (isort scans `.venv`); exit 0 with `--skip .venv` | README documents `--skip .venv`; config fix open |
| 28 | `mypy core/ services/ api/ config/ models/ database/` | no issues in 126 files | none |
| 29 | `pre-commit install` | installed | none |
| 30 | `npm run lint` | 0 errors, 18 warnings | none |
| 31 | `npx tsc --noEmit` | exit 0 | none |
| 32 | `npm run build` | exit 0 | none |

Totals: 32 rows run. Literal failures: row 17 (SQLite, documented as not booting) and row 27 (isort with an in-repo venv, documented with `--skip .venv`). Row 14 needed the host port override.

Screenshots: five PNGs (1440x900) from the Next.js app on a throwaway production-mode stack (`audit-pmp-shot-*`), with a demo user and a demo portfolio seeded through the API. Each is under 270 KB and visually not blank. The previous GitHub-hosted images showed the retired Streamlit UI and were replaced.

Env names: README configuration tables compared with `.env.example` (38 names, including commented ones), `config/settings.py` (33 fields) and the frontend `process.env` names. No name is missing from the README. The extra tokens in the README (`CMD`, `HS256`, `INFO`) are a Dockerfile keyword and values. The 13 `[unused]` names have no reader outside `config/settings.py` and tests.

Production URLs (read-only GET, 2026-11-02): web `https://portfolio-management-pro-taupe.vercel.app/login` 404; API `https://imnotkeril-portfolio-management-pro-api.hf.space/health` 503. The README does not present them as working.

Not fixed (outside the doc paths or feature code):

- `isort` scans an in-repo `.venv` (`[tool.isort]` has no skip for it).
- `.dockerignore` does not exclude `.venv`; a venv in the project root would enter the compose build context (not tested).
- `docker-compose.yml` sets `CACHE_DIR`, which nothing reads.
- SQLite boot still fails; `/llms.txt` still missing.

Cleanup: containers, networks, volumes and images `audit-pmp-shot-*` and `audit-pmp-readme*` removed; the dev servers started for rows 16 and 19 were stopped.

# Docs check — 2026-10-25 (branch chore/audit)

Every command documented in `README.md`, `USER_GUIDE.md`, `.env.example` and `docs/production/` was run on this machine (Windows 11, Docker 29.3.1, Compose v5.1.1, Python 3.11.9, Node v24.14.1, npm 11.11.0).

Setup for the run:

- Docker: compose project `audit-pmp-docs`, override file in the scratchpad remapping host ports to 31541 (Postgres), 31542 (API), 31543 (web); container names `audit-pmp-docs-*`; named volume `audit-pmp-docs-data` instead of `./data`; `--env-file` pointed to an empty file so the repo `.env` was not read by Compose.
- Host commands were run in a clean clone of `chore/audit` in the scratchpad (no `.env`, no `data/*.db` of the repo touched). Python commands used a fresh venv installed from `requirements.lock.txt` + `requirements-deploy.txt` + `requirements-dev.txt`.
- Ports 8000 and 3000 were avoided: the container `audit-pmp-review-tests` (another project) lists 8000/tcp, so port 8000 was replaced by 31547 in the uvicorn command, and 3000 by 31545 in `npm run dev`.

## Command results

| Command (as documented) | Literal result (summary) | Fix made |
|---|---|---|
| `docker compose up --build` | Run as `docker compose -p audit-pmp-docs -f docker-compose.yml -f <override> --env-file <empty> up --build -d` (fixed host ports and `wmc-*` container names cannot be used here). Exit 0. `audit-pmp-docs-postgres` healthy, `audit-pmp-docs-api` healthy, `audit-pmp-docs-web` up. API log: alembic upgraded 001 to 011_subscriptions_stripe. | None. README keeps the documented command. |
| `docker compose config` (with override) | Exit 0. Services postgres/api/web; published ports 31541/31542/31543 as overridden. | None |
| `curl http://localhost:<api>/health` | `200 {"status":"ok"}` | None |
| `curl http://localhost:<web>/` and `/login` | `200` and `200` | None |
| `curl .../openapi.json` (development) | `200`, 52 operations (see endpoint counts) | None |
| `curl .../llms.txt` | `404` | Not added (not a doc change; see remaining) |
| `docker exec ... id` | web runs as uid 1000 (node), api as uid 1000 (app) | None |
| `alembic upgrade head` (host venv, `DATABASE_URL` to compose Postgres) | Exit 0; log `Context impl PostgresqlImpl`; database already at head | None |
| `uvicorn api.main:app --reload --host 127.0.0.1 --port 8000` (Postgres path, venv) | Run without `--reload` and on port 31547 (port 8000 held by another project). Started, `GET /health` → `200 {"status":"ok"}`, no traceback. | None |
| Host Python `import api.main` (machine's global Python 3.11) | Fails: `AttributeError: 'FieldInfo' object has no attribute 'in_'` (host has fastapi 0.104.1; lock pins 0.162.0). Host environment is stale, not the repo. | README now has an explicit `pip install -c requirements.lock.txt ...` step |
| `pip install -c requirements.lock.txt -r requirements.txt` | Not run in full (TensorFlow, Prophet and other large packages). Slim set run instead: `pip install -c requirements.lock.txt -r requirements-deploy.txt -r requirements-dev.txt`, exit 0. | README documents the full command and the slim alternative; full set unverified |
| `DATABASE_URL=sqlite:///./data/wmc.db uvicorn api.main:app ...` (SQLite quick start) | **Fails.** Startup (`api/main.py` line 391 → `database/session.py` `init_db`) → `UserRepository.ensure_system_user` → `subscription_repository.ensure_free` raises `sqlite3.OperationalError: database is locked` on the `subscriptions` insert. Reproduced on a fresh DB file (log `sqlite_uvicorn2.log`). The first attempt overlapped with pytest in the same clone and hit the same error. | README "SQLite" section now states the API does not boot on SQLite. `.env.example` comment updated. Code not changed (feature code). |
| `npm install` (frontend) | Exit 0 | None |
| `npm run dev` | Documented default port 3000; run with `-- -p 31545`. `Ready in 622ms`, `GET /login` → `200`. Process stopped after check. | None |
| `npm run lint` | Exit 0: 0 errors, 18 warnings | None |
| `npm run build` | Exit 0 (routes compiled; middleware/proxy present) | None |
| `npx tsc --noEmit` | Exit 0 | None |
| `pytest` (venv, clean clone) | Exit 0: `385 passed, 130 warnings`; `core` coverage 78.43% (gate 70%). Later run: 388 passed, 78.45% | README "Testing" states the 388 run |
| `pytest --cov=core --cov-config=.coveragerc --cov-report=html --cov-report=term-missing` | See "Test run details" below | None |
| `pytest tests/unit/` | See "Test run details" below | None |
| `pytest tests/integration/` | See "Test run details" below | None |
| `pytest tests/performance/` | See "Test run details" below | None |
| `pytest` inside the Docker image (as docs/STATE.md described) | Not reproducible: `Dockerfile.api` installs `requirements.txt` only (no pytest, no pytest-cov), and `.dockerignore` excludes `tests/`. `python -m pytest` in `audit-pmp-api:test` and `audit-pmp-docs-api`: `No module named pytest`. | README documents the host-venv command. Remaining: see below. |
| `ruff check .` | Exit 0: `All checks passed!` (ruff 0.15.12) | None |
| `black --check .` | Exit 0: `175 files would be left unchanged` (black 26.3.1) | None |
| `isort --check-only .` | Exit 0 (isort 8.0.1) | None |
| `mypy core/ services/ api/ config/ models/ database/` | Exit 0: `Success: no issues found in 126 source files` (mypy 1.20.2) | None |
| `pre-commit install` | `pre-commit installed at .git/hooks/pre-commit` (run in the clone; pre-commit 4.7.0). Hooks also ran on the commits in this check. | None |
| `alembic upgrade head` vs. `docker compose up -d postgres` | Covered by the rows above | None |
| `.env.example` parsed with python-dotenv | 31 keys parse; inline `[unused]` comments do not leak into values | None |

## Test run details

| Command | Result | Exit |
|---|---|---|
| `pytest` (full, coverage gate on) | 385 passed (first run, `-q`); 388 passed in the later coverage run | 0 |
| `pytest --cov=core --cov-config=.coveragerc --cov-report=html --cov-report=term-missing` | 388 passed, `core` coverage 78.45% (gate 70%) | 0 |
| `pytest tests/unit/` | 355 passed with the gate off (`--no-cov`); with the gate on, 355 passed and `core` coverage 77.41% | 0 |
| `pytest tests/integration/` | 30 passed with `--no-cov`. Without it the run passes all tests but exits 1: `core` coverage 44.56% < 70% gate, which `pytest.ini` applies to partial runs | 0 with `--no-cov`, 1 without |
| `pytest tests/performance/` | 3 passed with `--no-cov`. Without it exit 1: `core` coverage 9.22% < 70% | 0 with `--no-cov`, 1 without |

Open point: the first full run (385) and the later full run (388) in the same clone differ; the per-tier counts moved from 353/29/3 to 355/30/3. Not explained. The clone had a `data/wmc.db` left by the SQLite attempt; not confirmed as the cause.

## Endpoint counts

- Real routes in code: 52 operations. `api/main.py` declares 44 `@app.*` routes; `api/routers/auth.py` 4 and `api/routers/billing.py` 4 (added via `include_router`).
- Live `GET /openapi.json` (development mode): 52 operations on 46 paths. Matches the code count.
- `/llms.txt`: not present (`404`). Endpoint count in `/llms.txt`: 0 of 52. `rules/api.md` requires it for an API; the README does not claim it, so the rule was not applied here (feature change, out of scope).

## Environment variable names

- README configuration table and `.env.example` compared after the fixes: the only README names missing from `.env.example` were `PORT` (added as a commented line) and two values (`HS256`, `INFO`) that are not names. Difference after the fix: empty.
- `.env.example` names not read by any code (grep for `settings.<name>` outside `config/settings.py` returns no reader): `APP_NAME`, `APP_VERSION`, `ALPHA_VANTAGE_API_KEY`, `IEX_CLOUD_API_KEY`, `YAHOO_FINANCE_ENABLED`, `CACHE_ENABLED`, `CACHE_TTL_SECONDS`, `MAX_WORKERS`, `CALCULATION_TIMEOUT_SECONDS`, `RISK_FREE_RATE`, `MAX_PORTFOLIO_POSITIONS`, `MIN_POSITION_WEIGHT`, `MAX_POSITION_WEIGHT`. Marked `[unused]` in `.env.example` and listed in README.
- Read by code but not in the example: `CORS_ORIGINS` (`api/main.py`, commented in the example), `NEXT_PUBLIC_API_URL`, `API_INTERNAL_URL`, `COOKIE_SECURE` (frontend, now in the example as comments).
- Set by `docker-compose.yml` but read by nothing: `CACHE_DIR` (settings uses `PRICE_CACHE_DIR`). README does not list it.

## Fixes made (commits on chore/audit, local only)

1. `f030521` docs(env): drop Streamlit references from `.env.example` comments
2. `ad245ed` docs(env): mark settings no code reads yet, document `NEXT_PUBLIC_API_URL`
3. `576fa4d` docs(readme): match real commands, ports, env names, tree and Python/Node versions. Python badge 3.9+ to 3.11, Node 18+ to 20+, Python install step added, SQLite path marked as not booting, `CACHE_DIR` replaced by `PRICE_CACHE_DIR`, project tree corrected (`auth/`, `services/billing/`, `api/routers/`, `Dockerfile` vs `Dockerfile.api`), env table added, test counts added, unverified performance table removed.
4. `0bac40e` docs(guide): rewrite user guide for the Next.js app. The old guide described the retired Streamlit UI (`python run.py`, port 8501, "Wild Market Capital").
5. `2680dbb` docs(production): drop Streamlit and Railway references from the phase index (force-added: `docs/` is gitignored). Also a CHANGELOG line.
6. `9d9cddc` docs(env): list unused settings in README and document `PORT`.
7. `e0b2c49` docs(readme): subset test commands need `--no-cov`; current counts.
8. This file and the STATE line (commit after this file).

Facts checked and correct: the compose port mappings and health checks, the three container services, the `/health` endpoint, the Dockerfile CMD, the Stripe webhook path `/stripe/webhook` (not reachable through the browser proxy), `FREE_PORTFOLIO_LIMIT = 1`, the `require_pro` guards on optimization and forecasting, the production guards in `config/settings.py` (default and short `JWT_SECRET`, `AUTH_DISABLED`), HSTS and docs hidden in production (`api/security_headers.py`), the five analysis tabs, the four creation methods, the forecasting families (classical, ML, deep learning, simple), and the first screenshot (current Next.js UI).

## Remaining (not fixed here)

- **Bug (feature code): SQLite boot fails.** `init_db()` creates tables with `Base.metadata.create_all` but the billing `subscriptions` insert in `ensure_system_user` fails with `database is locked` on a fresh SQLite file. The README documents the workaround (Postgres). Needs a code fix and a re-run of the SQLite row.
- **`/llms.txt` missing** for the API (0 of 52 operations). Required by `rules/api.md`. Needs a code change.
- **Test count drift**: 385 then 388 passed in the same clone (see Test run details); unexplained.
- **Test subsets fail the coverage gate** unless `--no-cov` is passed; `pytest.ini` applies the 70% gate to every run.
- **`pytest` in Docker** is not reproducible: the image has no pytest and tests are excluded from the build context. Either add a test stage to `Dockerfile.api` or change the STATE note; the README uses the host venv.
- **Python version conflict**: `runtime.txt` says `python-3.12`; CI, `Dockerfile.api`, mypy and README say 3.11; `pyproject.toml` ruff/black target `py39`. Owner decision which one is real.
- **Full `requirements.txt` install** (TensorFlow, Prophet, etc.) not verified; only the slim set was installed.
- **Host environment** of this machine has a stale global Python (fastapi 0.104); run from a venv.
- **Production URLs** (from README) at check time: the Vercel root returned `404` and the Hugging Face `/health` returned `503`. Not a docs problem, but the owner should confirm the live deployment.
- **Screenshots**: 5 GitHub-hosted images; 1 was viewed (current Next.js analysis page, matches the app). The other 4 were not viewed.
- **Out-of-scope docs still describe the Streamlit era**: `docs/ARCHITECTURE.md`, `docs/PLAN.md`, `docs/REQUIREMENTS.md`, `docs/PROJECT_VISION_COMPLETE.md`, `docs/production/ROADMAP.md`, `docs/production/phase-02-auth.md`, `docs/production/phase-04-frontend.md`, `docs/production/phases/phase-5-strategies.md`, `docs/internal/*`. Historical records; left as they are (owner decision: archive or banner). `docs/production/CHANGELOG.md` keeps its history entries.
- **`scripts/STRIPE-SETUP-ONCE.md`** (outside the owned doc paths) still describes the local flow in Russian; it is consistent with the compose stack but was not edited.
- **`frontend/README.md`** was not reviewed (outside the owned doc paths).
- **Owner guide** (change plans and prices, refund a customer, read MRR, rotate keys) required by `process/docs.md` is not written. Release/ops notes (backups, Stripe webhook events list) are also missing.
- **`docs/` is gitignored**: `docs/production/README.md` and `CHANGELOG.md` are now tracked with `git add -f`; the other `docs/` files stay untracked. Owner decision whether `docs/` should be tracked.
- **No root `CHANGELOG.md`**: the only changelog is `docs/production/CHANGELOG.md` (Keep-a-Changelog style not used there). A root `CHANGELOG.md` from `templates/CHANGELOG.md` is recommended.
- **Pushing, PRs and deploys**: none done.

Read: E:/Main/Dev/coding-brain/process/docs.md, E:/Main/Dev/coding-brain/knowledge/guides/engineering/29-documentation.md (first 120 lines), E:/Main/Dev/coding-brain/templates/CHANGELOG.md, E:/Main/Dev/coding-brain/routers/projects/webapp/_family.md (grep of sections Shared process notes, Shared checks, Traps), E:/Main/Dev/coding-brain/routers/projects/webapp/saas.md (grep of docs row, Done when, Traps)
