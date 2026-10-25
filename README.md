# Portfolio Management Terminal

**Professional portfolio management** with analytics, optimization, risk tools, and forecasting.

<p align="center">
  <a href="https://portfolio-management-pro-taupe.vercel.app/"><strong>Live app</strong></a>
  &nbsp;·&nbsp;
  <a href="#full-stack-nextjs--fastapi--docker">Full stack (Docker)</a>
  &nbsp;·&nbsp;
  <a href="USER_GUIDE.md">User guide</a>
</p>

<p align="center">
  <strong>Production:</strong>
  web <a href="https://portfolio-management-pro-taupe.vercel.app/">Vercel</a>
  · API <a href="https://imnotkeril-portfolio-management-pro-api.hf.space/health">Hugging Face Spaces</a>
  · DB Supabase Postgres · billing Stripe (test mode)
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11"></a>
  <a href="https://github.com/imnotkeril/Portfolio_Management_Pro/actions/workflows/ci.yml"><img src="https://github.com/imnotkeril/Portfolio_Management_Pro/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/imnotkeril/Portfolio_Management_Pro"><img src="https://codecov.io/gh/imnotkeril/Portfolio_Management_Pro/graph/badge.svg" alt="Codecov"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License MIT"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://nextjs.org/"><img src="https://img.shields.io/badge/Next.js-frontend-black?logo=next.js&logoColor=white" alt="Next.js"></a>
</p>

**Architecture** — a framework-agnostic `core/` + `services/` layer behind a **FastAPI** API, with a **Next.js** web UI. The Next.js app talks to the API through a same-origin proxy. Runs as one `docker compose` stack locally; the production setup is Vercel (web) + Hugging Face Spaces (API) + Supabase (Postgres).

CI runs **Ruff, Black, isort, Mypy, Pytest** on **Python 3.11** with **`core/` coverage ≥ 70%** (see `.coveragerc`), plus an ESLint / TypeScript / build job for `frontend/`.

---

## Contents

- [Screenshots](#screenshots)
- [Key features](#key-features)
- [Requirements](#requirements)
- [Project structure](#project-structure)
- [Run locally](#run-locally)
  - [Full stack (Next.js + FastAPI + Docker)](#full-stack-nextjs--fastapi--docker)
- [Testing & code quality](#testing--code-quality)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Screenshots

<p align="center">
  <img  width="1420" height="1007" alt="Снимок экрана 2026-06-12 233130" src="https://github.com/user-attachments/assets/9b129cc4-c246-43d0-a1b2-3e6203ffd919" />
</p>
<p align="center">
  <img width="1355" height="945" alt="Снимок экрана 2026-06-12 233219" src="https://github.com/user-attachments/assets/823d0397-3715-400b-8ae3-610d14c19cee" />
</p>
<p align="center">
 <img width="1355" height="1008" alt="Снимок экрана 2026-06-12 233313" src="https://github.com/user-attachments/assets/8ffa547f-aa3d-4e1e-a276-787847d5c747" />
</p>
<p align="center">
  <img width="1814" height="1008" alt="Снимок экрана 2026-06-07 060517" src="https://github.com/user-attachments/assets/55c9b4e9-3766-4819-a492-5bdb23dee2e6" />
</p>
<p align="center">
  <img width="1362" height="1011" alt="Снимок экрана 2026-06-12 233341" src="https://github.com/user-attachments/assets/49975018-512f-4b63-ae5b-7586e833f255" />
</p>

---

## Key features

### Portfolio management

- **Four creation methods** on `/create`: text, CSV file, manual entry, templates
- **Portfolio list** with edit and delete; positions with validation and editing
- **Transactions** (BUY / SELL / DEPOSIT / WITHDRAWAL, plus dividend and split sync endpoints) with a ledger mode per portfolio (`buy_hold` or `transactions`)

### Analytics

- **Analysis page** with five tabs: Overview, Performance, Risk, Assets & Correlations, Export & Reports
- Interactive charts, benchmark comparison against an index ETF or another portfolio

### Optimization (Pro)

- Mean-variance, Black–Litterman, risk parity, HRP, CVaR and other methods (see `core/optimization_engine/`)
- Efficient frontier, out-of-sample testing on a training/test window, constraints

### Risk analysis

- **VaR** (historical, parametric, Monte Carlo) and CVaR
- **Monte Carlo simulation**, **stress tests** and **custom scenarios**, scenario chains (`/risk/*`)

### Forecasting (Pro)

- Classical, machine-learning, deep-learning and simple model families for assets, portfolios and batches
- Validation window and horizon controls

### Billing

- Free plan (one portfolio) and Pro plan via Stripe Checkout; Customer Portal for managing the subscription; signed webhooks update the plan

---

## Requirements

- **Python** 3.11 (CI, `Dockerfile.api` and Mypy use 3.11)
- **Node.js** 20+ (Next.js 16; the frontend Docker image uses `node:20`)
- **Docker** with Compose v2 (for the full stack and the Postgres service)
- **OS**: Windows, macOS, or Linux
- **Network** for market data (yfinance) and Stripe (if billing is configured)

---

## Project structure

```text
Portfolio_Management_Pro/
├── api/                  # FastAPI app: main.py (routes), routers/ (auth, billing), schemas/, dependencies, rate limiting, security headers
├── core/                 # Business logic (framework-agnostic)
│   ├── analytics_engine/
│   ├── auth/
│   ├── data_manager/     # Repositories (portfolios, users, subscriptions, prices)
│   ├── forecasting_engine/
│   ├── optimization_engine/
│   ├── risk_engine/
│   └── scenario_engine/
├── services/             # Orchestration over core (portfolio, ledger, optimization, risk, forecasting, billing/)
├── models/               # SQLAlchemy models
├── database/             # Session, migration runner (migrate.py), Alembic migrations/
├── config/               # settings.py (environment variables)
├── frontend/             # Next.js 16 app (src/app pages, src/components, src/lib; BFF proxy at /api/proxy)
├── tests/                # unit/, integration/, performance/
├── scripts/              # Helper scripts (Stripe CLI listener, profiling)
├── alembic.ini
├── Dockerfile            # Hugging Face Spaces image for the API (port 7860)
├── Dockerfile.api        # Local compose image for the API (port 8000)
├── docker-compose.yml    # postgres + api + web
├── requirements*.txt     # requirements.txt (full), requirements-deploy.txt (slim), requirements-dev.txt, requirements.lock.txt (pins)
└── pyproject.toml        # Ruff, Black, isort, Mypy configuration
```

---

## Run locally

### Full stack (Next.js + FastAPI + Docker)

- `api/main.py` — FastAPI on top of `services/*` and `core/*`
- `frontend/` — Next.js web UI (talks to the API through the same-origin proxy at `/api/proxy/*`)
- `docker-compose.yml` — **PostgreSQL 16** + API + web. The API runs `alembic upgrade head` on start.

```bash
docker compose up --build
```

Open:

- **Frontend**: <http://localhost:3000>
- **API health**: <http://localhost:8000/health>

Host ports are fixed in `docker-compose.yml`: `3000` (web), `8000` (API), `5432` (Postgres, user/password/database `portfolio`). If one is taken by another program, change the host side of that `ports:` entry. Stripe variables are read from the host `.env` by Compose interpolation; if the file is absent, billing stays unconfigured and the rest works.

### Develop without Docker (Postgres in Docker, API and web on the host)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -c requirements.lock.txt -r requirements.txt   # full set (TensorFlow, Prophet, ...)
# slim set without the heaviest packages: -r requirements-deploy.txt
cp .env.example .env                 # DATABASE_URL already points at localhost:5432
docker compose up -d postgres
alembic upgrade head
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend (second terminal):

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000>. With no environment variables the frontend proxies to `http://localhost:8000`; to change that, set `API_INTERNAL_URL` in `frontend/.env.local` (see `frontend/.env.example`).

### SQLite (no Postgres)

`DATABASE_URL=sqlite:///./data/wmc.db` is the settings default and the tests use SQLite, but **the API does not start on SQLite at present**: on import of `api.main`, `init_db()` fails with `sqlite3.OperationalError: database is locked` while creating the system user's subscription. Use the Postgres path above until this is fixed.

---

## Testing & code quality

```bash
pytest
```

`pytest.ini` runs the whole suite with coverage of `core/` and the 70% gate (`--cov-fail-under=70`); it also writes `htmlcov/` and `coverage.xml`. Last run: 385 passed, `core/` coverage 78.43%.

```bash
pytest --cov=core --cov-config=.coveragerc --cov-report=html --cov-report=term-missing
pytest tests/unit/          # 353 tests
pytest tests/integration/   # 29 tests
pytest tests/performance/   # 3 tests
```

```bash
ruff check .
black --check .
isort --check-only .
mypy core/ services/ api/ config/ models/ database/
pre-commit install   # optional: git hooks from .pre-commit-config.yaml
```

Frontend checks (run in `frontend/`):

```bash
npm run lint         # eslint (0 errors; warnings are reported)
npx tsc --noEmit
npm run build
```

Tooling is configured in `pyproject.toml` (Ruff, Black, isort, Mypy).

---

## Configuration

Create `.env` in the project root (`cp .env.example .env`). Every variable is optional for local development because `config/settings.py` has defaults; the table lists the names the code reads. Values are never committed.

| Variable | Default | Meaning |
|----------|---------|---------|
| `ENVIRONMENT` | `development` | `development` or `production` (production enforces the checks below) |
| `DEBUG` | `false` | Debug flag |
| `JWT_SECRET` | `change-me-in-production` | Signing key for access tokens; must be ≥ 32 chars in production |
| `JWT_ALGORITHM` | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `1440` | Token lifetime |
| `AUTH_DISABLED` | `false` | Skips login (local only; refused in production) |
| `DATABASE_URL` | `sqlite:///./data/wmc.db` | SQLAlchemy URL; Docker uses Postgres |
| `DATABASE_ECHO` | `false` | Log SQL statements |
| `RATE_LIMIT_ENABLED` | `true` | IP-based rate limits on auth and public endpoints |
| `TRUSTED_PROXY_COUNT` | `0` | Number of reverse proxies in front of the API (0 = ignore `X-Forwarded-For`) |
| `CORS_ORIGINS` | empty | Extra allowed browser origins, comma-separated (read in `api/main.py`) |
| `PRICE_CACHE_DIR` | `data/cache/prices` | Price cache directory |
| `LOG_LEVEL` | `INFO` | Log level |
| `LOG_FILE` | `logs/app.log` | Log file path |
| `STRIPE_SECRET_KEY` | unset | Stripe secret key (test mode `sk_test_...` for development) |
| `STRIPE_WEBHOOK_SECRET` | unset | Signing secret for `/stripe/webhook` (`whsec_...`) |
| `STRIPE_PRICE_ID_PRO` | unset | Pro price (`price_...` or `prod_...`) |
| `FRONTEND_URL` | `http://localhost:3000` | Web app URL (Stripe return URLs, CORS) |
| `BILLING_ENFORCEMENT` | `true` | Enforces plan limits |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Frontend build-time API base (Docker build argument) |
| `API_INTERNAL_URL` | falls back to `NEXT_PUBLIC_API_URL`, then `http://localhost:8000` | Server-side API base used by the frontend proxy (Docker: `http://api:8000`) |
| `COOKIE_SECURE` | `false` | Frontend session cookie `Secure` flag; set `true` behind HTTPS |
| `PORT` | `7860` | Port for the Hugging Face Spaces image (`Dockerfile`) |

`.env.example` also lists settings that no code reads yet; they are marked `[unused]` there: `APP_NAME`, `APP_VERSION`, `ALPHA_VANTAGE_API_KEY`, `IEX_CLOUD_API_KEY`, `YAHOO_FINANCE_ENABLED`, `CACHE_ENABLED`, `CACHE_TTL_SECONDS`, `MAX_WORKERS`, `CALCULATION_TIMEOUT_SECONDS`, `RISK_FREE_RATE`, `MAX_PORTFOLIO_POSITIONS`, `MIN_POSITION_WEIGHT`, `MAX_POSITION_WEIGHT`.

**Production checklist**

- `ENVIRONMENT=production` and a strong `JWT_SECRET` (`python -c "import secrets; print(secrets.token_urlsafe(64))"`); the API refuses to boot with the default or a short (<32 chars) secret, or with `AUTH_DISABLED=true`. In production `/docs`, `/redoc` and `/openapi.json` are disabled and HSTS is sent.
- `TRUSTED_PROXY_COUNT` = number of reverse proxies in front of the API. Set `1` behind a single proxy such as HF Spaces so per-client rate limiting works; a wrong value lets clients spoof their IP.
- Keep `AUTH_DISABLED=false` and `DEBUG=false`.
- Frontend behind HTTPS: `COOKIE_SECURE=true`. Extra API origins: `CORS_ORIGINS`.
- Billing: `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_ID_PRO`, `FRONTEND_URL`. The webhook endpoint is `https://<api-host>/stripe/webhook` (not reachable through the browser proxy).
- Local Stripe webhooks: `stripe listen --forward-to localhost:8000/stripe/webhook` (see `scripts/start-stripe-listen.ps1`).

---

## Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** — task-oriented guide for the web app

Performance figures from earlier releases were removed from this README because they were not re-measured for the Next.js + FastAPI stack.

---

## Roadmap

### Done

- Portfolio management (CRUD, transactions, multiple creation flows)
- Analytics, optimization, risk, forecasting, scenarios
- **Next.js + FastAPI + Docker** full-stack web app, deployed to production
- CI: lint, format, types, tests, **core coverage gate**, frontend lint / typecheck / build
- **Transaction ledger** (`ledger_mode=transactions`) with scheduled rebalance maintenance
- **Optimization / Risk / Forecasting parity** for transaction-led portfolios (synthetic optimized ledger mirrors deposits, withdrawals, and rebalance interval)
- **Stripe billing** — Free/Pro tiers, checkout, customer portal, webhooks, plan limits
- **Rebalance strategy UI** — target weights editor

### In progress

- Reports & export (e.g. PDF) polish
- Continued UX polish across the Next.js app

---

## Contributing

1. Fork the repository
2. Branch: `git checkout -b feature/your-feature`
3. Commit with clear messages
4. Push and open a Pull Request

**Guidelines**

- Add or update **tests** for new behavior
- Keep **`core/` coverage ≥ 70%** as enforced by CI and `.coveragerc`
- Prefer **type hints**; follow existing layout and **SOLID** / **DRY**
- Run **Ruff, Black, isort, Mypy** before pushing (or use **pre-commit**)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- **yfinance**, **Next.js**, **FastAPI**, **Plotly**
- **CVXPy**, **NumPy**, **Pandas**, **scikit-learn**, **TensorFlow** (where used in forecasting)
