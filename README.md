# Portfolio Management Terminal

**Professional portfolio management** with analytics, optimization, risk tools, and forecasting.

<p align="center">
  <a href="https://proportfolio.streamlit.app/"><strong>Live demo (Streamlit)</strong></a>
  &nbsp;·&nbsp;
  <a href="#full-stack-nextjs--fastapi--docker">Full stack (Docker)</a>
  &nbsp;·&nbsp;
  <a href="USER_GUIDE.md">User guide</a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+"></a>
  <a href="https://github.com/imnotkeril/Portfolio_Management_Pro/actions/workflows/ci.yml"><img src="https://github.com/imnotkeril/Portfolio_Management_Pro/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/imnotkeril/Portfolio_Management_Pro"><img src="https://codecov.io/gh/imnotkeril/Portfolio_Management_Pro/graph/badge.svg" alt="Codecov"></a>
  <img src="https://img.shields.io/badge/CI%20Python-3.11-3776AB?logo=python&logoColor=white" alt="CI Python 3.11">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License MIT"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://nextjs.org/"><img src="https://img.shields.io/badge/Next.js-frontend-black?logo=next.js&logoColor=white" alt="Next.js"></a>
</p>

**Two ways to use the app**

| | **Streamlit** | **Next.js + FastAPI (Docker)** |
|---|----------------|--------------------------------|
| **Role** | Fast online preview; full feature set in the original UI | Modern web UI + API, same `core/` and `services/` |
| **Try it** | [proportfolio.streamlit.app](https://proportfolio.streamlit.app/) | `docker compose up --build` → [http://localhost:3000](http://localhost:3000) |

CI runs **Ruff, Black, isort, Mypy, Pytest** on **Python 3.11** with **`core/` coverage ≥ 70%** (see `.coveragerc`).

---

## Contents

- [Screenshots](#screenshots)
- [Key features](#key-features)
- [Requirements](#requirements)
- [Project structure](#project-structure)
- [Run locally](#run-locally)
  - [Full stack (Next.js + FastAPI + Docker)](#full-stack-nextjs--fastapi--docker)
  - [Streamlit (local)](#streamlit-local)
- [Testing & code quality](#testing--code-quality)
- [Performance](#performance)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Screenshots

<p align="center">
  <img width="780" alt="App screenshot 1" src="https://github.com/user-attachments/assets/4a8ebd9a-611c-4e7d-8d81-76232791b85d" />
</p>
<p align="center">
  <img width="780" alt="App screenshot 2" src="https://github.com/user-attachments/assets/0288e9ec-88d4-41ce-8cb2-b792a156f2ce" />
</p>

---

## Key features

### Portfolio management

- **4 creation methods**: text, CSV, manual entry, templates  
- **5-step wizard** for guided creation  
- **Full CRUD** and **transaction tracking** (BUY / SELL / DEPOSIT / WITHDRAWAL)  
- **Positions**: validation, inline editing, bulk actions  

### Analytics & metrics

- **70+ metrics**: performance, risk, ratios, market  
- **5 analysis tabs** with interactive **Plotly** charts  
- **Benchmark** comparison vs indices  

### Optimization

- **18 methods** (mean-variance, Black–Litterman, risk parity, HRP, CVaR, …)  
- **Efficient frontier**, out-of-sample tests, flexible constraints  

### Risk analysis

- **VaR**: historical, parametric, Cornish–Fisher, Monte Carlo  
- **Stress tests** and **scenario chains** (e.g. historical crisis templates)  

### Forecasting

- **9 model families** (ARIMA, GARCH, LSTM, XGBoost, Prophet, ensemble, …)  
- Validation metrics and portfolio-level horizons  

### Transactions

- History, CSV import/export, automatic positions, buy-and-hold vs with-transactions modes  

---

## Requirements

- **Python** 3.9+ locally (CI uses **3.11**)
- **OS**: Windows, macOS, or Linux  
- **Network** for market data ([yfinance](https://github.com/ranaroussi/yfinance))  
- **RAM** 4GB+ recommended for large portfolios  
- **Full stack**: Docker, Node.js 18+ (for `frontend/` when not using Docker)

---

## Project structure

```text
Portfolio_Management_Pro/
├── core/                 # Business logic (framework-agnostic)
│   ├── analytics_engine/
│   ├── data_manager/
│   ├── optimization_engine/
│   ├── risk_engine/
│   ├── forecasting_engine/
│   └── scenario_engine/
├── services/             # Orchestration over core
├── api/                  # FastAPI app (uses services + core)
├── frontend/             # Next.js UI
├── streamlit_app/        # Streamlit UI (live demo + legacy-style UI)
│   ├── app.py
│   ├── pages/
│   └── components/
├── models/               # SQLAlchemy models
├── database/             # Session, migrations
├── config/               # Settings
├── tests/                # Unit, integration, performance
└── docker-compose.yml    # API + web
```

---

## Run locally

### Full stack (Next.js + FastAPI + Docker)

- `api/main.py` — FastAPI on top of `services/*` and `core/*`  
- `frontend/` — Next.js, aligned with Streamlit navigation where applicable  
- `docker-compose.yml` — API + web in one command  

```bash
docker compose up --build
```

Open:

- **Frontend**: <http://localhost:3000>  
- **API health**: <http://localhost:8000/health>  

**Develop without Docker**

Backend:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

### Streamlit (local)

```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

Use a `.env` file if you override database or cache paths (see [Configuration](#configuration)).

---

## Testing & code quality

```bash
pytest
```

```bash
# Coverage (CI uses .coveragerc — core ≥ 70%)
pytest --cov=core --cov-config=.coveragerc --cov-report=html --cov-report=term-missing
```

```bash
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```

Tooling is configured in `pyproject.toml` (**Ruff**, **Black**, **isort**, **Mypy**). Mypy targets `core/`, `services/`, `api/`, `config/`, `models/`, `database/`.

```bash
ruff check .
black --check .
isort --check-only .
mypy core/ services/ api/ config/ models/ database/
pytest
pre-commit install   # optional: git hooks
```

---

## Performance

| Operation | Target | Actual | Status |
|-----------|--------|--------|--------|
| Portfolio creation | <100ms | ~50ms | OK |
| Fetch 1y (cached) | <10ms | <1ms | OK |
| Calculate 70 metrics | <500ms | ~14ms | OK |
| Bulk fetch (8 tickers) | <500ms | ~212ms | OK |
| Parallel fetch speedup | — | **6.83×** | OK |

---

## Configuration

Create `.env` in the project root (optional; defaults exist in `config/settings.py`):

```env
# Database (default in code: sqlite:///./data/wmc.db)
DATABASE_URL=sqlite:///./data/wmc.db

LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Cache (align with your settings / deploy)
CACHE_DIR=data/cache

RISK_FREE_RATE=0.0435
```

---

## Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** — manual, examples, workflows  

---

## Roadmap

### Done

- Portfolio management (CRUD, transactions, multiple creation flows)  
- Analytics (70+ metrics), optimization (18 methods), risk, forecasting, scenarios  
- **Next.js + FastAPI + Docker** stack alongside Streamlit  
- CI: lint, format, types, tests, **core coverage gate**  

### In progress

- Reports & export (e.g. PDF) polish  
- Next.js **parity** and UX polish vs Streamlit where needed  

### Planned

- Authentication & multi-user  
- Real-time or push-based data updates (where applicable)  

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

- **yfinance**, **Streamlit**, **Next.js**, **FastAPI**, **Plotly**  
- **CVXPy**, **NumPy**, **Pandas**, **scikit-learn**, **TensorFlow** (where used in forecasting)  
