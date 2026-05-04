# Portfolio Management Terminal

## 🌐 Live Application [Streamlit APP](https://proportfolio.streamlit.app/)

**Professional portfolio management system with comprehensive analytics, optimization, and risk management.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/imnotkeril/WMC_Portfolio_Management/actions/workflows/ci.yml/badge.svg)](https://github.com/imnotkeril/WMC_Portfolio_Management/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/imnotkeril/WMC_Portfolio_Management/graph/badge.svg)](https://codecov.io/gh/imnotkeril/WMC_Portfolio_Management)

---
<img width="1581" height="1190" alt="1" src="https://github.com/user-attachments/assets/4a8ebd9a-611c-4e7d-8d81-76232791b85d" />

<img width="1656" height="1200" alt="Screenshot_16" src="https://github.com/user-attachments/assets/0288e9ec-88d4-41ce-8cb2-b792a156f2ce" />

---

## ✨ Key Features

### Portfolio Management
- **4 Creation Methods**: Text Input, CSV Import, Manual Entry, Templates
- **5-Step Creation Process**: Guided interface for all creation methods
- **Full CRUD Operations**: Create, read, update, delete portfolios
- **Transaction Tracking**: Record BUY/SELL/DEPOSIT/WITHDRAWAL operations
- **Position Management**: Real-time validation, inline editing, bulk operations

### Analytics & Metrics
- **70+ Financial Metrics**: Performance, Risk, Ratios, Market metrics
- **5 Analysis Tabs**: Overview, Performance, Risk, Assets & Correlations, Export & Reports
- **Interactive Charts**: Plotly-based visualizations with zoom, pan, hover
- **Benchmark Comparison**: Compare portfolios against market indices

### Optimization
- **18 Optimization Methods**: Mean-Variance, Black-Litterman, Risk Parity, HRP, CVaR, and more
- **Efficient Frontier**: Visual risk-return optimization
- **Out-of-Sample Testing**: Validate optimization on unseen data
- **Flexible Constraints**: Weight limits, group constraints, cardinality

### Risk Analysis
- **VaR Calculation**: Multiple methods (Historical, Parametric, Cornish-Fisher, Monte Carlo)
- **Monte Carlo Simulation**: 1,000-100,000 simulation paths
- **Stress Testing**: Historical scenarios (2008 Crisis, COVID-19, etc.) and custom scenarios
- **Scenario Chains**: Sequential scenario testing with cumulative impact

### Forecasting
- **9 Forecasting Methods**: ARIMA, GARCH, LSTM, XGBoost, Prophet, Ensemble, and more
- **Out-of-Sample Validation**: MAE, RMSE, MAPE, Directional Accuracy metrics
- **Portfolio-Level Forecasting**: Aggregate portfolio predictions
- **Multiple Forecast Horizons**: 1 day to 1 year

### Transaction Management
- **Transaction History**: Full audit trail with dates, prices, fees, notes
- **CSV Import/Export**: Import transactions from files, export to CSV
- **Position Calculation**: Automatic position calculation from transaction history
- **Portfolio Modes**: Buy-and-Hold or With Transactions

---

## 📋 Requirements

- **Python**: 3.9 or higher
- **Operating System**: Windows, macOS, or Linux
- **Internet Connection**: Required for fetching market data (yfinance)
- **Memory**: 4GB+ RAM recommended for large portfolios

---

## 📁 Project Structure

```
Portfolio_Management_Pro/
├── core/                      # Core business logic (framework-agnostic)
│   ├── analytics_engine/      # 70+ metrics calculation
│   ├── data_manager/          # Price fetching, caching, validation
│   ├── optimization_engine/   # 18 optimization methods
│   ├── risk_engine/          # VaR, Monte Carlo, stress testing
│   ├── forecasting_engine/   # 9 forecasting models
│   └── scenario_engine/      # Historical and custom scenarios
├── services/                  # Service layer (orchestration)
├── streamlit_app/            # UI layer (Streamlit)
│   ├── app.py               # Main application
│   ├── pages/               # Page components
│   └── components/          # Reusable UI components
├── models/                   # SQLAlchemy ORM models
├── database/                 # Database utilities & migrations
├── config/                   # Configuration
└── tests/                    # Test suite
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=services --cov-report=html

# Run specific test categories
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
pytest tests/performance/    # Performance tests
```

### Code quality (local)

Configuration lives in `pyproject.toml` (Ruff, Black, isort, Mypy). Mypy runs on `core/`, `services/`, `api/`, `config/`, `models/`, `database/` with **assignment and return-value checking enabled** (alongside the remaining gradual-typing relaxations in `pyproject.toml`). **Coverage** for CI and default `pytest` runs uses `.coveragerc`: **`core/` must stay ≥70%** — heavy optional paths (e.g. TensorFlow forecasters, omitted CVX optimizers, chart serialization helpers) are excluded from the denominator on purpose; run broader smoke/integration tests for those.

```bash
ruff check .
black --check .
isort --check-only .
mypy core/ services/ api/ config/ models/ database/
pytest                                    # enforces core coverage ≥70%
pre-commit install   # optional: run hooks on every commit
```

---

## Next.js + FastAPI + Docker

The project now includes:

- `api/main.py` - FastAPI backend over existing `services/*` and `core/*`
- `frontend/` - Next.js frontend with pages matching Streamlit navigation
- `docker-compose.yml` - one-command startup for `api` and `web`

### Run with Docker

```bash
docker compose up --build
```

Then open:

- Frontend: `http://localhost:3000`
- API health: `http://localhost:8000/health`

### Local development

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

---

## 📊 Performance

| Operation | Target | Actual | Status |
|-----------|--------|--------|--------|
| Portfolio creation | <100ms | ~50ms | ✅ |
| Fetch 1-year (cached) | <10ms | <1ms | ✅ |
| Calculate 70 metrics | <500ms | ~14ms | ✅ |
| Bulk fetch (8 tickers) | <500ms | ~212ms | ✅ |
| Parallel fetch speedup | - | **6.83x** | ✅ |

---

## 🔧 Configuration

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=sqlite:///./data/portfolio.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Cache
CACHE_DIR=data/cache

# Risk-Free Rate (optional)
RISK_FREE_RATE=0.0435
```

---

## 📚 Documentation

- **[User Guide](USER_GUIDE.md)**: Complete user manual with detailed instructions, examples, and workflows

For detailed information about features, methods, and usage, see the [User Guide](USER_GUIDE.md).

---

## 🛣️ Roadmap

### Completed ✅
- Portfolio Management (4 creation methods, CRUD, transactions)
- Analytics Engine (70+ metrics, 5 tabs with sub-tabs)
- Optimization Engine (18 methods, efficient frontier)
- Risk Analysis (VaR, Monte Carlo, 5 scenario types)
- Forecasting (9 models, validation, ensemble)
- Transaction Tracking (full history, import/export)

### In Progress 🚧
- Reports & Export (PDF generation in progress)
- Documentation & Polish

### Planned 📋
- User Authentication
- Multi-user Support
- Real-time Data Updates
- Next.js Migration (Future)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Write tests for new features
- Maintain >80% test coverage for core modules
- Use type hints for all functions
- Follow SOLID principles and DRY
- Follow existing code structure and patterns

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **yfinance**: Market data fetching
- **Streamlit**: Web framework
- **Plotly**: Interactive charts
- **CVXPy**: Convex optimization
- **NumPy/Pandas**: Data manipulation and calculations
- **scikit-learn**: Machine learning models
- **TensorFlow**: Deep learning models

---
