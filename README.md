# Wild Market Capital - Portfolio Management Terminal

**Professional portfolio management system with comprehensive analytics, optimization, and risk management.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 Features

### Portfolio Management
- **5 Portfolio Creation Methods**: Wizard, Text Input, CSV Import, Manual Entry, Templates
- **Full CRUD Operations**: Create, Read, Update, Delete portfolios
- **Position Management**: Add, remove, update positions with real-time validation
- **Portfolio Cloning**: Duplicate existing portfolios with modifications

### Analytics & Metrics
- **70+ Financial Metrics** across 4 categories:
  - **Performance** (18): Total Return, CAGR, Annualized Return, YTD, MTD, QTD, Best/Worst Periods, Win Rate, etc.
  - **Risk** (22): Volatility (daily/weekly/monthly/annual), Max Drawdown, VaR (90%/95%/99%), CVaR, Downside Deviation, Skewness, Kurtosis, etc.
  - **Ratios** (15): Sharpe, Sortino, Calmar, Sterling, Burke, Treynor, Information Ratio, Omega, etc.
  - **Market** (15): Beta, Alpha (CAPM), R², Correlation, Tracking Error, Up/Down Capture, etc.

### Optimization
- **17 Optimization Methods**:
  - Equal Weight, Mean-Variance, Min Variance, Max Sharpe, Max Return
  - Risk Parity, Kelly Criterion, Hierarchical Risk Parity (HRP)
  - CVaR Optimization, Mean-CVaR, Robust Optimization
  - Max Diversification, Min Correlation, Inverse Correlation
  - Market Cap Weighting, Black-Litterman
- **Efficient Frontier Generation**
- **Flexible Constraints**: Weight limits, group constraints, turnover limits, cardinality

### Risk Analysis
- **VaR Calculation**: Historical, Parametric, Cornish-Fisher methods
- **CVaR (Conditional VaR)**: Expected shortfall analysis
- **Monte Carlo Simulation**: 10,000+ simulation paths
- **Stress Testing**: Historical and custom scenarios
- **Sensitivity Analysis**: Parameter variation studies

### Forecasting
- **Multiple Forecasting Methods**:
  - Classical: ARIMA, GARCH, ARIMA-GARCH
  - Machine Learning: Random Forest, SVM, XGBoost
  - Deep Learning: LSTM, TCN, SSA-MAEMD-TCN
  - Time Series: Prophet
  - Ensemble Forecasting
- **Out-of-Sample Validation**
- **Portfolio-Level Forecasting**

### Visualizations
- **7 Interactive Chart Types**:
  - Cumulative Returns, Drawdown Chart
  - Rolling Metrics, Correlation Heatmap
  - Returns Distribution, Monthly Returns Heatmap
  - Efficient Frontier (for optimization)

### Performance
- **Parallel Data Fetching**: 6.83x speedup for uncached data
- **Multi-Level Caching**: In-memory → Disk → Database
- **Optimized Calculations**: Vectorized operations with NumPy/Pandas
- **Fast Metrics Calculation**: <500ms for 1-year data (target: <500ms, actual: ~14ms)

---

## 📋 Requirements

- **Python**: 3.9 or higher
- **Operating System**: Windows, macOS, or Linux
- **Internet Connection**: Required for fetching market data

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd WMC_Portfolio_Management
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

### 5. Initialize Database

```bash
# Run database migrations
alembic upgrade head
```

---

## 🚀 Quick Start

### Run the Application

```bash
python run.py
```

The application will open in your default browser at `http://localhost:8501`.

### First Steps

1. **Create a Portfolio**:
   - Click "Create Portfolio" in the navigation
   - Choose one of 5 creation methods:
     - **Wizard**: Step-by-step guided process
     - **Text Input**: Natural language (e.g., "60% AAPL, 40% MSFT")
     - **CSV Import**: Upload your existing portfolio
     - **Manual Entry**: Add positions one by one
     - **Templates**: Pre-built strategies

2. **View Analytics**:
   - Navigate to "Portfolio Analysis"
   - Select a portfolio and date range
   - View 70+ calculated metrics

3. **Optimize Portfolio**:
   - Go to "Optimization" page
   - Select optimization method
   - Configure constraints
   - Generate optimized weights

---

## 📁 Project Structure

```
WMC_Portfolio_Management/
├── core/                      # Core business logic (framework-agnostic)
│   ├── analytics_engine/      # 70+ metrics calculation
│   ├── data_manager/          # Price fetching, caching, validation
│   ├── optimization_engine/   # 17 optimization methods
│   ├── risk_engine/          # VaR, Monte Carlo, stress testing
│   ├── forecasting_engine/   # Forecasting models
│   └── scenario_engine/      # Scenario analysis
├── services/                  # Service layer (orchestration)
│   ├── analytics_service.py
│   ├── data_service.py
│   ├── portfolio_service.py
│   ├── optimization_service.py
│   └── ...
├── streamlit_app/            # UI layer (Streamlit)
│   ├── app.py               # Main application
│   ├── pages/               # Page components
│   ├── components/          # Reusable UI components
│   └── utils/               # UI utilities
├── models/                   # SQLAlchemy ORM models
├── database/                 # Database utilities
│   └── migrations/          # Alembic migrations
├── config/                   # Configuration
├── tests/                    # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── performance/         # Performance tests
├── scripts/                  # Utility scripts
├── docs/                     # Documentation
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
└── run.py                    # Application entry point
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=core --cov=services --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/performance/ -m performance
```

---

## 📊 Performance

### Benchmarks

| Operation | Target | Actual | Status |
|-----------|--------|--------|--------|
| Portfolio creation | <100ms | ~50ms | ✅ |
| Fetch 1-year (cached) | <10ms | <1ms | ✅ |
| Fetch 1-year (uncached) | <2s | ~200-800ms | ✅ |
| Calculate 70 metrics | <500ms | ~14ms | ✅ |
| Bulk fetch (8 tickers, uncached) | <500ms | ~212ms | ✅ |
| Bulk fetch speedup | - | **6.83x** | ✅ |

See [Performance Report](docs/PERFORMANCE_REPORT.md) for detailed analysis.

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root (see `.env.example`):

```env
# Database
DATABASE_URL=sqlite:///./data/portfolio.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Cache
CACHE_DIR=data/cache
```

### Settings

Configuration is managed through `config/settings.py` using Pydantic Settings.

---

## 📚 Documentation

- **[Architecture](docs/ARCHITECTURE.md)**: System architecture and design decisions
- **[Implementation Plan](docs/PLAN.md)**: Development phases and tasks
- **[Requirements](docs/REQUIREMENTS.md)**: Business requirements and user stories
- **[Performance Report](docs/PERFORMANCE_REPORT.md)**: Performance profiling and optimization
- **[Architecture Rules](docs/ARC-RULES.md)**: Coding standards and best practices

---

## 🛣️ Roadmap

### Completed ✅
- [x] Phase 0: Project Setup
- [x] Phase 1: Data Infrastructure
- [x] Phase 2: Portfolio Core
- [x] Phase 3: Analytics Engine (70+ metrics)
- [x] Phase 4: Streamlit UI
- [x] Phase 5: Charts & Visualizations
- [x] Phase 6: Optimization Engine (17 methods)
- [x] Phase 7: Risk & Scenarios
- [x] Phase 9: Testing & Optimization (partial)

### In Progress 🚧
- [ ] Phase 8: Reports & Export
- [ ] Phase 9: Documentation & Polish

### Planned 📋
- [ ] User Authentication
- [ ] Multi-user Support
- [ ] Real-time Data Updates
- [ ] Mobile Responsive UI
- [ ] Next.js Migration (Future)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow [Architecture Rules](docs/ARC-RULES.md)
- Write tests for new features
- Maintain >80% test coverage for core modules
- Use type hints for all functions
- Follow SOLID principles and DRY

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

---

## 📧 Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**Built with ❤️ for portfolio managers and financial analysts**

