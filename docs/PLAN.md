# WILD MARKET CAPITAL - Implementation Plan

**Version**: 1.0  
**Last Updated**: 2025-10-29  
**Project Type**: Portfolio Management Terminal (MVP: Streamlit)

---

## PROJECT OVERVIEW

**Goal**: Build a professional portfolio management system with 70+ metrics, 17 optimization methods, risk analysis, and scenario testing.

**Technology Stack**:
- Backend: Python 3.9+ (Core business logic)
- Frontend: Streamlit (MVP) → Next.js (Future)
- Database: SQLite (MVP) → PostgreSQL (Production)
- Analytics: NumPy, Pandas, SciPy, Statsmodels
- Optimization: CVXPy, PyPortfolioOpt
- Charts: Plotly

**Development Strategy**: Incremental delivery with working application after each phase.

---

## MVP SCOPE

**Must Have** (Core Features):
- Portfolio creation (5 methods)
- Portfolio management (CRUD, positions)
- Performance metrics (70+ metrics)
- Interactive charts (7 chart types)
- Data infrastructure (price fetching, caching)

**Nice to Have** (Phase 2):
- Portfolio optimization (17 methods)
- Risk analysis (VaR, stress tests, Monte Carlo)
- Scenario analysis
- Advanced reports (PDF, Excel)
- User authentication

---

## PHASE OVERVIEW

| Phase | Name | Duration | Priority | Status |
|-------|------|----------|----------|--------|
| 0 | Project Setup | 1-2 days | **CRITICAL** | 🔲 Not Started |
| 1 | Data Infrastructure | 2-3 days | **CRITICAL** | 🔲 Not Started |
| 2 | Portfolio Core | 3-4 days | **CRITICAL** | 🔲 Not Started |
| 3 | Analytics Engine | 4-5 days | **CRITICAL** | 🔲 Not Started |
| 4 | Streamlit UI (Basic) | 3-4 days | **CRITICAL** | 🔲 Not Started |
| 5 | Charts & Visualizations | 2-3 days | **CRITICAL** | 🔲 Not Started |
| 6 | Optimization Engine | 4-5 days | **IMPORTANT** | 🔲 Not Started |
| 7 | Risk & Scenarios | 3-4 days | **IMPORTANT** | 🔲 Not Started |
| 8 | Reports & Export | 2-3 days | **NICE TO HAVE** | 🔲 Not Started |
| 9 | Testing & Polish | 2-3 days | **CRITICAL** | 🔲 Not Started |

**Total Estimated Time**: 26-36 days (5-7 weeks)

**MVP Milestone** (Phases 0-5): ~15-21 days (3-4 weeks)

---

## PHASE 0: PROJECT SETUP & INFRASTRUCTURE

**Priority**: 🔴 CRITICAL  
**Duration**: 1-2 days  
**Goal**: Set up project structure, configuration, database, and development tools.

### Description

Initialize the project with proper architecture, configuration management, logging, and database setup. This phase creates the foundation for all future development.

### Tasks Checklist

#### Project Structure
- [ ] Create directory structure according to `arc-rules.md`
  - [ ] `config/` - Configuration modules
  - [ ] `core/` - Business logic (data_manager, analytics_engine, etc.)
  - [ ] `services/` - Service layer
  - [ ] `models/` - SQLAlchemy ORM models
  - [ ] `database/` - Database utilities and migrations
  - [ ] `streamlit_app/` - UI layer
  - [ ] `tests/` - Test suite
  - [ ] `scripts/` - Utility scripts
  - [ ] `logs/` - Log files (gitignored)
  - [ ] `data/` - Local data (gitignored)

#### Configuration
- [ ] Create `pyproject.toml` with project metadata
- [ ] Create `requirements.txt` with core dependencies
- [ ] Create `requirements-dev.txt` with development dependencies
- [ ] Create `.env.example` with example environment variables
- [ ] Create `.env` file (add to `.gitignore`)
- [ ] Implement `config/settings.py` using Pydantic Settings
- [ ] Implement `config/logging_config.py`

#### Database Setup
- [ ] Create `database/session.py` with SQLAlchemy engine and session management
- [ ] Initialize Alembic for migrations (`alembic init database/migrations`)
- [ ] Configure `alembic.ini`
- [ ] Create initial database schema migration

#### Development Tools
- [ ] Configure `mypy` in `pyproject.toml` (strict mode)
- [ ] Configure `ruff` for linting
- [ ] Configure `pytest` in `pytest.ini`
- [ ] Create `.gitignore` (Python, data, logs, .env, __pycache__, etc.)
- [ ] Set up pre-commit hooks (optional but recommended)

#### Core Utilities
- [ ] Create `core/exceptions.py` with custom exception hierarchy
- [ ] Create `core/utils/` with common utilities
- [ ] Create `run.py` - main application entry point
- [ ] Test that project structure is correct

### Files to Create/Modify

```
WMC_Portfolio_Terminal/
├── config/
│   ├── __init__.py
│   ├── settings.py              # NEW: Pydantic settings
│   └── logging_config.py        # NEW: Logging configuration
├── core/
│   ├── __init__.py
│   ├── exceptions.py            # NEW: Custom exceptions
│   └── utils/
│       └── __init__.py
├── database/
│   ├── __init__.py
│   └── session.py               # NEW: DB session management
├── models/
│   └── __init__.py
├── services/
│   └── __init__.py
├── streamlit_app/
│   ├── __init__.py
│   └── app.py                   # NEW: Streamlit entry point (placeholder)
├── tests/
│   ├── __init__.py
│   └── conftest.py              # NEW: Pytest fixtures
├── .env.example                 # NEW: Example environment variables
├── .gitignore                   # NEW: Git ignore rules
├── pyproject.toml               # NEW: Project config
├── requirements.txt             # NEW: Python dependencies
├── requirements-dev.txt         # NEW: Dev dependencies
├── pytest.ini                   # NEW: Pytest config
├── alembic.ini                  # NEW: Alembic config
├── run.py                       # NEW: Application entry point
└── README.md                    # MODIFY: Update with setup instructions
```

### Dependencies to Install

```txt
# Core
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
pydantic>=2.0.0
pydantic-settings>=2.0.0

# Database
sqlalchemy>=2.0.0
alembic>=1.12.0

# Utilities
python-dotenv>=1.0.0

# Development
pytest>=7.0.0
pytest-cov>=4.0.0
mypy>=1.5.0
ruff>=0.1.0
```

### Expected Result

After Phase 0:
- ✅ Project structure is set up
- ✅ Configuration system works (loads from `.env`)
- ✅ Logging is configured and writes to `logs/app.log`
- ✅ Database connection works (SQLite created)
- ✅ Type checking passes (`mypy .`)
- ✅ Linting passes (`ruff check .`)
- ✅ Basic tests pass (`pytest`)

### Acceptance Criteria

- [ ] All directories created as per architecture rules
- [ ] `python run.py` executes without errors (even if it does nothing yet)
- [ ] Database file created at `data/wmc.db`
- [ ] Logging outputs to console and file
- [ ] `mypy` passes with no errors
- [ ] `ruff check` passes with no errors
- [ ] Environment variables load from `.env`
- [ ] README has setup instructions

---

## PHASE 1: DATA INFRASTRUCTURE

**Priority**: 🔴 CRITICAL  
**Duration**: 2-3 days  
**Goal**: Implement price data fetching, ticker validation, and caching system.

### Description

Build the data layer that fetches historical and current prices from Yahoo Finance (primary) with fallback options. Implement caching to minimize API calls and ensure fast data access.

### Tasks Checklist

#### Ticker Validation
- [ ] Create `core/data_manager/ticker_validator.py`
  - [ ] Implement `validate_ticker(ticker: str) -> bool`
  - [ ] Implement `validate_tickers(tickers: List[str]) -> Dict[str, bool]`
  - [ ] Implement `get_ticker_info(ticker: str) -> TickerInfo`
  - [ ] Handle ticker format validation (1-10 alphanumeric)
  - [ ] Cache validation results (24h TTL)

#### Price Data Manager
- [ ] Create `core/data_manager/price_manager.py`
  - [ ] Implement `fetch_historical_prices(ticker, start_date, end_date) -> pd.DataFrame`
  - [ ] Implement `fetch_current_price(ticker) -> float`
  - [ ] Implement `fetch_bulk_prices(tickers, start_date, end_date) -> pd.DataFrame`
  - [ ] Support multiple data sources (Yahoo Finance primary)
  - [ ] Implement fallback logic if primary fails
  - [ ] Add retry logic with exponential backoff

#### Caching System
- [ ] Create `core/data_manager/cache.py`
  - [ ] Implement in-memory cache with TTL
  - [ ] Implement disk cache for historical data (pickle or parquet)
  - [ ] Implement cache invalidation logic
  - [ ] Add cache statistics (hit rate, size)

#### Database Models
- [ ] Create `models/price_history.py`
  - [ ] Define `PriceHistory` model (ticker, date, OHLCV)
  - [ ] Add indexes on (ticker, date)
  - [ ] Add unique constraint on (ticker, date)
- [ ] Create Alembic migration for `price_history` table

#### Service Layer
- [ ] Create `services/data_service.py`
  - [ ] Orchestrate ticker validation
  - [ ] Orchestrate price fetching
  - [ ] Handle caching logic
  - [ ] Error handling and logging

#### Testing
- [ ] Write unit tests for ticker validation
- [ ] Write unit tests for price fetching (mock API calls)
- [ ] Write integration tests for caching
- [ ] Test fallback logic

### Files to Create/Modify

```
core/data_manager/
├── __init__.py
├── ticker_validator.py          # NEW: Ticker validation
├── price_manager.py             # NEW: Price data fetching
└── cache.py                     # NEW: Caching system

models/
├── __init__.py
└── price_history.py             # NEW: Price history ORM model

services/
├── __init__.py
└── data_service.py              # NEW: Data service orchestration

database/migrations/
└── versions/
    └── 001_create_price_history.py  # NEW: Alembic migration

tests/unit/
├── test_ticker_validator.py     # NEW
├── test_price_manager.py        # NEW
└── test_cache.py                # NEW

tests/integration/
└── test_data_service.py         # NEW
```

### Dependencies to Install

```txt
yfinance>=0.2.0          # Yahoo Finance data
requests>=2.31.0         # HTTP requests
```

### Expected Result

After Phase 1:
- ✅ Can validate tickers (e.g., "AAPL" → valid, "INVALID" → invalid)
- ✅ Can fetch historical prices for any ticker
- ✅ Can fetch current price for any ticker
- ✅ Price data is cached (second request is instant)
- ✅ Cached data is stored in `data/cache/prices/`
- ✅ Database stores price history
- ✅ Fallback works if Yahoo Finance fails

### Acceptance Criteria

- [ ] `validate_ticker("AAPL")` returns `True`
- [ ] `fetch_historical_prices("AAPL", "2024-01-01", "2024-12-31")` returns DataFrame
- [ ] Second call to `fetch_historical_prices` returns instantly from cache
- [ ] `fetch_current_price("AAPL")` returns float price
- [ ] Invalid tickers raise `TickerNotFoundError`
- [ ] All tests pass
- [ ] Cache hit rate > 80% in repeated tests

---

## PHASE 2: PORTFOLIO CORE (Models & CRUD)

**Priority**: 🔴 CRITICAL  
**Duration**: 3-4 days  
**Goal**: Implement portfolio and position models, CRUD operations, and basic business logic.

### Description

Create the core portfolio domain models and implement all CRUD operations. This phase establishes the fundamental data structures and operations that all other features will build upon.

### Tasks Checklist

#### Database Models
- [ ] Create `models/portfolio.py`
  - [ ] Define `Portfolio` model (id, name, description, starting_capital, currency, timestamps)
  - [ ] Add indexes on (name, created_at)
  - [ ] Add relationships to positions
- [ ] Create `models/position.py`
  - [ ] Define `Position` model (id, portfolio_id, ticker, shares, weight_target, purchase_price, purchase_date)
  - [ ] Add indexes on (portfolio_id, ticker)
  - [ ] Add unique constraint on (portfolio_id, ticker)
  - [ ] Add foreign key to portfolios with cascade delete
- [ ] Create Alembic migrations for both tables

#### Core Domain Logic
- [ ] Create `core/data_manager/portfolio.py`
  - [ ] Implement `Portfolio` domain class (pure Python, not ORM)
  - [ ] Implement `add_position(ticker, shares)`
  - [ ] Implement `remove_position(ticker)`
  - [ ] Implement `update_position(ticker, shares)`
  - [ ] Implement `get_position(ticker) -> Position`
  - [ ] Implement `get_all_positions() -> List[Position]`
  - [ ] Implement `calculate_current_weights()` using current prices
  - [ ] Implement `calculate_current_value()` using current prices
  - [ ] Validate business rules (weights sum to 1.0, shares > 0, etc.)

#### Repository Pattern
- [ ] Create `core/data_manager/portfolio_repository.py`
  - [ ] Implement `save(portfolio) -> Portfolio`
  - [ ] Implement `find_by_id(portfolio_id) -> Portfolio`
  - [ ] Implement `find_all(limit, offset) -> List[Portfolio]`
  - [ ] Implement `find_by_name(name) -> Portfolio`
  - [ ] Implement `delete(portfolio_id) -> bool`
  - [ ] Convert between ORM models and domain models

#### Service Layer
- [ ] Create `services/portfolio_service.py`
  - [ ] Implement `create_portfolio(data: CreatePortfolioRequest) -> Portfolio`
  - [ ] Implement `get_portfolio(portfolio_id) -> Portfolio`
  - [ ] Implement `list_portfolios(filters, pagination) -> List[Portfolio]`
  - [ ] Implement `update_portfolio(portfolio_id, data) -> Portfolio`
  - [ ] Implement `delete_portfolio(portfolio_id) -> bool`
  - [ ] Implement `add_position(portfolio_id, ticker, shares) -> Portfolio`
  - [ ] Implement `remove_position(portfolio_id, ticker) -> Portfolio`
  - [ ] Implement `clone_portfolio(portfolio_id, new_name) -> Portfolio`
  - [ ] Validate inputs using Pydantic
  - [ ] Integrate with data service for price fetching

#### Input Validation
- [ ] Create `services/schemas.py` (Pydantic models)
  - [ ] `CreatePortfolioRequest` schema
  - [ ] `UpdatePortfolioRequest` schema
  - [ ] `AddPositionRequest` schema
  - [ ] `UpdatePositionRequest` schema
  - [ ] Validators for weights, shares, tickers

#### Testing
- [ ] Write unit tests for domain logic
- [ ] Write unit tests for repository
- [ ] Write integration tests for service layer
- [ ] Test edge cases (duplicate names, invalid tickers, negative shares)

### Files to Create/Modify

```
models/
├── __init__.py
├── portfolio.py                 # NEW: Portfolio ORM model
└── position.py                  # NEW: Position ORM model

core/data_manager/
├── __init__.py
├── portfolio.py                 # NEW: Portfolio domain logic
└── portfolio_repository.py      # NEW: Repository pattern

services/
├── __init__.py
├── portfolio_service.py         # NEW: Portfolio service
└── schemas.py                   # NEW: Pydantic validation schemas

database/migrations/versions/
├── 002_create_portfolios.py     # NEW: Migration
└── 003_create_positions.py      # NEW: Migration

tests/unit/
├── test_portfolio_domain.py     # NEW
└── test_portfolio_repository.py # NEW

tests/integration/
└── test_portfolio_service.py    # NEW
```

### Dependencies to Install

```txt
# All already installed from Phase 0
```

### Expected Result

After Phase 2:
- ✅ Can create portfolio with positions
- ✅ Can retrieve portfolio by ID
- ✅ Can list all portfolios
- ✅ Can update portfolio name/description
- ✅ Can add/remove/update positions
- ✅ Can clone portfolio
- ✅ Can delete portfolio (cascades to positions)
- ✅ Current value and weights calculated correctly
- ✅ Database stores portfolios and positions

### Acceptance Criteria

- [ ] Create portfolio with 5 positions via service
- [ ] Retrieve portfolio and verify all data
- [ ] Update position shares and verify weights recalculate
- [ ] Clone portfolio and verify independent copy
- [ ] Delete portfolio and verify positions also deleted
- [ ] Duplicate portfolio name raises `ConflictError`
- [ ] Invalid ticker raises `ValidationError`
- [ ] All tests pass (>80% coverage)

---

## PHASE 3: ANALYTICS ENGINE (70+ Metrics)

**Priority**: 🔴 CRITICAL  
**Duration**: 4-5 days  
**Goal**: Implement all 70+ portfolio metrics (performance, risk, ratios, market metrics).

### Description

Build the analytics engine that calculates comprehensive portfolio metrics. This is the core value proposition of the system - accurate, professional-grade analytics.

### Tasks Checklist

#### Performance Metrics (18 metrics)
- [ ] Create `core/analytics_engine/performance.py`
  - [ ] `calculate_total_return()`
  - [ ] `calculate_cagr()`
  - [ ] `calculate_annualized_return()`
  - [ ] `calculate_period_returns()` (YTD, MTD, QTD, 1M, 3M, 6M, 1Y, 3Y, 5Y)
  - [ ] `calculate_best_worst_periods()`
  - [ ] `calculate_win_rate()`
  - [ ] `calculate_payoff_ratio()`
  - [ ] `calculate_profit_factor()`
  - [ ] `calculate_expectancy()`

#### Risk Metrics (22 metrics)
- [ ] Create `core/analytics_engine/risk_metrics.py`
  - [ ] `calculate_volatility()` (daily, weekly, monthly, annual)
  - [ ] `calculate_max_drawdown()` (with dates and duration)
  - [ ] `calculate_current_drawdown()`
  - [ ] `calculate_average_drawdown()`
  - [ ] `calculate_drawdown_duration()` (max, average)
  - [ ] `calculate_recovery_time()`
  - [ ] `calculate_ulcer_index()`
  - [ ] `calculate_pain_index()`
  - [ ] `calculate_var()` (90%, 95%, 99% - Historical, Parametric, Cornish-Fisher)
  - [ ] `calculate_cvar()` (90%, 95%, 99%)
  - [ ] `calculate_downside_deviation()`
  - [ ] `calculate_semi_deviation()`
  - [ ] `calculate_skewness()`
  - [ ] `calculate_kurtosis()`

#### Risk-Adjusted Ratios (15 ratios)
- [ ] Create `core/analytics_engine/ratios.py`
  - [ ] `calculate_sharpe_ratio()`
  - [ ] `calculate_sortino_ratio()`
  - [ ] `calculate_calmar_ratio()`
  - [ ] `calculate_sterling_ratio()`
  - [ ] `calculate_burke_ratio()`
  - [ ] `calculate_treynor_ratio()`
  - [ ] `calculate_information_ratio()`
  - [ ] `calculate_modigliani_m2()`
  - [ ] `calculate_omega_ratio()`
  - [ ] `calculate_kappa_3()`
  - [ ] `calculate_gain_pain_ratio()`
  - [ ] `calculate_martin_ratio()`
  - [ ] `calculate_tail_ratio()`
  - [ ] `calculate_common_sense_ratio()`
  - [ ] `calculate_rachev_ratio()`

#### Market-Related Metrics (15 metrics)
- [ ] Create `core/analytics_engine/market_metrics.py`
  - [ ] `calculate_beta()`
  - [ ] `calculate_alpha()`
  - [ ] `calculate_r_squared()`
  - [ ] `calculate_correlation()`
  - [ ] `calculate_tracking_error()`
  - [ ] `calculate_active_return()`
  - [ ] `calculate_up_capture()`
  - [ ] `calculate_down_capture()`
  - [ ] `calculate_capture_ratio()`
  - [ ] `calculate_jensens_alpha()`
  - [ ] `calculate_active_share()`
  - [ ] `calculate_batting_average()`
  - [ ] `calculate_benchmark_relative_return()`
  - [ ] `calculate_rolling_beta()`
  - [ ] `calculate_market_timing_ratio()`

#### Main Analytics Engine
- [ ] Create `core/analytics_engine/engine.py`
  - [ ] `AnalyticsEngine` class orchestrates all calculations
  - [ ] `calculate_all_metrics(portfolio, start_date, end_date, benchmark) -> Dict`
  - [ ] Handle missing data gracefully (return null for insufficient data)
  - [ ] Add calculation metadata (duration, data points, missing days)
  - [ ] Cache results (1 hour TTL)

#### Service Layer
- [ ] Create `services/analytics_service.py`
  - [ ] Orchestrate portfolio data + price data + analytics
  - [ ] Handle date range validation
  - [ ] Handle benchmark ticker validation
  - [ ] Error handling and logging
  - [ ] Return standardized response format

#### Testing
- [ ] Write unit tests for each metric category
- [ ] Test with known reference portfolios (validate against published numbers)
- [ ] Test edge cases (empty portfolio, single position, insufficient data)
- [ ] Test performance (calculations should complete <1s for 1-year daily data)

### Files to Create/Modify

```
core/analytics_engine/
├── __init__.py
├── engine.py                    # NEW: Main analytics orchestrator
├── performance.py               # NEW: Performance metrics (18)
├── risk_metrics.py              # NEW: Risk metrics (22)
├── ratios.py                    # NEW: Risk-adjusted ratios (15)
└── market_metrics.py            # NEW: Market-related metrics (15)

services/
├── __init__.py
└── analytics_service.py         # NEW: Analytics service

tests/unit/
├── test_performance.py          # NEW
├── test_risk_metrics.py         # NEW
├── test_ratios.py               # NEW
└── test_market_metrics.py       # NEW

tests/integration/
└── test_analytics_service.py    # NEW
```

### Dependencies to Install

```txt
statsmodels>=0.14.0      # Statistical models
```

### Expected Result

After Phase 3:
- ✅ Can calculate all 70+ metrics for any portfolio
- ✅ Metrics are accurate (validated against reference implementations)
- ✅ Calculations complete in <1 second for 1-year daily data
- ✅ Handles missing data gracefully
- ✅ Returns null for metrics requiring unavailable data (e.g., 3Y return when <3 years data)
- ✅ Benchmark metrics calculate correctly when benchmark provided

### Acceptance Criteria

- [ ] Create test portfolio with known metrics
- [ ] Calculate all metrics and verify accuracy
- [ ] Sharpe ratio matches manual calculation
- [ ] Max drawdown matches visual inspection
- [ ] VaR 95% is within expected range
- [ ] Beta calculation matches reference (when compared to S&P 500)
- [ ] All 70 metrics return valid values or null
- [ ] Performance: <1s for 1-year daily data (252 data points)
- [ ] All tests pass (>80% coverage)

---

## PHASE 4: STREAMLIT UI (Basic Pages)

**Priority**: 🔴 CRITICAL  
**Duration**: 3-4 days  
**Goal**: Build basic Streamlit UI with portfolio creation, list, detail, and analysis pages.

### Description

Create the MVP user interface using Streamlit. Implement all 5 portfolio creation methods and basic portfolio management pages.

### Tasks Checklist

#### Application Setup
- [ ] Create `streamlit_app/app.py`
  - [ ] Set up multi-page navigation
  - [ ] Configure page layout (wide mode)
  - [ ] Apply custom CSS (color palette from requirements)
  - [ ] Initialize session state management
  - [ ] Add sidebar with navigation

#### Dashboard Page
- [ ] Create `streamlit_app/pages/dashboard.py`
  - [ ] Display portfolio overview cards
  - [ ] Show total value across all portfolios
  - [ ] Show top performing portfolios
  - [ ] Quick actions (create, view)
  - [ ] Recent activity

#### Portfolio Creation Pages
- [ ] Create `streamlit_app/pages/create_portfolio.py`
  - [ ] Tab 1: **Text Description** (Natural Language)
    - [ ] Text input for portfolio description
    - [ ] Parse text and extract tickers/allocations
    - [ ] Preview parsed results
    - [ ] Confirm and create button
  - [ ] Tab 2: **CSV Import**
    - [ ] File uploader for CSV
    - [ ] Preview uploaded data
    - [ ] Validation messages
    - [ ] Import button
  - [ ] Tab 3: **Manual Entry**
    - [ ] Portfolio name and description inputs
    - [ ] Starting capital input
    - [ ] Dynamic form for adding positions (ticker, shares)
    - [ ] Real-time weight calculation
    - [ ] Add/remove position buttons
    - [ ] Create button
  - [ ] Tab 4: **Templates**
    - [ ] Display template cards (Conservative, Balanced, Aggressive, Tech Growth, Dividend)
    - [ ] Select template
    - [ ] Customize starting capital
    - [ ] Create from template button
  - [ ] Tab 5: **Clone Existing**
    - [ ] Select existing portfolio to clone
    - [ ] Enter new name
    - [ ] Optional: modify positions before creating
    - [ ] Clone button

#### Portfolio List Page
- [ ] Create `streamlit_app/pages/portfolio_list.py`
  - [ ] Display all portfolios in grid or table
  - [ ] Show: name, value, return, positions count, created date
  - [ ] Filters: by tag, date range, performance
  - [ ] Sorting: by name, value, return, date
  - [ ] Search by name
  - [ ] Click to navigate to detail page
  - [ ] Delete button (with confirmation)

#### Portfolio Detail Page
- [ ] Create `streamlit_app/pages/portfolio_detail.py`
  - [ ] Display portfolio header (name, description, value, return)
  - [ ] Positions table with columns: ticker, shares, price, value, weight, gain/loss
  - [ ] Edit portfolio name/description (inline)
  - [ ] Add position button
  - [ ] Remove position button
  - [ ] Update shares (inline editing)
  - [ ] Rebalance button
  - [ ] Navigation to analysis page

#### Portfolio Analysis Page
- [ ] Create `streamlit_app/pages/portfolio_analysis.py`
  - [ ] Date range selector (start, end)
  - [ ] Benchmark selector (optional)
  - [ ] Calculate button
  - [ ] Display all 70+ metrics in organized sections:
    - [ ] Performance Metrics (18)
    - [ ] Risk Metrics (22)
    - [ ] Risk-Adjusted Ratios (15)
    - [ ] Market Metrics (15, if benchmark provided)
  - [ ] Use tabs or expanders for metric categories
  - [ ] Format numbers nicely (percentages, decimals)
  - [ ] Color coding (positive = green, negative = red)

#### Reusable Components
- [ ] Create `streamlit_app/components/portfolio_card.py`
  - [ ] Portfolio summary card component
- [ ] Create `streamlit_app/components/metrics_display.py`
  - [ ] Metrics display component (2-4 column layout)
- [ ] Create `streamlit_app/components/position_table.py`
  - [ ] Position table component with sorting

#### Utilities
- [ ] Create `streamlit_app/utils/formatters.py`
  - [ ] Format currency ($100,000.00)
  - [ ] Format percentage (25.34%)
  - [ ] Format date (2025-10-29)
  - [ ] Format large numbers (1.5M, 2.3B)
- [ ] Create `streamlit_app/utils/validators.py`
  - [ ] Validate ticker format
  - [ ] Validate shares (positive)
  - [ ] Validate weights (sum to 1.0)

#### Styling
- [ ] Create `streamlit_app/styles.css`
  - [ ] Apply color palette (background #0D1015, primary #BF9FFB, etc.)
  - [ ] Custom button styles
  - [ ] Custom card styles
  - [ ] Responsive design

#### Testing
- [ ] Manual testing of all pages
- [ ] Test all 5 creation methods
- [ ] Test portfolio CRUD operations
- [ ] Test navigation between pages
- [ ] Test error handling (invalid inputs)

### Files to Create/Modify

```
streamlit_app/
├── __init__.py
├── app.py                       # MODIFY: Main entry point with navigation
├── pages/
│   ├── __init__.py
│   ├── dashboard.py             # NEW: Dashboard overview
│   ├── create_portfolio.py      # NEW: 5 creation methods
│   ├── portfolio_list.py        # NEW: List all portfolios
│   ├── portfolio_detail.py      # NEW: Detail and editing
│   └── portfolio_analysis.py    # NEW: Metrics display
├── components/
│   ├── __init__.py
│   ├── portfolio_card.py        # NEW: Portfolio card component
│   ├── metrics_display.py       # NEW: Metrics display component
│   └── position_table.py        # NEW: Position table component
├── utils/
│   ├── __init__.py
│   ├── formatters.py            # NEW: Formatting utilities
│   └── validators.py            # NEW: Validation utilities
└── styles.css                   # NEW: Custom CSS

run.py                           # MODIFY: Launch Streamlit app
```

### Dependencies to Install

```txt
streamlit>=1.28.0        # Web framework
```

### Expected Result

After Phase 4:
- ✅ Can launch app with `python run.py` → opens browser at `localhost:8501`
- ✅ Can create portfolio using all 5 methods
- ✅ Can view list of all portfolios
- ✅ Can click portfolio to see details
- ✅ Can edit portfolio (name, positions)
- ✅ Can delete portfolio
- ✅ Can view all 70+ metrics for a portfolio
- ✅ UI follows color palette and design guidelines
- ✅ Navigation works between all pages

### Acceptance Criteria

- [ ] Launch app and dashboard loads
- [ ] Create portfolio via text: "60% AAPL, 40% MSFT"
- [ ] Create portfolio via CSV upload
- [ ] Create portfolio manually (5 positions)
- [ ] Create portfolio from Conservative template
- [ ] Clone existing portfolio
- [ ] View portfolio list (shows all 5 portfolios)
- [ ] Click portfolio → see detail page
- [ ] Add position to portfolio
- [ ] Remove position from portfolio
- [ ] Navigate to analysis page
- [ ] Select date range and calculate metrics
- [ ] All 70 metrics display correctly
- [ ] Delete portfolio (confirmation works)
- [ ] UI is responsive and styled correctly

---

## PHASE 5: CHARTS & VISUALIZATIONS

**Priority**: 🔴 CRITICAL  
**Duration**: 2-3 days  
**Goal**: Add interactive charts for portfolio analysis (7 chart types).

### Description

Implement all chart types using Plotly for interactive, professional visualizations. Charts must be zoomable, exportable, and follow the color palette.

### Tasks Checklist

#### Chart Implementations
- [ ] Create `core/analytics_engine/chart_data.py`
  - [ ] `get_cumulative_returns_data(portfolio, start, end, benchmark)` → dict
  - [ ] `get_drawdown_data(portfolio, start, end)` → dict
  - [ ] `get_rolling_metric_data(portfolio, metric, window, start, end)` → dict
  - [ ] `get_return_distribution_data(portfolio, start, end, bins)` → dict
  - [ ] `get_calendar_data(portfolio, year, frequency)` → dict

- [ ] Create `streamlit_app/components/charts.py`
  - [ ] **Chart 1**: `plot_cumulative_returns(data)` - Line chart
    - [ ] Portfolio line (color: #BF9FFB)
    - [ ] Benchmark line (color: #90BFF9) if provided
    - [ ] Linear/log scale toggle
    - [ ] Zoom, pan, crosshair
    - [ ] Export buttons (PNG, SVG, CSV)
  - [ ] **Chart 2**: `plot_drawdown(data)` - Area chart
    - [ ] Red gradient fill (#FAA1A4)
    - [ ] Max drawdown annotation
    - [ ] Current drawdown marker
    - [ ] Hover tooltip
  - [ ] **Chart 3**: `plot_rolling_metric(data)` - Line chart
    - [ ] Configurable metric selector
    - [ ] Window size selector (20, 30, 60, 90, 120 days)
    - [ ] Threshold lines (e.g., Sharpe = 1.0)
    - [ ] Color zones (green/yellow/red)
  - [ ] **Chart 4**: `plot_return_distribution(data)` - Histogram
    - [ ] Bars (color: #BF9FFB)
    - [ ] Normal distribution overlay
    - [ ] VaR cutoff lines (90%, 95%, 99%)
    - [ ] Mean line (dashed)
    - [ ] Statistics in legend
  - [ ] **Chart 5**: `plot_qq_plot(data)` - Scatter plot
    - [ ] Points (color: #BF9FFB)
    - [ ] 45-degree reference line
    - [ ] Interpretation text
  - [ ] **Chart 6**: `plot_monthly_heatmap(data)` - Heatmap
    - [ ] Rows: Years
    - [ ] Columns: Months
    - [ ] Color scale: red (negative) to green (positive)
    - [ ] Cell values: return %
    - [ ] Click to drill down
  - [ ] **Chart 7**: `plot_calendar_view(data)` - Calendar grid
    - [ ] Full calendar layout
    - [ ] Daily returns colored
    - [ ] Non-trading days grayed out
    - [ ] Week/month summaries

#### Integration with Analysis Page
- [ ] Modify `streamlit_app/pages/portfolio_analysis.py`
  - [ ] Add "Charts" tab after "Metrics" tab
  - [ ] Add chart selector (dropdown or tabs)
  - [ ] Render selected chart
  - [ ] Add chart controls (date range, benchmark, window size)
  - [ ] Export button for each chart

#### Chart Utilities
- [ ] Create `streamlit_app/utils/chart_config.py`
  - [ ] Plotly theme configuration (color palette)
  - [ ] Common chart settings (font, margins, etc.)
  - [ ] Export configuration

#### Testing
- [ ] Test all charts render correctly
- [ ] Test interactivity (zoom, pan, hover)
- [ ] Test export functionality (PNG, SVG, CSV)
- [ ] Test with different date ranges
- [ ] Test with and without benchmark
- [ ] Test performance (charts render quickly)

### Files to Create/Modify

```
core/analytics_engine/
├── __init__.py
└── chart_data.py                # NEW: Chart data preparation

streamlit_app/components/
├── __init__.py
└── charts.py                    # NEW: All chart implementations

streamlit_app/utils/
├── __init__.py
└── chart_config.py              # NEW: Chart configuration

streamlit_app/pages/
└── portfolio_analysis.py        # MODIFY: Add charts tab

tests/unit/
└── test_chart_data.py           # NEW: Test chart data
```

### Dependencies to Install

```txt
plotly>=5.18.0           # Interactive charts
```

### Expected Result

After Phase 5:
- ✅ All 7 chart types render correctly
- ✅ Charts are interactive (zoom, pan, hover)
- ✅ Charts follow color palette
- ✅ Can export charts (PNG, SVG, CSV)
- ✅ Charts load quickly (<2 seconds)
- ✅ Charts update when date range or benchmark changes

### Acceptance Criteria

- [ ] Navigate to portfolio analysis page
- [ ] Click "Charts" tab
- [ ] Select "Cumulative Returns" → chart displays
- [ ] Zoom in on chart → works smoothly
- [ ] Export chart as PNG → downloads successfully
- [ ] Select "Drawdown" chart → displays correctly
- [ ] Max drawdown point is annotated
- [ ] Select "Monthly Heatmap" → shows all months
- [ ] Click on month cell → drills down to daily view
- [ ] Change date range → all charts update
- [ ] Add benchmark → benchmark line appears on cumulative returns
- [ ] All 7 charts tested and working

---

## 🎯 MVP COMPLETE (Phases 0-5)

**At this point, you have a working MVP with:**
- ✅ Portfolio creation (5 methods)
- ✅ Portfolio management (CRUD)
- ✅ 70+ metrics
- ✅ 7 interactive charts
- ✅ Professional UI

**This is the minimum viable product. Phases 6-9 are enhancements.**

---

## PHASE 6: OPTIMIZATION ENGINE (17 Methods)

**Priority**: 🟡 IMPORTANT  
**Duration**: 4-5 days  
**Goal**: Implement portfolio optimization with 17 methods and efficient frontier.

### Description

Build the optimization engine that can optimize portfolio weights using various algorithms. This is a key differentiator but not critical for MVP.

### Tasks Checklist

#### Base Optimizer
- [ ] Create `core/optimization_engine/base.py`
  - [ ] `BaseOptimizer` abstract class
  - [ ] Common constraint builders
  - [ ] Result format standardization
  - [ ] Error handling

#### Optimization Methods (17)
- [ ] Create optimizers in `core/optimization_engine/`:
  - [ ] `mean_variance.py` - Markowitz optimization
  - [ ] `min_variance.py` - Minimum variance
  - [ ] `max_sharpe.py` - Maximum Sharpe ratio
  - [ ] `max_return.py` - Maximum return
  - [ ] `risk_parity.py` - Risk parity
  - [ ] `hrp.py` - Hierarchical Risk Parity
  - [ ] `max_diversification.py` - Maximum diversification
  - [ ] `min_correlation.py` - Minimum correlation
  - [ ] `black_litterman.py` - Black-Litterman
  - [ ] `robust.py` - Robust optimization
  - [ ] `cvar_optimization.py` - CVaR optimization
  - [ ] `mean_cvar.py` - Mean-CVaR
  - [ ] `kelly_criterion.py` - Kelly Criterion
  - [ ] `equal_weight.py` - Equal weight (1/N)
  - [ ] `market_cap_weight.py` - Market cap weight
  - [ ] `min_tracking_error.py` - Minimum tracking error
  - [ ] `max_alpha.py` - Maximum alpha

#### Constraints System
- [ ] Create `core/optimization_engine/constraints.py`
  - [ ] Weight constraints (min, max, bounds)
  - [ ] Group constraints (sector limits)
  - [ ] Risk constraints (max vol, max VaR, etc.)
  - [ ] Turnover constraints
  - [ ] Transaction costs
  - [ ] Cardinality constraints

#### Efficient Frontier
- [ ] Create `core/optimization_engine/efficient_frontier.py`
  - [ ] Generate N points on efficient frontier
  - [ ] Find current portfolio position
  - [ ] Find optimal portfolio position

#### Service Layer
- [ ] Create `services/optimization_service.py`
  - [ ] Orchestrate optimization request
  - [ ] Validate constraints
  - [ ] Run optimization
  - [ ] Generate trade list
  - [ ] Calculate performance improvement
  - [ ] Cache results

#### Streamlit UI
- [ ] Create `streamlit_app/pages/optimization.py`
  - [ ] Method selector (17 methods)
  - [ ] Constraint inputs (weight bounds, risk limits, etc.)
  - [ ] Lookback period selector
  - [ ] Run optimization button
  - [ ] Display results:
    - [ ] Current vs optimal allocation (side-by-side)
    - [ ] Trade list table
    - [ ] Metrics comparison
    - [ ] Efficient frontier chart (if calculated)
  - [ ] Apply optimization button (updates portfolio)

#### Testing
- [ ] Test each optimization method
- [ ] Test constraint enforcement
- [ ] Test convergence
- [ ] Compare results to known solutions

### Files to Create

```
core/optimization_engine/
├── __init__.py
├── base.py                      # NEW: Base optimizer
├── constraints.py               # NEW: Constraint builders
├── efficient_frontier.py        # NEW: Efficient frontier
├── mean_variance.py             # NEW (+ 16 more)
└── ... (17 optimizer files)

services/
└── optimization_service.py      # NEW

streamlit_app/pages/
└── optimization.py              # NEW

tests/unit/
└── test_optimizers.py           # NEW
```

### Dependencies

```txt
cvxpy>=1.4.0             # Convex optimization
pypfopt>=1.5.0           # Portfolio optimization
```

### Expected Result

- ✅ Can optimize portfolio with any of 17 methods
- ✅ Constraints are enforced
- ✅ Receives optimal weights and trade list
- ✅ Can apply optimization to portfolio

### Acceptance Criteria

- [ ] Select "Maximum Sharpe" method
- [ ] Set max weight constraint = 30%
- [ ] Run optimization
- [ ] Optimal allocation displayed
- [ ] Trade list shows what to buy/sell
- [ ] Apply optimization updates portfolio
- [ ] Efficient frontier displays (if enabled)

---

## PHASE 7: RISK & SCENARIO ANALYSIS

**Priority**: 🟡 IMPORTANT  
**Duration**: 3-4 days  
**Goal**: Add advanced risk analysis (stress tests, Monte Carlo, scenarios).

### Tasks Checklist

#### Risk Analysis
- [ ] Create `core/risk_engine/var_calculator.py`
  - [ ] Historical VaR
  - [ ] Parametric VaR
  - [ ] Monte Carlo VaR
  - [ ] Cornish-Fisher VaR
- [ ] Create `core/risk_engine/monte_carlo.py`
  - [ ] Simulate portfolio paths (10,000+ simulations)
  - [ ] Calculate percentile outcomes
  - [ ] Generate distribution charts
- [ ] Create `core/risk_engine/stress_testing.py`
  - [ ] 25+ historical stress scenarios
  - [ ] Custom scenario builder
  - [ ] Calculate portfolio impact

#### Scenario Engine
- [ ] Create `core/scenario_engine/historical_scenarios.py`
  - [ ] Define 25+ scenarios (2008 crisis, COVID, etc.)
- [ ] Create `core/scenario_engine/custom_scenarios.py`
  - [ ] User-defined scenarios
- [ ] Create `core/scenario_engine/scenario_chain.py`
  - [ ] Chain multiple scenarios

#### UI Pages
- [ ] Create `streamlit_app/pages/risk_analysis.py`
  - [ ] VaR analysis section
  - [ ] Monte Carlo simulation
  - [ ] Stress test results
- [ ] Create `streamlit_app/pages/scenarios.py`
  - [ ] Scenario selector
  - [ ] Custom scenario builder
  - [ ] Scenario chain builder
  - [ ] Results visualization

### Expected Result

- ✅ VaR calculations with multiple methods
- ✅ Monte Carlo simulation (10k paths)
- ✅ Stress test against 25+ historical scenarios
- ✅ Custom scenario creation

---

## PHASE 8: REPORTS & EXPORT

**Priority**: 🟢 NICE TO HAVE  
**Duration**: 2-3 days  
**Goal**: Generate PDF reports and Excel exports.

### Tasks Checklist

#### Report Generation
- [ ] Create `core/reporting_engine/pdf_generator.py`
  - [ ] Generate comprehensive PDF report
  - [ ] Include metrics, charts, holdings
- [ ] Create `core/reporting_engine/excel_generator.py`
  - [ ] Export portfolio to Excel
  - [ ] Multiple sheets (summary, holdings, metrics, history)

#### Templates
- [ ] Create `core/reporting_engine/templates/`
  - [ ] PDF template (HTML → PDF)
  - [ ] Excel template structure

#### UI
- [ ] Create `streamlit_app/pages/reports.py`
  - [ ] Report configuration
  - [ ] Generate report button
  - [ ] Download button

### Dependencies

```txt
reportlab>=4.0.0         # PDF generation
openpyxl>=3.1.0          # Excel generation
```

### Expected Result

- ✅ Generate PDF report with all metrics and charts
- ✅ Export portfolio data to Excel
- ✅ Download reports from UI

---

## PHASE 9: TESTING, OPTIMIZATION & POLISH

**Priority**: 🔴 CRITICAL  
**Duration**: 2-3 days  
**Goal**: Comprehensive testing, performance optimization, bug fixes, documentation.

### Tasks Checklist

#### Testing
- [ ] Achieve >80% test coverage for core modules
- [ ] Write integration tests for all workflows
- [ ] Test error handling
- [ ] Test edge cases

#### Performance Optimization
- [ ] Profile slow functions
- [ ] Optimize data fetching (batch requests)
- [ ] Optimize calculations (vectorize)
- [ ] Implement caching where beneficial

#### Bug Fixes
- [ ] Fix all known bugs
- [ ] Test all user workflows end-to-end
- [ ] Fix UI issues

#### Documentation
- [ ] Update README with:
  - [ ] Installation instructions
  - [ ] Usage guide
  - [ ] Screenshots
  - [ ] API documentation (if applicable)
- [ ] Add docstrings to all public functions
- [ ] Create user guide (optional)

#### Polish
- [ ] Improve error messages
- [ ] Add loading indicators
- [ ] Improve UI responsiveness
- [ ] Final styling adjustments

### Expected Result

- ✅ Test coverage >80%
- ✅ All calculations < 1 second
- ✅ No critical bugs
- ✅ Comprehensive documentation
- ✅ Production-ready application

---

## TECHNICAL DETAILS

### Database Schema

**Portfolios Table**:
```sql
CREATE TABLE portfolios (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description VARCHAR(500),
    starting_capital FLOAT NOT NULL,
    base_currency VARCHAR(3) DEFAULT 'USD',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_name (name),
    INDEX idx_created (created_at)
);
```

**Positions Table**:
```sql
CREATE TABLE positions (
    id VARCHAR(36) PRIMARY KEY,
    portfolio_id VARCHAR(36) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    shares FLOAT NOT NULL,
    weight_target FLOAT,
    purchase_price FLOAT,
    purchase_date DATE,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE,
    UNIQUE (portfolio_id, ticker),
    INDEX idx_portfolio (portfolio_id),
    INDEX idx_ticker (ticker)
);
```

**Price History Table**:
```sql
CREATE TABLE price_history (
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT NOT NULL,
    adjusted_close FLOAT NOT NULL,
    volume BIGINT,
    PRIMARY KEY (ticker, date),
    INDEX idx_ticker (ticker),
    INDEX idx_date (date)
);
```

### Indexes Strategy

**Critical Indexes**:
- `portfolios(name)` - Fast name lookup
- `portfolios(created_at)` - Sorting by date
- `positions(portfolio_id)` - Fast position lookup
- `positions(portfolio_id, ticker)` - Unique constraint + fast lookup
- `price_history(ticker, date)` - Primary key for fast price retrieval

### Caching Strategy

**Level 1: In-Memory Cache** (Python dict with TTL):
- Current prices: 5 minutes TTL
- Calculated metrics: 1 hour TTL
- Ticker validation: 24 hours TTL

**Level 2: Disk Cache** (Pickle/Parquet):
- Historical prices: Permanent (immutable data)
- Stored in `data/cache/prices/{ticker}.parquet`

**Level 3: Database Cache**:
- Price history table stores all fetched prices
- Never fetch same date twice

**Cache Invalidation**:
- Portfolio modification → invalidate metrics cache
- New day → invalidate current prices
- Manual refresh → clear all caches

### Performance Targets

| Operation | Target | Acceptable |
|-----------|--------|------------|
| Portfolio creation | <100ms | <500ms |
| Fetch 1-year prices (cached) | <10ms | <50ms |
| Fetch 1-year prices (uncached) | <2s | <5s |
| Calculate 70 metrics (1-year) | <500ms | <1s |
| Calculate 70 metrics (5-year) | <1s | <2s |
| Optimization (20 positions) | <3s | <5s |
| Monte Carlo (10k paths) | <5s | <10s |
| Page load | <1s | <2s |
| Chart rendering | <500ms | <1s |

### Optimization Opportunities

**Data Fetching**:
- Batch requests (fetch multiple tickers at once)
- Parallel requests (use asyncio or threading)
- Pre-fetch common benchmarks (SPY, QQQ) on startup

**Calculations**:
- Vectorize all computations (NumPy/Pandas)
- Cache expensive calculations
- Pre-calculate common periods (YTD, 1Y, 3Y)

**UI Rendering**:
- Lazy load charts (only render when tab is active)
- Debounce user inputs
- Use st.cache_data for expensive operations

---

## DEPLOYMENT (Future)

### Local Development
```bash
# Already handled by run.py
python run.py
```

### Docker (Future)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501"]
```

### Cloud Deployment Options (Future)
- **Streamlit Cloud**: Free tier for Streamlit apps
- **Heroku**: Easy deployment with Procfile
- **AWS EC2**: Full control
- **Google Cloud Run**: Serverless containers

---

## PROGRESS TRACKING

### Phase Completion Checklist

- [ ] **Phase 0**: Project Setup ⏱️ 1-2 days
- [ ] **Phase 1**: Data Infrastructure ⏱️ 2-3 days
- [ ] **Phase 2**: Portfolio Core ⏱️ 3-4 days
- [ ] **Phase 3**: Analytics Engine ⏱️ 4-5 days
- [ ] **Phase 4**: Streamlit UI ⏱️ 3-4 days
- [ ] **Phase 5**: Charts ⏱️ 2-3 days

**→ MVP COMPLETE** 🎉 (15-21 days)

- [ ] **Phase 6**: Optimization ⏱️ 4-5 days
- [ ] **Phase 7**: Risk & Scenarios ⏱️ 3-4 days
- [ ] **Phase 8**: Reports ⏱️ 2-3 days
- [ ] **Phase 9**: Testing & Polish ⏱️ 2-3 days

**→ FULL VERSION COMPLETE** 🚀 (26-36 days)

---

## RISK MITIGATION

### Known Risks & Solutions

**Risk 1**: Yahoo Finance API rate limiting
- **Solution**: Implement caching, add fallback to Alpha Vantage
- **Prevention**: Batch requests, use disk cache

**Risk 2**: Slow metric calculations for large portfolios
- **Solution**: Optimize with NumPy, cache results, show loading indicators
- **Prevention**: Profile code, vectorize operations

**Risk 3**: Streamlit performance with many charts
- **Solution**: Lazy load charts, limit data points, use tabs
- **Prevention**: Test with real data sizes

**Risk 4**: Data inconsistencies (splits, dividends)
- **Solution**: Use adjusted_close prices, validate data quality
- **Prevention**: Data quality checks on import

**Risk 5**: Optimization fails to converge
- **Solution**: Implement timeout, relaxed constraints, better error messages
- **Prevention**: Validate inputs, reasonable defaults

---

## SUCCESS METRICS

### MVP Success Criteria
- [ ] Can create 5 portfolios using all 5 methods
- [ ] Can view and edit all portfolios
- [ ] Can calculate all 70 metrics for any portfolio
- [ ] Can view all 7 chart types
- [ ] Application is stable (no crashes in normal usage)
- [ ] Performance targets met (see table above)
- [ ] Test coverage >80% for core modules

### Full Version Success Criteria
- [ ] All MVP criteria met
- [ ] Can optimize portfolios with any of 17 methods
- [ ] Can run stress tests and Monte Carlo simulations
- [ ] Can generate PDF reports and Excel exports
- [ ] Application is production-ready
- [ ] Documentation is complete

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-29  
**Status**: Ready for Implementation

---

**END OF IMPLEMENTATION PLAN**

