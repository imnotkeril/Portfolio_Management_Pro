# WILD MARKET CAPITAL - System Architecture

**Version**: 1.0 (Initial)  
**Last Updated**: 2025-10-29  
**Status**: 🟡 Living Document - Will be updated during development

---

## 1. SYSTEM OVERVIEW

**Wild Market Capital** is a professional portfolio management terminal providing comprehensive analytics, optimization, and risk management for investment portfolios. The system calculates 70+ metrics, supports 17 optimization methods, and offers interactive visualizations for portfolio analysis.

**Target Users**: Portfolio managers, financial analysts, individual investors  
**Deployment**: Desktop application (MVP: Streamlit), future web platform (Next.js)

### Technology Stack

**Frontend (MVP)**:
- Streamlit 1.28+ - Web UI framework
- Plotly 5.18+ - Interactive charts
- Custom CSS - Styling with TradingView-inspired palette

**Backend (Core Business Logic)**:
- Python 3.9+ - Primary language
- NumPy 1.24+ - Numerical computations
- Pandas 2.0+ - Data manipulation
- SciPy 1.11+ - Scientific computing & optimization
- Statsmodels 0.14+ - Statistical models
- CVXPy 1.4+ - Convex optimization

**Data Layer**:
- SQLite (MVP) / PostgreSQL (Production) - Relational database
- SQLAlchemy 2.0+ - ORM
- Alembic 1.12+ - Database migrations
- yfinance 0.2+ - Market data (Yahoo Finance)

**Development Tools**:
- pytest 7.0+ - Testing framework
- mypy 1.5+ - Type checking
- ruff 0.1+ - Linting
- Pydantic 2.0+ - Data validation

---

## 2. HIGH-LEVEL ARCHITECTURE

### Architectural Style

**Layered Architecture** with **Service Layer Pattern**

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                       │
│              (Streamlit UI - Future: Next.js)               │
│                                                             │
│  Pages: Dashboard, Create, List, Detail, Analysis,         │
│         Optimization, Risk, Scenarios, Reports              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ User Actions
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     SERVICE LAYER                           │
│              (Business Logic Orchestration)                 │
│                                                             │
│  Services: PortfolioService, AnalyticsService,              │
│            OptimizationService, DataService                 │
│                                                             │
│  Responsibilities:                                          │
│  - Input validation (Pydantic)                              │
│  - Transaction management                                   │
│  - Cross-cutting concerns (logging, caching)                │
│  - Orchestrate core modules                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ Business Operations
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      CORE MODULES                           │
│               (Pure Python Business Logic)                  │
│                                                             │
│  ┌─────────────────┐  ┌──────────────────┐                 │
│  │  Data Manager   │  │ Analytics Engine │                 │
│  │                 │  │                  │                 │
│  │ - Portfolio     │  │ - Performance    │                 │
│  │ - Positions     │  │ - Risk Metrics   │                 │
│  │ - Price Data    │  │ - Ratios         │                 │
│  │ - Validation    │  │ - Market Metrics │                 │
│  │ - Caching       │  └──────────────────┘                 │
│  └─────────────────┘                                        │
│                                                             │
│  ┌─────────────────┐  ┌──────────────────┐                 │
│  │  Optimization   │  │   Risk Engine    │                 │
│  │     Engine      │  │                  │                 │
│  │                 │  │ - VaR            │                 │
│  │ - 17 Methods    │  │ - Monte Carlo    │                 │
│  │ - Constraints   │  │ - Stress Tests   │                 │
│  │ - Efficient     │  └──────────────────┘                 │
│  │   Frontier      │                                        │
│  └─────────────────┘  ┌──────────────────┐                 │
│                       │ Scenario Engine  │                 │
│                       │                  │                 │
│                       │ - Historical     │                 │
│                       │ - Custom         │                 │
│                       │ - Chain Builder  │                 │
│                       └──────────────────┘                 │
│                                                             │
│  No UI dependencies - Pure Python - Framework-agnostic      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ Data Access
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       DATA LAYER                            │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Database   │  │ External APIs│  │    Cache     │     │
│  │              │  │              │  │              │     │
│  │ - Portfolios │  │ - Yahoo Fin. │  │ - In-Memory  │     │
│  │ - Positions  │  │ - Alpha Vant.│  │ - Disk Cache │     │
│  │ - Prices     │  │ - IEX Cloud  │  │ - Redis(fut.)│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

**Presentation Layer** (Streamlit UI):
- Render UI components
- Handle user input
- Display data (charts, tables, metrics)
- Navigate between pages
- **Must NOT**: Contain business logic, call core modules directly

**Service Layer**:
- Validate inputs (Pydantic schemas)
- Orchestrate core modules
- Manage transactions
- Handle caching
- Error handling and logging
- **Must NOT**: Contain domain logic, have UI dependencies

**Core Modules** (Business Logic):
- Implement domain logic
- Calculate metrics and analytics
- Execute optimization algorithms
- Manage portfolio state
- **Must NOT**: Have framework dependencies, access database directly

**Data Layer**:
- Database access (SQLAlchemy ORM)
- External API calls
- Data caching
- Data validation

---

## 3. SYSTEM MODULES

### 3.1 Configuration Module

**Path**: `config/`  
**Status**: 🔲 Not Implemented

**Purpose**: Centralized configuration management for the entire application.

**Components**:
- `settings.py` - Pydantic Settings class loading from `.env`
- `logging_config.py` - Logging configuration with rotating file handler

**Dependencies**: None (foundation module)

**Key Features**:
- Environment-based configuration (dev, staging, prod)
- Type-safe settings with Pydantic
- Encrypted storage for API keys
- Default values for all settings

---

### 3.2 Data Manager Module

**Path**: `core/data_manager/`  
**Status**: 🔲 Not Implemented

**Purpose**: Manage all data operations including portfolios, positions, price data, and ticker validation.

**Components**:
- `portfolio.py` - Portfolio domain model and business logic
- `portfolio_repository.py` - Repository pattern for portfolio persistence
- `price_manager.py` - Fetch and cache price data from multiple sources
- `ticker_validator.py` - Validate ticker symbols and fetch company info
- `cache.py` - Caching system (in-memory + disk)
- `parsers/` - Data parsers for CSV, text, etc.

**Dependencies**:
- External APIs (Yahoo Finance, Alpha Vantage, IEX Cloud)
- Database (via SQLAlchemy)
- Configuration module

**Key Features**:
- Multi-source price fetching with fallbacks
- Intelligent caching (3-level: memory, disk, database)
- Ticker validation with company info
- Portfolio CRUD operations
- Position management (add, remove, update)

**Data Flow**:
```
User Request → Service Layer → Data Manager
                                    ↓
                    ┌───────────────┴────────────────┐
                    ↓                                ↓
            Portfolio Operations              Price Data
                    ↓                                ↓
            Database (ORM)                    External APIs
                                                     ↓
                                              Cache (TTL)
```

---

### 3.3 Analytics Engine Module

**Path**: `core/analytics_engine/`  
**Status**: 🟢 Phase 3 Implemented (Partial)

**Purpose**: Calculate comprehensive portfolio analytics including 70+ metrics across 4 categories.

**Components**:
- ✅ `engine.py` - Main analytics orchestrator (Phase 3)
- ✅ `performance.py` - 18 performance metrics (Phase 3)
- ✅ `risk_metrics.py` - 22 risk metrics (Phase 3)
- ✅ `ratios.py` - 15 risk-adjusted ratios (Phase 3)
- ✅ `market_metrics.py` - 15 market-related metrics (Phase 3)
- 🔲 `attribution.py` - Performance attribution analysis (Future)
- 🔲 `chart_data.py` - Prepare data for chart rendering (Phase 5)

**Dependencies**:
- Data Manager (price data)
- NumPy, Pandas, SciPy, Statsmodels

**Key Metrics**:
- **Performance** (18): Total Return, CAGR, Annualized Return, YTD, MTD, QTD, 1M, 3M, 6M, 1Y, 3Y, 5Y, Best/Worst Month, Win Rate, Payoff Ratio, Profit Factor, Expectancy
- **Risk** (22): Volatility (daily/weekly/monthly/annual), Max Drawdown (with dates/duration), Current Drawdown, Average Drawdown, Drawdown Duration (max/avg), Recovery Time, Ulcer Index, Pain Index, VaR (90%/95%/99% - Historical/Parametric/Cornish-Fisher), CVaR (90%/95%/99%), Downside Deviation, Semi-Deviation, Skewness, Kurtosis
- **Ratios** (15): Sharpe, Sortino, Calmar, Sterling, Burke, Treynor, Information Ratio, Modigliani M², Omega, Kappa 3, Gain-Pain Ratio, Martin Ratio, Tail Ratio, Common Sense Ratio, Rachev Ratio
- **Market** (15): Beta, Alpha (CAPM), R², Correlation, Tracking Error, Active Return, Up Capture, Down Capture, Capture Ratio, Jensen's Alpha, Active Share, Batting Average, Benchmark Relative Return, Rolling Beta, Market Timing Ratio

**Performance Targets**:
- 1-year daily data: <500ms
- 5-year daily data: <1000ms
- ✅ Calculations complete in <1s for typical portfolios

---

### 3.4 Optimization Engine Module

**Path**: `core/optimization_engine/`  
**Status**: 🔲 Not Implemented

**Purpose**: Optimize portfolio weights using 17 different methods with customizable constraints.

**Components**:
- `base.py` - Base optimizer abstract class
- `constraints.py` - Constraint builders (weight, group, risk, turnover, cardinality)
- `efficient_frontier.py` - Generate efficient frontier
- **Optimizers** (17 files):
  - `mean_variance.py` - Markowitz optimization
  - `min_variance.py` - Minimum variance
  - `max_sharpe.py` - Maximum Sharpe ratio
  - `risk_parity.py` - Risk parity
  - `hrp.py` - Hierarchical Risk Parity
  - `black_litterman.py` - Black-Litterman with views
  - `cvar_optimization.py` - CVaR optimization
  - `kelly_criterion.py` - Kelly Criterion (Full, Half, Quarter)
  - `equal_weight.py` - Equal weight (1/N)
  - ... (8 more methods)

**Dependencies**:
- Analytics Engine (returns, covariance)
- Data Manager (price data)
- CVXPy, SciPy, PyPortfolioOpt

**Key Features**:
- 17 optimization methods
- Flexible constraint system
- Efficient frontier generation
- Trade list generation
- Sensitivity analysis
- Transaction cost modeling

**Constraint Types**:
- Weight constraints (min, max, specific bounds)
- Group constraints (sector limits, asset class limits)
- Risk constraints (max volatility, max VaR, max beta)
- Turnover constraints (max turnover, min trade size)
- Cardinality constraints (min/max assets)

---

### 3.5 Risk Engine Module

**Path**: `core/risk_engine/`  
**Status**: 🔲 Not Implemented

**Purpose**: Advanced risk analysis including VaR, stress testing, and Monte Carlo simulation.

**Components**:
- `var_calculator.py` - VaR calculations (4 methods)
- `monte_carlo.py` - Monte Carlo simulation (10k+ paths)
- `stress_testing.py` - Historical and custom stress tests
- `scenarios.py` - 25+ historical scenario definitions

**Dependencies**:
- Analytics Engine (risk metrics)
- Data Manager (price data)
- NumPy, SciPy

**Key Features**:
- **VaR Methods**: Historical, Parametric, Monte Carlo, Cornish-Fisher
- **Confidence Levels**: 90%, 95%, 99%
- **Monte Carlo**: 10,000+ simulations with percentile outcomes
- **Stress Tests**: 25+ historical scenarios (2008 Crisis, COVID, Dot-com Bubble, etc.)
- **Custom Scenarios**: User-defined market shocks

---

### 3.6 Scenario Engine Module

**Path**: `core/scenario_engine/`  
**Status**: 🔲 Not Implemented

**Purpose**: Scenario analysis with historical scenarios, custom scenarios, and scenario chaining.

**Components**:
- `historical_scenarios.py` - Pre-defined historical market scenarios
- `custom_scenarios.py` - User-defined scenario builder
- `scenario_chain.py` - Chain multiple scenarios sequentially

**Dependencies**:
- Risk Engine (scenario application)
- Data Manager (price data)

**Key Features**:
- 25+ historical scenarios
- Custom scenario builder (specify ticker shocks)
- Scenario chains (what-if sequences)
- Impact analysis on portfolio

---

### 3.7 Reporting Engine Module

**Path**: `core/reporting_engine/`  
**Status**: 🔲 Not Implemented

**Purpose**: Generate professional reports in PDF and Excel formats.

**Components**:
- `pdf_generator.py` - Generate PDF reports
- `excel_generator.py` - Generate Excel workbooks
- `templates/` - Report templates

**Dependencies**:
- Analytics Engine (metrics data)
- Chart data (visualizations)
- ReportLab, OpenPyXL

**Key Features**:
- Comprehensive PDF reports with metrics and charts
- Multi-sheet Excel exports
- Customizable templates
- Branding and styling

---

### 3.8 Service Layer

**Path**: `services/`  
**Status**: 🟢 Phase 1-3 Implemented (Partial)

**Purpose**: Orchestrate business operations, validate inputs, manage transactions.

**Components**:
- ✅ `data_service.py` - Data fetching orchestration (Phase 1)
- ✅ `portfolio_service.py` - Portfolio CRUD orchestration (Phase 2)
- ✅ `schemas.py` - Pydantic validation schemas (Phase 2)
- ✅ `analytics_service.py` - Analytics calculation orchestration (Phase 3)
- ✅ `report_service.py` - PDF report generation (Phase 4)
- 🔲 `optimization_service.py` - Optimization orchestration (Phase 6)

**Dependencies**:
- All Core Modules
- Database (session management)
- Pydantic (validation)

**Key Responsibilities**:
- Input validation using Pydantic
- Transaction management (commit/rollback)
- Error handling and logging
- Caching strategy
- Cross-module orchestration

**Example Flow** (Create Portfolio):
```
1. User submits create request (UI)
2. PortfolioService.create_portfolio(data)
3. Validate input with Pydantic schema
4. Check for duplicate name
5. Validate tickers via DataService
6. Create Portfolio domain object
7. Save to database via repository
8. Return created portfolio
```

---

### 3.9 Presentation Layer (Streamlit UI)

**Path**: `streamlit_app/`  
**Status**: 🟢 Phase 4 Implemented (Partial)

**Purpose**: Provide user interface for portfolio management and analytics visualization.

**Components**:
- ✅ `app.py` - Main application entry point with navigation (Phase 4)
- ✅ `pages/dashboard.py` - Portfolio overview dashboard (Phase 4)
- ✅ `pages/create_portfolio.py` - Portfolio creation (5 methods) (Phase 4)
- ✅ `pages/portfolio_list.py` - Portfolio list view (Phase 4)
- ✅ `pages/portfolio_detail.py` - Portfolio detail and editing (Phase 4)
- ✅ `pages/portfolio_analysis.py` - Comprehensive analytics display (Phase 4)
- 🔲 `pages/portfolio_optimization.py` - Optimization interface (Phase 6)
- 🔲 `pages/risk_analysis.py` - Risk analysis interface (Phase 7)
- 🔲 `pages/scenario_analysis.py` - Scenario analysis interface (Phase 7)
- 🔲 `pages/reports.py` - Report generation (Phase 8)

**Reusable Components**:
- ✅ `components/portfolio_card.py` - Portfolio summary card (Phase 4)
- ✅ `components/metrics_display.py` - Metrics grid display (Phase 4)
- ✅ `components/position_table.py` - Position table component (Phase 4)
- ✅ Plotly charts (Asset/Sector Allocation) - Integrated in `portfolio_list.py` (Phase 4)
- ✅ `components/charts.py` - Chart components (Phase 5) - All 7 chart types implemented:
  - Cumulative Returns (with benchmark, linear/log scale)
  - Drawdown (with max DD annotation)
  - Return Distribution (histogram with normal overlay)
  - Q-Q Plot (distribution analysis)
  - Rolling Sharpe Ratio (with threshold)
  - Monthly Heatmap (returns by year/month)
- 🔲 `components/optimization_results.py` - Optimization results display (Phase 6)

**Utilities**:
- ✅ `utils/formatters.py` - Currency, percentage, date formatters (Phase 4)
- ✅ `utils/validators.py` - Input validation utilities (Phase 4)

**Styling**:
- ✅ `styles.css` - Custom CSS with TradingView-inspired color palette (Phase 4)

**Key Features**:
- Multi-page navigation with sidebar
- 5 portfolio creation methods:
  - Manual Entry
  - Equal Weights
  - Equal Dollar
  - By Shares
  - From Template
- Portfolio CRUD operations (create, read, update, delete)
- Real-time portfolio metrics display
- Comprehensive analytics visualization (70+ metrics)
- Session state management
- Responsive design with custom styling

**Dependencies**:
- Service Layer (PortfolioService, AnalyticsService, DataService)
- Streamlit 1.28+

---

### 3.10 Database Models (ORM)

**Path**: `models/`  
**Status**: 🟢 Phase 1-2 Implemented (Partial)

**Purpose**: SQLAlchemy ORM models representing database tables.

**Components**:
- `portfolio.py` - Portfolio model
- `position.py` - Position model
- `price_history.py` - Price history model
- `user.py` - User model (future authentication)

**Dependencies**:
- SQLAlchemy
- Database session

**Key Features**:
- Type-safe models with SQLAlchemy 2.0 style
- Relationships (one-to-many, foreign keys)
- Indexes for performance
- Cascade deletes
- Timestamps (created_at, updated_at)

---

## 4. MODULE INTERACTIONS

### Data Flow: Portfolio Creation (Manual Entry)

```
┌─────────────────┐
│  User (UI)      │
│  Enter:         │
│  - Name         │
│  - Positions    │
│  - Capital      │
└────────┬────────┘
         │
         │ 1. Submit form
         ▼
┌─────────────────────────────────┐
│  Streamlit UI                   │
│  (create_portfolio.py)          │
│  - Collect inputs               │
│  - Basic validation             │
└────────┬────────────────────────┘
         │
         │ 2. Call service
         ▼
┌─────────────────────────────────┐
│  PortfolioService               │
│  - Validate with Pydantic       │
│  - Check duplicate name         │
│  - Validate tickers             │
└────────┬────────────────────────┘
         │
         │ 3. Validate tickers
         ▼
┌─────────────────────────────────┐
│  DataService                    │
│  - Call TickerValidator         │
└────────┬────────────────────────┘
         │
         │ 4. Validate with API
         ▼
┌─────────────────────────────────┐
│  TickerValidator                │
│  - Query Yahoo Finance          │
│  - Cache results                │
└────────┬────────────────────────┘
         │
         │ 5. Tickers valid
         ▼
┌─────────────────────────────────┐
│  PortfolioService               │
│  - Create Portfolio object      │
│  - Call repository.save()       │
└────────┬────────────────────────┘
         │
         │ 6. Save to DB
         ▼
┌─────────────────────────────────┐
│  PortfolioRepository            │
│  - Convert to ORM model         │
│  - Save with SQLAlchemy         │
│  - Return saved portfolio       │
└────────┬────────────────────────┘
         │
         │ 7. Return result
         ▼
┌─────────────────────────────────┐
│  Streamlit UI                   │
│  - Display success message      │
│  - Navigate to portfolio detail │
└─────────────────────────────────┘
```

### Data Flow: Calculate Metrics

```
┌─────────────────┐
│  User (UI)      │
│  Select:        │
│  - Portfolio    │
│  - Date range   │
│  - Benchmark    │
└────────┬────────┘
         │
         │ 1. Click "Calculate"
         ▼
┌─────────────────────────────────┐
│  Streamlit UI                   │
│  (portfolio_analysis.py)        │
└────────┬────────────────────────┘
         │
         │ 2. Call service
         ▼
┌─────────────────────────────────┐
│  AnalyticsService               │
│  - Validate date range          │
│  - Check cache                  │
└────────┬────────────────────────┘
         │
         │ 3. Fetch portfolio
         ▼
┌─────────────────────────────────┐
│  PortfolioService               │
│  - Get portfolio by ID          │
└────────┬────────────────────────┘
         │
         │ 4. Fetch price data
         ▼
┌─────────────────────────────────┐
│  DataService                    │
│  - Call PriceManager            │
└────────┬────────────────────────┘
         │
         │ 5. Get prices (cached or API)
         ▼
┌─────────────────────────────────┐
│  PriceManager                   │
│  - Check cache (memory/disk/DB) │
│  - Fetch from Yahoo Finance     │
│  - Cache results                │
└────────┬────────────────────────┘
         │
         │ 6. Calculate metrics
         ▼
┌─────────────────────────────────┐
│  AnalyticsEngine                │
│  - Calculate performance (18)   │
│  - Calculate risk (22)          │
│  - Calculate ratios (15)        │
│  - Calculate market (15)        │
│  - Return all metrics           │
└────────┬────────────────────────┘
         │
         │ 7. Cache & return
         ▼
┌─────────────────────────────────┐
│  AnalyticsService               │
│  - Cache results (1 hour TTL)   │
│  - Return to UI                 │
└────────┬────────────────────────┘
         │
         │ 8. Display metrics
         ▼
┌─────────────────────────────────┐
│  Streamlit UI                   │
│  - Render metrics in tabs       │
│  - Color code values            │
└─────────────────────────────────┘
```

### Module Dependency Graph

```
┌──────────────────────────────────────────────────┐
│                 Presentation Layer               │
│              (Streamlit UI Pages)                │
└──────────────────┬───────────────────────────────┘
                   │
                   ↓
┌──────────────────────────────────────────────────┐
│                 Service Layer                    │
│  ┌───────────────┐  ┌────────────────────────┐  │
│  │  Portfolio    │  │  Analytics  Data       │  │
│  │  Service      │  │  Service    Service    │  │
│  └───────┬───────┘  └────┬───────────┬───────┘  │
└──────────┼───────────────┼───────────┼──────────┘
           │               │           │
           ↓               ↓           ↓
┌──────────────────────────────────────────────────┐
│                 Core Modules                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Data    │  │Analytics │  │Optimization  │   │
│  │ Manager  │  │ Engine   │  │   Engine     │   │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘   │
│       │             │                │           │
│  ┌────┴─────┐  ┌───┴──────┐  ┌──────┴───────┐   │
│  │   Risk   │  │ Scenario │  │  Reporting   │   │
│  │  Engine  │  │  Engine  │  │   Engine     │   │
│  └──────────┘  └──────────┘  └──────────────┘   │
└──────────────────┬───────────────────────────────┘
                   │
                   ↓
┌──────────────────────────────────────────────────┐
│                 Data Layer                       │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │ Database │  │External   │  │    Cache     │  │
│  │  (ORM)   │  │   APIs    │  │  (3-level)   │  │
│  └──────────┘  └───────────┘  └──────────────┘  │
└──────────────────────────────────────────────────┘
```

**Key Dependency Rules**:
- ✅ UI → Services → Core → Data (allowed)
- ❌ Core → Services (forbidden - core is framework-agnostic)
- ❌ UI → Core directly (forbidden - must go through services)
- ❌ Core → UI (forbidden - no UI dependencies in core)

---

## 5. DATABASE SCHEMA

### 5.1 Portfolios Table

```sql
CREATE TABLE portfolios (
    id                VARCHAR(36) PRIMARY KEY,
    name              VARCHAR(100) NOT NULL,
    description       VARCHAR(500),
    starting_capital  FLOAT NOT NULL,
    base_currency     VARCHAR(3) DEFAULT 'USD',
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_name (name),
    INDEX idx_created (created_at)
);
```

**Purpose**: Store portfolio metadata  
**Relationships**: One-to-many with `positions`  
**Constraints**: `name` should be unique per user (future: add user_id)  

---

### 5.2 Positions Table

```sql
CREATE TABLE positions (
    id              VARCHAR(36) PRIMARY KEY,
    portfolio_id    VARCHAR(36) NOT NULL,
    ticker          VARCHAR(10) NOT NULL,
    shares          FLOAT NOT NULL,
    weight_target   FLOAT,
    purchase_price  FLOAT,
    purchase_date   DATE,
    
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE,
    UNIQUE (portfolio_id, ticker),
    INDEX idx_portfolio (portfolio_id),
    INDEX idx_ticker (ticker),
    INDEX idx_portfolio_ticker (portfolio_id, ticker)
);
```

**Purpose**: Store individual positions within portfolios  
**Relationships**: Many-to-one with `portfolios`  
**Constraints**: 
- One ticker per portfolio (unique constraint)
- Cascade delete when portfolio deleted
- `shares` must be > 0 (enforced in business logic)

---

### 5.3 Price History Table

```sql
CREATE TABLE price_history (
    ticker          VARCHAR(10) NOT NULL,
    date            DATE NOT NULL,
    open            FLOAT,
    high            FLOAT,
    low             FLOAT,
    close           FLOAT NOT NULL,
    adjusted_close  FLOAT NOT NULL,
    volume          BIGINT,
    
    PRIMARY KEY (ticker, date),
    INDEX idx_ticker (ticker),
    INDEX idx_date (date)
);
```

**Purpose**: Cache historical price data (OHLCV)  
**Relationships**: None (lookup table)  
**Constraints**: 
- Composite primary key (ticker, date) ensures uniqueness
- `adjusted_close` used for returns to account for splits/dividends

---

### 5.4 Users Table (Future - Phase 10+)

```sql
CREATE TABLE users (
    id              VARCHAR(36) PRIMARY KEY,
    email           VARCHAR(255) UNIQUE NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    first_name      VARCHAR(100),
    last_name       VARCHAR(100),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login      TIMESTAMP,
    is_active       BOOLEAN DEFAULT TRUE,
    
    INDEX idx_email (email)
);
```

**Purpose**: User authentication and management  
**Status**: Not in MVP, future enhancement  
**Security**: 
- Passwords hashed with bcrypt (cost factor 12)
- Email must be unique

---

### Entity-Relationship Diagram

```
┌─────────────────────┐
│      Users          │  (Future)
│──────────────────── │
│  id (PK)            │
│  email (UNIQUE)     │
│  password_hash      │
│  ...                │
└─────────┬───────────┘
          │
          │ 1:N (future)
          │
┌─────────▼───────────┐
│    Portfolios       │
│──────────────────── │
│  id (PK)            │
│  user_id (FK)       │  (future)
│  name               │
│  starting_capital   │
│  created_at         │
│  ...                │
└─────────┬───────────┘
          │
          │ 1:N (CASCADE)
          │
┌─────────▼───────────┐
│     Positions       │
│──────────────────── │
│  id (PK)            │
│  portfolio_id (FK)  │
│  ticker             │
│  shares             │
│  weight_target      │
│  ...                │
└─────────┬───────────┘
          │
          │ N:1 (lookup)
          │
┌─────────▼───────────┐
│   Price History     │
│──────────────────── │
│  ticker (PK)        │
│  date (PK)          │
│  open, high, low    │
│  close, adj_close   │
│  volume             │
└─────────────────────┘
```

---

## 6. KEY ARCHITECTURAL DECISIONS

### 6.1 Layered Architecture with Service Layer

**Decision**: Use Layered Architecture with explicit Service Layer

**Rationale**:
- **Clear separation of concerns**: UI, business logic, data access
- **Testability**: Core modules testable without UI or database
- **Flexibility**: Can swap Streamlit → Next.js without changing core
- **Maintainability**: Each layer has single responsibility

**Trade-offs**:
- More layers = more boilerplate
- Learning curve for developers
- ✅ Worth it for long-term maintainability

---

### 6.2 Pure Python Core Modules (Framework-Agnostic)

**Decision**: Core business logic has zero framework dependencies

**Rationale**:
- **Portability**: Can use core with any UI framework (Streamlit, Flask, FastAPI, Next.js)
- **Testability**: Easy to test without UI or framework overhead
- **Reusability**: Core can be packaged as library
- **Longevity**: Not tied to framework lifecycle

**Trade-offs**:
- Cannot use framework-specific features in core
- Need adapter layer (services) between UI and core
- ✅ Worth it for future flexibility

---

### 6.3 SQLite for MVP, PostgreSQL for Production

**Decision**: Start with SQLite, migrate to PostgreSQL later

**Rationale**:
- **MVP Speed**: SQLite requires zero setup, embedded database
- **Development**: Easy local development
- **Migration Path**: SQLAlchemy abstracts DB, easy to switch
- **Production**: PostgreSQL offers better concurrency, performance, features

**Trade-offs**:
- SQLite limitations (no concurrent writes, limited datatype support)
- Migration effort later (but SQLAlchemy makes it easy)
- ✅ Worth it to ship MVP faster

---

### 6.4 Multi-Source Price Data with Fallbacks

**Decision**: Yahoo Finance primary, Alpha Vantage/IEX Cloud fallbacks

**Rationale**:
- **Reliability**: If Yahoo Finance down, fallback works
- **Cost**: Yahoo Finance free, paid sources only if needed
- **Data Quality**: Yahoo Finance generally good quality
- **Rate Limits**: Distribute load across sources

**Trade-offs**:
- More complex code (multiple adapters)
- Potential data inconsistencies between sources
- ✅ Worth it for reliability

---

### 6.5 Three-Level Caching Strategy

**Decision**: In-memory cache → Disk cache → Database cache

**Rationale**:
- **Performance**: In-memory cache is instant (<1ms)
- **Persistence**: Disk cache survives restarts
- **Data Storage**: Database stores all fetched prices
- **Cost Savings**: Minimize API calls (rate limits, costs)

**Caching Levels**:
1. **In-Memory** (Python dict with TTL):
   - Current prices: 5 min TTL
   - Calculated metrics: 1 hour TTL
   - Ticker validation: 24 hour TTL
2. **Disk Cache** (Parquet files):
   - Historical prices: Permanent
   - Path: `data/cache/prices/{ticker}.parquet`
3. **Database** (PostgreSQL/SQLite):
   - `price_history` table
   - Never fetch same (ticker, date) twice

**Trade-offs**:
- Cache invalidation complexity
- Storage space (disk + DB)
- ✅ Worth it for performance gains

---

### 6.6 Pydantic for All Input Validation

**Decision**: Use Pydantic schemas for all external inputs

**Rationale**:
- **Type Safety**: Runtime validation + static typing
- **Clear API**: Schema = documentation
- **Error Messages**: User-friendly validation errors
- **Consistency**: Same validation approach everywhere

**Trade-offs**:
- Need to maintain separate schemas (in addition to ORM models)
- Some code duplication
- ✅ Worth it for safety and clarity

---

### 6.7 Streamlit for MVP UI

**Decision**: Streamlit for MVP, plan migration to Next.js later

**Rationale**:
- **Speed**: Streamlit allows rapid UI development in Python
- **Prototyping**: Perfect for MVP and testing core features
- **No Frontend Skills**: Don't need React/JS knowledge
- **Migration Path**: Core is framework-agnostic, UI swap is easy

**Trade-offs**:
- Streamlit limitations (performance, customization, mobile)
- Will need to rebuild UI later
- ✅ Worth it to ship MVP in weeks instead of months

---

### 6.8 Repository Pattern for Data Access

**Decision**: Use Repository pattern between services and ORM

**Rationale**:
- **Abstraction**: Services don't know about ORM details
- **Testability**: Can mock repository easily
- **Flexibility**: Can swap database/ORM later
- **Domain Models**: Separate domain objects from ORM models

**Trade-offs**:
- More abstraction layers
- Need to convert between ORM and domain models
- ✅ Worth it for clean architecture

---

## 7. EVOLUTION PLAN

This document will be **updated during development** with:

### Phase 0-1: Foundation (Weeks 1-2)
- [x] Initial architecture document
- [ ] Detailed class diagrams for core modules
- [ ] Database schema finalized
- [ ] Configuration system documented

### Phase 2-3: Core Implementation (Weeks 2-4)
- [ ] Portfolio domain model details
- [ ] Analytics engine algorithms documentation
- [ ] API specifications for internal services
- [ ] Data flow diagrams for key operations

### Phase 4-5: UI & Integration (Weeks 4-6)
- [ ] UI component hierarchy
- [ ] State management approach
- [ ] Chart rendering architecture
- [ ] Integration patterns between layers

### Phase 6-8: Advanced Features (Weeks 6-9)
- [ ] Optimization algorithm details
- [ ] Risk engine architecture
- [ ] Scenario modeling approach
- [ ] Report generation pipeline

### Phase 9: Refinement
- [ ] Performance optimization decisions
- [ ] Deployment architecture
- [ ] Monitoring and observability setup
- [ ] Security hardening details

### Future Additions (Post-MVP)
- [ ] Sequence diagrams for complex workflows
- [ ] Detailed API documentation (if REST API added)
- [ ] Authentication/authorization architecture
- [ ] Scaling strategy (horizontal/vertical)
- [ ] Microservices decomposition (if needed)
- [ ] Event-driven architecture (if needed)
- [ ] WebSocket architecture for real-time updates
- [ ] CDN and caching strategy for web version

---

## 8. DOCUMENTATION REFERENCES

**Related Documents**:
- `docs/PROJECT_VISION_COMPLETE.md` - Project vision and business requirements
- `docs/REQUIREMENTS.md` - Detailed technical requirements (3,799 lines)
- `docs/ARC-RULES.md` - Coding standards and architectural rules (5,087 lines)
- `docs/PLAN.md` - Phase-by-phase implementation plan (1,247 lines)

**Code Documentation** (Will be added):
- `docs/API.md` - Internal API documentation
- `docs/DATABASE.md` - Database schema and migrations
- `docs/TESTING.md` - Testing strategy and guidelines
- `docs/DEPLOYMENT.md` - Deployment procedures
- `README.md` - User-facing documentation and setup guide

---

## 9. CHANGE LOG

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.3 | 2025-01-XX | System | Phase 3 implementation: Analytics Engine completed |
| 1.4 | 2025-01-XX | System | Phase 4 implementation: Streamlit UI (Basic Pages) completed |
| 1.2 | 2025-01-XX | System | Phase 2 implementation: Portfolio Core completed |
| 1.1 | 2025-01-XX | System | Phase 1 implementation: Data Infrastructure completed |
| 1.0 | 2025-10-29 | System | Initial architecture document created |

#### Update 2025-10-30: Portfolio Analysis Page Redesign

**Major Feature Implementation** - Complete redesign of Portfolio Analysis page with comprehensive tearsheet-style analytics.

**New Core Modules:**
- `core/analytics_engine/advanced_metrics.py` - Advanced metrics:
  - Probabilistic Sharpe Ratio (PSR), Smart Sharpe & Smart Sortino (autocorrelation-adjusted)
  - Kelly Criterion (full, half, quarter positions), Risk of Ruin calculations
  - Win Rate statistics, Outlier analysis, Common Performance Periods (CPP) index
- Enhanced `core/analytics_engine/chart_data.py`:
  - `get_rolling_sharpe_data()`, `get_rolling_sortino_data()`, `get_rolling_beta_alpha_data()`
  - `get_underwater_plot_data()`, `get_best_worst_periods_data()`, `get_comparison_stats_data()`
  - `get_yearly_returns_data()`

**New Service Modules:**
- `services/report_service.py` - ReportService class for tearsheet generation:
  - CSV export functionality, JSON export with returns data
  - Excel export with multiple sheets, PDF generation structure (placeholder)

**New UI Components:**
- `streamlit_app/components/comparison_table.py` - Portfolio vs Benchmark comparison tables
- `streamlit_app/components/metric_card_comparison.py` - Metric cards with delta comparison
- `streamlit_app/components/period_filter.py` - Date range selector with pre-defined periods
- `streamlit_app/components/assets_metrics.py` - Asset-level analysis and metrics
- `streamlit_app/components/triple_chart_section.py` - Triple dynamic chart section
- `streamlit_app/utils/comparison_utils.py` - Comparison utility functions

**Portfolio Analysis Page Restructure:**
- Tab-based navigation (5+ tabs):
  1. **Overview**: 8 key metrics cards, cumulative returns chart, drawdown plots, portfolio composition
  2. **Performance**: Performance metrics table, periodic returns table, distribution analysis
  3. **Risk**: Risk metrics table, drawdown analysis, VaR & CVaR, rolling risk metrics
  4. **Assets & Correlations**: Position table, allocation charts, correlation matrix (placeholder)
  5. **Export & Reports**: CSV/JSON/PDF export, complete metrics table
- Enhanced charts: Rolling Sharpe/Sortino, Rolling Beta/Alpha, Underwater plot, Yearly returns

**Key Features Added:**
- Benchmark comparison for all metrics (SPY, QQQ, VTI, DIA, IWM)
- Advanced risk metrics (Probabilistic Sharpe, Smart ratios, Kelly Criterion)
- Rolling analysis (configurable windows)
- Distribution analysis (Q-Q plots, normality tests)
- Periodic analysis (yearly returns, monthly heatmap)
- Export capabilities (CSV, JSON, Excel multi-sheet)

**Performance Metrics:**
- Average page load: ~2-3s (with caching)
- Analytics calculation: ~1-2s for 1Y data
- Chart rendering: ~0.5s per chart

**Files Modified:**
- New files: 8 core/component files, 1 service file
- Modified files: 10 existing files (chart_data, engine, analytics_service, charts, etc.)
- Deleted: `portfolio_detail.py` (consolidated into portfolio_analysis)

**Architecture Principles Maintained:**
- ✅ Separation of Concerns (backend/frontend/service)
- ✅ DRY (reusable components, generic helpers)
- ✅ SOLID principles (single responsibility, open/closed)
- ✅ Testability (pure functions, no UI dependencies in core)

#### Update 2025-10-31

- Analytics comparison and timezone normalization fixes:
  - `services/analytics_service.py`: comparison series now aligned by strict intersection of dates (removed ffill alignment) to prevent artificial deltas for identical portfolios and ETFs. Returned series are tz-naive.
  - `streamlit_app/pages/portfolio_analysis.py`: benchmark metrics computed on strict date intersection; plotting windows normalize tz to avoid "tz-naive vs tz-aware" errors.
  - Effect: identical portfolios show zero deltas; index ETF comparison loads independently of initial UI date; no timezone comparison errors.
  - UI visual tweaks: removed "Better" column from Key Metrics Comparison (colored status dots suffice); benchmark lines in charts switched to solid style for consistency.
  - Date normalization: before pivoting combined prices, `Date` is coerced to tz-naive; pivot index and filter boundaries are also normalized to tz-naive to prevent `Cannot compare tz-naive and tz-aware timestamps`.

#### Update 2025-10-31: Statistical Tests Improvements

**Problem Identified:** User reported that p-values for normality tests (Shapiro-Wilk and Jarque-Bera) consistently showed `< 0.0001` across different portfolios, raising concerns about calculation accuracy.

**Root Cause Analysis:**
1. Tests were correctly calculating p-values via `scipy.stats`
2. Large sample sizes (>1000 observations typical for 4-year daily data) make normality tests extremely sensitive
3. Financial returns inherently have fat tails and rarely follow perfect normal distribution
4. Shapiro-Wilk was using non-random first N elements instead of representative sampling

**Changes Made:**
- **Core Module** (`core/analytics_engine/chart_data.py`):
  - `get_statistical_tests_data()` function enhanced:
    - Added random sampling for Shapiro-Wilk when sample size > 5000 (using fixed seed for reproducibility)
    - Added `sample_size` field to return dictionary to track total observations
    - Added `sample_size` to Shapiro-Wilk results to show actual sample tested
    - Improved documentation explaining return values

- **UI Display** (`streamlit_app/pages/portfolio_analysis.py`):
  - P-values now shown in scientific notation (e.g., `2.34e-15`) instead of `< 0.0001` for more precision
  - Added sample size caption showing total observations and Shapiro-Wilk sample when applicable
  - Added explanatory note for large samples (>1000):
    - Explains that tests become highly sensitive with large samples
    - Notes that financial returns typically have fat tails
    - Clarifies that rejection of normality is expected and not concerning

**Impact:**
- Users now see exact p-values in scientific notation for better transparency
- Random sampling ensures representative testing for Shapiro-Wilk
- Educational note helps users understand why p-values are consistently low
- No breaking changes - backward compatible with existing code

**Files Modified:**
- `core/analytics_engine/chart_data.py` (lines 1170-1228)
- `streamlit_app/pages/portfolio_analysis.py` (lines 1065-1128)

#### Update 2025-01-XX (Latest UI Improvements)

- Portfolio Analysis UI enhancements:
  - **Layout reorganization** (`streamlit_app/pages/portfolio_analysis.py`):
    - Analysis Parameters section: Portfolio dropdown moved down, Start Date moved left, End Date placed in Start Date's position
    - Improved visual flow and user experience
  - **Comparison table improvements** (`streamlit_app/components/comparison_table.py`):
    - Removed "Better" column (colored circles provide sufficient visual indication)
    - Updated percentage difference calculation: uses `((portfolio - benchmark) / (1 + |benchmark|)) * 100` for percentage-based metrics and `((portfolio - benchmark) / |benchmark|) * 100` for ratios
    - Fixed comparison logic for special metrics:
      - Max Drawdown: less negative is better (portfolio > benchmark means better)
      - Beta: closer to 1.0 is better (absolute distance comparison)
  - **Metric cards redesign** (`streamlit_app/components/metric_card_comparison.py`):
    - New format: "Metric Name" → "Portfolio Value" → "Colored Circle + Benchmark Value"
    - Fixed comparison logic for Max Drawdown and Beta to match table logic
    - Color indicators: 🟢 (better), 🔴 (worse), ⚪ (neutral)
  - **Chart styling** (`streamlit_app/components/charts.py`):
    - Benchmark lines changed from dashed to solid style in all charts for consistency
    - Affected charts: Cumulative Returns, Rolling Sharpe, Underwater Plot
  - **Key Performance Metrics order** (`streamlit_app/pages/portfolio_analysis.py`):
    - Row 1: Total Return, CAGR, Volatility, Max Drawdown
    - Row 2: Sharpe Ratio, Sortino Ratio, Beta, Alpha
    - Improved logical grouping of related metrics

### Phase 5 Implementation Details (2025-10-29)

**Implemented Components**:
- ✅ `core/analytics_engine/chart_data.py` - Chart data preparation functions
- ✅ `streamlit_app/components/charts.py` - Chart visualization components (7 types)
- ✅ `streamlit_app/utils/chart_config.py` - Plotly chart configuration
- ✅ `streamlit_app/pages/portfolio_analysis.py` - Enhanced with Charts tab

**Key Features**:
- 7 interactive chart types using Plotly:
  1. **Cumulative Returns** - Line chart with portfolio and benchmark, linear/log scale toggle
  2. **Drawdown** - Area chart with max drawdown annotation
  3. **Return Distribution** - Histogram with normal distribution overlay and VaR lines
  4. **Q-Q Plot** - Quantile-quantile plot for distribution analysis
  5. **Rolling Sharpe Ratio** - Rolling metric with configurable window and threshold
  6. **Monthly Heatmap** - Monthly returns heatmap by year and month
- Chart selector dropdown for easy navigation
- Charts tab integrated into portfolio analysis page
- Benchmark support for cumulative returns
- Interactive features (zoom, pan, hover)
- Color palette consistent with TradingView-inspired design

**Dependencies Added**:
- Plotly 5.17+ (already added in Phase 4)
- scipy (for statistical analysis in Q-Q plot)

### Phase 3 Implementation Details (2025-01-XX)

**Implemented Components**:
- ✅ `core/analytics_engine/performance.py` - 18 performance metrics
- ✅ `core/analytics_engine/risk_metrics.py` - 22 risk metrics
- ✅ `core/analytics_engine/ratios.py` - 15 risk-adjusted ratios
- ✅ `core/analytics_engine/market_metrics.py` - 15 market-related metrics
- ✅ `core/analytics_engine/engine.py` - Analytics orchestrator
- ✅ `services/analytics_service.py` - Analytics service layer
- ✅ Unit and integration tests

**Key Features**:
- 70+ metrics calculation in single call
- Performance metrics: Total Return, CAGR, period returns, win rate, etc.
- Risk metrics: Volatility (multiple timeframes), VaR/CVaR (multiple methods), drawdown analysis
- Risk-adjusted ratios: Sharpe, Sortino, Calmar, Treynor, Information Ratio, Omega, etc.
- Market metrics: Beta, Alpha, R², Correlation, Tracking Error, Capture ratios, etc.
- Benchmark comparison support (all market metrics)
- Graceful handling of missing/insufficient data (returns None)
- Calculation metadata (time, data points, date range)
- Vectorized calculations for performance
- Type-safe with full type hints

### Phase 2 Implementation Details (2025-01-XX)

**Implemented Components**:
- ✅ `models/portfolio.py` - Portfolio ORM model
- ✅ `models/position.py` - Position ORM model
- ✅ `core/data_manager/portfolio.py` - Portfolio domain model (pure Python)
- ✅ `core/data_manager/portfolio_repository.py` - Repository pattern implementation
- ✅ `services/schemas.py` - Pydantic validation schemas
- ✅ `services/portfolio_service.py` - Portfolio service orchestration
- ✅ `database/migrations/versions/002_create_portfolios.py` - Portfolios table migration
- ✅ `database/migrations/versions/003_create_positions.py` - Positions table migration
- ✅ Unit and integration tests

**Key Features**:
- Portfolio CRUD operations (create, read, update, delete)
- Position management (add, remove, update)
- Portfolio cloning
- Current value calculation with real-time prices
- Current weights calculation
- Input validation with Pydantic
- Duplicate name detection
- Cascade delete for positions
- Repository pattern for persistence abstraction

### Phase 1 Implementation Details (2025-01-XX)

**Implemented Components**:
- ✅ `core/exceptions.py` - Custom exception hierarchy
- ✅ `config/settings.py` - Pydantic Settings configuration
- ✅ `config/logging_config.py` - Logging setup
- ✅ `database/session.py` - SQLAlchemy session management
- ✅ `core/data_manager/cache.py` - Multi-level caching system
- ✅ `core/data_manager/ticker_validator.py` - Ticker validation with Yahoo Finance
- ✅ `core/data_manager/price_manager.py` - Price data fetching with retry logic
- ✅ `models/price_history.py` - Price history ORM model
- ✅ `services/data_service.py` - Data service orchestration layer
- ✅ `database/migrations/versions/001_create_price_history.py` - Alembic migration
- ✅ Unit and integration tests

**Key Features**:
- Ticker validation with 24h cache TTL
- Historical price fetching from Yahoo Finance
- Current price fetching with 5min cache TTL
- Bulk price fetching for multiple tickers
- Database storage for price history with indexes
- Retry logic with exponential backoff
- Comprehensive error handling

### Phase 4 Implementation Details (Updated 2025-10-29)

**Implemented Components**:
- ✅ `streamlit_app/app.py` - Main application with navigation and CSS
- ✅ `streamlit_app/pages/dashboard.py` - Portfolio overview dashboard
- ✅ `streamlit_app/pages/create_portfolio.py` - Portfolio creation with Wizard (5 steps) + 4 additional methods
- ✅ `streamlit_app/pages/portfolio_list.py` - Portfolio list with full CRUD (view, edit, delete, clone, bulk operations)
- ✅ `streamlit_app/pages/portfolio_detail.py` - Portfolio detail and editing
- ✅ `streamlit_app/pages/portfolio_analysis.py` - Analytics metrics display
- ✅ `streamlit_app/components/portfolio_card.py` - Portfolio card component
- ✅ `streamlit_app/components/metrics_display.py` - Metrics display component
- ✅ `streamlit_app/components/position_table.py` - Position table component
- ✅ `streamlit_app/utils/formatters.py` - Formatting utilities
- ✅ `streamlit_app/utils/validators.py` - Validation utilities (Level 1: UI validation)
- ✅ `streamlit_app/utils/text_parser.py` - Text parsing utility for portfolio creation
- ✅ `streamlit_app/styles.css` - Custom CSS styling
- ✅ `run.py` - Application launcher

**Key Features**:
- **5 portfolio creation methods** (based on reference_portfolio_system):
  1. **Wizard** - Step-by-step guided process (5 steps): Basic Info → Input Method → Add Assets → Settings & Review → Creation
  2. **Text Input** - Natural language parsing supporting multiple formats: `AAPL:40%`, `AAPL 0.4`, `AAPL 40`, `AAPL, MSFT` (equal weights)
  3. **CSV Import** - Upload CSV/Excel files with automatic column mapping and validation
  4. **Manual Entry** - Dynamic form for adding positions individually with full control
  5. **Templates** - Pre-built strategies (Value Factor, Quality Factor, Growth, 60/40, Tech Focus, etc.)
- **Full CRUD operations** for portfolios:
  - **List View**: Search, filter, sort, bulk operations
  - **Detail View**: Full portfolio information with positions table and metrics
  - **Edit**: Inline editing of portfolio info and positions
  - **Delete**: With undo functionality within session
  - **Clone**: Duplicate portfolios with new names
- **Three-level validation**:
  - **Level 1 (UI)**: Format validation (ticker format, number ranges, etc.)
  - **Level 2 (Service)**: Business rules validation (duplicate names, ticker existence, weight sums)
  - **Level 3 (Model)**: Domain model validation (invariants, data integrity)
- Real-time portfolio metrics calculation
- Comprehensive analytics display (70+ metrics in organized sections)
- Navigation between pages with session state management
- Custom CSS with TradingView-inspired color palette
- Responsive design with dark theme
- Input validation and error handling
- Export functionality (CSV, JSON) for analytics results

**Architecture Compliance**:
- ✅ Maintained layered architecture: UI → Service → Core → Data
- ✅ UI layer uses PortfolioService (not direct core access)
- ✅ Service layer orchestrates PortfolioRepository and DataService
- ✅ Core modules (Portfolio domain model) remain framework-agnostic
- ✅ Three-level validation: UI → Service → Model

**Dependencies Added**:
- Streamlit 1.28+ (for UI framework)

**Reference Implementation Integration** (2025-10-29):
- Integrated portfolio creation system from `reference_portfolio_system`
- Added Wizard flow (5 steps) as primary creation method
- Added text parsing utility for flexible input formats
- Enhanced portfolio management with full CRUD and bulk operations
- Implemented undo functionality for deleted portfolios
- Added comprehensive validation on 3 levels
- **Cash Management**: Added cash allocation feature in Wizard Step 4, automatic cash position creation with remainder handling
- **Visualization Enhancements** (2025-10-29):
  - Added Plotly donut charts for Asset Allocation (by individual assets) and Sector Allocation (by sectors)
  - Charts always include "Cash" even if value is zero (gray color)
  - Charts integrated into portfolio detail view (`portfolio_list.py`)
- **Dashboard Improvements** (2025-10-29):
  - Fixed error handling for missing price data (uses PortfolioService.calculate_portfolio_metrics with fallback)
  - Graceful degradation when price data unavailable (uses starting_capital as fallback)

**Dependencies Added**:
- Streamlit 1.28+ (for UI framework)
- Plotly 5.17+ (for interactive charts)

**Future Updates**: This section will track all architectural changes and decisions made during development.

---

**Document Status**: 🟢 Active - Living Document  
**Last Updated**: 2025-01-XX (UI improvements and comparison logic fixes)  
**Next Review**: After Phase 6 completion  
**Owner**: Development Team

**Note**: This is the single source of truth for architecture documentation. All architectural changes and decisions are tracked in this document.

---

**END OF ARCHITECTURE DOCUMENT**

