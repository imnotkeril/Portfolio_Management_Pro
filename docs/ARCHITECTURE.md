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
**Status**: 🔲 Not Implemented

**Purpose**: Calculate comprehensive portfolio analytics including 70+ metrics across 4 categories.

**Components**:
- `engine.py` - Main analytics orchestrator
- `performance.py` - 18 performance metrics (Return, CAGR, Win Rate, etc.)
- `risk_metrics.py` - 22 risk metrics (Volatility, Drawdown, VaR, CVaR, etc.)
- `ratios.py` - 15 risk-adjusted ratios (Sharpe, Sortino, Calmar, etc.)
- `market_metrics.py` - 15 market-related metrics (Beta, Alpha, R², etc.)
- `attribution.py` - Performance attribution analysis
- `chart_data.py` - Prepare data for chart rendering

**Dependencies**:
- Data Manager (price data)
- NumPy, Pandas, SciPy, Statsmodels

**Key Metrics**:
- **Performance**: Total Return, CAGR, YTD, MTD, Best/Worst Month, Win Rate, Payoff Ratio, Profit Factor, Expectancy
- **Risk**: Volatility (multiple timeframes), Max Drawdown, VaR (90/95/99%), CVaR, Skewness, Kurtosis, Ulcer Index
- **Ratios**: Sharpe, Sortino, Calmar, Treynor, Information Ratio, Omega, Tail Ratio, Martin Ratio
- **Market**: Beta, Alpha, R², Correlation, Tracking Error, Up/Down Capture, Active Share, Batting Average

**Performance Targets**:
- 1-year daily data: <500ms
- 5-year daily data: <1000ms

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
**Status**: 🔲 Not Implemented

**Purpose**: Orchestrate business operations, validate inputs, manage transactions.

**Components**:
- `portfolio_service.py` - Portfolio CRUD orchestration
- `analytics_service.py` - Analytics calculation orchestration
- `optimization_service.py` - Optimization orchestration
- `data_service.py` - Data fetching orchestration
- `schemas.py` - Pydantic validation schemas

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

### 3.9 Database Models (ORM)

**Path**: `models/`  
**Status**: 🔲 Not Implemented

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

### 3.10 Streamlit UI (Presentation Layer)

**Path**: `streamlit_app/`  
**Status**: 🔲 Not Implemented

**Purpose**: User interface for portfolio management and analysis.

**Components**:

**Pages** (`pages/`):
- `dashboard.py` - Portfolio overview dashboard
- `create_portfolio.py` - 5 creation methods (Text, CSV, Manual, Template, Clone)
- `portfolio_list.py` - List all portfolios with filters
- `portfolio_detail.py` - Portfolio detail and editing
- `portfolio_analysis.py` - Metrics and charts display
- `optimization.py` - Optimization interface
- `risk_analysis.py` - Risk analysis interface
- `scenarios.py` - Scenario analysis interface
- `reports.py` - Report generation interface

**Components** (`components/`):
- `portfolio_card.py` - Portfolio summary card
- `metrics_display.py` - Metrics display component
- `position_table.py` - Position table with sorting
- `charts.py` - All chart implementations (7 types)

**Utilities** (`utils/`):
- `formatters.py` - Number, currency, date formatting
- `validators.py` - Input validation helpers
- `state_management.py` - Session state helpers
- `chart_config.py` - Chart configuration and theming

**Dependencies**:
- Service Layer (all services)
- Streamlit, Plotly

**Key Features**:
- Multi-page navigation
- 5 portfolio creation methods
- Interactive charts (zoom, pan, export)
- Real-time validation
- Custom styling (TradingView-inspired palette)
- Responsive design

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
| 1.0 | 2025-10-29 | System | Initial architecture document created |

**Future Updates**: This section will track all architectural changes and decisions made during development.

---

**Document Status**: 🟢 Active - Living Document  
**Next Review**: After Phase 2 completion  
**Owner**: Development Team

---

**END OF ARCHITECTURE DOCUMENT**

