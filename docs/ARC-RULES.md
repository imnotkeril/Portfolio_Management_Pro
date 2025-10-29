# WILD MARKET CAPITAL - Architecture Rules

**Version**: 1.0  
**Last Updated**: 2025-10-29  
**Status**: Active Architecture Guidelines

---

## 1. TECHNOLOGY STACK

### Core Technologies

**Backend (Core Business Logic)**:
- **Language**: Python 3.9+ (recommended 3.11+)
- **Required Libraries**:
  ```txt
  numpy>=1.24.0          # Numerical computations
  pandas>=2.0.0          # Data manipulation
  scipy>=1.11.0          # Scientific computing, optimization
  statsmodels>=0.14.0    # Statistical models, econometrics
  cvxpy>=1.4.0           # Convex optimization
  yfinance>=0.2.0        # Market data (Yahoo Finance)
  plotly>=5.18.0         # Interactive charts
  ```

**Frontend (MVP)**:
- **Framework**: Streamlit 1.28+
- **Charts**: Plotly 5.18+
- **Styling**: Custom CSS

**Data Storage**:
- **Database**: SQLite (MVP) → PostgreSQL 14+ (production)
- **ORM**: SQLAlchemy 2.0+
- **Migrations**: Alembic

**Development Tools**:
- **Type Checking**: mypy (strict mode)
- **Linting**: ruff (replaces flake8, isort, black)
- **Testing**: pytest 7.0+, pytest-cov
- **Pre-commit**: pre-commit hooks

### Version Requirements

```txt
# Core
python>=3.9,<3.14
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0

# Financial
yfinance>=0.2.0
statsmodels>=0.14.0
cvxpy>=1.4.0

# Web/UI (MVP)
streamlit>=1.28.0
plotly>=5.18.0

# Database
sqlalchemy>=2.0.0
alembic>=1.12.0

# Development
pytest>=7.0.0
pytest-cov>=4.0.0
mypy>=1.5.0
ruff>=0.1.0
```

---

## 2. ARCHITECTURAL PATTERNS

### 2.1 Architecture Style: **Layered Architecture + Service Layer**

```
┌──────────────────────────────────────┐
│         Presentation Layer           │
│    (Streamlit UI / Future API)       │
└──────────────────┬───────────────────┘
                   │
┌──────────────────▼───────────────────┐
│         Service Layer                │
│  (Business Logic Orchestration)      │
└──────────────────┬───────────────────┘
                   │
┌──────────────────▼───────────────────┐
│         Core Modules                 │
│  (Domain Logic - Pure Python)        │
│  ┌────────────────────────────────┐  │
│  │  Data Manager                  │  │
│  │  Analytics Engine              │  │
│  │  Optimization Engine           │  │
│  │  Risk Engine                   │  │
│  │  Scenario Engine               │  │
│  └────────────────────────────────┘  │
└──────────────────┬───────────────────┘
                   │
┌──────────────────▼───────────────────┐
│         Data Layer                   │
│  (Database, External APIs, Cache)    │
└──────────────────────────────────────┘
```

### 2.2 Project Structure

```
WMC_Portfolio_Terminal/
│
├── config/                         # Configuration
│   ├── __init__.py
│   ├── settings.py                 # Central config (loads from .env)
│   └── logging_config.py           # Logging setup
│
├── core/                           # Core Business Logic (Pure Python)
│   ├── __init__.py
│   │
│   ├── data_manager/               # Data access and management
│   │   ├── __init__.py
│   │   ├── portfolio.py            # Portfolio data operations
│   │   ├── price_manager.py        # Price data fetching/caching
│   │   ├── ticker_validator.py     # Ticker validation
│   │   └── parsers/                # Data parsers (CSV, text, etc.)
│   │       ├── __init__.py
│   │       ├── csv_parser.py
│   │       └── text_parser.py
│   │
│   ├── analytics_engine/           # Performance & analytics
│   │   ├── __init__.py
│   │   ├── performance.py          # Performance metrics (18 metrics)
│   │   ├── risk_metrics.py         # Risk metrics (22 metrics)
│   │   ├── ratios.py               # Risk-adjusted ratios (15 ratios)
│   │   ├── market_metrics.py       # Market-related metrics (15 metrics)
│   │   └── attribution.py          # Performance attribution
│   │
│   ├── optimization_engine/        # Portfolio optimization
│   │   ├── __init__.py
│   │   ├── base.py                 # Base optimizer class
│   │   ├── mean_variance.py        # Markowitz optimization
│   │   ├── risk_parity.py          # Risk parity
│   │   ├── hrp.py                  # Hierarchical Risk Parity
│   │   ├── black_litterman.py      # Black-Litterman
│   │   ├── constraints.py          # Constraint builders
│   │   └── utils.py                # Optimization utilities
│   │
│   ├── risk_engine/                # Risk management
│   │   ├── __init__.py
│   │   ├── var_calculator.py       # VaR calculations
│   │   ├── stress_testing.py       # Stress tests
│   │   ├── monte_carlo.py          # Monte Carlo simulation
│   │   └── scenarios.py            # Scenario definitions
│   │
│   ├── scenario_engine/            # Scenario analysis
│   │   ├── __init__.py
│   │   ├── historical_scenarios.py # Historical market scenarios
│   │   ├── custom_scenarios.py     # Custom scenario builder
│   │   └── scenario_chain.py       # Chained scenarios
│   │
│   └── reporting_engine/           # Report generation
│       ├── __init__.py
│       ├── pdf_generator.py        # PDF reports
│       ├── excel_generator.py      # Excel exports
│       └── templates/              # Report templates
│
├── services/                       # Service Layer (Orchestration)
│   ├── __init__.py
│   ├── portfolio_service.py        # Portfolio CRUD + orchestration
│   ├── analytics_service.py        # Analytics orchestration
│   ├── optimization_service.py     # Optimization orchestration
│   └── data_service.py             # Data fetching orchestration
│
├── models/                         # Data models (SQLAlchemy)
│   ├── __init__.py
│   ├── portfolio.py                # Portfolio ORM model
│   ├── position.py                 # Position ORM model
│   ├── price_history.py            # Price history ORM model
│   └── user.py                     # User model (future)
│
├── database/                       # Database utilities
│   ├── __init__.py
│   ├── session.py                  # DB session management
│   ├── migrations/                 # Alembic migrations
│   └── seed_data.py                # Sample data (dev/test)
│
├── streamlit_app/                  # Streamlit UI (MVP)
│   ├── __init__.py
│   ├── app.py                      # Main entry point
│   │
│   ├── pages/                      # Streamlit pages
│   │   ├── __init__.py
│   │   ├── dashboard.py
│   │   ├── create_portfolio.py
│   │   ├── portfolio_list.py
│   │   ├── portfolio_detail.py
│   │   ├── portfolio_analysis.py
│   │   ├── risk_analysis.py
│   │   ├── optimization.py
│   │   ├── scenarios.py
│   │   └── reports.py
│   │
│   ├── components/                 # Reusable UI components
│   │   ├── __init__.py
│   │   ├── portfolio_card.py
│   │   ├── metrics_display.py
│   │   ├── charts.py
│   │   └── forms.py
│   │
│   └── utils/                      # UI utilities
│       ├── __init__.py
│       ├── formatters.py           # Number/date formatting
│       ├── validators.py           # Input validation
│       └── state_management.py     # Session state helpers
│
├── tests/                          # Tests
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures
│   ├── unit/                       # Unit tests
│   │   ├── test_analytics.py
│   │   ├── test_optimization.py
│   │   └── test_risk.py
│   ├── integration/                # Integration tests
│   └── fixtures/                   # Test data
│
├── scripts/                        # Utility scripts
│   ├── seed_db.py                  # Seed database
│   ├── benchmark_calcs.py          # Benchmark performance
│   └── data_migration.py           # Data migration scripts
│
├── logs/                           # Log files (gitignored)
├── data/                           # Local data (gitignored)
│   ├── portfolios/                 # Portfolio data
│   └── cache/                      # Cached price data
│
├── .env                            # Environment variables (gitignored)
├── .env.example                    # Example env file
├── requirements.txt                # Python dependencies
├── requirements-dev.txt            # Dev dependencies
├── pyproject.toml                  # Python project config (ruff, mypy)
├── alembic.ini                     # Alembic config
├── pytest.ini                      # Pytest config
├── run.py                          # Application entry point
└── README.md                       # Project documentation
```

### 2.3 Module Responsibility

**Core Modules** (Pure Python - No UI/Framework Dependencies):
- **MUST**: Contain only business logic
- **MUST**: Be testable without UI
- **MUST**: Have no Streamlit imports
- **MUST**: Return typed data structures (not UI components)

**Service Layer**:
- **MUST**: Orchestrate core modules
- **MUST**: Handle cross-cutting concerns (transactions, caching, logging)
- **MUST NOT**: Contain business logic (delegate to core)
- **MUST NOT**: Have UI imports

**Presentation Layer** (Streamlit):
- **MUST**: Only handle UI rendering
- **MUST**: Call services (not core modules directly)
- **MUST**: Validate user inputs
- **MUST NOT**: Contain business logic

---

## 3. CODE RULES

### 3.1 Type Hints (MANDATORY)

```python
# ✅ CORRECT - Always use type hints
from typing import List, Dict, Optional, Tuple
from datetime import date, datetime
import numpy as np
import pandas as pd

def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Trading periods per year (default: 252)
    
    Returns:
        Annualized Sharpe ratio
    
    Raises:
        ValueError: If returns series is empty or all zero
    """
    if returns.empty:
        raise ValueError("Returns series cannot be empty")
    
    excess_returns = returns - risk_free_rate / periods_per_year
    return float(
        np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
    )


# ❌ WRONG - No type hints
def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    return np.sqrt(252) * returns.mean() / returns.std()
```

### 3.2 Type Checking Configuration

**pyproject.toml**:
```toml
[tool.mypy]
python_version = "3.9"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_generics = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true

# Per-module options
[[tool.mypy.overrides]]
module = "yfinance.*"
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "streamlit.*"
ignore_missing_imports = true
```

### 3.3 Naming Conventions

```python
# ✅ CORRECT Naming

# Classes: PascalCase
class PortfolioManager:
    pass

class MeanVarianceOptimizer:
    pass


# Functions/Methods: snake_case
def calculate_total_return(start_value: float, end_value: float) -> float:
    pass

def fetch_price_history(ticker: str, start_date: date) -> pd.DataFrame:
    pass


# Constants: UPPER_SNAKE_CASE
TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.0435
MAX_PORTFOLIO_POSITIONS = 100


# Private attributes/methods: _leading_underscore
class Portfolio:
    def __init__(self):
        self._positions: List[Position] = []
        self._cache: Dict[str, Any] = {}
    
    def _calculate_weights(self) -> np.ndarray:
        pass


# Protected/Internal: __double_underscore (use sparingly)
class BaseOptimizer:
    def __init__(self):
        self.__solver_params = {}  # Name mangling for true privacy


# Variables: snake_case
portfolio_value = 100000.0
sharpe_ratio = 1.35
current_drawdown = -0.0234
```

### 3.4 Forbidden Practices

```python
# ❌ FORBIDDEN: Using 'any' type
def process_data(data: any) -> any:  # WRONG!
    pass

# ✅ CORRECT: Proper typing
from typing import Union, Dict, List
def process_data(data: Union[Dict, List]) -> pd.DataFrame:
    pass


# ❌ FORBIDDEN: Bare except
try:
    result = risky_operation()
except:  # WRONG!
    pass

# ✅ CORRECT: Specific exceptions
try:
    result = risky_operation()
except (ValueError, TypeError) as e:
    logger.error(f"Operation failed: {e}")
    raise


# ❌ FORBIDDEN: Mutable default arguments
def add_position(ticker: str, positions: List = []) -> List:  # WRONG!
    positions.append(ticker)
    return positions

# ✅ CORRECT: Use None and create new instance
def add_position(ticker: str, positions: Optional[List] = None) -> List:
    if positions is None:
        positions = []
    positions.append(ticker)
    return positions


# ❌ FORBIDDEN: Direct environment variable access
import os
api_key = os.getenv("API_KEY")  # WRONG in business logic!

# ✅ CORRECT: Use config module
from config.settings import settings
api_key = settings.api_key


# ❌ FORBIDDEN: String concatenation for paths
file_path = "data/" + folder + "/" + filename  # WRONG!

# ✅ CORRECT: Use pathlib
from pathlib import Path
file_path = Path("data") / folder / filename


# ❌ FORBIDDEN: Magic numbers
if portfolio_value > 100000:  # WRONG! What is 100000?
    pass

# ✅ CORRECT: Named constants
MINIMUM_PORTFOLIO_VALUE = 100000.0
if portfolio_value > MINIMUM_PORTFOLIO_VALUE:
    pass
```

### 3.5 Mandatory Practices

```python
# ✅ MANDATORY: Docstrings for all public functions/classes
def calculate_portfolio_metrics(
    portfolio: Portfolio,
    start_date: date,
    end_date: date,
    benchmark: Optional[str] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio metrics over a date range.
    
    This function computes 70+ performance, risk, and market metrics
    including returns, volatility, Sharpe ratio, VaR, and more.
    
    Args:
        portfolio: Portfolio object containing positions
        start_date: Start of analysis period (inclusive)
        end_date: End of analysis period (inclusive)
        benchmark: Optional benchmark ticker for comparison (e.g., "SPY")
    
    Returns:
        Dictionary containing all calculated metrics with keys:
        - "total_return": Total period return
        - "sharpe_ratio": Annualized Sharpe ratio
        - "max_drawdown": Maximum drawdown
        ... (70+ metrics)
    
    Raises:
        ValueError: If start_date >= end_date or portfolio is empty
        DataFetchError: If price data cannot be retrieved
    
    Examples:
        >>> portfolio = Portfolio.load("my_portfolio")
        >>> metrics = calculate_portfolio_metrics(
        ...     portfolio,
        ...     date(2024, 1, 1),
        ...     date(2025, 10, 29),
        ...     benchmark="SPY"
        ... )
        >>> print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        Sharpe Ratio: 1.35
    """
    # Implementation...
    pass


# ✅ MANDATORY: Logging for important operations
import logging
logger = logging.getLogger(__name__)

def optimize_portfolio(portfolio: Portfolio, method: str) -> OptimizationResult:
    logger.info(f"Starting optimization: method={method}, portfolio_id={portfolio.id}")
    
    try:
        result = optimizer.optimize()
        logger.info(f"Optimization complete: sharpe={result.sharpe:.2f}")
        return result
    except OptimizationError as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise


# ✅ MANDATORY: Input validation at boundaries (service layer)
from typing import TypedDict

class CreatePortfolioRequest(TypedDict):
    name: str
    description: str
    starting_capital: float

def create_portfolio(request: CreatePortfolioRequest) -> Portfolio:
    # Validate at service boundary
    if not request["name"] or len(request["name"]) > 100:
        raise ValidationError("Portfolio name must be 1-100 characters")
    
    if request["starting_capital"] <= 0:
        raise ValidationError("Starting capital must be positive")
    
    # Proceed with business logic
    return Portfolio(**request)


# ✅ MANDATORY: Context managers for resources
from contextlib import contextmanager

@contextmanager
def get_db_session():
    """Provide a transactional database session."""
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# Usage
with get_db_session() as session:
    portfolio = session.query(Portfolio).first()
    portfolio.name = "Updated Name"
```

---

## 4. DATA VALIDATION

### 4.1 Validation Libraries

**Primary**: Pydantic 2.0+ (for data validation and settings)

```python
from pydantic import BaseModel, Field, validator, field_validator
from typing import List, Optional
from datetime import date

# ✅ CORRECT: Use Pydantic for validation
class Position(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, pattern=r"^[A-Z0-9]+$")
    shares: float = Field(..., gt=0)
    weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    purchase_date: Optional[date] = None
    
    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        return v.upper()
    
    class Config:
        str_strip_whitespace = True
        validate_assignment = True


class CreatePortfolioRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    starting_capital: float = Field(..., gt=0)
    positions: List[Position] = Field(..., min_items=1, max_items=100)
    
    @field_validator("positions")
    @classmethod
    def validate_weights_sum(cls, positions: List[Position]) -> List[Position]:
        total_weight = sum(p.weight or 0 for p in positions)
        if abs(total_weight - 1.0) > 0.0001 and total_weight > 0:
            raise ValueError(f"Position weights must sum to 1.0, got {total_weight}")
        return positions


# Usage in service layer
def create_portfolio(data: dict) -> Portfolio:
    # Pydantic validates automatically
    request = CreatePortfolioRequest(**data)  # Raises ValidationError if invalid
    
    # Now we have validated data
    portfolio = Portfolio(
        name=request.name,
        starting_capital=request.starting_capital,
    )
    for pos in request.positions:
        portfolio.add_position(pos.ticker, pos.shares)
    
    return portfolio
```

### 4.2 Where to Validate

```python
# ✅ CORRECT: Validation hierarchy

# 1. UI Layer - Basic format validation (Streamlit)
import streamlit as st

ticker = st.text_input("Ticker")
if ticker and not ticker.isalnum():
    st.error("Ticker must be alphanumeric")

# 2. Service Layer - Business rule validation (before core)
class PortfolioService:
    def create_portfolio(self, data: dict) -> Portfolio:
        # Validate request structure
        request = CreatePortfolioRequest(**data)  # Pydantic
        
        # Validate business rules
        if self._portfolio_name_exists(request.name):
            raise ConflictError("Portfolio name already exists")
        
        # Validate tickers are real
        invalid_tickers = self.data_service.validate_tickers(
            [p.ticker for p in request.positions]
        )
        if invalid_tickers:
            raise ValidationError(f"Invalid tickers: {invalid_tickers}")
        
        # Call core module
        return self.portfolio_manager.create(request)

# 3. Core Layer - Domain invariants (internal consistency)
class Portfolio:
    def add_position(self, ticker: str, shares: float) -> None:
        # Ensure domain invariants
        if shares <= 0:
            raise ValueError("Shares must be positive")
        
        if self._position_exists(ticker):
            raise ValueError(f"Position {ticker} already exists")
        
        self._positions.append(Position(ticker, shares))
```

---

## 5. CONFIGURATION MANAGEMENT

### 5.1 Settings Module (Pydantic Settings)

**config/settings.py**:
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "Wild Market Capital"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Database
    database_url: str = "sqlite:///./data/wmc.db"
    database_echo: bool = False
    
    # Data Sources
    yahoo_finance_enabled: bool = True
    alpha_vantage_api_key: Optional[str] = None
    iex_cloud_api_key: Optional[str] = None
    
    # Cache
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    price_cache_dir: Path = Path("data/cache/prices")
    
    # Performance
    max_workers: int = 4
    calculation_timeout_seconds: int = 30
    
    # Risk-Free Rate (annual)
    risk_free_rate: float = 0.0435
    
    # Constraints
    max_portfolio_positions: int = 100
    min_position_weight: float = 0.0
    max_position_weight: float = 1.0
    
    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("logs/app.log")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.price_cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()
```

**.env.example**:
```bash
# Application
DEBUG=false

# Database
DATABASE_URL=sqlite:///./data/wmc.db
DATABASE_ECHO=false

# API Keys (optional)
ALPHA_VANTAGE_API_KEY=your_key_here
IEX_CLOUD_API_KEY=your_key_here

# Cache
CACHE_ENABLED=true
CACHE_TTL_SECONDS=3600

# Risk-Free Rate
RISK_FREE_RATE=0.0435

# Logging
LOG_LEVEL=INFO
```

### 5.2 Usage Rules

```python
# ✅ CORRECT: Import settings from config module
from config.settings import settings

def fetch_price_data(ticker: str) -> pd.DataFrame:
    api_key = settings.alpha_vantage_api_key
    if not api_key:
        logger.warning("Alpha Vantage API key not configured, using Yahoo Finance")
        return fetch_from_yahoo(ticker)
    
    return fetch_from_alpha_vantage(ticker, api_key)


# ❌ WRONG: Direct environment access in business logic
import os
def fetch_price_data(ticker: str) -> pd.DataFrame:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")  # WRONG!
    pass


# ❌ WRONG: Magic strings
def calculate_sharpe(returns: pd.Series) -> float:
    return returns.mean() / returns.std() * np.sqrt(252)  # WRONG! Magic 252

# ✅ CORRECT: Use constants from settings
def calculate_sharpe(returns: pd.Series) -> float:
    periods_per_year = 252  # Or from settings if configurable
    return returns.mean() / returns.std() * np.sqrt(periods_per_year)
```

---

## 6. ERROR HANDLING

### 6.1 Custom Exception Hierarchy

**core/exceptions.py**:
```python
class WMCBaseException(Exception):
    """Base exception for all WMC exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


# Data-related errors
class DataError(WMCBaseException):
    """Base class for data-related errors."""
    pass

class TickerNotFoundError(DataError):
    """Ticker symbol not found in data source."""
    pass

class DataFetchError(DataError):
    """Error fetching data from external source."""
    pass

class InsufficientDataError(DataError):
    """Not enough data points for calculation."""
    pass


# Validation errors
class ValidationError(WMCBaseException):
    """Input validation failed."""
    pass

class ConflictError(WMCBaseException):
    """Resource conflict (e.g., duplicate name)."""
    pass


# Calculation errors
class CalculationError(WMCBaseException):
    """Base class for calculation errors."""
    pass

class OptimizationError(CalculationError):
    """Optimization failed to converge."""
    pass

class NumericalError(CalculationError):
    """Numerical computation error (overflow, NaN, etc.)."""
    pass


# Portfolio errors
class PortfolioError(WMCBaseException):
    """Base class for portfolio-related errors."""
    pass

class PortfolioNotFoundError(PortfolioError):
    """Portfolio not found in database."""
    pass

class PositionNotFoundError(PortfolioError):
    """Position not found in portfolio."""
    pass
```

### 6.2 Error Handling Patterns

```python
# ✅ CORRECT: Specific exception handling with logging
import logging
logger = logging.getLogger(__name__)

def calculate_metrics(portfolio: Portfolio) -> Dict[str, float]:
    """Calculate portfolio metrics with proper error handling."""
    
    try:
        # Fetch price data
        prices = fetch_price_history(portfolio.tickers)
    except TickerNotFoundError as e:
        logger.error(f"Invalid ticker in portfolio: {e.message}")
        raise  # Re-raise for caller to handle
    except DataFetchError as e:
        logger.warning(f"Data fetch failed, using cached data: {e.message}")
        prices = get_cached_prices(portfolio.tickers)
    
    try:
        # Calculate metrics
        returns = calculate_returns(prices)
        metrics = {
            "sharpe_ratio": calculate_sharpe(returns),
            "max_drawdown": calculate_max_drawdown(returns),
        }
    except InsufficientDataError as e:
        logger.error(f"Not enough data for calculations: {e.message}")
        raise ValidationError("Portfolio requires at least 30 days of price data")
    except NumericalError as e:
        logger.error(f"Numerical error in calculation: {e.message}", exc_info=True)
        raise CalculationError(f"Failed to calculate metrics: {e.message}")
    
    return metrics


# ✅ CORRECT: Contextual error information
def optimize_portfolio(portfolio: Portfolio, method: str) -> OptimizationResult:
    try:
        optimizer = get_optimizer(method)
        result = optimizer.optimize(portfolio)
    except OptimizationError as e:
        # Add context before re-raising
        raise OptimizationError(
            f"Optimization failed for method '{method}'",
            details={
                "portfolio_id": portfolio.id,
                "method": method,
                "position_count": len(portfolio.positions),
                "original_error": str(e)
            }
        ) from e
    
    return result


# ❌ WRONG: Swallowing exceptions
try:
    result = risky_operation()
except Exception:
    pass  # WRONG! Never do this


# ❌ WRONG: Catching too broad
try:
    result = operation()
except Exception as e:  # WRONG! Too broad
    logger.error("Something went wrong")


# ✅ CORRECT: Catch specific exceptions, let others propagate
try:
    result = operation()
except (ValueError, TypeError) as e:
    logger.error(f"Invalid input: {e}")
    raise ValidationError(f"Input validation failed: {e}") from e
# Other exceptions (KeyError, AttributeError, etc.) propagate naturally
```

### 6.3 Logging Standards

**config/logging_config.py**:
```python
import logging
import logging.handlers
from pathlib import Path
from config.settings import settings

def setup_logging() -> None:
    """Configure application logging."""
    
    # Create logs directory
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)
    
    # Format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (rotating)
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
```

**Usage**:
```python
import logging
logger = logging.getLogger(__name__)  # Always use __name__

# ✅ CORRECT: Structured logging
logger.info(
    "Portfolio created",
    extra={
        "portfolio_id": portfolio.id,
        "position_count": len(portfolio.positions),
        "starting_capital": portfolio.starting_capital
    }
)

logger.debug(f"Fetching prices for {len(tickers)} tickers")
logger.warning(f"API rate limit approaching: {remaining}/{limit}")
logger.error(f"Optimization failed: {error}", exc_info=True)


# ❌ WRONG: Print statements in production code
print("Portfolio created")  # WRONG! Use logger


# ❌ WRONG: String concatenation in logs (inefficient)
logger.debug("Processing " + str(len(items)) + " items")  # WRONG!

# ✅ CORRECT: f-strings or lazy formatting
logger.debug(f"Processing {len(items)} items")
```

---

## 7. DATABASE

### 7.1 ORM Models (SQLAlchemy 2.0)

**models/portfolio.py**:
```python
from sqlalchemy import Column, String, Float, DateTime, Integer, ForeignKey, Index
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from datetime import datetime
from typing import List
import uuid

from database.session import Base

# ✅ CORRECT: Use SQLAlchemy 2.0 style with type hints

class Portfolio(Base):
    __tablename__ = "portfolios"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    
    # Attributes
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    description: Mapped[str] = mapped_column(String(500), nullable=True)
    starting_capital: Mapped[float] = mapped_column(Float, nullable=False)
    base_currency: Mapped[str] = mapped_column(String(3), default="USD")
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationships
    positions: Mapped[List["Position"]] = relationship(
        "Position",
        back_populates="portfolio",
        cascade="all, delete-orphan",
        lazy="selectin"  # Eager load positions
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_portfolio_name", "name"),
        Index("idx_portfolio_created", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<Portfolio(id={self.id}, name={self.name})>"


class Position(Base):
    __tablename__ = "positions"
    
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    
    portfolio_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False
    )
    
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    shares: Mapped[float] = mapped_column(Float, nullable=False)
    weight_target: Mapped[float] = mapped_column(Float, nullable=True)
    purchase_price: Mapped[float] = mapped_column(Float, nullable=True)
    purchase_date: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    portfolio: Mapped["Portfolio"] = relationship(
        "Portfolio",
        back_populates="positions"
    )
    
    __table_args__ = (
        Index("idx_position_portfolio", "portfolio_id"),
        Index("idx_position_ticker", "ticker"),
        Index("idx_position_portfolio_ticker", "portfolio_id", "ticker", unique=True),
    )
    
    def __repr__(self) -> str:
        return f"<Position(ticker={self.ticker}, shares={self.shares})>"
```

### 7.2 Session Management

**database/session.py**:
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session
from contextlib import contextmanager
from typing import Generator
from config.settings import settings

# Base class for models
class Base(DeclarativeBase):
    pass

# Engine
engine = create_engine(
    settings.database_url,
    echo=settings.database_echo,
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,   # Recycle connections after 1 hour
)

# Session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)


# ✅ CORRECT: Context manager for sessions
@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Provide a transactional database session.
    
    Usage:
        with get_db_session() as session:
            portfolio = session.query(Portfolio).first()
            portfolio.name = "Updated"
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Initialize database (create tables)
def init_db() -> None:
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)
```

### 7.3 Database Operations

```python
# ✅ CORRECT: Using session context manager
from database.session import get_db_session
from models.portfolio import Portfolio

def get_portfolio_by_id(portfolio_id: str) -> Optional[Portfolio]:
    """Fetch portfolio by ID."""
    with get_db_session() as session:
        portfolio = session.query(Portfolio).filter(
            Portfolio.id == portfolio_id
        ).first()
        return portfolio  # Safe: expire_on_commit=False


def create_portfolio(name: str, starting_capital: float) -> Portfolio:
    """Create new portfolio."""
    with get_db_session() as session:
        portfolio = Portfolio(
            name=name,
            starting_capital=starting_capital
        )
        session.add(portfolio)
        session.flush()  # Get ID before commit
        portfolio_id = portfolio.id
    
    # Fetch again to return fresh instance
    return get_portfolio_by_id(portfolio_id)


# ✅ CORRECT: Bulk operations
def get_all_portfolios(limit: int = 100, offset: int = 0) -> List[Portfolio]:
    """Fetch all portfolios with pagination."""
    with get_db_session() as session:
        portfolios = session.query(Portfolio).limit(limit).offset(offset).all()
        return portfolios


# ✅ CORRECT: Updates
def update_portfolio_name(portfolio_id: str, new_name: str) -> Portfolio:
    """Update portfolio name."""
    with get_db_session() as session:
        portfolio = session.query(Portfolio).filter(
            Portfolio.id == portfolio_id
        ).first()
        
        if not portfolio:
            raise PortfolioNotFoundError(f"Portfolio {portfolio_id} not found")
        
        portfolio.name = new_name
        # Commit happens automatically in context manager
    
    return get_portfolio_by_id(portfolio_id)


# ❌ WRONG: Not using context manager
def bad_create_portfolio(name: str) -> Portfolio:
    session = SessionLocal()
    portfolio = Portfolio(name=name)
    session.add(portfolio)
    session.commit()
    # WRONG! Session not closed, potential leak
    return portfolio
```

### 7.4 Migrations (Alembic)

**Usage**:
```bash
# Initialize Alembic (already done)
alembic init database/migrations

# Create migration after model changes
alembic revision --autogenerate -m "Add tags column to portfolios"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

**Migration file example**:
```python
"""Add tags column to portfolios

Revision ID: abc123
"""
from alembic import op
import sqlalchemy as sa

def upgrade() -> None:
    op.add_column(
        'portfolios',
        sa.Column('tags', sa.JSON(), nullable=True)
    )

def downgrade() -> None:
    op.drop_column('portfolios', 'tags')
```

---

## 8. SOLID PRINCIPLES

### 8.1 Single Responsibility Principle

```python
# ❌ WRONG: Class doing too much
class Portfolio:
    def __init__(self):
        self.positions = []
    
    def add_position(self, ticker, shares):
        self.positions.append((ticker, shares))
    
    def fetch_prices(self):  # WRONG! Not portfolio's responsibility
        # Fetch from Yahoo Finance
        pass
    
    def calculate_metrics(self):  # WRONG! Calculation should be separate
        # Calculate Sharpe, Sortino, etc.
        pass
    
    def save_to_database(self):  # WRONG! Persistence should be separate
        # Save to DB
        pass


# ✅ CORRECT: Separate responsibilities
class Portfolio:
    """Domain model - only holds state and domain logic."""
    def __init__(self, name: str, starting_capital: float):
        self.name = name
        self.starting_capital = starting_capital
        self._positions: List[Position] = []
    
    def add_position(self, position: Position) -> None:
        """Add position (domain logic)."""
        if self._position_exists(position.ticker):
            raise ValueError(f"Position {position.ticker} already exists")
        self._positions.append(position)
    
    def _position_exists(self, ticker: str) -> bool:
        return any(p.ticker == ticker for p in self._positions)


class PriceDataService:
    """Separate service for price fetching."""
    def fetch_prices(self, tickers: List[str]) -> pd.DataFrame:
        # Fetch from data source
        pass


class AnalyticsEngine:
    """Separate engine for calculations."""
    def calculate_metrics(self, portfolio: Portfolio) -> Dict[str, float]:
        # Calculate metrics
        pass


class PortfolioRepository:
    """Separate repository for persistence."""
    def save(self, portfolio: Portfolio) -> None:
        # Save to database
        pass
```

### 8.2 Open/Closed Principle

```python
# ✅ CORRECT: Open for extension, closed for modification

from abc import ABC, abstractmethod
from typing import Dict

class BaseOptimizer(ABC):
    """Abstract base class for all optimizers."""
    
    @abstractmethod
    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: Dict
    ) -> OptimizationResult:
        """Optimize portfolio weights."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return optimizer name."""
        pass


class MeanVarianceOptimizer(BaseOptimizer):
    """Markowitz mean-variance optimization."""
    
    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: Dict
    ) -> OptimizationResult:
        # Implementation
        pass
    
    def get_name(self) -> str:
        return "Mean-Variance"


class RiskParityOptimizer(BaseOptimizer):
    """Risk parity optimization."""
    
    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: Dict
    ) -> OptimizationResult:
        # Different implementation
        pass
    
    def get_name(self) -> str:
        return "Risk Parity"


# Factory pattern for extensibility
class OptimizerFactory:
    _optimizers: Dict[str, type[BaseOptimizer]] = {
        "mean_variance": MeanVarianceOptimizer,
        "risk_parity": RiskParityOptimizer,
    }
    
    @classmethod
    def register(cls, name: str, optimizer_class: type[BaseOptimizer]) -> None:
        """Register new optimizer (extension point)."""
        cls._optimizers[name] = optimizer_class
    
    @classmethod
    def create(cls, name: str) -> BaseOptimizer:
        """Create optimizer instance."""
        if name not in cls._optimizers:
            raise ValueError(f"Unknown optimizer: {name}")
        return cls._optimizers[name]()

# Adding new optimizer without modifying existing code:
class HRPOptimizer(BaseOptimizer):
    def optimize(self, returns, constraints):
        # HRP implementation
        pass
    
    def get_name(self) -> str:
        return "Hierarchical Risk Parity"

# Register the new optimizer
OptimizerFactory.register("hrp", HRPOptimizer)
```

### 8.3 Liskov Substitution Principle

```python
# ✅ CORRECT: Subtypes must be substitutable

class MetricCalculator(ABC):
    """Base class for metric calculation."""
    
    @abstractmethod
    def calculate(self, returns: pd.Series) -> float:
        """Calculate metric. Must return float or raise exception."""
        pass


class SharpeRatioCalculator(MetricCalculator):
    def calculate(self, returns: pd.Series) -> float:
        if returns.empty:
            raise ValueError("Returns series is empty")
        return float(returns.mean() / returns.std() * np.sqrt(252))


class SortinoRatioCalculator(MetricCalculator):
    def calculate(self, returns: pd.Series) -> float:
        if returns.empty:
            raise ValueError("Returns series is empty")
        downside = returns[returns < 0].std()
        return float(returns.mean() / downside * np.sqrt(252))


# Both can be used interchangeably
def calculate_metric(calculator: MetricCalculator, returns: pd.Series) -> float:
    """Works with any MetricCalculator subtype."""
    return calculator.calculate(returns)

# Usage - both work the same
sharpe = calculate_metric(SharpeRatioCalculator(), returns)
sortino = calculate_metric(SortinoRatioCalculator(), returns)
```

### 8.4 Interface Segregation Principle

```python
# ❌ WRONG: Fat interface
class DataProvider(ABC):
    @abstractmethod
    def fetch_prices(self, ticker: str) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def fetch_fundamentals(self, ticker: str) -> Dict:
        pass
    
    @abstractmethod
    def fetch_options(self, ticker: str) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def fetch_news(self, ticker: str) -> List[Dict]:
        pass

# Some providers can't implement all methods!


# ✅ CORRECT: Segregated interfaces
class PriceProvider(ABC):
    @abstractmethod
    def fetch_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        pass


class FundamentalsProvider(ABC):
    @abstractmethod
    def fetch_fundamentals(self, ticker: str) -> Dict:
        pass


class OptionsProvider(ABC):
    @abstractmethod
    def fetch_options(self, ticker: str) -> pd.DataFrame:
        pass


# Providers implement only what they support
class YahooFinanceProvider(PriceProvider, FundamentalsProvider):
    """Yahoo Finance provides prices and fundamentals."""
    
    def fetch_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        # Implementation
        pass
    
    def fetch_fundamentals(self, ticker: str) -> Dict:
        # Implementation
        pass


class AlphaVantageProvider(PriceProvider):
    """Alpha Vantage only provides prices (free tier)."""
    
    def fetch_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        # Implementation
        pass
```

### 8.5 Dependency Inversion Principle

```python
# ❌ WRONG: High-level depends on low-level concrete class
class PortfolioService:
    def __init__(self):
        self.data_source = YahooFinanceAPI()  # WRONG! Concrete dependency
    
    def get_portfolio_value(self, portfolio):
        prices = self.data_source.fetch_prices(portfolio.tickers)
        return calculate_value(portfolio, prices)


# ✅ CORRECT: Both depend on abstraction
class PriceDataSource(ABC):
    """Abstract interface for price data."""
    
    @abstractmethod
    def fetch_prices(
        self,
        tickers: List[str],
        start: date,
        end: date
    ) -> pd.DataFrame:
        pass


class PortfolioService:
    """High-level service depends on abstraction."""
    
    def __init__(self, price_source: PriceDataSource):
        self.price_source = price_source  # Injected dependency
    
    def get_portfolio_value(
        self,
        portfolio: Portfolio,
        as_of_date: date
    ) -> float:
        prices = self.price_source.fetch_prices(
            portfolio.tickers,
            as_of_date,
            as_of_date
        )
        return self._calculate_value(portfolio, prices)


class YahooFinanceSource(PriceDataSource):
    """Low-level implementation."""
    
    def fetch_prices(
        self,
        tickers: List[str],
        start: date,
        end: date
    ) -> pd.DataFrame:
        import yfinance as yf
        data = yf.download(tickers, start=start, end=end)
        return data


# Usage with dependency injection
price_source = YahooFinanceSource()
portfolio_service = PortfolioService(price_source)

# Easy to swap implementations
alternative_source = AlphaVantageSource()
portfolio_service = PortfolioService(alternative_source)
```

---

## 9. TESTING

### 9.1 Test Structure

```
tests/
├── unit/                          # Unit tests (isolated)
│   ├── test_analytics.py
│   ├── test_optimization.py
│   └── test_risk.py
├── integration/                   # Integration tests (with DB/APIs)
│   ├── test_portfolio_service.py
│   └── test_data_fetching.py
├── fixtures/                      # Test data
│   ├── sample_portfolios.json
│   └── mock_price_data.csv
└── conftest.py                    # Pytest fixtures
```

### 9.2 Unit Test Examples

**tests/unit/test_analytics.py**:
```python
import pytest
import pandas as pd
import numpy as np
from datetime import date
from core.analytics_engine.performance import calculate_sharpe_ratio, calculate_total_return

class TestPerformanceMetrics:
    """Test performance calculation functions."""
    
    def test_sharpe_ratio_positive_returns(self):
        """Test Sharpe ratio calculation with positive returns."""
        # Arrange
        returns = pd.Series([0.01, 0.02, 0.015, 0.018, 0.012])
        risk_free_rate = 0.02  # 2% annual
        
        # Act
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
        
        # Assert
        assert isinstance(sharpe, float)
        assert sharpe > 0
        assert np.isfinite(sharpe)
    
    def test_sharpe_ratio_empty_series(self):
        """Test Sharpe ratio raises error with empty series."""
        # Arrange
        returns = pd.Series([], dtype=float)
        
        # Act & Assert
        with pytest.raises(ValueError, match="empty"):
            calculate_sharpe_ratio(returns)
    
    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility (constant returns)."""
        # Arrange
        returns = pd.Series([0.01] * 100)  # Constant returns
        
        # Act & Assert
        with pytest.raises(ZeroDivisionError):
            calculate_sharpe_ratio(returns)
    
    def test_total_return_calculation(self):
        """Test total return calculation."""
        # Arrange
        start_value = 100000.0
        end_value = 125000.0
        expected_return = 0.25  # 25%
        
        # Act
        total_return = calculate_total_return(start_value, end_value)
        
        # Assert
        assert total_return == pytest.approx(expected_return, rel=1e-6)


# ✅ CORRECT: Use fixtures for common test data
@pytest.fixture
def sample_returns():
    """Provide sample returns series for testing."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.001, 0.02, 252))


def test_with_fixture(sample_returns):
    """Test using fixture."""
    sharpe = calculate_sharpe_ratio(sample_returns)
    assert isinstance(sharpe, float)
```

### 9.3 Integration Test Examples

**tests/integration/test_portfolio_service.py**:
```python
import pytest
from services.portfolio_service import PortfolioService
from database.session import get_db_session, init_db
from models.portfolio import Portfolio

@pytest.fixture(scope="module")
def test_database():
    """Set up test database."""
    # Use in-memory SQLite for tests
    from config.settings import settings
    settings.database_url = "sqlite:///:memory:"
    init_db()
    yield
    # Cleanup happens automatically with in-memory DB


@pytest.fixture
def portfolio_service(test_database):
    """Provide portfolio service instance."""
    return PortfolioService()


class TestPortfolioService:
    """Integration tests for PortfolioService."""
    
    def test_create_and_retrieve_portfolio(self, portfolio_service):
        """Test creating and retrieving a portfolio."""
        # Arrange
        portfolio_data = {
            "name": "Test Portfolio",
            "description": "Integration test portfolio",
            "starting_capital": 100000.0,
            "positions": [
                {"ticker": "AAPL", "shares": 100},
                {"ticker": "MSFT", "shares": 50}
            ]
        }
        
        # Act
        created = portfolio_service.create_portfolio(portfolio_data)
        retrieved = portfolio_service.get_portfolio(created.id)
        
        # Assert
        assert retrieved is not None
        assert retrieved.name == "Test Portfolio"
        assert len(retrieved.positions) == 2
    
    def test_duplicate_portfolio_name_raises_error(self, portfolio_service):
        """Test that duplicate portfolio name raises ConflictError."""
        # Arrange
        data = {"name": "Duplicate", "starting_capital": 50000.0, "positions": []}
        portfolio_service.create_portfolio(data)
        
        # Act & Assert
        with pytest.raises(ConflictError, match="already exists"):
            portfolio_service.create_portfolio(data)
```

### 9.4 Test Coverage

**pytest.ini**:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --cov=core
    --cov=services
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
```

**Run tests**:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test file
pytest tests/unit/test_analytics.py

# Run specific test
pytest tests/unit/test_analytics.py::TestPerformanceMetrics::test_sharpe_ratio_positive_returns

# Run with markers
pytest -m "not slow"
```

---

## 10. SECURITY

### 10.1 Password Hashing (Future Phase)

```python
from passlib.context import CryptContext

# ✅ CORRECT: Use bcrypt with appropriate cost factor
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)

def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


# ❌ WRONG: Plain text passwords
def store_password(password: str):
    # NEVER store plain text!
    pass
```

### 10.2 Input Sanitization

```python
# ✅ CORRECT: Sanitize and validate all inputs
from pydantic import BaseModel, Field, field_validator
import re

class TickerInput(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    
    @field_validator("ticker")
    @classmethod
    def sanitize_ticker(cls, v: str) -> str:
        # Strip whitespace
        v = v.strip().upper()
        
        # Only allow alphanumeric
        if not re.match(r"^[A-Z0-9]+$", v):
            raise ValueError("Ticker must be alphanumeric")
        
        return v


# ✅ CORRECT: Parameterized queries (SQLAlchemy does this automatically)
def get_portfolio_by_name(name: str) -> Optional[Portfolio]:
    with get_db_session() as session:
        # SQLAlchemy uses parameterized queries - safe from SQL injection
        portfolio = session.query(Portfolio).filter(
            Portfolio.name == name  # Automatically parameterized
        ).first()
        return portfolio


# ❌ WRONG: String concatenation in queries (vulnerable to SQL injection)
def bad_get_portfolio(name: str):
    query = f"SELECT * FROM portfolios WHERE name = '{name}'"  # DANGEROUS!
    # Never do this!
```

### 10.3 API Key Management

```python
# ✅ CORRECT: Store API keys in settings, encrypt in database
from cryptography.fernet import Fernet
from config.settings import settings

class APIKeyManager:
    def __init__(self):
        # Encryption key from environment (not in code!)
        self.cipher = Fernet(settings.encryption_key.encode())
    
    def encrypt_api_key(self, api_key: str) -> bytes:
        """Encrypt API key for storage."""
        return self.cipher.encrypt(api_key.encode())
    
    def decrypt_api_key(self, encrypted_key: bytes) -> str:
        """Decrypt API key from storage."""
        return self.cipher.decrypt(encrypted_key).decode()


# ❌ WRONG: API keys in code
API_KEY = "abc123def456"  # NEVER!

# ❌ WRONG: API keys in version control
# .env file should be in .gitignore!
```

---

## 11. ANTI-PATTERNS (DON'T DO THIS)

### 11.1 God Object

```python
# ❌ ANTI-PATTERN: God object that does everything
class Portfolio:
    def __init__(self):
        self.positions = []
    
    def add_position(self): pass
    def fetch_prices(self): pass
    def calculate_all_metrics(self): pass
    def optimize(self): pass
    def generate_report(self): pass
    def save_to_db(self): pass
    def send_email(self): pass
    # ... 50 more methods

# ✅ SOLUTION: Single Responsibility - split into focused classes
```

### 11.2 Cargo Cult Programming

```python
# ❌ ANTI-PATTERN: Copy-paste without understanding
def calculate_metric(data):
    # Copied from Stack Overflow, not sure what it does
    result = data.rolling(window=20).apply(lambda x: x.mean() / x.std()).fillna(0)
    return result * 1.5  # Magic number from internet

# ✅ SOLUTION: Understand what you're implementing
def calculate_rolling_sharpe(
    returns: pd.Series,
    window: int = 20,
    annualization_factor: float = 1.0
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.
    
    Sharpe = (Mean Return) / (Std Dev of Returns)
    
    Args:
        returns: Series of periodic returns
        window: Rolling window size (default: 20 periods)
        annualization_factor: Factor to annualize (e.g., sqrt(252) for daily)
    
    Returns:
        Series of rolling Sharpe ratios
    """
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    sharpe = (rolling_mean / rolling_std) * annualization_factor
    return sharpe
```

### 11.3 Premature Optimization

```python
# ❌ ANTI-PATTERN: Optimizing before it's a problem
def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    # Complex caching, multiprocessing for 10 rows
    with multiprocessing.Pool() as pool:
        results = pool.map(lambda x: x.pct_change(), prices.values)
    return pd.DataFrame(results)

# ✅ SOLUTION: Start simple, optimize if needed
def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple returns."""
    return prices.pct_change()

# Optimize LATER if profiling shows this is a bottleneck
```

### 11.4 Magic Numbers

```python
# ❌ ANTI-PATTERN: Magic numbers everywhere
def calculate_sharpe(returns):
    return returns.mean() / returns.std() * 15.8745  # ???

def is_acceptable_volatility(vol):
    return vol < 0.25  # Why 0.25?

# ✅ SOLUTION: Named constants
TRADING_DAYS_PER_YEAR = 252
ANNUALIZATION_FACTOR = np.sqrt(TRADING_DAYS_PER_YEAR)  # ~15.8745
MAX_ACCEPTABLE_VOLATILITY = 0.25  # 25% annual volatility

def calculate_sharpe(returns: pd.Series) -> float:
    return returns.mean() / returns.std() * ANNUALIZATION_FACTOR

def is_acceptable_volatility(volatility: float) -> bool:
    return volatility < MAX_ACCEPTABLE_VOLATILITY
```

### 11.5 Leaky Abstractions

```python
# ❌ ANTI-PATTERN: Exposing implementation details
class Portfolio:
    def __init__(self):
        self._positions_dict = {}  # Internal implementation
    
    def get_positions_dict(self):
        return self._positions_dict  # LEAKING internal structure!

# Client code now depends on dict implementation
portfolio = Portfolio()
positions = portfolio.get_positions_dict()
positions["AAPL"] = {"shares": 100}  # Client modifies internal state!


# ✅ SOLUTION: Proper encapsulation
class Portfolio:
    def __init__(self):
        self._positions: List[Position] = []  # Private
    
    def get_positions(self) -> List[Position]:
        """Return copy of positions."""
        return list(self._positions)
    
    def add_position(self, position: Position) -> None:
        """Controlled way to modify portfolio."""
        self._positions.append(position)
```

### 11.6 Shotgun Surgery

```python
# ❌ ANTI-PATTERN: Changing one thing requires changes everywhere
# If you change how returns are calculated, you must change:
# - analytics_engine/performance.py (3 places)
# - analytics_engine/risk_metrics.py (5 places)
# - optimization_engine/mean_variance.py (2 places)
# - etc... (10 more files)

# ✅ SOLUTION: Centralize common logic
class ReturnsCalculator:
    """Single place for all return calculations."""
    
    @staticmethod
    def simple_returns(prices: pd.Series) -> pd.Series:
        return prices.pct_change()
    
    @staticmethod
    def log_returns(prices: pd.Series) -> pd.Series:
        return np.log(prices / prices.shift(1))
    
    @staticmethod
    def cumulative_returns(returns: pd.Series) -> pd.Series:
        return (1 + returns).cumprod() - 1

# Now everyone uses ReturnsCalculator - change once, applies everywhere
```

---

## 12. PERFORMANCE GUIDELINES

### 12.1 Efficient Data Operations

```python
# ❌ SLOW: Iterating over DataFrame rows
def calculate_returns_slow(prices: pd.DataFrame) -> pd.DataFrame:
    returns = pd.DataFrame()
    for i in range(1, len(prices)):
        returns.loc[i] = (prices.iloc[i] - prices.iloc[i-1]) / prices.iloc[i-1]
    return returns

# ✅ FAST: Vectorized operations
def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change()


# ❌ SLOW: Python loops for numerical computation
def calculate_max_drawdown_slow(returns: pd.Series) -> float:
    cumulative = 1.0
    peak = 1.0
    max_dd = 0.0
    for ret in returns:
        cumulative *= (1 + ret)
        if cumulative > peak:
            peak = cumulative
        dd = (cumulative - peak) / peak
        if dd < max_dd:
            max_dd = dd
    return max_dd

# ✅ FAST: NumPy vectorized
def calculate_max_drawdown(returns: pd.Series) -> float:
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - running_max) / running_max
    return float(drawdowns.min())
```

### 12.2 Caching

```python
from functools import lru_cache
from typing import Tuple

# ✅ CORRECT: Cache expensive computations
@lru_cache(maxsize=128)
def get_price_data(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch price data with caching."""
    # Expensive API call
    data = fetch_from_api(ticker, start_date, end_date)
    return data


# ✅ CORRECT: Cache at service layer
class PriceDataService:
    def __init__(self):
        self._cache: Dict[Tuple[str, date, date], pd.DataFrame] = {}
        self._cache_ttl = 3600  # 1 hour
    
    def get_prices(
        self,
        ticker: str,
        start: date,
        end: date
    ) -> pd.DataFrame:
        cache_key = (ticker, start, end)
        
        if cache_key in self._cache:
            logger.debug(f"Cache hit for {ticker}")
            return self._cache[cache_key]
        
        logger.debug(f"Cache miss for {ticker}, fetching...")
        data = self._fetch_from_source(ticker, start, end)
        self._cache[cache_key] = data
        return data
```

---

## SUMMARY: GOLDEN RULES

1. **Type Everything**: All functions must have type hints
2. **Single Responsibility**: One class/function = one job
3. **Validate at Boundaries**: Service layer validates, core assumes valid
4. **No Magic**: Constants for all numbers, clear names
5. **Exceptions for Errors**: Don't return None/empty on error
6. **Log Important Events**: INFO for business events, ERROR for failures
7. **Test Core Logic**: 80%+ coverage for business logic
8. **Immutable Data**: Prefer returning new objects over mutation
9. **Context Managers**: Use `with` for resources (DB, files)
10. **Document Public APIs**: Docstrings for all public functions/classes

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-29  
**Status**: Active - Must be followed for all code

---

**END OF ARCHITECTURE RULES**

