"""Application constants."""

# Cache TTL values (in seconds)
CACHE_TTL_HISTORICAL_PRICES = 86400  # 24 hours
CACHE_TTL_CURRENT_PRICE = 300  # 5 minutes
CACHE_TTL_METRICS = 3600  # 1 hour
CACHE_TTL_TICKER_VALIDATION = 86400  # 24 hours

# Cache size limits
CACHE_MAX_SIZE = 1000  # Maximum number of items in cache

# Performance settings
MAX_WORKERS_PARALLEL_FETCH = 5  # Number of parallel threads for data fetching
MAX_RETRIES_API = 3  # Maximum retry attempts for API calls
INITIAL_RETRY_DELAY = 1.0  # Initial retry delay in seconds
RETRY_BACKOFF_MULTIPLIER = 2.0  # Exponential backoff multiplier

# Trading days
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_QUARTER = 63

# Default risk-free rate (annual)
DEFAULT_RISK_FREE_RATE = 0.0435  # 4.35%

# Portfolio constraints
MAX_PORTFOLIO_POSITIONS = 100
MIN_POSITION_WEIGHT = 0.0
MAX_POSITION_WEIGHT = 1.0

# Ticker validation
MAX_TICKER_LENGTH = 10
MIN_TICKER_LENGTH = 1

# Date ranges
MAX_ANALYSIS_PERIOD_YEARS = 20
MIN_ANALYSIS_PERIOD_DAYS = 30

