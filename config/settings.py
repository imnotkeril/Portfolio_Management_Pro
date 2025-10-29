"""Application settings loaded from environment variables."""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


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
        extra="ignore",
    )

    def __init__(self, **kwargs) -> None:
        """Initialize settings and create necessary directories."""
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.price_cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()

