"""Application settings loaded from environment variables."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Portfolio Management Terminal"
    app_version: str = "1.0.0"
    debug: bool = False

    # Auth (Phase 2)
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24
    auth_disabled: bool = False

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

    # Stripe billing (Phase 5)
    stripe_secret_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None
    stripe_price_id_pro: Optional[str] = None
    frontend_url: str = "http://localhost:3000"
    billing_enforcement: bool = True

    @property
    def stripe_configured(self) -> bool:
        """True when Checkout / Portal can be created."""
        key = self.stripe_secret_key or ""
        price = self.stripe_price_id_pro or ""
        return bool(
            key.startswith("sk_")
            and len(key) > 20
            and "..." not in key
            and (price.startswith("price_") or price.startswith("prod_"))
        )

    @property
    def stripe_webhook_configured(self) -> bool:
        secret = self.stripe_webhook_secret or ""
        key = self.stripe_secret_key or ""
        return bool(
            key.startswith("sk_")
            and secret.startswith("whsec_")
            and "..." not in secret
            and len(secret) > 12
        )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def __init__(self, **kwargs) -> None:
        """Initialize settings and create necessary directories."""
        env_path = Path(__file__).resolve().parent.parent / ".env"
        preserved_db_url = os.environ.get("DATABASE_URL")
        if env_path.is_file():
            # override=True so .env wins over Windows placeholder STRIPE_* in system env
            load_dotenv(env_path, override=True)
        if preserved_db_url:
            # Docker / CI set DATABASE_URL before startup — do not replace with .env localhost
            os.environ["DATABASE_URL"] = preserved_db_url
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.price_cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()
