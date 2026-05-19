"""
Pydantic validation schemas for portfolio operations.

These models are shared by the FastAPI layer and Streamlit flows where payloads
must match domain constraints (tickers, weights summing to one, etc.).
"""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator

REBALANCE_INTERVALS = frozenset({1, 3, 6, 12})


class PositionSchema(BaseModel):
    """Position schema for validation."""

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        pattern=r"^[A-Z0-9-]+$",
        description="Ticker symbol (uppercase, alphanumeric, may contain hyphens)",
    )
    shares: Optional[float] = Field(
        None, gt=0, description="Number of shares (must be > 0 if provided)"
    )
    weight_target: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Target weight (0.0 to 1.0)"
    )
    purchase_price: Optional[float] = Field(
        None, gt=0, description="Purchase price per share"
    )
    purchase_date: Optional[date] = Field(None, description="Purchase date")

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return v.strip().upper()

    class Config:
        str_strip_whitespace = True
        validate_assignment = True


class CreatePortfolioRequest(BaseModel):
    """Request schema for creating a portfolio."""

    name: str = Field(..., min_length=1, max_length=100, description="Portfolio name")
    description: Optional[str] = Field(
        None, max_length=500, description="Portfolio description"
    )
    starting_capital: float = Field(..., gt=0, description="Starting capital amount")
    base_currency: str = Field(
        "USD", min_length=3, max_length=3, description="Base currency"
    )
    positions: list[PositionSchema] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of positions",
    )
    rebalance_interval_months: Optional[int] = Field(
        None,
        description="Rebalance to target weights every N months (1, 3, 6, 12); null = off",
    )
    ledger_mode: str = Field(
        "buy_hold",
        description="buy_hold = positions only; transactions = full ledger",
    )

    @field_validator("ledger_mode")
    @classmethod
    def validate_ledger_mode(cls, v: str) -> str:
        mode = (v or "buy_hold").strip().lower()
        if mode not in ("buy_hold", "transactions"):
            raise ValueError("ledger_mode must be 'buy_hold' or 'transactions'")
        return mode

    @field_validator("rebalance_interval_months")
    @classmethod
    def validate_rebalance_interval(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if v not in REBALANCE_INTERVALS:
            raise ValueError(
                "rebalance_interval_months must be 1, 3, 6, 12, or omitted"
            )
        return v

    @field_validator("name")
    @classmethod
    def name_stripped(cls, v: str) -> str:
        """Strip whitespace from name."""
        return v.strip()

    @field_validator("base_currency")
    @classmethod
    def currency_uppercase(cls, v: str) -> str:
        """Normalize currency to uppercase."""
        return v.strip().upper()

    @field_validator("positions")
    @classmethod
    def validate_no_duplicate_tickers(
        cls, positions: list[PositionSchema]
    ) -> list[PositionSchema]:
        """Validate no duplicate tickers."""
        tickers = [p.ticker for p in positions]
        if len(tickers) != len(set(tickers)):
            raise ValueError("Duplicate tickers are not allowed")
        return positions

    @field_validator("positions")
    @classmethod
    def validate_weights_sum(
        cls, positions: list[PositionSchema]
    ) -> list[PositionSchema]:
        """
        Validate that weights sum to 1.0 (±0.0001 tolerance).

        If no weights provided, validation passes.
        If weights provided, they must sum to 1.0.
        """
        weights = [p.weight_target for p in positions if p.weight_target]
        if not weights:
            # No weights specified, validation passes
            return positions

        total_weight = sum(weights)
        tolerance = 0.0001
        if abs(total_weight - 1.0) > tolerance:
            raise ValueError(
                f"Position weights must sum to 1.0, got {total_weight:.6f}"
            )
        return positions

    class Config:
        str_strip_whitespace = True
        validate_assignment = True


class UpdatePortfolioRequest(BaseModel):
    """Request schema for updating a portfolio."""

    name: Optional[str] = Field(
        None, min_length=1, max_length=100, description="Portfolio name"
    )
    description: Optional[str] = Field(
        None, max_length=500, description="Portfolio description"
    )
    starting_capital: Optional[float] = Field(
        None, gt=0, description="Starting capital amount"
    )
    base_currency: Optional[str] = Field(
        None, min_length=3, max_length=3, description="Base currency"
    )
    rebalance_interval_months: Optional[int] = Field(
        None,
        description="Rebalance schedule in months (1, 3, 6, 12); null disables",
    )

    @field_validator("rebalance_interval_months")
    @classmethod
    def validate_rebalance_interval_update(
        cls,
        v: Optional[int],
    ) -> Optional[int]:
        if v is None:
            return None
        if v not in REBALANCE_INTERVALS:
            raise ValueError("rebalance_interval_months must be 1, 3, 6, 12, or null")
        return v

    @field_validator("name")
    @classmethod
    def name_stripped(cls, v: Optional[str]) -> Optional[str]:
        """Strip whitespace from name."""
        return v.strip() if v else None

    @field_validator("base_currency")
    @classmethod
    def currency_uppercase(cls, v: Optional[str]) -> Optional[str]:
        """Normalize currency to uppercase."""
        return v.strip().upper() if v else None

    class Config:
        str_strip_whitespace = True
        validate_assignment = True


class AddPositionRequest(BaseModel):
    """Request schema for adding a position."""

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        pattern=r"^[A-Z0-9-]+$",
        description="Ticker symbol (may contain hyphens)",
    )
    shares: float = Field(..., gt=0, description="Number of shares (must be > 0)")
    weight_target: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Target weight"
    )
    purchase_price: Optional[float] = Field(
        None, gt=0, description="Purchase price per share"
    )
    purchase_date: Optional[date] = Field(None, description="Purchase date")

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return v.strip().upper()

    class Config:
        str_strip_whitespace = True
        validate_assignment = True


class UpdatePositionRequest(BaseModel):
    """Request schema for updating a position."""

    shares: Optional[float] = Field(None, gt=0, description="Number of shares")
    weight_target: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Target weight"
    )
    purchase_price: Optional[float] = Field(
        None, gt=0, description="Purchase price per share"
    )
    purchase_date: Optional[date] = Field(None, description="Purchase date")

    class Config:
        str_strip_whitespace = True
        validate_assignment = True
