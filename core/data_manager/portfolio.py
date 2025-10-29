"""Portfolio domain model (pure Python, no ORM dependencies)."""

import logging
from datetime import date
from typing import Dict, List, Optional

from core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class Position:
    """Position domain model."""

    def __init__(
        self,
        ticker: str,
        shares: float,
        weight_target: Optional[float] = None,
        purchase_price: Optional[float] = None,
        purchase_date: Optional[date] = None,
    ) -> None:
        """
        Initialize position.

        Args:
            ticker: Ticker symbol
            shares: Number of shares (must be > 0)
            weight_target: Target weight (0.0 to 1.0)
            purchase_price: Purchase price per share
            purchase_date: Purchase date
        """
        if shares <= 0:
            raise ValidationError("Shares must be greater than 0")

        if weight_target is not None and (
            weight_target < 0 or weight_target > 1
        ):
            raise ValidationError(
                "Weight target must be between 0.0 and 1.0"
            )

        self.ticker = ticker.strip().upper()
        self.shares = shares
        self.weight_target = weight_target
        self.purchase_price = purchase_price
        self.purchase_date = purchase_date

    def __eq__(self, other: object) -> bool:
        """Compare positions by ticker."""
        if not isinstance(other, Position):
            return False
        return self.ticker == other.ticker

    def __repr__(self) -> str:
        return (
            f"<Position(ticker={self.ticker}, shares={self.shares}, "
            f"weight={self.weight_target})>"
        )


class Portfolio:
    """Portfolio domain model (pure Python, no framework dependencies)."""

    def __init__(
        self,
        name: str,
        starting_capital: float,
        description: Optional[str] = None,
        base_currency: str = "USD",
        portfolio_id: Optional[str] = None,
    ) -> None:
        """
        Initialize portfolio.

        Args:
            name: Portfolio name
            starting_capital: Starting capital amount (must be > 0)
            description: Optional description
            base_currency: Base currency (default: USD)
            portfolio_id: Optional portfolio ID (for existing portfolios)
        """
        if not name or len(name) > 100:
            raise ValidationError("Name must be 1-100 characters")

        if starting_capital <= 0:
            raise ValidationError("Starting capital must be greater than 0")

        self.id = portfolio_id
        self.name = name.strip()
        self.description = description
        self.starting_capital = starting_capital
        self.base_currency = base_currency
        self._positions: List[Position] = []

    def add_position(
        self,
        ticker: str,
        shares: float,
        weight_target: Optional[float] = None,
        purchase_price: Optional[float] = None,
        purchase_date: Optional[date] = None,
    ) -> None:
        """
        Add position to portfolio.

        Args:
            ticker: Ticker symbol
            shares: Number of shares
            weight_target: Target weight (0.0 to 1.0)
            purchase_price: Purchase price per share
            purchase_date: Purchase date

        Raises:
            ValidationError: If position already exists or validation fails
        """
        ticker = ticker.strip().upper()

        if self._position_exists(ticker):
            raise ValidationError(
                f"Position {ticker} already exists in portfolio"
            )

        position = Position(
            ticker=ticker,
            shares=shares,
            weight_target=weight_target,
            purchase_price=purchase_price,
            purchase_date=purchase_date,
        )

        self._positions.append(position)
        logger.debug(f"Added position {ticker} to portfolio {self.name}")

    def remove_position(self, ticker: str) -> None:
        """
        Remove position from portfolio.

        Args:
            ticker: Ticker symbol to remove

        Raises:
            ValidationError: If position doesn't exist
        """
        ticker = ticker.strip().upper()

        position = self.get_position(ticker)
        if position:
            self._positions.remove(position)
            logger.debug(f"Removed position {ticker} from portfolio {self.name}")
        else:
            raise ValidationError(
                f"Position {ticker} not found in portfolio"
            )

    def update_position(
        self,
        ticker: str,
        shares: Optional[float] = None,
        weight_target: Optional[float] = None,
        purchase_price: Optional[float] = None,
        purchase_date: Optional[date] = None,
    ) -> None:
        """
        Update position in portfolio.

        Args:
            ticker: Ticker symbol
            shares: New number of shares
            weight_target: New target weight
            purchase_price: New purchase price
            purchase_date: New purchase date

        Raises:
            ValidationError: If position doesn't exist or validation fails
        """
        ticker = ticker.strip().upper()
        position = self.get_position(ticker)

        if not position:
            raise ValidationError(
                f"Position {ticker} not found in portfolio"
            )

        if shares is not None:
            if shares <= 0:
                raise ValidationError("Shares must be greater than 0")
            position.shares = shares

        if weight_target is not None:
            if weight_target < 0 or weight_target > 1:
                raise ValidationError(
                    "Weight target must be between 0.0 and 1.0"
                )
            position.weight_target = weight_target

        if purchase_price is not None:
            if purchase_price <= 0:
                raise ValidationError(
                    "Purchase price must be greater than 0"
                )
            position.purchase_price = purchase_price

        if purchase_date is not None:
            position.purchase_date = purchase_date

        logger.debug(f"Updated position {ticker} in portfolio {self.name}")

    def get_position(self, ticker: str) -> Optional[Position]:
        """
        Get position by ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Position object or None if not found
        """
        ticker = ticker.strip().upper()
        for position in self._positions:
            if position.ticker == ticker:
                return position
        return None

    def get_all_positions(self) -> List[Position]:
        """
        Get all positions in portfolio.

        Returns:
            List of Position objects (copy)
        """
        return list(self._positions)

    def calculate_current_weights(
        self, prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate current weights based on current prices.

        Args:
            prices: Dictionary mapping ticker to current price

        Returns:
            Dictionary mapping ticker to current weight (0.0 to 1.0)

        Raises:
            ValidationError: If price data is missing for any position
        """
        if not self._positions:
            return {}

        # Calculate total value
        total_value = 0.0
        position_values: Dict[str, float] = {}

        for position in self._positions:
            if position.ticker not in prices:
                raise ValidationError(
                    f"Price not available for ticker: {position.ticker}"
                )
            price = prices[position.ticker]
            value = position.shares * price
            position_values[position.ticker] = value
            total_value += value

        if total_value == 0:
            # All positions have zero value
            return {pos.ticker: 0.0 for pos in self._positions}

        # Calculate weights
        weights: Dict[str, float] = {}
        for ticker, value in position_values.items():
            weights[ticker] = value / total_value

        return weights

    def calculate_current_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value.

        Args:
            prices: Dictionary mapping ticker to current price

        Returns:
            Total current value

        Raises:
            ValidationError: If price data is missing for any position
        """
        total_value = 0.0

        for position in self._positions:
            if position.ticker not in prices:
                raise ValidationError(
                    f"Price not available for ticker: {position.ticker}"
                )
            price = prices[position.ticker]
            total_value += position.shares * price

        return total_value

    def _position_exists(self, ticker: str) -> bool:
        """
        Check if position exists in portfolio.

        Args:
            ticker: Ticker symbol

        Returns:
            True if position exists, False otherwise
        """
        return self.get_position(ticker) is not None

    def validate_weights(self, tolerance: float = 0.0001) -> bool:
        """
        Validate that position weights sum to approximately 1.0.

        Args:
            tolerance: Tolerance for weight sum (default: 0.0001)

        Returns:
            True if weights sum to 1.0 ± tolerance
        """
        total_weight = sum(
            pos.weight_target or 0.0 for pos in self._positions
        )
        return abs(total_weight - 1.0) <= tolerance

    def __repr__(self) -> str:
        return (
            f"<Portfolio(name={self.name}, "
            f"capital={self.starting_capital}, "
            f"positions={len(self._positions)})>"
        )

