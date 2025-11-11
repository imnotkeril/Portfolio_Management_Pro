"""Base forecaster class and result types."""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.exceptions import CalculationError, InsufficientDataError

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Result of forecasting operation."""

    method: str
    """Forecasting method name"""

    forecast_dates: pd.DatetimeIndex
    """Dates for forecasted values"""

    forecast_values: np.ndarray
    """Forecasted prices/values"""

    forecast_returns: np.ndarray
    """Forecasted returns (if applicable)"""

    confidence_intervals: Dict[str, np.ndarray] = field(default_factory=dict)
    """Confidence intervals: upper_95, lower_95, upper_80, lower_80"""

    final_value: float = 0.0
    """Final forecasted value"""

    change_pct: float = 0.0
    """Percentage change from last historical value"""

    validation_metrics: Optional[Dict[str, float]] = None
    """Out-of-sample validation metrics: MAPE, RMSE, MAE, direction_accuracy, r_squared"""

    model_info: Dict[str, any] = field(default_factory=dict)
    """Model information: parameters, AIC, BIC, training_time"""

    residuals: Optional[np.ndarray] = None
    """Model residuals for diagnostics"""

    success: bool = True
    """Whether forecast was successful"""

    message: str = ""
    """Status message"""

    def __post_init__(self) -> None:
        """Calculate final_value and change_pct if not provided."""
        if len(self.forecast_values) > 0:
            if self.final_value == 0.0:
                self.final_value = float(self.forecast_values[-1])
            
            # Calculate change_pct if not provided and we have a reference
            # Note: change_pct calculation requires last_price, which is not available here
            # So we leave it to the forecaster to set it explicitly

    def to_dict(self) -> Dict[str, any]:
        """
        Convert result to dictionary for serialization.
        
        Converts numpy arrays to lists and handles datetime serialization.
        
        Returns:
            Dictionary representation of ForecastResult
        """
        # Convert DatetimeIndex to list of ISO format strings
        forecast_dates_list = [
            date.isoformat() if hasattr(date, "isoformat") else str(date)
            for date in self.forecast_dates
        ]
        
        result = {
            "method": self.method,
            "forecast_dates": forecast_dates_list,
            "forecast_values": self.forecast_values.tolist() if isinstance(self.forecast_values, np.ndarray) else list(self.forecast_values),
            "forecast_returns": self.forecast_returns.tolist() if isinstance(self.forecast_returns, np.ndarray) else list(self.forecast_returns),
            "confidence_intervals": {
                k: v.tolist() if isinstance(v, np.ndarray) else (
                    list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else v
                )
                for k, v in self.confidence_intervals.items()
            },
            "final_value": float(self.final_value),
            "change_pct": float(self.change_pct),
            "validation_metrics": self.validation_metrics,
            "model_info": self.model_info,
            "residuals": (
                self.residuals.tolist() 
                if self.residuals is not None and isinstance(self.residuals, np.ndarray)
                else (
                    list(self.residuals) 
                    if self.residuals is not None and hasattr(self.residuals, "__iter__") and not isinstance(self.residuals, str)
                    else None
                )
            ),
            "success": bool(self.success),
            "message": str(self.message),
        }
        return result


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasters.

    All forecasting methods must inherit from this class and implement
    the forecast() method.
    """

    def __init__(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None,
    ) -> None:
        """
        Initialize base forecaster.

        Args:
            prices: Historical prices series with DatetimeIndex
            returns: Optional historical returns series (calculated if not provided)
        """
        if prices.empty:
            raise ValueError("Prices series cannot be empty")

        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("Prices must have DatetimeIndex")

        if len(prices) < 30:
            raise InsufficientDataError(
                f"Insufficient data points: {len(prices)} (minimum: 30)"
            )

        self.prices = prices.sort_index()
        
        # Calculate returns if not provided
        if returns is not None:
            self.returns = returns.sort_index()
            # Ensure returns are valid (no NaN, finite values)
            self.returns = self.returns.replace([np.inf, -np.inf], np.nan).dropna()
        else:
            self.returns = self.prices.pct_change().dropna()
            # Remove infinite values
            self.returns = self.returns.replace([np.inf, -np.inf], np.nan).dropna()

        # Ensure returns index is sorted
        if len(self.returns) > 0:
            self.returns = self.returns.sort_index()
            
            # Validate returns don't contain invalid values
            if not np.all(np.isfinite(self.returns.values)):
                logger.warning(
                    f"Returns contain non-finite values, removing them. "
                    f"Original: {len(self.returns)}, after cleanup: {len(self.returns[np.isfinite(self.returns.values)])}"
                )
                self.returns = self.returns[np.isfinite(self.returns.values)]

        logger.debug(
            f"Initialized forecaster: {len(self.prices)} price points, "
            f"{len(self.returns)} return points"
        )

    @abstractmethod
    def forecast(
        self,
        horizon: int,
        **kwargs,
    ) -> ForecastResult:
        """
        Generate forecast for specified horizon.

        Args:
            horizon: Number of periods ahead to forecast
            **kwargs: Method-specific parameters

        Returns:
            ForecastResult with forecasted values and metadata
        """
        pass

    def get_name(self) -> str:
        """Return forecaster name."""
        return self.__class__.__name__

    def evaluate_forecast(
        self,
        forecast: ForecastResult,
        actual: pd.Series,
    ) -> Dict[str, float]:
        """
        Evaluate forecast quality against actual values.

        Args:
            forecast: ForecastResult to evaluate
            actual: Actual values for comparison

        Returns:
            Dictionary with metrics: MAPE, RMSE, MAE, direction_accuracy, r_squared
        """
        # Import here to avoid circular dependency
        from core.forecasting_engine.utils import evaluate_forecast_metrics

        # Validate inputs
        if len(forecast.forecast_values) == 0:
            logger.warning("Forecast is empty, cannot evaluate")
            return {
                "mape": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "direction_accuracy": np.nan,
                "r_squared": np.nan,
            }

        if actual.empty:
            logger.warning("Actual values are empty, cannot evaluate")
            return {
                "mape": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "direction_accuracy": np.nan,
                "r_squared": np.nan,
            }

        # Normalize timezones for comparison
        forecast_dates = forecast.forecast_dates
        actual_index = actual.index
        
        # Remove timezone if present for comparison
        if hasattr(forecast_dates, "tz") and forecast_dates.tz is not None:
            forecast_dates = forecast_dates.tz_localize(None)
        if hasattr(actual_index, "tz") and actual_index.tz is not None:
            actual_index = actual_index.tz_localize(None)
            actual = pd.Series(actual.values, index=actual_index)

        # Align forecast and actual by dates
        forecast_series = pd.Series(
            forecast.forecast_values,
            index=forecast_dates,
        )

        # Find common dates
        common_dates = forecast_series.index.intersection(actual.index)

        if len(common_dates) == 0:
            logger.warning("No common dates between forecast and actual")
            return {
                "mape": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "direction_accuracy": np.nan,
                "r_squared": np.nan,
            }

        forecast_aligned = forecast_series.reindex(common_dates)
        actual_aligned = actual.reindex(common_dates)

        # Remove NaN values
        valid_mask = ~(forecast_aligned.isna() | actual_aligned.isna())
        forecast_clean = forecast_aligned[valid_mask]
        actual_clean = actual_aligned[valid_mask]

        if len(forecast_clean) == 0:
            logger.warning("No valid data points after alignment")
            return {
                "mape": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "direction_accuracy": np.nan,
                "r_squared": np.nan,
            }

        return evaluate_forecast_metrics(forecast_clean.values, actual_clean.values)

    def _calculate_change_pct(
        self,
        forecast_values: np.ndarray,
        last_price: float,
    ) -> float:
        """
        Calculate percentage change from last historical price.

        Args:
            forecast_values: Forecasted values
            last_price: Last historical price

        Returns:
            Percentage change (can be negative)
        """
        if len(forecast_values) == 0:
            logger.warning("Cannot calculate change_pct: forecast_values is empty")
            return 0.0

        if last_price == 0 or not np.isfinite(last_price):
            logger.warning(
                f"Cannot calculate change_pct: last_price is invalid ({last_price})"
            )
            return 0.0

        final_forecast = forecast_values[-1]
        
        # Validate final_forecast
        if not np.isfinite(final_forecast):
            logger.warning(
                f"Cannot calculate change_pct: final_forecast is invalid ({final_forecast})"
            )
            return 0.0

        try:
            change_pct = ((final_forecast - last_price) / last_price) * 100.0
            # Ensure result is finite
            if not np.isfinite(change_pct):
                logger.warning(f"Calculated change_pct is not finite: {change_pct}")
                return 0.0
            return float(change_pct)
        except (ZeroDivisionError, OverflowError) as e:
            logger.warning(f"Error calculating change_pct: {e}")
            return 0.0

    def _create_forecast_dates(
        self,
        horizon: int,
        last_date: pd.Timestamp,
    ) -> pd.DatetimeIndex:
        """
        Create forecast dates starting from day after last historical date.

        Uses business days (excludes weekends) for financial forecasting.

        Args:
            horizon: Number of periods to forecast
            last_date: Last date in historical data

        Returns:
            DatetimeIndex with forecast dates

        Raises:
            ValueError: If horizon is invalid or last_date is not a valid timestamp
        """
        if horizon <= 0:
            raise ValueError(f"Horizon must be positive, got {horizon}")

        if not isinstance(last_date, pd.Timestamp):
            try:
                last_date = pd.Timestamp(last_date)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"last_date must be a valid timestamp, got {last_date}: {e}"
                ) from e

        # Normalize timezone (remove timezone for consistency)
        if hasattr(last_date, "tz") and last_date.tz is not None:
            last_date = last_date.tz_localize(None)

        try:
            # Generate business days (excludes weekends)
            forecast_dates = pd.bdate_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
            )
            
            # Ensure we got the right number of dates
            if len(forecast_dates) != horizon:
                logger.warning(
                    f"Requested {horizon} dates but got {len(forecast_dates)}. "
                    f"Using bdate_range may skip weekends/holidays."
                )
                # If we need more dates, extend using business days
                if len(forecast_dates) < horizon:
                    additional_dates = pd.bdate_range(
                        start=forecast_dates[-1] + pd.Timedelta(days=1),
                        periods=horizon - len(forecast_dates),
                    )
                    forecast_dates = forecast_dates.union(additional_dates)[:horizon]
            
            return forecast_dates
        except Exception as e:
            logger.error(f"Error creating forecast dates: {e}", exc_info=True)
            # Fallback: use simple date range
            try:
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=horizon,
                    freq="D",
                )
                logger.warning("Using daily frequency instead of business days")
                return forecast_dates
            except Exception as e2:
                raise ValueError(
                    f"Failed to create forecast dates: {e2}"
                ) from e2

