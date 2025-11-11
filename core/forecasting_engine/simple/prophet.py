"""Prophet forecaster implementation."""

import logging
import time
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

from core.exceptions import CalculationError
from core.forecasting_engine.base import BaseForecaster, ForecastResult
from core.forecasting_engine.utils import calculate_confidence_intervals

logger = logging.getLogger(__name__)


class ProphetForecaster(BaseForecaster):
    """
    Prophet forecaster (Facebook/Meta).

    Fast, simple forecasting model good for seasonal patterns.
    """

    def __init__(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None,
    ) -> None:
        """
        Initialize Prophet forecaster.

        Args:
            prices: Historical prices series
            returns: Optional returns series (not used by Prophet)
        """
        if Prophet is None:
            raise ImportError(
                "Prophet library not installed. Install with: pip install prophet"
            )

        super().__init__(prices, returns)

    def forecast(
        self,
        horizon: int,
        growth: str = "linear",
        seasonality: bool = True,
        holidays: bool = False,
        **kwargs,
    ) -> ForecastResult:
        """
        Generate forecast using Prophet.

        Args:
            horizon: Number of days to forecast
            growth: Growth model ('linear' or 'logistic')
            seasonality: Whether to include seasonality
            holidays: Whether to include US holidays
            **kwargs: Additional Prophet parameters

        Returns:
            ForecastResult with forecasted values
        """
        if Prophet is None:
            raise ImportError("Prophet library not installed")

        start_time = time.time()

        try:
            # Validate data before using Prophet
            if len(self.prices) < 30:
                raise CalculationError(
                    f"Prophet requires at least 30 data points, "
                    f"got {len(self.prices)}"
                )

            # Work with a copy to avoid modifying original data
            prices_clean = self.prices.copy()

            # Check for invalid values
            if prices_clean.isna().any() or (prices_clean <= 0).any():
                logger.warning(
                    "Prophet: Data contains NaN or non-positive values, "
                    "filtering..."
                )
                prices_clean = prices_clean.dropna()
                prices_clean = prices_clean[prices_clean > 0]
                if len(prices_clean) < 30:
                    raise CalculationError(
                        "Prophet: Not enough valid data points after filtering"
                    )

            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            df = pd.DataFrame({
                "ds": prices_clean.index,
                "y": prices_clean.values,
            })

            # Convert dates to datetime and normalize timezone
            if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
                df["ds"] = pd.to_datetime(df["ds"])
            # Normalize timezone (remove timezone info for consistency)
            if df["ds"].dt.tz is not None:
                df["ds"] = df["ds"].dt.tz_localize(None)

            # Ensure growth is 'linear' unless 'cap' is provided
            # For financial data, linear growth is usually more appropriate
            if growth == "logistic" and "cap" not in kwargs:
                logger.warning(
                    "Prophet: Logistic growth requires 'cap' column. "
                    "Switching to linear growth."
                )
                growth = "linear"

            # Get holidays if requested
            holidays_df = None
            if holidays:
                holidays_df = self._get_holidays(
                    start_date=df["ds"].min(),
                    end_date=df["ds"].max() + pd.Timedelta(days=horizon),
                )

            # Initialize Prophet model with conservative settings for financial data
            model = Prophet(
                growth=growth,
                yearly_seasonality=seasonality and len(prices_clean) >= 365,
                weekly_seasonality=False,  # Disable weekly for daily financial data
                daily_seasonality=False,  # Not useful for daily stock data
                holidays=holidays_df,
                changepoint_prior_scale=0.05,  # More conservative for financial data
                seasonality_prior_scale=10.0,  # Default
                **kwargs,
            )

            # Fit model
            try:
                model.fit(df)
            except Exception as e:
                raise CalculationError(
                    f"Prophet model fitting failed: {e}"
                ) from e

            # Create future dataframe
            # Use business days for financial data (more appropriate than calendar days)
            # Prophet's make_future_dataframe uses calendar days, so we create custom dates
            last_date = df["ds"].iloc[-1]
            # Normalize timezone for last_date
            if hasattr(last_date, "tz") and last_date.tz is not None:
                last_date = last_date.tz_localize(None)

            # Generate business days for forecast
            forecast_dates = pd.bdate_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
            )
            future = pd.DataFrame({"ds": forecast_dates})

            # Combine with historical data (Prophet needs all dates)
            # Use sort=False to preserve order, then sort by date
            future_all = pd.concat([df[["ds"]], future], ignore_index=False)
            future_all = future_all.sort_values("ds").reset_index(drop=True)

            # Generate forecast
            try:
                forecast_df = model.predict(future_all)
            except Exception as e:
                raise CalculationError(
                    f"Prophet prediction failed: {e}"
                ) from e

            # Extract forecast values (only future period)
            # Normalize timezone for comparison
            last_historical_date = pd.Timestamp(prices_clean.index[-1])
            if last_historical_date.tz is not None:
                last_historical_date = last_historical_date.tz_localize(None)

            # Normalize forecast_df dates for comparison
            forecast_df_dates = pd.to_datetime(forecast_df["ds"])
            if forecast_df_dates.dt.tz is not None:
                forecast_df_dates = forecast_df_dates.dt.tz_localize(None)

            forecast_mask = forecast_df_dates > last_historical_date

            if not forecast_mask.any():
                raise CalculationError(
                    "Prophet: No forecast dates found after historical period"
                )

            forecast_values = forecast_df.loc[forecast_mask, "yhat"].values

            # Validate forecast values
            if np.any(np.isnan(forecast_values)) or np.any(
                np.isinf(forecast_values)
            ):
                logger.warning(
                    "Prophet forecast contains NaN or Inf values, cleaning"
                )
                forecast_values = np.nan_to_num(
                    forecast_values, nan=0.0, posinf=0.0, neginf=0.0
                )

            # Extract confidence intervals and dates
            forecast_dates = forecast_df_dates[forecast_mask]

            # Ensure forecast_dates is DatetimeIndex
            if not isinstance(forecast_dates, pd.DatetimeIndex):
                forecast_dates = pd.DatetimeIndex(forecast_dates)

            # Calculate confidence intervals
            upper_95 = forecast_df.loc[forecast_mask, "yhat_upper"].values
            lower_95 = forecast_df.loc[forecast_mask, "yhat_lower"].values

            # Validate confidence intervals
            upper_95 = np.nan_to_num(upper_95, nan=0.0, posinf=0.0, neginf=0.0)
            lower_95 = np.nan_to_num(lower_95, nan=0.0, posinf=0.0, neginf=0.0)

            # Calculate returns
            if len(forecast_values) > 1:
                forecast_returns = np.diff(forecast_values) / np.maximum(
                    forecast_values[:-1], 1e-10
                )
                last_price = float(prices_clean.iloc[-1])
                first_return = (forecast_values[0] - last_price) / max(
                    last_price, 1e-10
                )
                forecast_returns = np.insert(forecast_returns, 0, first_return)
            else:
                last_price = float(prices_clean.iloc[-1])
                first_return = (forecast_values[0] - last_price) / max(
                    last_price, 1e-10
                )
                forecast_returns = np.array([first_return])

            # Calculate change percentage
            change_pct = self._calculate_change_pct(
                forecast_values, prices_clean.iloc[-1]
            )

            # Get residuals - align by dates
            historical_mask = ~forecast_mask
            if historical_mask.any():
                # Get historical forecast and align with actual prices
                historical_forecast_df = forecast_df.loc[historical_mask].copy()
                historical_forecast_df["ds"] = pd.to_datetime(
                    historical_forecast_df["ds"]
                )
                if historical_forecast_df["ds"].dt.tz is not None:
                    historical_forecast_df["ds"] = (
                        historical_forecast_df["ds"].dt.tz_localize(None)
                    )

                # Create Series from historical forecast
                historical_forecast_series = pd.Series(
                    historical_forecast_df["yhat"].values,
                    index=historical_forecast_df["ds"],
                )

                # Align with actual prices by date
                prices_aligned = prices_clean.reindex(
                    historical_forecast_series.index, method="nearest"
                )

                # Calculate residuals only for matching dates
                valid_mask = ~(
                    prices_aligned.isna() | historical_forecast_series.isna()
                )
                if valid_mask.any():
                    residuals = (
                        prices_aligned[valid_mask].values
                        - historical_forecast_series[valid_mask].values
                    )
                else:
                    residuals = None
            else:
                residuals = None

            training_time = time.time() - start_time

            return ForecastResult(
                method="Prophet",
                forecast_dates=forecast_dates,
                forecast_values=forecast_values,
                forecast_returns=forecast_returns,
                confidence_intervals={
                    "upper_95": upper_95,
                    "lower_95": lower_95,
                },
                final_value=float(forecast_values[-1]),
                change_pct=change_pct,
                model_info={
                    "growth": growth,
                    "seasonality": seasonality,
                    "holidays": holidays,
                    "training_time": training_time,
                },
                residuals=residuals,
                success=True,
                message="Prophet forecast completed successfully",
            )

        except Exception as e:
            logger.error(f"Prophet forecast failed: {e}", exc_info=True)
            raise CalculationError(f"Prophet forecast failed: {e}") from e

    def _get_holidays(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Get US holidays for Prophet in the specified date range.

        Args:
            start_date: Start date for holiday range
            end_date: End date for holiday range

        Returns:
            DataFrame with holidays in Prophet format
        """
        # Generate US holidays dynamically
        holiday_list = []

        # Normalize dates
        if hasattr(start_date, "tz") and start_date.tz is not None:
            start_date = start_date.tz_localize(None)
        if hasattr(end_date, "tz") and end_date.tz is not None:
            end_date = end_date.tz_localize(None)

        start_year = start_date.year
        end_year = end_date.year

        # Generate holidays for each year in range
        for year in range(start_year, end_year + 1):
            # New Year's Day
            holiday_list.append(date(year, 1, 1))
            # Independence Day
            holiday_list.append(date(year, 7, 4))
            # Christmas
            holiday_list.append(date(year, 12, 25))

        # Filter holidays within date range
        holiday_dates = [
            d
            for d in holiday_list
            if start_date.date() <= d <= end_date.date()
        ]

        if not holiday_dates:
            # Return empty DataFrame with correct structure
            return pd.DataFrame(
                columns=["holiday", "ds", "lower_window", "upper_window"]
            )

        holidays = pd.DataFrame({
            "holiday": "us_holiday",
            "ds": pd.to_datetime(holiday_dates),
            "lower_window": 0,
            "upper_window": 0,
        })

        return holidays

