"""Forecasting service for orchestrating forecast operations."""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.exceptions import (
    CalculationError,
    InsufficientDataError,
    ValidationError,
)
from services.data_service import DataService
from services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)


class ForecastingService:
    """Service for orchestrating forecasting operations."""

    def __init__(
        self,
        data_service: Optional[DataService] = None,
        portfolio_service: Optional[PortfolioService] = None,
    ) -> None:
        """
        Initialize forecasting service.

        Args:
            data_service: Optional data service instance
            portfolio_service: Optional portfolio service instance
        """
        self._data_service = data_service or DataService()
        self._portfolio_service = portfolio_service or PortfolioService()

    def forecast_asset(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        horizon: int,
        method: str,
        method_params: Optional[Dict[str, any]] = None,
        out_of_sample: bool = False,
        training_ratio: float = 0.3,
    ) -> Dict[str, any]:
        """
        Forecast prices for a single asset.

        Args:
            ticker: Ticker symbol
            start_date: Start date for training data
            end_date: End date for training data (validation end if out_of_sample)
            horizon: Number of days to forecast ahead
            method: Forecasting method name
            method_params: Method-specific parameters
            out_of_sample: If True, use period BEFORE start_date for training
            training_ratio: Ratio of analysis period to use for training

        Returns:
            Dictionary with forecast result

        Raises:
            ValidationError: If date range is invalid
            InsufficientDataError: If insufficient data available
        """
        # Validate date range
        if start_date >= end_date:
            raise ValidationError("Start date must be before end date")

        # Determine training period
        if out_of_sample:
            analysis_days = (end_date - start_date).days
            training_days = int(analysis_days * training_ratio)
            training_start = start_date - timedelta(days=training_days)
            training_end = start_date
            logger.info(
                f"Out-of-sample mode: training {training_start} to {training_end}, "
                f"validation {start_date} to {end_date}"
            )
        else:
            training_start = start_date
            training_end = end_date

        # Fetch price data
        logger.info(
            f"Fetching price data for {ticker} from {training_start} to {training_end}"
        )

        price_data = self._data_service.fetch_historical_prices(
            ticker=ticker,
            start_date=training_start,
            end_date=training_end,
            use_cache=True,
            save_to_db=False,
        )

        if price_data.empty:
            raise InsufficientDataError(
                f"No price data available for {ticker} "
                f"from {training_start} to {training_end}"
            )

        # Convert to Series with DatetimeIndex
        if "Date" in price_data.columns:
            price_data.set_index("Date", inplace=True)
            price_data.index = pd.to_datetime(price_data.index, errors="coerce")
            price_data.index = price_data.index.tz_localize(None)

        # Extract adjusted close prices
        if "Adjusted_Close" in price_data.columns:
            prices = price_data["Adjusted_Close"]
        elif "Close" in price_data.columns:
            prices = price_data["Close"]
        else:
            raise InsufficientDataError("No price column found in data")

        prices = prices.sort_index().dropna()

        if len(prices) < 30:
            raise InsufficientDataError(
                f"Insufficient data points: {len(prices)} (minimum: 30)"
            )

        # Get forecaster class
        forecaster_class = self._get_forecaster_class(method)
        if forecaster_class is None:
            raise ValueError(f"Unknown forecasting method: {method}")

        # Initialize and run forecast
        try:
            forecaster = forecaster_class(prices=prices)
            
            # Calculate total forecast horizon
            # For out-of-sample: need to forecast Validation Period + Forecast Horizon
            # For regular: just Forecast Horizon
            if out_of_sample:
                validation_days = (end_date - start_date).days
                total_horizon = validation_days + horizon
                logger.info(
                    f"Out-of-sample forecast: {validation_days} days validation + "
                    f"{horizon} days forecast = {total_horizon} days total"
                )
            else:
                total_horizon = horizon
            
            forecast_result = forecaster.forecast(horizon=total_horizon, **(method_params or {}))

            # If out-of-sample, evaluate on validation period
            if out_of_sample:
                validation_days = (end_date - start_date).days
                validation_prices = self._data_service.fetch_historical_prices(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=True,
                    save_to_db=False,
                )

                validation_metrics = None
                validation_forecast_dates = forecast_result.forecast_dates[:validation_days] if len(forecast_result.forecast_dates) >= validation_days else forecast_result.forecast_dates
                validation_forecast_values = forecast_result.forecast_values[:validation_days] if len(forecast_result.forecast_values) >= validation_days else forecast_result.forecast_values

                if not validation_prices.empty:
                    if "Date" in validation_prices.columns:
                        validation_prices.set_index("Date", inplace=True)
                        validation_prices.index = pd.to_datetime(
                            validation_prices.index, errors="coerce"
                        )
                        validation_prices.index = validation_prices.index.tz_localize(None)

                    if "Adjusted_Close" in validation_prices.columns:
                        actual = validation_prices["Adjusted_Close"]
                    elif "Close" in validation_prices.columns:
                        actual = validation_prices["Close"]
                    else:
                        actual = pd.Series(dtype=float)

                    if len(actual) > 0:
                        actual = actual.sort_index().dropna()
                        # Align forecast dates with actual dates (take first validation_days)
                        # The forecast_result contains validation period + forecast period
                        # We need to take only the validation period part for comparison
                        
                        min_len = min(len(validation_forecast_dates), len(actual))
                        if min_len > 0:
                            actual_aligned = actual.iloc[:min_len]
                            
                            # Create temporary ForecastResult with only validation period for evaluation
                            from core.forecasting_engine.base import ForecastResult
                            validation_forecast_result = ForecastResult(
                                method=forecast_result.method,
                                forecast_dates=validation_forecast_dates[:min_len],
                                forecast_values=validation_forecast_values[:min_len],
                                forecast_returns=forecast_result.forecast_returns[:min_len] if len(forecast_result.forecast_returns) >= min_len else np.array([]),
                                confidence_intervals={
                                    k: v[:min_len] if len(v) >= min_len else v
                                    for k, v in forecast_result.confidence_intervals.items()
                                },
                                final_value=float(validation_forecast_values[min_len - 1]) if min_len > 0 else 0.0,
                                change_pct=0.0,
                                validation_metrics=None,
                                model_info=forecast_result.model_info,
                                residuals=forecast_result.residuals,
                                success=forecast_result.success,
                                message=forecast_result.message,
                            )

                            try:
                                validation_metrics = forecaster.evaluate_forecast(
                                    validation_forecast_result, actual_aligned
                                )
                                logger.info(f"Validation metrics calculated for {method}: {validation_metrics}")
                                
                                # Calculate residuals for validation period
                                # Residuals = actual - forecast
                                if len(actual_aligned) > 0 and len(validation_forecast_values[:min_len]) > 0:
                                    # Align actual and forecast by dates
                                    actual_values_aligned = actual_aligned.reindex(validation_forecast_dates[:min_len])
                                    forecast_values_aligned = validation_forecast_values[:min_len]
                                    
                                    # Find common valid indices
                                    valid_mask = ~(actual_values_aligned.isna() | ~np.isfinite(forecast_values_aligned))
                                    if valid_mask.any():
                                        actual_valid = actual_values_aligned[valid_mask].values
                                        forecast_valid = forecast_values_aligned[valid_mask]
                                        validation_residuals = actual_valid - forecast_valid
                                        
                                        # Filter out invalid values
                                        valid_residuals_mask = np.isfinite(validation_residuals)
                                        if np.any(valid_residuals_mask):
                                            validation_residuals_clean = validation_residuals[valid_residuals_mask]
                                            # Update residuals in forecast_result
                                            if len(validation_residuals_clean) > 0:
                                                forecast_result.residuals = validation_residuals_clean
                                                logger.debug(f"Updated residuals for validation period: {len(validation_residuals_clean)} values")
                            except Exception as e:
                                logger.warning(f"Failed to evaluate forecast for {method}: {e}")
                                validation_metrics = {
                                    "mape": np.nan,
                                    "rmse": np.nan,
                                    "mae": np.nan,
                                    "direction_accuracy": np.nan,
                                    "r_squared": np.nan,
                                }
                        else:
                            logger.warning(f"No overlapping dates between forecast and actual for {method}")
                            validation_metrics = {
                                "mape": np.nan,
                                "rmse": np.nan,
                                "mae": np.nan,
                                "direction_accuracy": np.nan,
                                "r_squared": np.nan,
                            }
                    else:
                        logger.warning(f"No actual values found for validation period for {method}")
                        validation_metrics = {
                            "mape": np.nan,
                            "rmse": np.nan,
                            "mae": np.nan,
                            "direction_accuracy": np.nan,
                            "r_squared": np.nan,
                        }
                else:
                    logger.warning(f"Validation prices are empty for {method}, cannot calculate validation metrics")
                    validation_metrics = {
                        "mape": np.nan,
                        "rmse": np.nan,
                        "mae": np.nan,
                        "direction_accuracy": np.nan,
                        "r_squared": np.nan,
                    }
                
                # Set validation_metrics (even if all NaN)
                forecast_result.validation_metrics = validation_metrics
            else:
                # If out_of_sample is False, use all forecast data as validation period
                validation_forecast_dates = forecast_result.forecast_dates
                validation_forecast_values = forecast_result.forecast_values
                # validation_metrics remain None when out_of_sample=False

                # Keep only Validation Period - do not create Forecast Period
                # Update forecast_result to contain only Validation Period
                if len(validation_forecast_values) > 0:
                    # Save residuals before update (they may be lost)
                    original_residuals = forecast_result.residuals
                    
                    forecast_result.forecast_values = validation_forecast_values
                    forecast_result.forecast_dates = validation_forecast_dates
                    
                    # Restore residuals if they existed
                    if original_residuals is not None:
                        forecast_result.residuals = original_residuals
                    
                    # Update final_value and change_pct based on last Validation Period value
                    last_historical_price = float(prices.iloc[-1])
                    if last_historical_price > 0 and len(validation_forecast_values) > 0:
                        forecast_result.final_value = float(validation_forecast_values[-1])
                        try:
                            forecast_result.change_pct = (
                                (forecast_result.final_value - last_historical_price) / last_historical_price
                            ) * 100.0
                            if not np.isfinite(forecast_result.change_pct):
                                forecast_result.change_pct = 0.0
                        except (ZeroDivisionError, OverflowError) as e:
                            logger.warning(f"Error calculating change_pct: {e}")
                            forecast_result.change_pct = 0.0
                    else:
                        if len(validation_forecast_values) > 0:
                            forecast_result.final_value = float(validation_forecast_values[-1])
                        forecast_result.change_pct = 0.0

            return forecast_result.to_dict()

        except Exception as e:
            logger.error(f"Error running forecast for {ticker} with {method}: {e}", exc_info=True)
            raise CalculationError(f"Forecast failed: {e}") from e

    def forecast_portfolio(
        self,
        portfolio_id: str,
        start_date: date,
        end_date: date,
        horizon: int,
        method: str,
        method_params: Optional[Dict[str, any]] = None,
        out_of_sample: bool = False,
        training_ratio: float = 0.3,
    ) -> Dict[str, any]:
        """
        Forecast portfolio returns.

        Args:
            portfolio_id: Portfolio ID
            start_date: Start date for training data
            end_date: End date for training data
            horizon: Number of days to forecast ahead
            method: Forecasting method name
            method_params: Method-specific parameters
            out_of_sample: If True, use period BEFORE start_date for training
            training_ratio: Ratio of analysis period to use for training

        Returns:
            Dictionary with portfolio forecast result
        """
        # Get portfolio
        portfolio = self._portfolio_service.get_portfolio(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_id} not found")

        positions = portfolio.get_all_positions()
        tickers = [pos.ticker for pos in positions if pos.ticker != "CASH"]

        if not tickers:
            raise ValueError("Portfolio has no non-CASH positions")

        # Fetch portfolio prices
        if out_of_sample:
            analysis_days = (end_date - start_date).days
            training_days = int(analysis_days * training_ratio)
            training_start = start_date - timedelta(days=training_days)
            training_end = start_date
        else:
            training_start = start_date
            training_end = end_date

        price_data = self._data_service.fetch_bulk_prices(
            tickers=tickers,
            start_date=training_start,
            end_date=training_end,
            use_cache=True,
            save_to_db=False,
        )

        if price_data.empty:
            raise InsufficientDataError(
                f"No price data available for portfolio from {training_start} to {training_end}"
            )

        # Convert to pivot format
        if "Ticker" in price_data.columns and "Adjusted_Close" in price_data.columns:
            if "Date" in price_data.columns:
                price_data["Date"] = pd.to_datetime(price_data["Date"], errors="coerce")
                price_data["Date"] = price_data["Date"].dt.tz_localize(None)
                pivot_df = price_data.pivot_table(
                    index="Date",
                    columns="Ticker",
                    values="Adjusted_Close",
                    aggfunc="last",
                )
                price_data = pivot_df

        # Calculate portfolio prices (cumulative portfolio value)
        portfolio_prices = self.calculate_portfolio_prices(price_data, positions)

        # Get forecaster class
        forecaster_class = self._get_forecaster_class(method)
        if forecaster_class is None:
            raise ValueError(f"Unknown forecasting method: {method}")

        # Initialize and run forecast
        try:
            forecaster = forecaster_class(prices=portfolio_prices)
            
            # Calculate total forecast horizon
            # For out-of-sample: need to forecast Validation Period + Forecast Horizon
            # For regular: just Forecast Horizon
            if out_of_sample:
                validation_days = (end_date - start_date).days
                total_horizon = validation_days + horizon
                logger.info(
                    f"Out-of-sample portfolio forecast: {validation_days} days validation + "
                    f"{horizon} days forecast = {total_horizon} days total"
                )
            else:
                total_horizon = horizon
            
            forecast_result = forecaster.forecast(horizon=total_horizon, **(method_params or {}))

            # If out-of-sample, evaluate on validation period
            if out_of_sample:
                validation_days = (end_date - start_date).days
                # Fetch validation period portfolio prices
                validation_price_data = self._data_service.fetch_bulk_prices(
                    tickers=tickers,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=True,
                    save_to_db=False,
                )

                validation_metrics = None
                validation_forecast_dates = forecast_result.forecast_dates[:validation_days] if len(forecast_result.forecast_dates) >= validation_days else forecast_result.forecast_dates
                validation_forecast_values = forecast_result.forecast_values[:validation_days] if len(forecast_result.forecast_values) >= validation_days else forecast_result.forecast_values

                if not validation_price_data.empty:
                    # Convert to pivot format
                    if "Ticker" in validation_price_data.columns and "Adjusted_Close" in validation_price_data.columns:
                        if "Date" in validation_price_data.columns:
                            validation_price_data["Date"] = pd.to_datetime(
                                validation_price_data["Date"], errors="coerce"
                            )
                            validation_price_data["Date"] = validation_price_data["Date"].dt.tz_localize(None)
                            validation_pivot = validation_price_data.pivot_table(
                                index="Date",
                                columns="Ticker",
                                values="Adjusted_Close",
                                aggfunc="last",
                            )
                            validation_price_data = validation_pivot

                    # Calculate validation portfolio prices
                    validation_portfolio_prices = self.calculate_portfolio_prices(
                        validation_price_data, positions
                    )

                    if len(validation_portfolio_prices) > 0:
                        validation_portfolio_prices = validation_portfolio_prices.sort_index().dropna()
                        # Align forecast dates with actual dates (take first validation_days)
                        # The forecast_result contains validation period + forecast period
                        # We need to take only the validation period part for comparison
                        
                        min_len = min(
                            len(validation_forecast_dates),
                            len(validation_portfolio_prices)
                        )
                        if min_len > 0:
                            actual_aligned = validation_portfolio_prices.iloc[:min_len]
                            
                            # Create temporary ForecastResult with only validation period for evaluation
                            from core.forecasting_engine.base import ForecastResult
                            validation_forecast_result = ForecastResult(
                                method=forecast_result.method,
                                forecast_dates=validation_forecast_dates[:min_len],
                                forecast_values=validation_forecast_values[:min_len],
                                forecast_returns=forecast_result.forecast_returns[:min_len] if len(forecast_result.forecast_returns) >= min_len else np.array([]),
                                confidence_intervals={
                                    k: v[:min_len] if len(v) >= min_len else v
                                    for k, v in forecast_result.confidence_intervals.items()
                                },
                                final_value=float(validation_forecast_values[min_len - 1]) if min_len > 0 else 0.0,
                                change_pct=0.0,
                                validation_metrics=None,
                                model_info=forecast_result.model_info,
                                residuals=forecast_result.residuals,
                                success=forecast_result.success,
                                message=forecast_result.message,
                            )

                            try:
                                validation_metrics = forecaster.evaluate_forecast(
                                    validation_forecast_result, actual_aligned
                                )
                                logger.info(f"Validation metrics calculated for portfolio {portfolio_id} with {method}: {validation_metrics}")
                                
                                # Calculate residuals for validation period
                                # Residuals = actual - forecast
                                if len(actual_aligned) > 0 and len(validation_forecast_values[:min_len]) > 0:
                                    # Align actual and forecast by dates
                                    actual_values_aligned = actual_aligned.reindex(validation_forecast_dates[:min_len])
                                    forecast_values_aligned = validation_forecast_values[:min_len]
                                    
                                    # Find common valid indices
                                    valid_mask = ~(actual_values_aligned.isna() | ~np.isfinite(forecast_values_aligned))
                                    if valid_mask.any():
                                        actual_valid = actual_values_aligned[valid_mask].values
                                        forecast_valid = forecast_values_aligned[valid_mask]
                                        validation_residuals = actual_valid - forecast_valid
                                        
                                        # Filter out invalid values
                                        valid_residuals_mask = np.isfinite(validation_residuals)
                                        if np.any(valid_residuals_mask):
                                            validation_residuals_clean = validation_residuals[valid_residuals_mask]
                                            # Update residuals in forecast_result
                                            if len(validation_residuals_clean) > 0:
                                                forecast_result.residuals = validation_residuals_clean
                                                logger.debug(f"Updated residuals for validation period (portfolio): {len(validation_residuals_clean)} values")
                            except Exception as e:
                                logger.warning(f"Failed to evaluate forecast for portfolio {portfolio_id} with {method}: {e}")
                                validation_metrics = {
                                    "mape": np.nan,
                                    "rmse": np.nan,
                                    "mae": np.nan,
                                    "direction_accuracy": np.nan,
                                    "r_squared": np.nan,
                                }
                        else:
                            logger.warning(f"No overlapping dates between forecast and actual for portfolio {portfolio_id} with {method}")
                            validation_metrics = {
                                "mape": np.nan,
                                "rmse": np.nan,
                                "mae": np.nan,
                                "direction_accuracy": np.nan,
                                "r_squared": np.nan,
                            }
                    else:
                        logger.warning(f"Validation portfolio prices are empty for portfolio {portfolio_id} with {method}")
                        validation_metrics = {
                            "mape": np.nan,
                            "rmse": np.nan,
                            "mae": np.nan,
                            "direction_accuracy": np.nan,
                            "r_squared": np.nan,
                        }
                else:
                    logger.warning(f"Validation price data is empty for portfolio {portfolio_id} with {method}, cannot calculate validation metrics")
                    validation_metrics = {
                        "mape": np.nan,
                        "rmse": np.nan,
                        "mae": np.nan,
                        "direction_accuracy": np.nan,
                        "r_squared": np.nan,
                    }
                
                # Set validation_metrics (even if all NaN)
                forecast_result.validation_metrics = validation_metrics
            else:
                # If out_of_sample is False, use all forecast data as validation period
                validation_forecast_dates = forecast_result.forecast_dates
                validation_forecast_values = forecast_result.forecast_values
                # validation_metrics remain None when out_of_sample=False

                # Keep only Validation Period - do not create Forecast Period
                # Update forecast_result to contain only Validation Period
                if len(validation_forecast_values) > 0:
                    # Save residuals before update (they may be lost)
                    original_residuals = forecast_result.residuals
                    
                    forecast_result.forecast_values = validation_forecast_values
                    forecast_result.forecast_dates = validation_forecast_dates
                    
                    # Restore residuals if they existed
                    if original_residuals is not None:
                        forecast_result.residuals = original_residuals
                    
                    # Update final_value and change_pct based on last Validation Period value
                    last_historical_price = float(portfolio_prices.iloc[-1])
                    if last_historical_price > 0 and len(validation_forecast_values) > 0:
                        forecast_result.final_value = float(validation_forecast_values[-1])
                        try:
                            forecast_result.change_pct = (
                                (forecast_result.final_value - last_historical_price) / last_historical_price
                            ) * 100.0
                            if not np.isfinite(forecast_result.change_pct):
                                forecast_result.change_pct = 0.0
                        except (ZeroDivisionError, OverflowError) as e:
                            logger.warning(f"Error calculating change_pct: {e}")
                            forecast_result.change_pct = 0.0
                    else:
                        if len(validation_forecast_values) > 0:
                            forecast_result.final_value = float(validation_forecast_values[-1])
                        forecast_result.change_pct = 0.0

            return forecast_result.to_dict()

        except Exception as e:
            logger.error(
                f"Error running portfolio forecast with {method}: {e}", exc_info=True
            )
            raise CalculationError(f"Portfolio forecast failed: {e}") from e

    def run_multiple_forecasts(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        horizon: int,
        methods: List[str],
        method_params: Optional[Dict[str, Dict[str, any]]] = None,
        out_of_sample: bool = False,
        training_ratio: float = 0.3,
    ) -> Dict[str, Dict[str, any]]:
        """
        Run multiple forecasting methods and return all results.

        Args:
            ticker: Ticker symbol
            start_date: Start date for training data
            end_date: End date for training data
            horizon: Number of days to forecast ahead
            methods: List of method names to run
            method_params: Dictionary mapping method name to parameters
            out_of_sample: If True, use out-of-sample testing
            training_ratio: Ratio for training period

        Returns:
            Dictionary mapping method name to forecast result
        """
        results = {}

        for method in methods:
            try:
                params = method_params.get(method, {}) if method_params else {}
                result = self.forecast_asset(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    horizon=horizon,
                    method=method,
                    method_params=params,
                    out_of_sample=out_of_sample,
                    training_ratio=training_ratio,
                )
                results[method] = result
            except Exception as e:
                logger.error(f"Error running {method} forecast: {e}", exc_info=True)
                results[method] = {
                    "success": False,
                    "message": str(e),
                }

        return results

    def create_ensemble(
        self,
        forecasts: Dict[str, Dict[str, any]],
        method: str = "weighted_average",
    ) -> Dict[str, any]:
        """
        Create ensemble forecast from multiple forecasts.

        Args:
            forecasts: Dictionary mapping method name to forecast result
            method: Ensemble method: 'weighted_average', 'simple_average', 'stacking'

        Returns:
            Ensemble forecast result
        """
        # Filter successful forecasts
        successful_forecasts = {
            k: v for k, v in forecasts.items() if v.get("success", False)
        }

        if len(successful_forecasts) < 2:
            raise ValueError("Need at least 2 successful forecasts for ensemble")

        # Get common forecast dates (use first forecast's dates)
        first_method = list(successful_forecasts.keys())[0]
        first_forecast = successful_forecasts[first_method]
        forecast_dates = pd.to_datetime(first_forecast["forecast_dates"])

        # Collect forecast values
        forecast_values_list = []
        weights = []

        for method_name, forecast_data in successful_forecasts.items():
            values = np.array(forecast_data["forecast_values"])
            if len(values) == len(forecast_dates):
                forecast_values_list.append(values)

                # Calculate weight based on MAPE (if available)
                if method == "weighted_average":
                    if "validation_metrics" in forecast_data:
                        mape = forecast_data["validation_metrics"].get("mape", np.nan)
                        if not np.isnan(mape) and mape > 0:
                            weights.append(1.0 / mape)
                        else:
                            weights.append(1.0)
                    else:
                        weights.append(1.0)
                else:
                    weights.append(1.0)

        if len(forecast_values_list) == 0:
            raise ValueError("No valid forecasts to combine")

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Combine forecasts
        if method == "weighted_average":
            ensemble_values = np.average(
                forecast_values_list, axis=0, weights=weights
            )
        else:  # simple_average
            ensemble_values = np.mean(forecast_values_list, axis=0)

        # Calculate final value and change
        if len(ensemble_values) == 0:
            raise ValueError("Ensemble values are empty")
        
        final_value = float(ensemble_values[-1])
        first_value = float(ensemble_values[0])
        
        # Calculate change_pct safely
        if first_value > 0 and np.isfinite(first_value) and np.isfinite(final_value):
            try:
                change_pct = ((final_value - first_value) / first_value) * 100.0
                if not np.isfinite(change_pct):
                    change_pct = 0.0
            except (ZeroDivisionError, OverflowError) as e:
                logger.warning(f"Error calculating ensemble change_pct: {e}")
                change_pct = 0.0
        else:
            change_pct = 0.0

        return {
            "method": "Ensemble",
            "forecast_dates": [
                date.isoformat() if hasattr(date, "isoformat") else str(date)
                for date in forecast_dates
            ],
            "forecast_values": ensemble_values.tolist(),
            "forecast_returns": (
                (np.diff(ensemble_values) / ensemble_values[:-1]).tolist()
                if len(ensemble_values) > 1 and np.all(ensemble_values[:-1] > 0)
                else [0.0] * (len(ensemble_values) - 1) if len(ensemble_values) > 1
                else []
            ),
            "confidence_intervals": {},
            "final_value": final_value,
            "change_pct": change_pct,
            "validation_metrics": None,
            "model_info": {
                "ensemble_method": method,
                "methods_used": list(successful_forecasts.keys()),
                "weights": weights.tolist(),
            },
            "residuals": None,
            "success": True,
            "message": f"Ensemble of {len(successful_forecasts)} methods",
        }

    def calculate_portfolio_prices(
        self,
        price_data: pd.DataFrame,
        positions: List,
    ) -> pd.Series:
        """
        Calculate weighted portfolio prices (cumulative portfolio value).

        Args:
            price_data: DataFrame with prices (dates as index, tickers as columns)
            positions: List of Position objects

        Returns:
            Series with portfolio prices (cumulative values)
        """
        # Calculate current weights based on first available date
        from services.data_service import DataService

        data_service = DataService()
        current_prices = {}
        for pos in positions:
            if pos.ticker != "CASH":
                current_price = data_service.fetch_current_price(pos.ticker)
                if current_price:
                    current_prices[pos.ticker] = current_price

        if not current_prices:
            raise ValueError("No valid prices for portfolio positions")

        # Calculate portfolio value
        portfolio_value = sum(
            current_prices.get(pos.ticker, 0) * pos.shares
            for pos in positions
            if pos.ticker != "CASH"
        )

        if portfolio_value == 0:
            raise ValueError("Portfolio value is zero")

        # Calculate weights
        weights = {}
        for pos in positions:
            if pos.ticker != "CASH" and pos.ticker in current_prices:
                weights[pos.ticker] = (current_prices[pos.ticker] * pos.shares) / portfolio_value

        # Calculate portfolio value over time
        portfolio_prices = pd.Series(index=price_data.index, dtype=float)
        
        for current_date in price_data.index:
            portfolio_value_at_date = 0.0
            
            # Calculate value for each position
            for pos in positions:
                ticker = pos.ticker
                shares = pos.shares
                
                if ticker == "CASH":
                    # CASH value is just the shares amount (cash amount)
                    portfolio_value_at_date += shares
                elif ticker in price_data.columns:
                    # Use .loc to access by column name (ticker)
                    try:
                        price_at_date = price_data.loc[current_date, ticker]
                        if pd.notna(price_at_date) and price_at_date > 0:
                            # Portfolio value = sum(price * shares)
                            # Shares already calculated based on weights and initial capital
                            portfolio_value_at_date += price_at_date * shares
                    except (KeyError, IndexError) as e:
                        logger.debug(
                            f"Could not get price for {ticker} on {current_date}: {e}"
                        )
                        continue
            
            portfolio_prices.loc[current_date] = portfolio_value_at_date
            
            # Log warning if value is zero (should not happen for valid portfolio)
            if portfolio_value_at_date == 0 and current_date == price_data.index[0]:
                logger.warning(
                    f"Portfolio value is zero on first date {current_date}. "
                    f"Tickers in data: {list(price_data.columns)}, "
                    f"Positions: {[p.ticker for p in positions]}"
                )

        # Ensure we have valid values
        portfolio_prices = portfolio_prices.dropna()
        if len(portfolio_prices) == 0:
            raise ValueError("Could not calculate portfolio prices")

        return portfolio_prices

    def _calculate_portfolio_returns(
        self,
        price_data: pd.DataFrame,
        positions: List,
    ) -> pd.Series:
        """
        Calculate weighted portfolio returns.

        Args:
            price_data: DataFrame with prices (dates as index, tickers as columns)
            positions: List of Position objects

        Returns:
            Series with portfolio returns
        """
        portfolio_prices = self.calculate_portfolio_prices(price_data, positions)
        return portfolio_prices.pct_change().dropna()

    def _create_forecast_period_from_validation(
        self,
        validation_forecast_values: np.ndarray,
        validation_forecast_dates: pd.DatetimeIndex,
        last_historical_price: float,
        last_historical_date: pd.Timestamp,
        horizon: int,
    ) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Create Forecast Period forecast based on Validation Period trend.
        
        Args:
            validation_forecast_values: Validation period forecast values
            validation_forecast_dates: Validation period forecast dates
            last_historical_price: Last historical price (end_date)
            last_historical_date: Last historical date (end_date)
            horizon: Number of days to forecast ahead
            
        Returns:
            Tuple of (forecast_period_values, forecast_period_dates)
        """
        if len(validation_forecast_values) <= 1 or horizon <= 0:
            return np.array([]), pd.DatetimeIndex([])
        
        # Calculate average daily return from validation period forecast
        # Safe division: avoid division by zero
        validation_forecast_safe = validation_forecast_values[:-1]
        # Filter zero and negative values
        valid_mask = validation_forecast_safe > 1e-10
        if np.any(valid_mask):
            validation_returns = np.diff(validation_forecast_values)[valid_mask] / validation_forecast_safe[valid_mask]
            avg_daily_return = np.mean(validation_returns)
            # Calculate volatility (standard deviation of returns) from validation period
            volatility = np.std(validation_returns)
            if not np.isfinite(volatility) or volatility <= 0:
                volatility = 0.01  # Default 1% volatility
        else:
            # Fallback: use zero return and small volatility
            avg_daily_return = 0.0
            volatility = 0.01  # Default 1% volatility
        
        # Get last price from Validation Period forecast
        last_validation_forecast_price = float(validation_forecast_values[-1])
        
        # Calculate difference between last historical price and last Validation Period forecast price
        # This is needed to "shift" the forecast so it starts from the last historical price
        price_offset = last_historical_price - last_validation_forecast_price
        
        # Create new forecast for forecast period
        # Start from last Validation Period forecast price + offset (to start from history)
        # but preserve the logic (trend) of Validation Period forecast
        # Do NOT add initial price - it will be added in forecast_charts.py for proper connection
        new_forecast_period_values = []
        current_price = last_validation_forecast_price + price_offset  # Start from history
        
        # Generate random shocks with volatility
        # Use local generator to avoid conflicts in parallel computations
        rng = np.random.default_rng(42)  # For reproducibility
        random_shocks = rng.normal(0, volatility, horizon)
        
        for i in range(horizon):
            # Apply average return + random shock with volatility
            # This preserves the logic (trend) of Validation Period forecast
            daily_return = avg_daily_return + random_shocks[i]
            current_price = current_price * (1 + daily_return)
            new_forecast_period_values.append(current_price)
        
        # Update dates - forecast period should start from end_date + 1
        # (without initial date, as it will be added in forecast_charts.py)
        if len(validation_forecast_dates) > 0:
            last_validation_date = pd.Timestamp(validation_forecast_dates[-1])
            if last_validation_date >= last_historical_date:
                # Validation dates already include last_historical_date, start from next day
                start_date = last_historical_date + pd.Timedelta(days=1)
            else:
                # Validation dates end earlier, start from next day after last_historical_date
                start_date = last_historical_date + pd.Timedelta(days=1)
        else:
            start_date = last_historical_date + pd.Timedelta(days=1)
        
        new_forecast_period_dates = pd.bdate_range(
            start=start_date,
            periods=horizon  # Without +1, as initial date will be added in forecast_charts.py
        )
        
        return np.array(new_forecast_period_values), new_forecast_period_dates

    def _get_forecaster_class(self, method: str):
        """Get forecaster class by method name."""
        method_map = {}

        # Simple models
        try:
            from core.forecasting_engine.simple.prophet import ProphetForecaster
            method_map["prophet"] = ProphetForecaster
        except ImportError:
            pass

        # Classical models
        try:
            from core.forecasting_engine.classical.arima import ARIMAForecaster
            method_map["arima"] = ARIMAForecaster
        except ImportError:
            pass

        try:
            from core.forecasting_engine.classical.garch import GARCHForecaster
            method_map["garch"] = GARCHForecaster
        except ImportError:
            pass

        try:
            from core.forecasting_engine.classical.arima_garch import (
                ARIMAGARCHForecaster,
            )
            method_map["arima_garch"] = ARIMAGARCHForecaster
            method_map["arima-garch"] = ARIMAGARCHForecaster
        except ImportError:
            pass

        # ML models
        try:
            from core.forecasting_engine.ml.svm_forecaster import SVMForecaster
            method_map["svm"] = SVMForecaster
            method_map["svr"] = SVMForecaster
        except ImportError:
            pass
        try:
            from core.forecasting_engine.ml.xgboost_forecaster import XGBoostForecaster
            method_map["xgboost"] = XGBoostForecaster
        except ImportError:
            pass

        try:
            from core.forecasting_engine.ml.random_forest import RandomForestForecaster
            method_map["random_forest"] = RandomForestForecaster
            method_map["randomforest"] = RandomForestForecaster  # Alternative name
        except ImportError:
            pass

        # Deep learning models
        try:
            from core.forecasting_engine.deep_learning.lstm import LSTMForecaster
            method_map["lstm"] = LSTMForecaster
        except ImportError:
            pass

        try:
            from core.forecasting_engine.deep_learning.tcn import TCNForecaster
            method_map["tcn"] = TCNForecaster
        except ImportError:
            pass

        try:
            from core.forecasting_engine.deep_learning.ssa_maemd_tcn import (
                SSAMAEEMDTCNForecaster,
            )
            method_map["ssa_maemd_tcn"] = SSAMAEEMDTCNForecaster
            method_map["ssa-maemd-tcn"] = SSAMAEEMDTCNForecaster
        except ImportError:
            pass

        forecaster_class = method_map.get(method.lower())
        if forecaster_class is None:
            raise ValueError(
                f"Forecaster class not found for method: {method}. "
                f"Available: {list(method_map.keys())}"
            )

        return forecaster_class

