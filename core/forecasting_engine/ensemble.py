"""Ensemble forecasting models."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.exceptions import CalculationError
from core.forecasting_engine.base import BaseForecaster, ForecastResult
from core.forecasting_engine.utils import calculate_confidence_intervals

logger = logging.getLogger(__name__)


class EnsembleForecaster(BaseForecaster):
    """
    Ensemble forecaster that combines multiple forecasting methods.

    Supports multiple ensemble methods:
    - weighted_average: Weight by inverse MAPE or RMSE
    - simple_average: Equal weights for all models
    - median: Median of all forecasts (robust to outliers)
    - trimmed_mean: Mean after removing outliers
    """

    def __init__(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None,
        forecasters: Optional[List[BaseForecaster]] = None,
    ) -> None:
        """
        Initialize ensemble forecaster.

        Args:
            prices: Historical prices series
            returns: Optional returns series
            forecasters: List of forecaster instances to combine
        """
        super().__init__(prices, returns)
        self.forecasters = forecasters or []

    def forecast(
        self,
        horizon: int,
        method: str = "weighted_average",
        forecast_results: Optional[List[ForecastResult]] = None,
        weight_metric: str = "mape",
        trim_percent: float = 0.1,
        **kwargs,
    ) -> ForecastResult:
        """
        Generate ensemble forecast from multiple forecasters.

        Args:
            horizon: Number of periods to forecast
            method: Ensemble method:
                - 'weighted_average': Weight by inverse error metric
                - 'simple_average': Equal weights for all models
                - 'median': Median of all forecasts (robust to outliers)
                - 'trimmed_mean': Mean after removing outliers
            forecast_results: Optional list of ForecastResult objects to combine
            weight_metric: Metric for weighting ('mape', 'rmse', 'mae')
            trim_percent: Percentage to trim for trimmed_mean (0.0-0.5)
            **kwargs: Additional parameters passed to individual forecasters

        Returns:
            ForecastResult with ensemble forecast
        """
        if forecast_results is None:
            # Run forecasts from all forecasters
            if not self.forecasters:
                raise CalculationError("No forecasters provided for ensemble")

            forecast_results = []
            for forecaster in self.forecasters:
                try:
                    result = forecaster.forecast(horizon=horizon, **kwargs)
                    forecast_results.append(result)
                except Exception as e:
                    forecaster_name = (
                        forecaster.__class__.__name__
                        if hasattr(forecaster, "__class__")
                        else "Unknown"
                    )
                    logger.warning(
                        f"Forecaster {forecaster_name} failed: {e}",
                        exc_info=False,
                    )

        # Filter successful forecasts
        successful_forecasts = [
            r for r in forecast_results if r.success
        ]

        if len(successful_forecasts) < 2:
            raise CalculationError(
                f"Need at least 2 successful forecasts for ensemble, got {len(successful_forecasts)}"
            )

        # Get common forecast dates (use first forecast's dates)
        first_forecast = successful_forecasts[0]
        forecast_dates = first_forecast.forecast_dates

        # Collect forecast values and align lengths
        forecast_values_list = []
        weights = []
        methods_used = []

        for forecast_result in successful_forecasts:
            values = forecast_result.forecast_values

            # Align lengths: truncate or pad to match forecast_dates
            if len(values) > len(forecast_dates):
                values = values[: len(forecast_dates)]
            elif len(values) < len(forecast_dates):
                # Pad with last value
                last_val = values[-1] if len(values) > 0 else self.prices.iloc[-1]
                padding = np.full(len(forecast_dates) - len(values), last_val)
                values = np.concatenate([values, padding])

            if len(values) == len(forecast_dates):
                forecast_values_list.append(values)
                methods_used.append(forecast_result.method)

                # Calculate weight based on error metric (if available)
                if method == "weighted_average":
                    weight = 1.0
                    if forecast_result.validation_metrics:
                        metric_value = forecast_result.validation_metrics.get(
                            weight_metric.lower(), np.nan
                        )
                        if not np.isnan(metric_value) and metric_value > 0:
                            # Inverse metric: lower error = higher weight
                            weight = 1.0 / metric_value
                    weights.append(weight)
                else:
                    weights.append(1.0)

        if len(forecast_values_list) == 0:
            raise CalculationError("No valid forecasts to combine")

        # Convert to numpy array for easier manipulation
        forecast_values_array = np.array(forecast_values_list)

        # Combine forecasts based on method
        if method == "weighted_average":
            # Normalize weights
            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                # Fallback to equal weights if all weights are zero
                weights = np.ones(len(weights)) / len(weights)
            ensemble_values = np.average(
                forecast_values_array, axis=0, weights=weights
            )
        elif method == "median":
            # Median is robust to outliers
            ensemble_values = np.median(forecast_values_array, axis=0)
        elif method == "trimmed_mean":
            # Trim extreme values before averaging
            trim_count = int(len(forecast_values_list) * trim_percent)
            if trim_count > 0:
                # Sort each timestep and trim
                ensemble_values = []
                for t in range(forecast_values_array.shape[1]):
                    values_at_t = forecast_values_array[:, t]
                    sorted_values = np.sort(values_at_t)
                    trimmed = sorted_values[trim_count:-trim_count] if trim_count > 0 else sorted_values
                    ensemble_values.append(np.mean(trimmed))
                ensemble_values = np.array(ensemble_values)
            else:
                ensemble_values = np.mean(forecast_values_array, axis=0)
        else:  # simple_average
            ensemble_values = np.mean(forecast_values_array, axis=0)

        # Calculate returns
        ensemble_returns = np.diff(ensemble_values) / ensemble_values[:-1]
        first_return = (ensemble_values[0] - self.prices.iloc[-1]) / self.prices.iloc[-1]
        ensemble_returns = np.insert(ensemble_returns, 0, first_return)

        # Calculate change percentage
        change_pct = self._calculate_change_pct(ensemble_values, self.prices.iloc[-1])

        # Calculate confidence intervals (combine individual CIs)
        confidence_intervals = {}
        ci_upper_list = []
        ci_lower_list = []

        for forecast_result in successful_forecasts:
            if (
                hasattr(forecast_result, "confidence_intervals")
                and forecast_result.confidence_intervals
            ):
                ci = forecast_result.confidence_intervals
                if "upper_95" in ci and "lower_95" in ci:
                    upper = ci["upper_95"]
                    lower = ci["lower_95"]
                    # Align lengths
                    if len(upper) > len(forecast_dates):
                        upper = upper[: len(forecast_dates)]
                    elif len(upper) < len(forecast_dates):
                        last_val = upper[-1] if len(upper) > 0 else ensemble_values[-1]
                        padding = np.full(len(forecast_dates) - len(upper), last_val)
                        upper = np.concatenate([upper, padding])

                    if len(lower) > len(forecast_dates):
                        lower = lower[: len(forecast_dates)]
                    elif len(lower) < len(forecast_dates):
                        last_val = lower[-1] if len(lower) > 0 else ensemble_values[0]
                        padding = np.full(len(forecast_dates) - len(lower), last_val)
                        lower = np.concatenate([lower, padding])

                    if len(upper) == len(forecast_dates) and len(lower) == len(
                        forecast_dates
                    ):
                        ci_upper_list.append(upper)
                        ci_lower_list.append(lower)

        if ci_upper_list and ci_lower_list:
            # Combine confidence intervals (use weighted average if weights available)
            if method == "weighted_average" and len(weights) == len(ci_upper_list):
                upper_95_ensemble = np.average(
                    np.array(ci_upper_list), axis=0, weights=weights
                )
                lower_95_ensemble = np.average(
                    np.array(ci_lower_list), axis=0, weights=weights
                )
            else:
                # Simple average or median
                if method == "median":
                    upper_95_ensemble = np.median(np.array(ci_upper_list), axis=0)
                    lower_95_ensemble = np.median(np.array(ci_lower_list), axis=0)
                else:
                    upper_95_ensemble = np.mean(np.array(ci_upper_list), axis=0)
                    lower_95_ensemble = np.mean(np.array(ci_lower_list), axis=0)

            confidence_intervals = {
                "upper_95": upper_95_ensemble,
                "lower_95": lower_95_ensemble,
            }

        # Combine residuals (if available)
        residuals_list = []
        for forecast_result in successful_forecasts:
            if (
                hasattr(forecast_result, "residuals")
                and forecast_result.residuals is not None
            ):
                residuals = forecast_result.residuals
                # Filter valid residuals
                valid_residuals = residuals[
                    np.isfinite(residuals) & ~np.isnan(residuals)
                ]
                if len(valid_residuals) > 0:
                    residuals_list.append(valid_residuals)

        # Combine residuals (use median for robustness)
        ensemble_residuals = None
        if residuals_list:
            # Align lengths by taking minimum length
            min_len = min(len(r) for r in residuals_list)
            aligned_residuals = [r[:min_len] for r in residuals_list]
            # Use median for robustness
            ensemble_residuals = np.median(np.array(aligned_residuals), axis=0)

        # Calculate weights for info (normalize if needed)
        weights_info = weights
        if isinstance(weights, np.ndarray) and method == "weighted_average":
            weights_info = weights.tolist()

        return ForecastResult(
            method="Ensemble",
            forecast_dates=forecast_dates,
            forecast_values=ensemble_values,
            forecast_returns=ensemble_returns,
            confidence_intervals=confidence_intervals,
            final_value=float(ensemble_values[-1]),
            change_pct=change_pct,
            validation_metrics=None,  # Would need to evaluate separately
            model_info={
                "ensemble_method": method,
                "methods_used": methods_used,
                "weights": weights_info,
                "weight_metric": weight_metric if method == "weighted_average" else None,
                "num_forecasters": len(successful_forecasts),
                "trim_percent": trim_percent if method == "trimmed_mean" else None,
            },
            residuals=ensemble_residuals,
            success=True,
            message=(
                f"Ensemble of {len(successful_forecasts)} methods "
                f"({', '.join(methods_used)}) using {method}"
            ),
        )

