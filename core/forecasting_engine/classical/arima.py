"""ARIMA forecaster implementation."""

import logging
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

try:
    from pmdarima import auto_arima
except ImportError:
    auto_arima = None

from core.exceptions import CalculationError
from core.forecasting_engine.base import BaseForecaster, ForecastResult
from core.forecasting_engine.utils import calculate_confidence_intervals

logger = logging.getLogger(__name__)


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA (AutoRegressive Integrated Moving Average) forecaster.

    Good for trend forecasting and short-term predictions.
    """

    def __init__(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None,
    ) -> None:
        """
        Initialize ARIMA forecaster.

        Args:
            prices: Historical prices series
            returns: Optional returns series (ARIMA works better on returns)
        """
        super().__init__(prices, returns)

    def _get_residuals(
        self, fitted_model, is_auto_arima: bool
    ) -> Optional[np.ndarray]:
        """
        Get residuals from fitted model.

        Handles both auto_arima and statsmodels ARIMA.

        Args:
            fitted_model: Fitted ARIMA model
            is_auto_arima: Whether model is from auto_arima

        Returns:
            Residuals array or None
        """
        try:
            if is_auto_arima:
                # pmdarima auto_arima: residuals are in arima_res_
                if (
                    hasattr(fitted_model, "arima_res_")
                    and hasattr(fitted_model.arima_res_, "resid")
                ):
                    residuals = fitted_model.arima_res_.resid
                    # Convert to numpy array if needed
                    if hasattr(residuals, "values"):
                        return residuals.values
                    return np.array(residuals)
            else:
                # statsmodels ARIMA: residuals are directly accessible
                if hasattr(fitted_model, "resid"):
                    residuals = fitted_model.resid
                    if hasattr(residuals, "values"):
                        return residuals.values
                    return np.array(residuals)
        except Exception as e:
            logger.warning(f"Could not extract residuals: {e}")
        return None

    def _get_confidence_intervals(
        self, fitted_model, horizon: int, is_auto_arima: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get confidence intervals from fitted model.

        Args:
            fitted_model: Fitted ARIMA model
            horizon: Forecast horizon
            is_auto_arima: Whether model is from auto_arima

        Returns:
            Tuple of (lower_95, upper_95) arrays
        """
        try:
            if is_auto_arima:
                # pmdarima auto_arima: use arima_res_ for get_forecast
                if hasattr(fitted_model, "arima_res_"):
                    forecast_obj = (
                        fitted_model.arima_res_.get_forecast(steps=horizon)
                    )
                else:
                    # Fallback: try direct access
                    forecast_obj = fitted_model.get_forecast(
                        steps=horizon
                    )
            else:
                # statsmodels ARIMA: direct access
                forecast_obj = fitted_model.get_forecast(
                    steps=horizon
                )

            conf_int = forecast_obj.conf_int()
            lower_95 = conf_int.iloc[:, 0].values
            upper_95 = conf_int.iloc[:, 1].values

            # Validate and clean
            lower_95 = np.nan_to_num(lower_95, nan=0.0, posinf=0.0, neginf=0.0)
            upper_95 = np.nan_to_num(upper_95, nan=0.0, posinf=0.0, neginf=0.0)

            return lower_95, upper_95
        except Exception as e:
            logger.warning(
                f"Could not get confidence intervals from model: {e}"
            )
            # Fallback: calculate using residuals
            residuals = self._get_residuals(fitted_model, is_auto_arima)
            forecast_values = self._get_forecast_values(
                fitted_model, horizon, is_auto_arima
            )
            ci_dict = calculate_confidence_intervals(
                forecast_values, residuals=residuals, confidence_level=0.95
            )
            lower = ci_dict.get("lower_95", np.zeros(horizon))
            upper = ci_dict.get("upper_95", np.zeros(horizon))
            return lower, upper

    def _get_forecast_values(
        self, fitted_model, horizon: int, is_auto_arima: bool
    ) -> np.ndarray:
        """
        Get forecast values from fitted model.

        Args:
            fitted_model: Fitted ARIMA model
            horizon: Forecast horizon
            is_auto_arima: Whether model is from auto_arima

        Returns:
            Forecast values array
        """
        try:
            if is_auto_arima:
                # pmdarima auto_arima: forecast returns pandas Series
                forecast_result = fitted_model.predict(n_periods=horizon)
            else:
                # statsmodels ARIMA: forecast returns pandas Series
                forecast_result = fitted_model.forecast(steps=horizon)

            # Convert to numpy array
            if hasattr(forecast_result, "values"):
                forecast_values = forecast_result.values
            elif isinstance(forecast_result, (list, tuple)):
                forecast_values = np.array(forecast_result)
            else:
                forecast_values = np.array(forecast_result)

            # Validate
            if np.any(np.isnan(forecast_values)) or np.any(
                np.isinf(forecast_values)
            ):
                logger.warning(
                    "ARIMA forecast contains NaN or Inf values, cleaning"
                )
                forecast_values = np.nan_to_num(
                    forecast_values, nan=0.0, posinf=0.0, neginf=0.0
                )

            return forecast_values
        except Exception as e:
            logger.error(f"Error getting forecast values: {e}")
            raise CalculationError(f"Could not generate forecast: {e}") from e

    def _convert_returns_to_prices(
        self,
        forecast_returns: np.ndarray,
        ci_lower: np.ndarray,
        ci_upper: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert returns forecast to prices forecast.

        Args:
            forecast_returns: Forecasted returns
            ci_lower: Lower confidence interval for returns
            ci_upper: Upper confidence interval for returns

        Returns:
            Tuple of (forecast_prices, ci_lower_prices, ci_upper_prices)
        """
        last_price = self.prices.iloc[-1]
        if last_price <= 0:
            raise CalculationError(f"Invalid last price: {last_price}")

        # Clamp returns to reasonable range to avoid extreme values
        # Use wider range: -0.99 to 0.99
        # (allows large moves, prevents overflow)
        forecast_returns_clamped = np.clip(forecast_returns, -0.99, 0.99)
        ci_lower_clamped = np.clip(ci_lower, -0.99, 0.99)
        ci_upper_clamped = np.clip(ci_upper, -0.99, 0.99)

        # Convert returns to cumulative prices
        # price_t = price_0 * cumprod(1 + returns)
        cumulative_returns = np.cumprod(1 + forecast_returns_clamped)
        forecast_prices = last_price * cumulative_returns

        # Convert confidence intervals
        ci_lower_prices = last_price * np.cumprod(1 + ci_lower_clamped)
        ci_upper_prices = last_price * np.cumprod(1 + ci_upper_clamped)

        # Ensure prices are positive
        forecast_prices = np.maximum(forecast_prices, 0.01 * last_price)
        ci_lower_prices = np.maximum(ci_lower_prices, 0.01 * last_price)
        ci_upper_prices = np.maximum(ci_upper_prices, 0.01 * last_price)

        return forecast_prices, ci_lower_prices, ci_upper_prices

    def forecast(
        self,
        horizon: int,
        auto: bool = True,
        p: Optional[int] = None,
        d: Optional[int] = None,
        q: Optional[int] = None,
        use_returns: bool = True,
        **kwargs,
    ) -> ForecastResult:
        """
        Generate forecast using ARIMA.

        Args:
            horizon: Number of periods to forecast
            auto: If True, use auto_arima to find best parameters
            p: AR order (if auto=False)
            d: Differencing order (if auto=False)
            q: MA order (if auto=False)
            use_returns: If True, forecast returns then convert to prices
            **kwargs: Additional ARIMA parameters

        Returns:
            ForecastResult with forecasted values
        """
        start_time = time.time()
        is_auto_arima = False

        try:
            # Use returns for better stationarity
            if use_returns:
                data = self.returns.dropna()
                if len(data) < 30:
                    # Fallback to prices if not enough returns
                    logger.warning(
                        "Not enough returns data, falling back to prices"
                    )
                    data = self.prices
                    use_returns = False
            else:
                data = self.prices

            # Check stationarity for prices
            if not use_returns:
                adf_result = adfuller(data.dropna())
                if adf_result[1] > 0.05:
                    logger.warning(
                        f"Data may not be stationary "
                        f"(p-value={adf_result[1]:.4f}). "
                        "Consider using returns or differencing."
                    )

            # Fit ARIMA model
            if auto and auto_arima is not None:
                # Use auto_arima to find best parameters
                model = auto_arima(
                    data,
                    start_p=0,
                    start_q=0,
                    max_p=5,
                    max_q=5,
                    d=None,  # Auto-detect
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    **kwargs,
                )
                order = model.order
                fitted_model = model
                is_auto_arima = True
            else:
                # Use manual parameters
                if p is None or d is None or q is None:
                    # Default values
                    p = p or 1
                    d = d or 1
                    q = q or 1

                order = (p, d, q)
                model = ARIMA(data, order=order, **kwargs)
                fitted_model = model.fit()
                is_auto_arima = False

            # Validate order format
            if not isinstance(order, (tuple, list)) or len(order) != 3:
                raise CalculationError(
                    f"Invalid order format: {order}. "
                    "Expected (p, d, q) tuple."
                )

            # Generate forecast
            forecast_values = self._get_forecast_values(
                fitted_model, horizon, is_auto_arima
            )

            # Get confidence intervals
            try:
                lower_95, upper_95 = self._get_confidence_intervals(
                    fitted_model, horizon, is_auto_arima
                )
            except Exception as e:
                logger.warning(
                    f"Could not get confidence intervals: {e}, "
                    "using fallback"
                )
                # Fallback: calculate using residuals
                residuals = self._get_residuals(fitted_model, is_auto_arima)
                ci_dict = calculate_confidence_intervals(
                    forecast_values, residuals=residuals, confidence_level=0.95
                )
                lower_95 = ci_dict.get("lower_95", np.zeros(horizon))
                upper_95 = ci_dict.get("upper_95", np.zeros(horizon))

            # If forecasting returns, convert to prices
            if use_returns:
                forecast_values, lower_95, upper_95 = (
                    self._convert_returns_to_prices(
                        forecast_values, lower_95, upper_95
                    )
                )
            else:
                # Direct price forecast - ensure positive values
                last_price = self.prices.iloc[-1]
                if np.any(forecast_values <= 0):
                    num_negative = np.sum(forecast_values <= 0)
                    logger.warning(
                        f"ARIMA forecast contains {num_negative} "
                        "non-positive values, replacing with last price"
                    )
                    # Only replace negative values, not all
                    negative_mask = forecast_values <= 0
                    forecast_values[negative_mask] = last_price
                    # Adjust CI for negative values
                    lower_95[negative_mask] = last_price * 0.9
                    upper_95[negative_mask] = last_price * 1.1

            # Create forecast dates
            last_date = self.prices.index[-1]
            forecast_dates = self._create_forecast_dates(horizon, last_date)

            # Calculate returns
            if len(forecast_values) > 1:
                # Avoid division by zero
                forecast_returns = np.diff(forecast_values) / np.maximum(
                    forecast_values[:-1], 1e-10
                )
                last_price = self.prices.iloc[-1]
                first_return = (forecast_values[0] - last_price) / max(
                    last_price, 1e-10
                )
                forecast_returns = np.insert(forecast_returns, 0, first_return)
            else:
                # Single value forecast
                last_price = self.prices.iloc[-1]
                first_return = (forecast_values[0] - last_price) / max(
                    last_price, 1e-10
                )
                forecast_returns = np.array([first_return])

            # Calculate change percentage
            change_pct = self._calculate_change_pct(
                forecast_values, self.prices.iloc[-1]
            )

            # Get residuals
            residuals = self._get_residuals(fitted_model, is_auto_arima)

            # Get model info
            aic = None
            bic = None
            try:
                if is_auto_arima:
                    # pmdarima: use methods
                    if hasattr(fitted_model, "aic"):
                        aic = float(fitted_model.aic())
                    if hasattr(fitted_model, "bic"):
                        bic = float(fitted_model.bic())
                    # Fallback: try arima_res_
                    if aic is None and hasattr(fitted_model, "arima_res_"):
                        if hasattr(fitted_model.arima_res_, "aic"):
                            aic = float(fitted_model.arima_res_.aic)
                        if hasattr(fitted_model.arima_res_, "bic"):
                            bic = float(fitted_model.arima_res_.bic)
                else:
                    # statsmodels: direct access
                    if hasattr(fitted_model, "aic"):
                        aic = float(fitted_model.aic)
                    if hasattr(fitted_model, "bic"):
                        bic = float(fitted_model.bic)
            except Exception as e:
                logger.warning(f"Could not extract AIC/BIC: {e}")

            training_time = time.time() - start_time

            # Format order for message
            if isinstance(order, (tuple, list)) and len(order) == 3:
                order_str = f"({order[0]},{order[1]},{order[2]})"
            else:
                order_str = str(order)

            return ForecastResult(
                method="ARIMA",
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
                    "order": order,
                    "auto": auto,
                    "use_returns": use_returns,
                    "aic": aic,
                    "bic": bic,
                    "training_time": training_time,
                },
                residuals=residuals,
                success=True,
                message=f"ARIMA{order_str} forecast completed",
            )

        except Exception as e:
            logger.error(f"ARIMA forecast failed: {e}", exc_info=True)
            raise CalculationError(f"ARIMA forecast failed: {e}") from e
