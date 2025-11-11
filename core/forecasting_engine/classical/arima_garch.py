"""ARIMA-GARCH combined forecaster implementation."""

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

try:
    from arch import arch_model
except ImportError:
    arch_model = None

try:
    from pmdarima import auto_arima
except ImportError:
    auto_arima = None

from core.exceptions import CalculationError
from core.forecasting_engine.base import BaseForecaster, ForecastResult

logger = logging.getLogger(__name__)


class ARIMAGARCHForecaster(BaseForecaster):
    """
    Combined ARIMA-GARCH forecaster.

    ARIMA models the mean (expected returns), while GARCH models
    the volatility (conditional variance) of the residuals.
    This combination provides more accurate forecasts for financial time series.
    """

    def __init__(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None,
    ) -> None:
        """
        Initialize ARIMA-GARCH forecaster.

        Args:
            prices: Historical prices series
            returns: Optional returns series (calculated if not provided)
        """
        if arch_model is None:
            raise ImportError(
                "arch library not installed. Install with: pip install arch"
            )

        super().__init__(prices, returns)

    def forecast(
        self,
        horizon: int,
        arima_auto: bool = True,
        arima_p: Optional[int] = None,
        arima_d: Optional[int] = None,
        arima_q: Optional[int] = None,
        garch_p: int = 1,
        garch_q: int = 1,
        garch_dist: str = "normal",
        **kwargs,
    ) -> ForecastResult:
        """
        Generate forecast using combined ARIMA-GARCH model.

        Process:
        1. Fit ARIMA to returns (models mean)
        2. Fit GARCH to ARIMA residuals (models volatility)
        3. Forecast mean with ARIMA
        4. Forecast volatility with GARCH
        5. Combine: forecast = mean ± volatility

        Args:
            horizon: Number of periods to forecast
            arima_auto: If True, use auto_arima for ARIMA parameters
            arima_p: AR order for ARIMA (if arima_auto=False)
            arima_d: Differencing order for ARIMA (if arima_auto=False)
            arima_q: MA order for ARIMA (if arima_auto=False)
            garch_p: GARCH order (number of ARCH terms)
            garch_q: GARCH order (number of GARCH terms)
            garch_dist: Distribution for GARCH residuals ('normal', 't', 'skewt')
            **kwargs: Additional parameters

        Returns:
            ForecastResult with forecasted values
        """
        if arch_model is None:
            raise ImportError("arch library not installed")

        start_time = time.time()

        try:
            # ARIMA-GARCH works with returns
            returns_data = self.returns.dropna()

            if len(returns_data) < 50:
                raise CalculationError(
                    f"ARIMA-GARCH requires at least 50 data points, "
                    f"got {len(returns_data)}"
                )

            # Validate returns data
            if returns_data.isna().any() or np.any(np.isinf(returns_data)):
                logger.warning(
                    "ARIMA-GARCH: Returns contain NaN or Inf values, "
                    "filtering..."
                )
                returns_data = returns_data[
                    ~(returns_data.isna() | np.isinf(returns_data))
                ]
                if len(returns_data) < 50:
                    raise CalculationError(
                        "ARIMA-GARCH: Not enough valid returns after filtering"
                    )

            # Step 1: Fit ARIMA model to returns (models mean)
            logger.debug("Fitting ARIMA model for mean returns...")
            try:
                if arima_auto and auto_arima is not None:
                    arima_model = auto_arima(
                        returns_data,
                        start_p=0,
                        start_q=0,
                        max_p=5,
                        max_q=5,
                        d=None,
                        seasonal=False,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action="ignore",
                    )
                    arima_order = arima_model.order
                    arima_fitted = arima_model
                    is_auto_arima = True
                else:
                    # Manual ARIMA parameters
                    if arima_p is None or arima_d is None or arima_q is None:
                        arima_p = arima_p or 1
                        arima_d = arima_d or 0
                        arima_q = arima_q or 1

                    arima_order = (arima_p, arima_d, arima_q)
                    arima_model = ARIMA(returns_data, order=arima_order)
                    arima_fitted = arima_model.fit()
                    is_auto_arima = False

                # Validate order
                if not isinstance(arima_order, (tuple, list)) or len(
                    arima_order
                ) != 3:
                    raise CalculationError(
                        f"Invalid ARIMA order: {arima_order}"
                    )

            except Exception as e:
                raise CalculationError(
                    f"ARIMA model fitting failed: {e}"
                ) from e

            # Get ARIMA residuals for GARCH
            try:
                if is_auto_arima:
                    if hasattr(arima_fitted, "arima_res_"):
                        arima_residuals_raw = arima_fitted.arima_res_.resid
                    else:
                        arima_residuals_raw = arima_fitted.resid
                else:
                    arima_residuals_raw = arima_fitted.resid

                # Convert to numpy array first
                if hasattr(arima_residuals_raw, "values"):
                    arima_residuals_array = arima_residuals_raw.values
                else:
                    arima_residuals_array = np.array(arima_residuals_raw)

                # Create Series with proper index alignment
                # Residuals length may be less than returns_data due to differencing
                # Align with the end of returns_data
                residuals_len = len(arima_residuals_array)
                if residuals_len <= len(returns_data):
                    # Use last N indices from returns_data
                    residual_index = returns_data.index[-residuals_len:]
                else:
                    # If residuals longer (shouldn't happen), truncate
                    logger.warning(
                        f"ARIMA residuals length ({residuals_len}) > "
                        f"returns length ({len(returns_data)}), truncating"
                    )
                    arima_residuals_array = arima_residuals_array[-len(returns_data):]
                    residual_index = returns_data.index

                arima_residuals = pd.Series(
                    arima_residuals_array, index=residual_index
                )
                arima_residuals = arima_residuals.dropna()

            except Exception as e:
                raise CalculationError(
                    f"Could not extract ARIMA residuals: {e}"
                ) from e

            if len(arima_residuals) < 30:
                raise CalculationError(
                    f"Not enough ARIMA residuals for GARCH: {len(arima_residuals)}"
                )

            # Step 2: Fit GARCH model to ARIMA residuals (models volatility)
            logger.debug("Fitting GARCH model for volatility...")
            try:
                # Filter out kwargs that are not valid for arch_model
                # Remove auto_arima and other ARIMA-specific parameters
                garch_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k not in ['auto_arima', 'arima_auto', 'arima_p', 'arima_d', 'arima_q']
                }
                garch_model = arch_model(
                    arima_residuals,
                    vol="Garch",
                    p=garch_p,
                    q=garch_q,
                    mean="Zero",  # Mean already modeled by ARIMA
                    dist=garch_dist,
                    **garch_kwargs,
                )
                garch_fitted = garch_model.fit(disp="off")
            except Exception as e:
                raise CalculationError(
                    f"GARCH model fitting failed: {e}"
                ) from e

            # Step 3: Forecast mean returns with ARIMA
            logger.debug("Forecasting mean returns with ARIMA...")
            try:
                if is_auto_arima:
                    arima_forecast = arima_fitted.predict(n_periods=horizon)
                else:
                    arima_forecast = arima_fitted.forecast(steps=horizon)

                # Convert to numpy array
                if hasattr(arima_forecast, "values"):
                    forecast_mean_returns = arima_forecast.values
                elif isinstance(arima_forecast, (list, tuple)):
                    forecast_mean_returns = np.array(arima_forecast)
                else:
                    forecast_mean_returns = np.array(arima_forecast)

                # Validate
                if len(forecast_mean_returns) != horizon:
                    if len(forecast_mean_returns) > horizon:
                        forecast_mean_returns = forecast_mean_returns[:horizon]
                    else:
                        last_val = (
                            forecast_mean_returns[-1]
                            if len(forecast_mean_returns) > 0
                            else 0.0
                        )
                        forecast_mean_returns = np.append(
                            forecast_mean_returns,
                            np.full(horizon - len(forecast_mean_returns), last_val),
                        )

            except Exception as e:
                raise CalculationError(
                    f"ARIMA forecast failed: {e}"
                ) from e

            # Step 4: Forecast volatility with GARCH
            logger.debug("Forecasting volatility with GARCH...")
            try:
                garch_forecast = garch_fitted.forecast(
                    horizon=horizon, reindex=False
                )
            except Exception as e:
                raise CalculationError(
                    f"GARCH volatility forecast failed: {e}"
                ) from e

            # Extract forecasted volatility
            forecast_volatility = None
            if hasattr(garch_forecast, "variance"):
                try:
                    variance_df = garch_forecast.variance
                    if isinstance(variance_df, pd.DataFrame):
                        if len(variance_df) > 0:
                            forecast_variance = variance_df.iloc[-1].values
                        else:
                            forecast_variance = variance_df.iloc[:, 0].values
                    elif hasattr(variance_df, "values"):
                        forecast_variance = variance_df.values
                        if forecast_variance.ndim == 2:
                            forecast_variance = forecast_variance[-1, :]
                    else:
                        forecast_variance = np.array(variance_df)
                        if forecast_variance.ndim == 2:
                            forecast_variance = forecast_variance[-1, :]

                    # Ensure correct length
                    if len(forecast_variance) != horizon:
                        if len(forecast_variance) > horizon:
                            forecast_variance = forecast_variance[:horizon]
                        else:
                            last_val = (
                                forecast_variance[-1]
                                if len(forecast_variance) > 0
                                else 0.01
                            )
                            forecast_variance = np.append(
                                forecast_variance,
                                np.full(
                                    horizon - len(forecast_variance), last_val
                                ),
                            )

                    forecast_volatility = np.sqrt(forecast_variance)
                except Exception as e:
                    logger.warning(
                        f"Error extracting GARCH variance: {e}, using fallback"
                    )
                    forecast_volatility = None

            if forecast_volatility is None:
                # Fallback: use conditional volatility
                try:
                    conditional_vol = garch_fitted.conditional_volatility
                    avg_vol = float(conditional_vol[-30:].mean())
                    forecast_volatility = np.full(horizon, avg_vol)
                except Exception as e:
                    logger.warning(
                        f"Could not use conditional volatility: {e}, "
                        "using historical"
                    )
                    hist_vol = float(arima_residuals.std())
                    forecast_volatility = np.full(horizon, hist_vol)

            # Validate volatility
            if np.any(np.isnan(forecast_volatility)) or np.any(
                np.isinf(forecast_volatility)
            ):
                logger.warning(
                    "ARIMA-GARCH forecast volatility contains NaN or Inf, "
                    "cleaning"
                )
                forecast_volatility = np.nan_to_num(
                    forecast_volatility, nan=0.01, posinf=0.1, neginf=0.01
                )

            # Step 5: Combine ARIMA mean and GARCH volatility
            # Forecast returns = ARIMA mean (volatility is used for CI)
            forecast_returns = forecast_mean_returns.copy()

            # Clamp returns to reasonable range
            forecast_returns = np.clip(forecast_returns, -0.99, 0.99)

            # Convert to prices
            last_price = float(self.prices.iloc[-1])
            cumulative_returns = np.cumprod(1 + forecast_returns)
            forecast_values = last_price * cumulative_returns

            # Confidence intervals: mean ± z_score * volatility
            z_score = 1.96  # 95% confidence interval
            upper_returns = forecast_returns + z_score * forecast_volatility
            lower_returns = forecast_returns - z_score * forecast_volatility

            # Clamp returns CI
            upper_returns = np.clip(upper_returns, -0.99, 0.99)
            lower_returns = np.clip(lower_returns, -0.99, 0.99)

            # Convert to prices CI
            upper_prices = last_price * np.cumprod(1 + upper_returns)
            lower_prices = last_price * np.cumprod(1 + lower_returns)

            # Ensure prices are positive
            forecast_values = np.maximum(forecast_values, 0.01 * last_price)
            upper_prices = np.maximum(upper_prices, 0.01 * last_price)
            lower_prices = np.maximum(lower_prices, 0.01 * last_price)

            # Create forecast dates
            last_date = self.prices.index[-1]
            forecast_dates = self._create_forecast_dates(horizon, last_date)

            # Calculate forecast returns array
            if len(forecast_values) > 1:
                forecast_returns_array = np.diff(forecast_values) / np.maximum(
                    forecast_values[:-1], 1e-10
                )
                first_return = (forecast_values[0] - last_price) / max(
                    last_price, 1e-10
                )
                forecast_returns_array = np.insert(
                    forecast_returns_array, 0, first_return
                )
            else:
                first_return = (forecast_values[0] - last_price) / max(
                    last_price, 1e-10
                )
                forecast_returns_array = np.array([first_return])

            # Calculate change percentage
            change_pct = self._calculate_change_pct(forecast_values, last_price)

            # Get residuals (standardized GARCH residuals)
            try:
                garch_residuals = (
                    garch_fitted.resid / garch_fitted.conditional_volatility
                )
                residuals = garch_residuals.dropna().values
            except Exception as e:
                logger.warning(
                    f"Could not extract ARIMA-GARCH residuals: {e}"
                )
                residuals = None

            # Get model info
            arima_aic = None
            arima_bic = None
            garch_aic = None
            garch_bic = None

            try:
                if is_auto_arima:
                    if hasattr(arima_fitted, "aic"):
                        arima_aic = float(arima_fitted.aic())
                    if hasattr(arima_fitted, "bic"):
                        arima_bic = float(arima_fitted.bic())
                    if arima_aic is None and hasattr(arima_fitted, "arima_res_"):
                        if hasattr(arima_fitted.arima_res_, "aic"):
                            arima_aic = float(arima_fitted.arima_res_.aic)
                        if hasattr(arima_fitted.arima_res_, "bic"):
                            arima_bic = float(arima_fitted.arima_res_.bic)
                else:
                    if hasattr(arima_fitted, "aic"):
                        arima_aic = float(arima_fitted.aic)
                    if hasattr(arima_fitted, "bic"):
                        arima_bic = float(arima_fitted.bic)
            except Exception as e:
                logger.warning(f"Could not extract ARIMA AIC/BIC: {e}")

            try:
                if hasattr(garch_fitted, "aic"):
                    garch_aic = float(garch_fitted.aic)
                if hasattr(garch_fitted, "bic"):
                    garch_bic = float(garch_fitted.bic)
            except Exception as e:
                logger.warning(f"Could not extract GARCH AIC/BIC: {e}")

            # Get parameters
            arima_params = None
            garch_params = None
            try:
                if hasattr(arima_fitted, "params"):
                    if is_auto_arima and hasattr(arima_fitted, "arima_res_"):
                        arima_params = arima_fitted.arima_res_.params.to_dict()
                    else:
                        arima_params = arima_fitted.params.to_dict()
            except Exception as e:
                logger.warning(f"Could not extract ARIMA parameters: {e}")

            try:
                if hasattr(garch_fitted, "params"):
                    garch_params = garch_fitted.params.to_dict()
            except Exception as e:
                logger.warning(f"Could not extract GARCH parameters: {e}")

            training_time = time.time() - start_time

            # Format ARIMA order for message
            if isinstance(arima_order, (tuple, list)) and len(arima_order) == 3:
                arima_order_str = f"({arima_order[0]},{arima_order[1]},{arima_order[2]})"
            else:
                arima_order_str = str(arima_order)

            return ForecastResult(
                method="ARIMA-GARCH",
                forecast_dates=forecast_dates,
                forecast_values=forecast_values,
                forecast_returns=forecast_returns_array,
                confidence_intervals={
                    "upper_95": upper_prices,
                    "lower_95": lower_prices,
                },
                final_value=float(forecast_values[-1]),
                change_pct=change_pct,
                model_info={
                    "arima_order": arima_order,
                    "arima_auto": arima_auto,
                    "arima_aic": arima_aic,
                    "arima_bic": arima_bic,
                    "arima_params": arima_params,
                    "garch_p": garch_p,
                    "garch_q": garch_q,
                    "garch_dist": garch_dist,
                    "garch_aic": garch_aic,
                    "garch_bic": garch_bic,
                    "garch_params": garch_params,
                    "avg_volatility": float(forecast_volatility.mean()),
                    "training_time": training_time,
                },
                residuals=residuals,
                success=True,
                message=(
                    f"ARIMA{arima_order_str}-GARCH({garch_p},{garch_q}) "
                    "forecast completed"
                ),
            )

        except Exception as e:
            logger.error(
                f"ARIMA-GARCH forecast failed: {e}", exc_info=True
            )
            raise CalculationError(f"ARIMA-GARCH forecast failed: {e}") from e

