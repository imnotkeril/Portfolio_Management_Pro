"""GARCH forecaster implementation for volatility forecasting."""

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

try:
    from arch import arch_model
except ImportError:
    arch_model = None

from core.exceptions import CalculationError
from core.forecasting_engine.base import BaseForecaster, ForecastResult
from core.forecasting_engine.utils import calculate_confidence_intervals

logger = logging.getLogger(__name__)


class GARCHForecaster(BaseForecaster):
    """
    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) forecaster.

    Used for volatility forecasting. GARCH models the conditional variance
    (volatility) of returns, which is then used to forecast prices.
    """

    def __init__(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None,
    ) -> None:
        """
        Initialize GARCH forecaster.

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
        p: int = 1,
        q: int = 1,
        mean: str = "Zero",
        dist: str = "normal",
        **kwargs,
    ) -> ForecastResult:
        """
        Generate forecast using GARCH model.

        GARCH forecasts volatility, which is then converted to price forecast
        using mean returns and volatility.

        Args:
            horizon: Number of periods to forecast
            p: GARCH order (number of ARCH terms)
            q: GARCH order (number of GARCH terms)
            mean: Mean model ('Zero', 'Constant', 'AR', 'LS')
            dist: Distribution for residuals ('normal', 't', 'skewt')
            **kwargs: Additional GARCH parameters

        Returns:
            ForecastResult with forecasted values
        """
        if arch_model is None:
            raise ImportError("arch library not installed")

        start_time = time.time()

        try:
            # GARCH works with returns, not prices
            returns_data = self.returns.dropna()

            if len(returns_data) < 50:
                raise CalculationError(
                    f"GARCH requires at least 50 data points, "
                    f"got {len(returns_data)}"
                )

            # Validate returns data
            if returns_data.isna().any() or np.any(np.isinf(returns_data)):
                logger.warning(
                    "GARCH: Returns contain NaN or Inf values, filtering..."
                )
                returns_data = returns_data[
                    ~(returns_data.isna() | np.isinf(returns_data))
                ]
                if len(returns_data) < 50:
                    raise CalculationError(
                        "GARCH: Not enough valid returns after filtering"
                    )

            # Fit GARCH model
            try:
                model = arch_model(
                    returns_data,
                    vol="Garch",
                    p=p,
                    q=q,
                    mean=mean,
                    dist=dist,
                    **kwargs,
                )
                model_fit = model.fit(disp="off")
            except Exception as e:
                raise CalculationError(
                    f"GARCH model fitting failed: {e}"
                ) from e

            # Forecast volatility
            try:
                forecast_result = model_fit.forecast(
                    horizon=horizon, reindex=False
                )
            except Exception as e:
                raise CalculationError(
                    f"GARCH volatility forecast failed: {e}"
                ) from e

            # Extract forecasted volatility (standard deviation)
            # forecast_result.variance is a DataFrame with forecasted variance
            forecast_volatility = None
            if hasattr(forecast_result, "variance"):
                try:
                    variance_df = forecast_result.variance
                    # variance_df is typically a DataFrame with columns for each step
                    # Get the last row (most recent forecast) or first column
                    if isinstance(variance_df, pd.DataFrame):
                        # If DataFrame, get the last row (forecast from last date)
                        if len(variance_df) > 0:
                            forecast_variance = variance_df.iloc[-1].values
                        else:
                            # If empty, try first column
                            forecast_variance = variance_df.iloc[:, 0].values
                    elif hasattr(variance_df, "values"):
                        forecast_variance = variance_df.values
                        if forecast_variance.ndim == 2:
                            forecast_variance = forecast_variance[-1, :]
                    else:
                        forecast_variance = np.array(variance_df)
                        if forecast_variance.ndim == 2:
                            forecast_variance = forecast_variance[-1, :]

                    # Ensure we have the right length
                    if len(forecast_variance) != horizon:
                        if len(forecast_variance) > horizon:
                            forecast_variance = forecast_variance[:horizon]
                        else:
                            # Pad with last value
                            last_val = forecast_variance[-1] if len(
                                forecast_variance
                            ) > 0 else 0.01
                            forecast_variance = np.append(
                                forecast_variance,
                                np.full(
                                    horizon - len(forecast_variance), last_val
                                ),
                            )

                    forecast_volatility = np.sqrt(forecast_variance)
                except Exception as e:
                    logger.warning(
                        f"Error extracting forecast variance: {e}, "
                        "using fallback"
                    )
                    forecast_volatility = None

            if forecast_volatility is None:
                # Fallback: use conditional volatility from model
                logger.warning(
                    "Could not extract forecast variance, "
                    "using conditional volatility"
                )
                try:
                    conditional_vol = model_fit.conditional_volatility
                    avg_vol = float(conditional_vol[-30:].mean())
                    forecast_volatility = np.full(horizon, avg_vol)
                except Exception as e:
                    logger.warning(
                        f"Could not use conditional volatility: {e}, "
                        "using default"
                    )
                    # Last resort: use historical returns volatility
                    hist_vol = float(returns_data.std())
                    forecast_volatility = np.full(horizon, hist_vol)

            # Validate volatility
            if np.any(np.isnan(forecast_volatility)) or np.any(
                np.isinf(forecast_volatility)
            ):
                logger.warning(
                    "GARCH forecast volatility contains NaN or Inf, cleaning"
                )
                forecast_volatility = np.nan_to_num(
                    forecast_volatility, nan=0.01, posinf=0.1, neginf=0.01
                )

            # Convert volatility forecast to price forecast
            # Strategy: Use mean return + volatility to generate price path
            # For GARCH, we forecast volatility, not returns directly
            # We use mean return as the expected return, with volatility as uncertainty
            mean_return = float(returns_data.mean())

            # If mean model is not Zero, try to get forecasted mean
            # Otherwise use historical mean
            if mean != "Zero" and hasattr(forecast_result, "mean"):
                try:
                    mean_forecast = forecast_result.mean
                    if isinstance(mean_forecast, pd.DataFrame):
                        forecast_mean_returns = mean_forecast.iloc[-1].values
                    elif hasattr(mean_forecast, "values"):
                        forecast_mean_returns = mean_forecast.values
                        if forecast_mean_returns.ndim == 2:
                            forecast_mean_returns = forecast_mean_returns[-1, :]
                    else:
                        forecast_mean_returns = np.array(mean_forecast)
                        if forecast_mean_returns.ndim == 2:
                            forecast_mean_returns = forecast_mean_returns[-1, :]

                    # Ensure correct length
                    if len(forecast_mean_returns) == horizon:
                        forecast_returns = forecast_mean_returns
                    else:
                        # Use mean return as fallback
                        forecast_returns = np.full(horizon, mean_return)
                except Exception as e:
                    logger.debug(
                        f"Could not extract forecast mean: {e}, "
                        "using historical mean"
                    )
                    forecast_returns = np.full(horizon, mean_return)
            else:
                # Use historical mean return
                forecast_returns = np.full(horizon, mean_return)

            last_price = float(self.prices.iloc[-1])

            # Generate price forecast from returns
            # price_t = price_0 * cumprod(1 + returns)
            cumulative_returns = np.cumprod(1 + forecast_returns)
            forecast_values = last_price * cumulative_returns

            # Use volatility for confidence intervals
            # Upper CI: forecast_return + 1.96 * volatility
            # Lower CI: forecast_return - 1.96 * volatility
            z_score = 1.96  # 95% confidence interval
            upper_returns = forecast_returns + z_score * forecast_volatility
            lower_returns = forecast_returns - z_score * forecast_volatility

            # Clamp returns to reasonable range
            upper_returns = np.clip(upper_returns, -0.99, 0.99)
            lower_returns = np.clip(lower_returns, -0.99, 0.99)

            # Convert returns CI to prices CI
            upper_prices = last_price * np.cumprod(1 + upper_returns)
            lower_prices = last_price * np.cumprod(1 + lower_returns)

            # Ensure prices are positive
            forecast_values = np.maximum(forecast_values, 0.01 * last_price)
            upper_prices = np.maximum(upper_prices, 0.01 * last_price)
            lower_prices = np.maximum(lower_prices, 0.01 * last_price)

            # Create forecast dates
            last_date = self.prices.index[-1]
            forecast_dates = self._create_forecast_dates(horizon, last_date)

            # Calculate forecast returns (for ForecastResult)
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

            # Get residuals (standardized residuals from GARCH)
            try:
                residuals = model_fit.resid / model_fit.conditional_volatility
                residuals = residuals.dropna().values
            except Exception as e:
                logger.warning(f"Could not extract GARCH residuals: {e}")
                residuals = None

            # Get model info
            aic = None
            bic = None
            try:
                if hasattr(model_fit, "aic"):
                    aic = float(model_fit.aic)
                if hasattr(model_fit, "bic"):
                    bic = float(model_fit.bic)
            except Exception as e:
                logger.warning(f"Could not extract AIC/BIC: {e}")

            # Get parameters
            params = None
            try:
                if hasattr(model_fit, "params"):
                    params = model_fit.params.to_dict()
            except Exception as e:
                logger.warning(f"Could not extract parameters: {e}")

            training_time = time.time() - start_time

            return ForecastResult(
                method="GARCH",
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
                    "p": p,
                    "q": q,
                    "mean": mean,
                    "dist": dist,
                    "aic": aic,
                    "bic": bic,
                    "params": params,
                    "mean_return": mean_return,
                    "avg_volatility": float(forecast_volatility.mean()),
                    "training_time": training_time,
                },
                residuals=residuals,
                success=True,
                message=f"GARCH({p},{q}) forecast completed",
            )

        except Exception as e:
            logger.error(f"GARCH forecast failed: {e}", exc_info=True)
            raise CalculationError(f"GARCH forecast failed: {e}") from e

