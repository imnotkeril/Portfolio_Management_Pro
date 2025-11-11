"""Utility functions for forecasting."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_forecast_metrics(
    forecast: np.ndarray,
    actual: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate forecast quality metrics.

    Args:
        forecast: Forecasted values
        actual: Actual values

    Returns:
        Dictionary with metrics: MAPE, RMSE, MAE, direction_accuracy, r_squared
    """
    if len(forecast) != len(actual):
        raise ValueError("Forecast and actual must have same length")

    if len(forecast) == 0:
        return {
            "mape": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "direction_accuracy": np.nan,
            "r_squared": np.nan,
        }

    # Remove any NaN or Inf values
    valid_mask = np.isfinite(forecast) & np.isfinite(actual)
    forecast_clean = forecast[valid_mask]
    actual_clean = actual[valid_mask]

    if len(forecast_clean) == 0:
        return {
            "mape": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "direction_accuracy": np.nan,
            "r_squared": np.nan,
        }

    # Mean Absolute Percentage Error (MAPE)
    # Handle division by zero: skip points where actual is zero or very small
    mape_mask = np.abs(actual_clean) > 1e-10
    if np.any(mape_mask):
        mape = np.mean(
            np.abs((actual_clean[mape_mask] - forecast_clean[mape_mask]) / actual_clean[mape_mask])
        ) * 100.0
    else:
        # If all actual values are zero, MAPE is undefined
        mape = np.nan

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((actual_clean - forecast_clean) ** 2))

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(actual_clean - forecast_clean))

    # Direction Accuracy (percentage of correct direction predictions)
    if len(forecast_clean) > 1:
        forecast_direction = np.diff(forecast_clean) > 0
        actual_direction = np.diff(actual_clean) > 0
        direction_accuracy = np.mean(forecast_direction == actual_direction) * 100.0
    else:
        direction_accuracy = np.nan

    # R-squared
    ss_res = np.sum((actual_clean - forecast_clean) ** 2)
    ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        "mape": float(mape),
        "rmse": float(rmse),
        "mae": float(mae),
        "direction_accuracy": float(direction_accuracy),
        "r_squared": float(r_squared),
    }


def calculate_confidence_intervals(
    forecast: np.ndarray,
    residuals: Optional[np.ndarray] = None,
    confidence_level: float = 0.95,
) -> Dict[str, np.ndarray]:
    """
    Calculate confidence intervals for forecast.

    Args:
        forecast: Forecasted values
        residuals: Model residuals (for variance estimation)
        confidence_level: Confidence level (0.95 for 95% CI)

    Returns:
        Dictionary with upper and lower bounds
    """
    from scipy import stats

    if residuals is None or len(residuals) == 0:
        # Use simple approach: assume constant variance based on forecast magnitude
        # Use percentage-based error for better scaling
        std_error = np.abs(forecast) * 0.1  # 10% of forecast value
    else:
        # Filter valid residuals
        valid_residuals = residuals[np.isfinite(residuals) & ~np.isnan(residuals)]
        if len(valid_residuals) == 0:
            # No valid residuals, use percentage-based
            std_error = np.abs(forecast) * 0.1
        else:
            # Use residual standard error, but scale it appropriately
            residual_std = np.std(valid_residuals)
            
            # Get average forecast magnitude for comparison
            if len(forecast) > 0:
                avg_forecast_magnitude = np.mean(np.abs(forecast))
                min_forecast_magnitude = np.maximum(np.min(np.abs(forecast)), 1e-10)
            else:
                avg_forecast_magnitude = 1.0
                min_forecast_magnitude = 1.0
            
            max_residual_magnitude = np.max(np.abs(valid_residuals))
            
            # For price forecasts, use percentage-based error if residuals are large relative to forecast
            if max_residual_magnitude > avg_forecast_magnitude * 0.5:
                # Residuals are large relative to forecast, use percentage-based
                std_error = np.abs(forecast) * 0.15  # 15% of forecast value
            else:
                # Use absolute residual std, but limit to reasonable percentage
                # Ensure std_error has same length as forecast
                std_error = np.minimum(
                    np.full_like(forecast, residual_std),
                    np.abs(forecast) * 0.2
                )
                # Ensure minimum error to avoid zero intervals
                std_error = np.maximum(std_error, min_forecast_magnitude * 0.01)

    # Z-score for confidence level
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha / 2)

    # Calculate intervals
    upper = forecast + z_score * std_error
    lower = forecast - z_score * std_error
    
    # Ensure reasonable bounds
    # Lower bound should not be negative for prices
    lower = np.maximum(lower, forecast * 0.1)  # At least 10% of forecast
    # Upper bound should not be more than 3x forecast
    upper = np.minimum(upper, forecast * 3.0)

    return {
        f"upper_{int(confidence_level * 100)}": upper,
        f"lower_{int(confidence_level * 100)}": lower,
    }


def prepare_features(
    prices: pd.Series,
    returns: pd.Series,
    lookback: int = 10,
    include_technical: bool = True,
) -> pd.DataFrame:
    """
    Prepare features for ML models.

    Args:
        prices: Historical prices
        returns: Historical returns
        lookback: Number of lagged periods to include
        include_technical: Whether to include technical indicators

    Returns:
        DataFrame with features
    """
    features = pd.DataFrame(index=returns.index)

    # Lagged returns
    for i in range(1, lookback + 1):
        features[f"return_lag_{i}"] = returns.shift(i)

    # Lagged prices (normalized)
    for i in range(1, min(lookback + 1, 6)):
        price_lag_i = prices.shift(i)
        price_lag_i1 = prices.shift(i + 1)
        # Avoid division by zero
        ratio = price_lag_i / price_lag_i1.replace(0, np.nan) - 1
        features[f"price_lag_{i}"] = ratio

    if include_technical:
        # Technical indicators
        tech_indicators = calculate_technical_indicators(prices, returns)
        features = pd.concat([features, tech_indicators], axis=1)

    # Target variable (next period return)
    features["target"] = returns.shift(-1)

    # Remove rows with NaN
    features = features.dropna()

    return features


def calculate_technical_indicators(
    prices: pd.Series,
    returns: pd.Series,
) -> pd.DataFrame:
    """
    Calculate technical indicators.

    Args:
        prices: Historical prices
        returns: Historical returns

    Returns:
        DataFrame with technical indicators
    """
    indicators = pd.DataFrame(index=prices.index)

    # Moving averages
    indicators["sma_5"] = prices.rolling(window=5).mean()
    indicators["sma_10"] = prices.rolling(window=10).mean()
    indicators["sma_20"] = prices.rolling(window=20).mean()

    # Price relative to moving averages
    # Avoid division by zero
    sma_5_safe = indicators["sma_5"].replace(0, np.nan)
    sma_10_safe = indicators["sma_10"].replace(0, np.nan)
    sma_20_safe = indicators["sma_20"].replace(0, np.nan)
    indicators["price_sma5_ratio"] = prices / sma_5_safe - 1
    indicators["price_sma10_ratio"] = prices / sma_10_safe - 1
    indicators["price_sma20_ratio"] = prices / sma_20_safe - 1

    # Volatility (rolling standard deviation)
    indicators["volatility_5"] = returns.rolling(window=5).std()
    indicators["volatility_10"] = returns.rolling(window=10).std()
    indicators["volatility_20"] = returns.rolling(window=20).std()

    # RSI (Relative Strength Index) - simplified
    delta = returns
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # Avoid division by zero: use small epsilon for loss
    loss_safe = loss.replace(0, 1e-10)
    rs = gain / loss_safe
    # Handle infinite RS values
    rs = rs.replace([np.inf, -np.inf], 0)
    indicators["rsi"] = 100 - (100 / (1 + rs))
    # Clamp RSI to valid range [0, 100]
    indicators["rsi"] = indicators["rsi"].clip(0, 100)

    # Momentum
    # Avoid division by zero
    prices_shift5 = prices.shift(5).replace(0, np.nan)
    prices_shift10 = prices.shift(10).replace(0, np.nan)
    indicators["momentum_5"] = prices / prices_shift5 - 1
    indicators["momentum_10"] = prices / prices_shift10 - 1

    # Bollinger Bands (simplified)
    sma_20 = prices.rolling(window=20).mean()
    std_20 = prices.rolling(window=20).std()
    indicators["bb_upper"] = sma_20 + 2 * std_20
    indicators["bb_lower"] = sma_20 - 2 * std_20
    # Avoid division by zero for bb_width
    sma_20_safe = sma_20.replace(0, np.nan)
    indicators["bb_width"] = (
        (indicators["bb_upper"] - indicators["bb_lower"]) / sma_20_safe
    )

    return indicators


def split_train_validation(
    data: pd.Series,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> Tuple[pd.Series, pd.Series]:
    """
    Split data into training and validation sets.

    Preserves temporal order (critical for time series).

    Args:
        data: Time series data
        validation_start: Start of validation period
        validation_end: End of validation period

    Returns:
        Tuple of (training_data, validation_data)

    Raises:
        ValueError: If validation period is invalid or no data available
    """
    if data.empty:
        raise ValueError("Cannot split empty data series")

    if validation_start >= validation_end:
        raise ValueError(
            f"Invalid validation period: start ({validation_start}) >= end ({validation_end})"
        )

    # Ensure dates are timezone-naive for comparison
    if hasattr(validation_start, "tz") and validation_start.tz is not None:
        validation_start = validation_start.tz_localize(None)
    if hasattr(validation_end, "tz") and validation_end.tz is not None:
        validation_end = validation_end.tz_localize(None)

    # Normalize data index timezone if needed
    data_index = data.index
    if hasattr(data_index, "tz") and data_index.tz is not None:
        data_index = data_index.tz_localize(None)
        data = pd.Series(data.values, index=data_index)

    training_data = data[data.index < validation_start]
    validation_data = data[
        (data.index >= validation_start) & (data.index <= validation_end)
    ]

    if len(training_data) == 0:
        raise ValueError(
            f"No training data available before {validation_start}. "
            f"Data range: {data.index.min()} to {data.index.max()}"
        )

    if len(validation_data) == 0:
        logger.warning(
            f"No validation data in period {validation_start} to {validation_end}. "
            f"Data range: {data.index.min()} to {data.index.max()}"
        )

    return training_data, validation_data

