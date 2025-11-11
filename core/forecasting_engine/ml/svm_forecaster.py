"""SVM/SVR forecaster implementation."""

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
except ImportError:
    SVR = None
    StandardScaler = None

from core.exceptions import CalculationError
from core.forecasting_engine.base import BaseForecaster, ForecastResult
from core.forecasting_engine.utils import (
    calculate_confidence_intervals,
    prepare_features,
)

logger = logging.getLogger(__name__)


class SVMForecaster(BaseForecaster):
    """
    Support Vector Regression (SVR) forecaster.

    SVR is effective for time series forecasting, especially with
    non-linear patterns. Requires feature scaling for optimal performance.
    """

    def __init__(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None,
    ) -> None:
        """
        Initialize SVM forecaster.

        Args:
            prices: Historical prices series
            returns: Optional returns series
        """
        if SVR is None:
            raise ImportError(
                "scikit-learn not installed. Install with: pip install scikit-learn"
            )

        super().__init__(prices, returns)

    def forecast(
        self,
        horizon: int,
        kernel: str = "rbf",
        C: float = 100.0,
        epsilon: float = 0.1,
        gamma: Optional[str] = "scale",
        use_technical_features: bool = True,
        lookback: int = 10,
        train_size: float = 0.8,
        scale_features: bool = True,
        **kwargs,
    ) -> ForecastResult:
        """
        Generate forecast using Support Vector Regression (SVR).

        Best practices applied:
        - Temporal train/validation split (no shuffle)
        - Feature scaling (critical for SVR)
        - Optimized default parameters for time series
        - RBF kernel for non-linear patterns

        Args:
            horizon: Number of periods to forecast
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            C: Regularization parameter (higher = less regularization)
            epsilon: Epsilon-tube width (tolerance for errors)
            gamma: Kernel coefficient ('scale', 'auto', or float)
            use_technical_features: Whether to include technical indicators
            lookback: Number of lagged periods for features
            train_size: Fraction of data for training (rest for validation)
            scale_features: Whether to scale features (recommended for SVR)
            **kwargs: Additional SVR parameters

        Returns:
            ForecastResult with forecasted values
        """
        if SVR is None:
            raise ImportError("scikit-learn not installed")

        start_time = time.time()

        try:
            # Prepare features
            features_df = prepare_features(
                prices=self.prices,
                returns=self.returns,
                lookback=lookback,
                include_technical=use_technical_features,
            )

            if len(features_df) == 0:
                raise CalculationError("No features generated")

            # Separate features and target
            X = features_df.drop(columns=["target"]).values
            y = features_df["target"].values

            # Remove NaN rows
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]

            if len(X) < 50:
                raise CalculationError(
                    f"Insufficient data for training: {len(X)} samples "
                    "(need at least 50)"
                )

            # Temporal train/validation split (CRITICAL for time series)
            # Do NOT shuffle - preserve temporal order
            split_idx = int(len(X) * train_size)
            if split_idx < 30:
                # If split too small, use all data for training
                logger.warning(
                    f"Train size too small ({split_idx}), using all data for training"
                )
                X_train, X_val = X, X
                y_train, y_val = y, y
            else:
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

            logger.debug(
                f"SVR: Training on {len(X_train)} samples, "
                f"validating on {len(X_val)} samples"
            )

            # Feature scaling (CRITICAL for SVR performance)
            scaler_X = None
            scaler_y = None
            if scale_features:
                scaler_X = StandardScaler()
                X_train_scaled = scaler_X.fit_transform(X_train)
                if len(X_val) > 0:
                    X_val_scaled = scaler_X.transform(X_val)
                else:
                    X_val_scaled = X_train_scaled

                # Scale target for better SVR performance
                scaler_y = StandardScaler()
                y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
                if len(y_val) > 0:
                    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
                else:
                    y_val_scaled = y_train_scaled
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
                y_train_scaled = y_train
                y_val_scaled = y_val

            # Train SVR model
            model = SVR(
                kernel=kernel,
                C=C,
                epsilon=epsilon,
                gamma=gamma,
                **kwargs,
            )

            model.fit(X_train_scaled, y_train_scaled)

            # Verify feature count matches
            expected_features = X.shape[1]
            logger.debug(
                f"SVR: Model expects {expected_features} features "
                f"(lookback={lookback}, technical={use_technical_features})"
            )

            # Calculate training metrics
            train_pred_scaled = model.predict(X_train_scaled)
            if scale_features and scaler_y is not None:
                train_pred = scaler_y.inverse_transform(
                    train_pred_scaled.reshape(-1, 1)
                ).ravel()
            else:
                train_pred = train_pred_scaled

            train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
            val_rmse = None
            if len(X_val) > 0:
                val_pred_scaled = model.predict(X_val_scaled)
                if scale_features and scaler_y is not None:
                    val_pred = scaler_y.inverse_transform(
                        val_pred_scaled.reshape(-1, 1)
                    ).ravel()
                else:
                    val_pred = val_pred_scaled
                val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
                logger.debug(
                    f"SVR: Train RMSE={train_rmse:.6f}, "
                    f"Val RMSE={val_rmse:.6f}"
                )

            # Generate forecast iteratively
            forecast_values = []

            # Get last lookback elements from both arrays
            last_prices = self.prices.iloc[
                -min(lookback, len(self.prices)) :
            ].values
            last_returns = self.returns.iloc[
                -min(lookback, len(self.returns)) :
            ].values

            # Pad both arrays to exactly lookback length
            if len(last_prices) < lookback:
                padding_value = (
                    last_prices[0] if len(last_prices) > 0 else self.prices.iloc[0]
                )
                padding = np.full(lookback - len(last_prices), padding_value)
                last_prices = np.concatenate([padding, last_prices])

            if len(last_returns) < lookback:
                padding = np.full(lookback - len(last_returns), 0.0)
                last_returns = np.concatenate([padding, last_returns])

            # Ensure both arrays have exactly lookback length
            last_prices = last_prices[-lookback:]
            last_returns = last_returns[-lookback:]

            # Final check: arrays must have the same length
            assert (
                len(last_prices) == lookback
            ), f"last_prices length {len(last_prices)} != lookback {lookback}"
            assert (
                len(last_returns) == lookback
            ), f"last_returns length {len(last_returns)} != lookback {lookback}"

            # Prepare initial feature vector
            current_features = self._prepare_single_features(
                last_prices,
                last_returns,
                lookback,
                use_technical_features,
            )

            # Verify feature count
            if len(current_features) != expected_features:
                raise CalculationError(
                    f"Feature count mismatch: expected {expected_features}, "
                    f"got {len(current_features)}. "
                    f"Lookback={lookback}, use_technical={use_technical_features}"
                )

            for _ in range(horizon):
                # Scale features if needed
                if scale_features and scaler_X is not None:
                    features_scaled = scaler_X.transform(
                        current_features.reshape(1, -1)
                    )
                else:
                    features_scaled = current_features.reshape(1, -1)

                # Predict next return (scaled)
                next_return_scaled = model.predict(features_scaled)[0]

                # Inverse transform if scaled
                if scale_features and scaler_y is not None:
                    next_return = scaler_y.inverse_transform(
                        np.array([[next_return_scaled]])
                    )[0, 0]
                else:
                    next_return = next_return_scaled

                # Clamp return to reasonable range
                next_return = np.clip(next_return, -0.99, 0.99)

                # Update prices and returns
                last_price = last_prices[-1]
                next_price = last_price * (1 + next_return)

                forecast_values.append(next_price)

                # Update feature vector - maintain lookback length
                last_prices = np.concatenate([last_prices[1:], [next_price]])
                last_returns = np.concatenate([last_returns[1:], [next_return]])

                # Ensure arrays maintain lookback length
                if len(last_prices) > lookback:
                    last_prices = last_prices[-lookback:]
                if len(last_returns) > lookback:
                    last_returns = last_returns[-lookback:]

                current_features = self._prepare_single_features(
                    last_prices,
                    last_returns,
                    lookback,
                    use_technical_features,
                )

            forecast_values = np.array(forecast_values)

            # Create forecast dates
            last_date = self.prices.index[-1]
            forecast_dates = self._create_forecast_dates(horizon, last_date)

            # Calculate returns
            forecast_returns = np.diff(forecast_values) / forecast_values[:-1]
            first_return = (forecast_values[0] - self.prices.iloc[-1]) / self.prices.iloc[-1]
            forecast_returns = np.insert(forecast_returns, 0, first_return)

            # Calculate change percentage
            change_pct = self._calculate_change_pct(
                forecast_values, self.prices.iloc[-1]
            )

            # Get residuals (use validation if available)
            if len(X_val) > 0:
                # Use validation residuals for better uncertainty estimation
                y_val_pred_scaled = model.predict(X_val_scaled)
                if scale_features and scaler_y is not None:
                    y_val_pred = scaler_y.inverse_transform(
                        y_val_pred_scaled.reshape(-1, 1)
                    ).ravel()
                else:
                    y_val_pred = y_val_pred_scaled
                residuals = y_val - y_val_pred
            else:
                # Fallback to training residuals
                residuals = y_train - train_pred

            # Filter out NaN/Inf residuals
            valid_residuals = residuals[
                np.isfinite(residuals) & ~np.isnan(residuals)
            ]
            if len(valid_residuals) == 0:
                logger.warning(
                    "No valid residuals, using default confidence intervals"
                )
                valid_residuals = None

            # Calculate confidence intervals
            confidence_intervals = calculate_confidence_intervals(
                forecast_values,
                residuals=valid_residuals,
                confidence_level=0.95,
            )

            training_time = time.time() - start_time

            # Get support vectors info
            n_support_vectors = (
                len(model.support_vectors_) if hasattr(model, "support_vectors_") else None
            )

            return ForecastResult(
                method="SVR",
                forecast_dates=forecast_dates,
                forecast_values=forecast_values,
                forecast_returns=forecast_returns,
                confidence_intervals=confidence_intervals,
                final_value=float(forecast_values[-1]),
                change_pct=change_pct,
                model_info={
                    "kernel": kernel,
                    "C": C,
                    "epsilon": epsilon,
                    "gamma": gamma,
                    "use_technical_features": use_technical_features,
                    "lookback": lookback,
                    "train_size": train_size,
                    "scale_features": scale_features,
                    "train_rmse": float(train_rmse),
                    "val_rmse": float(val_rmse) if val_rmse is not None else None,
                    "n_support_vectors": n_support_vectors,
                    "n_features": expected_features,
                    "training_time": training_time,
                },
                residuals=valid_residuals if valid_residuals is not None else residuals,
                success=True,
                message=(
                    f"SVR forecast completed "
                    f"(train_rmse={train_rmse:.6f}"
                    + (f", val_rmse={val_rmse:.6f}" if val_rmse is not None else "")
                    + f", kernel={kernel})"
                ),
            )

        except Exception as e:
            logger.error(f"SVR forecast failed: {e}", exc_info=True)
            raise CalculationError(f"SVR forecast failed: {e}") from e

    def _prepare_single_features(
        self,
        prices: np.ndarray,
        returns: np.ndarray,
        lookback: int,
        use_technical: bool,
    ) -> np.ndarray:
        """
        Prepare feature vector for single prediction.
        Must match the features created by prepare_features().
        """
        features = []

        # Ensure arrays are the right length (lookback)
        prices = prices[-lookback:] if len(prices) > lookback else prices
        returns = returns[-lookback:] if len(returns) > lookback else returns

        # Pad if needed
        if len(prices) < lookback:
            padding = np.full(
                lookback - len(prices), prices[0] if len(prices) > 0 else 0.0
            )
            prices = np.concatenate([padding, prices])
        if len(returns) < lookback:
            padding = np.full(lookback - len(returns), 0.0)
            returns = np.concatenate([padding, returns])

        # Lagged returns (lookback features)
        for i in range(1, lookback + 1):
            if i <= len(returns):
                features.append(returns[-i])
            else:
                features.append(0.0)

        # Lagged prices (normalized) - up to 5 features
        for i in range(1, min(lookback + 1, 6)):
            if i < len(prices):
                features.append(prices[-i] / prices[-i - 1] - 1)
            else:
                features.append(0.0)

        if use_technical:
            # Technical indicators - must match calculate_technical_indicators()
            # Moving averages (3 features: sma_5, sma_10, sma_20)
            sma_5 = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
            sma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else sma_5
            sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else sma_10
            features.extend([sma_5, sma_10, sma_20])

            # Price relative to moving averages (3 features)
            features.extend(
                [
                    prices[-1] / sma_5 - 1 if sma_5 > 0 else 0.0,
                    prices[-1] / sma_10 - 1 if sma_10 > 0 else 0.0,
                    prices[-1] / sma_20 - 1 if sma_20 > 0 else 0.0,
                ]
            )

            # Volatility (rolling standard deviation) (3 features)
            features.extend(
                [
                    np.std(returns[-5:]) if len(returns) >= 5 else 0.0,
                    np.std(returns[-10:]) if len(returns) >= 10 else 0.0,
                    np.std(returns[-20:]) if len(returns) >= 20 else 0.0,
                ]
            )

            # RSI (simplified) (1 feature)
            if len(returns) >= 14:
                delta = returns[-14:]
                gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0.0
                loss = np.mean(-delta[delta < 0]) if np.any(delta < 0) else 0.0001
                rs = gain / loss if loss > 0 else 0.0
                rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50.0
            else:
                rsi = 50.0
            features.append(rsi)

            # Momentum (2 features)
            features.extend(
                [
                    prices[-1] / prices[-6] - 1 if len(prices) >= 6 else 0.0,
                    prices[-1] / prices[-11] - 1 if len(prices) >= 11 else 0.0,
                ]
            )

            # Bollinger Bands (simplified) (3 features)
            if len(prices) >= 20:
                sma_20_bb = np.mean(prices[-20:])
                std_20_bb = np.std(prices[-20:])
                bb_upper = sma_20_bb + 2 * std_20_bb
                bb_lower = sma_20_bb - 2 * std_20_bb
                bb_width = (
                    (bb_upper - bb_lower) / sma_20_bb if sma_20_bb > 0 else 0.0
                )
            else:
                bb_upper = prices[-1]
                bb_lower = prices[-1]
                bb_width = 0.0
            features.extend([bb_upper, bb_lower, bb_width])
        else:
            # Pad with zeros if technical features disabled
            # 15 technical indicators
            features.extend([0.0] * 15)

        return np.array(features)

