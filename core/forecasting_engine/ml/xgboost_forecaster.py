"""XGBoost forecaster implementation."""

import logging
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from core.exceptions import CalculationError
from core.forecasting_engine.base import BaseForecaster, ForecastResult
from core.forecasting_engine.utils import (
    calculate_confidence_intervals,
    prepare_features,
)

logger = logging.getLogger(__name__)


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost forecaster.

    High accuracy ML model with feature engineering.
    """

    def __init__(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None,
    ) -> None:
        """
        Initialize XGBoost forecaster.

        Args:
            prices: Historical prices series
            returns: Optional returns series
        """
        if xgb is None:
            raise ImportError(
                "XGBoost library not installed. Install with: pip install xgboost"
            )

        super().__init__(prices, returns)

    def forecast(
        self,
        horizon: int,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        use_technical_features: bool = True,
        lookback: int = 10,
        train_size: float = 0.8,
        early_stopping_rounds: int = 20,
        **kwargs,
    ) -> ForecastResult:
        """
        Generate forecast using XGBoost.

        Best practices applied:
        - Temporal train/validation split (no shuffle)
        - Early stopping to prevent overfitting
        - Optimized default parameters
        - Feature importance tracking

        Args:
            horizon: Number of periods to forecast
            n_estimators: Maximum number of boosting rounds
            max_depth: Maximum tree depth (lower reduces overfitting)
            learning_rate: Learning rate (lower with more estimators)
            use_technical_features: Whether to include technical indicators
            lookback: Number of lagged periods for features
            train_size: Fraction of data for training (rest for validation)
            early_stopping_rounds: Early stopping patience
            **kwargs: Additional XGBoost parameters

        Returns:
            ForecastResult with forecasted values
        """
        if xgb is None:
            raise ImportError("XGBoost library not installed")

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
                f"XGBoost: Training on {len(X_train)} samples, "
                f"validating on {len(X_val)} samples"
            )

            # Train model with early stopping
            # Use eval_set for validation monitoring
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                # Regularization to prevent overfitting
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                subsample=0.8,  # Row sampling
                colsample_bytree=0.8,  # Column sampling
                min_child_weight=3,  # Minimum samples in leaf
                # Objective and evaluation
                objective="reg:squarederror",
                eval_metric="rmse",
                **kwargs,
            )

            # Fit with early stopping
            best_iteration = n_estimators
            if len(X_val) > 0 and len(X_val) >= 5:
                # Only use early stopping if we have enough validation data
                # XGBoost 2.0+ uses callbacks for early stopping
                try:
                    # Try new API with callbacks (XGBoost 2.0+)
                    # Check if callback module exists and version supports it
                    if hasattr(xgb, 'callback') and hasattr(xgb.callback, 'EarlyStopping'):
                        # For XGBoost 2.0+, use callbacks
                        callbacks = [xgb.callback.EarlyStopping(
                            rounds=early_stopping_rounds,
                            save_best=True
                        )]
                        model.fit(
                            X_train,
                            y_train,
                            eval_set=[(X_val, y_val)],
                            callbacks=callbacks,
                            verbose=False,
                        )
                    else:
                        # Fallback: try old API with early_stopping_rounds parameter
                        model.fit(
                            X_train,
                            y_train,
                            eval_set=[(X_train, y_train), (X_val, y_val)],
                            early_stopping_rounds=early_stopping_rounds,
                            verbose=False,
                        )
                    # Get best iteration
                    if hasattr(model, "best_iteration") and model.best_iteration is not None:
                        best_iteration = model.best_iteration
                    elif hasattr(model, "best_ntree_limit"):
                        best_iteration = model.best_ntree_limit
                    logger.debug(f"XGBoost: Best iteration: {best_iteration}")
                except (AttributeError, TypeError) as e:
                    # If callbacks/early_stopping_rounds is not supported, fit without it
                    logger.warning(
                        f"XGBoost early stopping not supported: {e}. "
                        "Using all estimators"
                    )
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_train, y_train), (X_val, y_val)],
                        verbose=False,
                    )
            else:
                model.fit(X_train, y_train)
                logger.debug("XGBoost: No validation set, using all estimators")

            # Verify feature count matches
            expected_features = X.shape[1]
            logger.debug(
                f"XGBoost: Model expects {expected_features} features "
                f"(lookback={lookback}, technical={use_technical_features})"
            )

            # Calculate training metrics
            train_pred = model.predict(X_train)
            train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
            val_rmse = None
            if len(X_val) > 0:
                val_pred = model.predict(X_val)
                val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
                logger.debug(
                    f"XGBoost: Train RMSE={train_rmse:.6f}, "
                    f"Val RMSE={val_rmse:.6f}"
                )

            # Generate forecast iteratively
            forecast_values = []
            
            # Get last lookback elements from both arrays
            # Take exactly lookback elements, padding if necessary
            last_prices = self.prices.iloc[-min(lookback, len(self.prices)):].values
            last_returns = self.returns.iloc[-min(lookback, len(self.returns)):].values
            
            # Pad both arrays to exactly lookback length
            if len(last_prices) < lookback:
                # Pad with the first (oldest) price
                padding_value = last_prices[0] if len(last_prices) > 0 else self.prices.iloc[0]
                padding = np.full(lookback - len(last_prices), padding_value)
                last_prices = np.concatenate([padding, last_prices])
            
            if len(last_returns) < lookback:
                # Pad with zero (no return)
                padding = np.full(lookback - len(last_returns), 0.0)
                last_returns = np.concatenate([padding, last_returns])
            
            # Ensure both arrays have exactly lookback length
            last_prices = last_prices[-lookback:]
            last_returns = last_returns[-lookback:]
            
            # Final check: arrays must have the same length
            assert len(last_prices) == lookback, f"last_prices length {len(last_prices)} != lookback {lookback}"
            assert len(last_returns) == lookback, f"last_returns length {len(last_returns)} != lookback {lookback}"

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
                    f"Feature count mismatch: expected {expected_features}, got {len(current_features)}. "
                    f"Lookback={lookback}, use_technical={use_technical_features}"
                )

            for _ in range(horizon):
                # Predict next return
                next_return = model.predict(current_features.reshape(1, -1))[0]

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
            change_pct = self._calculate_change_pct(forecast_values, self.prices.iloc[-1])

            # Get residuals from training (use validation if available)
            if len(X_val) > 0:
                # Use validation residuals for better uncertainty estimation
                y_val_pred = model.predict(X_val)
                residuals = y_val - y_val_pred
            else:
                # Fallback to training residuals
                y_train_pred = model.predict(X_train)
                residuals = y_train - y_train_pred

            # Filter out NaN/Inf residuals
            valid_residuals = residuals[
                np.isfinite(residuals) & ~np.isnan(residuals)
            ]
            if len(valid_residuals) == 0:
                logger.warning("No valid residuals, using default confidence intervals")
                valid_residuals = None

            # Calculate confidence intervals
            confidence_intervals = calculate_confidence_intervals(
                forecast_values,
                residuals=valid_residuals,
                confidence_level=0.95,
            )

            training_time = time.time() - start_time

            # Get feature importance
            feature_importance = None
            if hasattr(model, "feature_importances_"):
                feature_importance = model.feature_importances_.tolist()
            elif hasattr(model, "get_booster"):
                try:
                    # Get feature importance from booster
                    booster = model.get_booster()
                    importance_dict = booster.get_score(importance_type="gain")
                    if importance_dict:
                        # Convert to list in feature order
                        feature_importance = [
                            importance_dict.get(f"f{i}", 0.0)
                            for i in range(expected_features)
                        ]
                except Exception as e:
                    logger.warning(f"Could not extract feature importance: {e}")

            # best_iteration already set during training

            return ForecastResult(
                method="XGBoost",
                forecast_dates=forecast_dates,
                forecast_values=forecast_values,
                forecast_returns=forecast_returns,
                confidence_intervals=confidence_intervals,
                final_value=float(forecast_values[-1]),
                change_pct=change_pct,
                model_info={
                    "n_estimators": n_estimators,
                    "best_iteration": best_iteration,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "use_technical_features": use_technical_features,
                    "lookback": lookback,
                    "train_size": train_size,
                    "train_rmse": float(train_rmse),
                    "val_rmse": float(val_rmse) if val_rmse is not None else None,
                    "training_time": training_time,
                    "feature_importance": feature_importance,
                    "n_features": expected_features,
                },
                residuals=valid_residuals if valid_residuals is not None else residuals,
                success=True,
                message=(
                    f"XGBoost forecast completed "
                    f"(train_rmse={train_rmse:.6f}"
                    + (f", val_rmse={val_rmse:.6f}" if val_rmse is not None else "")
                    + f", best_iter={best_iteration})"
                ),
            )

        except Exception as e:
            logger.error(f"XGBoost forecast failed: {e}", exc_info=True)
            raise CalculationError(f"XGBoost forecast failed: {e}") from e

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
        # Take only the last lookback elements
        prices = prices[-lookback:] if len(prices) > lookback else prices
        returns = returns[-lookback:] if len(returns) > lookback else returns
        
        # Pad if needed
        if len(prices) < lookback:
            padding = np.full(lookback - len(prices), prices[0] if len(prices) > 0 else 0.0)
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
            features.extend([
                prices[-1] / sma_5 - 1 if sma_5 > 0 else 0.0,
                prices[-1] / sma_10 - 1 if sma_10 > 0 else 0.0,
                prices[-1] / sma_20 - 1 if sma_20 > 0 else 0.0,
            ])

            # Volatility (rolling standard deviation) (3 features)
            features.extend([
                np.std(returns[-5:]) if len(returns) >= 5 else 0.0,
                np.std(returns[-10:]) if len(returns) >= 10 else 0.0,
                np.std(returns[-20:]) if len(returns) >= 20 else 0.0,
            ])

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
            features.extend([
                prices[-1] / prices[-6] - 1 if len(prices) >= 6 else 0.0,
                prices[-1] / prices[-11] - 1 if len(prices) >= 11 else 0.0,
            ])

            # Bollinger Bands (simplified) (3 features)
            if len(prices) >= 20:
                sma_20_bb = np.mean(prices[-20:])
                std_20_bb = np.std(prices[-20:])
                bb_upper = sma_20_bb + 2 * std_20_bb
                bb_lower = sma_20_bb - 2 * std_20_bb
                bb_width = (bb_upper - bb_lower) / sma_20_bb if sma_20_bb > 0 else 0.0
            else:
                bb_upper = prices[-1]
                bb_lower = prices[-1]
                bb_width = 0.0
            features.extend([bb_upper, bb_lower, bb_width])
        else:
            # Pad with zeros if technical features disabled
            # 15 technical indicators: 3 sma ratios + 3 volatility + 1 rsi + 2 momentum + 3 bb + 3 sma values
            features.extend([0.0] * 15)

        return np.array(features)

