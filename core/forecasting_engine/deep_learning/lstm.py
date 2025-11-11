"""LSTM forecaster implementation using TensorFlow/Keras."""

import logging
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import StandardScaler
except ImportError:
    tf = None
    keras = None
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None
    Bidirectional = None
    EarlyStopping = None
    ReduceLROnPlateau = None
    StandardScaler = None

from core.exceptions import CalculationError
from core.forecasting_engine.base import BaseForecaster, ForecastResult
from core.forecasting_engine.utils import calculate_confidence_intervals

logger = logging.getLogger(__name__)


class LSTMForecaster(BaseForecaster):
    """
    Long Short-Term Memory (LSTM) forecaster.

    LSTM networks are effective for time series forecasting due to their
    ability to capture long-term dependencies in sequential data.
    """

    def __init__(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None,
    ) -> None:
        """
        Initialize LSTM forecaster.

        Args:
            prices: Historical prices series
            returns: Optional returns series
        """
        if tf is None or keras is None:
            raise ImportError(
                "TensorFlow/Keras not installed. "
                "Install with: pip install tensorflow scikit-learn"
            )

        super().__init__(prices, returns)

    def _create_sequences(
        self,
        data: np.ndarray,
        lookback: int,
        forecast_horizon: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.

        Args:
            data: Input data array
            lookback: Number of time steps to look back (sequence length)
            forecast_horizon: Number of steps ahead to forecast (default: 1)

        Returns:
            Tuple of (X, y) where X is sequences and y is targets
        """
        X, y = [], []
        for i in range(len(data) - lookback - forecast_horizon + 1):
            X.append(data[i:(i + lookback)])
            y.append(data[i + lookback + forecast_horizon - 1])
        return np.array(X), np.array(y)
    
    def _calculate_directional_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        """
        Calculate directional accuracy metrics.
        
        Important for trading: predicts if price will go up or down,
        which is often more important than exact value.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary with directional accuracy metrics
        """
        if len(y_true) < 2 or len(y_pred) < 2:
            return {
                'directional_accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
            }
        
        # Calculate directions
        true_direction = np.diff(y_true) > 0  # True if price increased
        pred_direction = np.diff(y_pred) > 0
        
        # Directional accuracy
        directional_accuracy = np.mean(true_direction == pred_direction)
        
        # Confusion matrix
        tp = np.sum((true_direction == True) & (pred_direction == True))
        tn = np.sum((true_direction == False) & (pred_direction == False))
        fp = np.sum((true_direction == False) & (pred_direction == True))
        fn = np.sum((true_direction == True) & (pred_direction == False))
        
        # Precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'directional_accuracy': float(directional_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
        }

    def forecast(
        self,
        horizon: int,
        units: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        lookback: int = 30,
        train_size: float = 0.8,
        use_returns: bool = False,
        use_log_returns: bool = False,
        bidirectional: bool = False,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        learning_rate: float = 0.001,
        **kwargs,
    ) -> ForecastResult:
        """
        Generate forecast using LSTM.

        Best practices applied (based on financial forecasting guides):
        - StandardScaler for normalization (NOT MinMaxScaler - critical for financial data)
        - Temporal train/validation split BEFORE scaling (avoid data leakage)
        - Sequence length 30 days (optimal for 1-day forecasts)
        - Early stopping to prevent overfitting
        - Learning rate reduction on plateau
        - Multiple LSTM layers with dropout
        - Optional bidirectional LSTM for better context
        - Support for log returns forecasting (recommended for trading)

        Args:
            horizon: Number of periods to forecast
            units: Number of LSTM units per layer (default: 64, recommended: 64-256)
            layers: Number of LSTM layers (default: 2, recommended: 2-3)
            dropout: Dropout rate (0.0-1.0, default: 0.2)
            epochs: Maximum number of training epochs (default: 100)
            batch_size: Batch size for training (default: 32)
            lookback: Number of time steps to look back (default: 30, recommended: 20-60)
            train_size: Fraction of data for training (default: 0.8)
            use_returns: Whether to forecast returns (True) or prices (False)
            use_log_returns: Whether to use log returns (recommended for trading)
            bidirectional: Use bidirectional LSTM (default: False)
            validation_split: Fraction of training data for validation (default: 0.2)
            early_stopping_patience: Early stopping patience (default: 10)
            learning_rate: Initial learning rate (default: 0.001)
            **kwargs: Additional parameters

        Returns:
            ForecastResult with forecasted values
        """
        if tf is None or keras is None:
            raise ImportError("TensorFlow/Keras not installed")

        start_time = time.time()

        try:
            # Auto-adapt parameters based on forecast horizon
            # For long horizons (>60 days), use log returns and increase lookback
            if horizon > 60 and not use_returns and not use_log_returns:
                logger.info(
                    f"Long forecast horizon ({horizon} days) detected. "
                    "Auto-enabling log returns for better stability."
                )
                use_log_returns = True
            
            # Adapt lookback based on horizon (rule: lookback ≈ horizon for long forecasts)
            # But cap at reasonable limits
            if horizon > 60:
                adapted_lookback = min(max(lookback, int(horizon * 0.3)), 120)
                if adapted_lookback > lookback:
                    logger.info(
                        f"Increasing lookback from {lookback} to {adapted_lookback} "
                        f"for long forecast horizon ({horizon} days)"
                    )
                    lookback = adapted_lookback
            
            # Prepare data - CRITICAL: Split BEFORE scaling to avoid data leakage
            if use_log_returns:
                # Calculate log returns: log(p_t / p_{t-1})
                if len(self.prices) < 2:
                    raise CalculationError("Need at least 2 prices for log returns")
                log_returns = np.diff(np.log(self.prices.values))
                data = log_returns
                logger.debug("Using log returns for forecasting")
            elif use_returns:
                data = self.returns.dropna().values
                if len(data) < lookback + 10:
                    logger.warning(
                        "Insufficient returns data, falling back to prices"
                    )
                    use_returns = False
                    use_log_returns = False
                    data = self.prices.values
            else:
                data = self.prices.values

            if len(data) < lookback + 50:
                raise CalculationError(
                    f"Insufficient data: {len(data)} samples "
                    f"(need at least {lookback + 50})"
                )

            # CRITICAL: Split data BEFORE scaling to avoid look-ahead bias
            # This is a common mistake that leads to overfitting
            split_point = int(len(data) * train_size)
            train_data = data[:split_point]
            test_data = data[split_point:] if split_point < len(data) else data[-1:]

            # Normalize data using StandardScaler (NOT MinMaxScaler)
            # StandardScaler is recommended for financial data because:
            # 1. Prices don't have upper bounds (MinMaxScaler distorts future predictions)
            # 2. Better handles outliers
            # 3. Preserves distribution properties
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).ravel()
            
            # Transform test data using training statistics (no fit!)
            if len(test_data) > 0:
                test_scaled = scaler.transform(test_data.reshape(-1, 1)).ravel()
            else:
                test_scaled = np.array([])

            # Create sequences from scaled data
            X_train, y_train = self._create_sequences(train_scaled, lookback)
            
            if len(test_scaled) > 0:
                # For validation, use test data
                X_val, y_val = self._create_sequences(test_scaled, lookback)
            else:
                # Fallback: use last portion of training data for validation
                val_split_idx = int(len(X_train) * (1 - validation_split))
                if val_split_idx > 0:
                    X_val = X_train[val_split_idx:]
                    y_val = y_train[val_split_idx:]
                    X_train = X_train[:val_split_idx]
                    y_train = y_train[:val_split_idx]
                else:
                    X_val = np.array([])
                    y_val = np.array([])

            if len(X_train) < 30:
                raise CalculationError(
                    f"Insufficient training sequences: {len(X_train)} "
                    "(need at least 30)"
                )
            logger.debug(
                f"LSTM: Training on {len(X_train)} sequences, "
                f"validating on {len(X_val)} sequences"
            )

            # Reshape for LSTM: (samples, time_steps, features)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            if len(X_val) > 0:
                X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

            # Build LSTM model
            model = Sequential()

            # First LSTM layer
            lstm_layer = LSTM(
                units=units,
                return_sequences=layers > 1,
                input_shape=(lookback, 1),
            )
            
            # Use bidirectional if requested (better for capturing context)
            if bidirectional:
                lstm_layer = Bidirectional(lstm_layer)
            
            model.add(lstm_layer)
            model.add(Dropout(dropout))

            # Additional LSTM layers
            for i in range(1, layers):
                is_last = i == layers - 1
                lstm_layer = LSTM(
                    units=units if not is_last else units // 2,
                    return_sequences=not is_last,
                )
                
                if bidirectional and not is_last:
                    lstm_layer = Bidirectional(lstm_layer)
                
                model.add(lstm_layer)
                model.add(Dropout(dropout))

            # Dense layers before output
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.1))
            
            # Output layer
            model.add(Dense(1))

            # Compile model with gradient clipping to prevent exploding gradients
            # This is critical for stable training, especially with LSTM
            optimizer = keras.optimizers.Adam(
                learning_rate=learning_rate,
                clipnorm=1.0  # Clip gradients to prevent explosion
            )
            model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

            logger.debug(
                f"LSTM: Model architecture - {layers} layers, "
                f"{units} units, dropout={dropout}"
            )

            # Callbacks
            callbacks = []
            if len(X_val) > 0 and early_stopping_patience > 0:
                callbacks.append(
                    EarlyStopping(
                        monitor="val_loss",
                        patience=early_stopping_patience,
                        restore_best_weights=True,
                        verbose=0,
                    )
                )
                callbacks.append(
                    ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.5,
                        patience=early_stopping_patience // 2,
                        min_lr=1e-7,
                        verbose=0,
                    )
                )

            # Train model
            validation_data = (
                (X_val, y_val) if len(X_val) > 0 else None
            )
            val_split = validation_split if validation_data is None else 0.0

            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                validation_split=val_split,
                callbacks=callbacks,
                verbose=0,
            )

            # Get best epoch
            best_epoch = len(history.history["loss"])
            if "val_loss" in history.history:
                best_epoch = np.argmin(history.history["val_loss"]) + 1
            logger.debug(f"LSTM: Best epoch: {best_epoch}/{epochs}")

            # Calculate training metrics
            train_pred_scaled = model.predict(X_train, verbose=0).ravel()
            train_pred = scaler.inverse_transform(
                train_pred_scaled.reshape(-1, 1)
            ).ravel()
            train_actual = scaler.inverse_transform(
                y_train.reshape(-1, 1)
            ).ravel()

            train_rmse = np.sqrt(np.mean((train_actual - train_pred) ** 2))
            val_rmse = None
            directional_metrics = None

            if len(X_val) > 0:
                val_pred_scaled = model.predict(X_val, verbose=0).ravel()
                val_pred = scaler.inverse_transform(
                    val_pred_scaled.reshape(-1, 1)
                ).ravel()
                val_actual = scaler.inverse_transform(
                    y_val.reshape(-1, 1)
                ).ravel()
                val_rmse = np.sqrt(np.mean((val_actual - val_pred) ** 2))
                
                # Calculate directional accuracy (important for trading)
                directional_metrics = self._calculate_directional_accuracy(
                    val_actual, val_pred
                )
                
                logger.debug(
                    f"LSTM: Train RMSE={train_rmse:.6f}, "
                    f"Val RMSE={val_rmse:.6f}, "
                    f"Directional Accuracy={directional_metrics['directional_accuracy']:.2%}"
                )
            else:
                # Calculate directional accuracy on training data if no validation
                directional_metrics = self._calculate_directional_accuracy(
                    train_actual, train_pred
                )
                logger.debug(
                    f"LSTM: Train RMSE={train_rmse:.6f}, "
                    f"Directional Accuracy={directional_metrics['directional_accuracy']:.2%}"
                )

            # Generate forecast iteratively
            # Use last sequence from training data (properly scaled)
            forecast_scaled = []
            last_sequence = train_scaled[-lookback:].copy()
            
            # Log initial sequence for debugging
            logger.debug(
                f"LSTM: Starting forecast with last_sequence range: "
                f"[{last_sequence.min():.4f}, {last_sequence.max():.4f}]"
            )

            # For very long horizons, add some noise reduction and stability checks
            # Track recent predictions to detect if model is stuck
            recent_predictions = []
            max_recent = min(10, horizon // 10)
            
            for step in range(horizon):
                # Reshape for prediction
                X_pred = last_sequence.reshape((1, lookback, 1))

                # Predict next value
                next_value_raw = model.predict(X_pred, verbose=0)[0, 0]
                
                # Log raw prediction
                if step < 3 or step == horizon - 1 or step % 50 == 0:
                    logger.debug(
                        f"LSTM: Step {step}/{horizon}, raw prediction: {next_value_raw:.6f}, "
                        f"sequence range: [{last_sequence.min():.4f}, "
                        f"{last_sequence.max():.4f}]"
                    )
                
                # Apply stability check: if predictions are too similar, add small variation
                # This prevents the model from getting stuck in a constant prediction
                if len(recent_predictions) >= max_recent:
                    recent_std = np.std(recent_predictions[-max_recent:])
                    if recent_std < 0.01:  # Very low variation - model might be stuck
                        # Add small random variation based on historical volatility
                        hist_std = np.std(train_scaled[-min(60, len(train_scaled)):])
                        noise = np.random.normal(0, hist_std * 0.1)
                        next_value_raw = next_value_raw + noise
                        logger.debug(
                            f"LSTM: Step {step}, detected low variation, "
                            f"adding stability noise: {noise:.6f}"
                        )
                
                # StandardScaler doesn't have hard bounds like MinMaxScaler
                # But we can still apply reasonable clipping to prevent extreme values
                # For log returns, clip to reasonable range (±5% per day is extreme)
                if use_log_returns:
                    next_value_scaled = np.clip(next_value_raw, -0.05, 0.05)
                else:
                    next_value_scaled = next_value_raw
                
                forecast_scaled.append(next_value_scaled)
                recent_predictions.append(next_value_scaled)

                # Update sequence (shift and append)
                # Use the predicted value directly for next prediction
                last_sequence = np.append(
                    last_sequence[1:], next_value_scaled
                )
            
            # Log forecast statistics
            forecast_array = np.array(forecast_scaled)
            logger.debug(
                f"LSTM: Forecast scaled stats - min: {forecast_array.min():.4f}, "
                f"max: {forecast_array.max():.4f}, "
                f"mean: {forecast_array.mean():.4f}, "
                f"std: {forecast_array.std():.4f}, "
                f"unique values: {len(np.unique(forecast_array))}"
            )

            # Inverse transform forecast
            forecast_scaled_array = np.array(forecast_scaled).reshape(-1, 1)
            forecast_unscaled = scaler.inverse_transform(
                forecast_scaled_array
            ).ravel()
            
            # Convert log returns or returns to prices if needed
            # According to guides: BEST APPROACH for trading - forecast log returns
            if use_log_returns:
                # forecast_unscaled contains log returns (unscaled)
                # Convert log returns back to prices: p_t = p_{t-1} * exp(log_return_t)
                # As shown in guide: pred_prices = prices[split:-1] * np.exp(pred_returns)
                last_price = float(self.prices.iloc[-1])
                forecast_prices = [last_price]
                
                for log_ret in forecast_unscaled:
                    # Log return: log(p_t / p_{t-1}) = log(p_t) - log(p_{t-1})
                    # So: p_t = p_{t-1} * exp(log_ret)
                    next_price = forecast_prices[-1] * np.exp(log_ret)
                    
                    # Only apply minimal safety checks - don't clip too aggressively
                    # This allows natural price evolution
                    if not np.isfinite(next_price) or next_price <= 0:
                        next_price = forecast_prices[-1]
                    
                    forecast_prices.append(float(next_price))
                
                forecast_values = np.array(forecast_prices[1:])
                
            elif use_returns:
                # forecast_unscaled contains returns (unscaled)
                last_price = float(self.prices.iloc[-1])
                forecast_prices = [last_price]
                
                for ret in forecast_unscaled:
                    # Limit returns to reasonable range (±10% per period)
                    ret = np.clip(ret, -0.10, 0.10)
                    
                    if np.isfinite(forecast_prices[-1]) and np.isfinite(ret):
                        next_price = forecast_prices[-1] * (1 + ret)
                        
                        if not np.isfinite(next_price) or next_price <= 0:
                            next_price = forecast_prices[-1]
                        
                        forecast_prices.append(float(next_price))
                    else:
                        forecast_prices.append(forecast_prices[-1])
                
                forecast_values = np.array(forecast_prices[1:])
                
            else:
                # forecast_unscaled contains prices (unscaled)
                forecast_values = forecast_unscaled
                
                # Apply minimal bounds only for safety, not to restrict variation
                if len(self.prices) > 0:
                    last_price = float(self.prices.iloc[-1])
                    # Only clip extreme outliers (beyond 3x price range)
                    price_range = float(self.prices.max() - self.prices.min())
                    max_allowed = last_price + 3 * price_range
                    min_allowed = max(last_price - 3 * price_range, 0.01 * last_price)
                    
                    forecast_values = np.clip(forecast_values, min_allowed, max_allowed)
                    
                    # Additional check: remove any inf/nan values
                    valid_mask = np.isfinite(forecast_values)
                    if not np.all(valid_mask):
                        logger.warning("LSTM: Found invalid forecast values, replacing with last price")
                        forecast_values[~valid_mask] = last_price

            # Create forecast dates
            last_date = self.prices.index[-1]
            forecast_dates = self._create_forecast_dates(horizon, last_date)

            # Calculate returns
            if len(forecast_values) > 1:
                forecast_returns = np.diff(forecast_values) / forecast_values[:-1]
                first_return = (
                    (forecast_values[0] - self.prices.iloc[-1])
                    / self.prices.iloc[-1]
                )
                forecast_returns = np.insert(forecast_returns, 0, first_return)
            else:
                first_return = (
                    (forecast_values[0] - self.prices.iloc[-1])
                    / self.prices.iloc[-1]
                )
                forecast_returns = np.array([first_return])

            # Calculate change percentage
            change_pct = self._calculate_change_pct(
                forecast_values, self.prices.iloc[-1]
            )

            # Get residuals (use validation if available)
            if len(X_val) > 0:
                residuals = val_actual - val_pred
            else:
                residuals = train_actual - train_pred

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

            # Get model summary info
            total_params = model.count_params()

            return ForecastResult(
                method="LSTM",
                forecast_dates=forecast_dates,
                forecast_values=forecast_values,
                forecast_returns=forecast_returns,
                confidence_intervals=confidence_intervals,
                final_value=float(forecast_values[-1]),
                change_pct=change_pct,
                model_info={
                    "units": units,
                    "layers": layers,
                    "dropout": dropout,
                    "epochs": epochs,
                    "best_epoch": best_epoch,
                    "batch_size": batch_size,
                    "lookback": lookback,
                    "train_size": train_size,
                    "use_returns": use_returns,
                    "use_log_returns": use_log_returns,
                    "bidirectional": bidirectional,
                    "learning_rate": learning_rate,
                    "train_rmse": float(train_rmse),
                    "val_rmse": float(val_rmse) if val_rmse is not None else None,
                    "directional_accuracy": directional_metrics['directional_accuracy'] if directional_metrics else None,
                    "total_params": total_params,
                    "training_time": training_time,
                },
                residuals=valid_residuals
                if valid_residuals is not None
                else residuals,
                success=True,
                message=(
                    f"LSTM forecast completed "
                    f"(train_rmse={train_rmse:.6f}"
                    + (f", val_rmse={val_rmse:.6f}" if val_rmse is not None else "")
                    + f", {layers} layers, {units} units)"
                ),
            )

        except Exception as e:
            logger.error(f"LSTM forecast failed: {e}", exc_info=True)
            raise CalculationError(f"LSTM forecast failed: {e}") from e

