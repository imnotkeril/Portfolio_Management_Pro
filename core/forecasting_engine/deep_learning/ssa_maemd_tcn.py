"""SSA-MAEMD-TCN hybrid forecaster implementation.

Combines:
- SSA (Singular Spectrum Analysis) for noise reduction
- MA-EMD (Multivariate Adaptive Empirical Mode Decomposition) for decomposition
- TCN (Temporal Convolutional Network) for forecasting decomposed components
"""

import logging
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input,
        Conv1D,
        Activation,
        Add,
        Dropout,
        Dense,
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    tf = None
    keras = None
    Model = None
    Input = None
    Conv1D = None
    Activation = None
    Add = None
    Dropout = None
    Dense = None
    EarlyStopping = None
    ReduceLROnPlateau = None
    MinMaxScaler = None

try:
    from PyEMD import EMD
except ImportError:
    EMD = None

from core.exceptions import CalculationError
from core.forecasting_engine.base import BaseForecaster, ForecastResult
from core.forecasting_engine.utils import calculate_confidence_intervals

logger = logging.getLogger(__name__)


class SSAMAEEMDTCNForecaster(BaseForecaster):
    """
    SSA-MAEMD-TCN hybrid forecaster.

    Combines Singular Spectrum Analysis (SSA), Multivariate Adaptive
    Empirical Mode Decomposition (MA-EMD), and Temporal Convolutional
    Network (TCN) for improved forecasting accuracy through decomposition
    and noise reduction.
    """

    def __init__(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None,
    ) -> None:
        """
        Initialize SSA-MAEMD-TCN forecaster.

        Args:
            prices: Historical prices series
            returns: Optional returns series
        """
        if tf is None or keras is None:
            raise ImportError(
                "TensorFlow/Keras not installed. "
                "Install with: pip install tensorflow scikit-learn"
            )

        if EMD is None:
            raise ImportError(
                "PyEMD not installed. Install with: pip install PyEMD"
            )

        super().__init__(prices, returns)

    def _ssa_decompose(
        self,
        data: np.ndarray,
        window_size: Optional[int] = None,
        n_components: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Singular Spectrum Analysis (SSA) decomposition.

        SSA decomposes time series into trend, seasonal, and noise components.

        Args:
            data: Input time series
            window_size: Embedding window size (default: min(N//3, 50))
            n_components: Number of components to keep (default: window_size//2)

        Returns:
            Tuple of (trend, seasonal, residual)
        """
        N = len(data)
        if N < 10:
            raise CalculationError(f"SSA requires at least 10 data points, got {N}")

        # Default window size
        if window_size is None:
            window_size = min(N // 3, 50, N - 1)
        window_size = max(3, min(window_size, N - 1))

        # Default number of components
        if n_components is None:
            n_components = max(2, window_size // 2)
        n_components = min(n_components, window_size)

        # Step 1: Embedding - create trajectory matrix
        L = window_size
        K = N - L + 1
        X = np.zeros((L, K))

        for i in range(K):
            X[:, i] = data[i : i + L]

        # Step 2: SVD decomposition
        try:
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError as e:
            raise CalculationError(f"SSA SVD failed: {e}") from e

        # Step 3: Grouping - separate components
        # First component (largest singular value) = trend
        # Next few components = seasonal/periodic
        # Remaining = noise/residual

        # Reconstruct trend (first component)
        trend_matrix = np.outer(U[:, 0], s[0] * Vt[0, :])
        trend = self._diagonal_averaging(trend_matrix)

        # Reconstruct seasonal (next few components)
        seasonal_matrix = np.zeros((L, K))
        n_seasonal = min(3, len(s) - 1)  # Use next 3 components for seasonal
        for i in range(1, n_seasonal + 1):
            if i < len(s):
                seasonal_matrix += np.outer(U[:, i], s[i] * Vt[i, :])

        seasonal = self._diagonal_averaging(seasonal_matrix)

        # Residual (remaining components)
        residual_matrix = X - trend_matrix - seasonal_matrix
        residual = self._diagonal_averaging(residual_matrix)

        # Ensure same length
        min_len = min(len(trend), len(seasonal), len(residual), N)
        trend = trend[:min_len]
        seasonal = seasonal[:min_len]
        residual = residual[:min_len]

        return trend, seasonal, residual

    def _diagonal_averaging(self, matrix: np.ndarray) -> np.ndarray:
        """
        Perform diagonal averaging (Hankelization) to convert matrix to time series.

        Args:
            matrix: Trajectory matrix

        Returns:
            Time series reconstructed from matrix
        """
        L, K = matrix.shape
        N = L + K - 1
        result = np.zeros(N)

        for i in range(N):
            # Sum along anti-diagonal
            total = 0.0
            count = 0
            for j in range(max(0, i - K + 1), min(L, i + 1)):
                k = i - j
                if 0 <= k < K:
                    total += matrix[j, k]
                    count += 1
            if count > 0:
                result[i] = total / count

        return result

    def _emd_decompose(
        self,
        data: np.ndarray,
        max_imf: int = 10,
    ) -> List[np.ndarray]:
        """
        Perform Empirical Mode Decomposition (EMD).

        Decomposes signal into Intrinsic Mode Functions (IMFs).

        Args:
            data: Input signal
            max_imf: Maximum number of IMFs to extract

        Returns:
            List of IMFs (last element is residual)
        """
        if EMD is None:
            raise ImportError("PyEMD not installed")

        try:
            emd = EMD()
            emd.emd(data, max_imf=max_imf)
            imfs = emd.imfs
            residue = emd.residue

            # Combine IMFs and residue
            components = []
            for imf in imfs:
                if len(imf) > 0:
                    components.append(imf)
            if len(residue) > 0:
                components.append(residue)

            # Filter out components that are too small or constant
            filtered_components = []
            for comp in components:
                if len(comp) > 0 and np.std(comp) > 1e-10:
                    filtered_components.append(comp)

            if len(filtered_components) == 0:
                # Fallback: return original data as single component
                return [data]

            return filtered_components

        except Exception as e:
            logger.warning(f"EMD decomposition failed: {e}, using original data")
            return [data]

    def _tcn_block(
        self,
        x,
        filters: int,
        kernel_size: int,
        dilation_rate: int,
        dropout: float,
        block_num: int,
    ):
        """Create a TCN residual block (same as in tcn.py)."""
        # Causal dilated convolution
        conv1 = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            name=f"ssa_tcn_conv1_block_{block_num}",
        )(x)
        conv1 = Activation("relu", name=f"ssa_tcn_act1_block_{block_num}")(conv1)
        conv1 = Dropout(dropout, name=f"ssa_tcn_dropout1_block_{block_num}")(conv1)

        # Second convolution
        conv2 = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            name=f"ssa_tcn_conv2_block_{block_num}",
        )(conv1)
        conv2 = Activation("relu", name=f"ssa_tcn_act2_block_{block_num}")(conv2)
        conv2 = Dropout(dropout, name=f"ssa_tcn_dropout2_block_{block_num}")(conv2)

        # Residual connection
        if x.shape[-1] != filters:
            residual = Conv1D(
                filters=filters,
                kernel_size=1,
                padding="same",
                name=f"ssa_tcn_residual_block_{block_num}",
            )(x)
        else:
            residual = x

        output = Add(name=f"ssa_tcn_add_block_{block_num}")([conv2, residual])
        output = Activation("relu", name=f"ssa_tcn_output_block_{block_num}")(output)

        return output

    def _build_tcn_model(
        self,
        input_shape: Tuple[int, int],
        num_filters: int,
        kernel_size: int,
        num_blocks: int,
        dropout: float,
        learning_rate: float,
    ) -> Model:
        """Build TCN model (same as in tcn.py)."""
        inputs = Input(shape=input_shape, name="ssa_tcn_input")

        x = inputs

        # Build TCN blocks
        for i in range(num_blocks):
            dilation_rate = 2 ** i
            x = self._tcn_block(
                x,
                filters=num_filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                dropout=dropout,
                block_num=i,
            )

        # Take last timestep
        x = x[:, -1, :]

        # Output layer
        outputs = Dense(1, name="ssa_tcn_output")(x)

        model = Model(inputs=inputs, outputs=outputs, name="SSA_TCN_Model")

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def _create_sequences(
        self,
        data: np.ndarray,
        lookback: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for TCN training."""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback : i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def forecast(
        self,
        horizon: int,
        ssa_window_size: Optional[int] = None,
        ssa_n_components: Optional[int] = None,
        emd_max_imf: int = 8,
        tcn_num_filters: int = 64,
        tcn_kernel_size: int = 3,
        tcn_num_blocks: int = 4,
        tcn_dropout: float = 0.2,
        tcn_epochs: int = 50,
        tcn_batch_size: int = 32,
        tcn_lookback: int = 20,
        tcn_train_size: float = 0.8,
        tcn_learning_rate: float = 0.001,
        tcn_early_stopping_patience: int = 10,
        use_returns: bool = True,
        **kwargs,
    ) -> ForecastResult:
        """
        Generate forecast using SSA-MAEMD-TCN hybrid model.

        Process:
        1. Apply SSA to decompose into trend, seasonal, residual
        2. Apply EMD to residual to get IMFs
        3. Train separate TCN for each component (trend, seasonal, each IMF)
        4. Forecast each component
        5. Reconstruct final forecast by summing components

        Args:
            horizon: Number of periods to forecast
            ssa_window_size: SSA embedding window size
            ssa_n_components: SSA number of components
            emd_max_imf: Maximum number of IMFs for EMD
            tcn_num_filters: TCN filters per block
            tcn_kernel_size: TCN kernel size
            tcn_num_blocks: TCN number of blocks
            tcn_dropout: TCN dropout rate
            tcn_epochs: TCN training epochs
            tcn_batch_size: TCN batch size
            tcn_lookback: TCN sequence length
            tcn_train_size: TCN train/val split
            tcn_learning_rate: TCN learning rate
            tcn_early_stopping_patience: TCN early stopping patience
            use_returns: Whether to forecast returns (True) or prices (False)
            **kwargs: Additional parameters

        Returns:
            ForecastResult with forecasted values
        """
        if tf is None or keras is None:
            raise ImportError("TensorFlow/Keras not installed")

        if EMD is None:
            raise ImportError("PyEMD not installed")

        start_time = time.time()

        try:
            # Prepare data
            if use_returns:
                data = self.returns.dropna().values
                if len(data) < 50:
                    logger.warning("Insufficient returns data, falling back to prices")
                    use_returns = False
                    data = self.prices.values
            else:
                data = self.prices.values

            if len(data) < 50:
                raise CalculationError(
                    f"Insufficient data: {len(data)} samples (need at least 50)"
                )

            logger.debug("Step 1: Applying SSA decomposition...")
            # Step 1: SSA decomposition
            trend, seasonal, residual = self._ssa_decompose(
                data,
                window_size=ssa_window_size,
                n_components=ssa_n_components,
            )

            logger.debug(
                f"SSA: Trend std={np.std(trend):.6f}, "
                f"Seasonal std={np.std(seasonal):.6f}, "
                f"Residual std={np.std(residual):.6f}"
            )

            # Step 2: EMD decomposition of residual
            logger.debug("Step 2: Applying EMD decomposition to residual...")
            imfs = self._emd_decompose(residual, max_imf=emd_max_imf)

            logger.debug(f"EMD: Extracted {len(imfs)} IMF components")

            # Prepare all components for forecasting
            components = [trend, seasonal] + imfs
            component_names = ["trend", "seasonal"] + [f"imf_{i}" for i in range(len(imfs))]

            # Normalize each component
            scalers = []
            components_scaled = []
            for comp in components:
                scaler = MinMaxScaler(feature_range=(0, 1))
                comp_scaled = scaler.fit_transform(comp.reshape(-1, 1)).ravel()
                scalers.append(scaler)
                components_scaled.append(comp_scaled)

            # Step 3: Train TCN for each component and forecast
            logger.debug("Step 3: Training TCN models for each component...")
            component_forecasts = []

            for i, (comp_scaled, scaler, comp_name) in enumerate(
                zip(components_scaled, scalers, component_names)
            ):
                try:
                    # Create sequences
                    X, y = self._create_sequences(comp_scaled, tcn_lookback)

                    if len(X) < 30:
                        logger.warning(
                            f"Component {comp_name}: insufficient sequences, "
                            "using simple extrapolation"
                        )
                        # Simple extrapolation
                        last_val = comp_scaled[-1]
                        forecast_scaled = np.full(horizon, last_val)
                    else:
                        # Train/val split
                        split_idx = int(len(X) * tcn_train_size)
                        if split_idx < 20:
                            X_train, X_val = X, X
                            y_train, y_val = y, y
                        else:
                            X_train, X_val = X[:split_idx], X[split_idx:]
                            y_train, y_val = y[:split_idx], y[split_idx:]

                        # Reshape for TCN
                        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                        if len(X_val) > 0:
                            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

                        # Build and train TCN
                        input_shape = (tcn_lookback, 1)
                        model = self._build_tcn_model(
                            input_shape=input_shape,
                            num_filters=tcn_num_filters,
                            kernel_size=tcn_kernel_size,
                            num_blocks=tcn_num_blocks,
                            dropout=tcn_dropout,
                            learning_rate=tcn_learning_rate,
                        )

                        # Callbacks
                        callbacks = []
                        if len(X_val) > 0 and tcn_early_stopping_patience > 0:
                            callbacks.append(
                                EarlyStopping(
                                    monitor="val_loss",
                                    patience=tcn_early_stopping_patience,
                                    restore_best_weights=True,
                                    verbose=0,
                                )
                            )

                        # Train
                        validation_data = (
                            (X_val, y_val) if len(X_val) > 0 else None
                        )
                        model.fit(
                            X_train,
                            y_train,
                            epochs=tcn_epochs,
                            batch_size=tcn_batch_size,
                            validation_data=validation_data,
                            callbacks=callbacks,
                            verbose=0,
                        )

                        # Forecast iteratively
                        forecast_scaled = []
                        last_sequence = comp_scaled[-tcn_lookback:].copy()

                        for _ in range(horizon):
                            X_pred = last_sequence.reshape((1, tcn_lookback, 1))
                            next_val = model.predict(X_pred, verbose=0)[0, 0]
                            forecast_scaled.append(next_val)
                            last_sequence = np.append(last_sequence[1:], next_val)

                        forecast_scaled = np.array(forecast_scaled)

                    # Inverse transform
                    forecast_scaled_2d = forecast_scaled.reshape(-1, 1)
                    forecast_comp = scaler.inverse_transform(forecast_scaled_2d).ravel()
                    component_forecasts.append(forecast_comp)

                except Exception as e:
                    logger.warning(
                        f"Component {comp_name} forecasting failed: {e}, "
                        "using simple extrapolation"
                    )
                    # Fallback: simple extrapolation
                    last_val = components[i][-1]
                    forecast_comp = np.full(horizon, last_val)
                    component_forecasts.append(forecast_comp)

            # Step 4: Reconstruct final forecast
            logger.debug("Step 4: Reconstructing final forecast...")
            forecast_values = np.sum(component_forecasts, axis=0)

            # Convert returns to prices if needed
            if use_returns:
                last_price = float(self.prices.iloc[-1])
                forecast_prices = [last_price]
                for ret in forecast_values:
                    next_price = forecast_prices[-1] * (1 + ret)
                    forecast_prices.append(next_price)
                forecast_values = np.array(forecast_prices[1:])

            # Ensure positive prices
            forecast_values = np.maximum(
                forecast_values, 0.01 * self.prices.iloc[-1]
            )

            # Create forecast dates
            last_date = self.prices.index[-1]
            forecast_dates = self._create_forecast_dates(horizon, last_date)

            # Calculate returns
            if len(forecast_values) > 1:
                forecast_returns = np.diff(forecast_values) / forecast_values[:-1]
                first_return = (
                    (forecast_values[0] - self.prices.iloc[-1]) / self.prices.iloc[-1]
                )
                forecast_returns = np.insert(forecast_returns, 0, first_return)
            else:
                first_return = (
                    (forecast_values[0] - self.prices.iloc[-1]) / self.prices.iloc[-1]
                )
                forecast_returns = np.array([first_return])

            # Calculate change percentage
            change_pct = self._calculate_change_pct(
                forecast_values, self.prices.iloc[-1]
            )

            # Calculate residuals (use trend component residuals as proxy)
            residuals = None
            try:
                # Use trend component for residuals estimation
                trend_X, trend_y = self._create_sequences(
                    components_scaled[0], tcn_lookback
                )
                if len(trend_X) > 0:
                    # Simple residual estimation
                    trend_pred = components_scaled[0][tcn_lookback:]
                    trend_actual = components_scaled[0][tcn_lookback:]
                    residuals = trend_actual - trend_pred
                    residuals = residuals[np.isfinite(residuals) & ~np.isnan(residuals)]
            except Exception as e:
                logger.debug(f"Could not calculate residuals: {e}")

            # Calculate confidence intervals
            confidence_intervals = calculate_confidence_intervals(
                forecast_values,
                residuals=residuals,
                confidence_level=0.95,
            )

            training_time = time.time() - start_time

            return ForecastResult(
                method="SSA-MAEMD-TCN",
                forecast_dates=forecast_dates,
                forecast_values=forecast_values,
                forecast_returns=forecast_returns,
                confidence_intervals=confidence_intervals,
                final_value=float(forecast_values[-1]),
                change_pct=change_pct,
                model_info={
                    "ssa_window_size": ssa_window_size,
                    "ssa_n_components": ssa_n_components,
                    "emd_max_imf": emd_max_imf,
                    "n_components": len(components),
                    "component_names": component_names,
                    "tcn_num_filters": tcn_num_filters,
                    "tcn_kernel_size": tcn_kernel_size,
                    "tcn_num_blocks": tcn_num_blocks,
                    "tcn_dropout": tcn_dropout,
                    "tcn_epochs": tcn_epochs,
                    "tcn_batch_size": tcn_batch_size,
                    "tcn_lookback": tcn_lookback,
                    "use_returns": use_returns,
                    "training_time": training_time,
                },
                residuals=residuals,
                success=True,
                message=(
                    f"SSA-MAEMD-TCN forecast completed "
                    f"({len(components)} components, {len(imfs)} IMFs)"
                ),
            )

        except Exception as e:
            logger.error(f"SSA-MAEMD-TCN forecast failed: {e}", exc_info=True)
            raise CalculationError(f"SSA-MAEMD-TCN forecast failed: {e}") from e

