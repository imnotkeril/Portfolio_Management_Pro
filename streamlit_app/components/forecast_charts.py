"""Forecast visualization components."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from streamlit_app.utils.chart_config import (
    COLORS,
    get_chart_layout,
    get_method_color,
)

logger = logging.getLogger(__name__)


def _get_last_historical_point(
    historical_dates: pd.DatetimeIndex,
    historical_values: np.ndarray,
    validation_end: Optional[pd.Timestamp],
) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
    """
    Get last historical point at or before validation_end.
    
    Args:
        historical_dates: Historical dates
        historical_values: Historical values
        validation_end: Optional validation period end date
        
    Returns:
        Tuple of (last_historical_date, last_historical_value)
    """
    if len(historical_dates) == 0 or len(historical_values) == 0:
        return None, None
    
    # Ensure dates and values are aligned
    min_len = min(len(historical_dates), len(historical_values))
    if min_len == 0:
        return None, None
    
    historical_dates = historical_dates[:min_len]
    historical_values = historical_values[:min_len]
    
    # Filter out invalid values
    valid_mask = np.isfinite(historical_values)
    if not np.any(valid_mask):
        return None, None
    
    historical_dates = historical_dates[valid_mask]
    historical_values = historical_values[valid_mask]
    
    # Filter historical data to end at validation_end if provided
    if validation_end:
        try:
            validation_end_ts = pd.Timestamp(validation_end)
            # Normalize timezone for comparison
            if hasattr(validation_end_ts, "tz") and validation_end_ts.tz is not None:
                validation_end_ts = validation_end_ts.tz_localize(None)
            
            # Normalize historical dates timezone
            if hasattr(historical_dates, "tz") and historical_dates.tz is not None:
                historical_dates = historical_dates.tz_localize(None)
            
            historical_mask = historical_dates <= validation_end_ts
            historical_dates_filtered = historical_dates[historical_mask]
            historical_values_filtered = historical_values[historical_mask]
            
            if len(historical_dates_filtered) > 0:
                last_date = pd.Timestamp(historical_dates_filtered[-1])
                last_value = float(historical_values_filtered[-1])
                if np.isfinite(last_value):
                    return last_date, last_value
                else:
                    # Find last valid value
                    valid_idx = np.where(np.isfinite(historical_values_filtered))[0]
                    if len(valid_idx) > 0:
                        return (
                            pd.Timestamp(historical_dates_filtered[valid_idx[-1]]),
                            float(historical_values_filtered[valid_idx[-1]])
                        )
        except (ValueError, TypeError) as e:
            logger.warning(f"Error processing validation_end: {e}")
    
    # No validation_end or error, use last historical point
    if len(historical_dates) > 0 and len(historical_values) > 0:
        last_date = pd.Timestamp(historical_dates[-1])
        last_value = float(historical_values[-1])
        if np.isfinite(last_value):
            return last_date, last_value
        else:
            # Find last valid value
            valid_idx = np.where(np.isfinite(historical_values))[0]
            if len(valid_idx) > 0:
                return (
                    pd.Timestamp(historical_dates[valid_idx[-1]]),
                    float(historical_values[valid_idx[-1]])
                )
    
    return None, None


def plot_forecast_comparison(
    historical_dates: pd.DatetimeIndex,
    historical_values: np.ndarray,
    forecasts: Dict[str, Dict],
    validation_start: Optional[pd.Timestamp] = None,
    validation_end: Optional[pd.Timestamp] = None,
    training_start: Optional[pd.Timestamp] = None,
    forecast_end: Optional[pd.Timestamp] = None,
) -> go.Figure:
    """
    Plot comparison of all forecasts with historical data.

    Args:
        historical_dates: Historical dates
        historical_values: Historical values (prices)
        forecasts: Dictionary mapping method name to forecast data
        validation_start: Optional validation period start (start_date)
        validation_end: Optional validation period end (end_date)
        training_start: Optional training period start
        forecast_end: Optional forecast end date (end of forecast horizon)

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # 1. HISTORICAL DATA (Purple solid line)
    # Historical data should end at validation_end (end_date)
    last_historical_date, last_historical_value = _get_last_historical_point(
        historical_dates, historical_values, validation_end
    )
    
    if last_historical_date is not None and last_historical_value is not None:
        # Filter historical data to end at validation_end if provided
        if validation_end:
            validation_end_ts = pd.Timestamp(validation_end)
            historical_mask = historical_dates <= validation_end_ts
            historical_dates_filtered = historical_dates[historical_mask]
            historical_values_filtered = historical_values[historical_mask]
        else:
            historical_dates_filtered = historical_dates
            historical_values_filtered = historical_values

        if len(historical_dates_filtered) > 0:
            fig.add_trace(go.Scatter(
                x=historical_dates_filtered,
                y=historical_values_filtered,
                name="Historical",
                line=dict(color=COLORS["primary"], width=2),
                mode="lines",
            ))

    # 2. FORECASTS
    all_forecast_values = []

    for method_name, forecast_data in forecasts.items():
        if not forecast_data.get("success", False):
            continue

        # Safely convert forecast dates
        try:
            forecast_dates_raw = forecast_data["forecast_dates"]
            if isinstance(forecast_dates_raw, str):
                forecast_dates = pd.to_datetime([forecast_dates_raw])
            elif isinstance(forecast_dates_raw, (list, tuple)):
                forecast_dates = pd.to_datetime(forecast_dates_raw, errors="coerce")
            else:
                forecast_dates = pd.to_datetime(forecast_dates_raw, errors="coerce")
            
            # Normalize timezone
            if hasattr(forecast_dates, "tz") and forecast_dates.tz is not None:
                forecast_dates = forecast_dates.tz_localize(None)
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing forecast_dates for {method_name}: {e}")
            forecast_dates = pd.DatetimeIndex([])
        
        # Safely convert forecast values
        try:
            forecast_values_raw = forecast_data["forecast_values"]
            if isinstance(forecast_values_raw, (list, tuple)):
                forecast_values = np.array(forecast_values_raw, dtype=float)
            elif isinstance(forecast_values_raw, np.ndarray):
                forecast_values = forecast_values_raw.astype(float)
            else:
                forecast_values = np.array([forecast_values_raw], dtype=float)
            
            # Filter invalid values
            valid_mask = np.isfinite(forecast_values)
            if np.any(valid_mask):
                forecast_values = forecast_values[valid_mask]
                if len(forecast_dates) > 0:
                    forecast_dates = forecast_dates[valid_mask[:len(forecast_dates)]]
            else:
                forecast_values = np.array([])
                forecast_dates = pd.DatetimeIndex([])
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing forecast_values for {method_name}: {e}")
            forecast_values = np.array([])
        
        method_color = get_method_color(method_name)

        if validation_end and validation_start:
            try:
                validation_end_ts = pd.Timestamp(validation_end)
                validation_start_ts = pd.Timestamp(validation_start)
                
                # Normalize timezones
                if hasattr(validation_end_ts, "tz") and validation_end_ts.tz is not None:
                    validation_end_ts = validation_end_ts.tz_localize(None)
                if hasattr(validation_start_ts, "tz") and validation_start_ts.tz is not None:
                    validation_start_ts = validation_start_ts.tz_localize(None)

                # Split forecast dates
                # Forecast period: from validation_end onwards (inclusive of validation_end)
                # This includes the first point which should be at validation_end
                # Note: forecast_mask is not used here, only validation_mask_strict

                # 2a. VALIDATION PERIOD FORECAST (Solid line)
                # Exclude the last point (at validation_end) from validation, as it belongs to forecast period
                validation_mask_strict = (
                    (forecast_dates >= validation_start_ts) &
                    (forecast_dates < validation_end_ts)  # Strictly less than validation_end
                )
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing validation dates for {method_name}: {e}")
                validation_mask_strict = pd.Series([False] * len(forecast_dates), index=forecast_dates)
            if validation_mask_strict.any():
                validation_dates = forecast_dates[validation_mask_strict]
                validation_values = forecast_values[validation_mask_strict]
                all_forecast_values.extend(validation_values.tolist())

                fig.add_trace(go.Scatter(
                    x=validation_dates,
                    y=validation_values,
                    name=f"{method_name} Forecast",
                    line=dict(color=method_color, width=2),
                    mode="lines",
                    showlegend=True,
                ))

            # Forecast Period not displayed - only Validation Period for testing

        else:
            # No validation period
            all_forecast_values.extend(forecast_values.tolist())

            if last_historical_date and last_historical_value is not None:
                forecast_dates = pd.DatetimeIndex(
                    [last_historical_date]
                ).append(forecast_dates)
                forecast_values = np.concatenate([
                    [last_historical_value],
                    forecast_values
                ])

            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                name=f"{method_name} Forecast",
                line=dict(color=method_color, width=2),
                mode="lines",
            ))

    # 3. PERIOD SHADING
    if training_start and validation_start:
        training_start_ts = pd.Timestamp(training_start)
        validation_start_ts = pd.Timestamp(validation_start)
        fig.add_vrect(
            x0=training_start_ts,
            x1=validation_start_ts,
            fillcolor="rgba(0, 150, 255, 0.15)",
            layer="below",
            annotation_text="Training Period",
            annotation_position="top left",
        )

    if validation_start and validation_end:
        validation_start_ts = pd.Timestamp(validation_start)
        validation_end_ts = pd.Timestamp(validation_end)
        fig.add_vrect(
            x0=validation_start_ts,
            x1=validation_end_ts,
            fillcolor="rgba(255, 0, 0, 0.1)",
            layer="below",
            annotation_text="Validation Period (Test)",
            annotation_position="top left",
        )

    # Forecast Period not displayed - only Validation Period for testing

    # 5. LAYOUT
    layout = get_chart_layout()
    
    # X-axis: EXACTLY from training_start to forecast_end (WITHOUT padding)
    chart_start = None
    if training_start:
        chart_start = pd.Timestamp(training_start)
    elif len(historical_dates) > 0:
        chart_start = pd.Timestamp(historical_dates[0])

    # Prepare xaxis settings before update
    xaxis_settings = {}
    chart_end = None
    if validation_end:
        chart_end = pd.Timestamp(validation_end)
    elif forecast_end:
        chart_end = pd.Timestamp(forecast_end)
    elif len(historical_dates) > 0:
        chart_end = pd.Timestamp(historical_dates[-1])
    
    if chart_start and chart_end:
        # WITHOUT padding - chart starts and ends exactly (up to validation_end)
        xaxis_settings["range"] = [chart_start, chart_end]
        xaxis_settings["autorange"] = False
    
    # Update layout with title and axis labels
    layout.update(
        title="Price Forecasts Comparison",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
    )
    
    # Update xaxis settings after layout.update to ensure they're not overwritten
    if xaxis_settings:
        if "xaxis" not in layout or layout["xaxis"] is None:
            layout["xaxis"] = {}
        elif not isinstance(layout["xaxis"], dict):
            layout["xaxis"] = dict(layout["xaxis"])
        layout["xaxis"].update(xaxis_settings)

    # Y-axis: from minimum (or 0) to maximum with padding
    all_values = list(historical_values) + all_forecast_values
    # Filter out invalid values
    all_values = [v for v in all_values if np.isfinite(v)]
    if all_values:
        min_value = float(np.min(all_values))
        max_value = float(np.max(all_values))
        if max_value > min_value:
            value_range = max_value - min_value
            padding = value_range * 0.1
        else:
            # If all values are the same, add small padding
            padding = abs(min_value) * 0.1 if min_value != 0 else 1.0
        layout["yaxis"]["range"] = [
            max(0, min_value - padding),
            max_value + padding,
        ]

    fig.update_layout(layout)

    return fig


def plot_individual_forecast(
    historical_dates: pd.DatetimeIndex,
    historical_values: np.ndarray,
    forecast_data: Dict,
    method_name: str,
    validation_start: Optional[pd.Timestamp] = None,
    validation_end: Optional[pd.Timestamp] = None,
    training_start: Optional[pd.Timestamp] = None,
    forecast_end: Optional[pd.Timestamp] = None,
) -> go.Figure:
    """
    Plot individual forecast with confidence intervals.

    Args:
        historical_dates: Historical dates
        historical_values: Historical values
        forecast_data: Forecast result dictionary
        method_name: Method name for title
        validation_start: Optional validation period start
        validation_end: Optional validation period end
        training_start: Optional training period start
        forecast_end: Optional forecast end date

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    if not forecast_data.get("success", False):
        logger.warning(f"Forecast data for {method_name} is not successful")
        # Still try to show what we have
        layout = get_chart_layout()
        layout.update(
            title=f"{method_name} Forecast (Unsuccessful)",
            xaxis_title="Date",
            yaxis_title="Price",
        )
        fig.update_layout(layout)
        return fig

    # Check if forecast_data has required keys
    if "forecast_dates" not in forecast_data or "forecast_values" not in forecast_data:
        logger.warning(f"Forecast data for {method_name} missing forecast_dates or forecast_values")
        layout = get_chart_layout()
        layout.update(
            title=f"{method_name} Forecast (No Data)",
            xaxis_title="Date",
            yaxis_title="Price",
        )
        fig.update_layout(layout)
        return fig
    
    # 1. HISTORICAL DATA
    # Historical data should end at validation_end (end_date)
    last_historical_date, last_historical_value = _get_last_historical_point(
        historical_dates, historical_values, validation_end
    )
    
    if last_historical_date is not None and last_historical_value is not None:
        # Filter historical data to end at validation_end if provided
        if validation_end:
            validation_end_ts = pd.Timestamp(validation_end)
            historical_mask = historical_dates <= validation_end_ts
            historical_dates_filtered = historical_dates[historical_mask]
            historical_values_filtered = historical_values[historical_mask]
        else:
            historical_dates_filtered = historical_dates
            historical_values_filtered = historical_values

        if len(historical_dates_filtered) > 0:
            fig.add_trace(go.Scatter(
                x=historical_dates_filtered,
                y=historical_values_filtered,
                name="Historical",
                line=dict(color=COLORS["primary"], width=2),
                mode="lines",
            ))

    # 2. FORECAST
    # Safely convert forecast dates
    try:
        forecast_dates_raw = forecast_data["forecast_dates"]
        if isinstance(forecast_dates_raw, str):
            forecast_dates = pd.to_datetime([forecast_dates_raw])
        elif isinstance(forecast_dates_raw, (list, tuple)):
            forecast_dates = pd.to_datetime(forecast_dates_raw, errors="coerce")
        else:
            forecast_dates = pd.to_datetime(forecast_dates_raw, errors="coerce")
        
        # Normalize timezone
        if hasattr(forecast_dates, "tz") and forecast_dates.tz is not None:
            forecast_dates = forecast_dates.tz_localize(None)
    except (ValueError, TypeError) as e:
        logger.warning(f"Error parsing forecast_dates for {method_name}: {e}")
        forecast_dates = pd.DatetimeIndex([])
    
    # Safely convert forecast values
    try:
        forecast_values_raw = forecast_data["forecast_values"]
        if isinstance(forecast_values_raw, (list, tuple)):
            forecast_values = np.array(forecast_values_raw, dtype=float)
        elif isinstance(forecast_values_raw, np.ndarray):
            forecast_values = forecast_values_raw.astype(float)
        else:
            forecast_values = np.array([forecast_values_raw], dtype=float)
        
        # Filter invalid values
        valid_mask = np.isfinite(forecast_values)
        if np.any(valid_mask):
            forecast_values = forecast_values[valid_mask]
            if len(forecast_dates) > 0:
                # Align dates with values
                min_len = min(len(forecast_dates), len(valid_mask))
                forecast_dates = forecast_dates[:min_len][valid_mask[:min_len]]
        else:
            forecast_values = np.array([])
            forecast_dates = pd.DatetimeIndex([])
    except (ValueError, TypeError) as e:
        logger.warning(f"Error parsing forecast_values for {method_name}: {e}")
        forecast_values = np.array([])
    
    # Debug: check if forecast data is empty
    if len(forecast_dates) == 0 or len(forecast_values) == 0:
        logger.warning(f"Forecast data for {method_name} is empty: dates={len(forecast_dates)}, values={len(forecast_values)}")
        layout = get_chart_layout()
        layout.update(
            title=f"{method_name} Forecast (Empty Data)",
            xaxis_title="Date",
            yaxis_title="Price",
        )
        fig.add_annotation(
            text="No forecast data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(layout)
        return fig
    
    method_color = get_method_color(method_name)

    # Always show forecast data - use validation period if available, otherwise show all
    if validation_end and validation_start:
        try:
            validation_end_ts = pd.Timestamp(validation_end)
            validation_start_ts = pd.Timestamp(validation_start)
            
            # Normalize timezones
            if hasattr(validation_end_ts, "tz") and validation_end_ts.tz is not None:
                validation_end_ts = validation_end_ts.tz_localize(None)
            if hasattr(validation_start_ts, "tz") and validation_start_ts.tz is not None:
                validation_start_ts = validation_start_ts.tz_localize(None)

            # Validation period: from validation_start to validation_end (inclusive of validation_end)
            # Since we only have Validation Period now, use all forecast dates that fall in this range
            validation_mask = (
                (forecast_dates >= validation_start_ts) &
                (forecast_dates <= validation_end_ts)  # Include validation_end
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Error processing validation dates for {method_name}: {e}")
            validation_mask = pd.Series([True] * len(forecast_dates), index=forecast_dates)

        # 2a. VALIDATION PERIOD FORECAST (Solid line)
        if validation_mask.any():
            validation_dates = forecast_dates[validation_mask]
            validation_values = forecast_values[validation_mask]

            fig.add_trace(go.Scatter(
                x=validation_dates,
                y=validation_values,
                name=f"{method_name} Forecast",
                line=dict(color=method_color, width=2),
                mode="lines",
                showlegend=True,
            ))
        else:
            # Fallback: if no validation period match, show all forecast data
            logger.warning(
                f"No forecast data matches validation period for {method_name}. "
                f"Forecast dates range: {forecast_dates.min()} to {forecast_dates.max()}. "
                f"Validation period: {validation_start_ts} to {validation_end_ts}. "
                f"Showing all forecast data."
            )
            if len(forecast_dates) > 0:
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_values,
                    name=f"{method_name} Forecast",
                    line=dict(color=method_color, width=2),
                    mode="lines",
                    showlegend=True,
                ))
    else:
        # No validation period - show all forecast data
        if len(forecast_dates) > 0:
            # If we have historical data, connect forecast to it
            if last_historical_date and last_historical_value is not None:
                # Connect forecast to historical data
                connected_dates = pd.DatetimeIndex(
                    [last_historical_date]
                ).append(forecast_dates)
                connected_values = np.concatenate([
                    [last_historical_value],
                    forecast_values
                ])
                fig.add_trace(go.Scatter(
                    x=connected_dates,
                    y=connected_values,
                    name=f"{method_name} Forecast",
                    line=dict(color=method_color, width=2),
                    mode="lines",
                    showlegend=True,
                ))
            else:
                # No historical data, just show forecast
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_values,
                    name=f"{method_name} Forecast",
                    line=dict(color=method_color, width=2),
                    mode="lines",
                    showlegend=True,
                ))

    # 3. PERIOD SHADING
    if training_start and validation_start:
        training_start_ts = pd.Timestamp(training_start)
        validation_start_ts = pd.Timestamp(validation_start)
        fig.add_vrect(
            x0=training_start_ts,
            x1=validation_start_ts,
            fillcolor="rgba(0, 150, 255, 0.15)",
            layer="below",
            annotation_text="Training Period",
            annotation_position="top left",
        )

    if validation_start and validation_end:
        validation_start_ts = pd.Timestamp(validation_start)
        validation_end_ts = pd.Timestamp(validation_end)
        fig.add_vrect(
            x0=validation_start_ts,
            x1=validation_end_ts,
            fillcolor="rgba(255, 0, 0, 0.1)",
            layer="below",
            annotation_text="Validation Period (Test)",
            annotation_position="top left",
        )

    # Forecast Period not displayed - only Validation Period for testing

    # 5. LAYOUT
    layout = get_chart_layout()
    
    # X-axis: EXACTLY from training_start to validation_end (WITHOUT padding, without Forecast Period)
    chart_start = None
    if training_start:
        chart_start = pd.Timestamp(training_start)
    elif len(historical_dates) > 0:
        chart_start = pd.Timestamp(historical_dates[0])

    # Prepare xaxis settings before update
    xaxis_settings = {}
    chart_end = None
    if validation_end:
        chart_end = pd.Timestamp(validation_end)
    elif forecast_end:
        chart_end = pd.Timestamp(forecast_end)
    elif len(historical_dates) > 0:
        chart_end = pd.Timestamp(historical_dates[-1])
    
    if chart_start and chart_end:
        # WITHOUT padding - chart starts and ends exactly (up to validation_end)
        xaxis_settings["range"] = [chart_start, chart_end]
        xaxis_settings["autorange"] = False
    
    # Update layout with title and axis labels
    layout.update(
        title=f"{method_name} Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
    )
    
    # Update xaxis settings after layout.update to ensure they're not overwritten
    if xaxis_settings:
        if "xaxis" not in layout or layout["xaxis"] is None:
            layout["xaxis"] = {}
        elif not isinstance(layout["xaxis"], dict):
            layout["xaxis"] = dict(layout["xaxis"])
        layout["xaxis"].update(xaxis_settings)

    # Y-axis: from minimum (or 0) to maximum with padding
    all_values = list(historical_values) if len(historical_values) > 0 else []
    # Filter out invalid historical values
    all_values = [v for v in all_values if np.isfinite(v)]
    
    if validation_end and validation_start:
        if len(forecast_values) > 0 and len(forecast_dates) > 0:
            try:
                validation_start_ts = pd.Timestamp(validation_start)
                validation_end_ts = pd.Timestamp(validation_end)
                # Normalize timezones
                if hasattr(validation_start_ts, "tz") and validation_start_ts.tz is not None:
                    validation_start_ts = validation_start_ts.tz_localize(None)
                if hasattr(validation_end_ts, "tz") and validation_end_ts.tz is not None:
                    validation_end_ts = validation_end_ts.tz_localize(None)
                
                # Validation period: from validation_start to validation_end (inclusive)
                validation_mask = (
                    (forecast_dates >= validation_start_ts) &
                    (forecast_dates <= validation_end_ts)
                )
                if validation_mask.any():
                    validation_values = forecast_values[validation_mask]
                    all_values.extend([v for v in validation_values if np.isfinite(v)])
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing validation period for y-axis: {e}")
    else:
        if len(forecast_values) > 0:
            all_values.extend([v for v in forecast_values if np.isfinite(v)])

    if all_values:
        min_value = float(np.min(all_values))
        max_value = float(np.max(all_values))
        if max_value > min_value:
            value_range = max_value - min_value
            padding = value_range * 0.1
        else:
            # If all values are the same, add small padding
            padding = abs(min_value) * 0.1 if min_value != 0 else 1.0
        layout["yaxis"]["range"] = [
            max(0, min_value - padding),
            max_value + padding,
        ]

    fig.update_layout(layout)

    return fig


def plot_forecast_quality(
    forecasts: Dict[str, Dict],
    metric: str,
) -> go.Figure:
    """
    Plot quality metrics comparison across methods.

    Args:
        forecasts: Dictionary mapping method name to forecast data
        metric: Metric name (MAPE, RMSE, MAE, direction_accuracy, r_squared)

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    if not forecasts:
        logger.warning(f"No forecasts provided for metric {metric}")
        return fig

    methods = []
    values = []
    colors = []

    # Map display metric names to internal metric keys
    metric_key_map = {
        "MAPE": "mape",
        "RMSE": "rmse",
        "MAE": "mae",
        "Direction Accuracy": "direction_accuracy",
        "R²": "r_squared",
        "R2": "r_squared",
    }
    
    metric_key = metric_key_map.get(metric, metric.lower())
    
    for method_name, forecast_data in forecasts.items():
        if not forecast_data.get("success", False):
            logger.debug(f"Skipping {method_name}: not successful")
            continue

        metrics = forecast_data.get("validation_metrics")
        
        # Improved logging with safe access
        if metrics is None:
            logger.debug(f"Skipping {method_name}: validation_metrics is None")
            continue
        elif isinstance(metrics, dict):
            logger.debug(f"Method {method_name}: validation_metrics type=dict, keys={list(metrics.keys())}")
            # Empty dict is valid - it means metrics were attempted but all are NaN
            if len(metrics) == 0:
                logger.debug(f"Skipping {method_name}: validation_metrics is empty dict")
                continue
        else:
            logger.warning(f"Skipping {method_name}: validation_metrics is not dict, type={type(metrics)}, value={metrics}")
            continue
            
        # Try both the mapped key and direct lookup
        metric_value = None
        if metric_key in metrics:
            metric_value = metrics[metric_key]
            logger.debug(f"Found {metric_key} in metrics for {method_name}: {metric_value}")
        elif metric.lower() in metrics:
            metric_value = metrics[metric.lower()]
            logger.debug(f"Found {metric.lower()} in metrics for {method_name}: {metric_value}")
        else:
            logger.warning(f"Skipping {method_name}: metric key '{metric_key}' not found in {list(metrics.keys())}")
            continue
        
        # Safely convert to float and check if valid (handles all numeric types)
        try:
            if metric_value is None:
                logger.debug(f"Skipping {method_name}: metric_value is None")
                continue
            
            # Convert to float (handles int, float, numpy.float64, etc.)
            metric_value_float = float(metric_value)
            
            # Check if finite (not NaN, not Inf) - works for all numeric types
            if not np.isfinite(metric_value_float):
                logger.debug(f"Skipping {method_name}: metric_value is not finite (value={metric_value_float})")
                continue
            
            # Add to lists
            methods.append(method_name)
            values.append(metric_value_float)
            colors.append(get_method_color(method_name))
            logger.info(f"Added {method_name}: {metric_key}={metric_value_float}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping {method_name}: cannot convert metric_value to float: {e}, type={type(metric_value)}, value={metric_value}")
            continue

    if not methods:
        logger.warning(
            f"No methods found with valid {metric} metric. "
            f"Available forecasts: {list(forecasts.keys())}. "
            f"Metric key searched: {metric_key}"
        )
        # Return empty figure with a message
        fig.add_annotation(
            text=f"No data available for {metric}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        layout = get_chart_layout()
        layout.update(
            title=f"Forecast Quality: {metric}",
            xaxis_title="Method",
            yaxis_title=metric,
        )
        fig.update_layout(layout)
        return fig

    fig.add_trace(go.Bar(
        x=methods,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
    ))

    # Use autorange for automatic scaling - let Plotly determine the best range
    layout = get_chart_layout()
    layout.update(
        title=f"Forecast Quality: {metric.upper()}",
        xaxis_title="Method",
        yaxis_title=metric.upper(),
        xaxis=dict(
            autorange=True,  # Enable automatic scaling for X-axis
            automargin=True,  # Automatically add margins for labels
        ),
        yaxis=dict(
            autorange=True,  # Enable automatic scaling for Y-axis
            automargin=True,  # Automatically add margins for labels
        ),
    )

    fig.update_layout(layout)
    # Double-check: explicitly enable autorange AFTER layout update for both axes
    fig.update_xaxes(autorange=True, automargin=True)
    fig.update_yaxes(autorange=True, automargin=True)

    return fig


def plot_residuals(
    residuals: np.ndarray,
    method_name: str,
) -> go.Figure:
    """
    Plot forecast residuals for diagnostics.

    Args:
        residuals: Residual values (actual - forecast) - can be list, array, or None
        method_name: Method name for title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Handle different input types
    if residuals is None:
        logger.warning(f"Residuals are None for {method_name}")
        layout = get_chart_layout()
        layout.update(
            title=f"{method_name} Residuals",
            xaxis_title="Observation",
            yaxis_title="Residual",
        )
        fig.add_annotation(
            text="No residuals data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(layout)
        return fig
    
    # Convert to numpy array if needed
    if not isinstance(residuals, np.ndarray):
        try:
            # Handle list, tuple, pandas Series, etc.
            if hasattr(residuals, 'values'):  # pandas Series
                residuals = residuals.values
            residuals = np.array(residuals, dtype=float)
            logger.debug(f"Converted residuals to numpy array: shape={residuals.shape}")
        except (TypeError, ValueError) as e:
            logger.error(f"Could not convert residuals to array: {e}, type={type(residuals)}")
            layout = get_chart_layout()
            layout.update(
                title=f"{method_name} Residuals",
                xaxis_title="Observation",
                yaxis_title="Residual",
            )
            fig.add_annotation(
                text=f"Invalid residuals data: {type(residuals)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray"),
            )
            fig.update_layout(layout)
            return fig
    
    # Filter out NaN and Inf values
    valid_mask = np.isfinite(residuals)
    residuals_clean = residuals[valid_mask]
    
    if len(residuals_clean) == 0:
        logger.warning(f"All residuals are NaN/Inf for {method_name}, original length={len(residuals)}")
        layout = get_chart_layout()
        layout.update(
            title=f"{method_name} Residuals",
            xaxis_title="Observation",
            yaxis_title="Residual",
        )
        fig.add_annotation(
            text="No valid residuals data (all NaN/Inf)",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(layout)
        return fig

    logger.debug(f"Plotting {len(residuals_clean)} residuals for {method_name}, range=[{residuals_clean.min():.2f}, {residuals_clean.max():.2f}]")

    fig.add_trace(go.Scatter(
        x=np.arange(len(residuals_clean)),
        y=residuals_clean,
        mode="markers",
        name="Residuals",
        marker=dict(color=COLORS["primary"], size=4),
    ))

    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Zero",
    )

    # Use autorange for automatic scaling - let Plotly determine the best range
    layout = get_chart_layout()
    layout.update(
        title=f"{method_name} Residuals",
        xaxis_title="Observation",
        yaxis_title="Residual",
        xaxis=dict(
            autorange=True,  # Enable automatic scaling for X-axis
            automargin=True,  # Automatically add margins for labels
        ),
        yaxis=dict(
            autorange=True,  # Enable automatic scaling for Y-axis
            automargin=True,  # Automatically add margins for labels
        ),
    )

    fig.update_layout(layout)
    # Double-check: explicitly enable autorange AFTER layout update for both axes
    fig.update_xaxes(autorange=True, automargin=True)
    fig.update_yaxes(autorange=True, automargin=True)

    return fig
