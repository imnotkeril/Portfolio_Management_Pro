"""Period filter component for charts."""

from datetime import date, timedelta
from typing import Optional, Tuple

import streamlit as st


def render_period_filter(
    key_prefix: str = "default",
    default_period: str = "1Y",
) -> Tuple[date, date]:
    """
    Render period filter with date inputs.

    Args:
        key_prefix: Unique key prefix for the filter
        default_period: Default period to select

    Returns:
        Tuple of (start_date, end_date)
    """
    # Date inputs
    col1, col2 = st.columns(2)
    
    # Calculate default dates based on period
    end_date_default = date.today()
    
    period_days = {
        "6M": 180,
        "1Y": 365,
        "2Y": 730,
        "3Y": 1095,
    }
    
    days_back = period_days.get(default_period, 365)
    start_date_default = end_date_default - timedelta(days=days_back)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=start_date_default,
            max_value=end_date_default,
            key=f"start_date_{key_prefix}",
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=end_date_default,
            min_value=start_date,
            max_value=date.today(),
            key=f"end_date_{key_prefix}",
        )
    
    return start_date, end_date


def get_period_dates(
    period: str,
    end_date: Optional[date] = None,
    original_start: Optional[date] = None,
) -> Tuple[date, date]:
    """
    Get start and end dates for a given period.

    Args:
        period: Period code ('6M', '1Y', '2Y', 'All')
        end_date: End date (defaults to today)
        original_start: Original start date for 'All' period

    Returns:
        Tuple of (start_date, end_date)
    """
    if end_date is None:
        end_date = date.today()

    period_days = {
        "6M": 180,
        "1Y": 365,
        "2Y": 730,
    }

    if period == "All":
        if original_start is None:
            # Default to 5 years if no original start
            start_date = end_date - timedelta(days=1825)
        else:
            start_date = original_start
    else:
        days = period_days.get(period, 365)
        start_date = end_date - timedelta(days=days)

        # Don't go before original start if provided
        if original_start is not None:
            start_date = max(start_date, original_start)

    return start_date, end_date


def filter_series_by_period(
    data_series,
    period: str,
    original_start: Optional[date] = None,
):
    """
    Filter pandas Series by period.

    Args:
        data_series: Pandas Series with DatetimeIndex
        period: Period code ('6M', '1Y', '2Y', 'All')
        original_start: Original start date

    Returns:
        Filtered Series
    """
    import pandas as pd

    if data_series.empty:
        return data_series

    # Ensure index is DatetimeIndex
    if not isinstance(data_series.index, pd.DatetimeIndex):
        # Try to convert to DatetimeIndex
        try:
            data_series.index = pd.to_datetime(data_series.index)
        except Exception:
            # If conversion fails, return as is
            return data_series

    if period == "All":
        return data_series

    # Get end_date from index
    end_date_ts = data_series.index.max()
    if hasattr(end_date_ts, 'date'):
        end_date = end_date_ts.date()
    else:
        end_date = pd.Timestamp(end_date_ts).date()

    start_date, _ = get_period_dates(period, end_date, original_start)

    # Convert to pandas Timestamp for comparison
    start_ts = pd.Timestamp(start_date)

    return data_series[data_series.index >= start_ts]

