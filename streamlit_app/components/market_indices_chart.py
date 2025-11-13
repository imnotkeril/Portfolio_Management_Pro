"""Market indices comparison chart component."""

from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

from services.data_service import DataService


def get_market_indices_data(
    indices: List[Dict[str, str]],
    period_days: int = 30,
    data_service: Optional[DataService] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch market indices data for charting.

    Args:
        indices: List of dicts with 'symbol' and 'name' keys
        period_days: Number of days to fetch (default: 30)
        data_service: Optional DataService instance

    Returns:
        DataFrame with columns: date, and one column per index (normalized to 100)
    """
    if data_service is None:
        data_service = DataService()

    import logging

    logger = logging.getLogger(__name__)

    end_date = date.today()
    start_date = end_date - timedelta(days=period_days)

    all_data = {}
    dates_set = set()

    # Fetch data for each index
    for index_info in indices:
        symbol = index_info["symbol"]
        name = index_info["name"]

        try:
            df = data_service.fetch_historical_prices(
                symbol,
                start_date,
                end_date,
                use_cache=True,
                save_to_db=True,
            )

            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol}")
                continue

            # Check for Date column and set as index if needed
            if "Date" in df.columns:
                df = df.copy()
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df.set_index("Date", inplace=True)
            elif df.index.name != "Date" and not isinstance(df.index, pd.DatetimeIndex):
                # Try to convert index to datetime
                try:
                    df.index = pd.to_datetime(df.index, errors="coerce")
                except Exception:
                    logger.warning(f"Could not convert index to datetime for {symbol}")
                    continue

            # Use Adjusted_Close if available, otherwise Close
            price_column = None
            if "Adjusted_Close" in df.columns:
                price_column = "Adjusted_Close"
            elif "Close" in df.columns:
                price_column = "Close"
            elif "close" in df.columns:
                price_column = "close"
            else:
                logger.warning(
                    f"No price column found for {symbol}. "
                    f"Available columns: {list(df.columns)}"
                )
                continue

            # Normalize to 100 at first date
            prices = df[price_column].dropna()
            if len(prices) == 0:
                logger.warning(f"No valid prices for {symbol}")
                continue

            # Sort by date to ensure first_price is earliest
            prices = prices.sort_index()
            first_price = prices.iloc[0]

            if first_price <= 0:
                logger.warning(f"Invalid first price for {symbol}: {first_price}")
                continue

            normalized = (prices / first_price) * 100
            all_data[name] = normalized
            dates_set.update(normalized.index)

        except Exception as e:
            # Log error but continue with other indices
            logger.warning(f"Failed to fetch data for {symbol}: {e}", exc_info=True)

    if not all_data:
        return None

    # Combine all data into single DataFrame
    result_df = pd.DataFrame(all_data)
    result_df = result_df.sort_index()

    return result_df


def plot_market_indices_comparison(
    data: pd.DataFrame,
    title: str = "Market Indices Comparison",
) -> go.Figure:
    """
    Create a plotly chart comparing market indices.

    Args:
        data: DataFrame with date index and index names as columns
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Color palette for indices (using unified colors)
    from streamlit_app.utils.chart_config import COLORS
    colors = [
        COLORS["primary"],  # Purple
        COLORS["secondary"],  # Blue
        COLORS["success"],  # Green
        COLORS["danger"],  # Red
        COLORS["additional"],  # Yellow
    ]

    for i, column in enumerate(data.columns):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[column],
                mode="lines",
                name=column,
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{column}</b><br>"
                + "Date: %{x}<br>"
                + "Value: %{y:.2f}<br>"
                + "<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color="#FFFFFF"),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text="Date", font=dict(color="#D1D4DC")),
            tickfont=dict(color="#D1D4DC"),
            gridcolor="#2A2E39",
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text="Change from Start (%)", font=dict(color="#D1D4DC")),
            tickfont=dict(color="#D1D4DC"),
            gridcolor="#2A2E39",
            showgrid=True,
            tickformat=".1f",
        ),
        plot_bgcolor="#0D1015",
        paper_bgcolor="#0D1015",
        legend=dict(
            bgcolor="#1A1E29",
            bordercolor="#2A2E39",
            borderwidth=1,
            font=dict(color="#D1D4DC"),
        ),
        hovermode="x unified",
        height=600,
    )

    return fig

