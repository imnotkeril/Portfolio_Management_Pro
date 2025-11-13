"""Market index card component for displaying index information."""

from typing import Optional

import streamlit as st

from streamlit_app.utils.chart_config import COLORS
from streamlit_app.utils.formatters import format_percentage


def render_market_index_card(
    name: str,
    symbol: str,
    current_price: Optional[float] = None,
    change: Optional[float] = None,
    change_pct: Optional[float] = None,
) -> None:
    """
    Render a market index card.

    Args:
        name: Index name (e.g., "S&P 500")
        symbol: Index symbol (e.g., "^GSPC")
        current_price: Current price value
        change: Absolute change value
        change_pct: Percentage change
    """
    # Format values (indices are not currency, so format as numbers without decimals)
    if current_price is not None:
        price_str = f"{current_price:,.0f}"
    else:
        price_str = "N/A"
    
    # Only show percentage change, no absolute change, no color
    # change_pct comes as percentage (e.g., 25.34), convert to decimal for formatter
    if change_pct is not None:
        change_pct_decimal = change_pct / 100.0
        change_pct_str = format_percentage(change_pct_decimal, decimals=2)
    else:
        change_pct_str = "N/A"

    # Card container
    st.markdown(
        f"""
        <div style="
            border: 1px solid #2A2E39;
            border-radius: 8px;
            padding: 16px;
            background-color: #1A1E29;
            text-align: center;
        ">
            <h4 style="color: {COLORS['primary']}; margin-top: 0; margin-bottom: 8px;">
                {name}
            </h4>
            <p style="color: {COLORS['secondary']}; font-size: 0.85em; margin: 4px 0;">
                {symbol}
            </p>
            <p style="color: #FFFFFF; font-size: 1.2em; font-weight: bold; margin: 8px 0;">
                {price_str}
            </p>
            <div style="margin-top: 8px;">
                <p style="color: #D1D4DC; font-size: 0.9em; margin: 2px 0; font-weight: bold;">
                    {change_pct_str}
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

