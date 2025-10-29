"""Position table component for displaying portfolio positions."""

from typing import List, Optional

import pandas as pd
import streamlit as st

from streamlit_app.utils.formatters import (
    format_currency,
    format_percentage,
)


def render_position_table(
    positions: List[dict],
    show_actions: bool = True,
) -> Optional[str]:
    """
    Render a table of portfolio positions.

    Args:
        positions: List of position dictionaries with keys:
            - ticker
            - shares
            - current_price (optional)
            - current_value (optional)
            - weight (optional)
            - gain_loss (optional)
        show_actions: Whether to show action buttons (default: True)

    Returns:
        Selected ticker for action or None
    """
    if not positions:
        st.info("No positions in portfolio")
        return None

    # Prepare DataFrame
    data = []
    for pos in positions:
        ticker = pos.get("ticker", "-")
        # For cash, show shares as dollar amount
        if ticker == "CASH":
            shares_display = format_currency(pos.get('shares', 0))
        else:
            shares_display = f"{pos.get('shares', 0):.2f}"
        row = {
            "Ticker": ticker,
            "Shares": shares_display,
            "Price": (
                format_currency(pos.get("current_price", 0))
                if pos.get("current_price")
                else ("$1.00" if ticker == "CASH" else "-")
            ),
            "Value": (
                format_currency(pos.get("current_value", 0))
                if pos.get("current_value")
                else "-"
            ),
            "Weight": (
                format_percentage(pos.get("weight", 0))
                if pos.get("weight") is not None
                else "-"
            ),
            "P&L": (
                format_currency(pos.get("gain_loss", 0))
                if pos.get("gain_loss") is not None
                else "-"
            ),
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    # Action buttons if enabled
    if show_actions:
        st.markdown("### Actions")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Add Position"):
                st.session_state.show_add_position = True

        with col2:
            if st.button("Rebalance to Targets"):
                st.session_state.show_rebalance = True

    return None
