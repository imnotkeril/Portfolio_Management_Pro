"""Transaction form component for adding/editing transactions."""

import logging
from datetime import date
from typing import Any, Dict, Optional

import streamlit as st

logger = logging.getLogger(__name__)


def render_transaction_form(
    portfolio_id: str,
    default_date: Optional[date] = None,
    default_type: Optional[str] = None,
    default_ticker: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Render transaction form in expander.

    Args:
        portfolio_id: Portfolio ID
        default_date: Default transaction date
        default_type: Default transaction type
        default_ticker: Default ticker symbol

    Returns:
        Dictionary with transaction data if submitted, None otherwise
    """
    with st.expander("Add Transaction", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            transaction_date = st.date_input(
                "Transaction Date",
                value=default_date or date.today(),
                max_value=date.today(),
                help="Date when the transaction occurred",
            )

            transaction_type = st.selectbox(
                "Transaction Type",
                options=["BUY", "SELL", "DEPOSIT", "WITHDRAWAL"],
                index=0
                if not default_type
                else ["BUY", "SELL", "DEPOSIT", "WITHDRAWAL"].index(default_type),
                help="Type of transaction",
            )

        with col2:
            if transaction_type in ["DEPOSIT", "WITHDRAWAL"]:
                ticker = "CASH"
                st.info("Ticker: CASH (automatic for deposits/withdrawals)")
            else:
                ticker = st.text_input(
                    "Ticker",
                    value=default_ticker or "",
                    help="Enter ticker symbol (e.g., AAPL, MSFT)",
                    placeholder="AAPL",
                ).upper()

            shares = st.number_input(
                "Shares / Amount",
                min_value=0.0001,
                value=1.0,
                step=0.01,
                format="%.4f",
                help="Number of shares for BUY/SELL, amount for DEPOSIT/WITHDRAWAL",
            )

        col3, col4 = st.columns(2)

        with col3:
            if transaction_type in ["DEPOSIT", "WITHDRAWAL"]:
                price = 1.0
                st.info("Price: $1.00 (automatic for CASH)")
            else:
                price = st.number_input(
                    "Price per Share",
                    min_value=0.01,
                    value=100.0,
                    step=0.01,
                    format="%.2f",
                    help="Price per share at time of transaction",
                )

        with col4:
            fees = st.number_input(
                "Fees",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                help="Transaction fees (optional)",
            )

        notes = st.text_area(
            "Notes (optional)",
            value="",
            max_chars=500,
            help="Optional notes about this transaction",
        )

        # Calculate total amount
        total_amount = shares * price
        st.info(f"**Total Amount:** ${total_amount:,.2f}")

        col5, col6 = st.columns(2)

        with col5:
            if st.button("Add Transaction", type="primary", use_container_width=True):
                # Validate
                if transaction_type in ["BUY", "SELL"] and not ticker:
                    st.error("Please enter a ticker symbol")
                    return None

                return {
                    "transaction_date": transaction_date,
                    "transaction_type": transaction_type,
                    "ticker": ticker
                    if transaction_type not in ["DEPOSIT", "WITHDRAWAL"]
                    else "CASH",
                    "shares": shares,
                    "price": price,
                    "fees": fees,
                    "notes": notes,
                }

        with col6:
            if st.button("Cancel", use_container_width=True):
                return None

        return None

